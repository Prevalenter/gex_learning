from __future__ import annotations

from dataclasses import asdict
import json

import numpy as np
import pytest

from gex_learning.tasks.direct.super_inair.grasp_data import (
    JOINT_NAMES, Screening, empty_states, failure_masks, joint_order,
    load_cache, local_pose, save_cache, validate_seed, world_pose,
)
from scripts.super_inair_grasps import generate, trial, tiled_seed, validated_batch


def seed():
    return dict(version=1, saved_by_user=True, joint_names=JOINT_NAMES.copy(), joint_pos=[0.0] * 16,
                hand_pose=[0, 0, 0.5, 1, 0, 0, 0], object_pose=[0, 0, 0.6, 1, 0, 0, 0],
                ball_diameter_m=0.06, ball_mass_kg=0.05)


def state(count=2):
    return dict(tiled_seed(seed(), count), joint_vel=np.zeros((count, 16)),
                object_velocity=np.zeros((count, 6)),
                contact_forces=np.tile([[0.2, 0, 0], [0, 0.2, 0], [0, 0, 0], [0, 0, 0]], (count, 1, 1)))


def test_seed_requires_explicit_save_and_maps_names():
    raw = seed()
    raw["saved_by_user"] = False
    with pytest.raises(ValueError, match="Save a seed"):
        validate_seed(raw)
    raw["saved_by_user"] = True
    raw["joint_names"] = JOINT_NAMES[::-1]
    raw["joint_pos"] = list(range(16))
    assert validate_seed(raw)["joint_pos"] == list(reversed(range(16)))
    with pytest.raises(ValueError):
        joint_order(["joint1"] * 16, JOINT_NAMES)
    raw["hand_pose"][3:] = [0, 0, 0, 0]
    with pytest.raises(ValueError, match="quaternion"):
        validate_seed(raw)


def test_origins_do_not_leak_between_environments():
    local = np.array([[0, 0.1, 0.6, 1, 0, 0, 0], [0.1, 0.2, 0.7, 0, 1, 0, 0]])
    origins = np.array([[5, -5, 0], [-5, 5, 0]])
    np.testing.assert_allclose(local_pose(world_pose(local, origins), origins), local)
    np.testing.assert_allclose(world_pose(local, origins)[:, 3:], local[:, 3:])


def test_screening_contact_drop_drift_and_nonfinite():
    s = state()
    initial = s["object_pose"][:, :3].copy()
    assert not any(x.any() for x in failure_masks(s, initial, initial, Screening(), True).values())
    s["contact_forces"][0] = 0
    s["object_pose"][1, 2] -= 0.06
    failures = failure_masks(s, initial, initial, Screening(), True)
    assert failures["contacts"].tolist() == [True, False]
    assert failures["drop"].tolist() == [False, True]
    s = state()
    s["object_pose"][0, 0] += 0.011
    s["joint_pos"][1, 0] = np.nan
    failures = failure_masks(s, initial, initial, Screening(), True)
    assert failures["drift"].tolist() == [True, False]
    assert failures["nonfinite"].tolist() == [False, True]
    assert "contacts" not in failure_masks(s, initial, initial, Screening(), False)


def test_cache_roundtrip_keeps_targets_and_configuration(tmp_path):
    states = tiled_seed(seed(), 3)
    states["joint_targets"][:] = 0.1
    meta = dict(seed_pose=seed(), screening=asdict(Screening()), status="complete")
    save_cache(tmp_path, states, meta)
    restored, metadata = load_cache(tmp_path)
    np.testing.assert_allclose(restored["joint_targets"], 0.1)
    np.testing.assert_allclose(restored["joint_pos"], 0)
    assert metadata["count"] == 3 and metadata["coordinate_frame"] == "environment"
    # Corruption cannot silently load as a valid cache.
    payload = json.loads((tmp_path / "metadata.json").read_text())
    payload["count"] = 4
    (tmp_path / "metadata.json").write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="Corrupt"):
        load_cache(tmp_path)


def test_empty_partial_cache_is_valid(tmp_path):
    save_cache(tmp_path, empty_states(), dict(seed_pose=seed(), status="seed_failed"))
    assert load_cache(tmp_path)[0]["object_pose"].shape == (0, 7)


class App:
    def is_running(self):
        return True


class FakeEnv:
    num_envs = 2
    step_dt = 0.1
    actions = np.zeros((2, 16))
    seed_pose = seed()

    def __init__(self, fail_step=None):
        self.steps = 0
        self.fail_step = fail_step
        self.restored = []

    def step(self, actions):
        self.steps += 1

    def snapshot(self):
        s = state()
        if self.steps == self.fail_step:
            s["contact_forces"][0] = 0
        return s

    def restore(self, values, ids=None):
        self.restored.append(ids)


def test_failure_is_latched_even_if_contacts_recover():
    env = FakeEnv(fail_step=2)
    accepted, _, reasons = trial(env, App(), Screening(), tiled_seed(seed(), 2),
                                 np.ones(2, dtype=bool), duration=0.4)
    assert accepted.tolist() == [False, True]
    assert reasons["contacts"] == 1
    assert env.restored[0].tolist() == [0]


def test_restoration_failure_does_not_enter_cache():
    env = FakeEnv(fail_step=3)
    limits = Screening(settle_s=0, hold_s=0.2, replay_s=0.2)
    accepted, _, reasons = validated_batch(env, App(), limits, tiled_seed(seed(), 2),
                                           np.ones(2, dtype=bool), float("inf"))
    assert accepted.tolist() == [False, True]
    assert reasons["replay_contacts"] == 1


def test_validation_reports_restore_failure_and_keeps_failed_frame(tmp_path):
    from scripts.super_inair_grasps import validate_pose
    env = FakeEnv(fail_step=3)
    env.physics_metadata = lambda: {}
    limits = Screening(settle_s=0, hold_s=0.2, replay_s=0.2)
    assert validate_pose(env, App(), limits, tmp_path, 1) == 2
    report = json.loads((tmp_path / "validation.json").read_text())
    assert not report["passed"]
    assert report["diagnostics"]["hold"]["reasons"] == []
    assert report["diagnostics"]["replay"]["reasons"] == ["contacts"]
    assert report["diagnostics"]["replay"]["contacts"] == 0
    env = FakeEnv()
    env.physics_metadata = lambda: {}
    assert validate_pose(env, App(), limits, tmp_path, 1) == 0
    assert json.loads((tmp_path / "validation.json").read_text())["passed"]




@pytest.mark.parametrize("max_attempts,expected_status,expected_count", [
    (10, "complete", 3), (2, "attempt_limit", 2),
])
def test_collector_saves_exact_target_or_honest_partial(tmp_path, max_attempts, expected_status, expected_count):
    from types import SimpleNamespace
    import torch

    env = FakeEnv()
    env.actuated_dof_indices = list(range(16))
    env.hand_dof_lower_limits = torch.full((2, 16), -1.0)
    env.hand_dof_upper_limits = torch.full((2, 16), 1.0)
    env.physics_metadata = lambda: {"fixed_base": True}
    args = SimpleNamespace(seed=0, max_minutes=1, config=tmp_path / "seed.json", count=3,
                           max_attempts=max_attempts)
    code = generate(env, App(), args, Screening(settle_s=0, hold_s=0.1, replay_s=0.1), tmp_path)
    states, meta = load_cache(tmp_path)
    assert len(states["joint_pos"]) == expected_count
    assert meta["status"] == expected_status
    assert code == (0 if expected_status == "complete" else 2)


def test_failed_seed_never_starts_sampling(tmp_path):
    from types import SimpleNamespace
    env = FakeEnv(fail_step=1)
    env.physics_metadata = lambda: {}
    args = SimpleNamespace(seed=0, max_minutes=1, config=tmp_path / "seed.json", count=3, max_attempts=10)
    assert generate(env, App(), args, Screening(settle_s=0, hold_s=0.1, replay_s=0.1), tmp_path) == 2
    states, meta = load_cache(tmp_path)
    assert len(states["joint_pos"]) == meta["attempts"] == 0
    assert meta["status"] == "seed_failed"


def test_stability_costs_use_velocity_and_episode_specific_actual_pose():
    import torch
    from gex_learning.tasks.direct.super_inair.reward_terms import grasp_stability_costs
    previous = torch.tensor([[2., 0., 0.], [0., 3., 0.]])
    position = previous + torch.tensor([[.01, -.02, 0.], [0., 0., 0.]])
    initial = torch.tensor([[1., 2.], [-1., -2.]])
    joints = initial + torch.tensor([[.1, -.2], [0., 0.]])
    speed, deviation = grasp_stability_costs(position, previous, .1, joints, initial)
    torch.testing.assert_close(speed, torch.tensor([.3, 0.]))
    torch.testing.assert_close(deviation, torch.tensor([.05, 0.]))
    torch.testing.assert_close(-.3 * speed - .3 * deviation, torch.tensor([-.105, 0.]))
    # Resetting one environment updates only its reference, removing its pose cost.
    initial[0] = joints[0]
    _, deviation = grasp_stability_costs(position, previous, .1, joints, initial)
    torch.testing.assert_close(deviation, torch.zeros(2))


def test_object_position_cost_per_environment_reset_and_translation_invariance():
    import torch
    from gex_learning.tasks.direct.super_inair.reward_terms import object_position_cost
    initial = torch.tensor([[1., 2., 3.], [-2., 1., 4.], [0., 0., 1.]])
    current = initial + torch.tensor([[0., 0., 0.], [.006, .008, 0.], [0., 0., .02]])
    cost, distance = object_position_cost(current, initial, .02)
    torch.testing.assert_close(distance, torch.tensor([0., .01, .02]))
    torch.testing.assert_close(-.2 * cost, torch.tensor([0., -.05, -.2]))
    offsets = torch.tensor([[3., 2., 1.], [7., 1., 3.], [-1., -2., 4.]])
    translated, _ = object_position_cost(current + offsets, initial + offsets, .02)
    torch.testing.assert_close(translated, cost, atol=1e-4, rtol=1e-4)
    initial[1] = current[1]  # Reset only one environment.
    after_reset, _ = object_position_cost(current, initial, .02)
    torch.testing.assert_close(after_reset, torch.tensor([0., 0., 1.]))




def test_hora_rotation_sign_quaternion_invariance_and_power_formula():
    import torch
    from gex_learning.tasks.direct.super_inair.reward_terms import rotation_velocity, hora_reward_terms
    angle = torch.tensor([-.1, .1, 0.])
    current = torch.stack((torch.cos(angle / 2), torch.zeros(3), torch.zeros(3), torch.sin(angle / 2)), -1)
    previous = torch.tensor([[1., 0., 0., 0.]]).repeat(3, 1)
    velocity = rotation_velocity(current, previous, .1)
    torch.testing.assert_close(velocity[:, 2], torch.tensor([-1., 1., 0.]))
    torch.testing.assert_close(rotation_velocity(-current, previous, .1), velocity)
    terms = hora_reward_terms(velocity, torch.tensor([0., 0., -1.]), torch.ones(3), torch.ones(3),
                              torch.tensor([[1., 2.]]).repeat(3, 1),
                              torch.tensor([[2., -1.], [1., 1.], [0., 0.]]))
    torch.testing.assert_close(terms['rotation'], torch.tensor([.5, -.5, 0.]))
    torch.testing.assert_close(terms['torque'], torch.full((3,), 5.))
    torch.testing.assert_close(terms['work'], torch.tensor([0., 9., 0.]))




def test_wrist_x_axis_rotates_with_base_and_preserves_reward_projection():
    import torch
    from gex_learning.tasks.direct.super_inair.base_pose import rotation_axis_world, rotate_vector
    from gex_learning.tasks.direct.super_inair.reward_terms import hora_reward_terms
    quaternions = torch.tensor([[1., 0., 0., 0.], [2**-.5, 0., 0., 2**-.5], [0., 0., 1., 0.]])
    axis = rotation_axis_world([1., 0., 0.], quaternions, 'wrist')
    torch.testing.assert_close(axis, torch.tensor([[1., 0., 0.], [0., 1., 0.], [-1., 0., 0.]]), atol=2e-7, rtol=0)
    speed = rotate_vector(quaternions, torch.tensor([[.3, .1, .2]]).expand(3, -1))
    terms = hora_reward_terms(speed, axis, torch.zeros(3), torch.zeros(3), torch.zeros(3, 16), torch.zeros(3, 16))
    torch.testing.assert_close(terms['rotation'], torch.full((3,), .3))
    torch.testing.assert_close(rotation_axis_world([0., 0., -1.], quaternions, 'world'),
                               torch.tensor([[0., 0., -1.]]).expand(3, -1))


def test_reverse_rotation_multiplier_only_scales_negative_clipped_reward():
    import torch
    from gex_learning.tasks.direct.super_inair.reward_terms import hora_reward_terms
    speed = torch.tensor([[-2., 0., 0.], [-.3, 0., 0.], [0., 0., 0.], [.3, 0., 0.], [2., 0., 0.]])
    args = (speed, torch.tensor([1., 0., 0.]), torch.zeros(5), torch.zeros(5), torch.zeros(5, 16), torch.zeros(5, 16))
    baseline = hora_reward_terms(*args)
    changed = hora_reward_terms(*args, reverse_multiplier=3.0)
    torch.testing.assert_close(changed['rotation'] * 2, torch.tensor([-3., -1.8, 0., .6, 1.]))
    for key in baseline:
        if key != 'rotation':
            torch.testing.assert_close(changed[key], baseline[key])


def test_unclipped_reverse_rotation_and_clipped_positive_rotation():
    import torch
    from gex_learning.tasks.direct.super_inair.reward_terms import hora_reward_terms
    speed = torch.tensor([[-2., 0., 0.], [-1., 0., 0.], [-.5, 0., 0.], [-.3, 0., 0.], [0., 0., 0.], [.3, 0., 0.], [2., 0., 0.]])
    terms = hora_reward_terms(speed, torch.tensor([1., 0., 0.]), torch.zeros(7), torch.zeros(7),
                              torch.zeros(7, 16), torch.zeros(7, 16), reverse_multiplier=6.0, clip_reverse=False)
    torch.testing.assert_close(terms['rotation'] * 2, torch.tensor([-24., -12., -6., -3.6, 0., .6, 1.]))


def test_random_base_rigid_transform_and_uniform_orientation():
    import torch
    from gex_learning.tasks.direct.super_inair.base_pose import (
        transform_grasp, sample_base_transform, rotate_vector, multiply_quaternions)
    torch.manual_seed(9)
    nominal = torch.tensor([.1, -.2, .5, .70710678, .70710678, 0., 0.])
    obj = torch.tensor([.12, -.1, .6, 1., 0., 0., 0.]).repeat(10000, 1)
    offset, rotation = sample_base_transform(len(obj), .02, 'cpu')
    hand, moved = transform_grasp(nominal, obj, offset, rotation)
    assert offset.abs().max() <= .02
    inverse = hand[:, 3:].clone()
    inverse[:, 1:] *= -1
    original_inverse = nominal[3:].repeat(len(obj), 1)
    original_inverse[:, 1:] *= -1
    torch.testing.assert_close(rotate_vector(inverse, moved[:, :3]-hand[:, :3]),
                               rotate_vector(original_inverse, obj[:, :3]-nominal[:3]), atol=2e-7, rtol=1e-5)
    torch.testing.assert_close(multiply_quaternions(inverse, moved[:, 3:]),
                               multiply_quaternions(original_inverse, obj[:, 3:]), atol=3e-7, rtol=1e-5)
    direction = rotate_vector(rotation, torch.tensor([0., 0., 1.]).expand(len(obj), -1))
    assert direction.mean(0).abs().max() < .025
    torch.testing.assert_close(direction.square().mean(0), torch.full((3,), 1/3), atol=.015, rtol=0)
    offset, rotation = sample_base_transform(len(obj), 0.0, 'cpu')
    hand, _ = transform_grasp(nominal, obj, offset, rotation)
    torch.testing.assert_close(hand[:, :3], nominal[:3].expand(len(obj), -1), atol=0, rtol=0)
