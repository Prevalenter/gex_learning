"""Generate, replay, or smoke-test a fixed-base single-ball grasp cache."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, fields
from datetime import datetime
import math
from pathlib import Path
import time

import numpy as np

from gex_learning.tasks.direct.super_inair.grasp_data import (
    DEFAULT_CONFIG, ROOT, STATE_FIELDS, Screening, empty_states, failure_masks,
    load_cache, load_seed, save_cache, write_json,
)


class RunStopped(Exception):
    pass


def tiled_seed(seed, count):
    return dict(joint_pos=np.tile(seed["joint_pos"], (count, 1)),
                joint_targets=np.tile(seed["joint_pos"], (count, 1)),
                object_pose=np.tile(seed["object_pose"], (count, 1)))


def trial(env, app, limits, initial, active, duration, settle=0.0, deadline=math.inf, real_time=False,
          diagnostics=None):
    """Latch failures before any reset; failed environments are parked at the seed pose."""
    alive = active.copy()
    reasons = Counter()
    start = initial["object_pose"][:, :3].copy()
    anchor = start.copy()
    settling_steps = math.ceil(settle / env.step_dt)
    total_steps = settling_steps + math.ceil(duration / env.step_dt)
    state = None
    for step in range(total_steps):
        if time.monotonic() >= deadline or not app.is_running():
            raise RunStopped("time_limit" if time.monotonic() >= deadline else "window_closed")
        begin = time.monotonic()
        env.step(env.actions * 0)
        state = env.snapshot()
        if step == settling_steps - 1:
            anchor = state["object_pose"][:, :3].copy()
        masks = failure_masks(state, start, anchor, limits, stable=step >= settling_steps)
        failed = np.zeros(env.num_envs, dtype=bool)
        for name, mask in masks.items():
            reasons[name] += int((alive & mask).sum())
            failed |= mask
        newly_failed = alive & failed
        if diagnostics is not None and alive[0]:
            forces = np.linalg.norm(state["contact_forces"][0], axis=-1)
            diagnostics.update(
                time_s=(step + 1) * env.step_dt, checking=step >= settling_steps,
                reasons=[name for name, mask in masks.items() if mask[0]],
                contacts=int((forces > limits.contact_force_n).sum()),
                contact_forces_n=forces.tolist(),
                linear_speed_m_s=float(np.linalg.norm(state["object_velocity"][0, :3])),
                angular_speed_rad_s=float(np.linalg.norm(state["object_velocity"][0, 3:])),
                displacement_m=float(np.linalg.norm(state["object_pose"][0, :3] - start[0])),
                drift_m=float(np.linalg.norm(state["object_pose"][0, :3] - anchor[0])),
            )
        alive &= ~failed
        if newly_failed.any():
            ids = np.flatnonzero(newly_failed)
            env.restore(tiled_seed(env.seed_pose, len(ids)), ids)
        if real_time:
            time.sleep(max(0, env.step_dt - (time.monotonic() - begin)))
        if not alive.any():
            break
    return alive, state, dict(reasons)


def validated_batch(env, app, limits, candidates, active, deadline, diagnostics=None):
    env.restore(candidates)
    hold = diagnostics.setdefault("hold", {}) if diagnostics is not None else None
    passed, snapshot, failures = trial(env, app, limits, candidates, active,
                                     limits.hold_s, limits.settle_s, deadline, diagnostics=hold)
    selected = {key: snapshot[key].copy() for key in STATE_FIELDS}
    if passed.any():
        # Failed states may contain NaN. Never write them back into the simulator.
        replay = tiled_seed(env.seed_pose, env.num_envs)
        for key in STATE_FIELDS:
            replay[key][passed] = selected[key][passed]
        env.restore(replay)
        replay_diagnostics = diagnostics.setdefault("replay", {}) if diagnostics is not None else None
        restored, _, replay_failures = trial(env, app, limits, replay, passed,
                                            limits.replay_s, deadline=deadline, diagnostics=replay_diagnostics)
        failures.update({f"replay_{key}": value for key, value in replay_failures.items()})
        passed &= restored
    return passed, selected, failures


def validate_pose(env, app, limits, directory, max_minutes):
    """Run precisely the collector's seed gate, including restore verification."""
    diagnostics = {}
    active = np.zeros(env.num_envs, dtype=bool)
    active[0] = True
    passed, _, failures = validated_batch(
        env, app, limits, tiled_seed(env.seed_pose, env.num_envs), active,
        time.monotonic() + max_minutes * 60, diagnostics=diagnostics)
    report = dict(passed=bool(passed[0]), failures=failures, diagnostics=diagnostics,
                  screening=asdict(limits), seed_pose=env.seed_pose, physics=env.physics_metadata())
    write_json(directory / "validation.json", report)
    print(f"[grasp] Validation passed={report['passed']}; report: {directory / 'validation.json'}", flush=True)
    return 0 if report["passed"] else 2


def generate(env, app, args, limits, directory):
    rng = np.random.default_rng(args.seed)
    states = empty_states()
    attempts, failures = 0, Counter()
    begin = time.monotonic()
    deadline = begin + args.max_minutes * 60
    metadata = dict(seed_pose=env.seed_pose, screening=asdict(limits), physics=env.physics_metadata(),
                    seed=args.seed, requested_count=args.count, attempts=0, status="running",
                    input_config=str(args.config.resolve()), failures={})

    def checkpoint(status):
        count = len(states["joint_pos"])
        metadata.update(status=status, attempts=attempts, failures=dict(failures),
                        elapsed_s=time.monotonic() - begin,
                        acceptance_rate=count / attempts if attempts else 0.0)
        save_cache(directory, states, metadata)

    try:
        seed_states = tiled_seed(env.seed_pose, env.num_envs)
        active = np.zeros(env.num_envs, dtype=bool)
        active[0] = True
        print("[grasp] Validating the saved, unperturbed pose and its restoration...", flush=True)
        passed, pilot_state, pilot_failures = validated_batch(env, app, limits, seed_states, active, deadline)
        pilot_delta = pilot_state["object_pose"][0, :3] - seed_states["object_pose"][0, :3]
        metadata["pilot"] = dict(passed=bool(passed[0]), failures=pilot_failures,
                                 final_object_pose=pilot_state["object_pose"][0].tolist(),
                                 displacement_xyz_m=pilot_delta.tolist())
        if not passed[0]:
            checkpoint("seed_failed")
            print(f"[grasp] Seed failed: {pilot_failures}. Adjust the pose in viser; no sampling started.", flush=True)
            return 2
        print("[grasp] Seed passed. Starting perturbed sampling.", flush=True)
        lower = env.hand_dof_lower_limits[:, env.actuated_dof_indices].cpu().numpy()
        upper = env.hand_dof_upper_limits[:, env.actuated_dof_indices].cpu().numpy()
        while len(states["joint_pos"]) < args.count and attempts < args.max_attempts:
            if time.monotonic() >= deadline:
                raise RunStopped("time_limit")
            count = min(env.num_envs, args.max_attempts - attempts)
            active = np.arange(env.num_envs) < count
            candidates = tiled_seed(env.seed_pose, env.num_envs)
            q = candidates["joint_pos"] + rng.uniform(-limits.joint_noise_rad, limits.joint_noise_rad,
                                                      candidates["joint_pos"].shape)
            candidates["joint_pos"] = np.clip(q, lower, upper)
            candidates["joint_targets"] = candidates["joint_pos"].copy()
            candidates["object_pose"][:, :3] += rng.uniform(-limits.position_noise_m, limits.position_noise_m,
                                                           (env.num_envs, 3))
            attempts += count
            passed, sample, rejected = validated_batch(env, app, limits, candidates, active, deadline)
            failures.update(rejected)
            keep = np.flatnonzero(passed)[:args.count - len(states["joint_pos"])]
            states = {key: np.concatenate((states[key], sample[key][keep])) for key in STATE_FIELDS}
            checkpoint("running")
            print(f"[grasp] accepted={len(states['joint_pos'])}/{args.count}, attempts={attempts}, "
                  f"batch_passed={int(passed.sum())}/{count}", flush=True)
        complete = len(states["joint_pos"]) == args.count
        checkpoint("complete" if complete else "attempt_limit")
        return 0 if complete else 2
    except (RunStopped, KeyboardInterrupt) as error:
        checkpoint(str(error) or "interrupted")
        return 2
    except Exception:
        checkpoint("error")
        raise
    finally:
        print(f"[grasp] Results: {directory}; accepted={len(states['joint_pos'])}; attempts={attempts}", flush=True)


def preview(env, app, args, limits, states):
    if not 0 <= args.index < len(states["joint_pos"]):
        raise ValueError(f"--index must be in [0, {len(states['joint_pos'])})")
    state = {key: states[key][args.index:args.index+1] for key in STATE_FIELDS}
    env.restore(state)
    deadline = math.inf if args.duration <= 0 else time.monotonic() + args.duration
    while app.is_running() and time.monotonic() < deadline:
        active = np.ones(1, dtype=bool)
        passed, snapshot, reasons = trial(env, app, limits, state, active, limits.replay_s,
                                          real_time=not args.headless)
        contacts = int((np.linalg.norm(snapshot["contact_forces"][0], axis=-1) > limits.contact_force_n).sum())
        print(f"[preview] index={args.index}, contacts={contacts}, "
              f"linear_speed={np.linalg.norm(snapshot['object_velocity'][0, :3]):.4f} m/s, "
              f"angular_speed={np.linalg.norm(snapshot['object_velocity'][0, 3:]):.4f} rad/s, "
              f"passed={bool(passed[0])}, failures={reasons}", flush=True)
        env.restore(state)
    return 0


def smoke(env, app, args, limits, directory):
    """Synthetic diagnostic only; does not mark a pose as user-approved or produce a cache."""
    import torch
    seed = tiled_seed(env.seed_pose, env.num_envs)
    # Offset positions per environment to catch both origin and index mapping errors.
    seed["object_pose"][:, 0] += np.arange(env.num_envs) * 0.0002
    env.restore(seed)
    np.testing.assert_allclose(env.snapshot()["object_pose"], seed["object_pose"], atol=1e-5)
    np.testing.assert_allclose(env.snapshot()["joint_pos"], seed["joint_pos"], atol=1e-5)
    base = env.hand.data.root_state_w.clone()
    for _ in range(30):
        env.step(torch.zeros_like(env.actions))
    torch.testing.assert_close(env.hand.data.root_state_w[:, :7], base[:, :7], atol=1e-5, rtol=0)
    np.testing.assert_allclose(env.snapshot()["joint_targets"], seed["joint_targets"], atol=1e-6)
    # Put balls far outside the hand: all finger-object contact channels must clear and gravity must act.
    free = tiled_seed(env.seed_pose, env.num_envs)
    free["object_pose"][:, 0] += 0.28
    free["object_pose"][:, 2] += 0.25
    env.restore(free)
    for _ in range(8):
        env.step(torch.zeros_like(env.actions))
    sample = env.snapshot()
    assert (sample["object_velocity"][:, 2] < -0.5).all(), "Gravity did not accelerate free balls"
    assert np.max(np.abs(sample["contact_forces"])) < 1e-5, "Unexpected finger-object contact away from the hand"
    # Deliberately intersect a fingertip in env 0 only; other environments must stay contact-free.
    from isaaclab.utils.math import quat_apply
    index = env.finger_bodies[0]
    tip = env.hand.data.body_pos_w[0, index] + quat_apply(
        env.hand.data.body_quat_w[0, index].unsqueeze(0), env.fingertip_offsets[0].unsqueeze(0))[0]
    free["object_pose"][0, :3] = (tip - env.scene.env_origins[0]).cpu().numpy()
    env.restore(free)
    saw_contact = False
    for _ in range(6):
        env.step(torch.zeros_like(env.actions))
        forces = env.snapshot()["contact_forces"]
        saw_contact |= bool(np.linalg.norm(forces[0], axis=-1).max() > 0.1)
        if env.num_envs > 1:
            assert np.max(np.abs(forces[1:])) < 1e-5, "Contact leaked between environments"
    assert saw_contact, "Filtered view failed to report a deliberate ball-finger collision"
    masses = env.object.root_physx_view.get_masses().cpu().numpy()
    np.testing.assert_allclose(masses, env.seed_pose["object"]["mass_kg"], atol=1e-6)
    for asset in (env.hand, env.object):
        mat = asset.root_physx_view.get_material_properties().cpu().numpy()
        np.testing.assert_allclose(mat[..., :2], 1.0)
        np.testing.assert_allclose(mat[..., 2], 0.0)
    assert not env.sim.stage.GetPrimAtPath("/World/ground").IsValid()
    write_json(directory / "smoke.json", dict(passed=True, num_envs=env.num_envs,
               physics=env.physics_metadata(), checks=["local_pose_roundtrip", "joint_mapping", "fixed_base",
               "held_targets", "gravity", "filtered_contacts_clear", "filtered_contacts_positive",
               "contact_environment_isolation", "mass", "materials", "no_ground"]))
    print(f"[smoke] Passed {env.num_envs} environments: {directory}", flush=True)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("generate", "preview", "smoke", "validate"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--duration", type=float, default=0, help="Preview wall seconds; 0 waits for window close")
    parser.add_argument("--num-envs", "--num_envs", type=int, default=256)
    parser.add_argument("--count", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-attempts", type=int, default=100000)
    parser.add_argument("--max-minutes", type=float, default=60)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/super_inair")
    for field in fields(Screening):
        parser.add_argument("--" + field.name.replace("_", "-"), type=int if field.name == "min_contacts" else float,
                            default=None)
    # Preload Windows DLL dependencies before Kit.
    import h5py  # noqa: F401
    import tensordict  # noqa: F401
    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.num_envs < 1 or args.count < 1 or args.max_attempts < 1 or not math.isfinite(args.max_minutes) or args.max_minutes <= 0:
        parser.error("Environment count, target count and run limits must be positive")
    states, metadata = None, {}
    if args.mode == "preview":
        if args.cache is None:
            parser.error("preview requires --cache <directory or grasps.npz>")
        states, metadata = load_cache(args.cache)
        if not 0 <= args.index < len(states["joint_pos"]):
            parser.error("Cache is empty or --index is out of range")
        seed = metadata["seed_pose"]
        args.num_envs = 1
    elif args.mode == "smoke":
        seed = load_seed(args.config)
    else:
        try:
            seed = load_seed(args.config)
        except (FileNotFoundError, ValueError, KeyError) as error:
            parser.error(f"Provide the saved tennis seed JSON: {args.config}. {error}")
    options = metadata.get("screening", {}) | {field.name: getattr(args, field.name) for field in fields(Screening)
                                               if getattr(args, field.name) is not None}
    limits = Screening(**options).validate()
    directory = args.output / (datetime.now().strftime("%Y%m%d_%H%M%S_%f") + "_" + args.mode)
    directory.mkdir(parents=True)
    if not args.kit_args:
        import isaacsim
        root = Path(isaacsim.__file__).resolve().parent
        folders = [root / name for name in ("exts", "extsDeprecated", "extscache", "extsPhysics")]
        folders += [root / "kit/data/Kit/Isaac-Sim/5.1/exts/3"]
        args.kit_args = "--/app/extensions/registryEnabled=0 " + " ".join(
            f'--ext-folder="{path.as_posix()}"'  for path in folders if path.exists())
    if seed.get("object", {}).get("id") != "tennis_ball_60":
        parser.error("Only tennis_ball_60 is supported")
    app = AppLauncher(args).app
    env = None
    try:
        from gex_learning.tasks.direct.super_inair.super_inair_env import Gx4mSuperInAirEnv, Gx4mSuperInAirEnvCfg
        cfg = Gx4mSuperInAirEnvCfg()
        cfg.seed_pose = seed
        cfg.scene.num_envs = args.num_envs
        cfg.sim.device = args.device
        cfg.seed = args.seed
        cfg.log_dir = str(directory)
        env = Gx4mSuperInAirEnv(cfg)
        env.reset()
        center = np.array(seed["object_pose"][:3]) + env.scene.env_origins[0].cpu().numpy()
        env.sim.set_camera_view(eye=center + np.array([0.35, -0.45, 0.3]), target=center)
        if args.mode == "generate":
            return generate(env, app, args, limits, directory)
        if args.mode == "validate":
            return validate_pose(env, app, limits, directory, args.max_minutes)
        if args.mode == "preview":
            try:
                return preview(env, app, args, limits, states)
            except (RunStopped, KeyboardInterrupt):
                return 0
        return smoke(env, app, args, limits, directory)
    finally:
        if env is not None:
            env.close()
        app.close()


if __name__ == "__main__":
    raise SystemExit(main())
