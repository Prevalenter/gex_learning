import math
import torch

from gex_learning.tasks.direct.super_inair.reward_terms import cross_axis_orientation_cost


def rotation(axis, degrees):
    axis = torch.tensor(axis, dtype=torch.float64)
    axis = axis / axis.norm()
    a = math.radians(degrees) / 2
    return torch.cat((torch.tensor([math.cos(a)], dtype=axis.dtype), axis * math.sin(a)))


def multiply(q, p):
    return torch.cat((q[:1] * p[:1] - (q[1:] * p[1:]).sum().reshape(1),
                      q[:1] * p[1:] + p[:1] * q[1:] + torch.linalg.cross(q[1:], p[1:])))


def test_pure_target_twist_has_no_cost_for_arbitrary_initial_pose_and_axis():
    axis = torch.tensor([1., 2., -3.], dtype=torch.float64)
    initial = rotation([0, 1, 1], 63)
    for angle in [0, 5, 90, 180, 270, 360, 720]:
        current = multiply(rotation(axis.tolist(), angle), initial)
        cost, swing = cross_axis_orientation_cost(current[None], initial[None], axis)
        assert cost.item() == 0
        assert swing.item() < 1e-12


def test_deadband_penalty_quaternion_sign_and_swing_twist():
    initial = rotation([1, 1, 0], 37)
    axis = torch.tensor([1., 0., 0.], dtype=torch.float64)
    for degrees, expected in [(0, 0), (4, 0), (5, 0), (10, 1), (15, 4), (180, 1225)]:
        for twist in [0, 90, 180, 360]:
            delta = multiply(rotation([0, 1, 0], degrees), rotation([1, 0, 0], twist))
            current = multiply(delta, initial)
            cost, angle = cross_axis_orientation_cost(current[None], initial[None], axis)
            torch.testing.assert_close(cost, torch.tensor([expected], dtype=torch.float64), atol=1e-10, rtol=1e-10)
            torch.testing.assert_close(angle, torch.tensor([math.radians(degrees)], dtype=torch.float64))
            flipped, _ = cross_axis_orientation_cost(-current[None], initial[None], axis)
            torch.testing.assert_close(flipped, cost)


def test_reset_one_environment_preserves_other_references():
    initial = torch.stack([rotation([1, 0, 0], 0)] * 2)
    current = torch.stack([rotation([0, 0, 1], 10), rotation([0, 1, 0], 15)])
    initial[0] = current[0]
    cost, _ = cross_axis_orientation_cost(current, initial, torch.tensor([1., 0., 0.]))
    torch.testing.assert_close(cost, torch.tensor([0., 4.], dtype=torch.float64))


def test_twenty_degree_deadband_and_scale():
    axis = torch.tensor([1., 0., 0.], dtype=torch.float64)
    initial = rotation([0, 1, 1], 42)
    for degrees, penalty in [(0, 0), (20, 0), (30, -.025), (40, -.1), (60, -.4)]:
        delta = multiply(rotation([0, 1, 0], degrees), rotation([1, 0, 0], 137))
        cost, _ = cross_axis_orientation_cost(multiply(delta, initial)[None], initial[None], axis, 20)
        torch.testing.assert_close(-.1*cost, torch.tensor([penalty], dtype=torch.float64), atol=1e-10, rtol=1e-10)
