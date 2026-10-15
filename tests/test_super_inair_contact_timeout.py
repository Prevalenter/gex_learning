import torch
from gex_learning.tasks.direct.super_inair.reward_terms import fingertip_contact_timeout


def test_timeout_boundary_persistence_and_independent_recontact():
    steps = torch.zeros((2, 4), dtype=torch.long)
    forces = torch.zeros((2, 4, 3))
    for _ in range(6):
        steps, overdue = fingertip_contact_timeout(steps, forces, 1/30, .2, .1)
        assert not overdue.any()
    steps, overdue = fingertip_contact_timeout(steps, forces, 1/30, .2, .1)
    assert overdue.all()
    assert torch.allclose(-.1 * overdue.sum(-1), torch.tensor([-.4, -.4]))
    forces[0, 1, 0] = .11
    steps, overdue = fingertip_contact_timeout(steps, forces, 1/30, .2, .1)
    assert steps[0, 1] == 0
    assert overdue.sum().item() == 7
    forces.zero_()
    steps, overdue = fingertip_contact_timeout(steps, forces, 1/30, .2, .1)
    assert steps[0, 1] == 1 and not overdue[0, 1]
    assert overdue[1].all()


def test_new_04_second_boundary_and_015_scale():
    steps = torch.zeros((1, 4), dtype=torch.long)
    forces = torch.zeros((1, 4, 3))
    for _ in range(12):
        steps, overdue = fingertip_contact_timeout(steps, forces, 1/30, .4, .1)
        assert not overdue.any()
    steps, overdue = fingertip_contact_timeout(steps, forces, 1/30, .4, .1)
    assert overdue.all()
    assert torch.allclose(-.15 * overdue.sum(-1), torch.tensor([-.6]))
