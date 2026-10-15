"""Categorical object identity with a checkpoint-persisted vocabulary."""
import torch

OBJECT_ID_NAMES = ('cylinder_d60_h48', 'cylinder_d60_h60', 'cylinder_d60_h72',
                   'cube_42', 'cuboid_40_45_50', 'tennis_ball_60')


def object_id_codes(names, vocabulary, device):
    if not names or not vocabulary or len(set(vocabulary)) != len(vocabulary):
        raise ValueError('Object ID observations require known names and a unique vocabulary')
    if any(name not in vocabulary for name in names):
        raise ValueError('Object ID is missing from the observation vocabulary')
    indices = torch.tensor([vocabulary.index(name) for name in names], device=device)
    return torch.eye(len(vocabulary), device=device)[indices]


def write_object_id_observations(observations, codes, critic_command_start):
    # Reserve base pose slots even if the base pose observation is disabled.
    policy_start, critic_start = 80, critic_command_start + 10
    for name, start in [('policy', policy_start), ('critic', critic_start)]:
        obs = observations[name]
        if codes.ndim != 2 or len(codes) != len(obs) or start + codes.shape[1] > obs.shape[1]:
            raise ValueError(f'Insufficient {name} padding for object ID observations')
        obs[:, start:start+codes.shape[1]] = codes
