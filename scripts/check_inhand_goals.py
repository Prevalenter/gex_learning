"""Check that Cubic's random targets reach the actor in a real Isaac Lab environment.

Run with the Isaac Lab Python environment: python scripts/check_inhand_goals.py
This checks the observation contract and simulation steps, not policy convergence.
"""
from pathlib import Path
import sys

import h5py  # noqa: F401  # Preload HDF5 before Kit on Windows.
import tensordict  # noqa: F401
from isaaclab.app import AppLauncher


def main():
    app = AppLauncher(headless=True).app
    env = None
    try:
        import gymnasium as gym
        import torch
        from isaaclab_tasks.utils import parse_env_cfg

        root = Path(__file__).resolve().parents[1]
        sys.path.insert(0, str(root / "source/gex_learning"))
        import gex_learning.tasks  # noqa: F401

        task = "Gx4m-InHandReorient-Direct-v0"
        cfg = parse_env_cfg(task, device="cuda:0", num_envs=64)
        assert cfg.goal_mode == "random" and cfg.goal_rotation_axes == "xyz"
        assert not cfg.policy_joint_action_only
        cfg.seed = 42
        cfg.log_dir = str(root / "outputs/validation/inhand_goal_check")
        env = gym.make(task, cfg=cfg)
        env.reset()
        raw = env.unwrapped
        saved_goals = raw.goal_rot.clone()

        # Keep the physical state fixed and vary only the target. The actor must
        # see each change, whereas its joint/action/object inputs stay identical.
        raw.goal_rot[:] = torch.tensor([1., 0., 0., 0.], device=raw.device)
        baseline = raw._get_observations()["policy"].clone()
        assert baseline.shape == (64, 47)
        for axis in range(3):
            target = torch.zeros(4, device=raw.device)
            target[0] = target[axis + 1] = 2 ** -0.5
            raw.goal_rot[:] = target
            obs = raw._get_observations()["policy"]
            assert torch.allclose(obs[:, :39], baseline[:, :39])
            assert torch.allclose(obs[:, 39:43], target.expand(64, -1))
            assert not torch.allclose(obs[:, 43:47], baseline[:, 43:47])
            assert torch.isfinite(obs).all()
        raw.goal_rot[:] = saved_goals

        sampled = raw._sample_cubic_rotations(torch.arange(64, device=raw.device))
        assert torch.allclose(sampled.norm(dim=-1), torch.ones(64, device=raw.device), atol=1e-5)
        # A Z-only target has zero quaternion X/Y components. Check that targets
        # include rotations out of that plane, as requested by the XYZ task.
        assert (sampled[:, 1].abs() > 0.1).any() and (sampled[:, 2].abs() > 0.1).any()
        actions = torch.zeros((64, raw.cfg.action_space), device=raw.device)
        for _ in range(32):
            obs, rewards, _, _, _ = env.step(actions)
            assert obs["policy"].shape == (64, 47)
            assert torch.isfinite(obs["policy"]).all() and torch.isfinite(rewards).all()
        print("INHAND_GOAL_CHECK_PASSED: XYZ targets visible to 47D actor; 64 envs, 32 steps", flush=True)
    finally:
        if env is not None:
            env.close()
        app.close()


if __name__ == "__main__":
    main()
