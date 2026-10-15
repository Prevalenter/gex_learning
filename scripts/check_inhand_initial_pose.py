"""Verify that saved editor poses actually reach Isaac Lab resets."""
from pathlib import Path
import sys
import tempfile

import h5py  # noqa: F401
import tensordict  # noqa: F401
from isaaclab.app import AppLauncher

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source/gex_learning"))


def main():
    app = AppLauncher(headless=True).app
    env = None
    try:
        import gymnasium as gym
        import torch
        from isaaclab_tasks.utils import parse_env_cfg
        import gex_learning.tasks  # noqa: F401
        from gex_learning.tasks.direct.inhand_reorient.initial_pose import JOINT_NAMES, load_pose, save_pose

        with tempfile.TemporaryDirectory(prefix="inhand_pose_check_") as temp:
            data = load_pose()
            data["enabled"] = True
            data["joint_positions_rad"]["joint1"] = .42
            data["object_position_m"] = [.025, .095, .61]
            data["object_orientation_wxyz"] = [.9238795325, .3826834324, 0., 0.]
            pose_path = save_pose(data, Path(temp) / "initial_pose.json")
            cfg = parse_env_cfg("Gx4m-InHandReorient-Direct-v0", device="cuda:0", num_envs=4)
            cfg.initial_pose_file = str(pose_path)
            cfg.seed = 42
            cfg.enable_object_size_randomization = False
            for name in vars(cfg.events):
                if not name.startswith("_"):
                    setattr(cfg.events, name, None)
            env = gym.make("Gx4m-InHandReorient-Direct-v0", cfg=cfg)
            raw = env.unwrapped
            for _ in range(2):
                env.reset()
                q = torch.tensor([data["joint_positions_rad"][name] for name in JOINT_NAMES], device=raw.device)
                torch.testing.assert_close(raw.hand.data.joint_pos[:, raw.actuated_dof_indices], q.expand(4, -1),
                                           atol=1e-5, rtol=1e-5)
                pos = torch.tensor(data["object_position_m"], device=raw.device)
                torch.testing.assert_close(raw.object.data.root_pos_w - raw.scene.env_origins,
                                           pos.expand(4, -1), atol=1e-5, rtol=1e-5)
                quat = torch.tensor(data["object_orientation_wxyz"], device=raw.device)
                assert torch.all(torch.abs((raw.object.data.root_quat_w * quat).sum(-1)) > .99999)
            for _ in range(8):
                obs, rewards, _, _, _ = env.step(torch.zeros((4, 16), device=raw.device))
                assert obs["policy"].shape == (4, 47)
                assert torch.isfinite(obs["policy"]).all() and torch.isfinite(rewards).all()
            print("INHAND_INITIAL_POSE_CHECK_PASSED: joints, position and rotation restored on two resets; 8 steps", flush=True)
    finally:
        if env is not None:
            env.close()
        app.close()


if __name__ == "__main__":
    main()
