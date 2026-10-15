from copy import deepcopy
from types import SimpleNamespace

import pytest

from gex_learning.tasks.direct.inhand_reorient.initial_pose import (
    JOINT_NAMES, apply_initial_pose, joint_limits, load_pose, save_pose, validate_pose,
)


def test_pose_roundtrip_normalizes_rotation_and_preserves_joint_names(tmp_path):
    data = load_pose()
    data["enabled"] = True
    data["object_orientation_wxyz"] = [2., 0., 0., 0.]
    path = save_pose(data, tmp_path / "pose.json")
    result = load_pose(path)
    assert result["object_orientation_wxyz"] == [1., 0., 0., 0.]
    assert tuple(result["joint_positions_rad"]) == JOINT_NAMES
    assert not path.with_suffix(".tmp").exists()


def test_pose_rejects_invalid_values():
    data = load_pose()
    data["object_orientation_wxyz"] = [0., 0., 0., 0.]
    with pytest.raises(ValueError, match="nonzero"):
        validate_pose(data)
    data = load_pose()
    data["joint_positions_rad"]["joint1"] = joint_limits()["joint1"][1] + .1
    with pytest.raises(ValueError, match="joint1"):
        validate_pose(data)
    data = load_pose()
    data["object_position_m"][0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        validate_pose(data)


def test_saved_pose_controls_resets_and_survives_reordered_joints(tmp_path):
    data = load_pose()
    data["enabled"] = True
    data["object_position_m"] = [.02, .09, .61]
    data["joint_positions_rad"]["joint1"] = .4
    path = save_pose(data, tmp_path / "pose.json")
    cfg = SimpleNamespace(use_initial_pose_config=True, initial_pose_file=str(path),
                          robot_cfg=SimpleNamespace(init_state=SimpleNamespace(joint_pos={})))
    names = list(reversed(JOINT_NAMES))
    apply_initial_pose(cfg, names)
    assert cfg.default_hand_joint_pos[-1] == .4
    assert cfg.object_start_pos == [.02, .09, .61]
    assert cfg.object_start_rot == data["object_orientation_wxyz"]
    assert cfg.reset_dof_pos_noise == cfg.reset_position_noise == cfg.reset_object_rotation_noise == 0.
    assert cfg.robot_cfg.init_state.joint_pos == data["joint_positions_rad"]
    assert not cfg.use_initial_pose_config


def test_disabled_pose_leaves_legacy_configuration_untouched(tmp_path):
    data = load_pose()
    data["enabled"] = False
    cfg = SimpleNamespace(use_initial_pose_config=True,
                          initial_pose_file=str(save_pose(data, tmp_path / "pose.json")),
                          default_hand_joint_pos=[.1] * 16, object_start_pos=[0., 0., .6],
                          reset_object_rotation_noise=3.14159)
    before = deepcopy(vars(cfg))
    apply_initial_pose(cfg, JOINT_NAMES)
    assert vars(cfg) == before
