import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from scripts import play_inhand_reorient as play


@pytest.fixture
def checkpoint(tmp_path):
    model = tmp_path / "model_2000.pt"
    model.touch()
    params = tmp_path / "params/env.yaml"
    params.parent.mkdir()
    params.write_text("""goal_mode: random
goal_rotation_axes: xyz
policy_joint_action_only: false
default_hand_joint_pos: [0.4, 0.5]
object_start_pos: [0.01, 0.1017, 0.6]
object_start_rot: !!python/tuple [1.0, 0.0, 0.0, 0.0]
reset_position_noise: 0.0
reset_position_noise_z: 0.0
reset_object_rotation_noise: 0.0
reset_dof_pos_noise: 0.0
reach_goal_bonus: 800.0
dist_reward_scale: -10.0
robot_cfg:
  spawn:
    merge_fixed_joints: true
ignored_slice: !!python/object/apply:builtins.slice [null, null, null]
""", encoding="utf-8")
    return model


def overrides(argv):
    return {arg.split("=", 1)[0]: arg.split("=", 1)[1] for arg in argv if arg.startswith("env.")}


def test_restores_checkpoint_pose_without_reading_live_editor(checkpoint):
    args = play._with_defaults(play._with_training_configuration(["--checkpoint", str(checkpoint)]))
    settings = overrides(args)
    assert json.loads(settings["env.object_start_pos"]) == [.01, .1017, .6]
    assert json.loads(settings["env.object_start_rot"]) == [1., 0., 0., 0.]
    assert settings["env.use_initial_pose_config"] == "False"
    assert settings["env.robot_cfg.spawn.merge_fixed_joints"] == "true"
    assert settings["env.reset_object_rotation_noise"] == "0.0"
    assert settings["env.reach_goal_bonus"] == "800.0"


def test_explicit_goal_and_pose_settings_take_precedence(checkpoint):
    args = [f"--checkpoint={checkpoint}", "env.goal_rotation_axes=z", "env.initial_pose_file=custom.json"]
    settings = overrides(play._with_training_configuration(args))
    assert settings["env.goal_rotation_axes"] == "z"
    assert settings["env.initial_pose_file"] == "custom.json"
    assert "env.default_hand_joint_pos" not in settings
    assert "env.use_initial_pose_config" not in settings


def test_missing_saved_configuration_preserves_training_articulation(tmp_path):
    args = play._with_defaults(play._with_training_configuration(["--checkpoint", str(tmp_path / "model.pt")]))
    assert overrides(args)["env.robot_cfg.spawn.merge_fixed_joints"] == "True"


def test_windows_backend_default_keeps_extension_paths_with_spaces(tmp_path, monkeypatch):
    install = tmp_path / "Isaac Sim"
    (install / "exts").mkdir(parents=True)
    monkeypatch.setitem(sys.modules, "isaacsim", SimpleNamespace(__file__=str(install / "__init__.py")))
    monkeypatch.setattr(sys, "platform", "win32")
    result = play._with_local_isaacsim_ext_folders([])
    kit = play._option_value(result, "--kit_args")
    assert '--/app/vulkan=false' in kit
    assert f'--ext-folder="{(install / "exts").as_posix()}"' in kit


@pytest.mark.parametrize("args", [["--kit_args", "--/app/vulkan=true"], ["--kit_args=--/app/vulkan=true"]])
def test_explicit_backend_is_preserved(args, monkeypatch):
    monkeypatch.setitem(sys.modules, "isaacsim", SimpleNamespace(__file__=__file__))
    monkeypatch.setattr(sys, "platform", "win32")
    result = play._with_local_isaacsim_ext_folders(args)
    assert play._option_value(result, "--kit_args") == "--/app/vulkan=true"
    assert result.count("--kit_args") == 1


def test_custom_kit_arguments_still_get_windows_backend_default(monkeypatch):
    monkeypatch.setitem(sys.modules, "isaacsim", SimpleNamespace(__file__=__file__))
    monkeypatch.setattr(sys, "platform", "win32")
    result = play._with_local_isaacsim_ext_folders(["--kit_args=--/app/other=true"])
    kit = play._option_value(result, "--kit_args")
    assert "--/app/other=true" in kit and "--/app/vulkan=false" in kit
