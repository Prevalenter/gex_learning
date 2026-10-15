"""Check the release boundary and the single-object observation contract."""
from __future__ import annotations

import ast
import json
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

import gymnasium as gym
import pytest

import gex_learning.tasks  # noqa: F401
from gex_learning.tasks.direct.dexanytwist.dataset import load_hand_spec, load_twist_object, load_twist_object_names
from scripts.prepare_assets import check_assets, required_files

ROOT = Path(__file__).resolve().parents[1]
TASKS = ("dexanygrasp", "dexanytwist", "inhand_reorient", "super_inair")


def test_only_four_project_tasks_registered():
    ids = {name for name in gym.registry if name.startswith("Gx4m-")}
    assert ids == {
        "Gx4m-DexAnyGrasp-Direct-v0", "Gx4m-DexAnyTwist-Direct-v0",
        "Gx4m-InHandReorient-Direct-v0",
        "Gx4m-SuperInAir-Rotation-Direct-v0",
    }
    for name in ids:
        assert gym.spec(name).entry_point.startswith("gex_learning.")


@pytest.mark.parametrize("split", ["train", "train_all", "all", "dataset_all", "cls", "configs"])
def test_only_selected_bulb_in_splits(split):
    assert load_twist_object_names(split) == ["bulb_custom_0017"]


def test_bulb_class_index_and_features():
    if check_assets():
        pytest.skip("Prepare external assets to run the asset integration test.")
    spec = load_twist_object()
    assert spec.name == "bulb_custom_0017"
    assert spec.class_index == 93
    assert spec.points_feature.shape == (128,)
    assert spec.keypoint_bias.shape == (4, 3)
    assert len(spec.action_mean_joint_pos) == 16


def test_removed_object_fails_clearly():
    with pytest.raises(KeyError, match="not found"):
        load_twist_object("not_in_release")


def test_urdf_asset_closure():
    missing = [name for name in required_files() if not (ROOT / name).is_file()]
    if missing:
        pytest.skip("Prepare external assets to run the asset integration test.")
    assert check_assets() == []


def test_prepared_hand_meshes_resolve_from_temporary_location():
    if check_assets():
        pytest.skip("Prepare external assets to run the asset integration test.")
    hand = load_hand_spec()
    meshes = ET.parse(hand.urdf_path).findall(".//mesh")
    assert meshes
    for mesh in meshes:
        assert Path(mesh.get("filename")).is_file()


def test_configurator_lists_only_registered_object():
    from scripts.configure_dexanytwist_viser import discover_twist_objects
    assert discover_twist_objects() == {"bulb_custom": ["bulb_custom_0017"]}


def test_launch_defaults_are_consistent():
    for prefix in ("train", "play", "test"):
        tree = ast.parse((ROOT / f"scripts/{prefix}_dexanytwist.py").read_text(encoding="utf-8"))
        constants = {n.targets[0].id: n.value.value for n in tree.body
                     if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
                     and isinstance(n.value, ast.Constant)}
        assert constants["OBJECT_NAME"] == "bulb_custom_0017"
    for name in ("train", "play", "sample", "configure_viser"):
        text = (ROOT / f"experiment/dexanytwist/{name}.ps1").read_text(encoding="utf-8")
        assert "bulb_custom_0017" in text


def test_project_has_no_excluded_imports_or_personal_paths():
    forbidden = ("gx4m_learning", "super_inhand", "dexlong", "dexrio_grasp")
    for folder in ("source", "scripts", "experiment", "apps"):
        for path in (ROOT / folder).rglob("*"):
            if path.suffix not in {".py", ".json", ".ps1", ".kit"}:
                continue
            text = path.read_text(encoding="utf-8")
            assert not any(name in text for name in forbidden), str(path)
            assert "43462" not in text, str(path)


def test_asset_check_from_different_working_directory(tmp_path):
    if check_assets():
        pytest.skip("Prepare external assets to run the asset integration test.")
    result = subprocess.run([sys.executable, str(ROOT / "scripts/prepare_assets.py"), "--check"],
                            cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "OK:" in result.stdout


def test_runtime_paths_missing_checkout_and_invalid_override(monkeypatch, tmp_path):
    from scripts import runtime_paths
    from scripts.runtime_paths import isaaclab_root
    monkeypatch.delenv("ISAACLAB_ROOT", raising=False)
    monkeypatch.setattr(runtime_paths, "find_spec", lambda _: None)
    with pytest.raises(RuntimeError, match="ISAACLAB_ROOT"):
        isaaclab_root()
    monkeypatch.setenv("ISAACLAB_ROOT", str(tmp_path))
    with pytest.raises(FileNotFoundError, match="RSL-RL"):
        isaaclab_root()


def test_runtime_paths_discover_editable_install(monkeypatch, tmp_path):
    from types import SimpleNamespace
    from scripts import runtime_paths
    checkout = tmp_path / "Isaac Lab"
    train = checkout / "scripts/reinforcement_learning/rsl_rl/train.py"
    train.parent.mkdir(parents=True)
    train.touch()
    spec = SimpleNamespace(submodule_search_locations=[str(checkout / "source/isaaclab/isaaclab")])
    monkeypatch.delenv("ISAACLAB_ROOT", raising=False)
    monkeypatch.setattr(runtime_paths, "find_spec", lambda _: spec)
    assert runtime_paths.isaaclab_root() == checkout.resolve()
    monkeypatch.setenv("ISAACLAB_ROOT", str(tmp_path / "invalid"))
    with pytest.raises(FileNotFoundError, match="RSL-RL"):
        runtime_paths.isaaclab_root()


def test_anygrasp_play_keeps_training_robot_asset(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    from scripts.play_dexanygrasp import _with_defaults
    argv = _with_defaults(["--checkpoint", "model_1000.pt"])
    assert not any(arg.startswith("env.arm_cfg.spawn.usd_path=") for arg in argv)
    override = "env.arm_cfg.spawn.usd_path=custom.usd"
    assert _with_defaults(["--checkpoint", "model_1000.pt", override]).count(override) == 1


def test_runtime_paths_support_checkout_with_spaces(monkeypatch, tmp_path):
    from scripts.runtime_paths import isaaclab_root
    checkout = tmp_path / "Isaac Lab with spaces"
    train = checkout / "scripts/reinforcement_learning/rsl_rl/train.py"
    train.parent.mkdir(parents=True)
    train.touch()
    monkeypatch.setenv("ISAACLAB_ROOT", str(checkout))
    assert isaaclab_root() == checkout.resolve()
