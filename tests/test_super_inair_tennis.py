"""Check the tennis-only migration boundary and portable cache lookup."""
import json
from pathlib import Path

import pytest
import torch

from gex_learning.tasks.direct.super_inair.grasp_data import load_seed
from gex_learning.tasks.direct.super_inair.mixed_cache import load_mixed_cache
from gex_learning.tasks.direct.super_inair.object_assets import selected_objects
from gex_learning.tasks.direct.super_inair.object_observations import OBJECT_ID_NAMES, object_id_codes

ROOT = Path(__file__).resolve().parents[1]


def test_tennis_only_assets_and_original_observation_slot():
    if not (ROOT / "asset/HORA/selected/tennis_ball_60.urdf").exists():
        pytest.skip("Prepare tennis assets first")
    assert list(selected_objects()) == ["tennis_ball_60"]
    seed = load_seed()
    assert seed["object"]["id"] == "tennis_ball_60"
    assert seed["object"]["mass_kg"] == .05
    torch.testing.assert_close(object_id_codes(["tennis_ball_60"], list(OBJECT_ID_NAMES), "cpu"),
                               torch.tensor([[0., 0., 0., 0., 0., 1.]]))


def test_manifest_rejects_other_objects_before_loading_cache(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"version": 1, "status": "complete", "objects": [
        {"id": "cube_42", "cache": "missing", "count": 1}]}), encoding="utf-8")
    with pytest.raises(ValueError, match="Only tennis_ball_60"):
        load_mixed_cache(manifest)


def test_tennis_cache_resolves_independently_of_working_directory(tmp_path, monkeypatch):
    if not (ROOT / "outputs/super_inair/tennis_grasps/grasps.npz").exists():
        pytest.skip("Local training cache is intentionally excluded from the release")
    monkeypatch.chdir(tmp_path)
    ids, seeds, states, offsets, counts = load_mixed_cache("experiment/super_inair/tennis_manifest.json")
    assert ids == ["tennis_ball_60"]
    assert seeds[0]["object"]["id"] == ids[0]
    assert offsets == [0] and counts == [len(states["joint_pos"])]


def test_tennis_entry_defaults_match():
    from scripts.train_super_inair import _with_defaults as train_defaults
    from scripts.play_super_inair import _with_defaults as play_defaults
    for args in (train_defaults([]), play_defaults(["--checkpoint", "model.pt"])):
        assert args[args.index("--task") + 1] == "Gx4m-SuperInAir-Rotation-Direct-v0"
        assert args[args.index("--experiment_name") + 1] == "gx4m_super_inair_tennis"
