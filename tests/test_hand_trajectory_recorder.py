from __future__ import annotations

from pathlib import Path
import sys

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "source" / "gex_learning"))

from gex_learning.tasks.direct.hand_trajectory_recorder import AsyncNpzChunkWriter, HandTrajectorySampler


def _record_frames(writer: AsyncNpzChunkWriter, count: int) -> None:
    for step in range(1, count + 1):
        writer.record(step, {"value": np.asarray([step, step + 1], dtype=np.float32)})


def test_partial_999_frames(tmp_path: Path) -> None:
    writer = AsyncNpzChunkWriter(tmp_path, "run", 1000, {"task_name": "test"})
    _record_frames(writer, 999)
    writer.close()
    files = list(tmp_path.glob("*.npz"))
    assert len(files) == 1
    with np.load(files[0], allow_pickle=False) as data:
        assert data["value"].shape == (999, 2)
        assert bool(data["partial"])


def test_exact_1000_frames(tmp_path: Path) -> None:
    writer = AsyncNpzChunkWriter(tmp_path, "run", 1000, {})
    _record_frames(writer, 1000)
    writer.close()
    files = list(tmp_path.glob("*.npz"))
    assert len(files) == 1
    with np.load(files[0], allow_pickle=False) as data:
        assert int(data["num_steps"]) == 1000
        assert not bool(data["partial"])


def test_1001_frames_create_tail_chunk(tmp_path: Path) -> None:
    writer = AsyncNpzChunkWriter(tmp_path, "run", 1000, {})
    _record_frames(writer, 1001)
    writer.close()
    files = sorted(tmp_path.glob("*.npz"))
    assert len(files) == 2
    with np.load(files[0], allow_pickle=False) as first, np.load(files[1], allow_pickle=False) as second:
        assert first["steps"].tolist() == list(range(1, 1001))
        assert second["steps"].tolist() == [1001]
        assert bool(second["partial"])


def test_duplicate_step_is_ignored(tmp_path: Path) -> None:
    writer = AsyncNpzChunkWriter(tmp_path, "run", 2, {})
    assert writer.record(1, {"value": np.asarray([1])})
    assert not writer.record(1, {"value": np.asarray([2])})
    writer.close()
    with np.load(next(tmp_path.glob("*.npz")), allow_pickle=False) as data:
        assert data["steps"].tolist() == [1]
        assert data["value"].tolist() == [[1]]


def test_quaternion_offset_rotation() -> None:
    quaternion = np.sqrt(0.5) * np.asarray([1.0, 0.0, 0.0, 1.0], dtype=np.float32)
    import torch

    rotated = HandTrajectorySampler._quat_apply(
        torch.from_numpy(quaternion).reshape(1, 1, 4),
        torch.tensor([[[1.0, 0.0, 0.0]]]),
    )
    np.testing.assert_allclose(rotated.numpy(), [[[0.0, 1.0, 0.0]]], atol=1.0e-6)
