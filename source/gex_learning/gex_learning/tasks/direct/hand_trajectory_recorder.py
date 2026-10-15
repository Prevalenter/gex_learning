from __future__ import annotations

from datetime import datetime
from pathlib import Path
from queue import Queue
import re
from threading import Thread
from typing import Any

import numpy as np
import torch


FINGER_BASE_BODY_NAMES = ("link1", "link5", "link9", "link13")


class AsyncNpzChunkWriter:
    """Buffers trajectory frames and writes bounded compressed chunks in a worker thread."""

    def __init__(
        self,
        output_dir: str | Path,
        prefix: str,
        chunk_size: int,
        metadata: dict[str, Any],
        queue_size: int = 2,
    ) -> None:
        if chunk_size < 1:
            raise ValueError(f"chunk_size must be >= 1, got {chunk_size}")
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.prefix = prefix
        self.chunk_size = int(chunk_size)
        self.metadata = {key: np.asarray(value) for key, value in metadata.items()}
        self._queue: Queue[tuple[Path, dict[str, np.ndarray]] | None] = Queue(maxsize=queue_size)
        self._buffer: list[dict[str, np.ndarray]] = []
        self._keys: tuple[str, ...] | None = None
        self._last_step: int | None = None
        self._chunk_index = 0
        self._error: BaseException | None = None
        self._closed = False
        self._worker = Thread(target=self._write_worker, name=f"{prefix}-writer", daemon=True)
        self._worker.start()

    def record(self, step: int, fields: dict[str, np.ndarray]) -> bool:
        self._raise_worker_error()
        if self._closed:
            raise RuntimeError("Cannot record to a closed trajectory writer")
        step = int(step)
        if self._last_step == step:
            return False
        if self._last_step is not None and step < self._last_step:
            raise ValueError(f"Trajectory steps must be monotonic: {step} < {self._last_step}")

        frame = {"steps": np.asarray(step, dtype=np.int64)}
        frame.update({key: np.asarray(value) for key, value in fields.items()})
        keys = tuple(frame)
        if self._keys is None:
            self._keys = keys
        elif keys != self._keys:
            raise ValueError(f"Trajectory fields changed from {self._keys} to {keys}")
        self._buffer.append(frame)
        self._last_step = step
        if len(self._buffer) >= self.chunk_size:
            self._enqueue_chunk(partial=False)
        return True

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._enqueue_chunk(partial=True)
        self._queue.put(None)
        self._worker.join()
        self._raise_worker_error()

    def _enqueue_chunk(self, partial: bool) -> None:
        if not self._buffer:
            return
        frames = self._buffer
        self._buffer = []
        self._chunk_index += 1
        start_step = int(frames[0]["steps"])
        end_step = int(frames[-1]["steps"])
        data = {key: np.stack([frame[key] for frame in frames]) for key in frames[0]}
        data.update(self.metadata)
        data.update(
            chunk_index=np.asarray(self._chunk_index, dtype=np.int64),
            start_step=np.asarray(start_step, dtype=np.int64),
            end_step=np.asarray(end_step, dtype=np.int64),
            num_steps=np.asarray(len(frames), dtype=np.int64),
            partial=np.asarray(partial and len(frames) < self.chunk_size, dtype=np.bool_),
        )
        path = self.output_dir / (
            f"{self.prefix}_chunk_{self._chunk_index:06d}_steps_{start_step}_{end_step}.npz"
        )
        self._queue.put((path, data))
        self._raise_worker_error()

    def _write_worker(self) -> None:
        while True:
            item = self._queue.get()
            try:
                if item is None:
                    return
                path, data = item
                temporary_path = path.with_suffix(".tmp.npz")
                np.savez_compressed(temporary_path, **data)
                temporary_path.replace(path)
            except BaseException as error:
                self._error = error
            finally:
                self._queue.task_done()

    def _raise_worker_error(self) -> None:
        if self._error is not None:
            raise RuntimeError("Trajectory writer failed") from self._error


class HandTrajectorySampler:
    """Collects the common GX4M hand trajectory fields from an Isaac Lab environment."""

    def __init__(
        self,
        env,
        task_name: str,
        output_dir: str | Path,
        chunk_size: int,
        actuated_joint_names: list[str],
        actuated_joint_indices: list[int],
        fingertip_body_names: list[str],
        fingertip_offsets: list[list[float]] | tuple[tuple[float, float, float], ...],
        extra_metadata: dict[str, Any] | None = None,
        run_label: str = "trajectory",
        max_frames: int = 0,
    ) -> None:
        self.env = env
        self.actuated_joint_indices = list(actuated_joint_indices)
        self.fingertip_body_names = list(fingertip_body_names)
        self.finger_base_body_names = list(FINGER_BASE_BODY_NAMES)
        self.hand_root_body_name = "base_link"
        self.hand_root_body_index = self._find_name(env.hand.body_names, self.hand_root_body_name)
        self.fingertip_body_indices = [self._find_name(env.hand.body_names, name) for name in fingertip_body_names]
        self.finger_base_body_indices = [
            self._find_name(env.hand.body_names, name) for name in self.finger_base_body_names
        ]
        offsets = torch.as_tensor(fingertip_offsets, dtype=torch.float, device=env.device)
        if offsets.shape != (len(self.fingertip_body_indices), 3):
            raise ValueError(
                f"fingertip_offsets must have shape ({len(self.fingertip_body_indices)}, 3), "
                f"got {tuple(offsets.shape)}"
            )
        self.fingertip_offsets = offsets.unsqueeze(0)
        self.episode_reset = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
        self.episode_end = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self.episode_success = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self.max_frames = max(0, int(max_frames))
        self.num_frames = 0
        self.complete = False

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_label = re.sub(r"[^A-Za-z0-9_.-]+", "_", run_label).strip("_") or "trajectory"
        prefix = f"hand_trajectory_{safe_label}_{timestamp}"
        metadata: dict[str, Any] = {
            "env_ids": np.arange(int(env.num_envs), dtype=np.int64),
            "joint_names": np.asarray(actuated_joint_names, dtype=str),
            "fingertip_body_names": np.asarray(fingertip_body_names, dtype=str),
            "finger_base_body_names": np.asarray(self.finger_base_body_names, dtype=str),
            "hand_root_body_name": np.asarray(self.hand_root_body_name, dtype=str),
            "num_envs": np.asarray(int(env.num_envs), dtype=np.int64),
            "step_dt": np.asarray(float(env.step_dt), dtype=np.float64),
            "task_name": np.asarray(task_name, dtype=str),
            "coordinate_frame": np.asarray("environment_local_world", dtype=str),
            "position_unit": np.asarray("m", dtype=str),
            "joint_position_unit": np.asarray("rad", dtype=str),
            "interval": np.asarray(int(chunk_size), dtype=np.int64),
        }
        if extra_metadata:
            metadata.update(extra_metadata)
        self.writer = AsyncNpzChunkWriter(output_dir, prefix, chunk_size, metadata)
        print(f"[{task_name}] saving hand trajectory chunks to: {self.writer.output_dir / prefix}", flush=True)

    def mark_reset(self, env_ids: torch.Tensor, success: torch.Tensor | None = None) -> None:
        self.episode_reset[env_ids] = True
        self.episode_end[env_ids] = True
        if success is None:
            self.episode_success[env_ids] = False
        else:
            self.episode_success[env_ids] = success.to(device=self.env.device, dtype=torch.bool)

    def record(self, step: int, extra_fields: dict[str, torch.Tensor | np.ndarray] | None = None) -> bool:
        if self.complete:
            return False
        hand = self.env.hand
        env_origins = self.env.scene.env_origins
        fingertip_origin = hand.data.body_pos_w[:, self.fingertip_body_indices] - env_origins[:, None, :]
        fingertip_quat = hand.data.body_quat_w[:, self.fingertip_body_indices]
        fingertip_pos = fingertip_origin + self._quat_apply(fingertip_quat, self.fingertip_offsets)
        finger_base_pos = hand.data.body_pos_w[:, self.finger_base_body_indices] - env_origins[:, None, :]
        hand_root_pose = torch.cat(
            (
                hand.data.body_pos_w[:, self.hand_root_body_index] - env_origins,
                hand.data.body_quat_w[:, self.hand_root_body_index],
            ),
            dim=-1,
        )
        tensor_fields: dict[str, torch.Tensor] = {
            "joint_pos": hand.data.joint_pos[:, self.actuated_joint_indices],
            "fingertip_pos": fingertip_pos,
            "finger_base_pos": finger_base_pos,
            "hand_root_pose": hand_root_pose,
        }
        numpy_fields: dict[str, np.ndarray] = {}
        if extra_fields:
            for key, value in extra_fields.items():
                if isinstance(value, torch.Tensor):
                    tensor_fields[key] = value
                else:
                    numpy_fields[key] = np.asarray(value)

        field_shapes = {key: tuple(value.shape[1:]) for key, value in tensor_fields.items()}
        field_widths = {key: int(value[0].numel()) for key, value in tensor_fields.items()}
        packed = torch.cat([value.reshape(self.env.num_envs, -1) for value in tensor_fields.values()], dim=1)
        packed_cpu = packed.detach().cpu().numpy().copy()
        fields: dict[str, np.ndarray] = {}
        start = 0
        for key in tensor_fields:
            end = start + field_widths[key]
            fields[key] = packed_cpu[:, start:end].reshape(self.env.num_envs, *field_shapes[key])
            start = end
        fields.update(numpy_fields)

        episode_flags = torch.stack((self.episode_reset, self.episode_end, self.episode_success), dim=1)
        episode_flags_cpu = episode_flags.detach().cpu().numpy().copy()
        fields["episode_reset"] = episode_flags_cpu[:, 0]
        fields["episode_end"] = episode_flags_cpu[:, 1]
        fields["episode_success"] = episode_flags_cpu[:, 2]
        recorded = self.writer.record(step, fields)
        if recorded:
            self.episode_reset.zero_()
            self.episode_end.zero_()
            self.episode_success.zero_()
            self.num_frames += 1
            if self.max_frames > 0 and self.num_frames >= self.max_frames:
                self.writer.close()
                self.complete = True
        return recorded

    def close(self) -> None:
        if not self.complete:
            self.writer.close()
            self.complete = True

    @staticmethod
    def _quat_apply(quaternion: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
        xyz = quaternion[..., 1:]
        t = 2.0 * torch.cross(xyz, vector.expand_as(xyz), dim=-1)
        return vector + quaternion[..., :1] * t + torch.cross(xyz, t, dim=-1)

    @staticmethod
    def _find_name(names: list[str], target: str) -> int:
        for index, name in enumerate(names):
            if name == target or name.endswith((f"/{target}", f":{target}")):
                return index
        raise ValueError(f"Could not find body '{target}' in: {names}")
