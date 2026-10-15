"""Bound an RSL-RL learning run by wall time and save at an iteration boundary."""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import json
import math
import os
from pathlib import Path
import time


def consume_time_limit(argv: list[str]) -> tuple[list[str], float | None]:
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--max-training-hours", type=float)
    options, forwarded = parser.parse_known_args(argv)
    hours = options.max_training_hours
    if hours is not None and (not math.isfinite(hours) or hours <= 0):
        parser.error("--max-training-hours must be a finite positive number")
    return forwarded, hours


def install_training_time_limit(runner_class, hours: float) -> None:
    original_learn = runner_class.learn

    class TimeLimitReached(Exception):
        pass

    def timed_learn(self, num_learning_iterations, init_at_random_ep_len=False):
        start = time.monotonic()
        started_at = datetime.now(timezone.utc)
        duration = hours * 3600
        log_dir = Path(self.logger.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        status_path = log_dir / "training_time_limit.json"
        status = {
            "status": "running", "pid": os.getpid(), "limit_hours": hours,
            "started_at_utc": started_at.isoformat(),
            "deadline_utc": (started_at + timedelta(seconds=duration)).isoformat(),
            "stop_policy": "Save and stop after the first completed iteration at or after the deadline",
        }
        last_status_time = start
        original_log = self.logger.log

        def write_status():
            status["elapsed_seconds"] = time.monotonic() - start
            status["iteration"] = int(self.current_learning_iteration)
            temporary = status_path.with_suffix(".tmp")
            temporary.write_text(json.dumps(status, indent=2), encoding="utf-8")
            temporary.replace(status_path)

        def timed_log(*args, **kwargs):
            nonlocal last_status_time
            result = original_log(*args, **kwargs)
            now = time.monotonic()
            if now - start >= duration:
                raise TimeLimitReached
            if now - last_status_time >= 30:
                write_status()
                last_status_time = now
            return result

        write_status()
        self.logger.log = timed_log
        print(f"[training-time-limit] {hours:g} hours; deadline UTC {status['deadline_utc']}", flush=True)
        try:
            try:
                original_learn(self, num_learning_iterations, init_at_random_ep_len)
            except TimeLimitReached:
                checkpoint = log_dir / f"model_{self.current_learning_iteration}.pt"
                self.save(str(checkpoint))
                self.logger.stop_logging_writer()
                status.update(status="time_limit_reached", final_checkpoint=str(checkpoint.resolve()))
                print(f"[training-time-limit] saved {checkpoint}; training finished.", flush=True)
            else:
                status["status"] = "iterations_complete"
        except BaseException as error:
            status.update(status="failed", error=repr(error))
            raise
        finally:
            self.logger.log = original_log
            if status["status"] != "running":
                status["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
            write_status()

    runner_class.learn = timed_learn
