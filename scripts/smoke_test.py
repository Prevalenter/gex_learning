"""Run sequential short GPU training and checkpoint playback for all four tasks."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
TASKS = {
    "dexanygrasp": "gx4m_dexanygrasp_ducky",
    "dexanytwist": "gx4m_dexanytwist",
    "inhand_reorient": "gx4m_inhand_reorient",
    "super_inair": "gx4m_super_inair_tennis",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", nargs="+", choices=list(TASKS), default=list(TASKS))
    parser.add_argument("--timeout", type=int, default=240)
    parser.add_argument("--playback-only", action="store_true", help="Replay the latest migration smoke checkpoints.")
    args = parser.parse_args()
    output = ROOT / "outputs/validation"
    output.mkdir(parents=True, exist_ok=True)
    tag = "migration_" + time.strftime("%Y%m%d_%H%M%S")
    report = []
    # Kit helpers may retain the cwd briefly after the simulation process exits.
    with tempfile.TemporaryDirectory(prefix="gex_smoke_", ignore_cleanup_errors=True) as cwd:
        for task in args.tasks:
            checkpoint = None
            if args.playback_only:
                candidates = list((ROOT / "logs/rsl_rl" / TASKS[task]).glob("*migration_*/model_*.pt"))
                if candidates:
                    checkpoint = max(candidates, key=lambda p: p.stat().st_mtime)
            for phase in (("test",) if args.playback_only else ("train", "test")):
                if phase == "test" and checkpoint is None:
                    continue
                command = [sys.executable, str(ROOT / f"scripts/{phase}_{task}.py"), "--num_envs", "4"]
                if phase == "train":
                    command += ["--max_iterations", "1", "--run_name", tag, "--seed", "42"]
                else:
                    command += ["--checkpoint", str(checkpoint), "--duration", "3", "--headless", "--no-real-time"]
                # Two scales cover the min/max domain with four smoke environments;
                # the normal training configuration retains all 30 scale variants.
                overrides = ["env.scale_variant_count=2"] if task == "dexanytwist" else []
                command += overrides
                log = output / f"{tag}_{phase}_{task}.log"
                start = time.monotonic()
                with log.open("w", encoding="utf-8") as stream:
                    try:
                        result = subprocess.run(command, cwd=cwd, stdout=stream, stderr=subprocess.STDOUT,
                                                timeout=args.timeout)
                        status = result.returncode
                    except subprocess.TimeoutExpired:
                        status = "timeout"
                row = {"task": task, "phase": phase, "exit_code": status,
                       "seconds": round(time.monotonic() - start, 2), "log": log.relative_to(ROOT).as_posix(),
                       "overrides": overrides}
                log_text = log.read_text(encoding="utf-8", errors="replace")
                if any(s in log_text for s in ("Failed to resolve mesh", "Traceback (most recent call last)", "Could not open asset")):
                    row["error"] = "Runtime log contains an exception or unresolved model asset."
                marker = "Training time:" if phase == "train" else "finished after"
                if status == 0 and marker not in log_text:
                    row["error"] = "Runtime did not report successful completion."
                report.append(row)
                print(json.dumps(row), flush=True)
                if phase == "train" and status == 0:
                    candidates = list((ROOT / "logs/rsl_rl" / TASKS[task]).glob(f"*{tag}/model_*.pt"))
                    if candidates:
                        checkpoint = max(candidates, key=lambda p: p.stat().st_mtime)
                    else:
                        row["error"] = "Training exited without a checkpoint."
                (output / f"{tag}_results.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    expected = len(args.tasks) * (1 if args.playback_only else 2)
    if len(report) != expected or any(r["exit_code"] != 0 or "error" in r for r in report):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
