"""Install a validated local tennis grasp cache, or check the current one."""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source/gex_learning"))
from gex_learning.tasks.direct.super_inair.grasp_data import load_cache, write_json
from gex_learning.tasks.direct.super_inair.mixed_cache import load_mixed_cache

MANIFEST = ROOT / "experiment/super_inair/tennis_manifest.json"


def prepare(cache: Path) -> int:
    cache = cache.expanduser().resolve()
    states, metadata = load_cache(cache)
    count = len(states["joint_pos"])
    if (metadata.get("status") != "complete" or count == 0
            or metadata["seed_pose"].get("object", {}).get("id") != "tennis_ball_60"):
        raise ValueError("A complete, nonempty tennis_ball_60 cache is required")
    source = cache if cache.is_dir() else cache.parent
    target = ROOT / "outputs/super_inair/tennis_grasps"
    target.mkdir(parents=True, exist_ok=True)
    for name in ("grasps.npz", "metadata.json"):
        if (source / name).resolve() != (target / name).resolve():
            shutil.copy2(source / name, target / name)
    manifest = {"version": 1, "status": "complete", "objects": [
        {"id": "tennis_ball_60", "cache": "../../outputs/super_inair/tennis_grasps", "count": count}]}
    write_json(MANIFEST, manifest)
    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, help="Validated cache directory containing grasps.npz and metadata.json")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if not args.check and args.cache is None:
        parser.error("Pass --cache <directory> or --check; see docs/SUPER_INAIR.md")
    if args.cache is not None and not args.check:
        prepare(args.cache)
    try:
        ids, _, states, _, _ = load_mixed_cache(MANIFEST)
    except FileNotFoundError as error:
        raise SystemExit(f"Missing tennis assets or cache: {error}. See docs/SUPER_INAIR.md for preparation steps.")
    print(f"OK: {ids[0]}, {len(states['joint_pos'])} validated grasps")


if __name__ == "__main__":
    main()
