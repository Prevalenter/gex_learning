"""Resolve external runtime paths without machine-specific defaults."""
from __future__ import annotations

import os
from importlib.util import find_spec
from pathlib import Path


def isaaclab_root() -> Path:
    value = os.environ.get("ISAACLAB_ROOT")
    if value:
        root = Path(value).expanduser().resolve()
        if not (root / "scripts/reinforcement_learning/rsl_rl/train.py").is_file():
            raise FileNotFoundError(f"ISAACLAB_ROOT does not contain the RSL-RL scripts: {root}")
        return root

    # Editable Isaac Lab installs point back to the checkout. Inspect the spec
    # without importing Isaac Lab before the simulator has been launched.
    spec = find_spec("isaaclab")
    for location in (spec.submodule_search_locations or ()) if spec else ():
        for root in Path(location).resolve().parents:
            if (root / "scripts/reinforcement_learning/rsl_rl/train.py").is_file():
                return root
    raise RuntimeError(
        "Could not locate an Isaac Lab checkout from the active Python environment. "
        "Set ISAACLAB_ROOT to your Isaac Lab checkout before launching simulation."
    )
