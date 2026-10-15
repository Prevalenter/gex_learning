"""Validate USD layers and their dependencies using the installed Isaac Sim runtime."""
from pathlib import Path
import re

from isaaclab.app import AppLauncher


def main():
    launcher = AppLauncher(headless=True)
    from pxr import Sdf

    root = Path(__file__).resolve().parents[1]
    errors = []
    try:
        for path in sorted((root / "asset").rglob("*.usd")):
            layer = Sdf.Layer.FindOrOpen(str(path))
            if layer is None:
                errors.append(f"Cannot open {path}")
                continue
            for dep in layer.GetExternalReferences():
                if not dep:
                    continue
                target = (path.parent / dep).resolve()
                if "://" in dep or not target.is_relative_to(root) or not target.is_file():
                    errors.append(f"{path.relative_to(root)} -> {dep}")
            text = layer.ExportToString()
            if re.search(r"[A-Za-z]:[/\\]|/home/|/Users/", text):
                errors.append(f"Machine-specific path in {path.relative_to(root)}")
        if errors:
            raise RuntimeError("Unresolved USD dependencies:\n" + "\n".join(errors))
        print("USD_ASSET_CHECK_PASSED", flush=True)
    finally:
        launcher.app.close()


if __name__ == "__main__":
    main()
