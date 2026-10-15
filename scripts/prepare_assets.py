"""Copy the minimal asset manifest from an authorized local asset bundle."""
from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]


def required_files() -> list[str]:
    return json.loads((ROOT / "asset/manifest.json").read_text(encoding="utf-8"))["files"]


def normalize_urdf(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    # All selected URDFs have meshes beside their urdf directory.
    normalized = re.sub(r"package://[^/]+/meshes/", "../meshes/", text)
    if normalized != text:
        path.write_text(normalized, encoding="utf-8")


def check_assets() -> list[str]:
    errors = []
    for name in required_files():
        path = ROOT / name
        if not path.is_file():
            errors.append(f"Missing: {name}")
            continue
        if path.suffix == ".urdf":
            for mesh in ET.parse(path).findall(".//mesh"):
                filename = mesh.get("filename", "")
                target = (path.parent / filename).resolve()
                if "://" in filename or not target.is_relative_to(ROOT) or not target.is_file():
                    errors.append(f"Unresolved mesh: {name}: {filename}")
        elif path.suffix.lower() in (".obj", ".mtl"):
            for line in path.read_text(encoding="utf-8").splitlines():
                fields = line.strip().split(maxsplit=1)
                if len(fields) != 2 or fields[0] not in ("mtllib", "map_Kd", "map_Ka", "map_Ks", "map_d", "bump", "map_Bump"):
                    continue
                target = (path.parent / fields[1].strip()).resolve()
                if not target.is_relative_to(ROOT) or not target.is_file():
                    errors.append(f"Unresolved material/texture: {name}: {fields[1]}")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, help="Authorized asset directory with gx4m/, franka_gx4m/, etc.")
    parser.add_argument("--check", action="store_true", help="Check existing files without copying.")
    args = parser.parse_args()
    if not args.check and args.source is None:
        parser.error("Pass --source <asset-directory> or --check; see asset/README.md.")
    if args.source is not None and not args.check:
        source = args.source.expanduser().resolve()
        missing = [name for name in required_files() if not (source / Path(name).relative_to("asset")).is_file()]
        if missing:
            raise SystemExit("Source bundle is incomplete:\n" + "\n".join(missing))
        for name in required_files():
            origin = source / Path(name).relative_to("asset")
            dest = ROOT / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            if origin.resolve() != dest.resolve():
                shutil.copy2(origin, dest)
            if dest.suffix == ".urdf":
                normalize_urdf(dest)
    errors = check_assets()
    if errors:
        raise SystemExit("\n".join(errors))
    print(f"OK: {len(required_files())} required asset files; URDF mesh references resolve locally.")


if __name__ == "__main__":
    main()
