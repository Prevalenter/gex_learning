"""Shared, simulation-independent identity and geometry for selected HORA objects."""
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[6]
SELECTION = ROOT / 'asset/HORA/selected/selection.json'
@lru_cache(maxsize=1)
def selected_objects():
    data = json.loads(SELECTION.read_text(encoding='utf-8'))
    result = {}
    if [item['id'] for item in data['objects']] != ['tennis_ball_60']:
        raise ValueError('Only tennis_ball_60 is supported')
    for item in data['objects']:
        path = (SELECTION.parent.parent / item['path']).resolve()
        if not path.is_relative_to(ROOT) or not path.is_file():
            raise ValueError(f'Invalid object path: {path}')
        result[item['id']] = dict(id=item['id'], urdf_path=path.relative_to(ROOT).as_posix(),
            urdf_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            dimensions_m=item['collision_dimensions_m'], mass_kg=item['mass_kg'])
    return result


def validate_object(raw):
    expected = selected_objects().get(raw.get('id'))
    if expected is None or raw != expected:
        raise ValueError('Object ID, asset, dimensions or mass do not match the selected asset')
    # Return an independent JSON-compatible copy.
    return json.loads(json.dumps(expected))


def seed_object_size(seed):
    return max(seed['object']['dimensions_m']) if seed['version'] == 2 else seed['ball_diameter_m']


def seed_object_mass(seed):
    return seed['object']['mass_kg'] if seed['version'] == 2 else seed['ball_mass_kg']


@lru_cache(maxsize=16)
def object_mesh(object_id, collision=False):
    """Load the same centered primitive/mesh geometry used for validation.

    HORA's near-spherical collision mesh uses PhysX's convex hull approximation.
    Show this hull in the editor so its collision overlay matches that setting.
    """
    return urdf_mesh(ROOT / selected_objects()[object_id]['urdf_path'], collision)


def urdf_mesh(path, collision=False):
    """Combine all pieces in link coordinates; convexify each collider separately."""
    import trimesh
    path = Path(path)
    root = ET.parse(path).getroot()
    kind = 'collision' if collision else 'visual'
    elements = root.findall(f'link/{kind}')
    meshes = [_geometry_mesh(path, root, element, collision) for element in elements]
    if not meshes:
        raise ValueError(f'No {kind} geometry: {path}')
    return meshes[0] if len(meshes) == 1 else trimesh.util.concatenate(meshes)


def _geometry_mesh(path, root, element, collision):
    import numpy as np
    import trimesh
    geometry = element.find('geometry')[0]
    if geometry.tag == 'cylinder':
        mesh = trimesh.creation.cylinder(radius=float(geometry.get('radius')),
                                         height=float(geometry.get('length')), sections=64)
    elif geometry.tag == 'box':
        mesh = trimesh.creation.box(extents=list(map(float, geometry.get('size').split())))
    elif geometry.tag == 'sphere':
        mesh = trimesh.creation.icosphere(radius=float(geometry.get('radius')))
    else:
        mesh = trimesh.load(path.parent/geometry.get('filename'), force='mesh', process=False)
        mesh.apply_scale(list(map(float, geometry.get('scale', '1 1 1').split())))
        if collision:
            mesh = mesh.convex_hull
    origin = element.find('origin')
    if origin is not None:
        rpy = list(map(float, origin.get('rpy', '0 0 0').split()))
        transform = trimesh.transformations.euler_matrix(*rpy)
        transform[:3, 3] = np.array(list(map(float, origin.get('xyz', '0 0 0').split())))
        mesh.apply_transform(transform)
    if not collision:
        material = element.find('material')
        if material is not None:
            color = material.find('color')
            if color is None:
                named = root.find(f'material[@name="{material.get("name")}"]')
                color = named.find('color') if named is not None else None
            if color is not None:
                mesh.visual.face_colors = np.rint(np.array(list(map(float, color.get('rgba').split()))) * 255).astype(np.uint8)
    return mesh


def normalized_single_body_urdf(source, output):
    """Give mixed assets a common rigid-body path without changing geometry/inertia."""
    source, output = Path(source).resolve(), Path(output).resolve()
    tree = ET.parse(source)
    links = tree.getroot().findall('link')
    if len(links) != 1 or tree.getroot().findall('joint'):
        raise ValueError('Mixed object URDF must contain exactly one link and no joints')
    links[0].set('name', 'object')
    for node in tree.getroot().iter():
        if node.tag in ('mesh', 'texture') and node.get('filename'):
            filename = node.get('filename')
            if '://' in filename:
                raise ValueError(f'Expected local asset filename: {filename}')
            asset = (source.parent / filename).resolve()
            if not asset.is_file():
                raise FileNotFoundError(asset)
            node.set('filename', asset.as_posix())
    payload = ET.tostring(tree.getroot(), encoding='unicode')
    output.parent.mkdir(parents=True, exist_ok=True)
    if not output.exists() or output.read_text(encoding='utf-8') != payload:
        output.write_text(payload, encoding='utf-8')
    return output
