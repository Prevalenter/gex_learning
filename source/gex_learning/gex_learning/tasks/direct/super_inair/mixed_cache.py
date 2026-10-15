"""Load an explicit object-to-cache manifest without importing the simulator."""
import json
from pathlib import Path
import numpy as np
from .grasp_data import ROOT, load_cache


def load_mixed_cache(path, object_index=-1):
    path = Path(path)
    path = (path if path.is_absolute() else ROOT / path).resolve()
    manifest = json.loads(path.read_text(encoding='utf-8'))
    if manifest.get('version') != 1 or manifest.get('status') != 'complete':
        raise ValueError('Mixed grasp manifest must be complete, version 1')
    ids, seeds, caches, offsets, counts = [], [], [], [], []
    entries = manifest['objects']
    if [entry['id'] for entry in entries] != ['tennis_ball_60']:
        raise ValueError('Only tennis_ball_60 is supported')
    if object_index != -1:
        if not 0 <= object_index < len(entries):
            raise ValueError(f'Object index must be between 0 and {len(entries)-1}')
        entries = [entries[object_index]]
    for entry in entries:
        cache = Path(entry['cache'])
        try:
            states, metadata = load_cache(cache if cache.is_absolute() else path.parent/cache)
        except FileNotFoundError as error:
            raise FileNotFoundError(
                f"Missing tennis assets/cache: {error}. "
                "Run scripts/prepare_super_inair_tennis.py --cache <directory>; see docs/SUPER_INAIR.md"
            ) from error
        seed = metadata['seed_pose']
        object_id = entry['id']
        if seed['version'] != 2 or seed['object']['id'] != object_id or object_id in ids:
            raise ValueError('Cache object identity mismatch or duplicate')
        count = len(states['joint_pos'])
        if not count or metadata['status'] != 'complete' or count != entry['count']:
            raise ValueError('Each object requires a complete, nonempty cache')
        ids.append(object_id)
        seeds.append(seed)
        caches.append(states)
        offsets.append(sum(counts))
        counts.append(count)
    if not ids:
        raise ValueError('Mixed grasp manifest is empty')
    states = {key: np.concatenate([cache[key] for cache in caches]) for key in caches[0]}
    return ids, seeds, states, offsets, counts


def sample_object_rows(object_indices, offsets, counts):
    """Uniform rows within each environment's assigned object; never cross caches."""
    import torch
    return offsets[object_indices] + (torch.rand(object_indices.shape, device=object_indices.device)
                                    * counts[object_indices]).long()
