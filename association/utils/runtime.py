"""Portable run provenance and non-overwriting output handling."""
from __future__ import annotations

import json
import os
from pathlib import Path
import platform
import subprocess
import time

import numpy as np
import torch


def jsonable(value):
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path, value):
    path = Path(path)
    payload = json.dumps(jsonable(value), ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(payload, encoding='utf-8')
    os.replace(temp, path)


def read_json(path):
    with Path(path).open(encoding='utf-8') as f:
        return json.load(f)


def seed_everything(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def device_from_string(name):
    device = torch.device(name)
    if device.type not in ('cpu', 'cuda'):
        raise ValueError('Only explicit cpu or cuda:<index> devices are supported')
    if device.type == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA requested but unavailable; CPU fallback is forbidden')
        index = device.index if device.index is not None else torch.cuda.current_device()
        if index >= torch.cuda.device_count():
            raise RuntimeError('Requested CUDA device does not exist in visible devices')
        torch.cuda.set_device(index)
        torch.cuda.reset_peak_memory_stats(index)
    return device


def provenance(device):
    root = Path(__file__).resolve().parents[2]
    def git(*args):
        result = subprocess.run(['git', *args], cwd=root, capture_output=True, text=True, timeout=10)
        return result.stdout.strip() if result.returncode == 0 else 'unavailable'
    result = {'python': platform.python_version(), 'torch': torch.__version__, 'numpy': np.__version__,
              'device': str(device), 'git_commit': git('rev-parse', 'HEAD'),
              'git_dirty': bool(git('status', '--porcelain')),
              'cuda_build': torch.version.cuda, 'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES')}
    if device.type == 'cuda':
        props = torch.cuda.get_device_properties(device)
        result.update(gpu_name=props.name, total_memory_bytes=props.total_memory)
    return result


class BudgetExceeded(RuntimeError):
    pass


class Run:
    def __init__(self, path, metadata, max_hours=None):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=False)
        self.started = time.monotonic()
        self.max_seconds = None if max_hours is None else float(max_hours)*3600
        if self.max_seconds is not None and self.max_seconds <= 0:
            raise ValueError('Time budget must be positive')
        self.metadata = dict(metadata)
        self.metadata['status'] = 'RUNNING'
        write_json(self.path/'run.json', self.metadata)

    def check(self):
        if self.max_seconds is not None and time.monotonic()-self.started > self.max_seconds:
            raise BudgetExceeded('Configured wall time cap reached; partial model is not an eligible reference')

    def finish(self, status='COMPLETED', **extra):
        device_name = self.metadata.get('provenance', {}).get('device', '')
        if device_name.startswith('cuda'):
            try:
                if torch.cuda.is_available():
                    # Nested fits inherit process peaks, not isolated fit peaks.
                    extra['cuda_process_peak_allocated_bytes'] = torch.cuda.max_memory_allocated(device_name)
                    extra['cuda_process_peak_reserved_bytes'] = torch.cuda.max_memory_reserved(device_name)
            except Exception as error:
                # Optional telemetry must not suppress the original failure
                # or terminal state. Keep any earlier peaks.
                extra['cuda_memory_measurement_error'] = f'{type(error).__name__}: {error}'
        self.metadata.update(status=status, elapsed_seconds=time.monotonic()-self.started, **extra)
        write_json(self.path/'run.json', self.metadata)


def save_records(directory, records, binding):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    entries = []
    for k, record in enumerate(records):
        name = f'scene_{k:06d}.npz'
        np.savez_compressed(directory/name, **{key: np.asarray(record[key]) for key in
                            ['scores', 'labels', 'pairs', 'sources', 'identities', 'intervals']})
        entries.append({'file': name, 'scene_id': record['scene_id'],
                        'component_id': record['component_id'], 'node_ids': record['node_ids']})
    index = {'schema': 1, 'binding': binding, 'records': entries}
    write_json(directory/'index.json', index)


def load_records(directory, expected_binding=None):
    directory = Path(directory)
    index = read_json(directory/'index.json')
    if expected_binding is not None and any(index['binding'].get(key) != value
                                            for key, value in expected_binding.items()):
        raise ValueError('Score cache provenance differs from the requested reference/split')
    records = []
    for item in index['records']:
        name = item['file']
        if Path(name).name != name:
            raise ValueError('Score cache file path invalid')
        with np.load(directory/name, allow_pickle=False) as arrays:
            record = {key: arrays[key] for key in arrays.files}
        record.update({key: item[key] for key in ['scene_id', 'component_id', 'node_ids']})
        records.append(record)
    # Check the data used by the experiment, rather than its serialization.
    from association.evaluation.association import _records
    _records(records)
    for record in records:
        n = len(record['sources'])
        identities, intervals = np.asarray(record['identities']), np.asarray(record['intervals'])
        if (identities.shape != (n,) or len(record['node_ids']) != n
                or len(set(record['node_ids'])) != n or record['component_id'] is None):
            raise ValueError('Score cache node identity/component metadata invalid')
        if (intervals.shape != (n, 2) or not np.isfinite(intervals).all()
                or (intervals[:, 1] < intervals[:, 0]).any()):
            raise ValueError('Score cache intervals invalid')
        pairs = record['pairs']
        if not np.array_equal(record['labels'], identities[pairs[:, 0]] == identities[pairs[:, 1]]):
            raise ValueError('Score cache labels disagree with identity pairs')
    return records


def validate_record_split(records, manifest, split):
    """Check a cache against the actual selected scene/component population."""
    actual = [str(record['scene_id']) for record in records]
    expected = [str(scene) for scene in manifest['splits'][split]]
    if len(actual) != len(expected) or set(actual) != set(expected):
        raise ValueError(f'{split} cache scenes differ from manifest')
    for record in records:
        if str(record['component_id']) != str(manifest['components'][str(record['scene_id'])]):
            raise ValueError(f'{split} cache component differs from manifest')
        n = len(record['sources'])
        if len(record['pairs']) != n * (n - 1) // 2:
            raise ValueError(f'{split} cache does not contain all candidate pairs')
