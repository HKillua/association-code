"""Portable run provenance and non-overwriting output handling."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import time

import numpy as np
import torch


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


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


def object_hash(value):
    return hashlib.sha256(json.dumps(jsonable(value), sort_keys=True, allow_nan=False).encode()).hexdigest()


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


def code_fingerprint():
    root = Path(__file__).resolve().parents[1]
    files = sorted((root/'association').glob('*.py'))
    return object_hash({str(p.relative_to(root)): digest(p) for p in files})


def provenance(device):
    root = Path(__file__).resolve().parents[1]
    def git(*args):
        result = subprocess.run(['git', *args], cwd=root, capture_output=True, text=True, timeout=10)
        return result.stdout.strip() if result.returncode == 0 else 'unavailable'
    result = {'python': platform.python_version(), 'torch': torch.__version__, 'numpy': np.__version__,
              'device': str(device), 'git_commit': git('rev-parse', 'HEAD'),
              'git_dirty': bool(git('status', '--porcelain')), 'code_sha256': code_fingerprint(),
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
        if device_name.startswith('cuda') and torch.cuda.is_available():
            # CLI resets counters before the job; nested fits inherit process
            # peaks. Do not mislabel this as an isolated per-fit measurement.
            extra['cuda_process_peak_allocated_bytes'] = torch.cuda.max_memory_allocated(device_name)
            extra['cuda_process_peak_reserved_bytes'] = torch.cuda.max_memory_reserved(device_name)
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
        entries.append({'file': name, 'sha256': digest(directory/name), 'scene_id': record['scene_id'],
                        'component_id': record['component_id'], 'node_ids': record['node_ids']})
    index = {'schema': 1, 'binding': binding, 'records': entries}
    index['content_sha256'] = object_hash(index)
    write_json(directory/'index.json', index)


def load_records(directory, expected_binding=None):
    directory = Path(directory)
    index = read_json(directory/'index.json')
    checksum = index.pop('content_sha256')
    if object_hash(index) != checksum:
        raise ValueError('Score cache index checksum mismatch')
    if expected_binding is not None and index['binding'] != expected_binding:
        raise ValueError('Score cache provenance differs from the requested reference/split')
    records = []
    for item in index['records']:
        name = item['file']
        if Path(name).name != name or digest(directory/name) != item['sha256']:
            raise ValueError('Score cache file checksum or path invalid')
        with np.load(directory/name, allow_pickle=False) as arrays:
            record = {key: arrays[key] for key in arrays.files}
        record.update({key: item[key] for key in ['scene_id', 'component_id', 'node_ids']})
        records.append(record)
    return records
