"""Explicit T_core supervision readers for post-prediction online evaluation."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import json
import math
from pathlib import Path
import re

from .measurements import _csv_rows, _read_development_member


_LABEL_PATH = re.compile(r'(?:^|/)关联表/关联结果-([0-9]+)\.csv$')
_LABEL_FIELDS = ['t_s', 't_e', 'mmsi', 'source', 'batch']
_REFERENCE_PATH = re.compile(r'(?:^|/)真实场景/场景-([0-9]+)\.csv$')
_REFERENCE_FIELDS = ['MMSI', 'time', 'lat', 'lon', 'vel', 'cou']
_ALIGNMENT_PATH = Path(__file__).with_name('data') / 'mtad-level1-alignment.json'

RawReference = tuple[float | None, float | None, float | None, float | None, float | None]


@dataclass(frozen=True)
class ReferenceData:
    """Original CSV order and times with the separately frozen clock policy.

    Each sample is (raw time, latitude, longitude, speed, course). Velocities
    keep their raw Level1 reference values. The project's physical-unit
    convention treats those as m/s. Aligned time is raw time minus the offset.
    Non-numeric fields are None. Repeated times and invalid numeric values are
    retained for per-attribute support checks by the evaluator.
    """

    by_identity: dict[str, tuple[RawReference, ...]]
    time_offset_seconds: float
    max_gap_seconds: float


def read_labels(parts, manifest_path, scene):
    """Read one T_core label member and return {(source, batch): 'A-B'}."""
    payload = _read_development_member(
        parts, manifest_path, scene, _LABEL_PATH, 'label', required_split='T_core')
    full_identities = {}
    for row in _csv_rows(payload, _LABEL_FIELDS, 'label'):
        key = (row['source'], row['batch'])
        identity = row['mmsi']
        components = identity.split('-')
        if not all(key) or len(components) != 3 or not all(components):
            raise ValueError(f'scene {scene} has an invalid label key or A-B-C identity')
        try:
            start, end = float(row['t_s']), float(row['t_e'])
        except ValueError as exc:
            raise ValueError(f'scene {scene} has a nonnumeric label interval') from exc
        if not math.isfinite(start) or not math.isfinite(end) or start > end:
            raise ValueError(f'scene {scene} has an invalid label interval')
        previous = full_identities.setdefault(key, identity)
        if previous != identity:
            raise ValueError(f'scene {scene} has conflicting A-B-C labels for one source/batch')
    return {key: identity.rsplit('-', 1)[0] for key, identity in full_identities.items()}


def _number(value):
    try:
        return float(value)
    except ValueError:
        return None


def read_references(parts, manifest_path, scene, *, alignment_path=None):
    """Read one T_core reference member and its explicit alignment policy."""
    payload = _read_development_member(
        parts, manifest_path, scene, _REFERENCE_PATH, 'reference', required_split='T_core')
    alignment = _ALIGNMENT_PATH if alignment_path is None else Path(alignment_path)
    try:
        policy = json.loads(alignment.read_text(encoding='utf-8'))[scene]
        offset = float(policy['time_offset_seconds'])
        max_gap = float(policy['max_gap_seconds'])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f'scene {scene} has no valid reference alignment policy') from exc
    if not math.isfinite(offset) or not math.isfinite(max_gap) or max_gap <= 0:
        raise ValueError(f'scene {scene} has no valid reference alignment policy')
    by_identity = defaultdict(list)
    for row in _csv_rows(payload, _REFERENCE_FIELDS, 'reference'):
        identity = row['MMSI']
        components = identity.split('-')
        if len(components) != 2 or not all(components):
            raise ValueError(f'scene {scene} has an invalid A-B reference identity')
        sample = tuple(_number(row[field]) for field in _REFERENCE_FIELDS[1:])
        by_identity[identity].append(sample)
    return ReferenceData({identity: tuple(rows) for identity, rows in by_identity.items()},
                         offset, max_gap)
