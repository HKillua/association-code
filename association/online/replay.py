"""Prepare a received measurement prefix for the existing engineering replay.

The cutoff constrains observation availability; the output remains the historical
Q64 query grid used by the original timing protocol, not a current-state query.
No real label or reference payload is read by this adapter.
"""
from __future__ import annotations

import csv
import io

import numpy as np

from association.data.archive import MTADArchive, HEADERS
from association.data.native import _record
from association.data.timed import prepare_timed_scene
from association.data.motion_evidence import DATA, scene_motion_evidence


def csv_rows(payload):
    try:
        text = payload.decode('utf-8-sig')
    except UnicodeDecodeError:
        text = payload.decode('gb18030')
    reader = csv.DictReader(io.StringIO(text, newline=''), strict=True)
    if reader.fieldnames != HEADERS['measurements']:
        raise ValueError('measurement header differs')
    rows = []
    for row in reader:
        if None in row or any(v is None for v in row.values()):
            raise ValueError('wrong measurement column count')
        rows.append({k: v.strip() for k, v in row.items()})
    return rows


class ReceivedMeasurements(MTADArchive):
    """Reuse canonical measurement checks with anonymous, per-node metadata.

    The legacy loader requires label rows to describe fragments. This adapter
    derives the same source/batch boundaries from sensor messages themselves;
    its distinct anonymous IDs have no cross-node identity information.
    """
    def __init__(self, payload):
        self.rows = csv_rows(payload)

    def _rows(self, scene_id, kind):
        if kind != 'measurements':
            raise AssertionError('label/reference payload access forbidden')
        return self.rows

    def read_labels(self, scene_id):
        bounds = {}
        for row in self.rows:
            key = (row['source'], row['batch'])
            stamp = float(row['time'])
            low, high = bounds.get(key, (stamp, stamp))
            bounds[key] = (min(low, stamp), max(high, stamp))
        return [dict(source=s, batch=b, t_s=bounds[s, b][0], t_e=bounds[s, b][1],
                     mmsi=f'{i}-anonymous-0')
                for i, (s, b) in enumerate(sorted(bounds))]


def preprocess(payload, sid, component, cutoff):
    source = ReceivedMeasurements(payload)
    stamps = np.asarray([float(r['time']) for r in source.rows])
    if not len(stamps) or (stamps > cutoff).any():
        raise AssertionError('future observation entered the received buffer')
    native = _record(source.load_scene(sid), component, 'V_select', 128)
    if len(native['tracks']) < 2:
        raise ValueError('at least two valid received fragments required')
    prepared = prepare_timed_scene(native, **DATA)
    motion = scene_motion_evidence(prepared, motion_observation_policy='canonical_all')
    motion['node_ids'] = np.asarray(prepared['node_ids'], str)
    if motion['query_seconds'][-1] > cutoff:
        raise AssertionError('output grid extends beyond arrival cutoff')
    return motion, native['pairs'], dict(
        input_bytes=len(payload), received_rows=len(stamps),
        earliest_timestamp=float(stamps.min()), latest_timestamp=float(stamps.max()),
        arrival_cutoff=cutoff, future_observation_rows=0,
        valid_nodes=len(native['tracks']), quarantined_nodes=native['audit']['quarantined_nodes'],
        query_points=64, query_start=float(motion['query_seconds'][0]),
        query_end=float(motion['query_seconds'][-1]))
