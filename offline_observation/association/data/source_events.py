"""Canonical real events for anonymous-source fusion, without targets.

Event identity is the original canonical node/index, not its numeric value.
Resegmented views retain ``canonical_event_ids`` from their parent; they do not
claim to recover a physical event shared by two different CSV rows or nodes.
"""
from __future__ import annotations

import numpy as np


def source_events(scene):
    """Return one uncapped canonical event record per original scene node.

    ``canonical_valid`` can further mask attributes in a view. Invalid attributes
    remain in the event population, with false validity; no replacement observation
    is invented. The archive's existing track quarantine is unchanged.
    """
    tracks = scene.get('tracks')
    if tracks is None or len(tracks) != len(scene['node_ids']):
        raise ValueError('source events require original canonical tracks')
    result = []
    for node, track in enumerate(tracks):
        node_id, source = str(track['node_id']), str(track['source'])
        if node_id != str(scene['node_ids'][node]) or source != str(scene['sources'][node]):
            raise ValueError('source events must retain original node/source order')
        times = np.asarray(track['canonical_times'], dtype=np.float64).copy()
        values = np.asarray(track['canonical_values'], dtype=np.float64).copy()
        if times.ndim != 1 or values.shape != (len(times), 4) or not np.isfinite(times).all():
            raise ValueError('canonical events require finite seconds and four attributes')
        ids = track.get('canonical_event_ids', [(node_id, i) for i in range(len(times))])
        if len(ids) != len(times) or any(
                len(key) != 2 or not isinstance(key[0], str)
                or isinstance(key[1], (bool, np.bool_)) or not isinstance(key[1], (int, np.integer))
                or key[1] < 0 for key in ids):
            raise ValueError('event IDs must preserve original node and canonical index')
        ids = [(key[0], int(key[1])) for key in ids]
        valid = np.isfinite(values)
        valid[:, 0] &= np.abs(values[:, 0]) <= 90.
        valid[:, 1] &= np.abs(values[:, 1]) <= 180.
        valid[:, 2] &= values[:, 2] >= 0.
        valid[:, 3] &= (values[:, 3] >= 0.) & (values[:, 3] <= 360.)
        if 'canonical_valid' in track:
            mask = np.asarray(track['canonical_valid'])
            if mask.dtype != bool or mask.shape != values.shape:
                raise ValueError('canonical_valid must be a per-event attribute mask')
            valid &= mask
        result.append(dict(node_id=node_id, source=source, event_ids=ids,
                           times=times, values=values, valid=valid))
    return result


def group_events(motion, members):
    """Merge member events into anonymous same-source blocks.

    Only repeated event IDs are removed. Numerically equal observations belonging
    to different original nodes remain distinct. Source names choose equality
    classes and deterministic presentation order; they are not returned as model
    features. Empty source blocks are preserved.
    """
    native = motion['native_events']
    blocks, seen = {}, {}
    for member in dict.fromkeys(members):
        if (isinstance(member, (bool, np.bool_)) or not isinstance(member, (int, np.integer))
                or not 0 <= member < len(native)):
            raise ValueError('event members must be original node indices')
        record = native[int(member)]
        source = str(record['source'])
        events = blocks.setdefault(source, {})
        count = len(record['times'])
        if (len(record['event_ids']) != count or record['values'].shape != (count, 4)
                or record['valid'].shape != (count, 4) or record['valid'].dtype != bool):
            raise ValueError('native event arrays and IDs disagree')
        for index, raw_id in enumerate(record['event_ids']):
            event_id = (str(raw_id[0]), int(raw_id[1]))
            value = (float(record['times'][index]), record['values'][index], record['valid'][index])
            if event_id in seen:
                old_source, old = seen[event_id]
                if (old_source != source or old[0] != value[0]
                        or not np.array_equal(old[1], value[1], equal_nan=True)
                        or not np.array_equal(old[2], value[2])):
                    raise ValueError('repeated original event has conflicting source/time/attributes')
                continue
            seen[event_id] = (source, value)
            events[event_id] = value
    result = []
    for source in sorted(blocks):
        events = blocks[source]
        ids = sorted(events, key=lambda key: (events[key][0], key))
        result.append(dict(event_ids=ids,
            times=np.asarray([events[key][0] for key in ids], np.float64),
            values=np.asarray([events[key][1] for key in ids], np.float64).reshape(-1, 4),
            valid=np.asarray([events[key][2] for key in ids], bool).reshape(-1, 4)))
    return result
