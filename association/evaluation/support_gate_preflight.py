"""068 bounded raw-input/cache preflight; no model or complete-split loader.

PASS establishes reproduction of the frozen data/clock/coordinate contract. It
is not a claim that unusually large but legal observations are physically right.
The frozen 056 groups and representatives are checked, never recomputed.
"""
from __future__ import annotations

from contextlib import closing
import gzip
import json
from pathlib import Path
import time

import numpy as np

from association.data.archive import DataError, MTADArchive, validate_manifest
from association.data.legacy_dataset import validate_archive
from association.data.motion_evidence import DATA, scene_motion_evidence
from association.data.native import _record
from association.data.reference_states import iter_reference_states
from association.data.targets import build_scene_fusion_targets
from association.data.timed import prepare_timed_scene
from association.evaluation.fusion_controls import attribute_errors

REQUESTED_SCENES = ('3035', '2535')
FOCAL_TARGETS = {'3035': '374302000-1', '2535': '477407900-0'}
DATA_CONFIG = {**DATA, 'reference_longitude_policy': 'shortest_arc'}


def _read(value):
    if isinstance(value, dict):
        return value
    return json.loads(Path(value).read_text(encoding='utf-8'))


def _write(path, value):
    text = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)
    if str(path).endswith('.gz'):
        with gzip.open(path, 'wt', encoding='utf-8') as handle:
            handle.write(text)
    else:
        path.write_text(text, encoding='utf-8')


def _same(actual, expected, name):
    a, b = np.asarray(actual), np.asarray(expected)
    equal = (a.shape == b.shape and ((a.dtype == bool) == (b.dtype == bool))
             and (np.array_equal(a, b, equal_nan=True)
             if a.dtype.kind in 'fci' and b.dtype.kind in 'fci' else np.array_equal(a, b)))
    if not equal:
        raise DataError('068 raw/cache mismatch: ' + name)


def _load(root, kind, sid):
    with np.load(root/kind/f'states-{sid}.npz', allow_pickle=False) as loaded:
        return {key: loaded[key] for key in loaded.files}


def _bind(manifest, alignment, source, requested):
    validate_manifest(manifest)
    if manifest.get('dataset_level') != 'Level1' or manifest.get('split_unit') != 'scene':
        raise DataError('068 requires the complete Level1 scene manifest')
    if (not requested or len(requested) != len(set(requested))
            or any(not isinstance(s, str) or not s.isdigit() for s in requested)
            or not set(requested).issubset(manifest['splits']['V_select'])):
        raise DataError('068 requests must be a distinct V_select subset; T_test is sealed')
    formal = tuple(requested) == REQUESTED_SCENES
    if formal and tuple(len(manifest['splits'][s]) for s in ('T_core', 'T_head', 'V_select', 'T_test')) != (2100, 0, 300, 600):
        raise DataError('formal068 requires intact T2100/V300/test600 manifest')
    policies = {}
    for sid in requested:
        item = alignment.get(sid)
        if (not isinstance(item, dict) or set(item) != {'time_offset_seconds', 'max_gap_seconds'}
                or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not np.isfinite(v) for v in item.values())
                or item['max_gap_seconds'] < 0):
            raise DataError('explicit finite reference alignment required: ' + sid)
        policies[sid] = item.copy()
    run = _read(source/'run.json')
    if (run.get('status') != 'COMPLETED' or run.get('dataset_level') != 'Level1'
            or run.get('split') != 'V_select' or run.get('association_endpoint') != 'final'
            or run.get('motion_observation_policy') != 'canonical_all'
            or run.get('test_opened') is not False
            or run.get('completed_scenes') != len(manifest['splits']['V_select'])):
        raise DataError('completed056 final canonical_all V cache required')
    summary = _read(source/'summary.json')
    if (summary.get('status') != 'COMPLETED' or summary.get('dataset_level') != 'Level1'
            or summary.get('split') != 'V_select' or summary.get('test_opened') is not False
            or summary.get('scene_ids') != manifest['splits']['V_select']
            or summary.get('association_endpoint') != 'final'
            or summary.get('association_endpoint_step') != 1000):
        raise DataError('original056 fixed final1000 complete V summary required')
    # Read metadata only: no model checkpoint and no train/test payload.
    recorded_manifest = _read(run['inputs']['manifest'])
    if recorded_manifest != manifest:
        raise DataError('068 manifest differs from original056 complete manifest')
    recorded_alignment = _read(run['inputs']['alignment'])
    if any(recorded_alignment.get(sid) != policies[sid] for sid in requested):
        raise DataError('068 selected alignment differs from original056')
    if formal and (source.name != 'system' or source.parent.name != 'LEVEL1-ASSOCIATION-20261022-056-attempt-1'):
        raise DataError('formal068 requires the original056 attempt-1 system cache')
    return formal, policies


def _cached_mapping(cache, system, scene, targets):
    n, q, t = len(scene['node_ids']), len(scene['grid']), len(targets)
    prefix = 'learned_assoc_anchor_'
    artifact = {k[len(prefix):]: v for k, v in system.items() if k.startswith(prefix)}
    target_ids = np.asarray(list(targets))
    truth = np.stack([item['states'] for item in targets.values()])
    valid = np.stack([item['valid'] for item in targets.values()])
    _same(cache['target_identity_ids'], target_ids, 'target identity population/order')
    _same(artifact['target_ids'], target_ids, 'full-system target population/order')
    _same(artifact['aligned_truth'][:t], truth, 'complete aligned reference')
    _same(artifact['aligned_truth_valid'][:t], valid, 'complete reference validity')
    index = {identity: i for i, identity in enumerate(target_ids)}
    rows = np.asarray([index[x] for x in scene['identities']], np.int64)
    _same(cache['target'], truth[rows], 'fragment reference')
    _same(cache['target_valid'], valid[rows], 'fragment reference validity')
    offsets, flat = cache['learned_assoc_group_offsets'], cache['learned_assoc_group_node_indices']
    if (offsets.ndim != 1 or offsets.dtype.kind not in 'iu' or len(offsets) < 2
            or offsets[0] != 0 or offsets[-1] != n or (np.diff(offsets) <= 0).any()
            or flat.shape != (n,) or flat.dtype.kind not in 'iu'
            or not np.array_equal(np.sort(flat), np.arange(n))):
        raise DataError('invalid frozen056 complete group partition')
    groups = [flat[a:b].tolist() for a, b in zip(offsets[:-1], offsets[1:])]
    anchors = np.asarray([g[0] for g in groups], np.int64)
    if any(g != sorted(g) for g in groups) or not np.array_equal(anchors, np.sort(anchors)):
        raise DataError('frozen056 representative/input group order invalid')
    membership = np.empty(n, np.int64)
    first, duplicates = {}, []
    for gid, group in enumerate(groups):
        membership[group] = gid
        target = int(rows[group[0]])
        if target in first:
            duplicates.append(gid)
        else:
            first[target] = gid
    selected = np.full(t+len(duplicates), -1, np.int64)
    for target, gid in first.items():
        selected[target] = gid
    selected[t:] = duplicates
    full_shape = (len(selected), q, 4)
    aligned_truth = np.full(full_shape, np.nan)
    aligned_valid = np.zeros(full_shape, bool)
    aligned_truth[:t], aligned_valid[:t] = truth, valid
    _same(artifact['aligned_truth'], aligned_truth, 'full reference including duplicate rows')
    _same(artifact['aligned_truth_valid'], aligned_valid, 'full truth mask including duplicate rows')
    _same(artifact['aligned_target_ids'], np.r_[target_ids, np.full(len(duplicates), '', str)],
          'full aligned target ids including duplicate rows')
    if artifact['aligned_prediction'].shape != full_shape or artifact['aligned_prediction_valid'].shape != full_shape:
        raise DataError('frozen aligned output population shape differs')
    for key, value in dict(group_offsets=offsets, group_node_indices=flat, node_group_indices=membership,
            group_anchor_target_indices=rows[anchors], aligned_group_indices=selected,
            group_states=scene['motion']['states'][anchors],
            group_valid=np.ones((len(groups), q, 4), bool),
            grouping_kind='predicted', correspondence_policy='anchor_identity_first', split='V_select').items():
        _same(artifact[key], value, 'frozen mapping/' + key)
    _same(cache['learned_assoc_membership'], membership, 'frozen membership')
    _same(cache['learned_assoc_anchor'], scene['motion']['states'], 'frozen anchor physical output')
    for row, gid in enumerate(selected):
        prediction = scene['motion']['states'][anchors[gid]] if gid >= 0 else np.full((q, 4), np.nan)
        _same(artifact['aligned_prediction'][row], prediction, 'frozen aligned prediction')
        _same(artifact['aligned_prediction_valid'][row], np.full((q, 4), gid >= 0, bool), 'frozen aligned mask')
    return groups, selected[:t], truth, valid


def _track_summary(scene, node):
    track, motion = scene['tracks'][node], scene['motion']
    times, values = track['canonical_times'], track['canonical_values']
    return dict(node_index=node, node_id=track['node_id'], identity=track['identity_id'], source=track['source'],
        native_count=len(times), first_time_seconds=float(times[0]), last_time_seconds=float(times[-1]),
        span_seconds=float(times[-1]-times[0]), min_gap_seconds=float(np.diff(times).min()),
        max_gap_seconds=float(np.diff(times).max()), first_state=values[0].tolist(), last_state=values[-1].tolist(),
        coordinate_min=values[:, :2].min(0).tolist(), coordinate_max=values[:, :2].max(0).tolist(),
        native_speed_min=float(values[:, 2].min()), native_speed_max=float(values[:, 2].max()),
        ols_velocity_mps_north_east=motion['ols_velocity_mps'][node].tolist(),
        endpoint_velocity_mps_north_east=motion['endpoint_velocity_mps'][node].tolist(),
        fit_rmse_m=float(motion['fit_rmse_m'][node]))


def _focal_summary(scene, groups, selected, targets, truth, valid):
    identities, motion = list(targets), scene['motion']
    sid = scene['scene_id']
    focus = [FOCAL_TARGETS[sid]] if sid in FOCAL_TARGETS else identities
    if not set(focus).issubset(identities):
        raise DataError('required focal target missing: ' + sid)
    result = []
    for identity in focus:
        target = identities.index(identity)
        gid = int(selected[target])
        row = dict(target_id=identity, adopted_group=gid, input_nodes=np.flatnonzero(scene['identities'] == identity).tolist())
        if gid < 0:
            row['has_representative'] = False
            result.append(row)
            continue
        group, mask = groups[gid], valid[target, :, :2].all(-1)
        anchor = group[0]
        errors = attribute_errors(motion['states'][group], np.broadcast_to(truth[target], (len(group), *truth[target].shape)))
        row.update(has_representative=True, anchor_node=anchor, group_nodes=group,
            tracks=[_track_summary(scene, node) for node in group])
        if mask.any():
            indices = np.flatnonzero(mask)
            time_index = int(indices[np.argmax(errors['position'][0, mask])])
            row['largest_anchor_position_error'] = dict(query_index=time_index,
                query_seconds=float(scene['grid'][time_index]),
                truth_state=[float(value) if good else None for value, good in
                             zip(truth[target, time_index], valid[target, time_index])],
                truth_valid=valid[target, time_index].tolist(),
                position_error_m=errors['position'][:, time_index].tolist(),
                node_states=motion['states'][group, time_index].tolist(),
                node_support=motion['support'][group, time_index].tolist(),
                node_nearest_seconds=motion['nearest_seconds'][group, time_index].tolist())
        speed_mask = valid[target, :, 2]
        row['reference_speed_range'] = ([float(truth[target, speed_mask, 2].min()), float(truth[target, speed_mask, 2].max())]
                                        if speed_mask.any() else None)
        result.append(row)
    return result


def run_preflight(archive_paths, manifest, alignment, source056, output, check=None, *,
                  requested_scene_ids=REQUESTED_SCENES):
    """Rebuild only two declared V scenes and compare actual arrays, not hashes.

    An explicit alternative scene list is for small diagnostic/test fixtures; its
    report cannot authorize the formal068 training gate. ``check`` is an optional
    no-argument interrupt/resource callback. Any mismatch raises after preserving
    a FAILED report; unusually large legal observations are retained and reported.
    """
    manifest, alignment = _read(manifest), _read(alignment)
    source, output = Path(source056).resolve(), Path(output).resolve()
    if output == source or source in output.parents:
        raise DataError('preflight output must not modify original056 cache')
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    report = dict(status='RUNNING', scope='068-raw-cache-preflight', requested_scene_ids=list(requested_scene_ids),
        completed_scenes=0, checkpoint_loaded=False, model_forwards=0, optimizer_updates=0,
        test_opened=False, groups_recomputed=False, threshold_reselected=False,
        data_contract_reproduced=False, training_gate_passed=False,
        semantic_physical_plausibility_established=False, anomalies_removed=False,
        comparison='exact arrays including NaNs; native prediction is deterministic NumPy in both source and rebuild',
        scene_summaries=[], observation_members_read=[], reference_members_read=[], error=None)
    _write(output/'report.json', report)
    try:
        requested = tuple(requested_scene_ids)
        formal, policies = _bind(manifest, alignment, source, requested)
        report['formal068_scenes'] = formal
        if check:
            check()
        with MTADArchive(archive_paths) as archive, closing(iter_reference_states(
                archive_paths, manifest, split='V_select', scene_ids=requested)) as references:
            validate_archive(archive, manifest)  # complete original inventory, not a reduced manifest
            for sid in requested:
                if check:
                    check()
                cache = _load(source, 'predictions', sid)
                system = _load(source, 'system-predictions', sid)
                native = _record(archive.load_scene(sid), manifest['components'][sid], 'V_select',
                                 DATA['max_observations'], split_unit='scene')
                labels = archive.read_labels(sid)
                target_ids = list(dict.fromkeys('-'.join(r['mmsi'].split('-')[:2]) for r in labels))
                query = None
                if not native['tracks']:
                    if not labels or max(r['t_e'] for r in labels) <= min(r['t_s'] for r in labels):
                        raise DataError('empty scene lacks explicit positive label interval')
                    query = np.linspace(min(r['t_s'] for r in labels), max(r['t_e'] for r in labels), DATA['length'])
                scene = prepare_timed_scene(native, **DATA, query_times=query)
                scene['motion'] = scene_motion_evidence(scene, motion_observation_policy='canonical_all')
                reference = next(references)
                if reference['scene_id'] != sid:
                    raise DataError('reference order mismatch')
                targets = build_scene_fusion_targets(scene, reference, target_identity_ids=target_ids,
                    longitude_policy=DATA_CONFIG['reference_longitude_policy'], **policies[sid])
                motion = scene['motion']
                expected = dict(node_ids=scene['node_ids'], identities=scene['identities'], sources=scene['sources'],
                    query_seconds=scene['grid'], native_states=motion['states'], support=motion['support'],
                    nearest_seconds=motion['nearest_seconds'], exact_observation=motion['exact_observation'])
                for key, value in expected.items():
                    _same(cache[key], value, key)
                groups, selected, truth, valid = _cached_mapping(cache, system, scene, targets)
                payload = {**expected, 'target_ids': np.asarray(target_ids), 'complete_truth': truth,
                    'complete_truth_valid': valid, 'group_offsets': cache['learned_assoc_group_offsets'],
                    'group_node_indices': cache['learned_assoc_group_node_indices'], 'selected_group_by_target': selected}
                payload.update({'motion_'+k: v for k, v in motion.items()})
                for node, track in enumerate(scene['tracks']):
                    payload[f'canonical_times_{node}'] = track['canonical_times']
                    payload[f'canonical_values_{node}'] = track['canonical_values']
                np.savez_compressed(output/f'scene-{sid}-evidence.npz', **payload)
                _write(output/f'scene-{sid}-raw-reference.json.gz',
                       dict(scene_id=sid, member=reference['member'], rows=[row['raw'] for row in reference['rows']]))
                row = dict(scene_id=sid, node_count=len(scene['node_ids']), target_count=len(targets),
                    alignment=policies[sid], input_audit=scene['audit'],
                    target_alignment_reports={k: v['report'] for k, v in targets.items()},
                    focal_targets=_focal_summary(scene, groups, selected, targets, truth, valid),
                    canonical_definition='existing archive cleaning; full valid original observations, no cap in motion',
                    evidence_file=f'scene-{sid}-evidence.npz', reference_file=f'scene-{sid}-raw-reference.json.gz',
                    contracts_reproduced=True, physical_plausibility_established=False)
                _write(output/f'scene-{sid}.json.gz', row)
                report['scene_summaries'].append({k: row[k] for k in ('scene_id', 'node_count', 'target_count', 'alignment', 'focal_targets')})
                report['observation_members_read'] = list(archive.members_read)
                report['reference_members_read'].append(dict(scene_id=sid, member=reference['member']))
                report['completed_scenes'] += 1
                _write(output/'report.json', report)
        report.update(status='PASS', data_contract_reproduced=True, training_gate_passed=formal,
            elapsed_seconds=time.monotonic()-started,
            interpretation='PASS reproduces fixed raw/cache and alignment contracts only; extreme legal values are retained. '
                'It cannot certify physical correctness, data semantics or model benefit. No thresholds or clocks were fitted.')
        _write(output/'report.json', report)
        return report
    except BaseException as exc:
        report.update(status='FAILED', error=str(exc), error_type=type(exc).__name__, elapsed_seconds=time.monotonic()-started)
        _write(output/'report.json', report)
        raise
