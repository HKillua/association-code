"""069 fixed, descriptive reference/canonical audit of existing068 artifacts.

No archive, model, training, new grouping, cleaning, or threshold selection. Named
reference intervals were fixed before execution; their complement is sensitivity,
never a replacement evaluation population. Native ``vel`` units are unverified.
"""
from __future__ import annotations

import csv
import gzip
import io
import json
from pathlib import Path

import numpy as np

from association.data.archive import DataError
from association.data.reference_states import REFERENCE_HEADER, _read_rows
from association.data.targets import align_reference_states

FOCAL = {
    '3035': dict(target='374302000-1', nodes=(4, 18, 133),
                 intervals=((5773., 6208.), (6219., 6223.))),
    '2535': dict(target='477407900-0', nodes=(6, 14),
                 intervals=((4090., 4212.), (4212., 4213.))),
}
ARMS = ('initial', 'final', 'anchor', 'mean_supported', 'attribute_hybrid', 'support_switch')
BASES = ('initial', 'support_switch')


def _json(path):
    with (gzip.open(path, 'rt') if str(path).endswith('.gz') else open(path)) as stream:
        return json.load(stream)


def _require(condition, detail):
    if not condition:
        raise DataError('069 reference audit: ' + detail)


def _same(a, b, label):
    a, b = np.asarray(a), np.asarray(b)
    equal = a.shape == b.shape and (np.array_equal(a, b, equal_nan=True)
        if a.dtype.kind in 'fc' and b.dtype.kind in 'fc' else np.array_equal(a, b))
    _require(equal, 'cache mismatch: ' + label)


def _distance(a, b):
    a, b = np.deg2rad(a), np.deg2rad(b)
    delta = b - a
    value = np.sin(delta[..., 0]/2)**2 + np.cos(a[..., 0])*np.cos(b[..., 0])*np.sin(delta[..., 1]/2)**2
    return 12742000.*np.arcsin(np.sqrt(np.clip(value, 0., 1.)))


def _nullable(values):
    return np.where(np.isfinite(values), values, None).tolist()


def _moments(final, base):
    difference = final-base
    return dict(count=int(final.size), final_error_sum_m=float(final.sum()),
        baseline_error_sum_m=float(base.sum()), signed_error_change_sum_m=float(difference.sum()),
        gross_improvement_sum_m=float(-difference[difference < 0].sum()),
        gross_harm_sum_m=float(difference[difference > 0].sum()),
        improved_count=int((difference < 0).sum()), harmed_count=int((difference > 0).sum()),
        tied_count=int((difference == 0).sum()),
        signed_squared_error_change_sum_m2=float(((final-base)*(final+base)).sum()))


def _contribution(final, base, totals):
    result = _moments(final, base)
    total_count, n = totals['count'], result['count']
    complement_count = total_count-n
    _require(complement_count >= 0, 'subset exceeds full population')
    result.update(full_population_count=total_count,
        contribution_to_full_population_mae_change_m=(result['signed_error_change_sum_m']/total_count if total_count else None),
        contribution_to_full_population_mse_change_m2=(result['signed_squared_error_change_sum_m2']/total_count if total_count else None),
        subset_mae_change_m=(result['signed_error_change_sum_m']/n if n else None),
        full_population_mae_change_m=(totals['signed_error_change_sum_m']/total_count if total_count else None),
        complement_sensitivity=dict(count=complement_count,
            mae_change_m=((totals['signed_error_change_sum_m']-result['signed_error_change_sum_m'])/complement_count
                          if complement_count else None),
            mse_change_m2=((totals['signed_squared_error_change_sum_m2']-result['signed_squared_error_change_sum_m2'])/complement_count
                          if complement_count else None)),
        primary_population_unchanged=True)
    return result


def _reader(raw, sid):
    _require(raw.get('scene_id') == sid and isinstance(raw.get('rows'), list), 'raw reference scene/rows missing')
    buffer = io.StringIO(newline='')
    writer = csv.DictWriter(buffer, fieldnames=REFERENCE_HEADER)
    writer.writeheader()
    for row in raw['rows']:
        _require(set(row) == set(REFERENCE_HEADER) and all(isinstance(v, str) for v in row.values()),
                 'raw reference must retain original named CSV strings')
        writer.writerow(row)
    rows, issues = _read_rows(buffer.getvalue().encode(), raw['member'])
    return dict(scene_id=sid, split='V_select', state_fields=REFERENCE_HEADER[2:], rows=rows, issues=issues)


def _canonical(cache, node, groups, selected_gid, target):
    times = cache[f'canonical_times_{node}']
    states = cache[f'canonical_values_{node}']
    _require(times.ndim == 1 and len(times) >= 2 and states.shape == (len(times), 4)
        and np.isfinite(times).all() and np.isfinite(states).all() and (np.diff(times) > 0).all(),
        f'invalid canonical observations at node {node}')
    _require(((np.abs(states[:, 0]) <= 90) & (np.abs(states[:, 1]) <= 180)).all()
        and (states[:, 2] >= 0).all(), 'invalid canonical coordinate/velocity')
    _require(str(cache['identities'][node]) == target, 'fixed focal node identity changed')
    query = cache['query_seconds']
    _same(cache['support'][node], (query >= times[0]) & (query <= times[-1]),
          f'canonical support/node{node}')
    _same(cache['nearest_seconds'][node], np.min(np.abs(times[:, None]-query[None, :]), axis=0),
          f'canonical nearest distance/node{node}')
    distance = _distance(states[:-1, :2], states[1:, :2])
    speed = distance/np.diff(times)
    peak = int(np.argmax(speed))
    gid = next(index for index, group in enumerate(groups) if node in group)
    return dict(node_index=node, node_id=str(cache['node_ids'][node]), identity=target,
        native_count=len(times), support_bounds_seconds=[float(times[0]), float(times[-1])],
        cached_support_and_nearest_reproduced=True,
        coordinate_min=states[:, :2].min(0).tolist(), coordinate_max=states[:, :2].max(0).tolist(),
        native_velocity_range=[float(states[:, 2].min()), float(states[:, 2].max())],
        native_velocity_units='native MTAD; unverified, no conversion',
        adjacent_displacement_m_quantiles=dict(zip(('min', 'median', 'p95', 'max'),
            np.quantile(distance, [0, .5, .95, 1]).tolist())),
        adjacent_geometric_speed_mps_quantiles=dict(zip(('min', 'median', 'p95', 'max'),
            np.quantile(speed, [0, .5, .95, 1]).tolist())),
        largest_geometric_speed_bracket=dict(indices=[peak, peak+1],
            times=times[peak:peak+2].tolist(), states=states[peak:peak+2].tolist(),
            elapsed_seconds=float(times[peak+1]-times[peak]), displacement_m=float(distance[peak]),
            geometric_speed_mps=float(speed[peak])),
        output_group=gid, group_nodes=groups[gid], is_group_representative=node == groups[gid][0],
        in_unique_adopted_group=gid == selected_gid,
        is_unique_adopted_representative=gid == selected_gid and node == groups[gid][0])


def _scene(source, row, totals):
    sid, policy = row['scene_id'], FOCAL[row['scene_id']]
    root = source/'preflight'
    report = _json(root/f'scene-{sid}.json.gz')
    raw = _json(root/f'scene-{sid}-raw-reference.json.gz')
    with np.load(root/f'scene-{sid}-evidence.npz', allow_pickle=False) as store:
        cache = {key: store[key] for key in store.files}
    required = {'query_seconds', 'target_ids', 'complete_truth', 'complete_truth_valid',
        'selected_group_by_target', 'group_offsets', 'group_node_indices', 'node_ids',
        'identities', 'native_states', 'support', 'nearest_seconds'}
    required |= {f'canonical_{field}_{node}' for node in policy['nodes'] for field in ('times', 'values')}
    _require(required.issubset(cache), 'missing preflight arrays: '+', '.join(sorted(required-set(cache))))
    _require(report.get('scene_id') == sid and report.get('contracts_reproduced') is True,
             'preflight scene contract not reproduced')
    q, tids = cache['query_seconds'], cache['target_ids'].astype(str)
    _same(tids, row['target_ids'], sid+'/target_ids')
    _same(q, row['query_seconds'], sid+'/query_seconds')
    truth, valid = cache['complete_truth'], cache['complete_truth_valid']
    _require(valid.dtype == bool and truth.shape == valid.shape == (len(tids), len(q), 4), 'complete truth shape/dtype')
    selected = cache['selected_group_by_target']
    _require(selected.dtype.kind in 'iu' and selected.shape == (len(tids),), 'selected group shape/dtype')
    offsets, indices = cache['group_offsets'], cache['group_node_indices']
    n = len(cache['identities'])
    _require(offsets.ndim == 1 and offsets.dtype.kind in 'iu' and indices.dtype.kind in 'iu'
        and offsets.size >= 2 and offsets[0] == 0 and offsets[-1] == len(indices)
        and (np.diff(offsets) > 0).all() and np.array_equal(np.sort(indices), np.arange(n)), 'group partition')
    groups = [indices[a:b].tolist() for a, b in zip(offsets[:-1], offsets[1:])]
    _require(((selected >= -1) & (selected < len(groups))).all(), 'invalid selected group')
    _same(truth, row['truth'], sid+'/endpoint truth')
    _same(valid, row['truth_valid'], sid+'/endpoint truth validity')
    _same(selected, row['selected_group_by_target'], sid+'/endpoint target correspondence')
    _require(groups == row['groups'], 'endpoint/preflight grouping mismatch')
    for name in ('node_ids', 'identities', 'native_states', 'support', 'nearest_seconds', 'query_seconds'):
        _same(cache[name], row['native_inputs'][name], sid+'/endpoint native/'+name)
    _same(np.isfinite(row['errors']['initial']), valid[..., :2].all(-1) & (selected >= 0)[:, None],
          sid+'/full unique output position mask')
    reader = _reader(raw, sid)
    alignment = report['alignment']
    for index, tid in enumerate(tids):
        aligned = align_reference_states(reader, str(tid), q, **alignment, longitude_policy='shortest_arc')
        _same(aligned['states'], truth[index], sid+'/'+tid+'/complete_truth')
        _same(aligned['valid'], valid[index], sid+'/'+tid+'/complete_truth_valid')
    target = policy['target']
    matches = np.flatnonzero(tids == target)
    _require(len(matches) == 1, 'fixed focal target missing or repeated')
    target_index = int(matches[0])
    focal = [r for r in reader['rows'] if r['identity_id'] == target]
    times = np.asarray([r['time'] for r in focal], float)
    _require(len(times) >= 2 and np.isfinite(times).all() and (np.diff(times) > 0).all(),
             'focal raw order must already be strictly time increasing')
    _require(all(r['raw']['MMSI'] == target for r in focal), 'focal raw identity requires normalization')
    focal_positions = np.asarray([r['state'][:2] for r in focal], float)
    _require(np.isfinite(focal_positions).all(), 'focal raw positions not finite')
    focal_speed = _distance(focal_positions[:-1], focal_positions[1:])/np.diff(times)
    union = np.zeros(len(q), bool)
    intervals = []
    for start, stop in policy['intervals']:
        at = np.flatnonzero(times == start)
        _require(len(at) == 1 and at[0]+1 < len(times) and times[at[0]+1] == stop, 'named raw interval not adjacent')
        left, right = focal[at[0]], focal[at[0]+1]
        position = np.asarray([left['state'][:2], right['state'][:2]], float)
        _require(np.isfinite(position).all(), 'named reference position not finite')
        offset = alignment['time_offset_seconds']
        interior = (q > start-offset) & (q < stop-offset)
        effective = interior & valid[target_index, :, :2].all(-1) & np.isfinite(row['errors']['initial'][target_index])
        union |= effective
        distance = float(_distance(position[0], position[1]))
        intervals.append(dict(raw_rows=[left['raw'], right['raw']], original_csv_lines=[left['line'], right['line']],
            elapsed_seconds=stop-start, displacement_m=distance, geometric_speed_mps=distance/(stop-start),
            aligned_time_bounds=[start-offset, stop-offset], gap_allowed=stop-start <= alignment['max_gap_seconds'],
            interior_query_indices=np.flatnonzero(interior).tolist(), interior_query_seconds=q[interior].tolist(),
            exact_endpoint_query_indices=np.flatnonzero((q == start-offset) | (q == stop-offset)).tolist(),
            interior_position_truth_valid=valid[target_index, interior, :2].all(-1).tolist(),
            interior_aligned_truth=_nullable(truth[target_index, interior]), evaluated_query_indices=np.flatnonzero(effective).tolist(),
            contributions={base: _contribution(row['errors']['final'][target_index, effective],
                row['errors'][base][target_index, effective], totals[base]) for base in BASES}))
    gid = int(selected[target_index])
    _require(gid >= 0 and str(cache['identities'][groups[gid][0]]) == target, 'focal adopted representative identity')
    all_valid = np.isfinite(row['errors']['initial'][target_index])
    contributions = {base: _contribution(row['errors']['final'][target_index, union],
                        row['errors'][base][target_index, union], totals[base]) for base in BASES}
    return dict(scene_id=sid, target_id=target, target_row=target_index, alignment=alignment,
        raw_member=raw['member'], raw_identity_strings_exact=True, raw_identity_order_already_increasing=True,
        raw_focal_rows=len(focal), raw_focal_adjacent_geometric_speed_mps_quantiles=dict(zip(
            ('min', 'median', 'p95', 'p99', 'max'), np.quantile(focal_speed, [0, .5, .95, .99, 1]).tolist())),
        all_scene_targets_truth_reproduced=True, adopted_group=gid, adopted_representative=groups[gid][0],
        focal_valid_position_points=int(all_valid.sum()), named_bracket_valid_position_points=int(union.sum()),
        named_bracket_query_indices=np.flatnonzero(union).tolist(), named_intervals=intervals,
        named_bracket_union_contributions=contributions,
        focal_all_query_contributions={base: _contribution(row['errors']['final'][target_index, all_valid],
            row['errors'][base][target_index, all_valid], totals[base]) for base in BASES},
        canonical_tracks=[_canonical(cache, node, groups, gid, target) for node in policy['nodes']]), union


def audit_reference(source068dir, scene_rows: list[dict]) -> dict:
    """Return JSON-safe fixed two-scene diagnostics using only saved068 files.

    scene_rows must cover the caller's already bound full V population. Six error
    arrays use identical unique-target masks, with NaN at unscored points. Missing
    required artifacts/fields and mismatched truth raise; there is no fallback.
    """
    _require(bool(scene_rows), 'full population rows required')
    by_scene, pooled = {}, {arm: [] for arm in ARMS}
    for row in scene_rows:
        sid, tids = row['scene_id'], np.asarray(row['target_ids']).astype(str)
        _require(isinstance(sid, str) and sid not in by_scene and len(tids) == len(set(tids)), 'scene/target identity duplication')
        shape = (len(tids), len(row['query_seconds']))
        mask = None
        for arm in ARMS:
            errors = np.asarray(row['errors'][arm])
            _require(errors.shape == shape and errors.dtype.kind == 'f' and not np.isinf(errors).any()
                and (errors[np.isfinite(errors)] >= 0).all(), 'invalid error values/shape: '+sid+'/'+arm)
            if mask is None:
                mask = np.isfinite(errors)
            _same(np.isfinite(errors), mask, sid+'/'+arm+'/evaluation population')
            pooled[arm].append(errors[mask])
        by_scene[sid] = row
    _require(set(FOCAL).issubset(by_scene), 'both frozen focal scenes required')
    pooled = {arm: np.concatenate(parts) for arm, parts in pooled.items()}
    totals = {base: _moments(pooled['final'], pooled[base]) for base in BASES}
    source = Path(source068dir)
    preflight = _json(source/'preflight/report.json')
    _require(preflight.get('status') == 'PASS' and preflight.get('data_contract_reproduced') is True,
             'original preflight must be PASS')
    scenes, union_pairs = [], {base: [[], []] for base in BASES}
    for sid in FOCAL:
        scene, union = _scene(source, by_scene[sid], totals)
        scenes.append(scene)
        for base in BASES:
            index = scene['target_row']
            union_pairs[base][0].append(by_scene[sid]['errors']['final'][index, union])
            union_pairs[base][1].append(by_scene[sid]['errors'][base][index, union])
    return dict(status='COMPLETED', scope='069-fixed-reference-canonical-cache-audit',
        full_population_scene_count=len(scene_rows), full_population_position_points=int(pooled['initial'].size),
        full_population_comparisons=totals, scenes=scenes,
        named_bracket_union_contributions={base: _contribution(np.concatenate(pair[0]), np.concatenate(pair[1]), totals[base])
                                           for base, pair in union_pairs.items()},
        test_opened=False, model_forwards=0, original_archives_read=False, anomalies_removed=False,
        limitations=['Two development scenes chosen for known extremes; not representative of the dataset.',
            'Raw reference jumps establish source-field inconsistency under the current contract, not its cause.',
            'Only four predeclared reference brackets are attributed; other intervals, including other2535 jumps, remain unquantified.',
            'Named bracket membership is descriptive, not a cleaning rule or a causal effect of labels.',
            'Complement results are sensitivity only; all original scored points remain primary.',
            'Canonical observations are saved068 parsed inputs; no independent raw measurement parser audit.',
            'Native velocity units remain unverified; geometric distance/time uses the existing degree/second contract.'])
