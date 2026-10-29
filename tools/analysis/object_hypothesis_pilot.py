"""129: native temporal link candidates, T-frozen rejection and real WS unions.

Native overlap/CV are working assumptions. All actions precede V identities and
reference evaluation; original groups, nodes and failed/unmatched cases remain.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from itertools import combinations
import os
from pathlib import Path
import resource
import signal
import sys
import time
import traceback

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from association.evaluation.group_prediction import emit_group_outputs
from association.evaluation.query_contribution import recover_group_states
from tools.analysis import query_contribution_control as qc

old = qc.old
IDEA = 'IDEA-OBJECT-HYPOTHESIS-PILOT-20261029'
PARTITIONS = ('WS', 'NO_REJECT', 'T_REJECT')
METHODS = ('WS_MEAN', 'WS_LOW_SOURCE_SUPPORTED', 'NO_REJECT_MEAN', 'NO_REJECT_LOW',
           'T_REJECT_MEAN', 'T_REJECT_LOW')
COMPARISONS = tuple(combinations(METHODS, 2))
CONTROL_RUN = 'LEVEL1-SOURCE-PREFERENCE-20261029-121-attempt-1'
RADIUS = 6371000.
NATIVE_KEYS = ('node_ids', 'sources', 'query_seconds', 'event_offsets', 'event_times_seconds',
    'event_values', 'event_valid', 'event_node_ids', 'event_canonical_indices')
PHYSICAL_KEYS = ('native_states', 'nearest_seconds', 'support', 'sources', 'node_ids', 'query_seconds')


def distance(a, b):
    a, b = np.deg2rad(np.asarray(a, np.float64)), np.deg2rad(np.asarray(b, np.float64))
    delta = b-a
    h = np.sin(delta[..., 0]/2.)**2+np.cos(a[..., 0])*np.cos(b[..., 0])*np.sin(delta[..., 1]/2.)**2
    return 2.*RADIUS*np.arcsin(np.sqrt(np.clip(h, 0., 1.)))


def forward(position, bearing, metres):
    latitude, longitude = np.deg2rad(position)
    angle = metres/RADIUS
    out_lat = np.arcsin(np.clip(np.sin(latitude)*np.cos(angle)+np.cos(latitude)*np.sin(angle)*np.cos(bearing), -1., 1.))
    out_lon = longitude+np.arctan2(np.sin(bearing)*np.sin(angle)*np.cos(latitude),
                                np.cos(angle)-np.sin(latitude)*np.sin(out_lat))
    result = np.rad2deg([out_lat, out_lon])
    result[1] = (result[1]+180.) % 360.-180.
    return result


def native_features(native):
    """Only native inputs; no identity/reference argument exists."""
    ids, sources = np.asarray(native['node_ids']), np.asarray(native['sources'])
    offsets = np.asarray(native['event_offsets'])
    times, values, valid = (np.asarray(native[key]) for key in ('event_times_seconds', 'event_values', 'event_valid'))
    event_nodes, event_indices = np.asarray(native['event_node_ids']), np.asarray(native['event_canonical_indices'])
    n, count = len(ids), len(times)
    if (ids.dtype.kind not in 'US' or len(set(ids.tolist())) != n or sources.shape != (n,)
            or sources.dtype.kind not in 'US' or any(not str(value) for value in sources)
            or offsets.shape != (n+1,) or offsets.dtype.kind not in 'iu' or offsets[0] != 0
            or offsets[-1] != count or (np.diff(offsets) < 0).any()
            or times.shape != (count,) or not np.isfinite(times).all()
            or values.shape != (count, 4) or valid.shape != values.shape or valid.dtype != bool
            or event_nodes.shape != (count,) or event_nodes.dtype.kind not in 'US'
            or event_indices.shape != (count,) or event_indices.dtype.kind not in 'iu' or (event_indices < 0).any()):
        raise ValueError('complete native event cache with original order/provenance required')
    starts, ends, first_time, last_time = (np.full(n, np.nan) for _ in range(4))
    first, last = np.full((n, 2), np.nan), np.full((n, 2), np.nan)
    speed, bearing, usable_count = np.zeros(n), np.zeros(n), np.zeros(n, np.int64)
    events, registry = [], {}
    for node in range(n):
        left, right = offsets[node:node+2]
        t, v, mask = times[left:right], values[left:right], valid[left:right]
        if (np.diff(t) < 0).any():
            raise ValueError('native event order must be chronological')
        keys = set()
        for index in range(left, right):
            key = (str(event_nodes[index]), int(event_indices[index]))
            value = (str(sources[node]), float(times[index]), values[index], valid[index])
            if key in registry:
                previous = registry[key]
                if (previous[0:2] != value[0:2] or not np.array_equal(previous[2], value[2], equal_nan=True)
                        or not np.array_equal(previous[3], value[3])):
                    raise ValueError('same original event ID has conflicting source/time/attributes')
            registry[key] = value
            keys.add(key)
        events.append(keys)
        if len(t):
            starts[node], ends[node] = t[0], t[-1]
        use = mask[:, :2].all(1) & np.isfinite(v[:, :2]).all(1)
        use &= (np.abs(v[:, 0]) <= 90.) & (np.abs(v[:, 1]) <= 180.)
        index = np.flatnonzero(use)
        usable_count[node] = len(index)
        if not len(index):
            continue
        a, b = int(index[0]), int(index[-1])
        first[node], last[node] = v[a, :2], v[b, :2]
        first_time[node], last_time[node] = t[a], t[b]
        earlier = index[t[index] < t[b]]
        if len(earlier):
            a = int(earlier[-1])
            speed[node] = distance(v[a, :2], v[b, :2])/(t[b]-t[a])
            lat1, lon1 = np.deg2rad(v[a, :2]); lat2, lon2 = np.deg2rad(v[b, :2])
            bearing[node] = np.arctan2(np.sin(lon2-lon1)*np.cos(lat2),
                np.cos(lat1)*np.sin(lat2)-np.sin(lat1)*np.cos(lat2)*np.cos(lon2-lon1))
    temporal = ((sources[:, None] == sources[None, :]) & (ends[:, None] < starts[None, :]))
    eligible_pairs = np.stack(np.nonzero(temporal), axis=-1)
    edges, residuals, gaps = [], [], []
    for node in range(n):
        ranking = []
        if usable_count[node]:
            for other in np.flatnonzero(temporal[node]):
                if not usable_count[other]:
                    continue
                dt = first_time[other]-last_time[node]
                if dt < 0:
                    raise ValueError('temporal candidate would extrapolate backward')
                prediction = forward(last[node], bearing[node], speed[node]*dt)
                residual = float(distance(prediction, first[other]))
                if not np.isfinite(residual):
                    raise FloatingPointError('nonfinite endpoint continuity score')
                ranking.append((residual, float(starts[other]-ends[node]), int(other)))
        for residual, gap, other in sorted(ranking)[:2]:
            edges.append((node, other)); residuals.append(residual); gaps.append(gap)
    overlap = ((sources[:, None] == sources[None, :])
        & (np.maximum(starts[:, None], starts[None, :]) <= np.minimum(ends[:, None], ends[None, :])))
    np.fill_diagonal(overlap, False)
    event_conflict = np.zeros((n, n), bool)
    for a in range(n):
        for b in range(a+1, n):
            event_conflict[a, b] = event_conflict[b, a] = bool(events[a] & events[b])
    return dict(node_start_seconds=starts, node_end_seconds=ends,
        first_position_time_seconds=first_time, last_position_time_seconds=last_time,
        first_position=first, last_position=last, terminal_speed_mps=speed, terminal_bearing_radians=bearing,
        position_event_count=usable_count, time_eligible_pairs=eligible_pairs,
        candidate_edges=np.asarray(edges, np.int64).reshape(-1, 2),
        candidate_distance_m=np.asarray(residuals), candidate_gap_seconds=np.asarray(gaps),
        native_overlap_conflict=overlap, repeated_event_conflict=event_conflict,
        singleton_node_indices=np.arange(n, dtype=np.int64))


def fit_rejection(distances, labels, known):
    distances, labels, known = np.asarray(distances, np.float64), np.asarray(labels, bool), np.asarray(known, bool)
    if labels.shape != distances.shape or known.shape != distances.shape or not np.isfinite(distances).all():
        raise ValueError('finite frozen T candidates and same-length labels required')
    x, y = distances[known], labels[known]
    positive, negative = int(y.sum()), int((~y).sum())
    base = dict(training_candidate_count=len(distances), known_candidate_count=len(x),
        positive_count=positive, negative_count=negative, fit_objective='balanced_accuracy',
        tie_rule='fewest accepted candidates', threshold_selected_from_V=False)
    if not positive or not negative:
        return dict(base, status='unavailable_single_or_no_class', mode='reject_all', threshold_metres=None,
                    balanced_accuracy=None), dict(threshold_metres=np.empty(0), balanced_accuracy=np.empty(0))
    order = np.argsort(x, kind='stable'); x, y = x[order], y[order]
    ends = np.r_[np.flatnonzero(np.diff(x) != 0), len(x)-1]
    true_positive = np.cumsum(y)[ends]
    false_positive = ends+1-true_positive
    ba = .5*(true_positive/positive+1.-false_positive/negative)
    curve = dict(threshold_metres=np.r_[np.nan, x[ends]], balanced_accuracy=np.r_[.5, ba],
                 accepted_count=np.r_[0, ends+1], true_positive=np.r_[0, true_positive],
                 false_positive=np.r_[0, false_positive])
    best = float(curve['balanced_accuracy'].max())
    selected = int(np.flatnonzero(np.isclose(curve['balanced_accuracy'], best, rtol=0., atol=1e-15))[0])
    return dict(base, status='fitted', mode='reject_all' if selected == 0 else 'threshold',
        threshold_metres=None if selected == 0 else float(curve['threshold_metres'][selected]),
        balanced_accuracy=float(curve['balanced_accuracy'][selected]), selected_curve_index=selected,
        scanned_operating_points=len(curve['threshold_metres'])), curve


def merge_partition(membership, features, model, *, reject):
    """Greedy finite unions with visible conflicts; identity-blind."""
    n = len(membership)
    groups = old.groups_from_membership(membership, n)
    parent = np.arange(len(groups)); active = {i: list(group) for i, group in enumerate(groups)}
    def find(group):
        while parent[group] != group:
            group = int(parent[group])
        return group
    conflict = features['native_overlap_conflict'] | features['repeated_event_conflict']
    initial_conflicts = sum(int(np.triu(conflict[np.ix_(g, g)], 1).sum()) for g in groups)
    edges, distances, gaps = (features[k] for k in ('candidate_edges', 'candidate_distance_m', 'candidate_gap_seconds'))
    order = sorted(range(len(edges)), key=lambda i: (float(distances[i]), float(gaps[i]), int(edges[i, 0]), int(edges[i, 1])))
    decision = np.full(len(edges), -1, np.int8)
    for index in order:
        a, b = edges[index]
        ga, gb = find(int(membership[a])), find(int(membership[b]))
        if ga == gb:
            decision[index] = 0
            continue
        if reject and (model['mode'] == 'reject_all' or distances[index] > model['threshold_metres']):
            decision[index] = 1
            continue
        merged = sorted(active[ga]+active[gb])
        if np.triu(conflict[np.ix_(merged, merged)], 1).any():
            decision[index] = 2
            continue
        left, right = sorted((ga, gb))
        active[left] = merged; del active[right]; parent[right] = left
        decision[index] = 3
    result = old.partition.membership_from_clusters([active[key] for key in sorted(active)], n)
    final_groups = old.groups_from_membership(result, n)
    final_conflicts = sum(int(np.triu(conflict[np.ix_(g, g)], 1).sum()) for g in final_groups)
    if final_conflicts != initial_conflicts:
        raise ValueError('new union added or silently removed an original working conflict')
    audit = dict(initial_group_count=len(groups), final_group_count=len(final_groups),
        initial_internal_working_conflict_pairs=initial_conflicts, final_internal_working_conflict_pairs=final_conflicts,
        already_same_group_count=int((decision == 0).sum()), threshold_rejected_count=int((decision == 1).sum()),
        working_conflict_rejected_count=int((decision == 2).sum()), accepted_merge_count=int((decision == 3).sum()),
        complete_original_node_count=n, partition_uses_identity_or_reference=False)
    return result, audit, dict(processing_order=np.asarray(order), decision_code=decision)


def link_evaluation(features, identities):
    identities = np.asarray(identities)
    result = dict(node_count=len(identities), no_position_node_count=int((features['position_event_count'] == 0).sum()),
        single_position_node_count=int((features['position_event_count'] == 1).sum()))
    for prefix, key in [('time_eligible', 'time_eligible_pairs'), ('candidate', 'candidate_edges')]:
        pairs = features[key]
        known = (identities[pairs[:, 0]] != '') & (identities[pairs[:, 1]] != '')
        labels = identities[pairs[:, 0]] == identities[pairs[:, 1]]
        result.update({prefix+'_count': len(pairs), prefix+'_known_count': int(known.sum()),
                       prefix+'_true_count': int((known & labels).sum()), prefix+'_false_count': int((known & ~labels).sum())})
    denominator = result['time_eligible_true_count']
    result['candidate_true_link_recall'] = result['candidate_true_count']/denominator if denominator else None
    return result


def construct_predictions(physical, memberships):
    n = len(physical['native_states'])
    outputs, all_groups = {}, {}
    order = sorted(set(physical['sources'].tolist()))
    if not 1 <= len(order) <= 2:
        raise ValueError('original one/two-source scene required')
    for partition in PARTITIONS:
        groups = old.groups_from_membership(memberships[partition], n)
        all_groups[partition] = groups
        first = np.asarray([g[0] for g in groups], np.int64)
        mean = old.ordinary_fusion(physical['native_states'], physical['nearest_seconds'], physical['support'], groups)
        mean = mean['predictions']['mean_supported'][first]
        weights = np.zeros((n, 64))
        for group in groups:
            members = np.asarray(group)
            low = members[physical['sources'][members] == order[0]]
            high = members[physical['sources'][members] == order[1]] if len(order) == 2 else np.empty(0, np.int64)
            count = physical['support'][low].sum(0)
            weights[low] = physical['support'][low]/np.maximum(count, 1)
            high_count = physical['support'][high].sum(0)
            weights[high] = physical['support'][high]*(count == 0)/np.maximum(high_count, 1)
        low = recover_group_states(physical['native_states'], weights, groups)['group_states']
        low[..., 2:] = mean[..., 2:]
        names = ('WS_MEAN', 'WS_LOW_SOURCE_SUPPORTED') if partition == 'WS' else (partition+'_MEAN', partition+'_LOW')
        outputs[names[0]], outputs[names[1]] = mean, low
    return outputs, all_groups


def evaluate_scene(fragment, predictions, groups, memberships, record, retained):
    # Use the complete registry from immutable121 rather than inventing a
    # different reference interpolation/first representative convention.
    saved = retained['WS_MEAN']
    targets = fragment['target_identity_ids']
    old._same(saved['target_ids'], targets, '121 full target registry')
    truth = saved['aligned_truth'][:len(targets)]
    valid = saved['aligned_truth_valid'][:len(targets)]
    registry = {str(identity): i for i, identity in enumerate(targets)}
    for node, identity in enumerate(fragment['identities']):
        old._same(fragment['target'][node], truth[registry[str(identity)]], '056/121 target')
        old._same(fragment['target_valid'][node], valid[registry[str(identity)]], '056/121 target valid')
    regions = old.fixed_regions(fragment, targets)
    artifacts, methods, arrays = {}, {}, {}
    for name in METHODS:
        partition = 'NO_REJECT' if name.startswith('NO_REJECT') else 'T_REJECT' if name.startswith('T_REJECT') else 'WS'
        emitted = emit_group_outputs(predictions[name], groups[partition], node_count=len(fragment['native_states']))
        report = old.evaluate_full_scene(emitted, node_identities=fragment['identities'], target_identities=targets,
            target_states=truth, target_valid=valid, node_observed=fragment['support'], split='V_select',
            grouping_kind='predicted', correspondence_policy='anchor_identity_first', return_artifact=True,
            node_input_support=dict(support=fragment['support'], nearest_seconds=fragment['nearest_seconds']))
        artifact = report.pop('artifact')
        if name in retained:
            if set(artifact) != set(retained[name]):
                raise ValueError('121 complete artifact schema differs')
            for key in artifact:
                old._same(artifact[key], retained[name][key], '121 exact '+name+'/'+key)
        methods[name], arrays[name] = qc.describe(report, artifact, fragment, regions)
        artifacts[name] = artifact
    pairs, pair_arrays = {}, {}
    for baseline, candidate in COMPARISONS:
        key = candidate+'_vs_'+baseline
        pairs[key], pair_arrays[key], _ = old.compare_pair(artifacts[baseline], artifacts[candidate], regions)
        pairs[key].update(baseline_method=baseline, candidate_method=candidate,
                         side_labels=dict(A0=baseline, candidate=candidate))
    raw = old.frozen.evaluate_scenes([record], old.frozen.THRESHOLD)
    association = {'RAW': raw}
    accepted = old.frozen.predict(record['scores'], old.frozen.THRESHOLD)
    truth_membership = old.frozen.truth_membership(record)
    details = {}
    for partition, membership in memberships.items():
        co, detail = old.frozen.partition_details(membership, record['pairs'], accepted, truth_membership)
        association[partition] = old.frozen.evaluate_scenes([dict(record, scores=co.astype(float))], .5)
        details[partition] = detail
    return dict(scene_id=record['scene_id'], component_id=record['component_id'],
        methods=methods, pairwise=pairs, association=association, partition_details=details,
        original121_two_artifacts_exact=True), dict(methods=arrays, pairs=pair_arrays), artifacts


def load_native(native_run, split, sid):
    with np.load(native_run/'private/native-events'/split/f'scene-{sid}.npz', allow_pickle=False) as saved:
        return {key: saved[key].copy() for key in NATIVE_KEYS}


def load_identities(native_run, split, sid, node_ids):
    with np.load(native_run/'private/native-evaluation'/split/f'scene-{sid}.npz', allow_pickle=False) as saved:
        old._same(saved['node_ids'], node_ids, 'native/evaluation original node order')
        ids = saved['identities'].copy()
    if ids.shape != node_ids.shape or ids.dtype.kind not in 'US':
        raise ValueError('one post-freeze identity per native node required')
    return ids


def aggregate(rows, arrays):
    result = qc.aggregate(rows, arrays, method_names=METHODS)
    result['association'] = {}
    for name in ('RAW', *PARTITIONS):
        values = [r['association'][name] for r in rows]
        result['association'][name] = dict(old.frozen._aggregate([v['per_scene'][0] for v in values]),
            itsa=old.frozen._aggregate([v['itsa']['per_scene'][0] for v in values]),
            mstsa=old.frozen._aggregate([v['mstsa']['per_scene'][0] for v in values]))
    return result


def run(args):
    source, partition, control, native_run, output = (Path(getattr(args, key)).resolve()
        for key in ('source_system', 'partition_run', 'control_run', 'native_run', 'output'))
    for protected in (source.parent, partition, control, native_run):
        if output == protected or protected in output.parents or output in protected.parents:
            raise ValueError('new output must be disjoint from immutable inputs')
    if not 1 <= args.cpu_threads <= 4:
        raise ValueError('cpu_threads must be1..4')
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    state = dict(status='RUNNING', idea_id=IDEA, plan_version='1.0', started_at=datetime.now(timezone.utc).isoformat(),
        source_system=str(source), partition_run=str(partition), control_run=str(control), native_run=str(native_run),
        completed_training_scenes=0, completed_scenes=0, methods=list(METHODS),
        models_constructed=0, model_forwards=0, optimizer_updates=0, gpu_used=False,
        neural_training=False, test_opened=False, threshold_fitted_on_T_only=True,
        shared_centers_applied=False, new_association_scores=False, native_archive_read=False,
        cached_reference_read=False, metadata_binding_uses_existing_evaluation_summaries=True)
    def checkpoint(**extra):
        state.update(elapsed_seconds=time.monotonic()-started, peak_rss_bytes=old.frozen.peak_rss_bytes(),
            advisory_exceeded=time.monotonic()-started > 1800, **extra)
        old.frozen.write(output/'run.json', state)
        if state['peak_rss_bytes'] > old.frozen.RAM_BUDGET:
            raise MemoryError('4GiB resident-memory budget exceeded')
    try:
        checkpoint()
        if sys.platform.startswith('linux'):
            soft, hard = resource.getrlimit(resource.RLIMIT_AS)
            limit = min([old.frozen.MAX_AS]+[v for v in (soft, hard) if v != resource.RLIM_INFINITY])
            resource.setrlimit(resource.RLIMIT_AS, (limit, hard))
            if hasattr(os, 'sched_getaffinity'):
                os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:args.cpu_threads])
        manifest = old.frozen.read(args.manifest)
        meta = old.metadata(source, partition, manifest)
        cache = old.frozen.read(native_run/'native-cache.json')
        training_ids = [str(sid) for sid in manifest['splits']['T_core'][:32]]
        if (cache.get('status') != 'CACHE_COMPLETED' or cache.get('scene_ids', {}).get('T_core') != training_ids
                or cache.get('scene_ids', {}).get('V_select') != meta['ids']):
            raise ValueError('completed frozen127 T32/V32 native cache required')
        run121, summary121 = (old.frozen.read(control/name) for name in ('run.json', 'summary.json'))
        if (control.name != CONTROL_RUN or run121.get('status') != 'COMPLETED' or run121.get('exit_code') != 0
                or summary121.get('status') != 'COMPLETED' or run121.get('scene_ids') != meta['ids']
                or run121.get('idea_id') != 'IDEA-SOURCE-PREFERENCE-CONTROL-20261029'
                or summary121.get('idea_id') != run121.get('idea_id')
                or run121.get('plan_version') != '1.0' or summary121.get('plan_version') != '1.0'
                or Path(run121['source_system']).resolve() != source or Path(run121['partition_run']).resolve() != partition):
            raise ValueError('completed original121 control/source/pilot binding required')
        checkpoint(training_scene_ids=training_ids, scene_ids=meta['ids'])
        (output/'scenes').mkdir(); (output/'private').mkdir(); (output/'training').mkdir()
        for split in ('T_core', 'V_select'):
            (output/'private'/split).mkdir()
        training, train_distances, train_labels, train_known = [], [], [], []
        for sid in training_ids:
            native = load_native(native_run, 'T_core', sid)
            features = native_features(native)
            np.savez_compressed(output/'private/T_core'/f'features-{sid}.npz', **features)
            identities = load_identities(native_run, 'T_core', sid, native['node_ids'])
            edges = features['candidate_edges']
            labels = identities[edges[:, 0]] == identities[edges[:, 1]]
            known = (identities[edges[:, 0]] != '') & (identities[edges[:, 1]] != '')
            np.savez_compressed(output/'private/T_core'/f'labels-{sid}.npz', labels=labels, known=known)
            row = dict(scene_id=sid, split='T_core', **link_evaluation(features, identities),
                       feature_file_written_before_identity_read=True)
            old.frozen.write(output/'training'/f'scene-{sid}.json.gz', row)
            training.append(row); train_distances.append(features['candidate_distance_m'])
            train_labels.append(labels); train_known.append(known)
            checkpoint(completed_training_scenes=len(training), last_training_scene_id=sid)
        model, curve = fit_rejection(np.concatenate(train_distances), np.concatenate(train_labels), np.concatenate(train_known))
        old.frozen.write(output/'rejection-model.json', model)
        np.savez_compressed(output/'private/training-threshold-curve.npz', **curve)
        old.frozen.write(output/'training-summary.json.gz', training)
        checkpoint(phase='V_evaluation', rejection_model=model)
        rows, arrays = [], []
        with np.load(meta['association']/'final-scores.npz', allow_pickle=False) as scores:
            old._same(scores['scene_ids'], meta['all_ids'], '056 full scene order')
            for sid in meta['ids']:
                native = load_native(native_run, 'V_select', sid)
                features = native_features(native)
                np.savez_compressed(output/'private/V_select'/f'features-{sid}.npz', **features)
                with np.load(source/'predictions'/f'states-{sid}.npz', allow_pickle=False) as saved:
                    physical = {key: saved[key].copy() for key in PHYSICAL_KEYS}
                for key in ('node_ids', 'sources', 'query_seconds'):
                    old._same(native[key], physical[key], 'native/056 '+key)
                membership = np.asarray(old.frozen.read(partition/'private'/f'scene-{sid}.json.gz')['memberships']['WS'])
                memberships, audits, traces = {'WS': membership}, {}, {}
                for name in ('NO_REJECT', 'T_REJECT'):
                    memberships[name], audits[name], traces[name] = merge_partition(membership, features, model, reject=name == 'T_REJECT')
                predictions, groups = construct_predictions(physical, memberships)
                frozen = {name+'_group_states': value for name, value in predictions.items()}
                frozen.update({name+'_membership': value for name, value in memberships.items()})
                frozen.update({name+'_'+key: value for name, trace in traces.items() for key, value in trace.items()})
                np.savez_compressed(output/'private'/f'predictions-{sid}.npz', **frozen)
                # The partition/physical predictions above are complete before
                # any V identity/reference field is opened.
                fragment, _ = old._load_arrays(source, sid); state['cached_reference_read'] = True
                native_ids = load_identities(native_run, 'V_select', sid, native['node_ids'])
                old._same(native_ids, fragment['identities'], 'native/056 identities after freeze')
                record = old.frozen.load_record(scores, sid, str(meta['components'][sid]))
                for key in ('node_ids', 'sources'):
                    old._same(record[key], native[key], '056 score/native '+key)
                index = meta['all_ids'].index(sid)
                old.frozen.counts_match(old.frozen.evaluate_scenes([record], old.frozen.THRESHOLD),
                    {scope: meta['expected_raw'][scope][index] for scope in old.frozen.SCOPES}, 'original056 raw')
                with np.load(control/'private'/f'states-{sid}.npz', allow_pickle=False) as saved:
                    keys = {key[len('WS_MEAN_'):] for key in saved.files if key.startswith('WS_MEAN_')}
                    retained = {name: {key: saved[name+'_'+key].copy() for key in keys}
                                for name in ('WS_MEAN', 'WS_LOW_SOURCE_SUPPORTED')}
                row, errors, artifacts = evaluate_scene(fragment, predictions, groups, memberships, record, retained)
                original114 = old.frozen.read(partition/'scenes'/f'scene-{sid}.json.gz')['arms']['WS']['decoded']
                old.frozen.counts_match(row['association']['WS'], old.frozen.metric_rows(original114), 'original114 WS decoded')
                row.update(link_candidates=link_evaluation(features, native_ids), merge_audits=audits,
                    output_file_written_before_V_identity_or_reference_read=True)
                for name, trace in traces.items():
                    edges = features['candidate_edges'][trace['decision_code'] == 3]
                    row['merge_audits'][name]['accepted_same_identity_node_links'] = int(
                        (native_ids[edges[:, 0]] == native_ids[edges[:, 1]]).sum())
                old.frozen.write(output/'scenes'/f'scene-{sid}.json.gz', row)
                payload = {name+'_'+key: value for name, artifact in artifacts.items() for key, value in artifact.items()}
                np.savez_compressed(output/'private'/f'states-{sid}.npz', **payload)
                rows.append(row); arrays.append(errors)
                checkpoint(completed_scenes=len(rows), last_scene_id=sid)
        components = sorted({row['component_id'] for row in rows})
        def subset(indices):
            return aggregate([rows[i] for i in indices], [arrays[i] for i in indices]) if indices else None
        largest = max(components, key=lambda cid: sum(row['component_id'] == cid for row in rows))
        result = dict(status='COMPLETED', idea_id=IDEA, plan_version='1.0', **aggregate(rows, arrays),
            training_scene_ids=training_ids, rejection_model=model,
            components={cid: subset([i for i, row in enumerate(rows) if row['component_id'] == cid]) for cid in components},
            without_largest_component=subset([i for i, row in enumerate(rows) if row['component_id'] != largest]),
            original121_two_artifacts_exact=True, no_expansion=True,
            limitations=['T/V share identities/components; not independent identity generalization',
                'native overlap/CV and greedy finite union are work assumptions, not calibrated posterior',
                'T balanced-accuracy rejection does not optimize full object/coverage/tail risk',
                'no shared centers, learned state model, credible peer or online end-to-end claim'])
        old.frozen.write(output/'scene-summary.json.gz', rows)
        checkpoint(phase='finalizing')
        result.update(elapsed_seconds=state['elapsed_seconds'], peak_rss_bytes=state['peak_rss_bytes'])
        old.frozen.write(output/'summary.json', result)
        checkpoint(status='COMPLETED', exit_code=0, finished_at=datetime.now(timezone.utc).isoformat())
        return result
    except BaseException as error:
        checkpoint(status='INTERRUPTED' if isinstance(error, (KeyboardInterrupt, InterruptedError)) else 'FAILED',
            exit_code=1, error=f'{type(error).__name__}: {error}', finished_at=datetime.now(timezone.utc).isoformat())
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('source-system', 'partition-run', 'control-run', 'native-run', 'manifest', 'output'):
        parser.add_argument('--'+key, type=Path, required=True)
    parser.add_argument('--cpu-threads', type=int, default=4)
    args = parser.parse_args()
    def interrupted(signum, frame):
        raise InterruptedError(f'signal {signum}')
    signal.signal(signal.SIGTERM, interrupted); signal.signal(signal.SIGINT, interrupted)
    try:
        run(args)
    except BaseException:
        traceback.print_exc(); return 1
    print(f'COMPLETED: {args.output}/summary.json')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
