"""One bounded independent NumPy audit of completed 136 saved predictions.

No production policy, evaluator, fitting or aggregation function is imported.
Native caches stay in place. Public output contains aggregate counts only.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import resource
import signal
import sys
import time

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_name] = '4'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import numpy as np
from tools.analysis import audit_observation_common as common

core = common.core
RADIUS = 6371000.
EQUAL = 'WS_SOURCE_EQUAL'
OLD = ('WS_MEAN', 'WS_NEAREST_SUPPORTED', 'WS_LOW_SOURCE_SUPPORTED',
       'WS_FIXED_LOW_SOURCE_SUPPORTED', 'WS_QUADRATIC_VALUE_GATE', EQUAL,
       'WS_SHARED_Q_SOFT', 'WS_SHARED_Q_HARD', 'WS_SHARED_LINEAR_SOFT', 'WS_SHARED_SHUFFLED_SOFT')
NEW = ('WS_SOURCE_NEAREST', 'WS_OBS_OLS_F1', 'WS_OBS_OLS_F100', 'WS_OBS_LOO_F1', 'WS_OBS_LOO_F100')
T_METHODS, V_METHODS = (EQUAL,)+NEW, OLD+NEW
PRIMARY = tuple(name+'_vs_'+EQUAL for name in NEW[1:])
MOTION_KEYS = {'states', 'support', 'nearest_seconds', 'sources', 'native_count', 'span_seconds',
               'fit_rmse_m', 'ols_velocity_mps', 'node_ids', 'query_seconds'}
NATIVE_KEYS = {'node_ids', 'sources', 'query_seconds', 'event_offsets', 'event_times_seconds',
               'event_values', 'event_valid', 'event_node_ids', 'event_canonical_indices'}
RAM, MAX_AS = 6*1024**3, 12*1024**3


def npz(path):
    with np.load(path, allow_pickle=False) as payload:
        return {key: payload[key] for key in payload.files}


def close(actual, expected, checks, label, atol=1e-9):
    common.close(actual, expected, checks, label, atol=atol)


def native_proxies(motion, native, checks):
    """Reconstruct interpolation residuals without target, reference or source role."""
    checks.exact(set(motion) == MOTION_KEYS and set(native) == NATIVE_KEYS, 'physical cache allowlists')
    n = len(motion['node_ids'])
    for key in ('node_ids', 'sources', 'query_seconds'):
        checks.array(native[key], motion[key], 'native binding '+key)
    t, values, valid, offsets = (native[k] for k in
        ('event_times_seconds', 'event_values', 'event_valid', 'event_offsets'))
    checks.exact(t.ndim == 1 and values.shape == valid.shape == (len(t), 4)
                 and valid.dtype == bool and np.isfinite(t).all(), 'native arrays')
    checks.exact(offsets.shape == (n+1,) and offsets.dtype.kind in 'iu' and offsets[0] == 0
                 and offsets[-1] == len(t) and np.all(np.diff(offsets) >= 2), 'native event offsets')
    checks.exact(valid[:, :2].all() and np.isfinite(values[:, :2]).all()
                 and (np.abs(values[:, 0]) <= 90).all() and (np.abs(values[:, 1]) <= 180).all(), 'native positions')
    checks.exact(native['event_node_ids'].shape == native['event_canonical_indices'].shape == (len(t),),
                 'native event lineage shape')
    loo, known, residual_count = np.zeros(n), np.zeros(n, bool), np.zeros(n, np.int64)
    for node, (start, stop) in enumerate(zip(offsets[:-1], offsets[1:])):
        ts, p = t[start:stop], values[start:stop, :2]
        checks.exact(np.all(np.diff(ts) > 0)
                     and np.all(native['event_node_ids'][start:stop] == motion['node_ids'][node]), 'native time/node lineage')
        checks.array(native['event_canonical_indices'][start:stop], np.arange(stop-start), 'canonical event order')
        if len(ts) == 2:
            continue
        longitude = np.r_[p[0, 1], p[0, 1]+np.cumsum((np.diff(p[:, 1])+180.) % 360.-180.)]
        xy = np.column_stack((np.deg2rad(p[:, 0]-p[0, 0])*RADIUS,
            np.deg2rad(longitude-longitude[0])*RADIUS*np.cos(np.deg2rad(p[:, 0]).mean())))
        alpha = (ts[1:-1]-ts[:-2])/(ts[2:]-ts[:-2])
        predicted = xy[:-2]*(1-alpha[:, None])+xy[2:]*alpha[:, None]
        residual2 = np.sum((xy[1:-1]-predicted)**2, axis=1)
        loo[node] = np.mean(residual2/(1.+(1.-alpha)**2+alpha**2))
        known[node], residual_count[node] = True, len(ts)-2
    rmse = motion['fit_rmse_m']
    checks.exact(rmse.shape == (n,) and np.isfinite(rmse).all() and (rmse >= 0).all(), 'OLS nonnegative finite cache')
    return dict(ols_proxy_m2=rmse**2, loo_proxy_m2=loo, loo_known=known,
        loo_residual_count=residual_count, native_event_count=len(t), native_fragment_count=n)


def structure(motion, membership, checks):
    n, q = motion['support'].shape
    checks.exact(motion['support'].dtype == bool and motion['states'].shape == (n, q, 4)
                 and q == 64 and np.isfinite(motion['states']).all(), 'native Q64 motion/support')
    checks.exact(motion['nearest_seconds'].shape == (n, q) and np.isfinite(motion['nearest_seconds']).all()
                 and (motion['nearest_seconds'] >= 0).all(), 'native nearest physical field')
    checks.exact(membership.shape == (n,) and membership.dtype.kind in 'iu', 'membership shape')
    unique = np.unique(membership)
    checks.array(unique, np.arange(len(unique)), 'contiguous group labels')
    groups = [np.flatnonzero(membership == gid) for gid in unique]
    members, offsets, group_index = [], [0], []
    mapping = np.empty(n, np.int64)
    for gid, group in enumerate(groups):
        # First-occurrence source equivalence only, never lexical source ranking.
        by_source = {}
        for node in group:
            by_source.setdefault(str(motion['sources'][node]), []).append(node)
        for selected in by_source.values():
            selected = np.asarray(selected, np.int64)
            mapping[selected] = len(members)
            members.append(selected)
            group_index.append(gid)
        offsets.append(len(members))
    counts = np.stack([motion['support'][selected].sum(0) for selected in members])
    return dict(groups=groups, members=members, group_offsets=np.asarray(offsets, np.int64),
        source_group_indices=np.asarray(group_index, np.int64), node_to_source_row=mapping,
        supported_counts=counts, available=counts > 0,
        source_member_count=np.asarray([len(selected) for selected in members], np.int64))


def source_proxies(motion, struct, node, label):
    quality = node[label+'_proxy_m2']
    known = node['loo_known'] if label == 'loo' else np.ones(len(quality), bool)
    support = motion['support']
    values = np.zeros(struct['available'].shape)
    information = np.zeros(values.shape, bool)
    for row, selected in enumerate(struct['members']):
        mask = support[selected]
        values[row] = (quality[selected, None]*mask).sum(0)/np.maximum(mask.sum(0), 1)
        information[row] = mask.any(0) & ~((~known[selected, None]) & mask).any(0)
    return values, information


def weights_for(method, motion, struct, proxies):
    available = struct['available']
    source_weights = np.zeros(available.shape)
    fallback = np.zeros((len(struct['groups']), available.shape[1]), bool)
    potential_floor = np.zeros(available.shape, bool)
    floor_hit = np.zeros_like(potential_floor)
    quality, known = proxies['loo' if '_LOO_' in method else 'ols']
    nearest = np.stack([np.min(np.where(motion['support'][selected], motion['nearest_seconds'][selected], np.inf), axis=0)
                        for selected in struct['members']])
    floor = (100. if method.endswith('F100') else 1.) if method.startswith('WS_OBS_') else None
    for gid, (a, b) in enumerate(zip(struct['group_offsets'][:-1], struct['group_offsets'][1:])):
        present = available[a:b]
        if method == EQUAL:
            mass = present.astype(float)
        elif method == 'WS_SOURCE_NEAREST':
            mass = (present & (nearest[a:b] == nearest[a:b].min(0))).astype(float)
        else:
            unknown = (present & ~known[a:b]).any(0)
            fallback[gid] = unknown
            potential_floor[a:b] = present & known[a:b] & (quality[a:b] < floor)
            floor_hit[a:b] = potential_floor[a:b] & ~unknown
            mass = present/np.maximum(quality[a:b], floor)
            mass[:, unknown] = present[:, unknown]
        total = mass.sum(0)
        source_weights[a:b] = np.divide(mass, total, out=np.zeros_like(mass), where=total > 0)
    mapping = struct['node_to_source_row']
    node_weights = source_weights[mapping]*motion['support']/np.maximum(struct['supported_counts'][mapping], 1)
    return dict(source_weights=source_weights, node_weights=node_weights, floor_hit=floor_hit,
                uninformative_fallback=fallback, potential_floor_hit=potential_floor)


def mean_and_recovery(motion, struct, node_weights=None):
    """Direct original-node operator; never normalize circular source means first."""
    states, support = motion['states'], motion['support']
    output, audit = [], {key: [] for key in ('zero_weight_fallback_count', 'longitude_zero_resultant_count',
                                            'heading_zero_resultant_count', 'single_member_copy_count')}
    for members in struct['groups']:
        values = states[members]
        mass = support[members].astype(float) if node_weights is None else node_weights[members]
        totals = mass.sum(0)
        norm = mass/np.maximum(totals, 1e-300)
        prediction = values[0].copy()
        live = totals > 0
        for column in (0, 2):
            # Original ordinary Mean sums before dividing; weighted recovery
            # divides the mass first. Preserve that declared numerical order.
            val = ((values[:, :, column]*mass).sum(0)/np.maximum(totals, 1)
                   if node_weights is None else (values[:, :, column]*norm).sum(0))
            prediction[live, column] = val[live]
        for column, name in ((1, 'longitude'), (3, 'heading')):
            angles = np.deg2rad(values[:, :, column])
            sine = (np.sin(angles)*norm).sum(0)
            cosine = (np.cos(angles)*norm).sum(0)
            if node_weights is None:
                sine = (np.sin(angles)*mass).sum(0)/np.maximum(totals, 1)
                cosine = (np.cos(angles)*mass).sum(0)/np.maximum(totals, 1)
            bad = live & (np.hypot(sine, cosine) <= 1e-12)
            value = np.rad2deg(np.arctan2(sine, cosine))
            value = (value+180) % 360-180 if column == 1 else value % 360
            prediction[live & ~bad, column] = value[live & ~bad]
            audit[name+'_zero_resultant_count'].append(int(bad.sum()))
        single = (mass > 0).sum(0) == 1
        if node_weights is not None:
            selected = mass.argmax(0)
            prediction[single] = values[selected[single], np.flatnonzero(single)]
        audit['zero_weight_fallback_count'].append(int((~live).sum()))
        audit['single_member_copy_count'].append(int(single.sum()))
        output.append(prediction)
    return np.stack(output), audit


def inspect_physical(run_path, native_path, source_path, value_path, partition_path, split, row, checks):
    sid = row['scene_id']
    motion_cache = npz(native_path/'private/native-motion'/split/f'scene-{sid}.npz')
    checks.exact(MOTION_KEYS.issubset(motion_cache), 'required physical motion fields')
    motion = {key: motion_cache[key] for key in MOTION_KEYS}
    native = npz(native_path/'private/native-events'/split/f'scene-{sid}.npz')
    saved = npz(run_path/'private'/f'policy-{split}-{sid}.npz')
    member = (npz(value_path/'private'/f'T-features-{sid}.npz')['membership'] if split == 'T_core'
              else np.asarray(core.read(partition_path/'private'/f'scene-{sid}.json.gz')['memberships']['WS'], np.int64))
    checks.array(saved['membership'], member, 'membership original125/114')
    for key in ('node_ids', 'query_seconds', 'support'):
        checks.array(saved[key], motion[key], 'saved/raw binding '+key)
    struct = structure(motion, member, checks)
    for key in ('group_offsets', 'source_group_indices', 'node_to_source_row', 'supported_counts', 'available', 'source_member_count'):
        checks.array(saved[key], struct[key], 'source structure '+key)
    supported_sources = np.stack([struct['available'][a:b].sum(0)
        for a, b in zip(struct['group_offsets'][:-1], struct['group_offsets'][1:])])
    support_scope = dict(zero=int((supported_sources == 0).sum()), one=int((supported_sources == 1).sum()),
                         at_least_two=int((supported_sources >= 2).sum()))
    checks.exact(row['source_support_scope'] == support_scope and row['node_count'] == len(motion['states']),
                 'full node and zero/one/multi-source counts')
    node = native_proxies(motion, native, checks)
    for key, value in node.items():
        close(saved['native_'+key], np.asarray(value), checks, 'native proxy '+key, atol=1e-7)
    checks.exact(row['native_event_count'] == node['native_event_count']
                 and row['native_loo_unknown_fragments'] == int((~node['loo_known']).sum())
                 and row['native_loo_residual_count'] == int(node['loo_residual_count'].sum()), 'native public counts')
    proxies = {label: source_proxies(motion, struct, node, label) for label in ('ols', 'loo')}
    for label, (values, known) in proxies.items():
        close(saved[label+'__source_proxy_m2'], values, checks, 'source proxy '+label, atol=1e-7)
        checks.array(saved[label+'__source_known'], known, 'source proxy known '+label)
    names = T_METHODS if split == 'T_core' else V_METHODS
    artifacts = core.artifacts(run_path/'private'/f'states-{split}-{sid}.npz', names, template=names[0])
    base = artifacts[names[0]]
    checks.array(base['node_group_indices'], member, 'artifact original membership')
    ev = npz(native_path/'private/native-evaluation'/split/f'scene-{sid}.npz')
    for key in ('node_ids', 'query_seconds'):
        checks.array(ev[key], motion[key], 'evaluation physical binding '+key)
    checks.array(base['target_ids'], ev['target_ids'], 'original target registry')
    target_count = len(ev['target_ids'])
    checks.array(base['aligned_truth'][:target_count], ev['target_states'], 'original target states')
    checks.array(base['aligned_truth_valid'][:target_count], ev['target_valid'], 'original target validity')
    target_lookup = {str(target): i for i, target in enumerate(ev['target_ids'])}
    anchor_indices = np.asarray([target_lookup[str(ev['identities'][group[0]])] for group in struct['groups']], np.int64)
    checks.array(base['group_anchor_target_indices'], anchor_indices, 'first node original identity binding')
    mean, _ = mean_and_recovery(motion, struct)
    diagnostics = {}
    for name in T_METHODS:
        expected = weights_for(name, motion, struct, proxies)
        for key in ('source_weights', 'node_weights', 'floor_hit', 'uninformative_fallback'):
            close(saved[name+'__'+key], expected[key], checks, name+' '+key)
        prediction, recover_audit = mean_and_recovery(motion, struct, expected['node_weights'])
        zero = np.stack([~motion['support'][group].any(0) for group in struct['groups']])
        prediction[zero] = mean[zero]
        prediction[..., 2:] = mean[..., 2:]
        close(saved[name+'__group_states'], prediction, checks, name+' frozen physical prediction')
        close(artifacts[name]['group_states'], prediction, checks, name+' evaluated physical prediction')
        checks.array(saved[name+'__group_states'], artifacts[name]['group_states'], name+' pre-evaluation prediction binding')
        checks.array(artifacts[name]['group_states'][..., 2:], mean[..., 2:], name+' unchanged speed/heading')
        public = row['source_contributions'][name]
        checks.exact(public['recovery'] == recover_audit, name+' recovery fallback audit')
        counts = dict(uninformative_fallback_group_queries=int(expected['uninformative_fallback'].sum()),
            floor_hit_source_queries=int(expected['floor_hit'].sum()), available_source_queries=int(struct['available'].sum()))
        for key, value in counts.items():
            checks.exact(public[key] == value, name+' '+key)
        diagnostics[name] = dict(counts, potential_floor_hit_source_queries=int(expected['potential_floor_hit'].sum()),
            longitude_zero_resultant_count=sum(recover_audit['longitude_zero_resultant_count']),
            zero_support_group_queries=int(zero.sum()))
    if split == 'V_select':
        old = core.artifacts(source_path/'private'/f'states-{sid}.npz', OLD)
        for name in OLD:
            checks.exact(set(old[name]) == set(artifacts[name]), 'original133 artifact schema')
            for key in old[name]:
                checks.array(artifacts[name][key], old[name][key], 'exact original133 '+name+' '+key)
        checks.array(artifacts['WS_MEAN']['group_states'], mean, 'original133 native Mean')
    return diagnostics, node['native_event_count'], node['native_fragment_count']


def harm_summary(results, rows):
    """Post-hoc description, not a decision rule or additional experiment."""
    def subset(indices):
        output = {}
        for name in NEW:
            delta = np.concatenate([results[i][name]['errors']-results[i][EQUAL]['errors'] for i in indices])
            positive = delta[delta > 0]
            largest_n = int(np.ceil(len(delta)*.01))
            largest_harm_n = int(np.ceil(len(positive)*.01))
            damage = float(positive.sum())
            output[name] = dict(common_position_points=len(delta), positive_harm_points=len(positive),
                positive_harm_over_1000m_points=int((delta > 1000).sum()), positive_harm_sum_m=damage,
                top_one_percent_all_points_count=largest_n,
                top_one_percent_all_points_harm_fraction=float(np.sort(positive)[-largest_n:].sum()/damage) if damage else None,
                top_one_percent_harmed_points_count=largest_harm_n,
                top_one_percent_harmed_points_harm_fraction=float(np.sort(positive)[-largest_harm_n:].sum()/damage) if damage else None,
                mean_candidate_minus_equal_m=float(delta.mean()) if len(delta) else None)
        return output
    return dict(scope='after-prediction descriptive positive radial-error increase; no threshold selection or causality',
        all_components=subset(list(range(len(rows)))),
        by_component={c: subset([i for i, row in enumerate(rows) if row['component_id'] == c])
                      for c in sorted({row['component_id'] for row in rows})})


def audit_run(run_path, native_path, source_path, manifest_path, checks):
    run, summary, manifest = (core.read(p) for p in (run_path/'run.json', run_path/'summary.json', manifest_path))
    checks.exact(run['status'] == summary['status'] == 'COMPLETED' and run['exit_code'] == 0, 'completed136')
    checks.exact(run['idea_id'] == summary['idea_id'] == 'IDEA-OBSERVATION-QUALITY-20261030'
                 and run['plan_version'] == summary['plan_version'] == '1.0', '136 fixed contract')
    checks.exact(tuple(run['T_methods']) == T_METHODS and tuple(run['methods']) == V_METHODS, 'complete declared arms')
    checks.exact(run['fixed_floors_m2'] == [1., 100.] and tuple(summary['primary_comparisons']) == PRIMARY, 'fixed floors and comparisons')
    for key, path in (('native_run', native_path), ('source_risk_run', source_path), ('manifest', manifest_path)):
        checks.exact(Path(run[key]).resolve() == path.resolve(), 'CLI lineage '+key)
    value_path, partition_path = Path(run['value_run']), Path(run['partition_run'])
    old = core.read(source_path/'run.json')
    for path in (native_path, value_path, partition_path, source_path):
        state = core.read(path/'run.json')
        checks.exact(state['status'] == 'COMPLETED' and state['exit_code'] == 0, 'completed original input')
    for key in ('native_run', 'value_run', 'partition_run'):
        checks.exact(Path(old[key]).resolve() == Path(run[key]).resolve(), '133 lineage '+key)
    native_cache = core.read(native_path/'native-cache.json')
    checks.exact(native_cache['status'] == 'CACHE_COMPLETED', 'native cache terminal')
    for key in ('teacher_forward_calls', 'neural_models_constructed', 'optimizer_updates', 'target_reference_fit_count', 'analytic_head_fits'):
        checks.exact(run[key] == 0, 'declared zero '+key)
    for key in ('archive_read', 'csv_read', 'test_opened', 'source_id_feature', 'gpu_used'):
        checks.exact(run[key] is False, 'declared disabled '+key)
    checks.exact(run['predictions_frozen_before_reference'] is True and run['time_policy'] == 'advisory'
                 and 1 <= run['cpu_threads'] <= 4, 'declared permission/resources')
    tids, vids = run['training_scene_ids'], run['scene_ids']
    checks.exact(tids == manifest['splits']['T_core'][:32] == old['training_scene_ids']
                 and vids == old['scene_ids'] and set(vids).issubset(manifest['splits']['V_select'])
                 and not set(tids) & set(vids), 'original development populations')
    checks.exact(native_cache['scene_ids'] == {'T_core': tids, 'V_select': vids}, 'native cache scene registry')
    scopes, counts, harms = {}, {}, {}
    event_count = fragment_count = 0
    for split, ids, names, component_count, rowfile, sm in (
        ('T_core', tids, T_METHODS, 12, 'training-scene-summary.json.gz', summary['training_summary']),
        ('V_select', vids, V_METHODS, 9, 'scene-summary.json.gz', summary)):
        rows = core.read(run_path/rowfile)
        checks.exact(set(sm['primary_bootstrap']) == set(PRIMARY), split+' four primary bootstraps')
        totals = {name: {} for name in T_METHODS}
        for row in rows:
            checks.exact(row['component_id'] == str(manifest['components'][row['scene_id']]) and row['split'] == split,
                         split+' original component/split')
            diagnostic, events, fragments = inspect_physical(run_path, native_path, source_path, value_path,
                                                            partition_path, split, row, checks)
            event_count += events
            fragment_count += fragments
            for name, local in diagnostic.items():
                for key, value in local.items():
                    totals[name][key] = totals[name].get(key, 0)+value
            checks.exact(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform == 'darwin' else 1024) <= RAM,
                         'bounded resident memory')
        for name in T_METHODS:
            for key, value in sm['contribution_counts'][name].items():
                checks.exact(value == totals[name][key], split+' aggregate contribution count')
        results, scope = common.inspect_metric_block(run_path, split, rows, sm, names, checks,
            expected_scene_ids=ids, expected_component_count=component_count, expected_primary_pairs=PRIMARY)
        scopes[split], counts[split], harms[split] = scope, totals, harm_summary(results, rows)
    checks.exact(run['completed_training_scene_count'] == run['completed_scene_count'] == 32, 'complete T/V endpoint')
    checks.exact(run['native_event_count'] == summary['native_event_count'] == event_count
                 and run['native_fragment_count'] == summary['native_fragment_count'] == fragment_count, 'native work counters')
    return dict(metric_scope=scopes, contribution_counts=counts, posthoc_harm=harms,
        native_event_count=event_count, native_fragment_count=fragment_count,
        floor_semantics='floor_hit is applied inverse only, excluding unknown fallback; potential_floor_hit includes known low-proxy entries on fallback queries')


def audit(run_path, native_path, source_path, manifest_path, output):
    started, before, checks = time.monotonic(), resource.getrusage(resource.RUSAGE_SELF), core.Checks()
    paths = [Path(p).resolve() for p in (run_path, native_path, source_path, manifest_path, output)]
    run_path, native_path, source_path, manifest_path, output = paths
    if any(output == p or p in output.parents for p in (native_path, source_path, manifest_path)):
        raise ValueError('audit output must not mutate original inputs')
    report = dict(status='RUNNING', started_at=datetime.now(timezone.utc).isoformat(),
        scope='independent136 raw-native proxies, source/node weights, physical outputs, original133 replay, population/main metrics and bootstrap',
        not_audited=['physical correctness of reference or sensor-noise identifiability',
            'production process access chronology beyond separated API and frozen-array equality',
            'speed/heading error metrics (outputs are exact-bound to original Mean)',
            'original127 OLS fitting correctness (cached fit_rmse squared is independently bound)',
            'source feature vectors or descriptive proxy-reference correlation',
            'additional support-conditioned pair strata beyond main all/observed/missing metrics'],
        time_policy='advisory fixed endpoint, no retries', gpu_used=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream)
    try:
        if sys.platform.startswith('linux'):
            if hasattr(os, 'sched_getaffinity'):
                os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:4])
            soft, hard = resource.getrlimit(resource.RLIMIT_AS)
            limit = min([MAX_AS]+[x for x in (soft, hard) if x != resource.RLIM_INFINITY])
            resource.setrlimit(resource.RLIMIT_AS, (limit, hard))
        report.update(audit_run(run_path, native_path, source_path, manifest_path, checks), status='COMPLETED', exit_code=0)
    except BaseException as error:
        report.update(status='FAILED', exit_code=1, error_type=type(error).__name__,
            error=str(error) if isinstance(error, ValueError) else 'independent audit did not complete; preserve original artifacts')
    finally:
        usage = resource.getrusage(resource.RUSAGE_SELF)
        report.update(exact_checks=checks.exact_checks, numeric_checks=checks.numeric_checks, null_checks=checks.null_checks,
            maximum_absolute_difference=checks.maximum_absolute_difference, elapsed_seconds=time.monotonic()-started,
            advisory_exceeded=time.monotonic()-started > 1800,
            user_cpu_seconds=usage.ru_utime-before.ru_utime, system_cpu_seconds=usage.ru_stime-before.ru_stime,
            peak_rss_bytes=int(usage.ru_maxrss*(1 if sys.platform == 'darwin' else 1024)),
            finished_at=datetime.now(timezone.utc).isoformat())
        with output.open('w', encoding='utf-8') as stream:
            json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('run', 'native-run', 'source-risk-run', 'manifest', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    def interrupted(signum, frame):
        raise InterruptedError('audit interrupted')
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    result = audit(args.run, args.native_run, args.source_risk_run, args.manifest, args.output)
    print(result['status']+': '+str(args.output))
    return result['exit_code']


if __name__ == '__main__':
    raise SystemExit(main())
