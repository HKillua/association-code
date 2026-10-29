"""Independent bounded NumPy audit of 138 T-only cached conflict statistics.

No production diagnostic, predictor, proxy calculator or grid search is used.
Endpoint credit is relative to the first-anchor target even in mixed groups;
Q support-side equalities within C/D are partition identities, not validation.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from itertools import combinations
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
from tools.analysis import audit_shared_control_arrays as core
from tools.analysis.audit_observation_common import close

PREF = ('LOO', 'OLS', 'NEAREST', 'QOOF')
METHODS = ('WS_SOURCE_EQUAL', 'WS_SOURCE_NEAREST', 'WS_OBS_OLS_F1', 'WS_OBS_OLS_F100',
           'WS_OBS_LOO_F1', 'WS_OBS_LOO_F100', 'QUADRATIC_OOF_HARD')
CELLS = ('A', 'B', 'C', 'D', 'E', 'U')
RAM, MAX_AS = 6*1024**3, 12*1024**3
ORACLE = ('endpoint_errors_m', 'best_single_error_m', 'grid_error_m', 'gain_m', 'gap_bound_m', 'min_resultant')


def npz(path):
    with np.load(path, allow_pickle=False) as payload:
        return {key: payload[key] for key in payload.files}


def radial(a, b):
    """Angle between unit vectors, independent of the production evaluator."""
    def unit(p):
        lat, lon = np.deg2rad(p[..., 0]), np.deg2rad(p[..., 1])
        return np.stack((np.cos(lat)*np.cos(lon), np.cos(lat)*np.sin(lon), np.sin(lat)), axis=-1)
    x, y = unit(a), unit(b)
    return 6371000.*np.arctan2(np.linalg.norm(np.cross(x, y), axis=-1), (x*y).sum(-1))


def preference_diagnostics(values, known, endpoint, roles):
    """Hypothetical endpoint choice only; never constructs a physical prediction."""
    n = len(endpoint)
    winner = np.full((n, 4), -2, np.int64)
    mass = np.zeros((n, 4, 2))
    for j in range(4):
        good = known[:, j]
        ties = good & (values[:, j, 0] == values[:, j, 1])
        winner[ties, j] = -1
        mass[ties, j] = .5
        for side in (0, 1):
            chosen = good & (values[:, j, side] < values[:, j, 1-side])
            winner[chosen, j] = side
            mass[chosen, j, side] = 1.
    cost = np.sum(mass*endpoint[:, None, :], axis=-1)
    best = endpoint.min(1)
    regret = cost-best[:, None]
    credit = np.sum(mass*(endpoint == best[:, None])[:, None, :], axis=-1)
    for array in (cost, regret, credit):
        array[~known] = np.nan
    cell = np.full(n, 5, np.int64)
    live = known[:, 0] & known[:, 2] & known[:, 3]
    strict = live & (winner[:, [0, 2, 3]] >= 0).all(1)
    cell[live & ~strict] = 4
    l, nn, q = winner[:, 0], winner[:, 2], winner[:, 3]
    cell[strict & (l == nn) & (nn == q)] = 0
    cell[strict & (l == nn) & (nn != q)] = 1
    cell[strict & (l != nn) & (q == l)] = 2
    cell[strict & (l != nn) & (q == nn)] = 3
    sub = np.full(n, -1, np.int64)
    conflict = (cell == 2) | (cell == 3)
    sub[conflict & ~known[:, 1]] = 3
    sub[conflict & known[:, 1] & (winner[:, 1] == -1)] = 2
    sub[conflict & (winner[:, 1] >= 0) & (winner[:, 1] == l)] = 0
    sub[conflict & (winner[:, 1] >= 0) & (winner[:, 1] == nn)] = 1
    role_best = np.full(n, 2, np.int64)
    for side in (0, 1):
        use = endpoint[:, side] < endpoint[:, 1-side]
        role_best[use] = roles[use, side]
    return dict(preference_mass=mass, preference_winner=winner, selection_cost_m=cost,
        selection_regret_m=regret, selection_credit=credit, cell=cell, ols_subcell=sub, role_best=role_best)


def source_members(pack, membership, motion, checks, label):
    mapping, offsets = pack['node_to_source_row'], pack['group_offsets']
    checks.exact(mapping.shape == membership.shape and mapping.dtype.kind in 'iu', label+' node-source shape')
    groups = [np.flatnonzero(membership == gid) for gid in range(int(membership.max())+1)]
    checks.array(np.unique(membership), np.arange(len(groups)), label+' complete group labels')
    checks.exact(offsets.shape == (len(groups)+1,) and offsets[0] == 0
                 and np.all(np.diff(offsets) > 0), label+' source offsets')
    members, nearest, counts = [], [], []
    for gid, group in enumerate(groups):
        classes = {}
        for node in group:
            classes.setdefault(str(motion['sources'][node]), []).append(int(node))
        rows = list(range(int(offsets[gid]), int(offsets[gid+1])))
        checks.exact(len(rows) == len(classes), label+' source equivalence class count')
        expected_sets = {tuple(v) for v in classes.values()}
        found_sets = set()
        for row in rows:
            nodes = np.flatnonzero(mapping == row)
            key = tuple(nodes.tolist())
            checks.exact(key in expected_sets and key not in found_sets, label+' source membership equivalence')
            found_sets.add(key)
            members.append(nodes)
            mask = motion['support'][nodes]
            counts.append(mask.sum(0))
            nearest.append(np.min(np.where(mask, motion['nearest_seconds'][nodes], np.inf), axis=0))
    count = np.stack(counts)
    checks.array(pack['supported_counts'], count, label+' supported counts')
    checks.array(pack['available'], count > 0, label+' availability')
    return groups, members, np.stack(nearest)


def inspect_join(paths, sid, component, checks, *, prior_mixture=None, prior_quality=None):
    """Rebuild the new join from old T artifacts, without any new model output."""
    qrun, srun, mrun, nrun = (paths[k] for k in ('quality_run', 'source_run', 'mixture_run', 'native_run'))
    quality = npz(qrun/'private'/f'policy-T_core-{sid}.npz')
    diag = npz(qrun/'private'/f'diagnostic-T_core-{sid}.npz')
    source = npz(srun/'private'/f'training-{sid}.npz')
    oof = npz(srun/'private'/f'oof-{sid}.npz')
    mixture = npz(mrun/'private'/f'scene-{sid}.npz')
    motion = npz(nrun/'private/native-motion/T_core'/f'scene-{sid}.npz')
    ev = npz(nrun/'private/native-evaluation/T_core'/f'scene-{sid}.npz')
    roles = npz(mrun/'private/roles.npz')
    role_map = {str(k): int(v) for k, v in zip(roles['source_names'], roles['indices'])}
    checks.exact(len(role_map) == len(roles['source_names']) and set(role_map.values()) == {0, 1}, 'original K2 roles')
    for old in (source, quality, mixture, ev):
        for key in ('node_ids', 'query_seconds'):
            checks.array(old[key], motion[key], 'original node/query join '+key)
    checks.exact(motion['support'].dtype == bool and motion['states'].shape == (*motion['support'].shape, 4)
                 and motion['support'].shape[1] == 64, 'native original Q64 scope')
    checks.exact(len(set(motion['node_ids'].tolist())) == len(motion['node_ids']), 'unique original node identifiers')
    checks.array(quality['support'], motion['support'], '136 original support')
    membership = source['membership']
    for old in (quality, mixture):
        checks.array(old['membership'], membership, 'original membership join')
    checks.exact(str(source['component_id']) == component, '133 original component')
    groups, members, nearest = source_members(source, membership, motion, checks, '133')
    _, qmembers, _ = source_members(quality, membership, motion, checks, '136')
    qkeys = {tuple(nodes.tolist()): row for row, nodes in enumerate(qmembers)}
    mapping = np.asarray([qkeys[tuple(nodes.tolist())] for nodes in members], np.int64)
    checks.exact(len(set(mapping.tolist())) == len(mapping), 'source row unique bijection')
    for key in ('features', 'source_states'):
        checks.array(quality[key][mapping], source[key], 'original physical pack binding '+key)
    for key in ('labeled', 'sample_weight', 'errors_m'):
        if key == 'errors_m':
            close(diag[key][mapping], source[key], checks, 'source diagnostic labels '+key, atol=1e-5)
        else:
            checks.array(diag[key][mapping], source[key], 'source diagnostic labels '+key)
    registry = {str(target): i for i, target in enumerate(ev['target_ids'])}
    checks.exact(len(registry) == len(ev['target_ids']), 'unique original target registry')
    scope = dict(total_group_queries=len(groups)*64, source_support_zero=0, source_support_one=0,
        source_support_multi=0, complete_target_reference_valid_query_count=int(ev['target_valid'][..., :2].all(-1).sum()),
        labeled_group_query_count=0, duplicate_anchor_group_count=0, unknown_anchor_group_count=0,
        first_anchor_target_count=0, target_without_anchor_count=0, first_anchor_reference_invalid_query_count=0,
        first_anchor_reference_valid_zero_source_count=0, first_anchor_reference_valid_one_source_count=0,
        first_anchor_reference_valid_multi_source_count=0, mixed_identity_group_count=0,
        duplicate_group_queries=0, unknown_group_queries=0, reference_valid_points_without_anchor=0)
    gi, qi, sr, targets, pure = [], [], [], [], []
    expected_labeled = np.zeros(source['available'].shape, bool)
    claimed, anchor_target, first_group = set(), [], {}
    source_counts = []
    for gid, nodes in enumerate(groups):
        a, b = source['group_offsets'][gid:gid+2]
        available = source['available'][a:b]
        count = available.sum(0)
        source_counts.append(count)
        checks.exact(np.max(count) <= 2, 'K2 conditional diagnostic only')
        for name, mask in (('zero', count == 0), ('one', count == 1), ('multi', count >= 2)):
            scope['source_support_'+name] += int(mask.sum())
        identity = str(ev['identities'][nodes[0]])
        target = registry.get(identity, -1)
        anchor_target.append(target)
        is_pure = len(set(ev['identities'][nodes].tolist())) == 1
        scope['mixed_identity_group_count'] += int(not is_pure)
        if target < 0:
            scope['unknown_anchor_group_count'] += 1
            scope['unknown_group_queries'] += 64
            continue
        if identity in claimed:
            scope['duplicate_anchor_group_count'] += 1
            scope['duplicate_group_queries'] += 64
            continue
        claimed.add(identity)
        first_group[target] = gid
        valid = ev['target_valid'][target, :, :2].all(-1)
        scope['first_anchor_reference_invalid_query_count'] += int((~valid).sum())
        for name, mask in (('zero', count == 0), ('one', count == 1), ('multi', count >= 2)):
            scope['first_anchor_reference_valid_'+name+'_source_count'] += int((valid & mask).sum())
        for query in np.flatnonzero(valid & (count == 2)):
            rows = np.flatnonzero(available[:, query])+a
            expected_labeled[rows, query] = True
            gi.append(gid); qi.append(query); sr.append(rows); targets.append(target); pure.append(is_pure)
    gi, qi = np.asarray(gi, np.int64), np.asarray(qi, np.int64)
    sr = np.asarray(sr, np.int64).reshape(-1, 2)
    targets = np.asarray(targets, np.int64)
    checks.array(source['labeled'], expected_labeled, 'first representative D2 labels')
    expected_label_weights = np.zeros(expected_labeled.shape)
    for a, b in zip(source['group_offsets'][:-1], source['group_offsets'][1:]):
        expected_label_weights[a:b] = expected_labeled[a:b]/np.maximum(source['available'][a:b].sum(0), 1)
    checks.array(source['sample_weight'], expected_label_weights, 'complete original label weights')
    scope.update(first_anchor_target_count=len(claimed), target_without_anchor_count=len(registry)-len(claimed),
        labeled_group_query_count=len(gi), reference_valid_points_without_anchor=int(sum(
            ev['target_valid'][target, :, :2].all(-1).sum() for name, target in registry.items() if name not in claimed)))
    for key, value in (('group_indices', gi), ('query_indices', qi), ('source_rows', sr)):
        checks.array(mixture[key], value, '134 D2 key binding '+key)
    checks.exact(len(set(zip(gi.tolist(), qi.tolist()))) == len(gi), 'unique D2 query keys')
    close(source['sample_weight'][sr, qi[:, None]].sum(1), np.ones(len(gi)), checks, 'one diagnostic weight per query')
    truth = ev['target_states'][targets, qi, :2]
    checks.array(mixture['truth'], truth, '134 first-anchor original reference')
    endpoint = mixture['endpoint_errors_m']
    checks.exact(endpoint.shape == (len(gi), 2) and np.isfinite(endpoint).all() and (endpoint >= 0).all(), 'finite endpoint errors')
    close(endpoint, source['errors_m'][sr, qi[:, None]], checks, 'endpoint source label binding', atol=1e-5)
    checks.array(mixture['best_single_error_m'], endpoint.min(1), 'best original endpoint')
    original_label_error = source['errors_m'][sr, qi[:, None]]
    scope['endpoint_label_preference_roundoff_disagreements'] = int(np.any(
        (endpoint == endpoint.min(1, keepdims=True)) !=
        (original_label_error == original_label_error.min(1, keepdims=True)), axis=1).sum())
    close(mixture['gain_m'], endpoint.min(1)-mixture['grid_error_m'], checks, 'existing grid gain')
    expected_roles = np.asarray([[role_map[str(motion['sources'][members[row][0]])] for row in pair] for pair in sr], np.int64).reshape(-1, 2)
    checks.array(mixture['role_indices'], expected_roles, 'original anonymous roles')
    checks.array(mixture['group_pure'], np.asarray(pure, bool), 'group identity purity')
    qrows = mapping[sr]
    values = np.stack((quality['loo__source_proxy_m2'][qrows, qi[:, None]],
        quality['ols__source_proxy_m2'][qrows, qi[:, None]], nearest[sr, qi[:, None]],
        oof['QUADRATIC__scores'][sr, qi[:, None]]), axis=1)
    known = np.column_stack((quality['loo__source_known'][qrows, qi[:, None]].all(1),
        quality['ols__source_known'][qrows, qi[:, None]].all(1), np.ones((len(gi), 2), bool)))
    checks.exact(np.isfinite(values[known]).all(), 'finite known preference values')
    checks.array(values[:, 3], mixture['oof_QUADRATIC_scores'], 'original OOF scores exact binding')
    pd = preference_diagnostics(values, known, endpoint, expected_roles)
    close(quality['WS_SOURCE_NEAREST__source_weights'][qrows, qi[:, None]], pd['preference_mass'][:, 2], checks,
          'nearest preference binds existing136 weights')
    close(pd['selection_regret_m'][:, 3], mixture['QUADRATIC_selection_regret_m'], checks, 'original Q selection regret', atol=1e-5)
    for cell_value, side in ((2, 0), (3, 2)):
        use = pd['cell'] == cell_value
        checks.array(pd['selection_regret_m'][use, 3], pd['selection_regret_m'][use, side], 'Q support partition identity')
    artifacts = core.artifacts(qrun/'private'/f'states-T_core-{sid}.npz', METHODS[:6], template=METHODS[0])
    base, nt = artifacts[METHODS[0]], len(ev['target_ids'])
    for key, expected in (('target_ids', ev['target_ids']), ('node_group_indices', membership),
                          ('group_node_indices', np.concatenate(groups)),
                          ('group_offsets', np.r_[0, np.cumsum([len(nodes) for nodes in groups])]),
                          ('group_anchor_target_indices', np.asarray(anchor_target, np.int64))):
        checks.array(base[key], expected, '136 complete artifact binding '+key)
    checks.array(base['aligned_target_ids'][:nt], ev['target_ids'], '136 first-target registry order')
    checks.exact(len(base['aligned_group_indices']) == nt+scope['duplicate_anchor_group_count']
                 and scope['unknown_anchor_group_count'] == 0, '136 retained duplicate population and known anchors')
    if prior_quality is None:
        quality_rows = [row for row in core.read(qrun/'training-scene-summary.json.gz') if row['scene_id'] == sid]
        checks.exact(len(quality_rows) == 1, 'unique original136 T public row')
        prior_quality = quality_rows[0]
    checks.exact(prior_quality['scene_id'] == sid and prior_quality['component_id'] == component,
                 'original136 public row binding')
    expected_population = dict(target_identity_count=nt, predicted_group_count=len(groups),
        duplicate_prediction_group_count=scope['duplicate_anchor_group_count'],
        anchor_matched_target_count=len(claimed),
        reference_unavailable_target_count=int((~ev['target_valid'].any(axis=(1, 2))).sum()))
    for name in METHODS[:6]:
        method = prior_quality['methods'][name]
        for key, value in expected_population.items():
            checks.exact(method['population'][key] == value, '136 original population '+name+'/'+key)
        checks.exact(method['unmatched_target_count'] == nt-len(claimed), '136 original unmatched population')
    checks.array(base['aligned_group_indices'][:nt], np.asarray([first_group.get(i, -1) for i in range(nt)], np.int64),
                 '136 full first representative mapping')
    checks.array(base['aligned_truth'][:nt], ev['target_states'], '136 original truth')
    checks.array(base['aligned_truth_valid'][:nt], ev['target_valid'], '136 original valid')
    checks.exact(str(base['split']) == 'T_core', 'T-only physical artifact')
    evalmask = base['aligned_truth_valid'][:nt, :, :2].all(-1) & base['aligned_prediction_valid'][:nt, :, :2].all(-1)
    ft, fq = np.nonzero(evalmask)
    fg = base['aligned_group_indices'][:nt][ft]
    checks.exact(np.all(fg >= 0), 'full evaluation requires first group')
    inverse = {(int(group), int(query)): i for i, (group, query) in enumerate(zip(fg, fq))}
    checks.exact(len(inverse) == len(ft), 'unique full evaluated group/query')
    d2full = np.asarray([inverse[int(g), int(q)] for g, q in zip(gi, qi)], np.int64)
    full, actual = [], []
    for name in METHODS[:6]:
        art = artifacts[name]
        for key in base:
            if key not in ('group_states', 'aligned_prediction'):
                checks.array(art[key], base[key], '136 unchanged population '+key)
        for target in range(nt):
            group = int(base['aligned_group_indices'][target])
            if group >= 0:
                checks.array(art['aligned_prediction'][target], art['group_states'][group], '136 existing group output binding')
        checks.array(quality[name+'__group_states'], art['group_states'], '136 frozen before-evaluation state binding')
        checks.array(art['group_states'][..., 2:], base['group_states'][..., 2:], '136 unchanged speed and heading')
        full.append(radial(art['aligned_prediction'][ft, fq, :2], ev['target_states'][ft, fq, :2]))
        actual.append(radial(art['group_states'][gi, qi, :2], truth))
    full = np.stack(full, axis=1)
    qhard = mixture['QUADRATIC_hard_error_m']
    close(qhard, radial(mixture['QUADRATIC_hard_position'], truth), checks, 'existing Qhard physical error', atol=1e-5)
    actual = np.column_stack((*actual, qhard))
    close(actual[:, :6], full[d2full], checks, 'D2 versus full136 errors', atol=1e-5)
    strictq = pd['preference_winner'][:, 3] >= 0
    close(actual[strictq, 6], pd['selection_cost_m'][strictq, 3], checks, 'strict Q endpoint versus actual hard', atol=1e-5)
    for a, b in ((2, 3), (4, 5)):
        checks.array(artifacts[METHODS[a]]['group_states'], artifacts[METHODS[b]]['group_states'], 'fixed floor original outputs identical')
        checks.array(quality[METHODS[a]+'__source_weights'], quality[METHODS[b]+'__source_weights'], 'fixed floor original weights identical')
    source_counts = np.stack(source_counts)
    fsource = source_counts[fg, fq]
    checks.array(fsource[d2full], np.full(len(gi), 2, np.int64), 'D2 full-source count binding')
    scope['full_evaluable_position_points'] = len(ft)
    expected = dict(node_ids=motion['node_ids'], query_seconds=motion['query_seconds'], membership=membership,
        component_id=np.asarray(component), group_indices=gi, query_indices=qi, source_rows=sr,
        quality_source_rows=qrows, source_row_133_to_136=mapping, role_indices=expected_roles,
        group_pure=np.asarray(pure, bool), preference_names=np.asarray(PREF), preference_values=values,
        preference_known=known, actual_method_names=np.asarray(METHODS), actual_error_m=actual,
        full_actual_error_m=full, full_target_indices=ft, full_query_indices=fq, full_group_indices=fg,
        full_source_count=fsource, D2_full_indices=d2full, **pd)
    expected.update({key: mixture[key] for key in ORACLE})
    if prior_mixture is None:
        prior = [r for r in core.read(mrun/'scene-summary.json.gz') if r['scene_id'] == sid]
        checks.exact(len(prior) == 1, 'unique original134 public row')
        old_row = prior[0]
    else:
        old_row = prior_mixture
    checks.exact(old_row['scene_id'] == sid and old_row['component_id'] == component, 'original134 public row binding')
    for key in old_row['scope']:
        checks.exact(scope[key] == old_row['scope'][key], 'complete134 scope '+key)
    return expected, scope


def stats(values):
    x = np.asarray(values, float)
    x = x[np.isfinite(x)]
    n = len(x)
    return dict(count=n, sum=float(x.sum()), mean=float(x.mean()) if n else None,
        median=float(np.median(x)) if n else None, rmse=float(np.sqrt(np.mean(x*x))) if n else None,
        p95=float(np.quantile(x, .95)) if n else None, p99=float(np.quantile(x, .99)) if n else None,
        min=float(x.min()) if n else None, max=float(x.max()) if n else None)


def delta_stats(values):
    x = np.asarray(values, float)
    x = x[np.isfinite(x)]
    positive, negative, large, small = x > 0, x < 0, x > 1000, (x > 0) & (x <= 1000)
    return dict(count=len(x), signed_sum=float(x.sum()), mean=float(x.mean()) if len(x) else None,
        positive_count=int(positive.sum()), negative_count=int(negative.sum()), zero_count=int((x == 0).sum()),
        positive_sum=float(x[positive].sum()), benefit_sum=float(-x[negative].sum()),
        positive_gt1000_count=int(large.sum()), positive_gt1000_sum=float(x[large].sum()),
        positive_0_to_1000_count=int(small.sum()), positive_0_to_1000_sum=float(x[small].sum()),
        min=float(x.min()) if len(x) else None, max=float(x.max()) if len(x) else None)


def compare_tree(actual, expected, checks, label):
    """Exact schema/count/null checks, numeric tolerance only for measured values."""
    if isinstance(expected, dict):
        checks.exact(isinstance(actual, dict) and set(actual) == set(expected), label+' schema')
        for key in expected:
            compare_tree(actual[key], expected[key], checks, label+'/'+key)
    elif isinstance(expected, (list, tuple)):
        checks.exact(isinstance(actual, (list, tuple)) and len(actual) == len(expected), label+' length')
        for i, (a, e) in enumerate(zip(actual, expected)):
            compare_tree(a, e, checks, label+'/'+str(i))
    elif expected is None:
        checks.null_checks += 1
        checks.exact(actual is None, label+' null')
    elif isinstance(expected, (bool, int, np.integer, str)):
        checks.exact(actual == expected, label)
    else:
        checks.numeric_checks += 1
        difference = abs(float(actual)-float(expected))
        checks.maximum_absolute_difference['statistic'] = max(checks.maximum_absolute_difference.get('statistic', 0.), difference)
        checks.exact(np.isfinite(actual) and np.isclose(actual, expected, rtol=1e-9, atol=1e-5), label+' numeric')


def layer(data, components, use):
    n = int(use.sum())
    roles = data['role_best'][use]
    pure = data['group_pure'][use]
    result = dict(query_count=n, component_count=len(set(components[use].tolist())),
        role_counts={name: int((roles == i).sum()) for i, name in enumerate(('role_0_best', 'role_1_best', 'reference_tie'))},
        purity_counts=dict(pure=int(pure.sum()), mixed=int((~pure).sum())), preferences={}, actual={})
    for j, name in enumerate(PREF):
        known = data['preference_known'][use, j]
        winner = data['preference_winner'][use, j]
        item = dict(known_count=int(known.sum()), unknown_count=int((~known).sum()),
                    strict_count=int((winner >= 0).sum()), tie_count=int((winner == -1).sum()))
        for key in ('selection_cost_m', 'selection_regret_m', 'selection_credit'):
            item[key] = stats(data[key][use, j])
        result['preferences'][name] = item
    errors = data['actual_error_m'][use]
    for j, name in enumerate(METHODS):
        result['actual'][name] = dict(error_m=stats(errors[:, j]),
            vs_equal=delta_stats(errors[:, j]-errors[:, 0]),
            vs_best_single=delta_stats(errors[:, j]-data['best_single_error_m'][use]),
            vs_grid=delta_stats(errors[:, j]-data['grid_error_m'][use]))
    gain, bounds = data['gain_m'][use], data['gap_bound_m'][use]
    result['oracle'] = dict(gain_m=stats(gain), gain_gt10_count=int((gain > 10).sum()),
        gain_gt10_sum=float(gain[gain > 10].sum()), gap_bound_m=stats(bounds),
        uncertified_count=int((~np.isfinite(bounds)).sum()), min_resultant=stats(data['min_resultant'][use]))
    return result


def primary_arrays(data):
    conflict = (data['cell'] == 2) | (data['cell'] == 3)
    regret = data['selection_regret_m']
    return dict(actual_LOO_minus_NEAREST=(data['actual_error_m'][:, 4]-data['actual_error_m'][:, 1], conflict),
        selection_L_minus_N=(regret[:, 0]-regret[:, 2], conflict & data['preference_known'][:, 0] & data['preference_known'][:, 2]),
        selection_O_minus_N=(regret[:, 1]-regret[:, 2], conflict & data['preference_known'][:, 1] & data['preference_known'][:, 2]),
        selection_Q_minus_N=(regret[:, 3]-regret[:, 2], conflict & data['preference_known'][:, 3] & data['preference_known'][:, 2]))


def joined(items):
    variable = ('group_indices', 'query_indices', 'endpoint_errors_m', 'best_single_error_m', 'grid_error_m',
        'gain_m', 'gap_bound_m', 'min_resultant', 'role_indices', 'group_pure', 'preference_values', 'preference_known',
        'preference_mass', 'preference_winner', 'selection_cost_m', 'selection_regret_m', 'selection_credit',
        'actual_error_m', 'cell', 'ols_subcell', 'role_best', 'full_actual_error_m', 'full_source_count')
    return {key: np.concatenate([item[key] for item in items], axis=0) for key in variable}


def analysis_block(items, scopes, component_names):
    data = joined(items)
    components = np.concatenate([np.repeat(component, len(item['cell'])) for item, component in zip(items, component_names)])
    scope = {key: sum(local[key] for local in scopes) for key in scopes[0]}
    n = len(data['cell'])
    masks = {name: data['cell'] == i for i, name in enumerate(CELLS)}
    out = dict(query_count=n, component_count=len(set(component_names)), scope=scope,
        full_population=dict(point_count=len(data['full_actual_error_m']), methods={
            name: stats(data['full_actual_error_m'][:, j]) for j, name in enumerate(METHODS[:6])}),
        overall=layer(data, components, np.ones(n, bool)),
        main_cells={name: layer(data, components, mask) for name, mask in masks.items()},
        ols_subcells={}, role_cells={}, purity_cells={}, pairwise_preferences={}, primary_estimands={}, scene_macro_estimands={})
    for name in ('C', 'D'):
        for sub, label in enumerate(('L', 'N', 'TIE', 'UNKNOWN')):
            out['ols_subcells'][name+'_OLS_'+label] = layer(data, components, masks[name] & (data['ols_subcell'] == sub))
    for role, label in enumerate(('role_0_best', 'role_1_best', 'reference_tie')):
        out['role_cells'][label] = {name: layer(data, components, mask & (data['role_best'] == role)) for name, mask in masks.items()}
    for pure, label in ((True, 'pure'), (False, 'mixed')):
        out['purity_cells'][label] = {name: layer(data, components, mask & (data['group_pure'] == pure)) for name, mask in masks.items()}
    for a, b in combinations(range(4), 2):
        winner, known = data['preference_winner'], data['preference_known']
        both = known[:, a] & known[:, b]
        strict = both & (winner[:, a] >= 0) & (winner[:, b] >= 0)
        pair_masks = dict(strict_same=strict & (winner[:, a] == winner[:, b]),
            strict_opposite=strict & (winner[:, a] != winner[:, b]),
            at_least_one_tie=both & ((winner[:, a] == -1) | (winner[:, b] == -1)), any_unknown=~both)
        out['pairwise_preferences'][PREF[a]+'_vs_'+PREF[b]] = {key: layer(data, components, use) for key, use in pair_masks.items()}
    conflict_count = int(((data['cell'] == 2) | (data['cell'] == 3)).sum())
    for key, (value, use) in primary_arrays(data).items():
        out['primary_estimands'][key] = dict(stats(value[use]), conflict_query_count=conflict_count,
            eligible_component_count=len(set(components[use].tolist())),
            coverage=int(use.sum())/conflict_count if conflict_count else None)
        means = []
        for item in items:
            local, mask = primary_arrays(item)[key]
            if mask.any():
                means.append(float(local[mask].mean()))
        out['scene_macro_estimands'][key] = dict(mean=float(np.mean(means)) if means else None, defined_scene_count=len(means))
    return out


def bootstraps(items, component_names):
    components = sorted(set(component_names))
    draws = np.random.default_rng(20).integers(0, len(components), size=(2000, len(components)))
    out = {}
    for key in primary_arrays(items[0]):
        sums, counts = [], []
        for component in components:
            values = []
            for item, name in zip(items, component_names):
                if name == component:
                    value, use = primary_arrays(item)[key]
                    values.append(value[use])
            x = np.concatenate(values)
            sums.append(float(x.sum())); counts.append(len(x))
        sums, counts = np.asarray(sums), np.asarray(counts)
        denominator = counts[draws].sum(1)
        keep = denominator > 0
        values = sums[draws].sum(1)[keep]/denominator[keep]
        contributing = int((counts > 0).sum())
        out[key] = dict(comparison=key, components=components, component_count=len(components),
            contributing_component_count=contributing, resamples=2000, valid_resamples=int(keep.sum()),
            empty_resamples=int((~keep).sum()), seed=20,
            point_estimate_m=float(sums.sum()/counts.sum()) if counts.sum() else None,
            percentile95_m=np.quantile(values, [.025, .975]).tolist() if contributing >= 2 and len(values) else None,
            independent_test=False)
    return out


def metadata(paths, manifest_path, checks):
    manifest = core.read(manifest_path)
    original = {key: core.read(path/'run.json') for key, path in paths.items()}
    ids = dict(quality_run='IDEA-OBSERVATION-QUALITY-20261030', source_run='IDEA-SOURCE-RISK-20261030',
               mixture_run='IDEA-MIXTURE-OPPORTUNITY-20261030', native_run='IDEA-NATIVE-CV-STATE-CONTROL-20261029')
    for key, state in original.items():
        checks.exact(state['status'] == 'COMPLETED' and state['exit_code'] == 0 and state['idea_id'] == ids[key],
                     'completed original input '+key)
    quality, source, mixture = (original[key] for key in ('quality_run', 'source_run', 'mixture_run'))
    bindings = ((quality, 'source_risk_run', paths['source_run']), (quality, 'native_run', paths['native_run']),
        (source, 'native_run', paths['native_run']), (mixture, 'source_run', paths['source_run']),
        (mixture, 'native_run', paths['native_run']), (quality, 'manifest', manifest_path))
    for state, field, expected in bindings:
        checks.exact(Path(state[field]).resolve() == expected.resolve(), 'original lineage '+field)
    for field in ('value_run', 'partition_run'):
        checks.exact(Path(quality[field]).resolve() == Path(source[field]).resolve(), 'same original grouping '+field)
    checks.exact(quality['completed_training_scene_count'] == source['completed_training_scene_count'] == 32
                 and mixture['completed_scene_count'] == 32 and source['analytic_head_fits'] == 39,
                 'original fixed32 completed counts')
    checks.exact(tuple(quality['T_methods']) == METHODS[:6] and mixture['grid_size'] == 4097
                 and mixture['reference_oracle_only'] is True, 'original six arms and grid contract')
    tids = quality['training_scene_ids']
    marker = core.read(paths['native_run']/'native-cache.json')
    checks.exact(tids == source['training_scene_ids'] == mixture['scene_ids'] == manifest['splits']['T_core'][:32]
                 == marker['scene_ids']['T_core'] and len(tids) == len(set(tids)) == 32, 'fixed T32 lineage/order')
    checks.exact(marker['status'] == 'CACHE_COMPLETED', 'native cache terminal')
    components = sorted({str(manifest['components'][sid]) for sid in tids})
    checks.exact(len(components) == 12, 'T12 components')
    folds = core.read(paths['source_run']/'T-fold-index.json')
    found = set()
    for fold in folds:
        held = fold['heldout_component']
        if held is None:
            continue
        checks.exact(held in components and held not in found and held not in fold['training_components'], 'old QOOF isolation')
        found.add(held)
        selected = [sid for sid in tids if str(manifest['components'][sid]) != held]
        checks.exact(fold['training_scene_ids'] == selected
                     and sorted(fold['training_components']) == sorted(set(components)-{held}), 'old QOOF training population')
    checks.exact(found == set(components), 'complete original12 QOOF folds')
    return tids, manifest


def audit_run(run_path, paths, manifest_path, checks):
    run, summary = core.read(run_path/'run.json'), core.read(run_path/'summary.json')
    checks.exact(run['status'] == summary['status'] == 'COMPLETED' and run['exit_code'] == 0, 'completed138')
    checks.exact(run['idea_id'] == summary['idea_id'] == 'IDEA-OBSERVATION-CONFLICT-PROBE-20261030'
                 and run['plan_version'] == summary['plan_version'] == '1.0', 'fixed138 contract')
    for key, path in dict(paths, manifest=manifest_path).items():
        checks.exact(Path(run[key]).resolve() == path.resolve(), 'audit CLI lineage '+key)
    for key in ('new_fits', 'new_physical_predictions', 'model_forwards', 'neural_updates', 'teacher_forwards', 'new_grid_evaluations'):
        checks.exact(run[key] == 0, 'declared zero '+key)
    for key in ('native_events_opened', 'V_arrays_opened', 'test_opened', 'archive_read', 'source_id_feature', 'gpu_used'):
        checks.exact(run[key] is False, 'declared excluded '+key)
    checks.exact(run['original_QOOF_reused'] is True and run['time_policy'] == 'advisory'
                 and run['advisory_seconds'] == 1800 and 1 <= run['cpu_threads'] <= 4, 'fixed resource and QOOF declarations')
    checks.exact(tuple(summary['preference_names']) == PREF and tuple(summary['actual_method_names']) == METHODS,
                 'explicit preference/output order')
    tids, manifest = metadata(paths, manifest_path, checks)
    checks.exact(run['scene_ids'] == tids and run['completed_scene_count'] == 32, 'new fixed T32 completion')
    rows = core.read(run_path/'scene-summary.json.gz')
    checks.exact([row['scene_id'] for row in rows] == tids, 'complete ordered new scene rows')
    old_qrows = core.read(paths['quality_run']/'training-scene-summary.json.gz')
    checks.exact([row['scene_id'] for row in old_qrows] == tids, 'original136 T-only scene rows')
    old_mrows = core.read(paths['mixture_run']/'scene-summary.json.gz')
    checks.exact([row['scene_id'] for row in old_mrows] == tids, 'original134 T-only scene rows')
    old_mrows = {row['scene_id']: row for row in old_mrows}
    arrays, scopes, components, align_rows = [], [], [], []
    for row, old_row in zip(rows, old_qrows):
        sid, component = row['scene_id'], str(manifest['components'][row['scene_id']])
        checks.exact(row['component_id'] == old_row['component_id'] == component, 'public original component')
        checks.exact(core.read(run_path/'scenes'/f'scene-{sid}.json.gz') == row, 'public per-scene exact copy')
        expected, scope = inspect_join(paths, sid, component, checks, prior_mixture=old_mrows[sid], prior_quality=old_row)
        saved = npz(run_path/'private'/f'aligned-{sid}.npz')
        checks.exact(set(saved) == set(expected), 'complete private aligned schema')
        for key in expected:
            if key in ('actual_error_m', 'full_actual_error_m'):
                close(saved[key], expected[key], checks, 'independent saved-position errors '+key, atol=1e-5)
            elif key in ('selection_cost_m', 'selection_regret_m', 'selection_credit'):
                close(saved[key], expected[key], checks, 'independent selection '+key, atol=1e-8)
            else:
                checks.array(saved[key], expected[key], 'original/new exact '+key)
        # Strict signed counts concern the stored scalar representation. Its
        # position errors have just been checked via a different spherical
        # formula; do not let nanometre roundoff alter exact-zero statistics.
        expected['actual_error_m'] = saved['actual_error_m']
        expected['full_actual_error_m'] = saved['full_actual_error_m']
        checks.array(expected['actual_error_m'][:, :6], expected['full_actual_error_m'][expected['D2_full_indices']],
                     'saved full/D2 physical errors exact')
        for j, name in enumerate(METHODS[:6]):
            public = old_row['methods'][name]['position']['all']
            independent = stats(expected['full_actual_error_m'][:, j])
            checks.exact(public['intersection_count'] == independent['count'], 'old136 full point count')
            for oldkey, statkey in (('mae', 'mean'), ('rmse', 'rmse'), ('p95', 'p95'), ('p99', 'p99')):
                compare_tree(public[oldkey], independent[statkey], checks, 'old136 '+name+'/'+oldkey)
        block = analysis_block([expected], [scope], [component])
        actual_block = {key: value for key, value in row.items() if key not in ('scene_id', 'component_id')}
        compare_tree(actual_block, block, checks, 'scene '+sid)
        arrays.append(expected); scopes.append(scope); components.append(component)
        mapping = expected['source_row_133_to_136']
        align_rows.append(dict(scene_id=sid, source_rows=len(mapping),
            source_row_reordering_used=not np.array_equal(mapping, np.arange(len(mapping))),
            D2_queries=len(expected['cell']), labeled_source_queries=2*len(expected['cell']),
            full_evaluable_position_points=len(expected['full_actual_error_m']), exact_floor_pairs=2, original_QOOF_reused=True))
        checks.exact(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform == 'darwin' else 1024) <= RAM,
                     'bounded resident memory')
    compare_tree(summary['all_components'], analysis_block(arrays, scopes, components), checks, 'all components')
    unique = sorted(set(components))
    checks.exact(set(summary['by_component']) == set(unique), 'all12 component summaries')
    for component in unique:
        chosen = [i for i, c in enumerate(components) if c == component]
        block = analysis_block([arrays[i] for i in chosen], [scopes[i] for i in chosen], [components[i] for i in chosen])
        compare_tree(summary['by_component'][component], block, checks, 'component '+component)
    largest = max(unique, key=components.count)
    checks.exact(summary['largest_component_id'] == largest, 'original largest by scene count')
    keep = [i for i, c in enumerate(components) if c != largest]
    block = analysis_block([arrays[i] for i in keep], [scopes[i] for i in keep], [components[i] for i in keep])
    compare_tree(summary['without_largest_component'], block, checks, 'without largest component')
    compare_tree(summary['primary_bootstrap'], bootstraps(arrays, components), checks, 'all four primary bootstraps')
    alignment = core.read(run_path/'alignment.json')
    checks.exact(summary['alignment'] == alignment, 'public alignment copy')
    compare_tree(alignment['scenes'], align_rows, checks, 'all32 alignment records')
    checks.exact(alignment['status'] == 'COMPLETED' and alignment['scene_count'] == 32
                 and alignment['original_QOOF_reused'] is True
                 and alignment['original134_query_count'] == sum(len(data['cell']) for data in arrays)
                 and alignment['original136_full_point_count'] == sum(len(data['full_actual_error_m']) for data in arrays),
                 'alignment scope totals')
    checks.exact(alignment['input_file_count'] == run['input_files_opened']
                 and alignment['input_array_count'] == run['input_arrays_loaded'], 'declared reader-count binding')
    compare_tree(summary['component_scene_counts'], {c: components.count(c) for c in unique}, checks, 'component scene counts')
    original_mixture = core.read(paths['mixture_run']/'summary.json')  # Entirely T-only.
    total_scope = {key: sum(scope[key] for scope in scopes) for key in scopes[0]}
    for key, value in original_mixture['scope'].items():
        checks.exact(total_scope[key] == value, 'original134 total scope '+key)
    checks.exact(sum(len(data['cell']) for data in arrays) == original_mixture['all_components']['query_count'], 'original D2 total')
    return dict(scene_count=32, component_count=12, D2_query_count=total_scope['labeled_group_query_count'],
        full_T_position_points=total_scope['full_evaluable_position_points'], full_group_queries=total_scope['total_group_queries'],
        main_cells=6, ols_subcells=8, role_cells=18, purity_cells=12, preference_pairs=6, primary_bootstraps=4,
        source_reordering_scene_count=sum(int(row['source_row_reordering_used']) for row in align_rows),
        alignment_rows=align_rows)


def audit(run_path, quality_run, source_run, mixture_run, native_run, manifest_path, output):
    started, before, checks = time.monotonic(), resource.getrusage(resource.RUSAGE_SELF), core.Checks()
    run_path, manifest_path, output = (Path(p).resolve() for p in (run_path, manifest_path, output))
    paths = {key: Path(value).resolve() for key, value in dict(quality_run=quality_run, source_run=source_run,
              mixture_run=mixture_run, native_run=native_run).items()}
    if any(output == p or p in output.parents for p in list(paths.values())+[manifest_path]):
        raise ValueError('audit output overlaps original input')
    report = dict(status='RUNNING', started_at=datetime.now(timezone.utc).isoformat(),
        scope='independent138 T-only join, preference/choice diagnostics, old scalar/output binding, fixed cells, full denominators and bootstrap',
        boundaries=['Endpoint credit/regret compares first-anchor reference, including mixed groups; not sensor health',
            'Q/OLS support-side regret equalities inside conflict partitions are construction identities',
            'Only original133 Q head is LOCO; no new rule generalization is validated',
            'Strict signed statistics use saved scalars independently checked against old positions; no epsilon relabeling'],
        not_audited=['original134 grid optimization or bound correctness; existing scalar binding only',
            'original136 proxy and policy generation; existing arrays and output binding only',
            'original133 fitting, upstream teacher or any new prediction',
            'physical reference correctness, covariance identification, causal mechanisms, external generalization',
            'complete process access history beyond audited file paths and software guards'],
        time_policy='1800s advisory, one original audit endpoint, no automatic retries', gpu_used=False)
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
        report.update(audit_run(run_path, paths, manifest_path, checks), status='COMPLETED', exit_code=0)
    except BaseException as error:
        report.update(status='FAILED', exit_code=1, error_type=type(error).__name__,
            error=str(error) if isinstance(error, ValueError) else 'independent audit incomplete; preserve original artifacts')
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
    for name in ('run', 'quality-run', 'source-run', 'mixture-run', 'native-run', 'manifest', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    def interrupted(signum, frame):
        raise InterruptedError('audit interrupted')
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    result = audit(args.run, args.quality_run, args.source_run, args.mixture_run, args.native_run, args.manifest, args.output)
    print(result['status']+': '+str(args.output))
    return result['exit_code']


if __name__ == '__main__':
    raise SystemExit(main())
