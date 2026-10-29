"""138: T-only joined diagnosis of already saved preference and output errors.

No physical prediction, model forward, fitting, native-event access or grid
search is performed. Endpoint selection costs are diagnostics, not trajectories.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gzip
from itertools import combinations
import json
import os
from pathlib import Path
import resource
import signal
import sys
import time
import traceback

for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_name] = '4'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import numpy as np

IDEA = 'IDEA-OBSERVATION-CONFLICT-PROBE-20261030'
PREFERENCES = ('LOO', 'OLS', 'NEAREST', 'QOOF')
METHODS = ('WS_SOURCE_EQUAL', 'WS_SOURCE_NEAREST', 'WS_OBS_OLS_F1', 'WS_OBS_OLS_F100',
           'WS_OBS_LOO_F1', 'WS_OBS_LOO_F100')
ACTUAL = METHODS+('QUADRATIC_OOF_HARD',)
CELLS = ('A', 'B', 'C', 'D', 'E', 'U')
OLS_SUBCELLS = ('OLS_L', 'OLS_N', 'OLS_TIE', 'OLS_UNKNOWN')
ROLES = ('role_0_best', 'role_1_best', 'reference_tie')
PRIMARY = ('actual_LOO_minus_NEAREST', 'selection_L_minus_N', 'selection_O_minus_N', 'selection_Q_minus_N')
RAM, ADDRESS_SPACE = 6*1024**3, 12*1024**3
QUERY_KEYS = ('group_indices', 'query_indices', 'source_rows', 'quality_source_rows',
    'endpoint_errors_m', 'best_single_error_m', 'grid_error_m', 'gain_m', 'gap_bound_m', 'min_resultant',
    'role_indices', 'group_pure', 'preference_values', 'preference_known', 'preference_mass',
    'preference_winner', 'selection_cost_m', 'selection_regret_m', 'selection_credit',
    'actual_error_m', 'cell', 'ols_subcell', 'role_best')


def write(path, value):
    opener = gzip.open if str(path).endswith('.gz') else open
    with opener(path, 'wt', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False, indent=2)


class Reader:
    def __init__(self):
        self.files = []
        self.array_count = 0

    def json(self, path):
        path = Path(path)
        self.files.append(str(path))
        opener = gzip.open if str(path).endswith('.gz') else open
        with opener(path, 'rt', encoding='utf-8') as stream:
            return json.load(stream)

    def npz(self, path, keys=None):
        path = Path(path)
        self.files.append(str(path))
        with np.load(path, allow_pickle=False) as stored:
            selected = stored.files if keys is None else keys
            result = {key: stored[key] for key in selected}
        self.array_count += len(result)
        return result


def require(condition, name):
    if not bool(condition):
        raise ValueError(name)


def exact(left, right, name):
    a, b = np.asarray(left), np.asarray(right)
    require(a.shape == b.shape and a.dtype.kind == b.dtype.kind, name+' shape/type')
    require(np.array_equal(a, b, equal_nan=True) if a.dtype.kind in 'fc' else np.array_equal(a, b), name+' exact')


def close(left, right, name, atol=1e-5):
    a, b = np.asarray(left), np.asarray(right)
    require(a.shape == b.shape and np.allclose(a, b, rtol=1e-9, atol=atol, equal_nan=True), name+' numeric binding')


def distance(position, truth):
    """Evaluate saved positions only; no new state is constructed."""
    position, truth = np.asarray(position), np.asarray(truth)
    require(position.shape == truth.shape and position.shape[-1] == 2, 'position error dimensions')
    require(np.isfinite(position).all() and np.isfinite(truth).all(), 'finite evaluated positions')
    lat1, lat2 = np.deg2rad(position[..., 0]), np.deg2rad(truth[..., 0])
    delta = np.deg2rad((position[..., 1]-truth[..., 1]+180.) % 360.-180.)
    h = np.clip(np.sin((lat1-lat2)/2)**2+np.cos(lat1)*np.cos(lat2)*np.sin(delta/2)**2, 0., 1.)
    return 2.*6371000.*np.arctan2(np.sqrt(h), np.sqrt(1.-h))


def statistics(values):
    x = np.asarray(values, float)
    require(x.ndim == 1 and np.isfinite(x).all(), 'finite one-dimensional statistics')
    n = len(x)
    return dict(count=n, sum=float(x.sum()), mean=float(x.mean()) if n else None,
        median=float(np.median(x)) if n else None, rmse=float(np.sqrt(np.mean(x*x))) if n else None,
        p95=float(np.quantile(x, .95)) if n else None, p99=float(np.quantile(x, .99)) if n else None,
        min=float(x.min()) if n else None, max=float(x.max()) if n else None)


def delta_statistics(values):
    x = np.asarray(values, float)
    require(x.ndim == 1 and np.isfinite(x).all(), 'finite one-dimensional differences')
    positive, negative = x > 0., x < 0.
    large, small = x > 1000., (x > 0.) & (x <= 1000.)
    return dict(count=len(x), signed_sum=float(x.sum()), mean=float(x.mean()) if len(x) else None,
        positive_count=int(positive.sum()), negative_count=int(negative.sum()), zero_count=int((x == 0.).sum()),
        positive_sum=float(x[positive].sum()), benefit_sum=float(-x[negative].sum()),
        positive_gt1000_count=int(large.sum()), positive_gt1000_sum=float(x[large].sum()),
        positive_0_to_1000_count=int(small.sum()), positive_0_to_1000_sum=float(x[small].sum()),
        min=float(x.min()) if len(x) else None, max=float(x.max()) if len(x) else None)


def preference_quantities(values, known, endpoint_errors):
    values, known, endpoint = np.asarray(values, float), np.asarray(known), np.asarray(endpoint_errors, float)
    m = len(endpoint)
    require(values.shape == (m, 4, 2) and known.shape == (m, 4) and known.dtype == bool, 'four two-source preferences')
    require(endpoint.shape == (m, 2) and np.isfinite(endpoint).all() and (endpoint >= 0).all(), 'endpoint errors')
    require(np.isfinite(values[known]).all(), 'finite known preference')
    choose = values == values.min(axis=-1, keepdims=True)
    denominator = choose.sum(-1, keepdims=True)
    mass = np.divide(choose, denominator, out=np.zeros(values.shape),
                     where=known[..., None] & (denominator > 0))
    winner = np.where((choose.sum(-1) == 1), np.argmin(values, axis=-1), -1).astype(np.int64)
    winner[~known] = -2
    cost = (mass*endpoint[:, None]).sum(-1)
    regret = cost-endpoint.min(-1)[:, None]
    credit = (mass*(endpoint == endpoint.min(-1, keepdims=True))[:, None]).sum(-1)
    for x in (cost, regret, credit):
        x[~known] = np.nan
    require(np.all(regret[known] >= -1e-7), 'endpoint expectation cannot materially beat best endpoint')
    l, o, n, q = winner.T
    require(np.all(known[:, 2:]), 'nearest and Q must be known on two-supported-source population')
    cell = np.full(m, -1, np.int64)
    strict = (l >= 0) & (n >= 0) & (q >= 0)
    masks = (strict & (l == n) & (n == q), strict & (l == n) & (n != q),
             strict & (l != n) & (q == l), strict & (l != n) & (q == n),
             known[:, 0] & ~strict, ~known[:, 0])
    require(np.all(np.sum(masks, axis=0) == 1), 'A-U partition must be mutually exclusive/exhaustive')
    for i, mask in enumerate(masks):
        cell[mask] = i
    subcell = np.full(m, -1, np.int64)
    conflict = (cell == 2) | (cell == 3)
    for i, mask in enumerate(((o >= 0) & (o == l), (o >= 0) & (o == n), o == -1, o == -2)):
        subcell[conflict & mask] = i
    require(np.all(subcell[conflict] >= 0), 'all OLS conflict subclasses required')
    return dict(preference_mass=mass, preference_winner=winner, selection_cost_m=cost,
        selection_regret_m=regret, selection_credit=credit, cell=cell, ols_subcell=subcell)


def layer(data, components, mask):
    use = np.asarray(mask, bool)
    endpoint, error = data['endpoint_errors_m'][use], data['actual_error_m'][use]
    role, pure = data['role_best'][use], data['group_pure'][use]
    out = dict(query_count=int(use.sum()), component_count=len(set(components[use].tolist())),
        role_counts={name: int((role == i).sum()) for i, name in enumerate(ROLES)},
        purity_counts=dict(pure=int(pure.sum()), mixed=int((~pure).sum())), preferences={}, actual={})
    for p, name in enumerate(PREFERENCES):
        known = data['preference_known'][use, p]
        winner = data['preference_winner'][use, p]
        out['preferences'][name] = dict(known_count=int(known.sum()), unknown_count=int((~known).sum()),
            strict_count=int((winner >= 0).sum()), tie_count=int((winner == -1).sum()),
            **{key: statistics(data[key][use, p][known]) for key in
               ('selection_cost_m', 'selection_regret_m', 'selection_credit')})
    for index, name in enumerate(ACTUAL):
        out['actual'][name] = dict(error_m=statistics(error[:, index]),
            vs_equal=delta_statistics(error[:, index]-error[:, 0]),
            vs_best_single=delta_statistics(error[:, index]-data['best_single_error_m'][use]),
            vs_grid=delta_statistics(error[:, index]-data['grid_error_m'][use]))
    gain, bound = data['gain_m'][use], data['gap_bound_m'][use]
    finite = np.isfinite(bound)
    out['oracle'] = dict(gain_m=statistics(gain), gain_gt10_count=int((gain > 10).sum()),
        gain_gt10_sum=float(gain[gain > 10].sum()), gap_bound_m=statistics(bound[finite]),
        uncertified_count=int((~finite).sum()), min_resultant=statistics(data['min_resultant'][use]))
    return out


def primary_values(data):
    regret = data['selection_regret_m']
    return dict(zip(PRIMARY, (data['actual_error_m'][:, 4]-data['actual_error_m'][:, 1],
        regret[:, 0]-regret[:, 2], regret[:, 1]-regret[:, 2], regret[:, 3]-regret[:, 2])))


def primary_summary(data, components):
    conflict = (data['cell'] == 2) | (data['cell'] == 3)
    result = {}
    for key, values in primary_values(data).items():
        use = conflict & np.isfinite(values)
        result[key] = dict(statistics(values[use]), conflict_query_count=int(conflict.sum()),
            eligible_component_count=len(set(components[use].tolist())),
            coverage=float(use.sum()/conflict.sum()) if conflict.any() else None)
    return result


def analysis_block(items):
    require(bool(items), 'nonempty selected scene list')
    data = {key: np.concatenate([item['data'][key] for item in items], axis=0) for key in QUERY_KEYS}
    components = np.concatenate([np.repeat(item['component_id'], len(item['data']['cell'])) for item in items])
    allmask = np.ones(len(components), bool)
    cell, winner = data['cell'], data['preference_winner']
    scope = {key: sum(item['scope'][key] for item in items) for key in items[0]['scope']}
    full = np.concatenate([item['data']['full_actual_error_m'] for item in items])
    result = dict(query_count=len(cell), component_count=len({item['component_id'] for item in items}), scope=scope,
        full_population=dict(point_count=len(full), methods={name: statistics(full[:, i]) for i, name in enumerate(METHODS)}),
        overall=layer(data, components, allmask),
        main_cells={name: layer(data, components, cell == i) for i, name in enumerate(CELLS)},
        ols_subcells={name+'_'+suffix: layer(data, components, (cell == i) & (data['ols_subcell'] == j))
                     for i, name in ((2, 'C'), (3, 'D')) for j, suffix in enumerate(OLS_SUBCELLS)},
        role_cells={role: {name: layer(data, components, (data['role_best'] == j) & (cell == i))
                          for i, name in enumerate(CELLS)} for j, role in enumerate(ROLES)},
        purity_cells={pure: {name: layer(data, components, (data['group_pure'] == value) & (cell == i))
                            for i, name in enumerate(CELLS)} for pure, value in (('pure', True), ('mixed', False))},
        pairwise_preferences={}, primary_estimands=primary_summary(data, components), scene_macro_estimands={})
    for a, b in combinations(range(4), 2):
        x, y = winner[:, a], winner[:, b]
        unknown = (x == -2) | (y == -2)
        tied = ~unknown & ((x == -1) | (y == -1))
        same = (x >= 0) & (y >= 0) & (x == y)
        opposite = (x >= 0) & (y >= 0) & (x != y)
        result['pairwise_preferences'][PREFERENCES[a]+'_vs_'+PREFERENCES[b]] = {
            name: layer(data, components, mask) for name, mask in
            (('strict_same', same), ('strict_opposite', opposite), ('at_least_one_tie', tied), ('any_unknown', unknown))}
    for key in PRIMARY:
        means = [primary_summary(item['data'], np.repeat(item['component_id'], len(item['data']['cell'])))[key]['mean']
                 for item in items]
        means = [value for value in means if value is not None]
        result['scene_macro_estimands'][key] = dict(mean=float(np.mean(means)) if means else None,
                                                   defined_scene_count=len(means))
    return result


def bootstrap(items):
    components = sorted({item['component_id'] for item in items})
    draws = np.random.default_rng(20).integers(0, len(components), size=(2000, len(components)))
    result = {}
    for key in PRIMARY:
        sums, counts = [], []
        for component in components:
            chosen = []
            for item in items:
                if item['component_id'] != component:
                    continue
                data = item['data']
                values = primary_values(data)[key]
                use = ((data['cell'] == 2) | (data['cell'] == 3)) & np.isfinite(values)
                chosen.append(values[use])
            values = np.concatenate(chosen)
            sums.append(float(values.sum()))
            counts.append(len(values))
        sums, counts = np.asarray(sums), np.asarray(counts)
        denominator = counts[draws].sum(1)
        valid = denominator > 0
        estimates = sums[draws].sum(1)[valid]/denominator[valid]
        contributing = int((counts > 0).sum())
        result[key] = dict(comparison=key, components=components, component_count=len(components),
            contributing_component_count=contributing, resamples=2000, valid_resamples=int(valid.sum()),
            empty_resamples=int((~valid).sum()), seed=20, independent_test=False,
            point_estimate_m=float(sums.sum()/counts.sum()) if counts.sum() else None,
            percentile95_m=np.quantile(estimates, [.025, .975]).tolist() if contributing >= 2 and len(estimates) else None)
    return result


def metadata(paths, reader, manifest_path):
    states = {name: reader.json(path/'run.json') for name, path in paths.items() if name != 'output'}
    expected = {'quality_run': 'IDEA-OBSERVATION-QUALITY-20261030',
                'source_run': 'IDEA-SOURCE-RISK-20261030',
                'mixture_run': 'IDEA-MIXTURE-OPPORTUNITY-20261030',
                'native_run': 'IDEA-NATIVE-CV-STATE-CONTROL-20261029'}
    for name, run in states.items():
        require(run.get('status') == 'COMPLETED' and run.get('exit_code') == 0
                and run.get('idea_id') == expected[name], 'completed fixed input required: '+name)
    quality, source, mixture = (states[key] for key in ('quality_run', 'source_run', 'mixture_run'))
    for run, field, expected_path in ((quality, 'source_risk_run', paths['source_run']),
        (quality, 'native_run', paths['native_run']), (source, 'native_run', paths['native_run']),
        (mixture, 'source_run', paths['source_run']), (mixture, 'native_run', paths['native_run'])):
        require(Path(run[field]).resolve() == expected_path, 'input run lineage '+field)
    require(Path(quality['manifest']).resolve() == manifest_path, 'original136 absolute manifest required')
    for field in ('value_run', 'partition_run'):
        require(Path(quality[field]).resolve() == Path(source[field]).resolve(), '133/136 '+field+' lineage')
    require(quality['completed_training_scene_count'] == source['completed_training_scene_count'] == 32
            and mixture['completed_scene_count'] == 32 and source['analytic_head_fits'] == 39,
            'original T completed counts required')
    require(tuple(quality['T_methods']) == METHODS and mixture['grid_size'] == 4097
            and mixture['reference_oracle_only'] is True, 'original methods and reference-only grid contract')
    manifest = reader.json(manifest_path)
    tids = source['training_scene_ids']
    marker = reader.json(paths['native_run']/'native-cache.json')
    require(tids == quality['training_scene_ids'] == mixture['scene_ids'] == manifest['splits']['T_core'][:32]
            == marker['scene_ids']['T_core'] and len(tids) == len(set(tids)) == 32,
            'original32 T scenes and order required')
    require(marker['status'] == 'CACHE_COMPLETED'
            and len({manifest['components'][sid] for sid in tids}) == 12, 'completed native T12 component cache')
    folds = reader.json(paths['source_run']/'T-fold-index.json')
    fold_by_component = {}
    for fold in folds:
        held = fold['heldout_component']
        if held is None:
            continue
        require(held not in fold_by_component and held not in fold['training_components'], 'OOF component leakage/duplicate')
        selected = [sid for sid in tids if str(manifest['components'][sid]) != held]
        require(fold['training_scene_ids'] == selected
                and sorted(fold['training_components']) == sorted({str(manifest['components'][sid]) for sid in selected}),
                'original Q OOF training population')
        fold_by_component[held] = fold
    require(set(fold_by_component) == {str(manifest['components'][sid]) for sid in tids}, 'all12 original OOF folds required')
    roles = reader.npz(paths['mixture_run']/'private/roles.npz')
    names, indices = roles['source_names'].tolist(), roles['indices'].tolist()
    require(len(names) == len(set(names)) == 2 and set(indices) == {0, 1}, 'original two anonymous source roles')
    role_map = dict(zip(names, indices))
    qrows = reader.json(paths['quality_run']/'training-scene-summary.json.gz')
    mrows = reader.json(paths['mixture_run']/'scene-summary.json.gz')
    msummary = reader.json(paths['mixture_run']/'summary.json')  # 134 is T-only.
    require([row['scene_id'] for row in qrows] == [row['scene_id'] for row in mrows] == tids,
            'complete prior T public records')
    return tids, manifest, role_map, {r['scene_id']: r for r in qrows}, {r['scene_id']: r for r in mrows}, msummary


def structural_join(pack, policy, motion):
    n, nq = motion['support'].shape
    require(nq == 64 and motion['support'].dtype == bool, 'original Q64 physical support')
    for key in ('node_ids', 'query_seconds'):
        exact(pack[key], motion[key], '133/127 '+key)
        exact(policy[key], motion[key], '136/127 '+key)
    exact(pack['membership'], policy['membership'], '133/136 node membership')
    exact(policy['support'], motion['support'], '136/127 support')
    membership = pack['membership']
    require(membership.shape == (n,) and membership.dtype.kind in 'iu', 'complete membership')
    unique = np.unique(membership)
    exact(unique, np.arange(len(unique)), 'consecutive groups')
    groups = [np.flatnonzero(membership == gid) for gid in unique]
    offsets, mapping = pack['group_offsets'], pack['node_to_source_row']
    exact(policy['group_offsets'], offsets, '133/136 group source offsets')
    require(mapping.shape == (n,) and offsets.shape == (len(groups)+1,) and offsets[0] == 0
            and np.all(np.diff(offsets) > 0) and (mapping >= 0).all() and (mapping < offsets[-1]).all(), 'source structure')
    source_count = int(offsets[-1])
    qmap = policy['node_to_source_row']
    require(qmap.shape == (n,) and qmap.dtype.kind in 'iu', '136 source mapping')
    quality_members = {}
    for row in range(len(policy['available'])):
        members = tuple(np.flatnonzero(qmap == row))
        require(members and members not in quality_members, 'unique nonempty136 source members')
        quality_members[members] = row
    remap = np.empty(source_count, np.int64)
    source_nodes = []
    actual_counts = np.zeros((source_count, nq), np.int64)
    for row in range(source_count):
        members = np.flatnonzero(mapping == row)
        key = tuple(members)
        require(key in quality_members and len(set(motion['sources'][members].tolist())) == 1,
                'unique source-member identity join')
        source_nodes.append(members)
        remap[row] = quality_members[key]
        actual_counts[row] = motion['support'][members].sum(0)
    require(len(set(remap.tolist())) == len(policy['available']) == source_count, 'source rows must be a complete bijection')
    for gid, members in enumerate(groups):
        rows = mapping[members]
        require(((rows >= offsets[gid]) & (rows < offsets[gid+1])).all(), 'source row crosses supplied group')
        require(len({str(motion['sources'][source_nodes[row][0]]) for row in range(offsets[gid], offsets[gid+1])})
                == offsets[gid+1]-offsets[gid], 'one equivalence class per group source')
    exact(actual_counts, pack['supported_counts'], '133 actual source counts')
    exact(actual_counts, policy['supported_counts'][remap], '136 joined source counts')
    exact(pack['available'], actual_counts > 0, '133 actual available')
    exact(pack['available'], policy['available'][remap], '136 joined available')
    exact(pack['features'], policy['features'][remap], 'source physical feature binding')
    exact(pack['source_states'], policy['source_states'][remap], 'saved source state binding')
    return groups, remap, source_nodes


def scope_and_keys(pack, motion, evaluation, groups):
    registry = {str(v): i for i, v in enumerate(evaluation['target_ids'])}
    require(len(registry) == len(evaluation['target_ids']), 'unique evaluation targets')
    nq = len(motion['query_seconds'])
    scope = dict(total_group_queries=len(groups)*nq, source_support_zero=0, source_support_one=0, source_support_multi=0,
        complete_target_reference_valid_query_count=int(evaluation['target_valid'][..., :2].all(-1).sum()),
        labeled_group_query_count=0, duplicate_anchor_group_count=0, unknown_anchor_group_count=0,
        first_anchor_target_count=0, target_without_anchor_count=0, first_anchor_reference_invalid_query_count=0,
        first_anchor_reference_valid_zero_source_count=0, first_anchor_reference_valid_one_source_count=0,
        first_anchor_reference_valid_multi_source_count=0, mixed_identity_group_count=0,
        duplicate_group_queries=0, unknown_group_queries=0, reference_valid_points_without_anchor=0)
    claimed, expected, gi, qi, sr, pure, target = {}, np.zeros_like(pack['labeled']), [], [], [], [], []
    for gid, members in enumerate(groups):
        begin, end = pack['group_offsets'][gid:gid+2]
        available = pack['available'][begin:end]
        counts = available.sum(0)
        require(np.all(counts <= 2), 'actual K>2 outside fixed138 K2 diagnostic contract')
        for name, mask in (('zero', counts == 0), ('one', counts == 1), ('multi', counts == 2)):
            scope['source_support_'+name] += int(mask.sum())
        identity = str(evaluation['identities'][members[0]])
        mixed = len(set(evaluation['identities'][members].tolist())) > 1
        scope['mixed_identity_group_count'] += int(mixed)
        if identity not in registry:
            scope['unknown_anchor_group_count'] += 1
            scope['unknown_group_queries'] += nq
        elif identity in claimed:
            scope['duplicate_anchor_group_count'] += 1
            scope['duplicate_group_queries'] += nq
        else:
            claimed[identity] = gid
            scope['first_anchor_target_count'] += 1
            valid = evaluation['target_valid'][registry[identity], :, :2].all(-1)
            scope['first_anchor_reference_invalid_query_count'] += int((~valid).sum())
            for name, mask in (('zero', counts == 0), ('one', counts == 1), ('multi', counts == 2)):
                scope['first_anchor_reference_valid_'+name+'_source_count'] += int((valid & mask).sum())
            expected[begin:end] = available & (valid & (counts == 2))
            for query in np.flatnonzero(valid & (counts == 2)):
                gi.append(gid)
                qi.append(query)
                sr.append(np.flatnonzero(available[:, query])+begin)
                pure.append(not mixed)
                target.append(registry[identity])
    exact(pack['labeled'], expected, 'first-anchor unique reference-valid D2 labels')
    for gid in range(len(groups)):
        a, b = pack['group_offsets'][gid:gid+2]
        weights = expected[a:b]/np.maximum(pack['available'][a:b].sum(0), 1)
        exact(pack['sample_weight'][a:b], weights, 'original query label mass')
    scope['labeled_group_query_count'] = len(gi)
    scope['target_without_anchor_count'] = len(registry)-len(claimed)
    scope['reference_valid_points_without_anchor'] = int(sum(evaluation['target_valid'][index, :, :2].all(-1).sum()
        for identity, index in registry.items() if identity not in claimed))
    scope['full_evaluable_position_points'] = (scope['complete_target_reference_valid_query_count']
                                              -scope['reference_valid_points_without_anchor'])
    return scope, np.asarray(gi, np.int64), np.asarray(qi, np.int64), np.asarray(sr, np.int64).reshape(-1, 2), \
        np.asarray(pure, bool), np.asarray(target, np.int64), claimed


def saved_artifacts(raw):
    prefix = METHODS[0]+'_'
    fields = [key[len(prefix):] for key in raw if key.startswith(prefix)]
    require('aligned_prediction' in fields and 'group_states' in fields, 'complete136 T artifacts')
    return {name: {field: raw[name+'_'+field] for field in fields} for name in METHODS}


def scene(paths, sid, component, roles, reader, prior_quality, prior_mixture):
    pack = reader.npz(paths['source_run']/'private'/f'training-{sid}.npz')
    policy = reader.npz(paths['quality_run']/'private'/f'policy-T_core-{sid}.npz')
    diagnostic = reader.npz(paths['quality_run']/'private'/f'diagnostic-T_core-{sid}.npz')
    raw_artifact = reader.npz(paths['quality_run']/'private'/f'states-T_core-{sid}.npz')
    mixture = reader.npz(paths['mixture_run']/'private'/f'scene-{sid}.npz')
    motion = reader.npz(paths['native_run']/'private/native-motion/T_core'/f'scene-{sid}.npz',
        ('node_ids', 'query_seconds', 'sources', 'support', 'nearest_seconds'))
    ev = reader.npz(paths['native_run']/'private/native-evaluation/T_core'/f'scene-{sid}.npz')
    oof = reader.npz(paths['source_run']/'private'/f'oof-{sid}.npz', ('QUADRATIC__scores',))['QUADRATIC__scores']
    require(str(pack['component_id']) == component == str(prior_quality['component_id'])
            == str(prior_mixture['component_id']), 'scene component binding')
    groups, remap, source_nodes = structural_join(pack, policy, motion)
    for key in ('node_ids', 'query_seconds'):
        exact(mixture[key], motion[key], '134/127 '+key)
        exact(ev[key], motion[key], 'evaluation/physical '+key)
    exact(mixture['membership'], pack['membership'], '134 membership')
    for key in ('labeled', 'sample_weight'):
        exact(pack[key], diagnostic[key][remap], '136 joined '+key)
    close(pack['errors_m'], diagnostic['errors_m'][remap], '136 joined label error')
    scope, gi, qi, sr, pure, target, claimed = scope_and_keys(pack, motion, ev, groups)
    for key, value in (('group_indices', gi), ('query_indices', qi), ('source_rows', sr), ('group_pure', pure)):
        exact(mixture[key], value, '134 complete D2 '+key)
    for key, value in prior_mixture['scope'].items():
        require(scope[key] == value, '134 original scope '+key)
    require(len({(int(g), int(q)) for g, q in zip(gi, qi)}) == len(gi), 'unique D2 join keys')
    nq, m = len(motion['query_seconds']), len(gi)
    qsr = remap[sr]
    truth = ev['target_states'][target, qi, :2]
    exact(mixture['truth'], truth, '134 original D2 reference')
    endpoint = mixture['endpoint_errors_m']
    close(endpoint, pack['errors_m'][sr, qi[:, None]], '133/134 endpoints')
    close(endpoint, diagnostic['errors_m'][qsr, qi[:, None]], '136/134 endpoints')
    close(mixture['best_single_error_m'], endpoint.min(1), 'saved best endpoint')
    close(mixture['gain_m'], mixture['best_single_error_m']-mixture['grid_error_m'], 'saved grid gain')
    require(np.shape(oof) == np.shape(pack['available']) and np.isfinite(oof).all(), 'complete finite original QOOF')
    qscores = oof[sr, qi[:, None]]
    exact(mixture['oof_QUADRATIC_scores'], qscores, 'original QOOF subset')
    nearest = np.full(pack['available'].shape, np.inf)
    for row, members in enumerate(source_nodes):
        nearest[row] = np.min(np.where(motion['support'][members], motion['nearest_seconds'][members], np.inf), axis=0)
    values = np.stack((policy['loo__source_proxy_m2'][qsr, qi[:, None]],
        policy['ols__source_proxy_m2'][qsr, qi[:, None]], nearest[sr, qi[:, None]], qscores), axis=1)
    known = np.column_stack((policy['loo__source_known'][qsr, qi[:, None]].all(1),
        policy['ols__source_known'][qsr, qi[:, None]].all(1), np.ones(m, bool), np.ones(m, bool)))
    preferences = preference_quantities(values, known, endpoint)
    exact(policy['WS_SOURCE_NEAREST__source_weights'][qsr, qi[:, None]],
          preferences['preference_mass'][:, 2], 'source-min nearest preference binding')
    close(preferences['selection_regret_m'][:, 3], mixture['QUADRATIC_selection_regret_m'], 'old Q selection regret')
    actual_roles = np.asarray([[roles[str(motion['sources'][source_nodes[row][0]])] for row in rows] for rows in sr], np.int64).reshape(-1, 2)
    exact(mixture['role_indices'], actual_roles, 'original anonymous role mapping')
    require(np.all(np.sort(actual_roles, axis=1) == [0, 1]), 'each D2 contains original two different roles')
    best = endpoint == endpoint.min(1, keepdims=True)
    role_best = np.where(best.sum(1) == 2, 2, actual_roles[np.arange(m), np.argmin(endpoint, axis=1)])
    label_best = pack['errors_m'][sr, qi[:, None]]
    label_best = label_best == label_best.min(1, keepdims=True)
    scope['endpoint_label_preference_roundoff_disagreements'] = int(np.any(best != label_best, axis=1).sum())
    artifacts = saved_artifacts(raw_artifact)
    base = artifacts[METHODS[0]]
    t = len(ev['target_ids'])
    exact(base['target_ids'], ev['target_ids'], '136 target registry')
    exact(base['aligned_truth'][:t], ev['target_states'], '136 full reference')
    exact(base['aligned_truth_valid'][:t], ev['target_valid'], '136 full reference validity')
    exact(base['node_group_indices'], pack['membership'], '136 full node grouping')
    exact(base['group_node_indices'], np.concatenate(groups), '136 full original group members')
    exact(base['group_offsets'], np.r_[0, np.cumsum([len(g) for g in groups])], '136 full original group offsets')
    expected_anchor = np.asarray([{str(v): i for i, v in enumerate(ev['target_ids'])}.get(str(ev['identities'][g[0]]), -1) for g in groups])
    exact(base['group_anchor_target_indices'], expected_anchor, '136 original group anchors')
    first_groups = np.asarray([claimed.get(str(identity), -1) for identity in ev['target_ids']], np.int64)
    exact(base['aligned_group_indices'][:t], first_groups, '136 full first-anchor target alignment')
    exact(base['aligned_target_ids'][:t], ev['target_ids'], '136 full first-target registry order')
    duplicate = scope['duplicate_anchor_group_count']
    require(len(base['aligned_group_indices']) == t+duplicate and scope['unknown_anchor_group_count'] == 0,
            '136 original duplicate rows retained; original evaluator requires known anchors')
    expected_population = dict(target_identity_count=t, predicted_group_count=len(groups),
        duplicate_prediction_group_count=duplicate, anchor_matched_target_count=len(claimed),
        reference_unavailable_target_count=int((~ev['target_valid'].any(axis=(1, 2))).sum()))
    for name in METHODS:
        prior_method = prior_quality['methods'][name]
        for key, value in expected_population.items():
            require(prior_method['population'][key] == value, '136 original population '+name+'/'+key)
        require(prior_method['unmatched_target_count'] == t-len(claimed), '136 original unmatched population')
    require(str(base['split']) == 'T_core', 'T-only saved artifact')
    full_use = ev['target_valid'][..., :2].all(-1) & base['aligned_prediction_valid'][:t, :, :2].all(-1)
    full_t, full_q = np.nonzero(full_use)
    full_g = base['aligned_group_indices'][full_t]
    require(len(full_t) == scope['full_evaluable_position_points'], 'full136 point denominator')
    full_lookup = np.full((t, nq), -1, np.int64)
    full_lookup[full_t, full_q] = np.arange(len(full_t))
    d2_full = full_lookup[target, qi]
    require((d2_full >= 0).all(), 'all D2 rows occur in complete136 population')
    full_source_count = np.asarray([pack['available'][a:b].sum(0) for a, b in
                                   zip(pack['group_offsets'][:-1], pack['group_offsets'][1:])])[full_g, full_q]
    exact(full_source_count[d2_full], np.full(m, 2), 'D2 full-population source counts')
    full_error = np.empty((len(full_t), 6))
    for i, name in enumerate(METHODS):
        art = artifacts[name]
        for key in base:
            if key not in ('aligned_prediction', 'group_states'):
                exact(art[key], base[key], '136 fixed population '+name+'/'+key)
        exact(art['group_states'], policy[name+'__group_states'], '136 frozen policy/artifact '+name)
        exact(art['group_states'][..., 2:], base['group_states'][..., 2:], '136 speed heading '+name)
        exact(art['aligned_prediction'][full_t, full_q], art['group_states'][full_g, full_q], '136 actual aligned output '+name)
        full_error[:, i] = distance(art['group_states'][full_g, full_q, :2], ev['target_states'][full_t, full_q, :2])
        observed = statistics(full_error[:, i])
        prior = prior_quality['methods'][name]['position']['all']
        require(len(full_t) == prior['intersection_count'], '136 complete method point count')
        for field, stat in (('mae', 'mean'), ('rmse', 'rmse'), ('p95', 'p95'), ('p99', 'p99')):
            close(observed[stat], prior[field], '136 original full '+name+'/'+field)
    for a, b in ((2, 3), (4, 5)):
        exact(policy[METHODS[a]+'__source_weights'], policy[METHODS[b]+'__source_weights'], '136 original floor pair mass')
        exact(artifacts[METHODS[a]]['group_states'], artifacts[METHODS[b]]['group_states'], '136 original floor pair output')
    close(distance(mixture['QUADRATIC_hard_position'], truth), mixture['QUADRATIC_hard_error_m'], '134 actual Qhard error')
    qstrict = preferences['preference_winner'][:, 3] >= 0
    close(mixture['QUADRATIC_hard_error_m'][qstrict], preferences['selection_cost_m'][qstrict, 3], 'strict Q endpoint/actual distinction')
    actual_error = np.column_stack((full_error[d2_full], mixture['QUADRATIC_hard_error_m']))
    data = {key: mixture[key].copy() for key in ('endpoint_errors_m', 'best_single_error_m', 'grid_error_m',
        'gain_m', 'gap_bound_m', 'min_resultant', 'role_indices', 'group_pure')}
    data.update(node_ids=motion['node_ids'], query_seconds=motion['query_seconds'], membership=pack['membership'],
        component_id=np.asarray(component), group_indices=gi, query_indices=qi, source_rows=sr,
        quality_source_rows=qsr, source_row_133_to_136=remap, preference_names=np.asarray(PREFERENCES),
        preference_values=values, preference_known=known, actual_method_names=np.asarray(ACTUAL),
        actual_error_m=actual_error, role_best=role_best, full_actual_error_m=full_error,
        full_target_indices=full_t, full_query_indices=full_q, full_group_indices=full_g,
        full_source_count=full_source_count, D2_full_indices=d2_full, **preferences)
    return dict(scene_id=sid, component_id=component, scope=scope, data=data), dict(
        scene_id=sid, source_rows=len(remap), source_row_reordering_used=not np.array_equal(remap, np.arange(len(remap))),
        D2_queries=m, labeled_source_queries=int(pack['labeled'].sum()), full_evaluable_position_points=len(full_t),
        exact_floor_pairs=2, original_QOOF_reused=True)


def run(args):
    keys = ('quality_run', 'source_run', 'mixture_run', 'native_run', 'output')
    paths = {key: Path(getattr(args, key)).resolve() for key in keys}
    output, manifest_path = paths['output'], Path(args.manifest).resolve()
    for key, path in paths.items():
        if key != 'output':
            require(output != path and output not in path.parents and path not in output.parents,
                    'new output overlaps frozen input')
    require(output != manifest_path and output not in manifest_path.parents, 'output contains fixed manifest')
    require(1 <= args.cpu_threads <= 4 and getattr(args, 'time_policy', 'advisory') == 'advisory',
            'CPU1..4 and advisory time policy required')
    output.mkdir(parents=True, exist_ok=False)
    (output/'private').mkdir()
    (output/'scenes').mkdir()
    started, reader = time.monotonic(), Reader()
    state = dict(status='RUNNING', idea_id=IDEA, plan_version='1.0',
        **{key: str(value) for key, value in paths.items() if key != 'output'}, manifest=str(manifest_path),
        command=sys.argv, started_at=datetime.now(timezone.utc).isoformat(), completed_scene_count=0,
        new_fits=0, new_physical_predictions=0, model_forwards=0, neural_updates=0, teacher_forwards=0,
        new_grid_evaluations=0, native_events_opened=False, V_arrays_opened=False, test_opened=False,
        archive_read=False, source_id_feature=False, original_QOOF_reused=True, cpu_threads=args.cpu_threads,
        gpu_used=False, time_policy='advisory', advisory_seconds=1800, costs={})
    state['completion_count_semantics'] = 'completed_scene_count counts only fully written scenes; work in a failed current scene is not included'

    def check(**extra):
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if sys.platform != 'darwin':
            rss *= 1024
        state.update(extra, elapsed_seconds=time.monotonic()-started, peak_rss_bytes=int(rss),
                     input_files_opened=len(reader.files), input_arrays_loaded=reader.array_count)
        state['advisory_exceeded'] = state['elapsed_seconds'] > 1800
        write(output/'run.json', state)
        if state['advisory_exceeded'] and not (output/'time_advisory.json').exists():
            write(output/'time_advisory.json', dict(elapsed_seconds=state['elapsed_seconds'],
                action='continue only fixed T32 cache diagnosis; no new prediction, fitting, grid or retry'))
        if rss > RAM:
            raise MemoryError('6GiB resident budget exceeded')

    try:
        check(phase='binding')
        if sys.platform.startswith('linux'):
            soft, hard = resource.getrlimit(resource.RLIMIT_AS)
            limit = min([ADDRESS_SPACE]+[v for v in (soft, hard) if v != resource.RLIM_INFINITY])
            resource.setrlimit(resource.RLIMIT_AS, (limit, hard))
        tids, manifest, roles, prior_quality, prior_mixture, mixture_summary = metadata(paths, reader, manifest_path)
        check(scene_ids=tids, training_scene_ids=tids)
        items, rows, alignments = [], [], []
        for sid in tids:
            check(phase='T_saved_array_join', last_scene_id=sid)
            item, alignment = scene(paths, sid, str(manifest['components'][sid]), roles, reader,
                                    prior_quality[sid], prior_mixture[sid])
            np.savez_compressed(output/'private'/f'aligned-{sid}.npz', **item['data'])
            row = dict(scene_id=sid, component_id=item['component_id'], **analysis_block([item]))
            write(output/'scenes'/f'scene-{sid}.json.gz', row)
            items.append(item)
            rows.append(row)
            alignments.append(alignment)
            check(completed_scene_count=len(rows))
        state['costs']['join_and_scene_statistics_seconds'] = time.monotonic()-started
        check(phase='aggregate_fixed_diagnostics')
        require(len(items) == 32 and sum(len(item['data']['cell']) for item in items)
                == mixture_summary['all_components']['query_count'], 'complete original134 joint denominator')
        for key, value in mixture_summary['scope'].items():
            require(sum(item['scope'][key] for item in items) == value, 'original134 total scope '+key)
        components = sorted({item['component_id'] for item in items})
        largest = max(components, key=lambda c: sum(item['component_id'] == c for item in items))
        alignment = dict(status='COMPLETED', scene_count=len(items), scenes=alignments,
            original134_query_count=sum(a['D2_queries'] for a in alignments),
            original136_full_point_count=sum(a['full_evaluable_position_points'] for a in alignments),
            original_QOOF_reused=True, input_file_count=len(reader.files), input_array_count=reader.array_count,
            bindings=['run lineage and original136 manifest', 'T32 order and T12 original OOF isolation',
                'node membership and per-source member-set bijection', 'query times and available-source masks',
                'first-anchor reference-valid D2 and complete target/duplicate population',
                'original134 endpoint/grid/Qhard scalar arrays', 'original136 six outputs and full-point metrics'])
        write(output/'alignment.json', alignment)
        summary = dict(status='COMPLETED', idea_id=IDEA, plan_version='1.0',
            all_components=analysis_block(items),
            by_component={component: analysis_block([item for item in items if item['component_id'] == component])
                          for component in components},
            without_largest_component=analysis_block([item for item in items if item['component_id'] != largest]),
            largest_component_id=largest, primary_bootstrap=bootstrap(items), alignment=alignment,
            component_scene_counts={c: sum(item['component_id'] == c for item in items) for c in components},
            preference_names=list(PREFERENCES), actual_method_names=list(ACTUAL),
            units=dict(LOO='m2', OLS='m2', NEAREST='seconds', QOOF='saved log-error score',
                selection_cost_m='m', selection_regret_m='m', selection_credit='fraction', actual_error_m='m'),
            limitations=['T-only joint conditional diagnosis; no new method prediction or independent test',
                'Original133 OOF isolates only its heads; no new LOCO or new model fitting',
                'Endpoint expectation is not geometric mixture error; signed actual-minus-grid may be negative',
                'C/D and OLS-subcell regret equalities follow from construction, not mechanism evidence',
                'Anonymous roles, target reference and group purity are diagnostic only',
                'Mixed-group endpoint errors refer to first-anchor reference, not physical source health',
                'Actual K2, reused development T and upstream supervised grouping limit transfer conclusions'])
        write(output/'scene-summary.json.gz', rows)
        check(phase='finalizing')
        summary.update(elapsed_seconds=state['elapsed_seconds'], peak_rss_bytes=state['peak_rss_bytes'], costs=state['costs'])
        write(output/'summary.json', summary)
        check(status='COMPLETED', exit_code=0, phase='completed', finished_at=datetime.now(timezone.utc).isoformat())
        return summary
    except BaseException as error:
        state.update(status='INTERRUPTED' if isinstance(error, (InterruptedError, KeyboardInterrupt)) else 'FAILED',
            exit_code=1, error=f'{type(error).__name__}: {error}', elapsed_seconds=time.monotonic()-started,
            finished_at=datetime.now(timezone.utc).isoformat())
        write(output/'run.json', state)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('quality-run', 'source-run', 'mixture-run', 'native-run', 'manifest', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--cpu-threads', type=int, default=4)
    parser.add_argument('--time-policy', choices=('advisory',), default='advisory')
    args = parser.parse_args()
    def interrupted(signum, frame):
        raise InterruptedError(f'signal{signum}')
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        run(args)
    except BaseException:
        traceback.print_exc()
        return 1
    print(f'COMPLETED: {args.output}/summary.json', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
