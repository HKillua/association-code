"""Independent NumPy audit of134 T-only saved mixture-opportunity diagnostics.

No production feature, recovery, oracle, model or evaluator is imported.
The4097-point search is an audit recomputation, not a new scientific run.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import resource
import signal
import sys
import time
import traceback

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.analysis import audit_shared_control_arrays as core
from tools.analysis.audit_source_risk_arrays import close, distances, load
import numpy as np

IDEA = 'IDEA-MIXTURE-OPPORTUNITY-20261030'
HEADS = ('LINEAR', 'QUADRATIC', 'SHUFFLED')
GRID = 4097
RAM, ADDRESS_SPACE = 6*1024**3, 12*1024**3


def position(moments, alpha, anchors):
    """Use raw-node sine/cosine moments, never means of source angles."""
    moments = np.asarray(moments, np.float64)
    alpha = np.broadcast_to(np.asarray(alpha, np.float64), (len(moments),))
    mixed = moments[:, 1]+alpha[:, None]*(moments[:, 0]-moments[:, 1])
    longitude = np.degrees(np.arctan2(mixed[:, 1], mixed[:, 2]))
    longitude = (longitude+180.) % 360.-180.
    longitude = np.where(np.hypot(mixed[:, 1], mixed[:, 2]) > 1e-12, longitude, anchors[:, 1])
    return np.column_stack((mixed[:, 0], longitude))


def independent_grid(moments, anchors, truth, checks):
    """Search each grid point in chunks; verify attainable gain and a global bound."""
    count = len(moments)
    endpoint = np.column_stack([distances(position(moments, a, anchors), truth) for a in (1., 0.)])
    best = np.full(count, np.inf)
    best_index = np.zeros(count, np.int64)
    # Chunk the alpha axis, unlike the production query-axis blocking.
    for begin in range(0, GRID, 64):
        indices = np.arange(begin, min(begin+64, GRID))
        alpha = indices/(GRID-1)
        mixed = moments[:, None, 1]+alpha[None, :, None]*(moments[:, None, 0]-moments[:, None, 1])
        lon = np.degrees(np.arctan2(mixed[..., 1], mixed[..., 2]))
        lon = (lon+180.) % 360.-180.
        lon = np.where(np.hypot(mixed[..., 1], mixed[..., 2]) > 1e-12, lon, anchors[:, None, 1])
        d = distances(np.stack((mixed[..., 0], lon), axis=-1), truth[:, None, :])
        for column, index in enumerate(indices):
            better = d[:, column] < best
            tied = d[:, column] == best
            nearer_center = abs(index-(GRID-1)//2) < np.abs(best_index-(GRID-1)//2)
            replace = better | (tied & nearer_center)
            best[replace] = d[replace, column]
            best_index[replace] = index
    # The minimum of a squared norm on an interval is a clamped quadratic vertex.
    start = moments[:, 1, 1:]
    difference = moments[:, 0, 1:]-start
    squared = np.sum(difference*difference, axis=1)
    vertex = np.zeros(count)
    use = squared > 0.
    vertex[use] = -np.sum(start[use]*difference[use], axis=1)/squared[use]
    vertex = np.minimum(1., np.maximum(0., vertex))
    resultant = np.linalg.norm(start+vertex[:, None]*difference, axis=1)
    regular = resultant > 1e-12
    cross = start[:, 0]*difference[:, 1]-start[:, 1]*difference[:, 0]
    dlat = np.radians(moments[:, 0, 0]-moments[:, 1, 0])
    gap = np.full(count, np.nan)
    gap[regular] = core.RADIUS*np.hypot(dlat[regular], cross[regular]/np.square(resultant[regular]))/(2*(GRID-1))+1e-6
    checks.exact(np.isfinite(best).all() and np.all(best <= endpoint.min(1)+1e-5), 'grid includes both endpoints')
    return dict(endpoint_errors_m=endpoint, best_single_error_m=endpoint.min(1), grid_error_m=best,
                alpha=best_index/(GRID-1), gain_m=endpoint.min(1)-best, gap_bound_m=gap,
                min_resultant=resultant, position=position(moments, best_index/(GRID-1), anchors))


def compare_oracle(saved, actual, checks):
    for key in ('endpoint_errors_m', 'best_single_error_m', 'grid_error_m', 'gain_m', 'min_resultant'):
        close(saved[key], actual[key], checks, 'oracle/'+key, 1e-5 if key != 'min_resultant' else 1e-12)
    defined = np.isfinite(actual['gap_bound_m'])
    checks.array(np.isfinite(saved['gap_bound_m']), defined, 'continuous gap bound availability')
    close(saved['gap_bound_m'][defined], actual['gap_bound_m'][defined], checks, 'oracle/gap_bound_m', 1e-5)
    alpha = saved['alpha']
    checks.exact(np.isfinite(alpha).all() and np.all((alpha >= 0.) & (alpha <= 1.)), 'saved alpha domain')
    checks.array(alpha*(GRID-1), np.rint(alpha*(GRID-1)), 'saved alpha belongs to fixed4097 grid')
    # Independent spherical arithmetic can perturb exact floating ties. The
    # stored grid point must attain the independently best error, without
    # claiming its arbitrary tie-breaking alpha is scientifically unique.
    expected_position = position(saved['moments'], alpha, saved['anchors'])
    close(distances(saved['position'], expected_position), np.zeros(len(alpha)), checks, 'oracle/chosen_position', 1e-5)
    close(distances(expected_position, saved['truth']), actual['grid_error_m'], checks, 'oracle/chosen_grid_error', 1e-5)


def inspect_scene_inputs(saved, prior, motion, evaluation, oof, checks, roles):
    for key in ('membership', 'node_ids', 'query_seconds'):
        checks.array(saved[key], prior[key], '133 T exact '+key)
    for key in ('node_ids', 'query_seconds'):
        checks.array(motion[key], evaluation[key], 'native evaluation '+key)
        checks.array(prior[key], motion[key], '133/native '+key)
    membership = prior['membership']
    groups = [np.flatnonzero(membership == gid) for gid in range(len(prior['group_offsets'])-1)]
    declared = []
    targets = {str(t): index for index, t in enumerate(evaluation['target_ids'])}
    claimed = set()
    scope = dict(total_group_queries=len(groups)*len(motion['query_seconds']), source_support_zero=0,
                 source_support_one=0, source_support_multi=0,
                 complete_target_reference_valid_query_count=int(evaluation['target_valid'][..., :2].all(-1).sum()),
                 labeled_group_query_count=0, duplicate_anchor_group_count=0, unknown_anchor_group_count=0,
                 first_anchor_target_count=0, target_without_anchor_count=0,
                 first_anchor_reference_invalid_query_count=0, first_anchor_reference_valid_zero_source_count=0,
                 first_anchor_reference_valid_one_source_count=0, first_anchor_reference_valid_multi_source_count=0,
                 mixed_identity_group_count=0, duplicate_group_queries=0, unknown_group_queries=0,
                 reference_valid_points_without_anchor=0)
    for gid, (begin, end) in enumerate(zip(prior['group_offsets'][:-1], prior['group_offsets'][1:])):
        count = prior['available'][begin:end].sum(0)
        checks.exact(np.all(count <= 2), 'actual source count at most two')
        for name, mask in (('zero', count == 0), ('one', count == 1), ('multi', count >= 2)):
            scope['source_support_'+name] += int(mask.sum())
        identity = str(evaluation['identities'][groups[gid][0]])
        scope['mixed_identity_group_count'] += int(len(set(evaluation['identities'][groups[gid]].tolist())) > 1)
        expected = np.zeros(len(count), bool)
        if identity not in targets:
            scope['unknown_anchor_group_count'] += 1
            scope['unknown_group_queries'] += len(count)
        elif identity in claimed:
            scope['duplicate_anchor_group_count'] += 1
            scope['duplicate_group_queries'] += len(count)
        else:
            claimed.add(identity)
            valid = evaluation['target_valid'][targets[identity], :, :2].all(-1)
            expected = valid & (count == 2)
            scope['first_anchor_target_count'] += 1
            scope['first_anchor_reference_invalid_query_count'] += int(np.sum(~valid))
            for name, mask in (('zero', count == 0), ('one', count == 1), ('multi', count >= 2)):
                scope['first_anchor_reference_valid_'+name+'_source_count'] += int(np.sum(mask & valid))
        checks.array(prior['labeled'][begin:end].any(0), expected, 'independent first representative legal T labels')
        for query in np.flatnonzero(expected):
            sources = np.flatnonzero(prior['labeled'][begin:end, query])+begin
            checks.exact(len(sources) == 2, 'actual K2 eligible query')
            declared.append((gid, int(query), sources))
    expected_pairs = {(gid, query) for gid, query, _ in declared}
    actual_pairs = list(zip(saved['group_indices'].tolist(), saved['query_indices'].tolist()))
    checks.exact(len(actual_pairs) == len(expected_pairs) and set(actual_pairs) == expected_pairs, 'complete unique133 label population')
    lookup = {(gid, query): sources for gid, query, sources in declared}
    source_rows = np.asarray([lookup[pair] for pair in actual_pairs], np.int64).reshape(-1, 2)
    checks.array(saved['source_rows'], source_rows, 'original source order')
    m = len(actual_pairs)
    moments = np.zeros((m, 2, 3))
    anchors = np.empty((m, 2)); truth = np.empty((m, 2)); pure = np.empty(m, bool)
    features = np.empty((m, 2, 10)); source_errors = np.empty((m, 2))
    for index, (gid, query) in enumerate(actual_pairs):
        group = groups[gid]; anchor = group[0]
        identity = str(evaluation['identities'][anchor])
        checks.exact(identity in targets and evaluation['target_valid'][targets[identity], query, :2].all(), 'first anchor reference eligibility')
        truth[index] = evaluation['target_states'][targets[identity], query, :2]
        anchors[index] = motion['states'][anchor, query, :2]
        pure[index] = np.all(evaluation['identities'][group] == identity)
        for slot, source in enumerate(source_rows[index]):
            members = group[(prior['node_to_source_row'][group] == source) & motion['support'][group, query]]
            checks.exact(len(members) > 0, 'each eligible source supported')
            xy = motion['states'][members, query, :2]
            radians = np.radians(xy[:, 1])
            moments[index, slot] = [xy[:, 0].mean(), np.sin(radians).mean(), np.cos(radians).mean()]
        features[index] = prior['features'][source_rows[index], query]
        source_errors[index] = prior['errors_m'][source_rows[index], query]
    close(saved['moments'], moments, checks, 'binding/moments', 1e-12)
    checks.array(saved['anchors'], anchors, 'raw node anchor exact')
    checks.array(saved['truth'], truth, 'T reference exact')
    checks.array(saved['group_pure'], pure, 'evaluation-only group identity purity')
    checks.array(saved['features'], features, '133 feature exact')
    checks.array(saved['source_errors_m'], source_errors, '133 label source errors exact')
    role_rows = np.empty(len(prior['available']), np.int64)
    for row in range(len(role_rows)):
        member = np.flatnonzero(prior['node_to_source_row'] == row)[0]
        source = str(motion['sources'][member])
        if source not in roles:
            roles[source] = len(roles)
        role_rows[row] = roles[source]
    checks.array(saved['role_indices'], role_rows[source_rows], 'anonymous physical source equality roles')
    endpoints = [position(moments, a, anchors) for a in (1., 0.)]
    close(saved['source_distance_m'], distances(*endpoints), checks, 'source_distance/metres', 1e-5)
    # Independently compute reference-tangent bearings in local north/east,
    # rather than projecting3D source vectors as the production diagnostic.
    source_position = np.stack(endpoints, axis=1)
    lat, ref_lat = np.radians(source_position[..., 0]), np.radians(truth[:, None, 0])
    delta_lon = np.radians(source_position[..., 1]-truth[:, None, 1])
    north = np.cos(ref_lat)*np.sin(lat)-np.sin(ref_lat)*np.cos(lat)*np.cos(delta_lon)
    east = np.cos(lat)*np.sin(delta_lon)
    length = np.hypot(north, east)
    defined = np.all(length > 1e-12, axis=1)
    cosine = np.clip((north[:, 0]*north[:, 1]+east[:, 0]*east[:, 1])[defined]
                     /(length[:, 0]*length[:, 1])[defined], -1., 1.)
    checks.array(np.isfinite(saved['error_direction_cos']), defined, 'defined reference error direction')
    close(saved['error_direction_cos'][defined], cosine, checks, 'error_direction/cosine', 1e-8)
    for head in HEADS:
        expected = np.asarray([oof[head+'__scores'][sources, query] for (_, query), sources in zip(actual_pairs, source_rows)]).reshape(-1, 2)
        checks.array(saved['oof_'+head+'_scores'], expected, '133 heldout scores exact '+head)
    scope['labeled_group_query_count'] = m
    scope['target_without_anchor_count'] = len(targets)-len(claimed)
    scope['reference_valid_points_without_anchor'] = int(sum(evaluation['target_valid'][target, :, :2].all(-1).sum()
            for identity, target in targets.items() if identity not in claimed))
    return dict(query_count=m, scope=scope)


def inspect_arrays(saved, checks):
    oracle = independent_grid(saved['moments'], saved['anchors'], saved['truth'], checks)
    compare_oracle(saved, oracle, checks)
    close(oracle['endpoint_errors_m'], saved['source_errors_m'], checks, '133_endpoint/error', 1e-5)
    heads = {}
    for head in HEADS:
        scores = saved['oof_'+head+'_scores']
        hard = scores == scores.min(axis=1, keepdims=True)
        weights = hard/hard.sum(axis=1, keepdims=True)
        predicted = position(saved['moments'], weights[:, 0], saved['anchors'])
        error = distances(predicted, saved['truth'])
        close(distances(saved[head+'_hard_position'], predicted), np.zeros(len(error)), checks, 'hard_position/'+head, 1e-5)
        close(saved[head+'_hard_error_m'], error, checks, 'hard_error/'+head, 1e-5)
        regret = np.sum(weights*oracle['endpoint_errors_m'], axis=1)-oracle['best_single_error_m']
        close(saved[head+'_selection_regret_m'], regret, checks, 'selection_regret/'+head, 1e-5)
        # Keep actual recovered-state excess separate from expected selected
        # endpoint regret: averaging tied endpoints can improve over either.
        optimal = saved['source_errors_m'] == saved['source_errors_m'].min(1, keepdims=True)
        heads[head] = dict(error=error, excess=error-oracle['best_single_error_m'],
                           selection_regret=regret, credit=np.sum(weights*optimal, axis=1),
                           score_margin=np.abs(scores[:, 0]-scores[:, 1]), weights=weights)
    return dict(oracle=oracle, heads=heads)


def describe(values):
    values = np.asarray(values)
    result = dict(count=len(values), sum=float(values.sum()))
    for key, probability in (('median', .5), ('p95', .95), ('p99', .99)):
        result[key] = float(np.quantile(values, probability)) if len(values) else None
    result.update(mean=float(values.mean()) if len(values) else None,
                  min=float(values.min()) if len(values) else None, max=float(values.max()) if len(values) else None)
    return result


def head_stats(data, head, mask=None):
    mask = np.ones(len(data['alpha']), bool) if mask is None else mask
    error = data[head+'_hard_error_m'][mask]
    excess = error-data['best_single_error_m'][mask]
    positive = np.maximum(excess, 0.)
    scores, endpoints = data['oof_'+head+'_scores'][mask], data['endpoint_errors_m'][mask]
    tie = scores == scores.min(1, keepdims=True)
    mass = tie/tie.sum(1, keepdims=True)
    optimal = endpoints == endpoints.min(1, keepdims=True)
    credit = np.sum(mass*optimal, axis=1)
    top_count = (len(error)+99)//100
    top_sum = float(np.sort(positive)[-top_count:].sum()) if top_count else 0.
    return dict(query_count=len(error), hard_error_m=describe(error), signed_excess_m=describe(excess),
                selection_regret_m=describe(data[head+'_selection_regret_m'][mask]),
                positive_harm_sum_m=float(positive.sum()), negative_excess_benefit_sum_m=float(np.maximum(-excess, 0.).sum()),
                score_selection_credit_sum=float(credit.sum()), score_selection_accuracy=float(credit.mean()) if len(credit) else None,
                imperfect_selection_query_count=int(np.sum(credit < 1)), score_tie_query_count=int(np.sum(tie.sum(1) > 1)),
                positive_harm_gt1000m_count=int(np.sum(excess > 1000.)), top1pct_query_count=top_count,
                top1pct_positive_harm_sum_m=top_sum,
                top1pct_harm_share=top_sum/positive.sum() if positive.sum() else None)


def compare_tree(actual, expected, checks, label):
    if isinstance(actual, dict):
        checks.exact(isinstance(expected, dict) and set(actual) <= set(expected), label+' schema')
        for key, value in actual.items():
            compare_tree(value, expected[key], checks, label+'/'+key)
    else:
        checks.value(actual, expected, label.rsplit('/', 1)[-1], label)


def inspect_summary(data, public, checks, label):
    finite = np.isfinite(data['gap_bound_m'])
    oracle = {key: describe(data[key]) for key in ('best_single_error_m', 'grid_error_m', 'gain_m', 'alpha')}
    oracle.update(interior_alpha_001_099_count=int(np.sum((data['alpha'] > .01) & (data['alpha'] < .99))),
                  endpoint_alpha_count=int(np.sum((data['alpha'] == 0.) | (data['alpha'] == 1.))),
                  chosen_alpha_counts_only=True,
                  endpoint_attains_grid_min_count=int(np.sum(np.isclose(data['best_single_error_m'], data['grid_error_m'], rtol=1e-10, atol=1e-7))),
                  strict_positive_gain_interior_gt1e_6m_count=int(np.sum((data['alpha'] > 0.) & (data['alpha'] < 1.) & (data['gain_m'] > 1e-6))),
                  gains_gt_m={str(v): int(np.sum(data['gain_m'] > v)) for v in (1, 10, 100)},
                  gain_sums_gt_m={str(v): float(data['gain_m'][data['gain_m'] > v].sum()) for v in (1, 10, 100)},
                  gain_fractions_gt_m={str(v): float(np.mean(data['gain_m'] > v)) if len(data['alpha']) else None for v in (1, 10, 100)},
                  error_direction_cos=describe(data['error_direction_cos'][np.isfinite(data['error_direction_cos'])]),
                  undefined_direction_count=int(np.sum(~np.isfinite(data['error_direction_cos']))),
                  gap_bound_m=describe(data['gap_bound_m'][finite]), undefined_gap_count=int(np.sum(~finite)),
                  gap_le_m={str(v): int(np.sum(data['gap_bound_m'] <= v)) for v in (.1, 1, 10)})
    compare_tree(dict(query_count=len(data['alpha']), oracle=oracle), public, checks, label)
    endpoints = data['endpoint_errors_m']
    ties = endpoints == endpoints.min(1, keepdims=True)
    credits = ties/ties.sum(1, keepdims=True)
    role = {'role_'+str(r): float(np.sum(credits*(data['role_indices'] == r))) for r in np.unique(data['role_indices'])}
    checks.exact(set(public['role_best_credit']) == set(role), label+' all anonymous roles')
    compare_tree(role, public['role_best_credit'], checks, label+'/roles')
    for name, mask in (('pure', data['group_pure']), ('mixed', ~data['group_pure'])):
        compare_tree(dict(query_count=int(mask.sum()), gain_m=describe(data['gain_m'][mask])),
                     public['group_purity'][name], checks, label+'/'+name)
    for head in HEADS:
        compare_tree(head_stats(data, head), public['heads'][head], checks, label+'/'+head)
        score = data['oof_'+head+'_scores']; chosen = score == score.min(1, keepdims=True)
        mass = chosen/chosen.sum(1, keepdims=True)
        # These fixed bins are diagnostics from the predeclared T-only contract.
        specifications = {
            'score_margin': (np.abs(score[:, 0]-score[:, 1]), [0,.05,.1,.25,.5,1,np.inf]),
            'source_distance_m': (data['source_distance_m'], [0,250,500,1000,3000,10000,np.inf]),
        }
        for name, feature, scale, bins in (
            ('selected_nearest_seconds', 0, 60., [0,60,300,900,1800,np.inf]),
            ('selected_fit_rmse_m', 1, 1000., [0,100,500,1000,3000,np.inf]),
            ('selected_within_source_rms_m', 6, 1000., [0,100,500,1000,3000,np.inf]),
            ('selected_native_count', 2, 1., [0,4,8,16,32,np.inf]),
            ('selected_supported_member_count', 4, 1., [0,1,2,4,np.inf])):
            value = np.sum(mass*np.expm1(data['features'][:, :, feature]), axis=1)*scale
            specifications[name] = value, bins
        for name, (value, bins) in specifications.items():
            value = value.copy()
            for edge in bins:
                if np.isfinite(edge):
                    value[np.abs(value-edge) <= 1e-9+1e-12*abs(edge)] = edge
            for lower, upper in zip(bins[:-1], bins[1:]):
                mask = (value >= lower) & (value < upper)
                key = f'{lower:g}..{upper:g}'
                compare_tree(head_stats(data, head, mask), public['heads'][head]['strata'][name][key],
                             checks, label+'/'+head+'/'+name+'/'+key)
        for name, mask in (('pure', data['group_pure']), ('mixed', ~data['group_pure'])):
            compare_tree(head_stats(data, head, mask), public['heads'][head]['strata']['group_purity'][name],
                         checks, label+'/'+head+'/purity/'+name)


def join(items):
    keys = ('endpoint_errors_m', 'best_single_error_m', 'grid_error_m', 'gain_m', 'alpha', 'gap_bound_m',
            'role_indices', 'group_pure', 'source_distance_m', 'features', 'error_direction_cos')
    keys += tuple(key for head in HEADS for key in ('oof_'+head+'_scores', head+'_hard_error_m', head+'_selection_regret_m'))
    return {key: np.concatenate([item[key] for item in items]) for key in keys}


def audit_run(run_path, source_path, native_path, manifest_path, checks):
    run, summary = [core.read(run_path/name) for name in ('run.json', 'summary.json')]
    prior, marker, manifest = core.read(source_path/'run.json'), core.read(native_path/'native-cache.json'), core.read(manifest_path)
    ids = manifest['splits']['T_core'][:32]
    checks.exact(run.get('status') == summary.get('status') == 'COMPLETED' and run.get('exit_code') == 0, 'completed134 only')
    checks.exact(run.get('idea_id') == summary.get('idea_id') == IDEA and run.get('plan_version') == summary.get('plan_version') == '1.0', '134 identity/version')
    checks.exact(prior.get('status') == 'COMPLETED' and prior.get('exit_code') == 0
                 and prior.get('idea_id') == 'IDEA-SOURCE-RISK-20261030' and prior.get('analytic_head_fits') == 39, 'completed133 frozen OOF')
    checks.exact(marker.get('status') == 'CACHE_COMPLETED' and marker['scene_ids']['T_core'] == ids, 'completed127 exactT population')
    checks.exact(len(ids) == len(set(ids)) == run['completed_scene_count'] == 32
                 and run['scene_ids'] == prior['training_scene_ids'] == ids, 'same complete32 T scenes')
    for key, path in (('source_run', source_path), ('native_run', native_path), ('manifest', manifest_path)):
        checks.exact(Path(run[key]).resolve() == path.resolve(), 'declared '+key)
    checks.exact(run['grid_size'] == summary['grid_size'] == GRID and run['reference_oracle_only'] is True
                 and summary['reference_oracle_only'] is True, 'declared reference-only fixed grid')
    for key in ('new_fits', 'neural_updates', 'teacher_forwards'):
        checks.exact(run[key] == 0, 'declared zero '+key)
    for key in ('V_arrays_opened', 'test_opened', 'archive_read'):
        checks.exact(run[key] is False, 'declared no '+key)
    rows = core.read(run_path/'scene-summary.json.gz')
    checks.exact([row['scene_id'] for row in rows] == ids, 'ordered complete public scenes')
    arrays = []; roles = {}; scopes = []
    for row in rows:
        sid = row['scene_id']
        checks.exact(row['component_id'] == str(manifest['components'][sid]), 'manifest T component')
        saved = load(run_path/'private'/f'scene-{sid}.npz')
        saved['moments'] = saved['moment_pairs']
        prior_arrays = load(source_path/'private'/f'training-{sid}.npz')
        motion = load(native_path/'private/native-motion/T_core'/f'scene-{sid}.npz')
        evaluation = load(native_path/'private/native-evaluation/T_core'/f'scene-{sid}.npz')
        oof = load(source_path/'private'/f'oof-{sid}.npz')
        binding = inspect_scene_inputs(saved, prior_arrays, motion, evaluation, oof, checks, roles)
        checks.exact(row['scope'] == binding['scope'], 'complete scope counts')
        inspect_arrays(saved, checks)
        inspect_summary(saved, row, checks, 'scene')
        arrays.append(saved); scopes.append(binding['scope'])
        checks.exact(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform == 'darwin' else 1024) <= RAM, 'audit memory budget')
    role_file = load(run_path/'private/roles.npz')
    checks.array(role_file['source_names'], np.asarray(list(roles)), 'complete anonymous role mapping')
    checks.array(role_file['indices'], np.asarray(list(roles.values())), 'complete anonymous role indices')
    components = sorted({row['component_id'] for row in rows})
    checks.exact(len(components) == 12 and set(summary['by_component']) == set(components), 'twelve T components')
    frequency = Counter(row['component_id'] for row in rows)
    largest = max(components, key=lambda c: frequency[c])
    checks.exact(summary['largest_component_id'] == largest and summary['component_scene_counts'] == dict(frequency), 'largest component definition')
    checks.exact(summary['scope'] == {key: sum(scope[key] for scope in scopes) for key in scopes[0]}, 'wholeT scope')
    checks.exact(summary['source_role_count'] == len(roles), 'source role count')
    inspect_summary(join(arrays), summary['all_components'], checks, 'all')
    for component in components:
        selected = [data for data, row in zip(arrays, rows) if row['component_id'] == component]
        inspect_summary(join(selected), summary['by_component'][component], checks, 'component')
    selected = [data for data, row in zip(arrays, rows) if row['component_id'] != largest]
    inspect_summary(join(selected), summary['without_largest_component'], checks, 'without_largest')
    sums = np.array([sum(float(data['gain_m'].sum()) for data, row in zip(arrays, rows) if row['component_id'] == c) for c in components])
    counts = np.array([sum(len(data['gain_m']) for data, row in zip(arrays, rows) if row['component_id'] == c) for c in components])
    draws = np.random.default_rng(20).integers(0, len(components), size=(2000, len(components)))
    values = sums[draws].sum(1)/counts[draws].sum(1)
    public = summary['primary_bootstrap']
    compare_tree(dict(component_count=12, resamples=2000, seed=20, point_estimate_m=float(sums.sum()/counts.sum())),
                 public, checks, 'bootstrap')
    checks.exact(public['independent_test'] is False, 'T bootstrap is descriptive')
    close(public['percentile95_m'], np.quantile(values, [.025, .975]), checks, 'bootstrap/percentile95_m', 1e-5)
    return dict(T_scene_count=len(rows), T_component_count=len(components), query_count=int(counts.sum()), grid_size=GRID,
                head_count=len(HEADS), aggregate_subset_count=len(components)+2,
                main_array_metrics_and_strata_compared=True, public_bootstrap_compared=True)


def audit(run_path, source_path, native_path, manifest_path, output):
    started = time.monotonic(); before = resource.getrusage(resource.RUSAGE_SELF)
    checks = core.Checks()
    report = dict(status='RUNNING', started_at=datetime.now(timezone.utc).isoformat(),
                  scope='independent134 raw-node binding,4097 grid and continuous gap bound, frozen T OOF hard outcomes and main aggregates',
                  not_audited=['native cache construction and physical reference truth',
                               '133 head training and historical upstream models',
                               'exact floating grid tie multiplicity/order; chosen grid error and position are independently verified',
                               'causal source health, real K>=3 or physical sensor generalization'],
                  tolerances=dict(rtol=1e-9, metres_atol=1e-5))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, ensure_ascii=False)
    try:
        if sys.platform.startswith('linux'):
            if hasattr(os, 'sched_getaffinity'):
                os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:4])
            soft, hard = resource.getrlimit(resource.RLIMIT_AS)
            limit = min([ADDRESS_SPACE]+[v for v in (soft, hard) if v != resource.RLIM_INFINITY])
            resource.setrlimit(resource.RLIMIT_AS, (limit, hard))
        report.update(audit_run(run_path, source_path, native_path, manifest_path, checks), status='COMPLETED', exit_code=0)
    except BaseException as error:
        report.update(status='FAILED', exit_code=1, error_type=type(error).__name__, error=str(error))
        traceback.print_exc()
    finally:
        usage = resource.getrusage(resource.RUSAGE_SELF)
        report.update(exact_checks=checks.exact_checks, numeric_checks=checks.numeric_checks, null_checks=checks.null_checks,
                      maximum_absolute_difference=checks.maximum_absolute_difference, elapsed_seconds=time.monotonic()-started,
                      user_cpu_seconds=usage.ru_utime-before.ru_utime, system_cpu_seconds=usage.ru_stime-before.ru_stime,
                      peak_rss_bytes=int(usage.ru_maxrss*(1 if sys.platform == 'darwin' else 1024)), finished_at=datetime.now(timezone.utc).isoformat())
        with output.open('w', encoding='utf-8') as stream:
            json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('run', 'source-run', 'native-run', 'manifest', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    def interrupted(signum, frame):
        raise InterruptedError('audit interrupted')
    signal.signal(signal.SIGTERM, interrupted); signal.signal(signal.SIGINT, interrupted)
    result = audit(args.run.resolve(), args.source_run.resolve(), args.native_run.resolve(), args.manifest.resolve(), args.output.resolve())
    print(result['status']+': '+str(args.output))
    return result['exit_code']


if __name__ == '__main__':
    raise SystemExit(main())
