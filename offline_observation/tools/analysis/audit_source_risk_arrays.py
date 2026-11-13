"""Independent133 NumPy audit of saved source policies, T fits and V arrays.

Imports only an earlier independent NumPy audit's geometry/statistics helpers.
No production feature builder, labeler, model, predictor or evaluator is used.
Recomputed fits are audit calculations only: no new model is persisted.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
from itertools import combinations
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
import numpy as np

IDEA = 'IDEA-SOURCE-RISK-20261030'
HEADS = ('LINEAR', 'QUADRATIC', 'SHUFFLED')
IMPORTED = ('WS_MEAN', 'WS_NEAREST_SUPPORTED', 'WS_LOW_SOURCE_SUPPORTED',
            'WS_FIXED_LOW_SOURCE_SUPPORTED', 'WS_QUADRATIC_VALUE_GATE')
NEW = ('WS_SOURCE_EQUAL', 'WS_SHARED_Q_SOFT', 'WS_SHARED_Q_HARD',
       'WS_SHARED_LINEAR_SOFT', 'WS_SHARED_SHUFFLED_SOFT')
METHODS = IMPORTED + NEW
HEAD_FOR = dict(WS_SHARED_Q_SOFT='QUADRATIC', WS_SHARED_Q_HARD='QUADRATIC',
                WS_SHARED_LINEAR_SOFT='LINEAR', WS_SHARED_SHUFFLED_SOFT='SHUFFLED')
PAIRS = tuple(combinations(METHODS, 2))
PAIR_KEYS = {right+'_vs_'+left for left, right in PAIRS}
RAM, AS = 6*1024**3, 12*1024**3


def load(path):
    with np.load(path, allow_pickle=False) as payload:
        return {key: payload[key] for key in payload.files}


def close(actual, expected, checks, label, atol=1e-9):
    actual, expected = np.asarray(actual), np.asarray(expected)
    checks.exact(actual.shape == expected.shape, label+' shape')
    checks.exact(np.isfinite(actual).all() and np.isfinite(expected).all(), label+' finite')
    difference = float(np.max(np.abs(actual-expected))) if actual.size else 0.
    key = label.split('/')[0]
    checks.maximum_absolute_difference[key] = max(checks.maximum_absolute_difference.get(key, 0.), difference)
    checks.numeric_checks += 1
    checks.exact(np.allclose(actual, expected, rtol=1e-9, atol=atol), label+' values')


def distances(a, b):
    """Independent unit-vector spherical distance, unlike production haversine."""
    def unit(x):
        lat, lon = np.deg2rad(x[..., 0]), np.deg2rad(x[..., 1])
        return np.stack((np.cos(lat)*np.cos(lon), np.cos(lat)*np.sin(lon), np.sin(lat)), axis=-1)
    a, b = unit(np.asarray(a)), unit(np.asarray(b))
    return core.RADIUS*np.arctan2(np.linalg.norm(np.cross(a, b), axis=-1), np.sum(a*b, axis=-1))


def mean_states(values, weights, fallback, circular_fallback=None):
    """Independent weighted state reconstruction with declared angle fallbacks."""
    values, weights = np.asarray(values), np.asarray(weights, np.float64)
    total = weights.sum(axis=0)
    result = np.asarray(fallback).copy()
    circular_fallback = np.asarray(fallback if circular_fallback is None else circular_fallback)
    active = total > 0
    norm = weights/np.where(active, total, 1.)
    result[active, 0] = np.sum(values[..., 0]*norm, axis=0)[active]
    result[active, 2] = np.sum(values[..., 2]*norm, axis=0)[active]
    for column in (1, 3):
        radians = np.deg2rad(values[..., column])
        sine = np.einsum('nq,nq->q', np.sin(radians), norm)
        cosine = np.einsum('nq,nq->q', np.cos(radians), norm)
        use = active & (np.hypot(sine, cosine) > 1e-12)
        ambiguous = active & ~use
        result[ambiguous, column] = circular_fallback[ambiguous, column]
        angle = np.rad2deg(np.arctan2(sine, cosine))
        result[use, column] = ((angle[use]+180.) % 360.-180.) if column == 1 else (angle[use] % 360.)
    one = np.sum(weights > 0, axis=0) == 1
    if one.any():
        chosen = np.argmax(weights, axis=0)
        result[one] = values[chosen[one], np.flatnonzero(one)]
    return result


def groups_from_membership(membership, n, checks):
    m = np.asarray(membership)
    checks.exact(m.shape == (n,) and m.dtype.kind in 'iu', 'membership shape/type')
    unique = np.unique(m)
    checks.array(unique, np.arange(len(unique)), 'complete membership labels')
    groups = [np.flatnonzero(m == gid) for gid in unique]
    checks.exact(not groups or np.all(np.diff([g[0] for g in groups]) > 0), 'canonical group order')
    return groups


def inspect_pack(pack, motion, membership, checks, label):
    """Rebuild source equality classes, support, source states and all10 features."""
    states, support = motion['states'], motion['support']
    n, q = support.shape
    checks.exact(q == 64 and support.dtype == bool and states.shape == (n, q, 4), label+' original Q64')
    checks.array(pack['node_ids'], motion['node_ids'], label+' node order')
    checks.array(pack['query_seconds'], motion['query_seconds'], label+' query order')
    groups = groups_from_membership(membership, n, checks)
    rows, gids, offsets = [], [], [0]
    for gid, group in enumerate(groups):
        names = []
        for node in group:
            name = str(motion['sources'][node])
            if name not in names:
                names.append(name)
        rows.extend(np.array([node for node in group if str(motion['sources'][node]) == name], np.int64)
                    for name in names)
        gids.extend([gid]*len(names)); offsets.append(len(rows))
    s = len(rows)
    mapping = np.empty(n, np.int64)
    count = np.empty((s, q), np.int64)
    states_by_source = np.empty((s, q, 4), np.float64)
    source_velocity = np.zeros((s, q, 2), np.float64)
    features = np.zeros((s, q, 10), np.float64)
    for row, members in enumerate(rows):
        mapping[members] = row
        mask = support[members]
        count[row] = mask.sum(axis=0)
        norm = mask/np.maximum(count[row], 1)
        states_by_source[row] = mean_states(states[members], mask, states[members[0]], states[groups[gids[row]][0]])
        source_velocity[row] = np.einsum('nq,nd->qd', norm, motion['ols_velocity_mps'][members])
        for column, values, scale in (
            (0, motion['nearest_seconds'][members], 60.),
            (1, motion['fit_rmse_m'][members, None], 1000.),
            (2, motion['native_count'][members, None], 1.),
            (3, motion['span_seconds'][members, None], 60.)):
            features[row, :, column] = np.log1p(np.sum(norm*values, axis=0)/scale)
        features[row, :, 4] = np.log1p(count[row])
        features[row, :, 5] = count[row]/len(members)
        spread = np.sqrt(np.sum(norm*distances(states[members], states_by_source[row])**2, axis=0))
        features[row, :, 6] = np.log1p(spread/1000.)
    available = count > 0
    for gid, group in enumerate(groups):
        local = np.arange(offsets[gid], offsets[gid+1])
        for row in local:
            other = local[local != row]
            mask = available[other]
            k = mask.sum(axis=0)
            active = available[row] & (k > 0)
            if not active.any():
                continue
            context = mean_states(states_by_source[other], mask, states[group[0]])
            features[row, active, 7] = np.log1p(distances(states_by_source[row], context)[active]/1000.)
            context_velocity = np.sum(source_velocity[other]*mask[..., None], axis=0)/np.maximum(k[:, None], 1)
            features[row, active, 8] = np.log1p(np.linalg.norm(source_velocity[row]-context_velocity, axis=-1)[active])
            features[row, active, 9] = 1.-np.cos(np.deg2rad(states_by_source[row, active, 3]-context[active, 3]))
    features[~available] = 0.
    for key, value in dict(group_offsets=np.asarray(offsets, np.int64), node_to_source_row=mapping,
                           source_group_indices=np.asarray(gids, np.int64), supported_counts=count,
                           source_member_count=np.array([len(m) for m in rows], np.int64), available=available).items():
        checks.array(pack[key], value, label+' '+key)
    close(pack['source_states'], states_by_source, checks, 'source_states/'+label, 1e-9)
    close(pack['features'], features, checks, 'source_features/'+label, 1e-8)
    return groups


def inspect_labels(pack, groups, evaluation, checks):
    available, offsets = pack['available'], pack['group_offsets']
    targets, truth, valid = (evaluation[key] for key in ('target_ids', 'target_states', 'target_valid'))
    checks.exact(len(set(targets.tolist())) == len(targets), 'unique T targets')
    registry = {str(identity): index for index, identity in enumerate(targets)}
    labels = np.zeros_like(available)
    errors = np.full(available.shape, np.nan)
    weight = np.zeros(available.shape, np.float64)
    claimed = set(); duplicate = unknown = denominator = query_count = 0
    for gid, members in enumerate(groups):
        identity = str(evaluation['identities'][members[0]])
        if identity not in registry:
            unknown += 1; continue
        if identity in claimed:
            duplicate += 1; continue
        claimed.add(identity); target = registry[identity]
        valid_query = valid[target, :, :2].all(axis=-1)
        denominator += int(valid_query.sum())
        begin, end = offsets[gid:gid+2]
        k = available[begin:end].sum(axis=0)
        eligible = valid_query & (k >= 2)
        query_count += int(eligible.sum())
        labels[begin:end] = available[begin:end] & eligible
        for source in range(begin, end):
            use = labels[source]
            errors[source, use] = distances(pack['source_states'][source, use], truth[target, use])
            weight[source, use] = 1./k[use]
        close(weight[begin:end].sum(axis=0), eligible.astype(float), checks, 'label_weight/query_total', 1e-12)
    checks.array(pack['labeled'], labels, 'first representative source label mask')
    checks.exact(np.isnan(pack['y'][~labels]).all() and np.isnan(pack['errors_m'][~labels]).all(), 'unlabeled outcomes unavailable')
    close(pack['errors_m'][labels], errors[labels], checks, 'T_error/metres', 1e-5)
    close(pack['y'][labels], np.log1p(errors[labels]/1000.), checks, 'T_label/log_error', 1e-8)
    close(pack['sample_weight'], weight, checks, 'label_weight/source', 1e-12)
    return dict(labeled_source_query_count=int(labels.sum()), labeled_group_query_count=query_count,
                training_weight_sum=float(weight.sum()), duplicate_anchor_groups_without_training_labels=duplicate,
                unknown_anchor_groups=unknown, target_count=len(targets), first_anchor_target_count=len(claimed),
                target_without_group_anchor_count=len(targets)-len(claimed),
                all_matched_reference_valid_query_count=denominator,
                unlabeled_source_query_count=int(labels.size-labels.sum()))


def design(x, mean, std, head):
    z = np.maximum(-8., np.minimum(8., (x-mean)/std))
    columns = [np.ones(len(x))] + [z[:, j] for j in range(10)]
    if head != 'LINEAR':
        columns += [z[:, i]*z[:, j] for i in range(10) for j in range(i, 10)]
    return np.column_stack(columns)


def inspect_models(payload, name, training, checks):
    x = np.concatenate([row['features'][row['labeled']] for row in training]) if training else np.empty((0, 10))
    y = np.concatenate([row['y'][row['labeled']] for row in training]) if training else np.empty(0)
    w = np.concatenate([row['sample_weight'][row['labeled']] for row in training]) if training else np.empty(0)
    total = float(w.sum()); normalized = w/total if len(w) else w
    mean = np.sum(x*normalized[:, None], axis=0) if len(w) else np.zeros(10)
    std = np.sqrt(np.sum(np.square(x-mean)*normalized[:, None], axis=0)) if len(w) else np.ones(10)
    std[std < 1e-6] = 1.
    models = {}
    for head in HEADS:
        prefix = name+'__'+head+'__'
        saved = {key: payload[prefix+key] for key in
                 ('mean', 'std', 'beta', 'training_label_count', 'training_weight_sum', 'no_labels_fallback', 'permutation')}
        checks.exact(str(payload[prefix+'head']) == head, name+'/'+head+' declared head')
        close(saved['mean'], mean, checks, 'T_mean/'+name+'/'+head)
        close(saved['std'], std, checks, 'T_std/'+name+'/'+head)
        checks.value(len(y), int(saved['training_label_count']), 'training_label_count', name+'/'+head)
        checks.value(total, float(saved['training_weight_sum']), 'training_weight_sum', name+'/'+head)
        checks.exact(bool(saved['no_labels_fallback']) == (len(y) == 0), name+'/'+head+' empty fallback')
        permutation = np.random.default_rng(20).permutation(len(y)) if head == 'SHUFFLED' else np.arange(len(y))
        checks.array(saved['permutation'], permutation, name+'/'+head+' fixed permutation')
        matrix = design(x, mean, std, head)
        penalty = np.diag(np.r_[0., np.ones(matrix.shape[1]-1)])
        beta = (np.linalg.solve((matrix*normalized[:, None]).T @ matrix+penalty,
                              (matrix*normalized[:, None]).T @ y[permutation])
                if len(y) else np.zeros(matrix.shape[1]))
        close(saved['beta'], beta, checks, 'T_beta/'+name+'/'+head, 1e-8)
        models[head] = saved
    return models


def predict(model, features, head):
    return (design(features.reshape(-1, 10), model['mean'], model['std'], head) @ model['beta']).reshape(features.shape[:2])


def independent_oof(rows, predictions):
    result = {}
    for head in HEADS:
        weight_sum = mse = mae = correct = 0.
        query_count = labeled_count = 0
        for row, by_head in zip(rows, predictions):
            labeled = row['labeled']; weights = row['sample_weight'][labeled]
            error = by_head[head][labeled]-row['y'][labeled]
            weight_sum += float(weights.sum()); labeled_count += int(labeled.sum())
            mse += float(weights @ np.square(error)); mae += float(weights @ np.abs(error))
            for begin, end in zip(row['group_offsets'][:-1], row['group_offsets'][1:]):
                for query in np.flatnonzero(labeled[begin:end].any(axis=0)):
                    use = labeled[begin:end, query]
                    scores = by_head[head][begin:end, query][use]
                    actual = row['errors_m'][begin:end, query][use]
                    selected = scores == scores.min()
                    # Fraction of tie-averaged hard mass placed on true minimum sources.
                    correct += float(np.mean(actual[selected] == actual.min()))
                    query_count += 1
        result[head] = dict(weighted_mse=mse/weight_sum if weight_sum else None,
                            weighted_mae=mae/weight_sum if weight_sum else None,
                            source_selection_accuracy=correct/query_count if query_count else None,
                            selection_query_count=query_count, labeled_source_query_count=labeled_count,
                            training_weight_sum=weight_sum, weighted_squared_error_sum=mse,
                            weighted_absolute_error_sum=mae, selection_credit_sum=correct)
    return result


def inspect_training(run_path, value_path, native_path, manifest, ids, checks):
    rows = []
    for sid in ids:
        row = load(run_path/'private'/f'training-{sid}.npz')
        motion = load(native_path/'private/native-motion/T_core'/f'scene-{sid}.npz')
        evaluation = load(native_path/'private/native-evaluation/T_core'/f'scene-{sid}.npz')
        with np.load(value_path/'private'/f'T-features-{sid}.npz', allow_pickle=False) as prior:
            checks.array(row['membership'], prior['membership'], 'immutable125 T partition')
        checks.array(evaluation['node_ids'], motion['node_ids'], 'T cached evaluation node order')
        checks.array(evaluation['query_seconds'], motion['query_seconds'], 'T cached evaluation query order')
        groups = inspect_pack(row, motion, row['membership'], checks, 'T source pack')
        audit = inspect_labels(row, groups, evaluation, checks)
        component = str(manifest['components'][sid])
        checks.exact(str(row['component_id']) == component, 'T manifest component')
        rows.append(dict(row, scene_id=sid, component_id=component, audit=audit))
    components = sorted({row['component_id'] for row in rows})
    checks.exact(len(components) == 12, 'predeclared twelve T components')
    payload = load(run_path/'private/models.npz')
    expected_prefixes = ['full']+['leave_'+component for component in components]
    expected_keys = {name+'__'+head+'__'+key for name in expected_prefixes for head in HEADS for key in
                     ('head', 'mean', 'std', 'beta', 'training_label_count', 'training_weight_sum', 'no_labels_fallback', 'permutation')}
    checks.exact(set(payload) == expected_keys, 'exact39 saved model schema')
    predictions = {}
    for component in components:
        training = [row for row in rows if row['component_id'] != component]
        heldout = [row for row in rows if row['component_id'] == component]
        models = inspect_models(payload, 'leave_'+component, training, checks)
        for row in heldout:
            stored = load(run_path/'private'/f'oof-{row["scene_id"]}.npz')
            current = {}
            for head in HEADS:
                current[head] = predict(models[head], row['features'], head)
                close(stored[head+'__scores'], current[head], checks, 'OOF_scores/'+head, 1e-8)
            predictions[row['scene_id']] = current
    full = inspect_models(payload, 'full', rows, checks)
    oof = independent_oof(rows, [predictions[row['scene_id']] for row in rows])
    public = core.read(run_path/'T-oof.json')
    checks.exact(set(public['heads']) == set(HEADS), 'all public T OOF heads')
    checks.exact(set(public['components']) == set(components), 'all public T OOF components')
    public_scenes = {row['scene_id']: row for row in public['scenes']}
    checks.exact(len(public['scenes']) == len(rows) and set(public_scenes) == {row['scene_id'] for row in rows},
                 'all public T OOF scene rows')
    def compare(actual, expected, label):
        for head in HEADS:
            for key, value in actual[head].items():
                checks.value(value, expected[head][key], key, label+'/'+head)
    compare(oof, public['heads'], 'OOF/all')
    for row in rows:
        report = public_scenes[row['scene_id']]
        checks.exact(report['component_id'] == row['component_id'], 'OOF scene component')
        compare(independent_oof([row], [predictions[row['scene_id']]]), report['heads'], 'OOF/scene')
    for component in components:
        selected = [row for row in rows if row['component_id'] == component]
        compare(independent_oof(selected, [predictions[row['scene_id']] for row in selected]),
                public['components'][component], 'OOF/component')
    return full, dict(T_scene_count=len(rows), T_component_count=len(components), audited_model_count=len(expected_prefixes)*3,
                      T_label_populations=[dict(scene_id=row['scene_id'], component_id=row['component_id'], **row['audit']) for row in rows],
                      T_oof_independent=oof)


def policy_weights(scores, available, offsets, kind):
    out = np.zeros_like(scores, dtype=np.float64)
    for begin, end in zip(offsets[:-1], offsets[1:]):
        for query in range(scores.shape[1]):
            use = np.flatnonzero(available[begin:end, query])+begin
            if not len(use):
                continue
            selected = scores[use, query]
            if kind == 'equal':
                mass = np.ones(len(use))
            elif kind == 'hard':
                mass = (selected == np.min(selected)).astype(float)
            else:
                shifted = selected-np.min(selected)
                mass = np.exp(-shifted)
            out[use, query] = mass/mass.sum()
    return out


def inspect_policies(policy, motion, saved, models, checks):
    base = saved['WS_MEAN']; membership = base['node_group_indices']
    groups = inspect_pack(policy, motion, membership, checks, 'V source pack')
    offsets, available = policy['group_offsets'], policy['available']
    mapping, counts = policy['node_to_source_row'], policy['supported_counts']
    for name in NEW:
        kind = 'equal' if name == 'WS_SOURCE_EQUAL' else 'hard' if name == 'WS_SHARED_Q_HARD' else 'soft'
        head = HEAD_FOR.get(name)
        scores = np.zeros(available.shape) if head is None else predict(models[head], policy['features'], head)
        close(policy[name+'__scores'], scores, checks, 'V_scores/'+name, 1e-8)
        mass = policy_weights(scores, available, offsets, kind)
        close(policy[name+'__source_weights'], mass, checks, 'source_weights/'+name, 1e-10)
        checks.exact(np.all(mass[~available] == 0.), 'no unavailable source contribution')
        for begin, end in zip(offsets[:-1], offsets[1:]):
            close(mass[begin:end].sum(0), available[begin:end].any(0).astype(float), checks,
                  'source_weight_total/'+name, 1e-12)
        node_mass = mass[mapping]*motion['support']/np.maximum(counts[mapping], 1)
        close(policy[name+'__node_weights'], node_mass, checks, 'node_weights/'+name, 1e-10)
        expected = base['group_states'].copy()
        for gid, members in enumerate(groups):
            expected[gid] = mean_states(motion['states'][members], node_mass[members], motion['states'][members[0]])
            absent = ~motion['support'][members].any(0)
            expected[gid, absent] = base['group_states'][gid, absent]
        expected[..., 2:] = base['group_states'][..., 2:]
        close(saved[name]['group_states'][..., :2], expected[..., :2], checks, 'policy_position/'+name, 1e-9)
        checks.array(saved[name]['group_states'][..., 2:], base['group_states'][..., 2:], 'new policy speed heading exact Mean')
        checks.array(saved[name]['aligned_prediction'][..., 2:], base['aligned_prediction'][..., 2:], 'aligned speed heading exact Mean')


def bootstrap(results, rows, components):
    sums = np.array([sum(float(np.sum(result['WS_SHARED_Q_SOFT']['errors']-result['WS_SOURCE_EQUAL']['errors']))
                            for result, row in zip(results, rows) if row['component_id'] == component) for component in components])
    counts = np.array([sum(len(result['WS_SOURCE_EQUAL']['errors']) for result, row in zip(results, rows)
                          if row['component_id'] == component) for component in components])
    rng = np.random.default_rng(20)
    draws = rng.integers(0, len(components), size=(2000, len(components)))
    values = sums[draws].sum(1)/counts[draws].sum(1)
    return dict(replicates=2000, seed=20, component_count=len(components),
                point_estimate_m=float(sums.sum()/counts.sum()), ci95=np.quantile(values, [.025, .975]).tolist())


def audit_run(run_path, value_path, native_path, manifest_path, checks):
    run, summary = (core.read(run_path/name) for name in ('run.json', 'summary.json'))
    prior, marker, manifest = (core.read(value_path/'run.json'), core.read(native_path/'native-cache.json'), core.read(manifest_path))
    checks.exact(run.get('status') == summary.get('status') == 'COMPLETED' and run.get('exit_code') == 0, 'completed133 only')
    checks.exact(run.get('idea_id') == summary.get('idea_id') == IDEA and run.get('plan_version') == summary.get('plan_version') == '1.0', '133 idea/version')
    checks.exact(prior.get('status') == 'COMPLETED' and prior.get('exit_code') == 0
                 and prior.get('idea_id') == 'IDEA-CONDITIONAL-SOURCE-VALUE-20261029', 'completed125 controls')
    checks.exact(marker.get('status') == 'CACHE_COMPLETED', 'completed127 native cache')
    training_ids, ids = manifest['splits']['T_core'][:32], prior['scene_ids']
    checks.exact(len(training_ids) == len(set(training_ids)) == len(ids) == len(set(ids)) == 32, 'fixed T32 and originalV32')
    checks.exact(not set(training_ids) & set(ids) and set(ids) <= set(manifest['splits']['V_select']), 'manifest T/V roles')
    checks.exact(run['training_scene_ids'] == prior['training_scene_ids'] == training_ids
                 and run['scene_ids'] == ids and run['completed_training_scene_count'] == run['completed_scene_count'] == 32,
                 'completed declared32+32 population')
    checks.exact(tuple(run['methods']) == METHODS and tuple(run['imported_methods']) == IMPORTED, 'declared ten/five methods')
    checks.exact(marker['scene_ids']['T_core'] == training_ids and marker['scene_ids']['V_select'] == ids, 'native cache exact population')
    for key, path in (('value_run', value_path), ('native_run', native_path)):
        checks.exact(Path(run[key]).resolve() == path.resolve(), 'declared input '+key)
    models, training_report = inspect_training(run_path, value_path, native_path, manifest, training_ids, checks)
    checks.exact(summary['T_oof'] == core.read(run_path/'T-oof.json'), 'public summary/standalone OOF identical')
    rows = core.read(run_path/'scene-summary.json.gz')
    checks.exact([row['scene_id'] for row in rows] == ids, 'ordered complete scene rows')
    results = []
    for row in rows:
        sid = row['scene_id']
        checks.exact(row['component_id'] == str(manifest['components'][sid]), 'V manifest component')
        checks.exact(set(row['methods']) == set(METHODS) and set(row['pairwise']) == PAIR_KEYS, 'ten arms and45 pairs')
        for left, right in PAIRS:
            pair = row['pairwise'][right+'_vs_'+left]
            checks.exact(pair['baseline_method'] == left and pair['candidate_method'] == right, 'pair orientation')
        saved = core.artifacts(run_path/'private'/f'states-{sid}.npz', METHODS)
        original = core.artifacts(value_path/'private'/f'states-{sid}.npz', IMPORTED)
        for name in IMPORTED:
            checks.exact(set(saved[name]) == set(original[name]), 'old artifact schema')
            for key in original[name]:
                checks.array(saved[name][key], original[name][key], 'exact125 imported '+name+'/'+key)
        results.append(core.inspect_scene(saved, row, METHODS, checks))
        policy = load(run_path/'private'/f'policy-{sid}.npz')
        motion = load(native_path/'private/native-motion/V_select'/f'scene-{sid}.npz')
        inspect_policies(policy, motion, saved, models, checks)
        checks.exact(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform == 'darwin' else 1024) <= RAM, 'audit RSS bound')
    components = sorted({row['component_id'] for row in rows})
    checks.exact(len(components) == 9 and set(summary['by_component']) == set(components), 'nine V components')
    core.inspect_summary(summary['all_components'], list(range(len(rows))), results, rows, METHODS, checks, 'all')
    for component in components:
        core.inspect_summary(summary['by_component'][component], [i for i, row in enumerate(rows) if row['component_id'] == component],
                             results, rows, METHODS, checks, 'component')
    frequency = Counter(row['component_id'] for row in rows)
    largest = max(components, key=lambda component: frequency[component])
    checks.exact(summary['largest_component_id'] == largest, 'largest component by scene count')
    core.inspect_summary(summary['without_largest_component'], [i for i, row in enumerate(rows) if row['component_id'] != largest],
                         results, rows, METHODS, checks, 'without_largest')
    independent_bootstrap = bootstrap(results, rows, components)
    public = summary['primary_bootstrap']
    checks.exact(public['comparison'] == 'WS_SHARED_Q_SOFT_vs_WS_SOURCE_EQUAL'
                 and public['components'] == components and public['independent_test'] is False, 'bootstrap estimand/scope')
    for key, public_key in (('replicates', 'resamples'), ('seed', 'seed'), ('component_count', 'component_count'),
                            ('point_estimate_m', 'point_estimate_m')):
        checks.value(independent_bootstrap[key], public[public_key], key, 'component bootstrap')
    close(public['percentile95_m'], independent_bootstrap['ci95'], checks, 'bootstrap_ci95/metres', 1e-5)
    return dict(**training_report, V_scene_count=len(rows), V_component_count=len(components), method_count=len(METHODS),
                pair_count_per_scene=len(PAIRS), exact_imported_scene_arms=len(rows)*len(IMPORTED), aggregate_subset_count=2+len(components),
                primary_component_bootstrap_independent=independent_bootstrap,
                public_oof_compared=True, public_bootstrap_compared=True)


def audit(run_path, value_path, native_path, manifest_path, output):
    started = time.monotonic(); before = resource.getrusage(resource.RUSAGE_SELF); checks = core.Checks()
    report = dict(status='RUNNING', started_at=datetime.now(timezone.utc).isoformat(),
                  scope='independent133 source construction/features, T labels/LOCO39 fits/OOF, V policies and main position/population/45pairs',
                  not_audited=['raw archive/native construction and reference physical correctness',
                               'historical teacher/WS and runtime file-opening order or absence of other reads',
                               'paper metrics, additional support strata, source dynamics summaries and per-point benefit/harm counts',
                               'real K>=3 effectiveness, physical confidence calibration or independent whole-pipeline generalization'],
                  tolerances=dict(rtol=1e-9, metres_atol=1e-5, mse_atol=.001, features_atol=1e-8, model_atol=1e-8))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, ensure_ascii=False)
    try:
        if sys.platform.startswith('linux'):
            if hasattr(os, 'sched_getaffinity'):
                os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:4])
            soft, hard = resource.getrlimit(resource.RLIMIT_AS)
            limit = min([AS]+[v for v in (soft, hard) if v != resource.RLIM_INFINITY])
            resource.setrlimit(resource.RLIMIT_AS, (limit, hard))
        report.update(audit_run(run_path, value_path, native_path, manifest_path, checks), status='COMPLETED', exit_code=0)
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
    for name in ('run', 'value-run', 'native-run', 'manifest', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    def interrupted(signum, frame):
        raise InterruptedError('audit interrupted')
    signal.signal(signal.SIGTERM, interrupted); signal.signal(signal.SIGINT, interrupted)
    result = audit(args.run.resolve(), args.value_run.resolve(), args.native_run.resolve(), args.manifest.resolve(), args.output.resolve())
    print(result['status']+': '+str(args.output))
    return result['exit_code']


if __name__ == '__main__':
    raise SystemExit(main())
