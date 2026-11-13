"""139 independent saved-array audit: fixed features, meter-target Q and hard output.

Only earlier independent NumPy auditors are imported. Recomputed coefficients
are validation calculations, not a persisted new model or a new experiment.
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
for setting in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[setting] = '4'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import numpy as np
from tools.analysis import audit_observation_common as common
from tools.analysis import audit_source_risk_arrays as original
from tools.analysis import audit_observation_quality as physical

core = common.core
IDEA = 'IDEA-SOURCE-SELECTION-EVIDENCE-20261030'
EQUAL, NEAREST = 'WS_SOURCE_EQUAL', 'WS_SOURCE_NEAREST'
LOG_T, METER_T, METER_V = 'WS_LOG_Q_OOF_HARD', 'WS_METER_Q_OOF_HARD', 'WS_METER_Q_HARD'
T_METHODS = (EQUAL, NEAREST, LOG_T, METER_T)
V_METHODS = original.METHODS+(NEAREST, METER_V)
T_PRIMARY, V_PRIMARY = METER_T+'_vs_'+LOG_T, METER_V+'_vs_WS_SHARED_Q_HARD'
MODEL_FIELDS = ('head', 'mean', 'std', 'beta', 'training_label_count', 'training_weight_sum',
                'no_labels_fallback', 'permutation', 'target_kind')
RAM, MAX_AS = 6*1024**3, 12*1024**3
load = original.load


def close(actual, expected, checks, label, atol=1e-9):
    common.close(actual, expected, checks, label, atol=atol)


def meter_fit(training):
    """Independent weighted normal equations with only the response transformed."""
    x = np.concatenate([r['features'][r['labeled']] for r in training]) if training else np.empty((0, 10))
    y = np.concatenate([r['errors_m'][r['labeled']]/1000. for r in training]) if training else np.empty(0)
    weight = np.concatenate([r['sample_weight'][r['labeled']] for r in training]) if training else np.empty(0)
    total = float(weight.sum())
    w = weight/total if len(weight) else weight
    mean = np.sum(x*w[:, None], axis=0) if len(y) else np.zeros(10)
    std = np.sqrt(np.sum((x-mean)**2*w[:, None], axis=0)) if len(y) else np.ones(10)
    std[std < 1e-6] = 1.
    design = original.design(x, mean, std, 'QUADRATIC')
    penalty = np.diag(np.r_[0., np.ones(65)])
    beta = (np.linalg.solve((design*w[:, None]).T@design+penalty, (design*w[:, None]).T@y)
            if len(y) else np.zeros(66))
    return dict(mean=mean, std=std, beta=beta, training_label_count=len(y),
                training_weight_sum=total, no_labels_fallback=not len(y), permutation=np.arange(len(y)),
                head='QUADRATIC', target_kind='radial_error_km')


def inspect_fit(payload, fold, training, old_payload, checks):
    model = meter_fit(training)
    prefix = fold+'__QUADRATIC__'
    saved = {key: payload[prefix+key] for key in MODEL_FIELDS}
    for key in ('head', 'target_kind'):
        checks.exact(str(saved[key]) == model[key], fold+' '+key)
    for key in ('training_label_count', 'no_labels_fallback'):
        checks.exact(saved[key].item() == model[key], fold+' '+key)
    checks.value(float(saved['training_weight_sum']), model['training_weight_sum'], 'training_weight_sum', fold)
    checks.array(saved['permutation'], model['permutation'], fold+' no shuffled responses')
    for key in ('mean', 'std', 'beta'):
        close(saved[key], model[key], checks, fold+' independent '+key, 1e-8)
    for key in ('mean', 'std', 'training_label_count', 'training_weight_sum', 'no_labels_fallback', 'permutation'):
        close(saved[key], old_payload[prefix+key], checks, fold+' unchanged original133 '+key, 1e-8)
    return model


def inspect_training(run_path, source_path, native_path, value_path, manifest, ids, checks):
    rows = []
    for sid in ids:
        row = load(run_path/'private'/f'training-{sid}.npz')
        before = load(source_path/'private'/f'training-{sid}.npz')
        for key, value in before.items():
            # The old response is explicitly named y_log in this target contract.
            retained = row['y_log'] if key == 'y' else row[key]
            checks.array(retained, value, 'immutable original133 training '+key)
        motion = load(native_path/'private/native-motion/T_core'/f'scene-{sid}.npz')
        evaluation = load(native_path/'private/native-evaluation/T_core'/f'scene-{sid}.npz')
        member = load(value_path/'private'/f'T-features-{sid}.npz')['membership']
        checks.array(row['membership'], member, 'original125 T groups')
        for key in ('node_ids', 'query_seconds'):
            checks.array(evaluation[key], motion[key], 'original127 evaluation '+key)
        groups = original.inspect_pack(row, motion, member, checks, '139 T source pack')
        label_audit = original.inspect_labels(dict(row, y=row['y_log']), groups, evaluation, checks)
        checks.exact(np.isnan(row['y_meter'][~row['labeled']]).all(), 'unlabeled meter outcomes unavailable')
        close(row['y_meter'][row['labeled']], row['errors_m'][row['labeled']]/1000., checks, 'meter labels exactly km')
        component = str(manifest['components'][sid])
        checks.exact(str(row['component_id']) == component, 'original T component')
        rows.append(dict(row, scene_id=sid, component_id=component, audit=label_audit))
    components = sorted({row['component_id'] for row in rows})
    checks.exact(len(rows) == 32 and len(components) == 12, 'fixed T32 twelve components')
    folds = ['full']+['leave_'+c for c in components]
    payload, old_payload = load(run_path/'private/models.npz'), load(source_path/'private/models.npz')
    checks.exact(set(payload) == {f+'__QUADRATIC__'+k for f in folds for k in MODEL_FIELDS}, 'exact13 model schema')
    predictions = {}
    for component in components:
        training = [r for r in rows if r['component_id'] != component]
        model = inspect_fit(payload, 'leave_'+component, training, old_payload, checks)
        for row in (r for r in rows if r['component_id'] == component):
            sid = row['scene_id']
            saved = load(run_path/'private'/f'oof-{sid}.npz')
            old_oof = load(source_path/'private'/f'oof-{sid}.npz')
            score = original.predict(model, row['features'], 'QUADRATIC')
            close(saved['METER__scores'], score, checks, '139 complete OOF scores', 1e-8)
            checks.array(saved['LOG__scores'], old_oof['QUADRATIC__scores'], 'original log OOF score binding')
            predictions[sid] = dict(METER=score, LOG=saved['LOG__scores'], model=model)
    full = inspect_fit(payload, 'full', rows, old_payload, checks)
    folds = core.read(run_path/'T-fold-index.json')
    checks.exact(folds == core.read(source_path/'T-fold-index.json'), 'unchanged original fold index')
    checks.exact(len(folds) == 13 and {r['fold'] for r in folds} == set(['full']+['leave_'+c for c in components]),
                 'complete thirteen fold index entries')
    for fold in folds:
        selected = [r for r in rows if r['component_id'] != fold['heldout_component']]
        checks.exact(fold['training_scene_ids'] == [r['scene_id'] for r in selected]
                     and fold['training_components'] == sorted({r['component_id'] for r in selected}),
                     'explicit heldout training exclusion')
    public = core.read(run_path/'T-oof.json')
    checks.exact(set(public['heads']) == {'LOG', 'METER'} and set(public['components']) == set(components), 'all OOF heads/components')
    reports = {r['scene_id']: r for r in public['scenes']}
    checks.exact(len(reports) == len(public['scenes']) == 32 and set(reports) == set(ids), 'complete OOF scenes')
    def compare_oof(chosen, report):
        for head in ('LOG', 'METER'):
            total = square = absolute = credit_sum = 0.
            label_count = query_count = 0
            for r in chosen:
                labeled = r['labeled']; w = r['sample_weight'][labeled]
                response = r['y_log' if head == 'LOG' else 'y_meter'][labeled]
                delta = predictions[r['scene_id']][head][labeled]-response
                total += w.sum(); square += w@np.square(delta); absolute += w@np.abs(delta)
                label_count += int(labeled.sum())
                for a, b in zip(r['group_offsets'][:-1], r['group_offsets'][1:]):
                    for qi in np.flatnonzero(labeled[a:b].any(0)):
                        use = labeled[a:b, qi]
                        score = predictions[r['scene_id']][head][a:b, qi][use]
                        error = r['errors_m'][a:b, qi][use]
                        credit_sum += float(np.mean(error[score == score.min()] == error.min()))
                        query_count += 1
            expected = dict(weighted_mse=square/total if total else None, weighted_mae=absolute/total if total else None,
                source_selection_accuracy=credit_sum/query_count if query_count else None,
                selection_query_count=query_count, labeled_source_query_count=label_count,
                training_weight_sum=float(total), weighted_squared_error_sum=float(square),
                weighted_absolute_error_sum=float(absolute), selection_credit_sum=credit_sum)
            tree_equal(expected, report[head], checks, 'OOF '+head)
    compare_oof(rows, public['heads'])
    for r in rows:
        checks.exact(reports[r['scene_id']]['component_id'] == r['component_id'], 'OOF scene fold component')
        compare_oof([r], reports[r['scene_id']]['heads'])
    for component in components:
        compare_oof([r for r in rows if r['component_id'] == component], public['components'][component])
    return rows, predictions, full


def bind_original_artifacts(actual, path, names, checks, label):
    before = core.artifacts(path, names, template=names[0])
    for name in names:
        checks.exact(set(actual[name]) == set(before[name]), label+' exact original schema '+name)
        for key in actual[name]:
            checks.array(actual[name][key], before[name][key], label+' exact original '+name+'/'+key)


def inspect_policy(run_path, paths, split, row, predictions, full, checks):
    sid = row['scene_id']
    names = T_METHODS if split == 'T_core' else V_METHODS
    saved = core.artifacts(run_path/'private'/f'states-{split}-{sid}.npz', names, template=names[0])
    policy = load(run_path/'private'/f'policy-{split}-{sid}.npz')
    motion = load(paths['native_run']/'private/native-motion'/split/f'scene-{sid}.npz')
    membership = (load(paths['value_run']/'private'/f'T-features-{sid}.npz')['membership'] if split == 'T_core'
                  else np.asarray(core.read(paths['partition_run']/'private'/f'scene-{sid}.json.gz')['memberships']['WS']))
    checks.array(policy['membership'], membership, 'fixed original groups')
    groups = original.inspect_pack(policy, motion, membership, checks, split+' original source features')
    if split == 'T_core':
        bind_original_artifacts(saved, paths['quality_run']/'private'/f'states-{split}-{sid}.npz',
                                (EQUAL, NEAREST), checks, split)
        scores = {LOG_T: predictions[sid]['LOG'], METER_T: predictions[sid]['METER']}
    else:
        bind_original_artifacts(saved, paths['source_risk_run']/'private'/f'states-{sid}.npz',
                                original.METHODS, checks, split)
        bind_original_artifacts(saved, paths['quality_run']/'private'/f'states-{split}-{sid}.npz',
                                (NEAREST,), checks, split)
        scores = {METER_V: original.predict(full, policy['features'], 'QUADRATIC')}
    offset, available, mapping, counts = (policy[k] for k in
        ('group_offsets', 'available', 'node_to_source_row', 'supported_counts'))
    group_count = np.stack([available[a:b].sum(0) for a, b in zip(offset[:-1], offset[1:])])
    checks.exact(row['source_support_scope'] == dict(zero=int((group_count == 0).sum()),
        one=int((group_count == 1).sum()), at_least_two=int((group_count >= 2).sum()))
        and row['node_count'] == len(motion['node_ids']), 'complete zero/one/multi-source scope')
    mean, _ = physical.mean_and_recovery(motion, {'groups': groups})
    for name, score in scores.items():
        close(policy[name+'__scores'], score, checks, name+' complete score', 1e-8)
        weights = original.policy_weights(score, available, offset, 'hard')
        close(policy[name+'__source_weights'], weights, checks, name+' source_weights', 1e-12)
        node_weights = weights[mapping]*motion['support']/np.maximum(counts[mapping], 1)
        close(policy[name+'__node_weights'], node_weights, checks, name+' node_weights', 1e-12)
        output, _ = physical.mean_and_recovery(motion, {'groups': groups}, node_weights)
        output[..., 2:] = mean[..., 2:]
        close(policy[name+'__group_states'], output, checks, name+' independent node recovery', 1e-8)
        close(saved[name]['group_states'], output, checks, name+' frozen artifact recovery', 1e-8)
        checks.array(saved[name]['group_states'][..., 2:], saved[EQUAL]['group_states'][..., 2:], 'original speed/heading exact')
        head = 'LOG' if name == LOG_T else 'METER'
        model = (predictions[sid]['model'] if split == 'T_core' else full) if head == 'METER' else None
        values = score[available]
        selected = (weights > 0) & available
        stats = dict(available_source_queries=len(values), negative_score_count=int((values < 0).sum()),
            negative_score_fraction=float((values < 0).mean()) if len(values) else None,
            available_score_quantiles=np.quantile(values, [0, .01, .5, .99, 1]).tolist() if len(values) else [],
            selected_source_queries=int(selected.sum()), selected_negative_count=int(((score < 0) & selected).sum()),
            selected_negative_fraction=float(((score < 0) & selected).sum()/selected.sum()) if selected.sum() else None,
            exact_tied_group_queries=sum(int(((weights[a:b] > 0).sum(0) > 1).sum()) for a, b in zip(offset[:-1], offset[1:])))
        if model is not None:
            raw = (policy['features']-model['mean'])/model['std']
            stats.update(available_feature_value_count=int(available.sum()*10),
                         available_feature_clipped_count=int((np.abs(raw[available]) > 8).sum()))
        tree_equal(stats, row['score_diagnostics'][head], checks, 'score diagnostics '+head)
    return saved, policy, scores


def tree_equal(expected, actual, checks, label):
    if isinstance(expected, dict):
        checks.exact(set(expected) == set(actual), label+' object keys')
        for key, value in expected.items():
            tree_equal(value, actual[key], checks, label+'/'+key)
    elif isinstance(expected, list):
        close(actual, expected, checks, label)
    else:
        checks.value(expected, actual, label.rsplit('/', 1)[-1], label)


def describe(values):
    values = np.asarray(values, float)
    return dict(count=len(values), sum=float(values.sum()), mean=float(values.mean()) if len(values) else None,
        rmse=float(np.sqrt(np.mean(values**2))) if len(values) else None,
        median=float(np.median(values)) if len(values) else None,
        p95=float(np.percentile(values, 95)) if len(values) else None,
        p99=float(np.percentile(values, 99)) if len(values) else None,
        maximum=float(values.max()) if len(values) else None)


def diagnostic_layer(data, mask):
    out = dict(query_count=int(mask.sum()), heads={})
    for head in ('LOG', 'METER'):
        r = data[head+'__selection_regret_m'][mask]
        positive = np.maximum(r, 0.)
        n = int(np.ceil(.01*len(r)))
        out['heads'][head] = {key: describe(data[head+'__'+key][mask]) for key in
            ('actual_error_m', 'selection_cost_m', 'selection_regret_m', 'selection_credit')}
        out['heads'][head].update(strict_correct_count=int(data[head+'__strict_correct'][mask].sum()),
            strict_wrong_count=int(data[head+'__strict_wrong'][mask].sum()), tie_query_count=int(data[head+'__tie'][mask].sum()),
            harm_gt_1000m_count=int((r > 1000.).sum()), harm_up_to_1000m_count=int(((r > 0) & (r <= 1000.)).sum()),
            harm_gt_1000m_sum=float(r[r > 1000.].sum()), harm_up_to_1000m_sum=float(r[(r > 0) & (r <= 1000.)].sum()),
            top_one_percent_query_count=n,
            top_one_percent_regret_share=float(np.sort(positive)[-n:].sum()/positive.sum()) if positive.sum() else None)
    delta = data['METER__actual_error_m'][mask]-data['LOG__actual_error_m'][mask]
    regret_delta = data['METER__selection_regret_m'][mask]-data['LOG__selection_regret_m'][mask]
    out.update(meter_minus_log_actual_m=describe(delta), positive_harm_m=describe(delta[delta > 0]),
        benefit_m=describe(-delta[delta < 0]), meter_minus_log_regret_m=describe(regret_delta), strict_transition={})
    for old in ('correct', 'wrong'):
        for new in ('correct', 'wrong'):
            use = mask & data['LOG__strict_'+old] & data['METER__strict_'+new]
            out['strict_transition'][old+'_to_'+new] = dict(count=int(use.sum()),
                log_regret_sum_m=float(data['LOG__selection_regret_m'][use].sum()),
                meter_regret_sum_m=float(data['METER__selection_regret_m'][use].sum()))
    out['at_least_one_score_tie_count'] = int((mask & (data['LOG__tie'] | data['METER__tie'])).sum())
    return out


def diagnostic_summary(data):
    n = len(data['group_indices'])
    return dict(overall=diagnostic_layer(data, np.ones(n, bool)),
        by_role={name: diagnostic_layer(data, data['role_best'] == i)
                 for i, name in enumerate(('role_0_best', 'role_1_best', 'reference_tie'))},
        by_purity={name: diagnostic_layer(data, data['group_pure'] == flag)
                   for name, flag in (('pure', True), ('mixed', False))})


def inspect_diagnostic(run_path, paths, row, policy, scores, training, checks):
    sid = row['scene_id']
    data = load(run_path/'private'/f'diagnostic-{sid}.npz')
    old = load(paths['mixture_run']/'private'/f'scene-{sid}.npz')
    evaluation = load(paths['native_run']/'private/native-evaluation/T_core'/f'scene-{sid}.npz')
    for key in ('node_ids', 'query_seconds', 'membership'):
        checks.array(old[key], policy[key], '134 original D2 lineage '+key)
    for key in ('group_indices', 'query_indices', 'source_rows', 'role_indices', 'group_pure',
                'truth', 'endpoint_errors_m', 'best_single_error_m'):
        checks.array(data[key], old[key], '134 unchanged D2 '+key)
    gi, qi, sr = (data[key] for key in ('group_indices', 'query_indices', 'source_rows'))
    n = len(gi)
    checks.exact(qi.shape == gi.shape == (n,) and sr.shape == (n, 2)
                 and all(a.dtype.kind in 'iu' for a in (gi, qi, sr)), 'D2 integer shape')
    offsets = policy['group_offsets']
    checks.exact(((gi >= 0) & (gi < len(offsets)-1)).all() and ((qi >= 0) & (qi < 64)).all()
                 and ((sr >= offsets[gi, None]) & (sr < offsets[gi+1, None])).all(), 'D2 group/query/source ownership')
    expected = np.zeros_like(training['labeled'])
    expected[sr, qi[:, None]] = True
    checks.array(expected, training['labeled'], 'D2 full original133 label population')
    checks.exact(int(expected.sum()) == 2*n and np.all(policy['available'][sr, qi[:, None]]), 'unique complete dual-source keys')
    endpoint = data['endpoint_errors_m']
    close(endpoint, training['errors_m'][sr, qi[:, None]], checks, 'endpoint true labels', 1e-5)
    best = endpoint.min(1)
    close(best, data['best_single_error_m'], checks, 'best single endpoint', 1e-5)
    groups = original.groups_from_membership(policy['membership'], len(policy['node_ids']), checks)
    registry = {str(value): i for i, value in enumerate(evaluation['target_ids'])}
    target = np.asarray([registry[str(evaluation['identities'][groups[g][0]])] for g in gi])
    checks.array(data['truth'], evaluation['target_states'][target, qi, :2], 'D2 original first-anchor reference')
    checks.array(data['group_pure'], np.asarray([len(set(evaluation['identities'][groups[g]].tolist())) == 1 for g in gi]), 'D2 pure/mixed identity')
    role = np.where(endpoint[:, 0] < endpoint[:, 1], data['role_indices'][:, 0], data['role_indices'][:, 1])
    role[endpoint[:, 0] == endpoint[:, 1]] = 2
    checks.array(data['role_best'], role, 'D2 diagnostic-only best role')
    for head, name in (('LOG', LOG_T), ('METER', METER_T)):
        score = scores[name][sr, qi[:, None]]
        mass = (score == score.min(1, keepdims=True)).astype(float)
        mass /= mass.sum(1, keepdims=True)
        cost, credit = (mass*endpoint).sum(1), (mass*(endpoint == best[:, None])).sum(1)
        tie = (mass > 0).sum(1) > 1
        position = policy[name+'__group_states'][gi, qi, :2]
        computed = dict(scores=score, source_weights=mass, actual_position=position,
            actual_error_m=original.distances(position, data['truth']), selection_cost_m=cost,
            selection_regret_m=cost-best, selection_credit=credit, tie=tie,
            strict_correct=(~tie) & (credit == 1.), strict_wrong=(~tie) & (credit == 0.))
        for key, value in computed.items():
            close(data[head+'__'+key], value, checks, head+' D2 '+key, 1e-5 if key == 'actual_error_m' else 1e-9)
    checks.array(data['LOG__scores'], old['oof_QUADRATIC_scores'], '134 original Q OOF scores')
    checks.array(data['LOG__actual_position'], old['QUADRATIC_hard_position'], '134 original Q hard positions')
    close(data['LOG__actual_error_m'], old['QUADRATIC_hard_error_m'], checks, '134 original Q hard error', 1e-5)
    # After independent physical checks, exact sign/threshold summaries use the
    # preserved production floats, not a second spherical formula's roundoff.
    tree_equal(diagnostic_summary(data), row['D2_diagnostic'], checks, 'D2 scene statistics')
    return data


def inspect_diagnostic_aggregates(run_path, records, checks):
    public = core.read(run_path/'T-selection-diagnostics.json')
    def selected_summary(chosen):
        arrays = {key: np.concatenate([r['data'][key] for r in chosen]) for key in records[0]['data']}
        return diagnostic_summary(arrays)
    components = sorted({r['component_id'] for r in records})
    largest = max(components, key=lambda c: sum(r['component_id'] == c for r in records))
    expected = dict(all_components=selected_summary(records),
        by_component={c: selected_summary([r for r in records if r['component_id'] == c]) for c in components},
        without_largest_component=selected_summary([r for r in records if r['component_id'] != largest]),
        largest_component_id=largest,
        scenes=[dict(scene_id=r['scene_id'], component_id=r['component_id'], **diagnostic_summary(r['data'])) for r in records])
    # Non-statistic labels are exact; tree_equal's numeric leaves intentionally
    # never silently convert strings to numeric values.
    checks.exact(public['largest_component_id'] == largest, 'D2 largest component by scene count')
    for key in ('all_components', 'by_component', 'without_largest_component'):
        tree_equal(expected[key], public[key], checks, 'D2 aggregate '+key)
    checks.exact(len(public['scenes']) == 32, 'D2 complete public scene list')
    for actual, wanted in zip(public['scenes'], expected['scenes']):
        checks.exact(actual['scene_id'] == wanted['scene_id'] and actual['component_id'] == wanted['component_id'], 'D2 public scene order')
        tree_equal({k: v for k, v in wanted.items() if k not in ('scene_id', 'component_id')},
                   {k: v for k, v in actual.items() if k not in ('scene_id', 'component_id')}, checks, 'D2 aggregate scene')


def audit_run(run_path, paths, manifest_path, checks):
    state, summary, manifest = (core.read(p) for p in (run_path/'run.json', run_path/'summary.json', manifest_path))
    checks.exact(state['status'] == summary['status'] == 'COMPLETED' and state['exit_code'] == 0, 'completed139')
    checks.exact(state['idea_id'] == summary['idea_id'] == IDEA and state['plan_version'] == summary['plan_version'] == '1.0', 'fixed139 contract')
    for key, expected in paths.items():
        checks.exact(Path(state[key]).resolve() == expected.resolve(), 'CLI original lineage '+key)
    checks.exact(Path(state['manifest']).resolve() == manifest_path.resolve(), 'CLI original manifest')
    paths = dict(paths, value_run=Path(state['value_run']), partition_run=Path(state['partition_run']))
    for path in paths.values():
        original_state = core.read(path/'run.json')
        checks.exact(original_state['status'] == 'COMPLETED' and original_state['exit_code'] == 0, 'completed original inputs')
    source = core.read(paths['source_risk_run']/'run.json')
    quality = core.read(paths['quality_run']/'run.json')
    mixture = core.read(paths['mixture_run']/'run.json')
    for key in ('native_run', 'value_run', 'partition_run'):
        checks.exact(Path(source[key]).resolve() == paths[key].resolve(), 'source133 original lineage '+key)
    for key in ('native_run', 'value_run', 'partition_run', 'source_risk_run'):
        checks.exact(Path(quality[key]).resolve() == paths[key].resolve(), 'quality136 original lineage '+key)
    checks.exact(Path(mixture['native_run']).resolve() == paths['native_run'] and Path(mixture['source_run']).resolve() == paths['source_risk_run'], 'mixture134 original lineage')
    tids, vids = state['training_scene_ids'], state['scene_ids']
    checks.exact(tids == source['training_scene_ids'] == quality['training_scene_ids'] == mixture['scene_ids'] == manifest['splits']['T_core'][:32]
                 and vids == source['scene_ids'] == quality['scene_ids'] and set(vids).issubset(manifest['splits']['V_select'])
                 and not set(tids) & set(vids), 'fixed original T/V populations')
    checks.exact(tuple(state['methods']) == V_METHODS and tuple(state['training_methods']) == T_METHODS, 'all declared methods')
    checks.exact(state['target_kind'] == summary['target_kind'] == 'radial_error_km', 'meter regression rather than MSE critic')
    checks.exact(state['analytic_head_fits'] == summary['analytic_head_fits'] == 13, 'exactly thirteen fits')
    checks.exact(state['ridge_system_solves']+state['no_label_fallback_heads'] == 13, 'fit actual solve/fallback counts')
    for key in ('neural_models_constructed', 'optimizer_updates', 'teacher_forward_calls', 'new_state_estimates', 'new_groupings', 'new_oracle_searches'):
        checks.exact(state[key] == 0, 'declared zero '+key)
    for key in ('archive_read', 'csv_read', 'test_opened', 'gpu_used'):
        checks.exact(state[key] is False, 'declared false '+key)
    checks.exact(state['full_T_models_frozen_before_V'] is True and state['source_features_frozen_before_labels'] is True,
                 'declared separated prediction/label stages')
    checks.exact(state['time_policy'] == 'advisory' and 1 <= state['cpu_threads'] <= 4, 'fixed bounded resource policy')
    training, predictions, full = inspect_training(run_path, paths['source_risk_run'], paths['native_run'], paths['value_run'], manifest, tids, checks)
    training_by_id = {r['scene_id']: r for r in training}
    diagnostics, scopes = [], {}
    for split, ids, names, nc, rowfile, sm, primary in (
        ('T_core', tids, T_METHODS, 12, 'T-scene-summary.json.gz', core.read(run_path/'T-summary.json'), T_PRIMARY),
        ('V_select', vids, V_METHODS, 9, 'scene-summary.json.gz', summary, V_PRIMARY)):
        rows = core.read(run_path/rowfile)
        for row in rows:
            checks.exact(row['component_id'] == str(manifest['components'][row['scene_id']]) and row['split'] == split, 'original scene component/split')
            _, policy, scores = inspect_policy(run_path, paths, split, row, predictions, full, checks)
            if split == 'T_core':
                data = inspect_diagnostic(run_path, paths, row, policy, scores, training_by_id[row['scene_id']], checks)
                diagnostics.append(dict(scene_id=row['scene_id'], component_id=row['component_id'], data=data))
            checks.exact(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform == 'darwin' else 1024) <= RAM, 'bounded audit RSS')
        _, scopes[split] = common.inspect_metric_block(run_path, split, rows, sm, names, checks,
            expected_scene_ids=ids, expected_component_count=nc, expected_primary_pairs=(primary,),
            native_evaluation_root=paths['native_run']/'private/native-evaluation')
    inspect_diagnostic_aggregates(run_path, diagnostics, checks)
    checks.exact(state['completed_label_scene_count'] == state['completed_training_scene_count'] == state['completed_scene_count'] == 32, 'complete fixed endpoint')
    checks.exact(state['newly_recovered_old_T_log_scene_count'] == 32 and state['newly_recovered_policy_scene_count'] == 96, 'complete old/new replay accounting')
    return dict(metric_scope=scopes, audited_model_count=13,
        labeled_source_query_count=sum(int(r['labeled'].sum()) for r in training),
        D2_query_count=sum(len(r['data']['group_indices']) for r in diagnostics),
        T_components=12, V_components=9, original_T_controls=2, original_V_controls=11)


def audit(run_path, native_path, source_path, quality_path, mixture_path, manifest_path, output):
    started, before, checks = time.monotonic(), resource.getrusage(resource.RUSAGE_SELF), core.Checks()
    run_path, manifest_path, output = map(lambda p: Path(p).resolve(), (run_path, manifest_path, output))
    paths = {k: Path(v).resolve() for k, v in zip(('native_run', 'source_risk_run', 'quality_run', 'mixture_run'),
                                               (native_path, source_path, quality_path, mixture_path))}
    if any(output == p or p in output.parents for p in (*paths.values(), manifest_path)):
        raise ValueError('audit output cannot alter original inputs')
    report = dict(status='RUNNING', started_at=datetime.now(timezone.utc).isoformat(),
        scope='Independent139 raw pack/labels,13 meter ridge/OOF/full scores,hard node recovery,old controls,D2 and main metrics/bootstrap',
        not_audited=['physical correctness of reference or calibrated sensor health',
            'upstream teacher or original127 OLS algorithm correctness', 'physical causal mechanisms or external generalization',
            'chronological file-access proof beyond API inspection and saved-array equality',
            'speed/heading metrics beyond original-output equality',
            'additional support-conditioned pair strata beyond common all/observed/missing metrics'],
        time_policy='advisory one fixed endpoint, no retries', gpu_used=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream)
    try:
        if sys.platform.startswith('linux'):
            if hasattr(os, 'sched_getaffinity'):
                os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:4])
            soft, hard = resource.getrlimit(resource.RLIMIT_AS)
            bound = min([MAX_AS]+[v for v in (soft, hard) if v != resource.RLIM_INFINITY])
            resource.setrlimit(resource.RLIMIT_AS, (bound, hard))
        report.update(audit_run(run_path, paths, manifest_path, checks), status='COMPLETED', exit_code=0)
    except BaseException as error:
        report.update(status='FAILED', exit_code=1, error_type=type(error).__name__,
                      error=str(error) if isinstance(error, ValueError) else 'audit incomplete; preserve frozen inputs')
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
    for key in ('run', 'native-run', 'source-risk-run', 'quality-run', 'mixture-run', 'manifest', 'output'):
        parser.add_argument('--'+key, type=Path, required=True)
    args = parser.parse_args()
    def interrupted(signum, frame):
        raise InterruptedError('audit signal '+str(signum))
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    result = audit(args.run, args.native_run, args.source_risk_run, args.quality_run, args.mixture_run, args.manifest, args.output)
    print(result['status']+': '+str(args.output))
    return result['exit_code']


if __name__ == '__main__':
    raise SystemExit(main())
