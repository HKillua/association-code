"""Independent125 T-only labels, component folds, saved heads and OOF audit.

Reads immutable125/127 arrays and the frozen manifest. No production module,
teacher model, archive, V payload, fitter or evaluator is imported. Ridge
systems are recomputed only to audit the saved parameters; no model is saved
or selected. Stored gains retain the original floating-point sign convention
after an independent spherical-distance consistency check.
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
import traceback

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.analysis import audit_shared_control_arrays as core
import numpy as np

IDEA = 'IDEA-CONDITIONAL-SOURCE-VALUE-20261029'
HEADS = ('TMEAN', 'LINEAR', 'QUADRATIC', 'SHUFFLED')
HEAD_KEYS = {h+'_'+k for h in HEADS for k in ('mean', 'std', 'beta', 'permutation')}
FEATURE_NAMES = ('log_corrected_distance_km', 'log_raw_distance_km', 'log_ols_velocity_difference_mps',
    'log_low_fit_rmse_km', 'log_high_fit_rmse_km', 'log_low_nearest_minutes', 'log_high_nearest_minutes',
    'log_low_native_count', 'log_high_native_count', 'log_low_supported_count', 'log_high_supported_count',
    'log_center_norm_km', 'log_other_fold_eligible_groups', 'mean_internal_signed_margin', 'negative_margin_fraction')


def load(path):
    with np.load(path, allow_pickle=False) as saved:
        return {k: saved[k] for k in saved.files}


def close_array(actual, expected, checks, label, atol):
    actual, expected = np.asarray(actual), np.asarray(expected)
    checks.exact(actual.shape == expected.shape and actual.dtype == expected.dtype, label+' shape/dtype')
    checks.exact(np.isfinite(actual).all() and np.isfinite(expected).all(), label+' finite')
    checks.numeric_checks += 1
    difference = float(np.max(np.abs(actual-expected))) if actual.size else 0.
    checks.maximum_absolute_difference[label.split('/')[0]] = max(
        checks.maximum_absolute_difference.get(label.split('/')[0], 0.), difference)
    checks.exact(np.allclose(actual, expected, rtol=1e-9, atol=atol), label+' numeric consistency')


def label_scene(feature, labels, evaluation, motion, checks):
    """Independent canonical groups/first-anchor matching and physical gain."""
    x, both, hard, soft, membership = (feature[k] for k in
        ('features', 'both', 'hard_states', 'soft_states', 'membership'))
    n, g = len(evaluation['node_ids']), len(hard)
    checks.exact(membership.shape == (n,) and membership.dtype.kind in 'iu'
                 and np.array_equal(np.unique(membership), np.arange(g)), 'canonical complete membership')
    groups = [np.flatnonzero(membership == i) for i in range(g)]
    checks.exact(not groups or np.all(np.diff([v[0] for v in groups]) > 0), 'canonical group first-node order')
    checks.exact(x.shape == (g, 64, 15) and x.dtype == np.float64 and np.isfinite(x).all()
                 and both.shape == (g, 64) and both.dtype == bool
                 and hard.shape == soft.shape == (g, 64, 4)
                 and hard.dtype == soft.dtype == np.float64
                 and np.isfinite(hard[..., :2]).all() and np.isfinite(soft[..., :2]).all(), 'T physical/feature shapes')
    checks.exact(feature['scores'].shape == (n*(n-1)//2,), 'complete stored teacher score population')
    checks.array(evaluation['node_ids'], motion['node_ids'], '127 T node order')
    checks.array(evaluation['query_seconds'], motion['query_seconds'], '127 T query order')
    checks.exact(evaluation['identities'].shape == (n,), 'T identity population')
    targets = evaluation['target_ids']; truth, valid = evaluation['target_states'], evaluation['target_valid']
    checks.exact(targets.ndim == 1 and len(set(targets.tolist())) == len(targets)
                 and truth.shape == valid.shape == (len(targets), 64, 4) and valid.dtype == bool,
                 'complete unique T reference registry')
    registry = {str(v): i for i, v in enumerate(targets)}
    expected = np.zeros((g, 64), bool); gains = np.full((g, 64), np.nan)
    claimed = set(); repeated = unknown = denominator = 0
    for gid, members in enumerate(groups):
        identity = str(evaluation['identities'][members[0]])
        if identity not in registry:
            unknown += 1; continue
        if identity in claimed:
            repeated += 1; continue
        claimed.add(identity); row = registry[identity]
        use = valid[row, :, :2].all(-1)
        checks.exact(np.isfinite(truth[row, use, :2]).all(), 'finite valid reference positions')
        he = core.position(hard[gid:gid+1], truth[row:row+1], use[None])[0]
        se = core.position(soft[gid:gid+1], truth[row:row+1], use[None])[0]
        denominator += int(use.sum()); expected[gid] = use & both[gid]
        gains[gid, expected[gid]] = (he-se)[expected[gid]]
    checks.exact(set(labels) == {'gain', 'labeled'} and labels['gain'].shape == expected.shape
                 and labels['gain'].dtype == np.float64 and labels['labeled'].dtype == bool, 'saved label schema')
    checks.array(labels['labeled'], expected, 'first-anchor/both/reference labeled mask')
    checks.exact(np.isnan(labels['gain'][~expected]).all(), 'unlabeled gains remain unavailable')
    close_array(labels['gain'][expected], gains[expected], checks, 'physical_gain/metres', 1e-5)
    audit = dict(duplicate_anchor_groups_without_training_labels=repeated, unknown_anchor_groups=unknown,
        labeled_both_query_count=int(expected.sum()), unlabeled_group_query_count=int(expected.size-expected.sum()),
        target_count=len(targets), first_anchor_target_count=len(claimed),
        target_without_group_anchor_count=len(targets)-len(claimed), all_matched_reference_valid_query_count=denominator)
    return dict(x=x[expected], y=labels['gain'][expected], audit=audit,
        independent_gain_sign_disagreement_count=int(((labels['gain'][expected] > 0.) != (gains[expected] > 0.)).sum()))


def design(x, mean, std, quadratic):
    z = np.clip((x-mean)/std, -8., 8.)
    columns = [np.ones(len(x))]+[z[:, i] for i in range(15)]
    if quadratic:
        columns += [z[:, i]*z[:, j] for i in range(15) for j in range(i, 15)]
    return np.column_stack(columns)


def inspect_heads(path, x, y, checks):
    """Recompute only the declared train rows' moments/ridge objective."""
    saved = load(path); checks.exact(set(saved) == HEAD_KEYS, 'sixteen saved head arrays')
    mean = x.mean(0) if len(y) else np.zeros(15)
    std = x.std(0) if len(y) else np.ones(15); std[std < 1e-6] = 1.
    heads = {}; solves = 0
    for head in HEADS:
        model = {k: saved[head+'_'+k] for k in ('mean', 'std', 'beta', 'permutation')}
        close_array(model['mean'], mean, checks, 'train_only_mean/'+head, 1e-12)
        close_array(model['std'], std, checks, 'train_only_std/'+head, 1e-12)
        order = np.random.default_rng(20).permutation(len(y)) if head == 'SHUFFLED' else np.arange(len(y))
        checks.array(model['permutation'], order, 'fixed target permutation '+head)
        size = 1 if head == 'TMEAN' else 16 if head == 'LINEAR' else 136
        beta = np.zeros(size)
        if len(y):
            if head == 'TMEAN':
                beta[0] = y.mean()
            else:
                phi = design(x, mean, std, head != 'LINEAR')
                penalty = np.eye(size); penalty[0, 0] = 0.
                beta = np.linalg.solve(phi.T@phi/len(y)+penalty, phi.T@y[order]/len(y)); solves += 1
        close_array(model['beta'], beta, checks, 'train_only_beta/'+head, 1e-6)
        heads[head] = model
    return heads, solves


def predict(model, x, head):
    if head == 'TMEAN':
        return np.full(len(x), model['beta'][0], np.float64)
    return design(x, model['mean'], model['std'], head != 'LINEAR')@model['beta']


def auc(y, prediction):
    """Mann–Whitney counts of negative scores below/tied to each positive."""
    y = np.asarray(y, bool); positive = int(y.sum()); negative = len(y)-positive
    if not positive or not negative:
        return None
    order = np.argsort(prediction, kind='stable'); scores = prediction[order]; labels = y[order]
    starts = np.r_[0, np.flatnonzero(np.diff(scores) != 0.)+1]; ends = np.r_[starts[1:], len(y)]
    wins = 0.; negatives_before = 0
    for start, end in zip(starts, ends):
        pos = int(labels[start:end].sum()); neg = end-start-pos
        wins += pos*(negatives_before+.5*neg); negatives_before += neg
    return float(wins/(positive*negative))


def gain_report(y, prediction, denominator):
    n = len(y)
    if denominator < n:
        raise ValueError('full matched reference denominator smaller than both-source population')
    if not n:
        return dict(labeled_query_count=0, positive_count=0, gain_auc=None, mean_realized_gain_m=None,
            total_realized_gain_m=0., mean_regret_m=None, selected_count=0,
            mean_realized_gain_scope='eligible both-source/reference-valid queries only',
            all_matched_reference_valid_query_count=denominator,
            full_common_point_mae_change_m=0. if denominator else None,
            full_common_point_mean_regret_m=0. if denominator else None)
    selected = prediction > 0.; realized = np.where(selected, y, 0.); best = np.maximum(y, 0.)
    return dict(labeled_query_count=n, positive_count=int((y > 0.).sum()), gain_auc=auc(y > 0., prediction),
        mean_realized_gain_m=float(realized.mean()), total_realized_gain_m=float(realized.sum()),
        mean_regret_m=float((best-realized).mean()), hard_baseline_mean_regret_m=float(best.mean()),
        selected_count=int(selected.sum()), gain_prediction_mse=float(np.square(y-prediction).mean()),
        mean_realized_gain_scope='eligible both-source/reference-valid queries only',
        all_matched_reference_valid_query_count=denominator,
        full_common_point_mae_change_m=-float(realized.sum())/denominator if denominator else None,
        full_common_point_mean_regret_m=float((best-realized).sum())/denominator if denominator else None,
        observed_positive_gain_threshold_m=0., near_zero_abs_observed_gain_le_1e_6_m_count=int((np.abs(y) <= 1e-6).sum()))


def compare_metrics(actual, public, checks, label):
    checks.exact(set(actual) == set(public), label+' metric fields')
    for key, value in actual.items():
        if isinstance(value, str):
            checks.exact(value == public[key], label+'/'+key)
        elif key == 'gain_auc' and value is not None and public[key] is not None:
            checks.numeric_checks += 1
            checks.exact(abs(value-public[key]) <= 1e-12, label+'/gain_auc')
        else:
            checks.value(value, public[key], key, label)


def audit_run(run_path, manifest_path, checks):
    run, summary = (core.read(run_path/name) for name in ('run.json', 'summary.json'))
    manifest = core.read(manifest_path); ids = manifest['splits']['T_core'][:32]
    checks.exact(run.get('status') == summary.get('status') == 'COMPLETED' and run.get('exit_code') == 0
                 and run.get('idea_id') == summary.get('idea_id') == IDEA
                 and run.get('plan_version') == summary.get('plan_version') == '1.0', 'completed125 only')
    checks.exact(len(ids) == len(set(ids)) == 32 and run['training_scene_ids'] == ids
                 and run['completed_T_scenes'] == run['total_T_scenes'] == 32, 'manifest exact first32 T_core')
    checks.exact(tuple(run['feature_names']) == FEATURE_NAMES, 'original fifteen physical feature order')
    native = Path(run['native_run']); marker = core.read(native/'native-cache.json')
    checks.exact(marker['status'] == 'CACHE_COMPLETED' and marker['scene_ids']['T_core'] == ids, 'immutable127 T cache')
    components = sorted({str(manifest['components'][sid]) for sid in ids})
    checks.exact(run['training_component_ids'] == components
                 and run['T_training_component_count'] == summary['T_training_component_count'] == len(components), 'manifest T components')
    population = summary['T_label_populations']; checks.exact([r['scene_id'] for r in population] == ids, 'all T label population rows')
    rows = []
    for sid, public in zip(ids, population):
        row = label_scene(load(run_path/'private'/f'T-features-{sid}.npz'),
            load(run_path/'private'/f'T-labels-{sid}.npz'),
            load(native/'private/native-evaluation/T_core'/f'scene-{sid}.npz'),
            load(native/'private/native-motion/T_core'/f'scene-{sid}.npz'), checks)
        cid = str(manifest['components'][sid]); checks.exact(public['component_id'] == cid, 'T label component')
        checks.exact(row['audit'] == public['audit'], 'complete independent T label population')
        rows.append(dict(row, scene_id=sid, component_id=cid))
    folds = core.read(run_path/'T-fold-index.json'); public_oof = core.read(run_path/'T-oof.json')
    checks.exact(public_oof == summary['T_oof'] and set(public_oof) == set(HEADS), 'public OOF summary binding')
    checks.exact(len(folds) == len(components), 'one heldout fold per component')
    oof = {h: [] for h in HEADS}; solves = constants = empty_heads = 0
    for index, component in enumerate(components):
        train = [r for r in rows if r['component_id'] != component]
        heldout = [r for r in rows if r['component_id'] == component]
        x = np.concatenate([r['x'] for r in train]) if train else np.empty((0, 15))
        y = np.concatenate([r['y'] for r in train]) if train else np.empty(0)
        expected = dict(fold_index=index, heldout_component=component,
            training_components=[c for c in components if c != component],
            training_scene_ids=[r['scene_id'] for r in train], heldout_scene_ids=[r['scene_id'] for r in heldout],
            labeled_training_query_count=len(y), no_label_fallback=not len(y))
        checks.exact(folds[index] == expected, 'exact component-LOCO exclusions/order')
        models, count = inspect_heads(run_path/'private'/f'head-fold-{index}.npz', x, y, checks)
        solves += count; constants += int(bool(len(y))); empty_heads += 4*int(not len(y))
        for row in heldout:
            for h in HEADS:
                prediction = predict(models[h], row['x'], h)
                checks.exact(np.isfinite(prediction).all(), 'finite heldout predictions')
                oof[h].append(dict(scene_id=row['scene_id'], component_id=component, y=row['y'], prediction=prediction,
                    denominator=row['audit']['all_matched_reference_valid_query_count']))
    x = np.concatenate([r['x'] for r in rows]); y = np.concatenate([r['y'] for r in rows])
    _, count = inspect_heads(run_path/'private/full-T-heads.npz', x, y, checks)
    solves += count; constants += int(bool(len(y))); empty_heads += 4*int(not len(y))
    for h, items in oof.items():
        public = public_oof[h]
        computed = gain_report(np.concatenate([r['y'] for r in items]), np.concatenate([r['prediction'] for r in items]),
                               sum(r['denominator'] for r in items))
        compare_metrics(computed, {k:v for k,v in public.items() if k not in ('per_scene', 'components')}, checks, 'OOF all/'+h)
        checks.exact(len(public['per_scene']) == len(items) and set(public['components']) == set(components), 'complete OOF strata')
        for row, reported in zip(items, public['per_scene']):
            checks.exact(reported['scene_id'] == row['scene_id'] and reported['component_id'] == row['component_id'], 'OOF scene order/component')
            compare_metrics(gain_report(row['y'], row['prediction'], row['denominator']),
                {k:v for k,v in reported.items() if k not in ('scene_id', 'component_id')}, checks, 'OOF scene/'+h)
        for component in components:
            selected = [r for r in items if r['component_id'] == component]
            compare_metrics(gain_report(np.concatenate([r['y'] for r in selected]),
                np.concatenate([r['prediction'] for r in selected]), sum(r['denominator'] for r in selected)),
                public['components'][component], checks, 'OOF component/'+h)
    for key, value in (('analytic_head_fits', 4*(len(components)+1)), ('ridge_system_solves', solves),
                       ('constant_means_fitted', constants), ('no_label_fallback_heads', empty_heads)):
        checks.value(value, run[key], key, 'original declared fit amount')
        checks.value(value, summary[key], key, 'summary declared fit amount')
    return dict(T_scene_count=len(ids), T_component_count=len(components), head_count=4,
        T_labeled_both_query_count=len(y), all_matched_reference_valid_query_count=sum(r['audit']['all_matched_reference_valid_query_count'] for r in rows),
        saved_fold_head_count=4*len(folds), saved_fullT_head_count=4,
        audit_ridge_parameter_recomputations=solves, audit_constant_mean_recomputations=constants,
        independent_gain_sign_disagreement_count=sum(r['independent_gain_sign_disagreement_count'] for r in rows),
        neural_forward_calls=0, model_parameters_saved=False, V_payload_read=False,
        isolation_scope='saved new heads only; teacher056 trained allT and original threshold V-developed')


def audit(run_path, manifest_path, output):
    started=time.monotonic(); before=resource.getrusage(resource.RUSAGE_SELF); checks=core.Checks()
    report=dict(status='RUNNING', started_at=datetime.now(timezone.utc).isoformat(),
        scope='independent125 T labels/component-LOCO head parameters and four-head OOF metrics',
        tolerances=dict(rtol=1e-9, physical_gain_metres_atol=1e-5, normalizer_atol=1e-12, beta_atol=1e-6, auc_atol=1e-12),
        not_audited=['teacher inference, feature legality/construction, WS grouping or center calibration',
            'physical reference correctness, timestamps/pre-reference file creation order or whole-pipeline independence',
            'V performance, credible peers, calibrated physical source quality or causal mechanism'])
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as stream: json.dump(report,stream)
    try:
        if sys.platform.startswith('linux'):
            if hasattr(os,'sched_getaffinity'): os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:4])
            soft,hard=resource.getrlimit(resource.RLIMIT_AS)
            limit=min([core.MAX_AS]+[v for v in (soft,hard) if v != resource.RLIM_INFINITY])
            resource.setrlimit(resource.RLIMIT_AS,(limit,hard))
        report.update(audit_run(run_path,manifest_path,checks),status='COMPLETED',exit_code=0)
    except BaseException as error:
        report.update(status='FAILED',exit_code=1,error_type=type(error).__name__,error=str(error));traceback.print_exc()
    finally:
        usage=resource.getrusage(resource.RUSAGE_SELF)
        report.update(exact_checks=checks.exact_checks,numeric_checks=checks.numeric_checks,null_checks=checks.null_checks,
            maximum_absolute_difference=checks.maximum_absolute_difference,elapsed_seconds=time.monotonic()-started,
            user_cpu_seconds=usage.ru_utime-before.ru_utime,system_cpu_seconds=usage.ru_stime-before.ru_stime,
            peak_rss_bytes=int(usage.ru_maxrss*(1 if sys.platform=='darwin' else 1024)),finished_at=datetime.now(timezone.utc).isoformat())
        with output.open('w',encoding='utf-8') as stream:json.dump(report,stream,ensure_ascii=False,indent=2,allow_nan=False)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True);parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    def interrupted(signum,frame): raise InterruptedError('audit interrupted')
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    result=audit(args.run.resolve(),args.manifest.resolve(),args.output.resolve())
    print(result['status']+': '+str(args.output));return result['exit_code']


if __name__=='__main__':raise SystemExit(main())
