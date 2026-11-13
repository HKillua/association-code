"""125: T-component-heldout prediction of the benefit of a fixed soft action.

No new neural training. A frozen056 teacher supplies T scores; the new four
analytic heads use T-only labels/statistics, then freeze before originalV32.
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
import torch

from association.evaluation.fusion_controls import ordinary_fusion, attribute_errors
from association.evaluation.group_prediction import emit_group_outputs
from association.evaluation.query_contribution import recover_group_states
from association.models.evidence_association import features_from_motion
from association.training.evidence_association import load_checkpoint, normalized_features, _auc
from tools.analysis import query_contribution_control as qc
from tools.analysis import shared_component_control as shared
from tools.analysis import shared_component_full_control as full
from tools.analysis import static_partition_pilot as partition
from tools.analysis import source_mix_shrink_control as mix
from association.utils import runtime

old = qc.old
IDEA = 'IDEA-CONDITIONAL-SOURCE-VALUE-20261029'
BASES = ('WS_MEAN', 'WS_FIXED_MEAN', 'WS_LOW_SOURCE_SUPPORTED',
         'WS_FIXED_LOW_SOURCE_SUPPORTED', 'WS_ANCHOR', 'WS_NEAREST_SUPPORTED')
SOFT = 'WS_FIXED_SOFT75_SOURCE_SUPPORTED'
HEADS = ('TMEAN', 'LINEAR', 'QUADRATIC', 'SHUFFLED')
HEAD_METHODS = tuple('WS_'+head+'_VALUE_GATE' for head in HEADS)
METHODS = BASES+(SOFT,)+HEAD_METHODS
FEATURE_NAMES = ('log_corrected_distance_km', 'log_raw_distance_km', 'log_ols_velocity_difference_mps',
    'log_low_fit_rmse_km', 'log_high_fit_rmse_km', 'log_low_nearest_minutes', 'log_high_nearest_minutes',
    'log_low_native_count', 'log_high_native_count', 'log_low_supported_count', 'log_high_supported_count',
    'log_center_norm_km', 'log_other_fold_eligible_groups', 'mean_internal_signed_margin', 'negative_margin_fraction')
MOTION_KEYS = ('states', 'support', 'nearest_seconds', 'native_count', 'span_seconds', 'fit_rmse_m',
               'ols_velocity_mps', 'endpoint_velocity_mps', 'sources', 'query_seconds', 'exact_observation')
THRESHOLD = 1.7320103112387182
RAM, AS = 6*1024**3, 12*1024**3
POLY_I, POLY_J = np.triu_indices(15)


def load_motion(path):
    with np.load(path, allow_pickle=False) as saved:
        motion = {key: saved[key] for key in MOTION_KEYS}
        nodes, pairs = saved['node_ids'], saved['pairs']
    if not np.array_equal(pairs, np.column_stack(np.triu_indices(len(nodes), 1))):
        raise ValueError('all original unordered pairs required')
    return motion, nodes, pairs


@torch.no_grad()
def teacher_scores(model, normalizer, motion, pairs, check):
    scores = np.empty(len(pairs), np.float64)
    calls = 0
    for start in range(0, len(pairs), 4096):
        check()
        selected = pairs[start:start+4096]
        f, base = features_from_motion(motion, selected)
        # Match the original float32 cache followed by frozen T normalizer.
        x = normalized_features(f.astype(np.float32), normalizer)
        scores[start:start+len(selected)] = model(torch.as_tensor(x), torch.as_tensor(base, dtype=torch.float64)).numpy()
        calls += 1
        check(teacher_forward_calls_increment=1)
    if not np.isfinite(scores).all():
        raise FloatingPointError('nonfinite teacher scores')
    return scores, calls


def make_actions_features(*, states, support, nearest_seconds, sources, groups, calibration,
                          native_count, fit_rmse_m, ols_velocity_mps, pairs, scores, mean_states,
                          imported_hard=None, imported_soft=None):
    """Legal physical/score-only API; no target, reference or identity argument."""
    states, support = np.asarray(states, np.float64), np.asarray(support)
    sources, nearest = np.asarray(sources), np.asarray(nearest_seconds)
    groups = old.groups_from_membership(partition.membership_from_clusters(groups, len(states)), len(states))
    order = np.asarray(calibration['source_order']).tolist()
    folds = np.asarray(calibration['group_fold'])
    centers = np.asarray(calibration['fold_center_ne_m'], np.float64)
    eligible = np.asarray(calibration['group_fit_eligible'], bool)
    if (states.shape[1:] != (64, 4) or support.shape != states.shape[:2] or support.dtype != bool
            or nearest.shape != support.shape or sources.shape != (len(states),)
            or eligible.shape != (len(groups),)
            or folds.shape != (len(groups),) or not np.array_equal(folds, np.arange(len(groups)) % 3)
            or order != sorted(set(sources.tolist())) or len(order) > 2
            or centers.shape != (3, 2) or not np.isfinite(centers).all()):
        raise ValueError('complete lawful Q64/frozen source order/group calibration required')
    for key, value, shape in (('native_count', native_count, (len(states),)),
                              ('fit_rmse_m', fit_rmse_m, (len(states),)),
                              ('ols_velocity_mps', ols_velocity_mps, (len(states), 2))):
        if np.asarray(value).shape != shape or not np.isfinite(value).all():
            raise ValueError('invalid '+key)
    if (np.asarray(scores).shape != (len(pairs),) or not np.isfinite(scores).all()
            or (np.asarray(native_count) < 0).any() or (np.asarray(fit_rmse_m) < 0).any()):
        raise ValueError('finite scores and nonnegative count/residual proxies required')
    n = len(states); g = len(groups)
    weights = {key: np.zeros((n, 64), np.float64) for key in ('low', 'high', 'hard', 'soft')}
    features = np.zeros((g, 64, 15), np.float64)
    both = np.zeros((g, 64), bool)
    corrected = states.copy()
    margins = np.full((n, n), np.nan)
    if len(pairs):
        margins[pairs[:, 0], pairs[:, 1]] = np.asarray(scores)-THRESHOLD
        margins[pairs[:, 1], pairs[:, 0]] = np.asarray(scores)-THRESHOLD
    for gid, members in enumerate(groups):
        members = np.asarray(members)
        low = members[sources[members] == order[0]]
        high = members[sources[members] == order[1]] if len(order) == 2 else np.empty(0, np.int64)
        lc, hc = support[low].sum(0), support[high].sum(0)
        hl, hh = lc > 0, hc > 0
        both[gid] = hl & hh
        weights['low'][low] = support[low]/np.maximum(lc, 1)
        weights['high'][high] = support[high]/np.maximum(hc, 1)
        weights['hard'][low] = weights['low'][low]
        weights['hard'][high] = weights['high'][high]*(~hl)
        weights['soft'][low] = weights['low'][low]*np.where(hh, .75, 1.)
        weights['soft'][high] = weights['high'][high]*np.where(hl, .25, 1.)
        if len(high):
            corrected[high] = shared.spherical_shift(states[high], -centers[folds[gid]])
        def mean(values, which):
            # values[N] or[N,D], output[Q] or[Q,D].
            mass = weights[which][members]
            return mass.T @ np.asarray(values)[members]
        velocity = mean(ols_velocity_mps, 'low')-mean(ols_velocity_mps, 'high')
        features[gid, :, 2] = np.log1p(np.linalg.norm(velocity, axis=1))
        for field, values, scale in ((3, fit_rmse_m, 1000.), (7, native_count, 1.)):
            features[gid, :, field] = np.log1p(mean(values, 'low')/scale)
            features[gid, :, field+1] = np.log1p(mean(values, 'high')/scale)
        features[gid, :, 5] = np.log1p((weights['low'][members]*nearest[members]).sum(0)/60.)
        features[gid, :, 6] = np.log1p((weights['high'][members]*nearest[members]).sum(0)/60.)
        features[gid, :, 9:11] = np.column_stack((np.log1p(lc), np.log1p(hc)))
        features[gid, :, 11] = np.log1p(np.linalg.norm(centers[folds[gid]])/1000.)
        features[gid, :, 12] = np.log1p(np.sum((folds != folds[gid]) & eligible))
        if len(members) > 1:
            a, b = np.triu_indices(len(members), 1)
            values = margins[members[a], members[b]]
            if not np.isfinite(values).all():
                raise ValueError('complete group internal pair margins required')
            features[gid, :, 13] = values.mean()
            features[gid, :, 14] = (values < 0).mean()
    low = recover_group_states(states, weights['low'], groups)['group_states']
    raw_high = recover_group_states(states, weights['high'], groups)['group_states']
    shifted_high = recover_group_states(corrected, weights['high'], groups)['group_states']
    features[..., 0] = np.log1p(attribute_errors(low, shifted_high)['position']/1000.)
    features[..., 1] = np.log1p(attribute_errors(low, raw_high)['position']/1000.)
    features[~both] = 0.
    if not np.isfinite(features).all():
        raise FloatingPointError('nonfinite lawful query features')
    hard = (recover_group_states(corrected, weights['hard'], groups)['group_states']
            if imported_hard is None else np.asarray(imported_hard).copy())
    soft = (recover_group_states(corrected, weights['soft'], groups)['group_states']
            if imported_soft is None else np.asarray(imported_soft).copy())
    if hard.shape != (g, 64, 4) or soft.shape != hard.shape:
        raise ValueError('complete hard/soft group states required')
    hard[..., 2:] = np.asarray(mean_states)[..., 2:]
    soft[..., 2:] = np.asarray(mean_states)[..., 2:]
    old._same(hard[~both], soft[~both], 'fixed source action outside both')
    return features, both, hard, soft


def gain_labels(hard, soft, both, groups, identities, target_ids, truth, valid):
    """Evaluation-only labels: first anchor group per target, position-valid only."""
    g = len(groups)
    gain = np.full((g, 64), np.nan)
    labeled = np.zeros((g, 64), bool)
    registry = {str(identity): i for i, identity in enumerate(target_ids)}
    if len(registry) != len(target_ids):
        raise ValueError('target registry must be unique')
    claimed = set(); repeated = unknown = full_denominator = 0
    for gid, members in enumerate(groups):
        identity = str(identities[members[0]])
        if identity not in registry:
            unknown += 1; continue
        if identity in claimed:
            repeated += 1; continue
        claimed.add(identity); row = registry[identity]
        h = attribute_errors(hard[gid], truth[row])['position']
        s = attribute_errors(soft[gid], truth[row])['position']
        full_use = np.asarray(valid[row])[..., :2].all(-1) & np.isfinite(h) & np.isfinite(s)
        full_denominator += int(full_use.sum())
        use = full_use & both[gid]
        gain[gid, use] = h[use]-s[use]
        labeled[gid] = use
    return gain, labeled, dict(duplicate_anchor_groups_without_training_labels=repeated,
        unknown_anchor_groups=unknown, labeled_both_query_count=int(labeled.sum()),
        unlabeled_group_query_count=int(labeled.size-labeled.sum()), target_count=len(registry),
        first_anchor_target_count=len(claimed), target_without_group_anchor_count=len(registry)-len(claimed),
        all_matched_reference_valid_query_count=full_denominator)


def phi(x, mean, std, quadratic):
    z = np.clip((np.asarray(x)-mean)/std, -8., 8.)
    columns = [np.ones((len(z), 1)), z]
    if quadratic:
        columns.append(z[:, POLY_I]*z[:, POLY_J])
    return np.column_stack(columns)


def fit_heads(x, y, on_fit=None):
    x, y = np.asarray(x, np.float64), np.asarray(y, np.float64)
    if x.shape != (len(y), 15) or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('finite T-only feature/label training rows required')
    mean = x.mean(0) if len(x) else np.zeros(15)
    std = x.std(0) if len(x) else np.ones(15)
    std[std < 1e-6] = 1.
    models = {}
    for head in HEADS:
        quadratic = head in ('QUADRATIC', 'SHUFFLED')
        size = 1 if head == 'TMEAN' else 136 if quadratic else 16
        beta = np.zeros(size)
        permutation = np.random.default_rng(20).permutation(len(y)) if head == 'SHUFFLED' else np.arange(len(y))
        if len(y):
            if head == 'TMEAN':
                beta[0] = y.mean()
            else:
                design = phi(x, mean, std, quadratic)
                regularizer = np.eye(size); regularizer[0, 0] = 0.
                beta = np.linalg.solve(design.T @ design/len(y)+regularizer,
                                       design.T @ y[permutation]/len(y))
        models[head] = dict(head=head, mean=mean.copy(), std=std.copy(), beta=beta,
            training_label_count=len(y), no_labels_fallback=not len(y), permutation=permutation,
            objective='mean squared gain error + 1*sum(nonintercept coefficient squared)')
        if not np.isfinite(beta).all():
            raise FloatingPointError('nonfinite analytic head fit')
        if on_fit is not None:
            on_fit(head, len(y))
    return models


def predict_head(model, x):
    if model['head'] == 'TMEAN':
        return np.full(len(x), model['beta'][0], np.float64)
    return phi(np.asarray(x), model['mean'], model['std'], model['head'] in ('QUADRATIC', 'SHUFFLED')) @ model['beta']


def fit_component_fold(rows, heldout_component, on_fit=None):
    if not rows or any(row.get('split') != 'T_core' for row in rows):
        raise ValueError('new head folds must contain T_core only')
    if heldout_component not in {row['component_id'] for row in rows}:
        raise ValueError('heldout component absent')
    train = [row for row in rows if row['component_id'] != heldout_component]
    x = np.concatenate([row['x'] for row in train]) if train else np.empty((0, 15))
    y = np.concatenate([row['y'] for row in train]) if train else np.empty(0)
    return fit_heads(x, y, on_fit), train


def apply_heads(models, x, both, hard, soft):
    outputs, gains, gates = {}, {}, {}
    for head in HEADS:
        prediction = predict_head(models[head], x.reshape(-1, 15)).reshape(both.shape)
        gate = both & (prediction > 0.)
        value = hard.copy(); value[gate] = soft[gate]
        old._same(value[~both], hard[~both], 'head outside both support')
        outputs['WS_'+head+'_VALUE_GATE'] = value
        gains[head], gates[head] = prediction, gate
    return outputs, gains, gates


def gain_metrics(y, prediction, full_denominator=None):
    y, prediction = np.asarray(y), np.asarray(prediction)
    if full_denominator is not None and full_denominator < len(y):
        raise ValueError('full matched-reference denominator cannot be less than eligible rows')
    if not len(y):
        return dict(labeled_query_count=0, positive_count=0, gain_auc=None, mean_realized_gain_m=None,
                    total_realized_gain_m=0., mean_regret_m=None, selected_count=0,
                    mean_realized_gain_scope='eligible both-source/reference-valid queries only',
                    all_matched_reference_valid_query_count=full_denominator,
                    full_common_point_mae_change_m=0. if full_denominator else None,
                    full_common_point_mean_regret_m=0. if full_denominator else None)
    selected = prediction > 0.
    realized = selected*y
    return dict(labeled_query_count=len(y), positive_count=int((y > 0).sum()),
        gain_auc=_auc(y > 0., prediction), mean_realized_gain_m=float(realized.mean()),
        total_realized_gain_m=float(realized.sum()), mean_regret_m=float((np.maximum(y, 0.)-realized).mean()),
        hard_baseline_mean_regret_m=float(np.maximum(y, 0.).mean()), selected_count=int(selected.sum()),
        gain_prediction_mse=float(np.square(y-prediction).mean()),
        mean_realized_gain_scope='eligible both-source/reference-valid queries only',
        all_matched_reference_valid_query_count=full_denominator,
        full_common_point_mae_change_m=-float(realized.sum())/full_denominator if full_denominator else None,
        full_common_point_mean_regret_m=float((np.maximum(y, 0.)-realized).sum())/full_denominator if full_denominator else None,
        observed_positive_gain_threshold_m=0., near_zero_abs_observed_gain_le_1e_6_m_count=int((np.abs(y) <= 1e-6).sum()))


def evaluate_v(fragment, groups, predictions, retained, both):
    """All policy decisions must have finished before this identity/ref API."""
    targets = fragment['target_identity_ids']; saved = retained['WS_MEAN']
    old._same(saved['target_ids'], targets, 'source target registry')
    truth, valid = saved['aligned_truth'][:len(targets)], saved['aligned_truth_valid'][:len(targets)]
    registry = {str(t): i for i, t in enumerate(targets)}
    for node, identity in enumerate(fragment['identities']):
        old._same(fragment['target'][node], truth[registry[str(identity)]], '056 source truth')
        old._same(fragment['target_valid'][node], valid[registry[str(identity)]], '056 source truth valid')
    regions = old.fixed_regions(fragment, targets)
    aligned = saved['aligned_group_indices'][:len(targets)]
    matched = aligned >= 0
    region = np.zeros((len(targets), 64), bool); region[matched] = both[aligned[matched]]
    regions['predicted_group_both_sources'] = region
    methods, arrays, artifacts = {}, {}, {}
    for name in METHODS:
        outputs = emit_group_outputs(predictions[name], groups, node_count=len(fragment['native_states']))
        report = old.evaluate_full_scene(outputs, node_identities=fragment['identities'], target_identities=targets,
            target_states=truth, target_valid=valid, node_observed=fragment['support'], split='V_select',
            grouping_kind='predicted', correspondence_policy='anchor_identity_first', return_artifact=True,
            node_input_support=dict(support=fragment['support'], nearest_seconds=fragment['nearest_seconds']))
        artifact = report.pop('artifact')
        if name in retained:
            if set(artifact) != set(retained[name]):
                raise ValueError('source artifact schema differs')
            for key in artifact:
                old._same(artifact[key], retained[name][key], 'exact imported '+name+'/'+key)
        else:
            for key in retained['WS_MEAN']:
                if key not in ('aligned_prediction', 'group_states'):
                    old._same(artifact[key], retained['WS_MEAN'][key], 'fixed output '+key)
        methods[name], arrays[name] = qc.describe(report, artifact, fragment, regions)
        artifacts[name] = artifact
    pairs, pair_arrays = {}, {}
    for base, candidate in combinations(METHODS, 2):
        key = candidate+'_vs_'+base
        pairs[key], pair_arrays[key], _ = old.compare_pair(artifacts[base], artifacts[candidate], regions)
        pairs[key].update(baseline_method=base, candidate_method=candidate, side_labels=dict(A0=base, candidate=candidate))
    return dict(methods=methods, pairwise=pairs), dict(methods=arrays, pairs=pair_arrays), artifacts


def run(args):
    paths = {k: Path(getattr(args, k)).resolve() for k in ('native_run', 'source_system', 'partition_run',
              'source_control_run', 'mix_run', 'output')}
    output = paths['output']; source = paths['source_system']
    for path in [source.parent, paths['native_run'], paths['partition_run'], paths['source_control_run'], paths['mix_run']]:
        if output == path or output in path.parents or path in output.parents:
            raise ValueError('new output overlaps immutable input')
    if not 1 <= args.cpu_threads <= 4:
        raise ValueError('CPU threads1..4')
    output.mkdir(parents=True, exist_ok=False); (output/'private').mkdir(); (output/'scenes').mkdir()
    started = time.monotonic()
    state = dict(status='RUNNING', idea_id=IDEA, plan_version='1.0', started_at=datetime.now(timezone.utc).isoformat(),
        **{k: str(v) for k, v in paths.items() if k != 'output'}, models_constructed=0, teacher_forward_calls=0,
        optimizer_updates=0, analytic_head_fits=0, completed_T_scenes=0, completed_V_scenes=0,
        ridge_system_solves=0, constant_means_fitted=0, no_label_fallback_heads=0,
        archive_read=False, csv_read=False, cached_reference_read=False,
        feature_names=list(FEATURE_NAMES), methods=list(METHODS), threshold=THRESHOLD,
        gpu_used=False, test_opened=False, upstream_teacher_trained_on_all_T=True,
        upstream_threshold_historically_V_developed=True, loco_isolation_scope='new head only, not upstream teacher')
    def check(**extra):
        for name in ('teacher_forward_calls', 'analytic_head_fits', 'ridge_system_solves',
                     'constant_means_fitted', 'no_label_fallback_heads'):
            state[name] += extra.pop(name+'_increment', 0)
        state.update(extra, elapsed_seconds=time.monotonic()-started,
            peak_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform == 'darwin' else 1024)),
            advisory_exceeded=time.monotonic()-started > 1800)
        old.frozen.write(output/'run.json', state)
        if state['advisory_exceeded'] and not (output/'time_advisory.json').exists():
            old.frozen.write(output/'time_advisory.json', dict(elapsed_seconds=state['elapsed_seconds'],
                action='continue only fixed52 analytic heads/T32/V32; no additional fits or training'))
        if state['peak_rss_bytes'] > RAM:
            raise MemoryError('6GiB resident budget exceeded')
    try:
        check(phase='binding')
        torch.set_num_threads(args.cpu_threads)
        if sys.platform.startswith('linux'):
            soft_limit, hard_limit = resource.getrlimit(resource.RLIMIT_AS)
            limit = min([AS]+[v for v in (soft_limit, hard_limit) if v != resource.RLIM_INFINITY])
            resource.setrlimit(resource.RLIMIT_AS, (limit, hard_limit))
        manifest = runtime.read_json(args.manifest)
        meta = old.metadata(source, paths['partition_run'], manifest)
        train_ids, validation_ids = manifest['splits']['T_core'][:32], meta['ids']
        marker = runtime.read_json(paths['native_run']/'native-cache.json')
        native_run = runtime.read_json(paths['native_run']/'run.json')
        if (marker.get('status') != 'CACHE_COMPLETED' or marker['scene_ids']['T_core'] != train_ids
                or marker['scene_ids']['V_select'] != validation_ids or partition.THRESHOLD != THRESHOLD
                or Path(native_run['source_system']).resolve() != source
                or Path(native_run['partition_run']).resolve() != paths['partition_run']):
            raise ValueError('complete127 T32/V32 cache and original threshold required')
        control_run = runtime.read_json(paths['source_control_run']/'run.json')
        mix_run = runtime.read_json(paths['mix_run']/'run.json')
        if (control_run.get('status') != 'COMPLETED' or control_run.get('exit_code') != 0
                or control_run.get('idea_id') != mix.FULL_IDEA or mix_run.get('idea_id') != mix.IDEA
                or control_run.get('original32_ids') != validation_ids or mix_run.get('status') != 'COMPLETED'
                or mix_run.get('exit_code') != 0 or mix_run.get('scene_ids') != validation_ids
                or control_run.get('scene_ids') != meta['all_ids']
                or any(Path(record['source_system']).resolve() != source for record in (control_run, mix_run))
                or any(Path(record['partition_run']).resolve() != paths['partition_run'] for record in (control_run, mix_run))
                or Path(mix_run['source_full_run']).resolve() != paths['source_control_run']):
            raise ValueError('completed original123/124 populations required')
        prior = Path(control_run['pilot_run'])
        if Path(mix_run['source_preference_run']).resolve() != prior.resolve():
            raise ValueError('123/124 original121 lineage differs')
        check(scene_ids=validation_ids, training_scene_ids=train_ids,
              total_T_scenes=len(train_ids), total_V_scenes=len(validation_ids))
        teacher, checkpoint = load_checkpoint(meta['association']/'final.pt', torch.device('cpu'))
        if checkpoint['config'].get('motion_observation_policy') != 'canonical_all':
            raise ValueError('frozen teacher must use canonical_all')
        state['models_constructed'] = 1
        training = []
        for sid in train_ids:
            motion, nodes, pairs = load_motion(paths['native_run']/'private'/'native-motion'/'T_core'/f'scene-{sid}.npz')
            scores, calls = teacher_scores(teacher, checkpoint['normalizer'], motion, pairs, check)
            membership, _, _ = partition.agglomerate(len(nodes), pairs, scores, 'WS', check)
            groups = old.groups_from_membership(membership, len(nodes))
            if len(nodes):
                _, _, _, calibration = shared.heldout_centers(motion['states'], motion['support'], motion['sources'], groups)
            else:
                calibration = dict(source_order=np.array([], dtype=str), group_fold=np.array([], np.int64),
                    fold_center_ne_m=np.zeros((3, 2)), group_fit_eligible=np.array([], bool))
            first = np.asarray([g[0] for g in groups], np.int64)
            mean = ordinary_fusion(motion['states'], motion['nearest_seconds'], motion['support'], groups)['predictions']['mean_supported'][first]
            x, both, hard, soft = make_actions_features(**{k: motion[k] for k in
                ('states', 'support', 'nearest_seconds', 'sources', 'native_count', 'fit_rmse_m', 'ols_velocity_mps')},
                groups=groups, calibration=calibration, pairs=pairs, scores=scores, mean_states=mean)
            # Freeze all legal features and actions BEFORE opening eval file.
            np.savez_compressed(output/'private'/f'T-features-{sid}.npz', features=x, both=both,
                hard_states=hard, soft_states=soft, membership=membership, scores=scores, **calibration)
            state['cached_reference_read'] = True
            with np.load(paths['native_run']/'private'/'native-evaluation'/'T_core'/f'scene-{sid}.npz', allow_pickle=False) as ev:
                old._same(nodes, ev['node_ids'], 'T eval node order')
                old._same(motion['query_seconds'], ev['query_seconds'], 'T eval query')
                gain, labeled, audit = gain_labels(hard, soft, both, groups, ev['identities'], ev['target_ids'],
                                                  ev['target_states'], ev['target_valid'])
            np.savez_compressed(output/'private'/f'T-labels-{sid}.npz', gain=gain, labeled=labeled)
            training.append(dict(scene_id=sid, split='T_core', component_id=str(manifest['components'][sid]), x=x[labeled],
                                 y=gain[labeled], audit=audit))
            check(completed_T_scenes=len(training), last_scene_id=sid)
        del teacher
        components = sorted({r['component_id'] for r in training})
        check(training_component_ids=components, T_training_component_count=len(components))
        oof = {head: [] for head in HEADS}
        fold_index = []
        def fitted(head, count):
            check(analytic_head_fits_increment=1, last_head=head, last_fit_labeled_query_count=count,
                  no_label_fallback_heads_increment=int(count == 0),
                  constant_means_fitted_increment=int(count > 0 and head == 'TMEAN'),
                  ridge_system_solves_increment=int(count > 0 and head != 'TMEAN'))
        for fold, component in enumerate(components):
            check(phase='T_component_holdout', heldout_component=component)
            models, train = fit_component_fold(training, component, fitted)
            fold_index.append(dict(fold_index=fold, heldout_component=component,
                training_components=[c for c in components if c != component],
                training_scene_ids=[r['scene_id'] for r in train],
                heldout_scene_ids=[r['scene_id'] for r in training if r['component_id'] == component],
                labeled_training_query_count=sum(len(r['y']) for r in train),
                no_label_fallback=not sum(len(r['y']) for r in train)))
            old.frozen.write(output/'T-fold-index.json', fold_index)
            np.savez_compressed(output/'private'/f'head-fold-{fold}.npz',
                **{head+'_'+key: value for head, model in models.items() for key, value in model.items()
                   if isinstance(value, np.ndarray)})
            for row in training:
                if row['component_id'] != component:
                    continue
                for head in HEADS:
                    prediction = predict_head(models[head], row['x'])
                    oof[head].append(dict(scene_id=row['scene_id'], component_id=component,
                        y=row['y'], prediction=prediction,
                        **gain_metrics(row['y'], prediction, row['audit']['all_matched_reference_valid_query_count'])))
            check(phase='T_component_holdout', heldout_component=component)
        old.frozen.write(output/'T-fold-index.json', fold_index)
        check(phase='full_T_fit', heldout_component=None)
        models = fit_heads(np.concatenate([r['x'] for r in training]), np.concatenate([r['y'] for r in training]), fitted)
        np.savez_compressed(output/'private'/'full-T-heads.npz',
            **{head+'_'+key: value for head, model in models.items() for key, value in model.items() if isinstance(value, np.ndarray)})
        public_oof = {}
        for head, rows in oof.items():
            public_oof[head] = dict(gain_metrics(np.concatenate([r['y'] for r in rows]),
                np.concatenate([r['prediction'] for r in rows]), sum(r['all_matched_reference_valid_query_count'] for r in rows)),
                per_scene=[{k: v for k, v in r.items() if k not in ('y', 'prediction')} for r in rows],
                components={c: gain_metrics(np.concatenate([r['y'] for r in rows if r['component_id'] == c]),
                    np.concatenate([r['prediction'] for r in rows if r['component_id'] == c]),
                    sum(r['all_matched_reference_valid_query_count'] for r in rows if r['component_id'] == c)) for c in components})
        old.frozen.write(output/'T-oof.json', public_oof)
        rows, errors = [], []
        with np.load(meta['association']/'final-scores.npz', allow_pickle=False) as score_cache:
            for sid in validation_ids:
                motion, nodes, pairs = load_motion(paths['native_run']/'private'/'native-motion'/'V_select'/f'scene-{sid}.npz')
                old._same(nodes, score_cache[sid+'_node_ids'], 'V source nodes')
                old._same(pairs, score_cache[sid+'_pairs'], 'V source pairs')
                scores = score_cache[sid+'_scores']
                membership = np.asarray(old.frozen.read(paths['partition_run']/'private'/f'scene-{sid}.json.gz')['memberships']['WS'])
                groups = old.groups_from_membership(membership, len(nodes))
                with np.load(prior/'private'/f'states-{sid}.npz', allow_pickle=False) as saved:
                    prediction = {name: saved[name+'_group_states'] for name in BASES}
                    calibration = {key: saved[key] for key in mix.p.CALIBRATION_KEYS}
                with np.load(paths['mix_run']/'private'/f'states-{sid}.npz', allow_pickle=False) as saved:
                    prediction[SOFT] = saved[SOFT+'_group_states']
                x, both, hard, soft = make_actions_features(**{k: motion[k] for k in
                    ('states', 'support', 'nearest_seconds', 'sources', 'native_count', 'fit_rmse_m', 'ols_velocity_mps')},
                    groups=groups, calibration=calibration, pairs=pairs, scores=scores, mean_states=prediction['WS_MEAN'],
                    imported_hard=prediction['WS_FIXED_LOW_SOURCE_SUPPORTED'], imported_soft=prediction[SOFT])
                added, predicted_gains, gates = apply_heads(models, x, both, hard, soft)
                prediction.update(added)
                np.savez_compressed(output/'private'/f'V-features-{sid}.npz', features=x, both=both,
                    **{head+'_predicted_gain': value for head, value in predicted_gains.items()},
                    **{head+'_gate': value for head, value in gates.items()})
                # Every V decision is frozen. Only now bind evaluation artifacts.
                retained = full.load_artifacts(prior/'private'/f'states-{sid}.npz', BASES)
                retained.update(full.load_artifacts(paths['mix_run']/'private'/f'states-{sid}.npz', (SOFT,)))
                fragment, _ = old._load_arrays(source, sid)
                for k, previous in (('states', 'native_states'), ('support', 'support'),
                                    ('nearest_seconds', 'nearest_seconds'), ('query_seconds', 'query_seconds'), ('sources', 'sources')):
                    old._same(motion[k], fragment[previous], 'V127/056 '+k)
                row, error_arrays, artifacts = evaluate_v(fragment, groups, prediction, retained, both)
                row.update(scene_id=sid, component_id=str(meta['components'][sid]), node_count=len(nodes),
                    fixed_R_population_invariants_verified=True,
                    policy_action_counts={head: int(gate.sum()) for head, gate in gates.items()})
                old.frozen.write(output/'scenes'/f'scene-{sid}.json.gz', row)
                np.savez_compressed(output/'private'/f'states-{sid}.npz',
                    **{name+'_'+key: value for name, artifact in artifacts.items() for key, value in artifact.items()})
                rows.append(row); errors.append(error_arrays)
                check(phase='V_frozen_evaluation', completed_V_scenes=len(rows), last_scene_id=sid)
        components_v = sorted({r['component_id'] for r in rows})
        def aggregate(indices):
            return qc.aggregate([rows[i] for i in indices], [errors[i] for i in indices], method_names=METHODS) if indices else None
        largest = max(components_v, key=lambda c: sum(r['component_id'] == c for r in rows))
        result = dict(status='COMPLETED', idea_id=IDEA, plan_version='1.0', **aggregate(list(range(len(rows)))),
            components={c: aggregate([i for i, r in enumerate(rows) if r['component_id'] == c]) for c in components_v},
            without_largest_component=aggregate([i for i, r in enumerate(rows) if r['component_id'] != largest]),
            T_oof=public_oof, analytic_head_fits=state['analytic_head_fits'], teacher_forward_calls=state['teacher_forward_calls'],
            T_label_populations=[{k: r[k] for k in ('scene_id', 'component_id', 'audit')} for r in training],
            T_training_component_count=len(components),
            ridge_system_solves=state['ridge_system_solves'], constant_means_fitted=state['constant_means_fitted'],
            no_label_fallback_heads=state['no_label_fallback_heads'],
            optimizer_updates=0, models_constructed=1, fixed_R_population_invariants_verified=True,
            position_only=True, no_expansion=True, full_T_models_frozen_before_V=True,
            limitations=['T LOCO isolates only new analytic heads; upstream056 trained allT and threshold historicallyV developed',
                'T/V share identities; originalV32 repeatedly developed, not independent generalization',
                'raw source fit residuals/counts are proxies, not covariance or effective independent samples',
                'binary selection between two frozen position candidates; no online or credible peer claim'])
        old.frozen.write(output/'scene-summary.json.gz', rows)
        check(phase='finalizing')
        result.update(elapsed_seconds=state['elapsed_seconds'], peak_rss_bytes=state['peak_rss_bytes'])
        old.frozen.write(output/'summary.json', result)
        check(status='COMPLETED', exit_code=0, finished_at=datetime.now(timezone.utc).isoformat())
        return result
    except BaseException as error:
        state.update(status='INTERRUPTED' if isinstance(error, (KeyboardInterrupt, InterruptedError)) else 'FAILED',
            exit_code=1, error=f'{type(error).__name__}: {error}', elapsed_seconds=time.monotonic()-started,
            finished_at=datetime.now(timezone.utc).isoformat())
        old.frozen.write(output/'run.json', state); raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('native-run', 'source-system', 'partition-run', 'source-control-run', 'mix-run', 'manifest', 'output'):
        parser.add_argument('--'+key, type=Path, required=True)
    parser.add_argument('--cpu-threads', type=int, default=4)
    parser.add_argument('--time-policy', choices=('advisory',), default='advisory')
    args = parser.parse_args()
    def interrupted(signum, frame):
        raise InterruptedError(f'signal{signum}')
    signal.signal(signal.SIGTERM, interrupted); signal.signal(signal.SIGINT, interrupted)
    try:
        run(args)
    except BaseException:
        traceback.print_exc(); return 1
    print(f'COMPLETED: {args.output}/summary.json', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
