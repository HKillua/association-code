"""One-shot independent126 saved-score/statistics and training-record audit.

No production evaluator, predictor, sampler, feature reader or model is called.
Torch is used only to deserialize restricted CPU checkpoint tensors. Stored
plans/checkpoints do not prove that every runtime step used that plan/initial.
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

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.analysis import audit_shared_control_arrays as core
from tools.analysis.audit_geometry_arrays import rank_auc
import numpy as np

IDEA = 'IDEA-ORDERED-SHAPE-SIGNAL-20261029'
CONTINUATION_IDEA = 'IDEA-ORDERED-SHAPE-CONTINUATION-20261029'
ARMS = ('SUMMARY', 'ORDERED', 'DISTRIBUTION', 'SHUFFLED')
CONFUSION = ('tp', 'fp', 'fn', 'tn')
SCORE_FIELDS = ('pairs', 'labels', 'node_ids', 'sources', 'identities', 'intervals', 'joint_support_queries')
SHAPES = {'network.0.weight': (64, 30), 'network.0.bias': (64,), 'network.2.weight': (64, 64),
          'network.2.bias': (64,), 'network.4.weight': (1, 64), 'network.4.bias': (1,)}


class ScoreChecks(core.Checks):
    def value(self, actual, expected, key, label):
        if actual is None or expected is None or isinstance(actual, (int, np.integer, bool)):
            return super().value(actual, expected, key, label)
        self.numeric_checks += 1
        difference = abs(float(actual)-float(expected))
        self.maximum_absolute_difference[key] = max(self.maximum_absolute_difference.get(key, 0.), difference)
        if not np.isfinite(expected) or not np.isclose(actual, expected, rtol=1e-10, atol=1e-12):
            raise ValueError('numeric check failed: '+label+'/'+key)


def compare(actual, public, checks, label):
    """Check independently computed fields; do not invent unstored metrics."""
    if isinstance(actual, dict):
        for key, value in actual.items():
            checks.exact(key in public, label+'/'+key+' present')
            compare(value, public[key], checks, label+'/'+key)
    elif isinstance(actual, list):
        checks.exact(len(actual) == len(public), label+' row count')
        for i, (left, right) in enumerate(zip(actual, public)):
            compare(left, right, checks, label+'/'+str(i))
    elif isinstance(actual, str):
        checks.exact(actual == public, label)
    else:
        checks.value(actual, public, label.rsplit('/', 1)[-1], label)


def eligibility(record):
    pairs, spans = record['pairs'], record['intervals']
    a, b = spans[pairs[:, 0]], spans[pairs[:, 1]]
    same = record['sources'][pairs[:, 0]] == record['sources'][pairs[:, 1]]
    gap = np.maximum(a[:, 0], b[:, 0])-np.minimum(a[:, 1], b[:, 1])
    return dict(raw=np.ones(len(pairs), bool), joint_support=record['joint_support_queries'] > 0,
        no_joint_support=record['joint_support_queries'] == 0,
        paper_itsa=same & (gap >= 0.) & (gap < 1200.) & ((a[:, 1]-a[:, 0]) > 120.) & ((b[:, 1]-b[:, 0]) > 120.),
        paper_mstsa=(~same) & (gap < -120.)), same


def scene_counts(record, decision, mask):
    y, p = record['labels'][mask], decision[mask]
    tp, fp, fn, tn = map(int, ((y & p).sum(), (~y & p).sum(), (y & ~p).sum(), (~y & ~p).sum()))
    precision = tp/(tp+fp) if tp+fp else None
    recall = tp/(tp+fn) if tp+fn else None
    return dict(scene_id=record['scene_id'], component_id=record['component_id'], node_count=len(record['sources']),
        pair_count=len(y), positive_count=int(y.sum()), tp=tp, fp=fp, fn=fn, tn=tn,
        f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 1. if len(y) else None,
        precision=precision, recall=recall, p_ca=recall, p_fa=1.-precision if precision is not None else None)


def aggregate_counts(rows):
    active = [r for r in rows if r['pair_count']]
    totals = {k: sum(r[k] for r in active) for k in CONFUSION}
    tp, fp, fn, tn = (totals[k] for k in CONFUSION)
    positives = [r['f1'] for r in active if r['positive_count']]
    negatives = [r for r in active if not r['positive_count']]
    recall = tp/(tp+fn) if tp+fn else None; precision = tp/(tp+fp) if tp+fp else None
    defined = {name: [r[key] for r in active if r[key] is not None]
               for name, key in (('m_pca', 'recall'), ('m_pfa', 'p_fa'))}
    means = {k: float(np.mean(v)) if v else None for k, v in defined.items()}
    weighted = [r for r in active if r['tp']+r['fp']]
    mass = sum(r['tp']+r['fp'] for r in weighted)
    paper = dict(unit='fraction', eligible_scene_count=len(active),
        **{k: means[k] if len(defined[k]) == len(active) else None for k in means},
        mw_pca=sum((r['tp']+r['fp'])*r['recall'] for r in weighted)/mass
            if mass and all(r['recall'] is not None for r in weighted) else None,
        mw_pfa=sum(r['fp'] for r in weighted)/mass if mass else None,
        defined_scene_count={k: len(v) for k, v in defined.items()}, defined_only_mean=means,
        predicted_positive_count=mass, prediction_weighted_scene_count=len(weighted))
    return dict(scene_count=len(active), pair_count=sum(r['pair_count'] for r in active), **totals,
        macro_f1=float(np.mean([r['f1'] for r in active])) if active else None,
        positive_scene_macro_f1=float(np.mean(positives)) if positives else None,
        positive_scene_count=len(positives), negative_scene_count=len(negatives),
        negative_scene_false_acceptance_rate=sum(r['fp'] > 0 for r in negatives)/len(negatives) if negatives else None,
        micro_f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 1. if active else None,
        micro_precision=precision, micro_recall=recall, p_ca=recall,
        p_fa=1.-precision if precision is not None else None, paper_metrics=paper)


def raw_report(count_rows, indices, theta):
    selected = {scope: [count_rows[scope][i] for i in indices] for scope in ('all', 'itsa', 'mstsa')}
    result = dict(aggregate_counts(selected['all']), threshold=theta, per_scene=selected['all'],
        itsa=dict(aggregate_counts(selected['itsa']), per_scene=selected['itsa']),
        mstsa=dict(aggregate_counts(selected['mstsa']), per_scene=selected['mstsa']))
    nonempty = sum(bool(r['pair_count']) for r in selected['all'])
    result['coverage'] = dict(all_scenes=len(indices), eligible_scenes=nonempty, scenes_without_pairs=len(indices)-nonempty)
    return result


def count_all(records, scores, theta):
    rows = {name: {scope: [] for scope in ('all', 'itsa', 'mstsa')}
            for name in ('raw', 'joint_support', 'no_joint_support', 'paper_itsa', 'paper_mstsa')}
    for record, values in zip(records, scores):
        masks, same = eligibility(record)
        decision = np.zeros(len(values), bool) if theta is None else values >= theta
        for family, eligible in masks.items():
            for scope, select in (('all', np.ones(len(values), bool)), ('itsa', same), ('mstsa', ~same)):
                rows[family][scope].append(scene_counts(record, decision, eligible & select))
    return rows


def inspect_description(public, indices, records, scores, counts, theta, checks, label):
    compare(raw_report(counts['raw'], indices, theta), public['raw'], checks, label+'/raw')
    y = np.concatenate([records[i]['labels'] for i in indices]); x = np.concatenate([scores[i] for i in indices])
    checks.value(rank_auc(y, x), public['pooled_auc'], 'pooled_auc', label)
    for family in ('joint_support', 'no_joint_support'):
        compare(raw_report(counts[family], indices, theta), public['support'][family], checks, label+'/'+family)
    for task in ('itsa', 'mstsa'):
        rows = counts['paper_'+task]['all']
        report = dict(metrics=raw_report(counts['paper_'+task], indices, theta),
            candidate_count=sum(rows[i]['pair_count'] for i in indices),
            positive_count=sum(rows[i]['positive_count'] for i in indices), threshold=theta, threshold_reselected=False)
        compare(report, public['paper_eligible'][task], checks, label+'/paper_'+task)


def load_validation(run_path, ids, manifest, checks):
    records, scores = [], {arm: [] for arm in ARMS}
    for sid in ids:
        cache = run_path/'private/cache/V_select'
        with np.load(cache/f'{sid}-metadata.npz', allow_pickle=False) as z:
            r = {k: z[k] for k in SCORE_FIELDS if k not in ('pairs', 'labels')}
        r.update(scene_id=sid, component_id=str(manifest['components'][sid]),
            pairs=np.load(cache/f'{sid}-pairs.npy', allow_pickle=False), labels=np.load(cache/f'{sid}-labels.npy', allow_pickle=False))
        n, e = len(r['sources']), len(r['pairs'])
        checks.array(r['pairs'], np.column_stack(np.triu_indices(n, 1)), 'full i<j pair population')
        checks.exact(r['labels'].shape == (e,) and r['labels'].dtype == bool, 'binary cached labels')
        checks.array(r['labels'], r['identities'][r['pairs'][:, 0]] == r['identities'][r['pairs'][:, 1]], 'identity/label binding')
        checks.exact(r['intervals'].shape == (n, 2) and np.isfinite(r['intervals']).all()
            and (r['intervals'][:, 1] >= r['intervals'][:, 0]).all(), 'finite native intervals')
        checks.exact(r['joint_support_queries'].shape == (e,) and r['joint_support_queries'].dtype.kind in 'iu'
            and ((r['joint_support_queries'] >= 0) & (r['joint_support_queries'] <= 64)).all(), 'Q64 joint support count')
        for arm in ARMS:
            with np.load(run_path/'private'/arm/f'scores-{sid}.npz', allow_pickle=False) as z:
                checks.exact(set(z.files) == {'scores', *SCORE_FIELDS}, 'complete score artifact schema')
                for key in SCORE_FIELDS:
                    checks.array(z[key], r[key], 'cross-arm/cache '+key)
                values = z['scores']
            checks.exact(values.shape == (e,) and values.dtype == np.float64 and np.isfinite(values).all(), 'finite FP64 final scores')
            scores[arm].append(values)
        records.append(r)
    return records, scores


def plan_exposure(plan, ids, cache, checks, *, steps=1000, batch=256):
    checks.exact(plan.shape == (steps, batch, 2) and plan.dtype == np.int64, 'actual shared batch shape/dtype')
    flat = plan.reshape(-1, 2); scene_indices, pair_indices = flat.T
    checks.exact(((scene_indices >= 0) & (scene_indices < len(ids))).all(), 'plan scene indices')
    positive_per_step = np.zeros(steps, np.int64); scene_rows = []
    for index, sid in enumerate(ids):
        labels = np.load(cache/f'{sid}-labels.npy', mmap_mode='r', allow_pickle=False)
        pairs = np.load(cache/f'{sid}-pairs.npy', mmap_mode='r', allow_pickle=False)
        try:
            checks.exact(labels.ndim == 1 and labels.dtype == bool and pairs.shape == (len(labels), 2), 'T label/pair header')
            positions = np.flatnonzero(scene_indices == index); selected = pair_indices[positions]
            checks.exact(((selected >= 0) & (selected < len(labels))).all(), 'plan original pair indices')
            values = np.asarray(labels[selected]); unique = np.unique(selected)
            batches = positions//batch
            np.add.at(positive_per_step, batches, values)
            pos, neg = int(values.sum()), int(len(values)-values.sum())
            dp = int(np.asarray(labels[unique]).sum()); dn = len(unique)-dp
            ep = int(labels.sum()); en = len(labels)-ep
            row = dict(scene_id=sid, scene_index=index, eligible_positive_pair_count=ep, eligible_negative_pair_count=en,
                sampled_batch_count=len(np.unique(batches)), positive_batch_count=len(np.unique(batches[values])),
                negative_batch_count=len(np.unique(batches[~values])), sampled_positive_count=pos, sampled_negative_count=neg,
                distinct_positive_count=dp, distinct_negative_count=dn, repeated_positive_count=pos-dp, repeated_negative_count=neg-dn,
                sampled_pair_count=pos+neg, distinct_pair_count=dp+dn, repeated_pair_count=pos+neg-dp-dn,
                eligible_pair_count=ep+en, distinct_positive_coverage=dp/ep if ep else None,
                distinct_negative_coverage=dn/en if en else None, distinct_pair_coverage=(dp+dn)/(ep+en) if ep+en else None)
            scene_rows.append(row)
        finally:
            for value in (labels, pairs):
                if getattr(value, '_mmap', None) is not None: value._mmap.close()
    checks.exact((positive_per_step == batch//2).all(), 'every shared batch half positive/negative')
    totals = {k: sum(r[k] for r in scene_rows) for k in ('eligible_positive_pair_count', 'eligible_negative_pair_count',
        'sampled_positive_count', 'sampled_negative_count', 'distinct_positive_count', 'distinct_negative_count',
        'repeated_positive_count', 'repeated_negative_count', 'sampled_pair_count', 'distinct_pair_count',
        'repeated_pair_count', 'eligible_pair_count')}
    for name in ('positive', 'negative', 'pair'):
        denominator = totals['eligible_'+name+'_count'] if name == 'pair' else totals['eligible_'+name+'_pair_count']
        totals['distinct_'+name+'_coverage'] = totals['distinct_'+name+'_count']/denominator if denominator else None
    return dict(seed=20, completed_batch_count=steps, scene_count=len(ids),
        eligible_scene_count=sum(r['eligible_pair_count'] > 0 for r in scene_rows),
        visited_scene_count=sum(r['sampled_batch_count'] > 0 for r in scene_rows),
        eligible_positive_scene_count=sum(r['eligible_positive_pair_count'] > 0 for r in scene_rows),
        eligible_negative_scene_count=sum(r['eligible_negative_pair_count'] > 0 for r in scene_rows),
        visited_positive_scene_count=sum(r['positive_batch_count'] > 0 for r in scene_rows),
        visited_negative_scene_count=sum(r['negative_batch_count'] > 0 for r in scene_rows),
        scene_batch_count_distribution={name: sum(low <= r['sampled_batch_count'] < high for r in scene_rows)
            for name, low, high in (('0', 0, 1), ('1', 1, 2), ('2-4', 2, 5), ('5-9', 5, 10), ('10+', 10, float('inf')))},
        **totals, scenes=scene_rows)


def inspect_training(run_path, run, summary, manifest, checks):
    train_ids = manifest['splits']['T_core']
    data = core.read(run_path/'data-audit.json'); normalizers = core.read(run_path/'normalizers.json')
    sampling = core.read(run_path/'sampling-plan-audit.json')
    checks.exact(data == summary['data_audit'], 'declared data audit copy')
    checks.exact(data['train_scene_ids'] == train_ids and data['validation_scene_ids'] == manifest['splits']['V_select'], 'T2100/V300 IDs')
    checks.exact(data['train_scene_count'] == len(train_ids) == 2100 and data['validation_scene_count'] == 300, 'declared T/V counts')
    plan = np.load(run_path/'private/shared-batch-plan.npy', allow_pickle=False)
    exposure = plan_exposure(plan, train_ids, run_path/'private/cache/T_core', checks)
    compare(exposure, sampling, checks, 'shared plan exposure')
    checks.exact(sampling['actual_per_arm_steps'] == 1000 and sampling['plan_is_shared_across_arms'] is True
        and sampling['positive_per_batch'] == sampling['negative_per_batch'] == 128, 'shared plan metadata')
    checks.exact(exposure['eligible_pair_count'] == data['train_pair_count']
        and exposure['eligible_positive_pair_count'] == data['train_positive_count'], 'full T label population')
    checks.exact(run['optimizer_updates_by_arm'] == {arm: 1000 for arm in ARMS}
        and run['completed_optimizer_updates'] == summary['total_optimizer_updates'] == 4000
        and summary['optimizer_updates_per_arm'] == run['fixed_updates_per_arm'] == 1000
        and run['total_fixed_updates'] == 4000, 'declared successful updates')
    import torch
    initial = torch.load(run_path/'private/shared-initial.pt', map_location='cpu', weights_only=True)
    def parameters(state):
        checks.exact(set(state) == set(SHAPES), 'exact six parameter tensors')
        for name, shape in SHAPES.items():
            checks.exact(tuple(state[name].shape) == shape and state[name].dtype == torch.float32
                and torch.isfinite(state[name]).all().item(), 'parameter shape/dtype/finite')
        checks.exact(sum(t.numel() for t in state.values()) == 6209, '6209 saved parameters')
    parameters(initial)
    checks.exact((initial['network.0.weight'][:, 22:] == 0).all().item()
        and (initial['network.4.weight'] == 0).all().item() and (initial['network.4.bias'] == 0).all().item(), 'shared initial function zeros')
    checks.exact(set(normalizers) == set(ARMS), 'four saved normalizers')
    for arm in ARMS:
        history = core.read(run_path/'private'/arm/'history.json')
        checks.exact(len(history) == 1000 and [r['update'] for r in history] == list(range(1, 1001)), 'history 1000 successful updates')
        checks.exact(all(np.isfinite(r['loss']) and np.isfinite(r['gradient_norm']) for r in history), 'finite recorded training')
        checkpoint = torch.load(run_path/'private'/arm/'final.pt', map_location='cpu', weights_only=True)
        checks.exact(checkpoint['idea_id'] == IDEA and checkpoint['arm'] == arm and checkpoint['update'] == 1000
            and checkpoint['parameter_count'] == summary['parameter_count_per_arm'] == data['parameter_count_per_arm'] == 6209, 'final checkpoint metadata')
        parameters(checkpoint['state_dict'])
        norm = normalizers[arm]
        checks.exact(checkpoint['normalizer'] == norm and norm['fitted_split'] == 'T_core'
            and norm['pair_count'] == data['train_pair_count'] and norm['extra_feature_rule'] == arm, 'frozen T-only normalizer metadata')
        mean, std = np.asarray(norm['mean']), np.asarray(norm['std'])
        checks.exact(mean.shape == std.shape == (30,) and np.isfinite(mean).all() and np.isfinite(std).all()
            and (std > 0).all(), 'finite30 normalizer')
        checks.array(mean[:22], np.asarray(normalizers['SUMMARY']['mean'])[:22], 'shared22 mean')
        checks.array(std[:22], np.asarray(normalizers['SUMMARY']['std'])[:22], 'shared22 std')
        optimizer = checkpoint['optimizer_state']
        checks.exact(len(optimizer['state']) == 6 and all(float(v['step']) == 1000 for v in optimizer['state'].values()), 'Adam six step1000')
        checks.exact(len(optimizer['param_groups']) == 1 and optimizer['param_groups'][0]['lr'] == .001
            and len(optimizer['param_groups'][0]['params']) == 6, 'fixed optimizer group')
    checks.array(np.asarray(normalizers['SUMMARY']['mean'])[22:], np.zeros(8), 'SUMMARY extra means')
    checks.array(np.asarray(normalizers['SUMMARY']['std'])[22:], np.ones(8), 'SUMMARY extra scales')
    return dict(shared_plan_shape=list(plan.shape), planned_draws_per_arm=256000, distinct_planned_pairs_per_arm=exposure['distinct_pair_count'],
        restored_checkpoint_count=4, shared_initial_parameter_count=6209, successful_updates_per_arm=1000,
        provenance_boundary='saved plan/initial, frozen-code assertions, histories and checkpoints only; actual per-step use not independently replayed')


def inspect_pairs(public, records, scores, thresholds, macros, checks):
    keys = {right+'_vs_'+left for left, right in combinations(ARMS, 2)}
    checks.exact(set(public) == keys, 'complete six comparisons')
    for left, right in combinations(ARMS, 2):
        rows = []
        for index, record in enumerate(records):
            a = np.zeros(len(record['labels']), bool) if thresholds[left] is None else scores[left][index] >= thresholds[left]
            b = np.zeros(len(record['labels']), bool) if thresholds[right] is None else scores[right][index] >= thresholds[right]
            y = record['labels']; fixed = (a != y) & (b == y); broken = (a == y) & (b != y)
            rows.append(dict(scene_id=record['scene_id'], wrong_to_right=int(fixed.sum()), right_to_wrong=int(broken.sum()),
                corrected_fp=int((fixed & ~y).sum()), corrected_fn=int((fixed & y).sum()),
                introduced_fp=int((broken & ~y).sum()), introduced_fn=int((broken & y).sum())))
        totals = {k: sum(r[k] for r in rows) for k in rows[0] if k != 'scene_id'}
        result = dict(totals, per_scene=rows, net_corrected_edges=totals['wrong_to_right']-totals['right_to_wrong'],
            macro_f1_difference=macros[right]-macros[left], baseline_method=left, candidate_method=right)
        compare(result, public[right+'_vs_'+left], checks, 'pair '+right+'/'+left)


def inspect_run_identity(run_path, run, summary, checks):
    if run['idea_id'] == IDEA:
        checks.exact(summary['idea_id'] == IDEA and run['plan_version'] == summary['plan_version'] == '1.1', 'exact126/version')
        return
    checks.exact(run['idea_id'] == summary['idea_id'] == CONTINUATION_IDEA
        and run['plan_version'] == summary['plan_version'] == '1.0'
        and run['scientific_idea_id'] == summary['scientific_idea_id'] == IDEA
        and run['scientific_plan_version'] == summary['scientific_plan_version'] == '1.1', 'exact132/scientific126 provenance')
    source_path = Path(run['failed_run'])
    source = core.read(source_path/'run.json')
    checks.exact(source['idea_id'] == IDEA and source['plan_version'] == '1.1'
        and source['status'] == 'FAILED' and source['exit_code'] == 1
        and source['completed_optimizer_updates'] == 0 and source['completed_arms'] == []
        and source['optimizer_updates_by_arm'] == {arm: 0 for arm in ARMS}
        and 'Too many open files' in source['error'], 'original126 zero-update FD failure')
    checks.exact(Path(summary['failed_run']).resolve() == source_path.resolve()
        and (run_path/'private/cache').resolve() == (source_path/'private/cache').resolve(), '132 immutable cache lineage')
    checks.exact(core.read(Path(run['manifest'])) == core.read(Path(source['manifest']))
        and Path(run['source_system']).resolve() == Path(source['source_system']).resolve(), '132 same manifest/upstream')
    checks.exact(core.read(run_path/'normalizers.json') == core.read(source_path/'normalizers.json'), '132 exact inherited normalizers')
    for key in ('original126_restarted', 'raw_cache_rebuilt', 'normalizers_refitted'):
        checks.exact(run[key] is False, '132 no redo '+key)
    checks.exact(summary['original126_restarted'] is False and summary['normalizers_refitted'] is False
        and run['cache_read_only_reference'] is True and run['archive_read'] is False, '132 permissions')
    checks.value(source['elapsed_seconds'], run['prior_failed_elapsed_seconds'], 'prior_failed_elapsed_seconds', '132 consumed cost')


def audit_run(run_path, checks, check_resources=lambda: None):
    run, summary = (core.read(run_path/p) for p in ('run.json', 'summary.json'))
    checks.exact(run['status'] == summary['status'] == 'COMPLETED' and run['exit_code'] == 0, 'completed126 only')
    inspect_run_identity(run_path, run, summary, checks)
    checks.exact(tuple(run['arms']) == tuple(run['completed_arms']) == ARMS and set(summary['methods']) == set(ARMS), 'all four completed arms')
    checks.exact(summary['final_only'] is True and summary['intermediate_validation'] is False
        and run['validation_forward_passes_per_arm'] == 1 and not run['reference_arrays_read']
        and not run['reference_csv_read'] and not run['test_opened'] and not summary['reference_read'] and not summary['test_opened'], 'declared endpoint/permissions')
    manifest = core.read(Path(run['manifest'])); ids = manifest['splits']['V_select']
    checks.exact(len(ids) == len(set(ids)) == 300 and len(manifest['splits']['T_core']) == 2100, 'formal T2100/V300')
    records, scores = load_validation(run_path, ids, manifest, checks)
    checks.value(sum(len(r['pairs']) for r in records), summary['data_audit']['validation_pair_count'], 'V pairs', 'population')
    checks.value(sum(int(r['labels'].sum()) for r in records), summary['data_audit']['validation_positive_count'], 'V positives', 'population')
    components = sorted({r['component_id'] for r in records})
    checks.exact(len(components) == 38, 'complete38 components')
    largest = max(components, key=lambda c: sum(r['component_id'] == c for r in records))
    thresholds, macros = {}, {}
    for arm in ARMS:
        check_resources()
        result = summary['methods'][arm]
        checks.exact(core.read(run_path/(arm+'-result.json')) == result, 'arm result/public copy')
        theta = summary['threshold_selection'][arm]['threshold']; thresholds[arm] = theta
        checks.exact(theta is None or np.isfinite(theta), 'finite threshold or all reject')
        counts = count_all(records, scores[arm], theta)
        inspect_description(result, list(range(300)), records, scores[arm], counts, theta, checks, arm+'/full')
        checks.exact(set(result['components']) == set(components), 'exact component set')
        for component in components:
            indices = [i for i, r in enumerate(records) if r['component_id'] == component]
            inspect_description(result['components'][component], indices, records, scores[arm], counts, theta, checks, arm+'/component')
        kept = [i for i, r in enumerate(records) if r['component_id'] != largest]
        inspect_description(result['without_largest_component'], kept, records, scores[arm], counts, theta, checks, arm+'/without-largest')
        macros[arm] = aggregate_counts(counts['raw']['all'])['macro_f1']
        compare(dict(macro_f1=macros[arm], scene_count=sum(bool(len(r['pairs'])) for r in records),
                     candidate_count=len(np.unique(np.concatenate(scores[arm])))+1), summary['threshold_selection'][arm], checks, arm+'/selection declaration')
    inspect_pairs(summary['pairwise'], records, scores, thresholds, macros, checks)
    check_resources()
    training = inspect_training(run_path, run, summary, manifest, checks)
    return dict(scene_count=300, component_count=38, arm_count=4, comparison_count=6,
        largest_component_id=largest, excluded_scene_count=sum(r['component_id'] == largest for r in records),
        threshold_optimality_recomputed=False, training_record_audit=training)


def audit(run_path, output):
    start = time.monotonic(); before = resource.getrusage(resource.RUSAGE_SELF); checks = ScoreChecks()
    report = dict(status='RUNNING', started_at=datetime.now(timezone.utc).isoformat(),
        scope='126 scientific recipe or strict132 continuation: final saved scores/raw and paper qualification/counts/metrics, paired changes and training records',
        tolerances=dict(rtol=1e-10, atol=1e-12, integer_counts='exact'),
        not_audited=['raw ZIP/physical correctness, reference truth or unobserved identity errors',
            'feature generation, full T normalizer recomputation, checkpoint-to-score inference or training replay',
            'independent proof each optimizer step actually used the shared plan and initial state',
            'threshold sweep optimality, physical fusion effect, independent test or credible peer qualification'])
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as stream: json.dump(report, stream)
    def check_resources():
        usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform == 'darwin' else 1024)
        checks.exact(usage <= core.RAM, 'RSS within4GiB')
    try:
        if sys.platform.startswith('linux'):
            if hasattr(os, 'sched_getaffinity'): os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:4])
            soft, hard = resource.getrlimit(resource.RLIMIT_AS)
            limit = min([core.MAX_AS]+[v for v in (soft, hard) if v != resource.RLIM_INFINITY])
            resource.setrlimit(resource.RLIMIT_AS, (limit, hard))
        report.update(audit_run(run_path, checks, check_resources), status='PASSED', exit_code=0)
        check_resources()
    except BaseException as error:
        report.update(status='FAILED', exit_code=1, error_type=type(error).__name__, error=str(error))
    finally:
        usage = resource.getrusage(resource.RUSAGE_SELF)
        report.update(exact_checks=checks.exact_checks, numeric_checks=checks.numeric_checks, null_checks=checks.null_checks,
            maximum_absolute_difference=checks.maximum_absolute_difference, elapsed_seconds=time.monotonic()-start,
            user_cpu_seconds=usage.ru_utime-before.ru_utime, system_cpu_seconds=usage.ru_stime-before.ru_stime,
            peak_rss_bytes=int(usage.ru_maxrss*(1 if sys.platform == 'darwin' else 1024)), finished_at=datetime.now(timezone.utc).isoformat())
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--run', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path); args = parser.parse_args()
    def interrupted(signum, frame): raise InterruptedError('independent126 score audit interrupted')
    signal.signal(signal.SIGTERM, interrupted); signal.signal(signal.SIGINT, interrupted)
    result = audit(args.run.resolve(), args.output.resolve()); print(result['status']+': '+str(args.output), flush=True)
    return result['exit_code']


if __name__ == '__main__': raise SystemExit(main())
