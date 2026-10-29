"""129 independent saved-array audit with variable partitions and coverage.

Only NumPy/JSON plus the prior independent geometric/counting primitives are
used. No production predictor, evaluator, grouping or aggregation is imported.
Each arm is audited on its own population; each pair uses its true common mask.
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
import numpy as np

IDEA = 'IDEA-OBJECT-HYPOTHESIS-PILOT-20261029'
METHODS = ('WS_MEAN', 'WS_LOW_SOURCE_SUPPORTED', 'NO_REJECT_MEAN', 'NO_REJECT_LOW',
           'T_REJECT_MEAN', 'T_REJECT_LOW')
PAIRS = tuple(combinations(METHODS, 2))
PAIR_KEYS = {candidate+'_vs_'+base for base, candidate in PAIRS}
TRANSITIONS = ('gained_target_points', 'lost_target_points', 'rescued_target_count', 'lost_target_count')
PARTITIONS = dict(WS='WS_MEAN', NO_REJECT='NO_REJECT_MEAN', T_REJECT='T_REJECT_MEAN')


def representatives(artifact, t):
    first = artifact['group_node_indices'][artifact['group_offsets'][:-1]]
    group = artifact['aligned_group_indices'][:t]
    result = np.full(t, -1, np.int64)
    use = group >= 0
    result[use] = first[group[use]]
    return result


def inspect_variable_scene(saved, row, checks):
    checks.exact(set(saved) == set(METHODS) and set(row['methods']) == set(METHODS)
                 and set(row['pairwise']) == PAIR_KEYS, 'complete six arms/fifteen pairs')
    base = saved[METHODS[0]]
    t = len(base['target_ids']); n = len(base['node_group_indices'])
    outcomes, target = {}, {}
    for name in METHODS:
        artifact = saved[name]
        checks.exact(len(artifact['node_group_indices']) == n, 'same original node population')
        for key in ('target_ids',):
            checks.array(artifact[key], base[key], 'cross-arm target registry')
        for key in ('aligned_truth', 'aligned_truth_valid', 'aligned_observed'):
            checks.array(artifact[key][:t], base[key][:t], 'cross-arm firstT '+key)
        # A one-arm call checks that arm's independent grouping/duplicates and
        # valid/population metrics; it does not impose fixed-R across arms.
        outcomes[name] = core.inspect_scene({name: artifact},
            dict(methods={name: row['methods'][name]}, pairwise={}), [name], checks)[name]
        tv = artifact['aligned_truth_valid'][:t, :, :2].all(-1)
        pv = artifact['aligned_prediction_valid'][:t, :, :2].all(-1)
        errors = core.position(artifact['aligned_prediction'][:t], artifact['aligned_truth'][:t], tv & pv)
        target[name] = dict(tv=tv, pv=pv, errors=errors, representative=representatives(artifact, t))
    pair_results = {}
    for left, right in PAIRS:
        key = right+'_vs_'+left; public = row['pairwise'][key]
        checks.exact(public['baseline_method'] == left and public['candidate_method'] == right, 'pair direction')
        a, b = target[left], target[right]
        common_prediction = a['pv'] & b['pv']
        common_use = a['tv'] & common_prediction
        old_use, new_use = a['tv'] & a['pv'], b['tv'] & b['pv']
        gain, loss = new_use & ~old_use, old_use & ~new_use
        transitions = dict(zip(TRANSITIONS, (int(gain.sum()), int(loss.sum()),
            int((new_use.any(1) & ~old_use.any(1)).sum()), int((old_use.any(1) & ~new_use.any(1)).sum()))))
        side_results = {}
        for side, item in (('A0', a), ('candidate', b)):
            metric, values = core.metrics(item['tv'], common_prediction, item['errors'])
            core_keys = core.COUNTS+core.RATES+('mae', 'mse', 'rmse')
            checks.fields(metric, public['common_valid_position'][side], core_keys, 'scene true-common '+side)
            checks.fields(dict(point_count=len(values), **core.statistics(values)),
                public['strata']['all'][side], ('point_count',)+core.STATS, 'scene common tails '+side)
            side_results[side] = (metric, values)
        checks.fields(transitions, public['coverage_transitions'], TRANSITIONS, 'scene coverage transitions')
        checks.fields(core.stats(b['errors'][gain]), public['newly_covered_position'],
                      ('point_count',)+core.STATS, 'newly covered errors')
        checks.fields(core.stats(a['errors'][loss]), public['lost_coverage_position'],
                      ('point_count',)+core.STATS, 'lost coverage errors')
        represented = (a['representative'] >= 0) & (b['representative'] >= 0)
        same = represented & (a['representative'] == b['representative'])
        changed = represented & ~same
        checks.value(int(same.sum()), public['same_anchor_target_count'], 'same_anchor_target_count', key)
        checks.value(int(changed.sum()), public['changed_anchor_target_count'], 'changed_anchor_target_count', key)
        for region, mask in (('same_anchor', same), ('changed_anchor', changed)):
            use = common_use & mask[:, None]
            for side, item in (('A0', a), ('candidate', b)):
                checks.fields(core.stats(item['errors'][use]), public['strata'][region][side],
                              ('point_count',)+core.STATS, 'scene '+region+' '+side)
        delta = b['errors'][common_use]-a['errors'][common_use]
        checks.value(float(delta.mean()) if len(delta) else None,
            public['strata']['all']['candidate_minus_A0']['mean_error_change'], 'mean_error_change', key)
        pair_results[key] = dict(sides=side_results, transitions=transitions, delta=delta)
    return dict(methods=outcomes, pairs=pair_results)


def inspect_variable_summary(report, indices, results, rows, checks, label):
    if not indices:
        checks.exact(report is None, label+' empty subset null')
        return
    selected = [results[i] for i in indices]
    checks.value(len(indices), report['scene_count'], 'scene_count', label)
    checks.exact(report['scene_ids'] == [rows[i]['scene_id'] for i in indices], label+' scene order')
    checks.exact(set(report['methods']) == set(METHODS) and set(report['pairwise']) == PAIR_KEYS,
                 label+' complete methods/pairs')
    for name in METHODS:
        items = [item['methods'][name] for item in selected]
        metric = core.pooled([(item['position'], item['errors']) for item in items], scene_level=True)
        checks.fields(metric, report['methods'][name]['position'], core.COUNTS+core.RATES+core.STATS+
                      ('scene_count', 'defined_mae_scene_count', 'scene_macro_mae'), label+' own '+name)
        population = {k: sum(item['population'][k] for item in items) for k in core.POPULATION}
        checks.fields(population, report['methods'][name]['population'], core.POPULATION, label+' population '+name)
        checks.value(sum(item['unmatched_target_count'] for item in items),
            report['methods'][name]['unmatched_target_count'], 'unmatched_target_count', label+' '+name)
    for key in PAIR_KEYS:
        pairs = [item['pairs'][key] for item in selected]
        public = report['pairwise'][key]
        left, right = key.split('_vs_')[1], key.split('_vs_')[0]
        checks.exact(public['baseline_method'] == left and public['candidate_method'] == right, label+' pair direction')
        for side in ('A0', 'candidate'):
            pooled = core.pooled([item['sides'][side] for item in pairs], scene_level=True)
            checks.fields(pooled, public['common_valid_position'][side], core.COUNTS+core.RATES+core.STATS+
                          ('scene_count', 'defined_mae_scene_count', 'scene_macro_mae'), label+' true-common '+side)
            values = np.concatenate([item['sides'][side][1] for item in pairs])
            checks.fields(core.stats(values), public['strata']['all'][side], ('point_count',)+core.STATS,
                          label+' common tails '+side)
        transitions = {k: sum(item['transitions'][k] for item in pairs) for k in TRANSITIONS}
        checks.fields(transitions, public['coverage_transitions'], TRANSITIONS, label+' coverage transitions')
        delta = np.concatenate([item['delta'] for item in pairs])
        checks.value(float(delta.mean()) if len(delta) else None,
            public['strata']['all']['candidate_minus_A0']['mean_error_change'], 'mean_error_change', label+' '+key)
        for index, item in zip(indices, pairs):
            sid = rows[index]['scene_id']; values = item['delta']
            checks.value(float(values.mean()) if len(values) else None, public['scene_mae_changes'][sid],
                         'scene_mae_change', label+' '+key)


def binary_metrics(labels, decisions):
    labels, decisions = np.asarray(labels, bool), np.asarray(decisions, bool)
    tp = int((labels & decisions).sum()); fp = int((~labels & decisions).sum())
    fn = int((labels & ~decisions).sum()); tn = int((~labels & ~decisions).sum())
    return dict(pair_count=len(labels), positive_count=int(labels.sum()), tp=tp, fp=fp, fn=fn, tn=tn,
        f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 1.)


def inspect_decoded_association(saved, row, score_cache, checks):
    sid = row['scene_id']; pairs = score_cache[sid+'_pairs']; sources = score_cache[sid+'_sources']
    labels = score_cache[sid+'_labels']; nodes = score_cache[sid+'_node_ids']
    checks.exact(np.isin(labels, [0, 1]).all(), 'binary source labels')
    checks.array(pairs, np.column_stack(np.triu_indices(len(nodes), 1)), 'complete original source pair order')
    same = sources[pairs[:, 0]] == sources[pairs[:, 1]]
    outcome = {}
    for partition, method in PARTITIONS.items():
        membership = saved[method]['node_group_indices']
        checks.exact(len(membership) == len(nodes), 'decoded association node count')
        partner = 'WS_LOW_SOURCE_SUPPORTED' if partition == 'WS' else partition+'_LOW'
        checks.array(saved[partner]['node_group_indices'], membership, 'same partition for Mean/LOW')
        accepted = membership[pairs[:, 0]] == membership[pairs[:, 1]]
        outcome[partition] = {}
        for scope, mask in (('all', np.ones(len(pairs), bool)), ('itsa', same), ('mstsa', ~same)):
            metric = binary_metrics(labels[mask], accepted[mask])
            public = row['association'][partition] if scope == 'all' else row['association'][partition][scope]
            checks.fields(metric, public['per_scene'][0], ('pair_count', 'positive_count', 'tp', 'fp', 'fn', 'tn'),
                          'decoded '+partition+'/'+scope)
            checks.value(metric['f1'] if metric['pair_count'] else None, public['macro_f1'], 'macro_f1',
                         'decoded '+partition+'/'+scope)
            outcome[partition][scope] = metric
    return outcome


def inspect_decoded_summary(report, indices, decoded, checks, label):
    for partition in PARTITIONS:
        for scope in ('all', 'itsa', 'mstsa'):
            items = [decoded[i][partition][scope] for i in indices]
            eligible = [item for item in items if item['pair_count']]
            totals = {k: sum(item[k] for item in items) for k in ('pair_count', 'tp', 'fp', 'fn', 'tn')}
            public = report['association'][partition] if scope == 'all' else report['association'][partition][scope]
            checks.fields(totals, public, tuple(totals), label+' decoded '+partition+'/'+scope)
            checks.value(float(np.mean([item['f1'] for item in eligible])) if eligible else None,
                         public['macro_f1'], 'macro_f1', label+' decoded '+partition+'/'+scope)
            denominator = 2*totals['tp']+totals['fp']+totals['fn']
            micro = 2*totals['tp']/denominator if denominator else 1. if eligible else None
            checks.value(micro, public['micro_f1'], 'micro_f1', label+' decoded '+partition+'/'+scope)


def audit_run(run_path, checks):
    run, summary = (core.read(run_path/name) for name in ('run.json', 'summary.json'))
    checks.exact(run.get('status') == summary.get('status') == 'COMPLETED' and run.get('exit_code') == 0,
                 'completed129 only')
    checks.exact(run.get('idea_id') == summary.get('idea_id') == IDEA
                 and run.get('plan_version') == summary.get('plan_version') == '1.0', 'exact129 scope/version')
    checks.exact(tuple(run['methods']) == METHODS, 'declared six methods')
    rows = core.read(run_path/'scene-summary.json.gz')
    checks.exact(len(rows) == 32 and len({row['scene_id'] for row in rows}) == 32, 'complete fixed32')
    checks.exact(run['scene_ids'] == summary['scene_ids'] == [row['scene_id'] for row in rows]
                 and run['completed_scenes'] == 32, 'complete ordered scene population')
    results, decoded = [], []
    score_path = Path(run['source_system']).parent/'association'/'final-scores.npz'
    score_cache = np.load(score_path, allow_pickle=False) if score_path.exists() else None
    try:
        for row in rows:
            sid = row['scene_id']
            checks.exact(core.read(run_path/'scenes'/f'scene-{sid}.json.gz') == row, 'public scene copy')
            saved = core.artifacts(run_path/'private'/f'states-{sid}.npz', METHODS)
            results.append(inspect_variable_scene(saved, row, checks))
            if score_cache is not None:
                decoded.append(inspect_decoded_association(saved, row, score_cache, checks))
            usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform == 'darwin' else 1024)
            checks.exact(usage <= core.RAM, 'RSS within4GiB')
    finally:
        if score_cache is not None:
            score_cache.close()
    components = sorted({row['component_id'] for row in rows})
    checks.exact(len(components) == 9 and set(summary['components']) == set(components), 'complete nine components')
    selections = [('full', summary, list(range(len(rows))))]
    selections += [('component-'+str(i), summary['components'][c],
                    [j for j, row in enumerate(rows) if row['component_id'] == c]) for i, c in enumerate(components)]
    sizes = Counter(row['component_id'] for row in rows)
    largest = max(components, key=lambda c: sizes[c])
    selections.append(('without_largest_component', summary['without_largest_component'],
                       [i for i, row in enumerate(rows) if row['component_id'] != largest]))
    for label, report, indices in selections:
        inspect_variable_summary(report, indices, results, rows, checks, label)
        if decoded:
            inspect_decoded_summary(report, indices, decoded, checks, label)
    return dict(scene_count=len(rows), component_count=len(components), method_count=len(METHODS),
        pair_count_per_scene=len(PAIRS), aggregate_subset_count=len(selections),
        decoded_association_audited=bool(decoded), decoded_partition_count=len(PARTITIONS) if decoded else 0,
        association_scope='full all-pair/same-source/cross-source decoded counts and macro/microF1 only; RAW not audited')


def audit(run_path, output):
    started = time.monotonic(); before = resource.getrusage(resource.RUSAGE_SELF)
    checks = core.Checks()
    report = dict(status='RUNNING', started_at=datetime.now(timezone.utc).isoformat(),
        scope='129 independent variable-partition main-position/population/true-common/coverage audit',
        earth_radius_metres=core.RADIUS,
        tolerances=dict(rtol=1e-9, metres_atol=1e-5, mse_atol=.001, rate_atol=1e-12),
        not_audited=['native CV/event/conflict assumptions, candidate construction or T rejection fit',
            'reference clock/physical correctness or identity truth semantics',
            'source contribution routes, speed/heading, paper coordinate metrics or support-conditioned strata',
            'pointwise benefit/harm classification counts and signed sums, RAW056 neural score/threshold validity',
            'author reproduction, independent generalization or credible peer claims'])
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream)
    try:
        if sys.platform.startswith('linux'):
            if hasattr(os, 'sched_getaffinity'):
                os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:4])
            soft, hard = resource.getrlimit(resource.RLIMIT_AS)
            limit = min([core.MAX_AS]+[value for value in (soft, hard) if value != resource.RLIM_INFINITY])
            resource.setrlimit(resource.RLIMIT_AS, (limit, hard))
        report.update(audit_run(run_path, checks), status='COMPLETED', exit_code=0)
    except BaseException as error:
        report.update(status='FAILED', exit_code=1, error_type=type(error).__name__,
                      error=str(error) if isinstance(error, ValueError) else 'audit did not complete; preserve source run')
    finally:
        usage = resource.getrusage(resource.RUSAGE_SELF)
        report.update(exact_checks=checks.exact_checks, numeric_checks=checks.numeric_checks,
            null_checks=checks.null_checks, maximum_absolute_difference=checks.maximum_absolute_difference,
            elapsed_seconds=time.monotonic()-started, user_cpu_seconds=usage.ru_utime-before.ru_utime,
            system_cpu_seconds=usage.ru_stime-before.ru_stime,
            peak_rss_bytes=int(usage.ru_maxrss*(1 if sys.platform == 'darwin' else 1024)),
            finished_at=datetime.now(timezone.utc).isoformat())
        with output.open('w', encoding='utf-8') as stream:
            json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True); parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    def interrupted(signum, frame):
        raise InterruptedError('independent object output audit interrupted')
    signal.signal(signal.SIGTERM, interrupted); signal.signal(signal.SIGINT, interrupted)
    result = audit(args.run.resolve(), args.output.resolve())
    print(result['status']+': '+str(args.output), flush=True)
    return result['exit_code']


if __name__ == '__main__':
    raise SystemExit(main())
