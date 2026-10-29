"""One-shot independent saved-array position audit for completed120/121/123/124.

Uses only standard JSON/gzip readers and NumPy. No production evaluator,
predictor, fitting routine, aggregation helper, or reference source is called.
The independent stats/counts/position/pooled core is preserved from the prior
118/119 evidence audit, audit_parallel_saved_arrays.py (2026-10-29). The119
domain-specific checks are deliberately absent from this120/121 audit.
"""
from __future__ import annotations

import argparse
from collections import Counter
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

for thread_setting in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[thread_setting] = '4'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import numpy as np

RADIUS = 6371000.
IDS = ('IDEA-SHARED-COMPONENT-FULL-CONTROL-20261029', 'IDEA-SOURCE-PREFERENCE-CONTROL-20261029',
       'IDEA-SOURCE-PREFERENCE-FULL-CONTROL-20261029', 'IDEA-SOURCE-MIX-SHRINK-CONTROL-20261029',
       'IDEA-NATIVE-CV-STATE-CONTROL-20261029', 'IDEA-CONDITIONAL-SOURCE-VALUE-20261029')
M118 = ('WS_MEAN', 'WS_HUBER', 'WS_FIXED_MEAN', 'WS_FIXED_HUBER',
        'WS_SYMMETRIC_MEAN', 'WS_SYMMETRIC_HUBER', 'WS_ANCHOR', 'WS_NEAREST_SUPPORTED')
P118 = tuple(('WS_MEAN', m) for m in M118[1:]) + (
    ('WS_FIXED_MEAN', 'WS_FIXED_HUBER'), ('WS_HUBER', 'WS_FIXED_HUBER'),
    ('WS_SYMMETRIC_MEAN', 'WS_SYMMETRIC_HUBER'), ('WS_FIXED_MEAN', 'WS_SYMMETRIC_MEAN'),
    ('WS_FIXED_HUBER', 'WS_SYMMETRIC_HUBER')) + tuple(
    (b, m) for b in ('WS_ANCHOR', 'WS_NEAREST_SUPPORTED') for m in M118[1:6])
M121 = M118 + ('WS_LOW_SOURCE_SUPPORTED', 'WS_FIXED_LOW_SOURCE_SUPPORTED')
M124 = M121 + ('WS_SOFT75_SOURCE_SUPPORTED', 'WS_FIXED_SOFT75_SOURCE_SUPPORTED',
              'WS_SHRINK_LOW_SOURCE_SUPPORTED', 'WS_SHRINK_SOFT75_SOURCE_SUPPORTED',
              'WS_FIXED_EQUAL_SOURCE_SUPPORTED')
M127 = ('WS_MEAN', 'WS_LOW_SOURCE_SUPPORTED', 'WS_CV_FORWARD_MEAN',
        'WS_CV_FORWARD_LOW', 'WS_CV_RTS_MEAN', 'WS_CV_RTS_LOW')
M125 = ('WS_MEAN', 'WS_FIXED_MEAN', 'WS_LOW_SOURCE_SUPPORTED', 'WS_FIXED_LOW_SOURCE_SUPPORTED',
        'WS_ANCHOR', 'WS_NEAREST_SUPPORTED', 'WS_FIXED_SOFT75_SOURCE_SUPPORTED',
        'WS_TMEAN_VALUE_GATE', 'WS_LINEAR_VALUE_GATE', 'WS_QUADRATIC_VALUE_GATE', 'WS_SHUFFLED_VALUE_GATE')
COUNTS = ('available_target_count', 'available_prediction_count', 'intersection_count',
          'missing_prediction_count', 'prediction_without_target_count',
          'available_target_trajectory_count', 'covered_target_trajectory_count',
          'missing_target_trajectory_count')
STATS = ('mae', 'mse', 'rmse', 'p95', 'p99')
RAM, MAX_AS = 4*1024**3, 8*1024**3
RATES = ('coverage', 'trajectory_coverage')
POPULATION = ('target_identity_count', 'predicted_group_count', 'duplicate_prediction_group_count',
              'anchor_matched_target_count', 'reference_unavailable_target_count')


def read(path):
    opener = gzip.open if str(path).endswith('.gz') else open
    with opener(path, 'rt', encoding='utf-8') as stream:
        return json.load(stream)


class Checks:
    def __init__(self):
        self.exact_checks = 0
        self.numeric_checks = 0
        self.null_checks = 0
        self.maximum_absolute_difference = {}

    def exact(self, condition, label):
        self.exact_checks += 1
        if not bool(condition):
            raise ValueError('exact check failed: '+label)

    def array(self, actual, expected, label):
        a, b = np.asarray(actual), np.asarray(expected)
        same = a.shape == b.shape and a.dtype.kind == b.dtype.kind
        if same:
            same = np.array_equal(a, b, equal_nan=True) if a.dtype.kind in 'fc' else np.array_equal(a, b)
        self.exact(same, label)

    def value(self, actual, expected, key, label):
        if actual is None or expected is None:
            self.null_checks += 1
            self.exact(actual is expected, label+'/'+key+' null')
        elif isinstance(actual, (int, np.integer)):
            self.exact(actual == expected, label+'/'+key)
        else:
            self.numeric_checks += 1
            difference = abs(float(actual)-float(expected))
            self.maximum_absolute_difference[key] = max(self.maximum_absolute_difference.get(key, 0.), difference)
            atol = .001 if key == 'mse' else 1e-12 if key in RATES else 1e-5
            if not np.isfinite(expected) or not np.isclose(actual, expected, rtol=1e-9, atol=atol):
                raise ValueError('numeric check failed: '+label+'/'+key)

    def fields(self, actual, expected, keys, label):
        for key in keys:
            self.exact(key in expected and key in actual, label+'/'+key+' present')
            self.value(actual[key], expected[key], key, label)


def stats(values):
    n = len(values)
    return dict(point_count=n, mae=float(values.mean()) if n else None, mse=float(np.mean(values**2)) if n else None,
        rmse=float(np.sqrt(np.mean(values**2))) if n else None,
        p95=float(np.quantile(values, .95)) if n else None, p99=float(np.quantile(values, .99)) if n else None)

def counts(truth, prediction):
    use = truth & prediction; nt, ni = int(truth.sum()), int(use.sum())
    tt, ti = int(truth.any(1).sum()), int(use.any(1).sum())
    return dict(zip(COUNTS, (nt, int(prediction.sum()), ni, nt-ni, int((prediction & ~truth).sum()), tt, ti, tt-ti)))

def position(prediction, truth, use):
    def xyz(p):
        lat, lon = np.deg2rad(p[:, 0]), np.deg2rad(p[:, 1])
        return np.column_stack((np.cos(lat)*np.cos(lon), np.cos(lat)*np.sin(lon), np.sin(lat)))
    a, b = xyz(prediction[use, :2]), xyz(truth[use, :2]); result = np.full(use.shape, np.nan)
    result[use] = 6371000.*np.arctan2(np.linalg.norm(np.cross(a, b), axis=1), np.sum(a*b, axis=1))
    if not np.isfinite(result[use]).all(): raise ValueError('nonfinite valid independent position error')
    return result

def pooled(parts, *, scene_level=False):
    c = {k: sum(x[k] for x, _ in parts) for k in COUNTS}; values = np.concatenate([v for _, v in parts])
    s = stats(values); del s['point_count']; c.update(s)
    c.update(coverage=c['intersection_count']/c['available_target_count'] if c['available_target_count'] else None,
        trajectory_coverage=c['covered_target_trajectory_count']/c['available_target_trajectory_count'] if c['available_target_trajectory_count'] else None)
    if scene_level:
        defined = [float(v.mean()) for _, v in parts if len(v)]
        c.update(scene_macro_mae=float(np.mean(defined)) if defined else None,
                 defined_mae_scene_count=len(defined), scene_count=len(parts))
    return c

def statistics(values):
    values = np.asarray(values, np.float64)
    if not np.isfinite(values).all():
        raise ValueError('nonfinite valid position error')
    result = stats(values)
    del result['point_count']
    return result


def metrics(target_available, prediction_available, errors):
    use = target_available & prediction_available
    result = counts(target_available, prediction_available)
    values = errors[use]
    result.update(statistics(values),
        coverage=result['intersection_count']/result['available_target_count'] if result['available_target_count'] else None,
        trajectory_coverage=result['covered_target_trajectory_count']/result['available_target_trajectory_count']
            if result['available_target_trajectory_count'] else None)
    return result, values


def artifacts(path, names):
    with np.load(path, allow_pickle=False) as payload:
        prefix = 'WS_MEAN_'
        keys = {key[len(prefix):] for key in payload.files if key.startswith(prefix)}
        if 'aligned_prediction' not in keys:
            raise ValueError('saved complete artifact required')
        return {name: {key: payload[name+'_'+key] for key in keys} for name in names}


def inspect_scene(saved, report, names, checks):
    base = saved[names[0]]
    checks.exact(str(base['split']) == 'V_select' and str(base['grouping_kind']) == 'predicted'
                 and str(base['correspondence_policy']) == 'anchor_identity_first', 'development-only artifact policy')
    t, g, n = len(base['target_ids']), len(base['group_states']), len(base['node_group_indices'])
    checks.exact(len(set(base['target_ids'].tolist())) == t, 'unique targets')
    offsets, flat, anchors = base['group_offsets'], base['group_node_indices'], base['group_anchor_target_indices']
    checks.exact(offsets.shape == (g+1,) and offsets[0] == 0 and offsets[-1] == n
                 and np.all(np.diff(offsets) > 0), 'complete group offsets')
    checks.array(np.sort(flat), np.arange(n), 'complete node partition')
    membership = np.empty(n, np.int64)
    for group in range(g):
        members = flat[offsets[group]:offsets[group+1]]
        checks.exact(np.all(np.diff(members) > 0), 'member order')
        membership[members] = group
    checks.array(membership, base['node_group_indices'], 'membership binding')
    checks.exact(anchors.shape == (g,) and np.all((anchors >= 0) & (anchors < t)), 'anchor indices')
    first = {}
    duplicates = []
    for group, target in enumerate(anchors):
        if int(target) in first:
            duplicates.append(group)
        else:
            first[int(target)] = group
    aligned = np.full(t+len(duplicates), -1, np.int64)
    for target, group in first.items():
        aligned[target] = group
    aligned[t:] = duplicates
    checks.array(aligned, base['aligned_group_indices'], 'first representative and duplicate rows')
    checks.exact(base['aligned_prediction'].shape == (len(aligned), 64, 4), 'complete Q64 grid')
    checks.exact(not base['aligned_truth_valid'][t:].any(), 'duplicates have no reference')
    checks.array(base['aligned_target_ids'][:t], base['target_ids'], 'aligned target registry')
    checks.exact(np.all(base['aligned_target_ids'][t:] == ''), 'duplicate target absence')
    population = dict(zip(POPULATION, (t, g, len(duplicates), len(first),
        int((~base['aligned_truth_valid'][:t].any((1, 2))).sum()))))
    outcomes = {}
    for name in names:
        art = saved[name]
        checks.exact(set(art) == set(base), 'same artifact schema')
        for key in base:
            if key not in ('aligned_prediction', 'group_states'):
                checks.array(art[key], base[key], 'fixed '+key)
        for row, group in enumerate(aligned):
            if group >= 0:
                checks.array(art['aligned_prediction'][row], art['group_states'][group], 'aligned group states')
                checks.array(art['aligned_prediction_valid'][row], art['group_valid'][group], 'aligned valid states')
            else:
                checks.exact(not art['aligned_prediction_valid'][row].any(), 'unmatched prediction unavailable')
        truth_valid, pred_valid = art['aligned_truth_valid'], art['aligned_prediction_valid']
        checks.exact(truth_valid.dtype == bool and pred_valid.dtype == bool
                     and art['aligned_prediction'].shape == art['aligned_truth'].shape == truth_valid.shape == pred_valid.shape
                     and art['group_states'].shape == art['group_valid'].shape == (g, 64, 4)
                     and art['group_valid'].dtype == bool, 'complete values and boolean validity')
        tv, pv = truth_valid[..., :2].all(-1), pred_valid[..., :2].all(-1)
        use = tv & pv
        pred, truth = art['aligned_prediction'][use], art['aligned_truth'][use]
        checks.exact(np.isfinite(pred[:, :2]).all() and np.isfinite(truth[:, :2]).all(), 'finite valid positions')
        checks.exact(np.all(np.abs(pred[:, 0]) <= 90.) and np.all(np.abs(truth[:, 0]) <= 90.)
                     and np.all(np.abs(pred[:, 1]) <= 180.) and np.all(np.abs(truth[:, 1]) <= 180.), 'valid degree range')
        errors = position(art['aligned_prediction'], art['aligned_truth'], use)
        scene_metrics = {}
        for region, mask in (('all', np.ones(tv.shape, bool)), ('observed', art['aligned_observed']),
                             ('missing', ~art['aligned_observed'])):
            region_stats, values = metrics(tv & mask, pv & mask, errors)
            checks.fields(region_stats, report['methods'][name]['position'][region], COUNTS+RATES+STATS, 'scene position '+region)
            if region == 'all':
                full_metrics, full_values = region_stats, values
        common_metrics, common_values = metrics(tv[:t], pv[:t], errors[:t])
        checks.fields(population, report['methods'][name]['population'], POPULATION, 'scene population')
        checks.value(t-len(first), report['methods'][name]['unmatched_target_count'], 'unmatched_target_count', 'scene population')
        checks.exact(np.array_equal(full_values, common_values), 'no duplicate errors in main population')
        outcomes[name] = dict(position=full_metrics, common=common_metrics, errors=common_values,
                              population=population, unmatched_target_count=t-len(first))
    for key, pair in report['pairwise'].items():
        left, right = pair['baseline_method'], pair['candidate_method']
        common = (saved[left]['aligned_prediction_valid'][:t, :, :2].all(-1)
                  & saved[right]['aligned_prediction_valid'][:t, :, :2].all(-1))
        checks.array(common, base['aligned_prediction_valid'][:t, :, :2].all(-1), 'pairwise common mask')
        for side, name in (('A0', left), ('candidate', right)):
            checks.fields(outcomes[name]['common'], pair['common_valid_position'][side],
                          COUNTS+RATES+('mae', 'mse', 'rmse'), 'scene pair position')
            checks.fields(dict(point_count=len(outcomes[name]['errors']), **statistics(outcomes[name]['errors'])),
                          pair['strata']['all'][side], ('point_count',)+STATS, 'scene pair pooled tails')
        delta = outcomes[right]['errors']-outcomes[left]['errors']
        checks.value(float(np.mean(delta)) if len(delta) else None,
                     pair['strata']['all']['candidate_minus_A0']['mean_error_change'], 'mean_error_change', 'scene pair delta')
        for transition in ('gained_target_points', 'lost_target_points', 'rescued_target_count', 'lost_target_count'):
            checks.value(0, pair['coverage_transitions'][transition], transition, 'scene fixed population')
    return outcomes


def pooled_methods(items, name, *, common=False):
    rows = [item[name]['common' if common else 'position'] for item in items]
    return pooled([(row, item[name]['errors']) for row, item in zip(rows, items)], scene_level=True)


def inspect_summary(report, indices, results, rows, names, checks, label):
    if not indices:
        checks.exact(report is None, label+' empty subset remains null')
        return
    chosen = [results[i] for i in indices]
    checks.value(len(indices), report['scene_count'], 'scene_count', label)
    checks.exact(report['scene_ids'] == [rows[i]['scene_id'] for i in indices], label+' scene order')
    checks.exact(set(report['methods']) == set(names) and set(report['pairwise']) == set(rows[0]['pairwise']), label+' complete arms and pairs')
    pool_cache = {(name, common): pooled_methods(chosen, name, common=common)
                  for name in names for common in (False, True)}
    for name in names:
        checks.fields(pool_cache[name, False], report['methods'][name]['position'],
                      COUNTS+RATES+STATS+('scene_count', 'defined_mae_scene_count', 'scene_macro_mae'), label+' position')
        population = {key: sum(item[name]['population'][key] for item in chosen) for key in POPULATION}
        checks.fields(population, report['methods'][name]['population'], POPULATION, label+' population')
        checks.value(sum(item[name]['unmatched_target_count'] for item in chosen),
            report['methods'][name]['unmatched_target_count'], 'unmatched_target_count', label+' population')
    for pair in report['pairwise'].values():
        for side, name in (('A0', pair['baseline_method']), ('candidate', pair['candidate_method'])):
            pooled_common = pool_cache[name, True]
            checks.fields(pooled_common, pair['common_valid_position'][side],
                COUNTS+RATES+STATS+('scene_count', 'defined_mae_scene_count', 'scene_macro_mae'), label+' pair position')
            checks.fields(dict(point_count=pooled_common['intersection_count'],
                               **{key: pooled_common[key] for key in STATS}),
                          pair['strata']['all'][side], ('point_count',)+STATS, label+' pair tails')
        for transition in ('gained_target_points', 'lost_target_points', 'rescued_target_count', 'lost_target_count'):
            checks.value(0, pair['coverage_transitions'][transition], transition, label+' fixed population')


def audit_run(run_path, checks):
    run, summary = (read(run_path/name) for name in ('run.json', 'summary.json'))
    checks.exact(run.get('status') == summary.get('status') == 'COMPLETED' and run.get('exit_code') == 0, 'completed run')
    checks.exact(run['idea_id'] in IDS and run['idea_id'] == summary['idea_id'], 'supported completed source control')
    checks.exact(run.get('plan_version') == summary.get('plan_version') == '1.0', 'declared plan version')
    names = run['methods']
    expected_names = (M118 if run['idea_id'] == IDS[0] else M124 if run['idea_id'] == IDS[3]
                      else M127 if run['idea_id'] == IDS[4] else M125 if run['idea_id'] == IDS[5] else M121)
    checks.exact(tuple(names) == expected_names, 'complete declared methods')
    required_pairs = P118 if run['idea_id'] == IDS[0] else tuple(combinations(expected_names, 2))
    required_keys = {candidate+'_vs_'+base for base, candidate in required_pairs}
    rows = read(run_path/'scene-summary.json.gz')
    checks.exact(run['scene_ids'] == summary['scene_ids'] == [row['scene_id'] for row in rows], 'complete scene order')
    checks.exact(run['completed_scenes'] == len(rows) and len({row['scene_id'] for row in rows}) == len(rows), 'complete scene population')
    pilot_ids = set(run.get('original32_ids', []))
    pilot_path = Path(run['pilot_run']) if run['idea_id'] in (IDS[0], IDS[2]) else None
    if pilot_path is not None:
        pilot_run = read(pilot_path/'run.json')
        checks.exact(pilot_run['status'] == 'COMPLETED' and set(pilot_run['scene_ids']) == pilot_ids, 'immutable pilot original32 binding')
    results = []
    for row in rows:
        sid = row['scene_id']
        checks.exact(read(run_path/'scenes'/f'scene-{sid}.json.gz') == row, 'public per-scene copy')
        checks.exact(set(row['methods']) == set(names) and set(row['pairwise']) == required_keys, 'complete scene arms/pairs')
        for base, candidate in required_pairs:
            pair = row['pairwise'][candidate+'_vs_'+base]
            checks.exact(pair['baseline_method'] == base and pair['candidate_method'] == candidate, 'pair orientation')
        source = pilot_path if sid in pilot_ids else run_path
        saved = artifacts(source/'private'/f'states-{sid}.npz', names)
        results.append(inspect_scene(saved, row, names, checks))
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform == 'darwin' else 1024)
        checks.exact(rss <= RAM, 'resident memory within4GiB')
    components = sorted({row['component_id'] for row in rows})
    checks.exact(set(summary['components']) == set(components), 'complete components')
    inspect_summary(summary, list(range(len(rows))), results, rows, names, checks, 'full')
    for number, component in enumerate(components):
        inspect_summary(summary['components'][component], [i for i,row in enumerate(rows) if row['component_id'] == component],
                        results, rows, names, checks, 'component-'+str(number))
    subsets = 0
    if pilot_path is not None:
        pilot_components = {row['component_id'] for row in rows if row['scene_id'] in pilot_ids}
        for row in rows:
            checks.exact(row['stratum'] == ('original32' if row['scene_id'] in pilot_ids else 'remaining268'), 'original/new classification')
            checks.exact(row['component_seen_in_original32'] == (row['component_id'] in pilot_components), 'component seen classification')
        counts = Counter(row['component_id'] for row in rows)
        largest = max(components, key=lambda component: counts[component])
        selections = dict(original32=[i for i,row in enumerate(rows) if row['scene_id'] in pilot_ids],
            remaining268=[i for i,row in enumerate(rows) if row['scene_id'] not in pilot_ids],
            remaining268_seen_components=[i for i,row in enumerate(rows) if row['scene_id'] not in pilot_ids and row['component_id'] in pilot_components],
            remaining268_unseen_components=[i for i,row in enumerate(rows) if row['scene_id'] not in pilot_ids and row['component_id'] not in pilot_components],
            without_largest_component=[i for i,row in enumerate(rows) if row['component_id'] != largest])
        for label, indices in selections.items():
            inspect_summary(summary[label], indices, results, rows, names, checks, label)
        subsets = len(selections)
    elif run['idea_id'] in (IDS[3], IDS[4], IDS[5]):
        counts = Counter(row['component_id'] for row in rows)
        largest = max(components, key=lambda component: counts[component])
        indices = [i for i, row in enumerate(rows) if row['component_id'] != largest]
        if indices:
            inspect_summary(summary['without_largest_component'], indices, results, rows, names, checks, 'without_largest_component')
            subsets = 1
    return dict(scene_count=len(rows), component_count=len(components), method_count=len(names),
                pair_count_per_scene=len(rows[0]['pairwise']), imported_original32_scene_count=len(pilot_ids),
                aggregate_subset_count=1+len(components)+subsets)


def audit(run_path, output):
    started = time.monotonic()
    cpu_before = resource.getrusage(resource.RUSAGE_SELF)
    checks = Checks()
    report = dict(status='RUNNING', started_at=datetime.now(timezone.utc).isoformat(),
        scope='independent NumPy saved-array main position/population audit for120/121/123/124',
        earth_radius_metres=RADIUS,
        tolerances=dict(rtol=1e-9, metres_atol=1e-5, mse_atol=.001, rate_atol=1e-12),
        not_audited=['shared center fitting/folds/Huber mechanism or physical bias',
            'manifest/true component identity definitions or reference physical correctness',
            'input support/time/observed semantics beyond stored masks',
            'support-conditioned strata, source-preference routing, paper metrics, speed or heading',
            'per-point benefit/harm counts, training, credible peers or generalization claims'])
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream)
    try:
        if sys.platform.startswith('linux'):
            if hasattr(os, 'sched_getaffinity'):
                os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:4])
            soft, hard = resource.getrlimit(resource.RLIMIT_AS)
            limit = min([MAX_AS]+[value for value in (soft, hard) if value != resource.RLIM_INFINITY])
            resource.setrlimit(resource.RLIMIT_AS, (limit, hard))
        report.update(audit_run(run_path, checks), status='COMPLETED', exit_code=0)
    except BaseException as error:
        report.update(status='FAILED', exit_code=1, error_type=type(error).__name__,
                      error=str(error) if isinstance(error, ValueError) else 'audit could not complete; preserve original run')
    finally:
        usage = resource.getrusage(resource.RUSAGE_SELF)
        report.update(exact_checks=checks.exact_checks, numeric_checks=checks.numeric_checks,
            null_checks=checks.null_checks, maximum_absolute_difference=checks.maximum_absolute_difference,
            elapsed_seconds=time.monotonic()-started,
            user_cpu_seconds=usage.ru_utime-cpu_before.ru_utime,
            system_cpu_seconds=usage.ru_stime-cpu_before.ru_stime,
            peak_rss_bytes=int(usage.ru_maxrss*(1 if sys.platform == 'darwin' else 1024)),
            finished_at=datetime.now(timezone.utc).isoformat())
        with output.open('w', encoding='utf-8') as stream:
            json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    def interrupted(signum, frame):
        raise InterruptedError('audit interrupted')
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    result = audit(args.run.resolve(), args.output.resolve())
    print(result['status']+': '+str(args.output))
    return result['exit_code']


if __name__ == '__main__':
    raise SystemExit(main())
