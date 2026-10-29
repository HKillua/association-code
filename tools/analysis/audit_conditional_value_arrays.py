"""Independent saved-array main-position audit for completed125 only.

125 records completed_T_scenes/completed_V_scenes rather than completed_scenes.
This separate entry point does not modify the failed generic audit or its output.
It reuses only independent NumPy geometry/counting primitives, never production
predictors, evaluators, policy heads or aggregation. All eleven arms and all55
pairs retain the fixed-R population, grouping, first representative and validity.
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

IDEA = 'IDEA-CONDITIONAL-SOURCE-VALUE-20261029'
METHODS = core.M125
PAIRS = tuple(combinations(METHODS, 2))
PAIR_KEYS = {candidate+'_vs_'+base for base, candidate in PAIRS}
IMPORTED = METHODS[:7]
TRANSITIONS = ('gained_target_points', 'lost_target_points', 'rescued_target_count', 'lost_target_count')


def metadata(run_path, checks):
    run, summary = (core.read(run_path/name) for name in ('run.json', 'summary.json'))
    checks.exact(run.get('status') == summary.get('status') == 'COMPLETED' and run.get('exit_code') == 0,
                 'completed125 only')
    checks.exact(run.get('idea_id') == summary.get('idea_id') == IDEA
                 and run.get('plan_version') == summary.get('plan_version') == '1.0', 'exact125 scope/version')
    checks.exact(tuple(run['methods']) == METHODS, 'complete declared eleven methods')
    rows = core.read(run_path/'scene-summary.json.gz')
    checks.exact(len(rows) == 32 and len({row['scene_id'] for row in rows}) == 32, 'complete fixed V32')
    checks.exact(run['scene_ids'] == summary['scene_ids'] == [row['scene_id'] for row in rows],
                 'complete ordered V scene population')
    checks.exact(run.get('completed_V_scenes') == run.get('total_V_scenes') == len(rows),
                 '125 completed_V_scenes and total_V_scenes')
    checks.exact(run.get('completed_T_scenes') == run.get('total_T_scenes') == 32
                 and len(run['training_scene_ids']) == len(set(run['training_scene_ids'])) == 32,
                 '125 complete T32 metadata')
    checks.exact(not set(run['training_scene_ids']) & set(run['scene_ids']), 'T/V scene disjointness')
    checks.exact(summary.get('fixed_R_population_invariants_verified') is True
                 and summary.get('full_T_models_frozen_before_V') is True, 'declared fixed R/frozen heads')
    return run, summary, rows


def inspect_fixed_scene(saved, row, checks):
    checks.exact(set(saved) == set(row['methods']) == set(METHODS)
                 and set(row['pairwise']) == PAIR_KEYS, 'complete eleven arms/fifty-five pairs')
    checks.exact(row.get('fixed_R_population_invariants_verified') is True, 'scene fixed-R declaration')
    for left, right in PAIRS:
        pair = row['pairwise'][right+'_vs_'+left]
        checks.exact(pair['baseline_method'] == left and pair['candidate_method'] == right
                     and pair['side_labels'] == dict(A0=left, candidate=right), 'pair direction and side labels')
    # Includes complete group offsets/node partition, first-anchor mapping,
    # duplicate/unmatched rows, cross-arm truth/membership/validity equality,
    # all/observed/missing metrics and each pair's true common population/tails.
    outcomes = core.inspect_scene(saved, row, METHODS, checks)
    base = saved[METHODS[0]]
    t = len(base['target_ids']); represented = base['aligned_group_indices'][:t] >= 0
    checks.value(len(base['node_group_indices']), row['node_count'], 'node_count', 'scene original population')
    for left, right in PAIRS:
        key = right+'_vs_'+left; pair = row['pairwise'][key]
        checks.value(int(represented.sum()), pair['same_anchor_target_count'], 'same_anchor_target_count', key)
        checks.value(0, pair['changed_anchor_target_count'], 'changed_anchor_target_count', key)
        for region in ('same_anchor', 'changed_anchor'):
            for side, name in (('A0', left), ('candidate', right)):
                values = outcomes[name]['errors'] if region == 'same_anchor' else np.empty(0)
                checks.fields(core.stats(values), pair['strata'][region][side], ('point_count',)+core.STATS,
                              'scene fixed-anchor '+region+'/'+side)
        for transition in TRANSITIONS:
            checks.value(0, pair['coverage_transitions'][transition], transition, 'scene fixed population')
        for key in ('newly_covered_position', 'lost_coverage_position'):
            checks.fields(core.stats(np.empty(0)), pair[key], ('point_count',)+core.STATS, 'scene fixed '+key)
    return outcomes


def imported_sources(run, checks):
    """Resolve123's declared original32 import to121; never guess by existence.

    123 deliberately has no private NPZ for its original32. Those immutable
    arrays remain at123.run.pilot_run (121). 124 owns its own soft75 artifact.
    No production loader or missing-file fallback is used.
    """
    control = Path(run['source_control_run']); prior = core.read(control/'run.json')
    original = prior['original32_ids']; pilot = Path(prior['pilot_run'])
    checks.exact(prior.get('status') == 'COMPLETED' and prior.get('exit_code') == 0
                 and prior.get('idea_id') == 'IDEA-SOURCE-PREFERENCE-FULL-CONTROL-20261029'
                 and prior.get('plan_version') == '1.0', 'completed123 retained-control lineage')
    checks.exact(original == run['scene_ids'] and len(original) == len(set(original)) == 32
                 and set(original) <= set(prior['scene_ids']), '123 declared original32 exact125 population')
    checks.exact(pilot.resolve() != control.resolve(), '123 explicit distinct121 pilot')
    pilot_run = core.read(pilot/'run.json')
    checks.exact(pilot_run.get('status') == 'COMPLETED' and pilot_run.get('exit_code') == 0
                 and pilot_run.get('idea_id') == 'IDEA-SOURCE-PREFERENCE-CONTROL-20261029'
                 and pilot_run.get('plan_version') == '1.0'
                 and pilot_run.get('scene_ids') == original and pilot_run.get('completed_scenes') == 32,
                 'completed121 original32 lineage')
    return ((pilot, IMPORTED[:6]), (Path(run['mix_run']), IMPORTED[6:]))


def imported_artifacts(sources, sid, saved, checks):
    """Seven retained controls must remain exact to123→121 and124 arrays."""
    for source, names in sources:
        original = core.artifacts(source/'private'/f'states-{sid}.npz', names)
        for name in names:
            checks.exact(set(original[name]) == set(saved[name]), 'imported control schema '+name)
            for key in original[name]:
                checks.array(saved[name][key], original[name][key], 'exact imported control '+name+'/'+key)


def inspect_fixed_summary(report, indices, results, rows, checks, label):
    core.inspect_summary(report, indices, results, rows, METHODS, checks, label)
    if not indices:
        return
    for left, right in PAIRS:
        key = right+'_vs_'+left; pair = report['pairwise'][key]
        checks.exact(pair['baseline_method'] == left and pair['candidate_method'] == right
                     and pair['side_labels'] == dict(A0=left, candidate=right), label+' pair direction')
        for region in ('same_anchor', 'changed_anchor'):
            for side, name in (('A0', left), ('candidate', right)):
                values = (np.concatenate([results[i][name]['errors'] for i in indices])
                          if region == 'same_anchor' else np.empty(0))
                checks.fields(core.stats(values), pair['strata'][region][side], ('point_count',)+core.STATS,
                              label+' fixed-anchor '+region+'/'+side)
        for i in indices:
            a, b = results[i][left]['errors'], results[i][right]['errors']
            delta = float((b-a).mean()) if len(a) else None
            checks.value(delta, pair['scene_mae_changes'][rows[i]['scene_id']], 'scene_mae_change', label+' '+key)
        # Summary125 exposes summed coverage transitions; only per-scene125
        # exposes newly/lost-covered error distributions (checked above).


def audit_run(run_path, checks):
    run, summary, rows = metadata(run_path, checks)
    sources = imported_sources(run, checks)
    results = []
    for row in rows:
        sid = row['scene_id']
        checks.exact(core.read(run_path/'scenes'/f'scene-{sid}.json.gz') == row, 'public per-scene copy')
        saved = core.artifacts(run_path/'private'/f'states-{sid}.npz', METHODS)
        results.append(inspect_fixed_scene(saved, row, checks))
        imported_artifacts(sources, sid, saved, checks)
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform == 'darwin' else 1024)
        checks.exact(rss <= core.RAM, 'resident memory within4GiB')
    components = sorted({row['component_id'] for row in rows})
    checks.exact(set(summary['components']) == set(components), 'complete component population')
    inspect_fixed_summary(summary, list(range(len(rows))), results, rows, checks, 'full')
    for component in components:
        inspect_fixed_summary(summary['components'][component],
            [i for i, row in enumerate(rows) if row['component_id'] == component], results, rows, checks, 'component')
    frequencies = Counter(row['component_id'] for row in rows)
    largest = max(components, key=lambda component: frequencies[component])
    inspect_fixed_summary(summary['without_largest_component'],
        [i for i, row in enumerate(rows) if row['component_id'] != largest], results, rows, checks, 'without_largest')
    return dict(scene_count=len(rows), component_count=len(components), method_count=len(METHODS),
        pair_count_per_scene=len(PAIRS), exact_imported_control_scene_arm_count=len(rows)*len(IMPORTED),
        aggregate_subset_count=2+len(components), completion_field='completed_V_scenes')


def audit(run_path, output):
    started = time.monotonic(); before = resource.getrusage(resource.RUSAGE_SELF); checks = core.Checks()
    report = dict(status='RUNNING', started_at=datetime.now(timezone.utc).isoformat(),
        scope='independent125 fixed-R eleven-arm/fifty-five-pair saved-array main position/population audit',
        earth_radius_metres=core.RADIUS,
        tolerances=dict(rtol=1e-9, metres_atol=1e-5, mse_atol=.001, rate_atol=1e-12),
        not_audited=['training/teacher inference/head fitting or feature legality',
            'manifest/physical reference correctness or identity-independent generalization',
            'input time/support semantics beyond stored masks, support-conditioned regions or policy routing',
            'paper attribute metrics, speed/heading, per-point benefit/harm or credible peer claims'])
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream)
    try:
        if sys.platform.startswith('linux'):
            if hasattr(os, 'sched_getaffinity'):
                os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:4])
            soft, hard = resource.getrlimit(resource.RLIMIT_AS)
            limit = min([core.MAX_AS]+[v for v in (soft, hard) if v != resource.RLIM_INFINITY])
            resource.setrlimit(resource.RLIMIT_AS, (limit, hard))
        report.update(audit_run(run_path, checks), status='COMPLETED', exit_code=0)
    except BaseException as error:
        report.update(status='FAILED', exit_code=1, error_type=type(error).__name__, error=str(error))
        traceback.print_exc()
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
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    def interrupted(signum, frame):
        raise InterruptedError('audit interrupted')
    signal.signal(signal.SIGTERM, interrupted); signal.signal(signal.SIGINT, interrupted)
    result = audit(args.run.resolve(), args.output.resolve())
    print(result['status']+': '+str(args.output))
    return result['exit_code']


if __name__ == '__main__':
    raise SystemExit(main())
