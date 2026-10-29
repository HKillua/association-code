"""Independent131 saved-array position/population and frozen-import audit.

Acceptance: exact manifest V300/T32/component membership; every declared
subset, eleven arms and55 pairs; independent saved-array counts/geometry/tails;
125 original32 eleven-arm +123 remaining268 six-arm imports and16 head arrays
exact. No production evaluator, feature builder, fitter or forward is called.
Native/head application counts are metadata consistency, not physical/support
semantics, feature legality or head-fitting certification. Writes only a new
audit sidecar; incomplete131 and pre-existing audit outputs are refused.
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

from tools.analysis import audit_conditional_value_arrays as fixed
import numpy as np

core = fixed.core
IDEA = 'IDEA-CONDITIONAL-SOURCE-VALUE-FULL-20261029'
CONTROL_IDEA = 'IDEA-SOURCE-PREFERENCE-FULL-CONTROL-20261029'
METHODS, BASES = fixed.METHODS, fixed.METHODS[:6]
HEADS = ('TMEAN', 'LINEAR', 'QUADRATIC', 'SHUFFLED')
HEAD_KEYS = {h+'_'+k for h in HEADS for k in ('mean', 'std', 'beta', 'permutation')}


def same_arrays(actual, expected, checks, label):
    checks.exact(set(actual) == set(expected), label+' schema')
    for key in expected:
        left, right = np.asarray(actual[key]), np.asarray(expected[key])
        checks.exact(left.dtype == right.dtype, label+'/'+key+' exact dtype')
        checks.array(left, right, label+'/'+key)


def metadata(run_path, manifest, checks):
    run, summary = (core.read(run_path/name) for name in ('run.json', 'summary.json'))
    checks.exact(run.get('status') == summary.get('status') == 'COMPLETED'
                 and run.get('exit_code') == 0, 'completed131 only')
    checks.exact(run.get('idea_id') == summary.get('idea_id') == IDEA
                 and run.get('plan_version') == summary.get('plan_version') == '1.0', '131 scope/version')
    rows = core.read(run_path/'scene-summary.json.gz')
    ids = manifest['splits']['V_select']; train = manifest['splits']['T_core'][:32]
    checks.exact(len(ids) == len(set(ids)) == 300 and len(train) == len(set(train)) == 32
                 and not set(train) & set(ids), 'manifest fixed V300/T32')
    checks.exact(run.get('scene_ids') == summary.get('scene_ids') == [r['scene_id'] for r in rows] == ids,
                 'complete ordered manifest V300')
    checks.exact(run.get('completed_scenes') == run.get('total_scenes') == summary.get('scene_count') == 300,
                 '131 completed_scenes schema, not125 completed_T/V_scenes')
    checks.exact(tuple(run['methods']) == METHODS and set(summary['methods']) == set(METHODS), 'eleven methods')
    prior_path, control_path = Path(run['pilot_run']), Path(run['source_control_run'])
    prior, prior_summary = (core.read(prior_path/name) for name in ('run.json', 'summary.json'))
    control = core.read(control_path/'run.json')
    checks.exact(prior.get('status') == prior_summary.get('status') == 'COMPLETED' and prior.get('exit_code') == 0
                 and prior.get('idea_id') == prior_summary.get('idea_id') == fixed.IDEA
                 and prior.get('plan_version') == prior_summary.get('plan_version') == '1.0'
                 and prior.get('completed_V_scenes') == 32, 'completed125 declared pilot')
    original = prior['scene_ids']
    checks.exact(len(original) == len(set(original)) == 32 and set(original) <= set(ids)
                 and run.get('original32_ids') == summary.get('original32_ids') == original
                 and prior_summary.get('scene_ids') == original, 'original32 from125, not arbitrary manifest prefix')
    checks.exact(control.get('status') == 'COMPLETED' and control.get('exit_code') == 0
                 and control.get('idea_id') == CONTROL_IDEA and control.get('plan_version') == '1.0'
                 and control.get('scene_ids') == ids and control.get('original32_ids') == original,
                 'completed123 full-control population')
    for key in ('source_system', 'partition_run'):
        checks.exact(Path(run[key]).resolve() == Path(prior[key]).resolve() == Path(control[key]).resolve(),
                     'same upstream '+key)
    checks.exact(Path(prior['source_control_run']).resolve() == control_path.resolve(), '125 to123 lineage')
    checks.exact(run.get('training_scene_ids') == summary.get('training_scene_ids') == prior.get('training_scene_ids') == train,
                 'head training scenes are manifest first32 T_core')
    checks.exact(run.get('feature_names') == prior.get('feature_names') and len(run['feature_names']) == 15
                 and tuple(prior['methods']) == METHODS and prior.get('threshold') == 1.7320103112387182
                 and prior_summary.get('full_T_models_frozen_before_V') is True, '125 frozen feature/recipe lineage')
    train_components = {str(manifest['components'][sid]) for sid in train}
    original_components = {str(manifest['components'][sid]) for sid in original}
    components = sorted({str(manifest['components'][sid]) for sid in ids})
    checks.exact(len(components) == 38 and len(original_components) == 9 and len(train_components) == 12
                 and run.get('training_component_ids') == sorted(train_components)
                 and set(summary['components']) == set(components), 'manifest full/original/T component sets')
    for key in ('models_constructed', 'teacher_forward_calls', 'model_forwards', 'neural_model_forwards',
                'optimizer_updates', 'analytic_head_fits', 'ridge_system_solves'):
        checks.value(0, run[key], key, 'no new fit/forward metadata')
    for key in ('models_constructed', 'teacher_forward_calls', 'optimizer_updates', 'analytic_head_fits', 'neural_model_forwards'):
        checks.value(0, summary[key], key, 'summary no new fit/forward metadata')
    for key in ('scores_recomputed', 'groups_recomputed', 'threshold_reselected', 'normalizer_refitted',
                'centers_refitted', 'head_reselected', 'test_opened'):
        checks.exact(run.get(key) is False, 'fixed recipe '+key)
    checks.exact(run.get('declared_primary_head') == summary.get('declared_primary_head') == 'QUADRATIC'
                 and run.get('restored_head_count') == summary.get('restored_head_count') == 4
                 and run.get('restored_head_coefficient_counts') == dict(TMEAN=1, LINEAR=16, QUADRATIC=136, SHUFFLED=136),
                 'frozen four heads and primary Q')
    for key in ('all125_fullT_head_arrays_imported_exact', 'original125_public_and_artifacts_imported_exact',
                'fixed_R_population_invariants_verified', 'no_expansion_beyond_V300', 'position_only'):
        checks.exact(summary.get(key) is True, '131 scope '+key)
    counters = Counter(); native_events = applications = queries = 0
    for row in rows:
        sid = row['scene_id']; cid = str(manifest['components'][sid]); was_original = sid in set(original)
        checks.exact(row['component_id'] == cid and row['stratum'] == ('original32' if was_original else 'remaining268')
                     and row['component_seen_in_original32'] == (cid in original_components)
                     and row['component_seen_in_T32_head_training'] == (cid in train_components), 'manifest subset membership '+sid)
        imports = METHODS if was_original else BASES
        added = () if was_original else METHODS[6:]
        checks.exact(tuple(row['imported_prediction_methods']) == imports
                     and tuple(row['newly_computed_prediction_methods']) == added
                     and row['original125_public_and_artifacts_imported_exact'] is was_original,
                     'per-scene import/new methods '+sid)
        counters['imported'] += len(imports); counters['computed'] += len(added)
        if not was_original:
            native = row['native_input_audit']
            checks.exact(native['node_count'] == row['node_count'] and type(native['canonical_event_count']) is int
                         and native['canonical_event_count'] >= 0 and native.get('canonical_all') is True
                         and native.get('physical_binding_exact') is True
                         and native.get('native_quality_reconstructed_from_events') is True, 'native metadata '+sid)
            counters['native'] += 1; native_events += native['canonical_event_count']; applications += 4
            queries += 4*row['methods'][METHODS[0]]['population']['predicted_group_count']*64
    checks.exact(dict(counters) == dict(imported=1960, computed=1340, native=268), '1960 imports/1340 new/268 native')
    for key, value in (('imported_prediction_scene_arms', 1960), ('newly_computed_prediction_scene_arms', 1340),
                       ('completed_native_scenes', 268), ('canonical_event_count', native_events),
                       ('analytic_head_application_count', applications), ('analytic_query_prediction_count', queries)):
        checks.value(value, run[key], key, 'run amount metadata')
    for key, value in (('scene_prediction_arms_imported', 1960), ('scene_prediction_arms_computed', 1340),
                       ('completed_native_scenes', 268), ('canonical_event_count', native_events),
                       ('analytic_head_application_count', applications), ('analytic_query_prediction_count', queries)):
        checks.value(value, summary[key], key, 'summary amount metadata')
    largest = max(components, key=lambda cid: sum(row['component_id'] == cid for row in rows))
    checks.exact(summary['largest_component_id'] == largest, 'predeclared scene-count largest component')
    original_set = set(original)
    subsets = dict(
        original32=[i for i,r in enumerate(rows) if r['scene_id'] in original_set],
        remaining268=[i for i,r in enumerate(rows) if r['scene_id'] not in original_set],
        remaining268_seen_components=[i for i,r in enumerate(rows) if r['scene_id'] not in original_set and r['component_id'] in original_components],
        remaining268_unseen_components=[i for i,r in enumerate(rows) if r['scene_id'] not in original_set and r['component_id'] not in original_components],
        T32_head_training_seen_components=[i for i,r in enumerate(rows) if r['component_id'] in train_components],
        T32_head_training_unseen_components=[i for i,r in enumerate(rows) if r['component_id'] not in train_components],
        remaining268_T32_head_training_seen_components=[i for i,r in enumerate(rows) if r['scene_id'] not in original_set and r['component_id'] in train_components],
        remaining268_T32_head_training_unseen_components=[i for i,r in enumerate(rows) if r['scene_id'] not in original_set and r['component_id'] not in train_components],
        without_largest_component=[i for i,r in enumerate(rows) if r['component_id'] != largest])
    checks.exact(all(key in summary for key in subsets), 'all nine declared subset reports present')
    return run, summary, rows, subsets, prior_path, control_path


def inspect_heads(run_path, prior_path, run, checks):
    def load(path):
        with np.load(path/'private'/'full-T-heads.npz', allow_pickle=False) as saved:
            checks.exact(set(saved.files) == HEAD_KEYS, 'all16 frozen fullT arrays')
            return {key: saved[key] for key in saved.files}
    actual, original = load(run_path), load(prior_path)
    same_arrays(actual, original, checks, 'exact125 frozen head')
    for head in HEADS:
        size = dict(TMEAN=1, LINEAR=16, QUADRATIC=136, SHUFFLED=136)[head]
        for key, shape in (('mean', (15,)), ('std', (15,)), ('beta', (size,))):
            value = actual[head+'_'+key]
            checks.exact(value.shape == shape and value.dtype == np.float64 and np.isfinite(value).all(), 'frozen head shape '+head+'/'+key)
        permutation = actual[head+'_permutation']
        checks.exact(permutation.shape == (run['restored_T_label_count'],) and permutation.dtype.kind in 'iu', 'frozen label population '+head)


def inspect_imports(run_path, prior_path, control_path, sid, was_original, saved, checks):
    source, names = (prior_path, METHODS) if was_original else (control_path, BASES)
    original = core.artifacts(source/'private'/f'states-{sid}.npz', names)
    for name in names:
        same_arrays(saved[name], original[name], checks, 'exact imported131 '+name)


def inspect_aggregate(report, indices, results, rows, checks, label):
    fixed.inspect_fixed_summary(report, indices, results, rows, checks, label)
    if not indices:
        return
    for key, row_key in [('scene_prediction_arms_imported', 'imported_prediction_methods'),
                         ('scene_prediction_arms_computed', 'newly_computed_prediction_methods')]:
        checks.value(sum(len(rows[i][row_key]) for i in indices), report[key], key, label)
    for base, candidate in fixed.PAIRS:
        changes = np.concatenate([results[i][candidate]['errors']-results[i][base]['errors'] for i in indices])
        actual = float(changes.mean()) if len(changes) else None
        key = candidate+'_vs_'+base
        checks.value(actual, report['pairwise'][key]['strata']['all']['candidate_minus_A0']['mean_error_change'],
                     'mean_error_change', label+'/'+key)


def audit_run(run_path, manifest_path, checks):
    manifest = core.read(manifest_path)
    run, summary, rows, subsets, prior_path, control_path = metadata(run_path, manifest, checks)
    inspect_heads(run_path, prior_path, run, checks)
    results = []
    for row in rows:
        sid = row['scene_id']
        checks.exact(core.read(run_path/'scenes'/f'scene-{sid}.json.gz') == row, 'public per-scene copy')
        saved = core.artifacts(run_path/'private'/f'states-{sid}.npz', METHODS)
        results.append(fixed.inspect_fixed_scene(saved, row, checks))
        inspect_imports(run_path, prior_path, control_path, sid, row['stratum'] == 'original32', saved, checks)
        checks.exact(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform == 'darwin' else 1024) <= core.RAM,
                     'resident memory within4GiB')
    inspect_aggregate(summary, list(range(len(rows))), results, rows, checks, 'full300')
    for cid, report in summary['components'].items():
        inspect_aggregate(report, [i for i,r in enumerate(rows) if r['component_id'] == cid],
                          results, rows, checks, 'component/'+cid)
    for label, indices in subsets.items():
        inspect_aggregate(summary[label], indices, results, rows, checks, label)
    return dict(scene_count=300, component_count=38, method_count=11, pair_count_per_scene=55,
        exact_imported_scene_arm_count=1960, newly_computed_scene_arm_count=1340,
        native_scene_count=268, exact_original125_scene_arm_count=352,
        exact_remaining123_scene_arm_count=1608, exact_frozen_head_array_count=16,
        aggregate_subset_count=1+38+len(subsets), manifest=str(manifest_path),
        subset_scene_counts={key:len(value) for key,value in subsets.items()}, completion_field='completed_scenes')


def audit(run_path, manifest_path, output):
    started = time.monotonic(); before = resource.getrusage(resource.RUSAGE_SELF); checks = core.Checks()
    report = dict(status='RUNNING', started_at=datetime.now(timezone.utc).isoformat(),
        scope='independent131 V300 eleven-arm55pair saved-array main position/population, all declared subsets and frozen imports',
        earth_radius_metres=core.RADIUS, tolerances=dict(rtol=1e-9, metres_atol=1e-5, mse_atol=.001, rate_atol=1e-12),
        not_audited=['feature legality, raw native event/support semantics or physical binding beyond metadata',
            'teacher/head fitting, action routing or physical reference/identity correctness',
            'support-conditioned strata, paper metrics, speed/heading or per-point benefit/harm',
            'physical mechanism, whole-pipeline independence, online latency or credible peers'])
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as stream: json.dump(report, stream)
    try:
        if sys.platform.startswith('linux'):
            if hasattr(os, 'sched_getaffinity'): os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:4])
            soft, hard = resource.getrlimit(resource.RLIMIT_AS)
            limit = min([core.MAX_AS]+[v for v in (soft,hard) if v != resource.RLIM_INFINITY])
            resource.setrlimit(resource.RLIMIT_AS, (limit, hard))
        report.update(audit_run(run_path, manifest_path, checks), status='COMPLETED', exit_code=0)
    except BaseException as error:
        report.update(status='FAILED', exit_code=1, error_type=type(error).__name__, error=str(error)); traceback.print_exc()
    finally:
        usage = resource.getrusage(resource.RUSAGE_SELF)
        report.update(exact_checks=checks.exact_checks, numeric_checks=checks.numeric_checks,
            null_checks=checks.null_checks, maximum_absolute_difference=checks.maximum_absolute_difference,
            elapsed_seconds=time.monotonic()-started, user_cpu_seconds=usage.ru_utime-before.ru_utime,
            system_cpu_seconds=usage.ru_stime-before.ru_stime,
            peak_rss_bytes=int(usage.ru_maxrss*(1 if sys.platform == 'darwin' else 1024)),
            finished_at=datetime.now(timezone.utc).isoformat())
        with output.open('w', encoding='utf-8') as stream: json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    def interrupted(signum, frame): raise InterruptedError('audit interrupted')
    signal.signal(signal.SIGTERM, interrupted); signal.signal(signal.SIGINT, interrupted)
    result = audit(args.run.resolve(), args.manifest.resolve(), args.output.resolve())
    print(result['status']+': '+str(args.output)); return result['exit_code']


if __name__ == '__main__':
    raise SystemExit(main())
