"""Local audit of 071A/B returned aggregates; no checkpoint, NPZ, model or ZIP reads.

Actual parameter/batch pairing was executed by the server closeout. This audit
checks its implementation and result contract, and independently re-aggregates
returned scene sufficient statistics, never pretending compact summaries are
raw predictions or training logs.
"""
from pathlib import Path
import argparse
import copy
import gzip
import json
import math

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT/'HandOver/results'
PREFIX = {'071A': 'LEVEL1-GATE-LOG-20261023-071A',
          '071B': 'LEVEL1-GATE-LINEAR-20261023-071B',
          '068': 'LEVEL1-SUPPORT-GATE-20261023-068',
          '070': 'LEVEL1-LINEAR-GATE-20261023-070'}
SUMMARY = RESULTS/'LEVEL1-POSITION-BENCHMARK-20261023-071-summary'
WEIGHTS = ('point_equal', 'scene_target_equal', 'scene_equal_point_mean', 'scene_equal_target_mean')
REGIONS = ('all', 'anchor_inside_support', 'anchor_outside_helper_supported',
           'predicted_group_no_support', 'no_predicted_group')
ARM_NAMES = {'071A': '071A_log', '071B': '071B_linear'}
CONFIGS = {'071A': 'support-gate-replica-log-level1.json',
           '071B': 'support-gate-replica-linear-level1.json'}
CHECKS = {'numeric_comparisons': 0, 'maximum_absolute_difference': {}, 'maximum_relative_difference': 0.}


def read(path):
    data = path.read_bytes()
    return json.loads(gzip.decompress(data) if path.suffix == '.gz' else data)


def equal(a, b, key):
    if a is None or b is None:
        assert a is b, (key, a, b)
        return
    assert math.isfinite(a) and math.isfinite(b), (key, a, b)
    error = abs(a-b)
    CHECKS['numeric_comparisons'] += 1
    CHECKS['maximum_absolute_difference'][key] = max(error, CHECKS['maximum_absolute_difference'].get(key, 0.))
    CHECKS['maximum_relative_difference'] = max(error/max(abs(b), 1.), CHECKS['maximum_relative_difference'])
    assert math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-8), (key, a, b)


def pool(items, keys=('mae', 'mse', 'log1p')):
    n = sum(x['unit_count'] for x in items)
    out = {'unit_count': n}
    for key in keys:
        if key not in items[0]:
            continue
        out[key] = sum(x['unit_count']*x[key] for x in items if x['unit_count'])/n if n else None
    out['rmse'] = math.sqrt(out['mse']) if n else None
    return out


def audit_pool(rows, expected, getter):
    actual = {}
    for region in REGIONS:
        actual[region] = {}
        for method, weights in expected[region].items():
            actual[region][method] = {}
            for weight, target in weights.items():
                value = pool([getter(row, region, method, weight) for row in rows])
                assert value['unit_count'] == target['unit_count']
                for k in value.keys()-{'unit_count'}:
                    equal(value[k], target[k], k)
                actual[region][method][weight] = value
    return actual


def sign_counts(delta):
    return dict(improved=sum(x < 0 for x in delta), worsened=sum(x > 0 for x in delta),
                unchanged=sum(x == 0 for x in delta))


def method_counts(rows, getter, method, control):
    return {metric: sign_counts([getter(row, method)[metric]-getter(row, control)[metric] for row in rows])
            for metric in ('mae', 'rmse', 'log1p') if metric in getter(rows[0], method)}


def check_launch(directory, gpu=True):
    request, status = read(directory/'launch/request.json'), read(directory/'launch/status.json')
    assert status['status'] == 'COMPLETED' and status['returncode'] == int((directory/'launch/exit_code.txt').read_text()) == 0
    assert not status['timed_out'] and not status['time_advisory_issued']
    argv = request['argv']
    assert argv[argv.index('--cpu-threads')+1] == '4' and request['time_policy'] == 'advisory'
    if gpu:
        assert argv[argv.index('--device')+1] == 'cuda:0'
        assert argv[argv.index('--time-policy')+1] == 'advisory'
        assert 'support_gate_replication.py' in argv[1] and argv[2] == 'fit'
    else:
        assert 'compare_position_benchmarks.py' in argv[1]
    return dict(status='COMPLETED', returncode=0, elapsed_seconds=status['elapsed_seconds'], argv=argv)


def audit_training(tag, manifest):
    path = RESULTS/PREFIX[tag]
    run, summary, fit, execution = [read(path/name) for name in
        ('run.json', 'summary.json', 'fit/run.json.gz', 'execution.json')]
    config = read(ROOT/'configs'/CONFIGS[tag])
    for record in (run, summary, fit, execution):
        assert record['configuration'] == config and record['test_opened'] is False
    for record in (run, summary, fit):
        assert record['status'] == 'COMPLETED' and record['optimizer_updates'] == 1000
    assert run['completed_endpoints'] == ['initial', 'final'] and run['validation_selection'] is False
    assert run['retries'] == 0 and execution['resumed'] is False
    assert not any(run[k] for k in ('association_recomputed', 'threshold_reselected', 'validation_groups_recomputed'))
    assert fit['training_scene_ids'] == manifest['splits']['T_core'] and len(fit['training_scene_ids']) == 2100
    exposure = fit['exposure']
    assert exposure['successful_updates'] == 1000 and exposure['sampled_anchors'] == 8000
    assert sum(exposure['scenes'].values()) == sum(exposure['anchors'].values()) == 8000
    assert len(exposure['scenes']) == fit['trained_scene_count'] == 2053
    assert len(exposure['anchors']) == 7459
    for key in ('successful_updates', 'sampled_anchors', 'eligible_query_points',
                'eligible_valid_position_points', 'valid_position_points'):
        mode_key = {'successful_updates': 'updates', 'sampled_anchors': 'anchors'}.get(key, key)
        assert sum(v[mode_key] for v in exposure['modes'].values()) == exposure[key]
    for label, step in (('initial', 0), ('final', 1000)):
        endpoint, report = read(path/label/'endpoint-run.json'), read(path/label/'summary.json.gz')
        for record in (endpoint, report):
            assert record['status'] == 'COMPLETED' and record['device'] == 'cpu'
            assert record['optimizer_updates'] == 0 and not record['test_opened']
            assert record['configuration'] == config
            assert record['checkpoint']['step'] == record['checkpoint']['actual_updates'] == step
            assert record['checkpoint']['configured_final_updates'] == 1000
            assert record['scene_ids'] == manifest['splits']['V_select']
            assert all(not record[k] for k in ('groups_recomputed', 'scores_recomputed', 'threshold_reselected'))
        assert endpoint['model_forwards'] == report['model_forwards'] == 9353
        assert report['ordinary_source_replay_exact'] and report['fixed_population_verified']
        for method in report['methods'].values():
            pop = method['population']
            assert pop['target_identity_count'] == 9100 and pop['predicted_group_count'] == 9353
            assert pop['duplicate_prediction_group_count'] == 610
    windows = read(path/'fit/training-windows.json')
    assert windows['rows'] == 1000 and len(windows['windows']) == 10
    for i, row in enumerate(windows['windows']):
        assert row['step_from'] == 100*i+1 and row['step_to'] == 100*(i+1)
        assert sum(row['modes'].values()) == 100
    clipped = sum(w['clipped_updates'] for w in windows['windows'])
    assert clipped == windows['whole_run']['clipped_updates'] == fit['gradient_clipping']['clipped_updates']
    for key, window_key in (('valid_position_points', 'valid_position_points_mean'),
                            ('eligible_valid_position_points', 'eligible_valid_position_points_mean')):
        equal(sum(100*w[window_key] for w in windows['windows']), exposure[key], 'exposure_count')
    launch = check_launch(path)
    argv = launch['argv']
    args = execution['arguments']
    for name in ('source_system', 'manifest', 'alignment', 'config', 'output', 'device', 'cpu_threads', 'time_policy'):
        assert str(args[name]) == argv[argv.index('--'+name.replace('_', '-'))+1]
    return dict(configuration=config, launch=launch, parameters=summary['resources']['parameter_count'],
                fit_elapsed_seconds=fit['elapsed_seconds'], exposure={k: v for k, v in exposure.items() if k not in ('scenes', 'anchors')},
                distinct_scenes=len(exposure['scenes']), distinct_anchors=len(exposure['anchors']),
                clipping=fit['gradient_clipping'], training_windows=windows,
                gradients_are_returned_log_aggregates_not_locally_reconstructed=True), fit


def audit_objective(path, benchmark, arm_name):
    report = read(path/'comparison/objective-summary.json.gz')
    rows = read(path/'comparison/scene-summary.json.gz')
    objective = report['objectives']
    assert report['status'] == 'COMPLETED' and report['model_forwards'] == 0 and not report['test_opened']
    assert all(report[k] for k in ('population_and_masks_exact', 'source_population_verified', 'source_position_metrics_verified'))
    getter = lambda row, r, m, w: row['regions'][r]['methods'][m][w]
    expected = {r: objective['regions'][r]['methods'] for r in REGIONS}
    actual = audit_pool(rows, expected, getter)
    for region in REGIONS:
        for own, common in (('initial', 'fixed_prior095'), ('final', arm_name), ('support_switch', 'support_switch')):
            for weight in WEIGHTS:
                a, b = actual[region][own][weight], benchmark[region][common][weight]
                assert a['unit_count'] == b['unit_count']
                for metric in ('mae', 'mse', 'rmse'):
                    equal(a[metric], b[metric], metric)
        # Every paired signed sum and sign count is additive across scene records.
        for pair, weightings in objective['regions'][region]['paired'].items():
            for weight, target in weightings.items():
                parts = [row['regions'][region]['paired'][pair][weight] for row in rows]
                n = sum(x['unit_count'] for x in parts)
                assert n == target['unit_count']
                for metric in ('mae', 'mse', 'log1p'):
                    for key in ('signed_sum', 'negative_unit_count', 'positive_unit_count', 'exact_tie_unit_count'):
                        equal(sum(x[metric][key] for x in parts), target[metric][key], 'paired_'+metric+'_'+key)
                    equal(target[metric]['signed_sum']/n if n else None,
                          target[metric]['mean_difference'], 'paired_mean_'+metric)
    component_counts = {}
    for component, result in objective['components'].items():
        subset = [r for r in rows if r['component_id'] == component]
        assert [r['scene_id'] for r in subset] == result['scene_ids']
        audit_pool(subset, {r: result['regions'][r]['methods'] for r in REGIONS}, getter)
    getterall = lambda row, m: row['regions']['all']['methods'][m]['point_equal']
    scene_counts = {control: method_counts(rows, getterall, 'final', control) for control in ('initial', 'support_switch')}
    component_records = list(objective['components'].values())
    component_counts = {control: method_counts(component_records, getterall, 'final', control)
                        for control in ('initial', 'support_switch')}
    final = read(path/'final/summary.json.gz')
    return dict(four_weights={r: actual[r] for r in REGIONS}, scene_counts=scene_counts,
                component_counts=component_counts, routing=final['routing'],
                tails_from_returned_array_evaluation={r: {m: objective['regions'][r]['methods'][m]['point_equal']
                    for m in ('initial', 'final', 'support_switch')} for r in ('all', 'anchor_outside_helper_supported')},
                leave_one_named_target_sensitivity=final['leave_one_named_target_sensitivity'],
                paired_final_minus_initial=objective['regions']['all']['paired']['final__minus__initial'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    manifest = read(ROOT/'configs/data/mtad-level1-scene3000.json')
    summary, rows = read(SUMMARY/'summary.json.gz'), read(SUMMARY/'scene-summary.json.gz')
    run = read(SUMMARY/'run.json')
    assert summary['status'] == run['status'] == 'COMPLETED'
    assert run['completed_scenes'] == len(rows) == len(summary['scene_ids']) == 300
    assert summary['scene_ids'] == manifest['splits']['V_select']
    assert all(run[k] == 0 for k in ('model_forwards', 'optimizer_updates')) and not run['test_opened']
    assert [r['scene_ids'][0] for r in rows] == summary['scene_ids']
    assert all(r['component_id'] == manifest['components'][r['scene_ids'][0]] for r in rows)
    for key in summary['population']:
        assert sum(r['population'][key] for r in rows) == summary['population'][key]
    for key in summary['position_population'].keys()-{'point_coverage'}:
        assert sum(r['position_population'][key] for r in rows) == summary['position_population'][key]
    getter = lambda row, r, m, w: row['position_weightings'][r][m][w]
    actual = audit_pool(rows, summary['position_weightings'], getter)
    components = {}
    for component, record in summary['components'].items():
        part = [r for r in rows if r['component_id'] == component]
        assert record['scene_ids'] == [r['scene_ids'][0] for r in part]
        components[component] = audit_pool(part, record['position_weightings'], getter)
    assert len(components) == 38
    largest = min(components, key=lambda k: (-components[k]['all']['anchor']['point_equal']['unit_count'], k))
    assert largest == summary['largest_component_id']
    remainder = [r for r in rows if r['component_id'] != largest]
    assert len(remainder) == 170 and summary['remaining_components']['scene_count'] == 170
    remaining = audit_pool(remainder, summary['remaining_components']['position_weightings'], getter)
    training, fits = {}, {}
    for tag in ARM_NAMES:
        training[tag], fits[tag] = audit_training(tag, manifest)
    assert fits['071A']['exposure'] == fits['071B']['exposure']
    ca, cb = copy.deepcopy(fits['071A']['configuration']), copy.deepcopy(fits['071B']['configuration'])
    assert ca.pop('position_loss') == 'log1p' and cb.pop('position_loss') == 'linear' and ca == cb
    pair = summary['pairing']
    assert pair['status'] == 'PAIR_VERIFIED' and pair['matched_updates'] == 1000 and pair['matched_anchors'] == 8000
    assert all(pair[k] is True for k in ('initial_parameters_exact', 'ordered_training_scene_ids_exact',
                                      'actual_successful_batches_exact', 'sampling_exposure_exact', 'sampler_rng_exact'))
    objectives = {tag: audit_objective(RESULTS/PREFIX[tag], actual, name) for tag, name in ARM_NAMES.items()}
    old = read(RESULTS/PREFIX['070']/'comparison/objective-summary.json.gz')['comparisons']
    old068 = old['vs_log_final']['objectives']['regions']
    old070 = old['vs_initial']['objectives']['regions']
    seeds = {}
    for weight in WEIGHTS:
        seeds[weight] = {tag: source['all']['methods'][method][weight] for tag, source, method in
                         (('068_log', old068, 'initial'), ('070_linear', old070, 'final'))}
        seeds[weight].update({ARM_NAMES[tag]: objectives[tag]['four_weights']['all']['final'][weight] for tag in ARM_NAMES})
        for metric in ('mae', 'mse', 'rmse'):
            equal(seeds[weight]['070_linear'][metric], actual['all']['070_linear'][weight][metric], metric)
    for tag, oldtag in (('071A', '068'), ('071B', '070')):
        c = copy.deepcopy(read(RESULTS/PREFIX[oldtag]/'fit/run.json.gz')['configuration'])
        c.setdefault('position_loss', 'log1p')
        c['training']['seed'] = 20261023
        assert c == fits[tag]['configuration']
    result = dict(status='COMPLETED_LOCAL_AGGREGATE_AUDIT', scope=__doc__,
                  model_forwards=0, optimizer_updates=0, test_opened=False, archive_reads=0,
                  summary_launch=check_launch(SUMMARY, gpu=False),
                  population=summary['population'], position_population=summary['position_population'],
                  checks=CHECKS, training=training, server_pairing=pair,
                  pairing_evidence=dict(local_exposure_dictionary_equality=True,
                      local_initial_tensor_and_batch_recheck=False,
                      reason='step-0.pt/final.pt/training.jsonl stay on server; local fit aggregates and returned server audit are available.',
                      reviewed_procedure='association/evaluation/support_gate_replication.py::_fit/verify_pairing compares '
                          'every initial tensor, all successful step/mode/ordered scene-anchor-members-count batches, '
                          'reconstructs exposure from logs, and compares initial/final NumPy sampler states. '
                          'CPU closeout refuses non-PAIR_VERIFIED before array comparison.',
                      conclusion='Server executed actual pairing, not merely equal seed/counts; local independently validates '
                          'code/contract plus aggregate exposure, not raw checkpoint or batch equality.'),
                  gate_objectives=objectives, two_seed_four_weights=seeds,
                  benchmark_four_weights=actual['all'], opportunity=actual['anchor_outside_helper_supported'],
                  largest_component=dict(id=largest, scenes=130, metrics=components[largest]['all']),
                  remaining_components=dict(scenes=170, metrics=remaining['all']),
                  component_rank_counts={name: method_counts(list(summary['components'].values()),
                        lambda r, m: r['position_weightings']['all'][m]['point_equal'], name, 'fixed_prior095')
                        for name in ARM_NAMES.values()},
                  limitations=['Only returned scene sufficient statistics are re-aggregated locally; this is not a new raw-point replay.',
                      'Global P95/P99 and per-point routing cannot be reconstructed from scene quantiles. They are cross-linked to server array evaluations.',
                      'Training-window derived summaries are additive-checked against exposure/clipping; medians and actual tensor updates remain server evidence.',
                      'Two seeds support effect direction, not absence of seed sensitivity, independent generalization or statistical significance.',
                      'Small gradients and saturation co-occur; log loss, weighting, sigmoid and Adam dynamics have not been causally decomposed.'])
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    print(json.dumps({'status': result['status'], 'checks': CHECKS,
                      'A': actual['all']['071A_log']['point_equal'],
                      'B': actual['all']['071B_linear']['point_equal']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
