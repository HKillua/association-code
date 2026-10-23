"""Audit 072 returned sufficient statistics, finite T-grid and execution records.

No raw ZIP, checkpoint, training-sample replay, NPZ or model forward is performed.
Raw initial/batch/prediction equality is server evidence; additive scene metrics
are independently re-aggregated here. Quantiles cannot be pooled from summaries.
"""
from pathlib import Path
import argparse
import json
import math

import numpy as np
import torch

from audit_071_gates import (ROOT, RESULTS, CHECKS, WEIGHTS, REGIONS, read, equal,
                             audit_pool, method_counts)

SUMMARY = RESULTS/'LEVEL1-POSITION-BENCHMARK-20261023-072-summary'
CONSTANT = RESULTS/'LEVEL1-CONSTANT-GATE-20261023-072A'
NEW = ('072A_constant', '072B_lstm4000', '072C_transformer4000')
FOCUS = ('fixed_prior095', 'support_switch', '071B_linear', *NEW)


def execution(path, entry):
    request, status = read(path/'launch/request.json'), read(path/'launch/status.json')
    assert status['status'] == 'COMPLETED'
    assert status['returncode'] == int((path/'launch/exit_code.txt').read_text()) == 0
    assert not status['timed_out'] and not status['time_advisory_issued']
    assert request['time_policy'] == 'advisory' and request['deadline_monotonic'] is None
    assert request['environment']['CUDA_VISIBLE_DEVICES'] == ''
    assert all(request['environment'][k] == '4' for k in
               ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'))
    argv = request['argv']
    assert Path(argv[1]).name == entry and argv[argv.index('--cpu-threads')+1] == '4'
    return dict(status='COMPLETED', exit_code=0, elapsed_seconds=status['elapsed_seconds'],
                arguments=argv, advisory_seconds=request['max_seconds'])


def fit_a(manifest):
    fit, run, summary, actual = [read(CONSTANT/name) for name in
                               ('fit.json', 'run.json', 'summary.json', 'execution.json')]
    config = read(ROOT/'configs/constant-gate-level1.json')
    for record in (fit, run, summary, actual):
        assert record['configuration'] == config and record['optimizer_updates'] == 0
        assert not record['test_opened']
    for record in (fit, run, summary):
        assert record['status'] == 'COMPLETED' and record['train_only']
    assert run['completed_endpoints'] == ['final'] and not run['validation_selection']
    assert fit['training_scene_ids'] == manifest['splits']['T_core']
    assert not set(fit['training_scene_ids']) & set(manifest['splits']['V_select'])
    assert fit['candidate_count'] == 1001 and fit['sampling_replay_verified']
    old = read(RESULTS/'LEVEL1-GATE-LINEAR-20261023-071B/fit/run.json.gz')
    assert fit['successful_updates_replayed'] == old['exposure']['successful_updates'] == 1000
    for key in ('sampled_anchors', 'valid_position_points', 'eligible_valid_position_points'):
        assert fit[key] == old['exposure'][key]
    nominal = np.arange(1001, dtype=np.float64)/1000.
    logits = np.empty(1001, dtype=np.float64)
    logits[[0, -1]] = [-100., 100.]
    logits[1:-1] = np.log(nominal[1:-1]/(1.-nominal[1:-1]))
    logits = torch.tensor(logits, dtype=torch.float32)
    gates = torch.sigmoid(logits)
    assert np.array_equal(logits.numpy(), np.asarray(fit['grid_logits'], dtype=np.float32))
    assert np.array_equal(gates.numpy(), np.asarray(fit['effective_gates'], dtype=np.float32))
    losses = np.asarray(fit['training_losses'])
    assert len(losses) == 1001 and np.isfinite(losses).all()
    best = int(np.argmin(losses))
    assert best == fit['selected_index'] == 28
    assert losses[best] == fit['selected_training_loss']
    assert fit['selected_logit'] == float(logits[best]) == run['selected_logit']
    assert fit['selected_gate'] == float(gates[best]) == run['selected_gate']
    launch = execution(CONSTANT, 'constant_gate_fit.py')
    argv = launch['arguments']
    for k, v in actual['arguments'].items():
        flag = '--'+k.replace('_', '-')
        if k == 'archive':
            assert argv[argv.index(flag)+1:argv.index(flag)+1+len(v)] == v
        else:
            assert argv[argv.index(flag)+1] == str(v)
    return dict(execution=launch, configuration=config, source071B=fit['source071B'],
                source_system=fit['source_system'], training_scene_count=len(fit['training_scene_ids']),
                selected_index=best, selected_logit=fit['selected_logit'], selected_gate=fit['selected_gate'],
                selected_training_loss=fit['selected_training_loss'],
                minimum_indices=np.flatnonzero(losses == losses.min()).tolist(),
                curve_decreasing_edges=int((np.diff(losses) < 0).sum()),
                curve_increasing_edges=int((np.diff(losses) > 0).sum()),
                decreasing_until_selected=bool(np.all(np.diff(losses[:best+1]) < 0)),
                increasing_after_selected=bool(np.all(np.diff(losses[best:]) > 0)),
                sampled_curve={str(i/1000): float(losses[i]) for i in (0, 1, 28, 100, 250, 500, 750, 950, 1000)},
                sampled_anchors=fit['sampled_anchors'], valid_position_points=fit['valid_position_points'],
                eligible_valid_position_points=fit['eligible_valid_position_points'],
                training_exposure_counts_equal_original071B=True,
                successful_updates_replayed=fit['successful_updates_replayed'],
                local_raw_training_sample_or_objective_recalculation=False,
                evidence_boundary='fit-sampling.json, training.jsonl and preparation remain on server. '
                    'Local verifies returned full grid argmin, declared FP32 grid and exposure aggregates; '
                    'not the raw per-anchor reconstruction or objective from coordinates.',
                reviewed_implementation='constant_gate.collect_training_points retains repeated exposures, '
                    'per-anchor all-valid denominator, zero-supervision anchors and R-outside fixed loss; '
                    'constant_gate_fit saves fit.json before the sole V forward. No V selection.')


def delta(current, control):
    return {weight: {k: current[weight][k]-control[weight][k]
                     for k in ('mae', 'rmse')}
            for weight in WEIGHTS}


def constant_comparison(tag, common, benchmark):
    report = read(CONSTANT/tag/'objective-summary.json.gz')
    rows = read(CONSTANT/tag/'scene-summary.json.gz')
    objective = report['objectives']
    assert report['status'] == 'COMPLETED' and report['model_forwards'] == 0 and not report['test_opened']
    assert all(report[k] for k in ('population_and_masks_exact', 'source_population_verified',
                                 'source_position_metrics_verified'))
    getter = lambda row, r, m, w: row['regions'][r]['methods'][m][w]
    actual = audit_pool(rows, {r: objective['regions'][r]['methods'] for r in REGIONS}, getter)
    for r in REGIONS:
        for own, shared in (('initial', common), ('final', '072A_constant'),
                            ('support_switch', 'support_switch')):
            for w in WEIGHTS:
                assert actual[r][own][w]['unit_count'] == benchmark[r][shared][w]['unit_count']
                for k in ('mae', 'mse', 'rmse'):
                    equal(actual[r][own][w][k], benchmark[r][shared][w][k], k)
        for name, weights in objective['regions'][r]['paired'].items():
            for w, target in weights.items():
                parts = [row['regions'][r]['paired'][name][w] for row in rows]
                n = sum(x['unit_count'] for x in parts)
                assert n == target['unit_count']
                for metric in ('mae', 'mse', 'log1p'):
                    for key in ('signed_sum', 'negative_unit_count', 'positive_unit_count', 'exact_tie_unit_count'):
                        equal(sum(x[metric][key] for x in parts), target[metric][key], 'paired_'+metric+'_'+key)
                    equal(target[metric]['signed_sum']/n if n else None,
                          target[metric]['mean_difference'], 'paired_mean_'+metric)
    for c, result in objective['components'].items():
        part = [r for r in rows if r['component_id'] == c]
        assert result['scene_ids'] == [r['scene_id'] for r in part]
        audit_pool(part, {r: result['regions'][r]['methods'] for r in REGIONS}, getter)
    return dict(four_weights=actual['all'],
                paired_constant_minus_control=objective['regions']['all']['paired']['final__minus__initial'],
                source_array_equality_is_server_evidence=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    manifest = read(ROOT/'configs/data/mtad-level1-scene3000.json')
    s, rows, run = [read(SUMMARY/f) for f in ('summary.json.gz', 'scene-summary.json.gz', 'run.json')]
    assert s['status'] == run['status'] == 'COMPLETED'
    assert run['model_forwards'] == run['optimizer_updates'] == 0 and not run['test_opened']
    assert len(rows) == s['scene_count'] == run['completed_scenes'] == 300
    assert len(s['methods']) == 13 and s['scene_ids'] == manifest['splits']['V_select']
    assert [r['scene_ids'][0] for r in rows] == s['scene_ids']
    for row in rows:
        assert row['component_id'] == manifest['components'][row['scene_ids'][0]]
    for key, value in s['population'].items():
        assert sum(row['population'][key] for row in rows) == value
    for key in s['position_population'].keys()-{'point_coverage'}:
        assert sum(row['position_population'][key] for row in rows) == s['position_population'][key]
    equal(s['position_population']['covered_target_points']/s['position_population']['available_target_points'],
          s['position_population']['point_coverage'], 'coverage')
    getter = lambda row, r, m, w: row['position_weightings'][r][m][w]
    actual = audit_pool(rows, s['position_weightings'], getter)
    components = {}
    for c, result in s['components'].items():
        part = [r for r in rows if r['component_id'] == c]
        assert result['scene_ids'] == [r['scene_ids'][0] for r in part]
        components[c] = audit_pool(part, result['position_weightings'], getter)
    assert len(components) == 38
    largest = min(components, key=lambda k: (-components[k]['all']['anchor']['point_equal']['unit_count'], k))
    assert largest == s['largest_component_id']
    remainder = [r for r in rows if r['component_id'] != largest]
    assert len(remainder) == s['remaining_components']['scene_count'] == 170
    remaining = audit_pool(remainder, s['remaining_components']['position_weightings'], getter)
    # Existing ten outputs must remain the original 071 table, including point quantiles.
    old = read(RESULTS/'LEVEL1-POSITION-BENCHMARK-20261023-071-summary/summary.json.gz')
    assert old['population'] == s['population'] and old['position_population'] == s['position_population']
    for r in REGIONS:
        for method in old['methods']:
            for w in WEIGHTS:
                assert old['position_weightings'][r][method][w] == s['position_weightings'][r][method][w]
    summary_launch = execution(SUMMARY, 'compare_position_benchmarks.py')
    argv = summary_launch['arguments']
    flags = {'071A_log': '--log-run071A', '071B_linear': '--linear-run071B',
             '071C_lstm': '--lstm-run071C', '071D_transformer': '--transformer-run071D',
             '070_linear': '--source-070', '072A_constant': '--constant-run072A',
             '072B_lstm4000': '--lstm-run072B', '072C_transformer4000': '--transformer-run072C'}
    for method, flag in flags.items():
        assert s['inputs'][method] == argv[argv.index(flag)+1]
    fit = fit_a(manifest)
    assert s['source_system'] == fit['source_system']
    for k in ('selected_logit', 'selected_gate'):
        assert s['constant_fit'][k] == fit[k]
    assert s['constant_fit']['train_only']
    comparisons = {tag: constant_comparison(tag, shared, actual) for tag, shared in
                   (('versus_fixed095', 'fixed_prior095'), ('versus_learned071B', '071B_linear'))}
    pairs = [(m, c) for m in NEW for c in ('fixed_prior095', 'support_switch', '071B_linear')]
    pairs.extend([('072B_lstm4000', '071C_lstm'), ('072C_transformer4000', '071D_transformer')])
    results = {}
    getall = lambda r, m: r['position_weightings']['all'][m]['point_equal']
    for m, c in pairs:
        results[m+'__minus__'+c] = dict(four_weights=delta(actual['all'][m], actual['all'][c]),
            scene_counts=method_counts(rows, getall, m, c),
            component_counts=method_counts(list(s['components'].values()), getall, m, c),
            remaining_170=delta(remaining['all'][m], remaining['all'][c]))
    pair = s['extended_sequence_pairing']
    assert pair['status'] == 'PAIR_VERIFIED' and pair['fresh_training']
    assert pair['matched_updates'] == 4000 and pair['matched_anchors'] == 32000
    assert all(pair[k] for k in ('actual_successful_batches_exact', 'ordered_training_scene_ids_exact', 'sampling_exposure_exact'))
    for prefix in pair['prefixes'].values():
        assert prefix['matched_prefix_updates'] == 1000
        assert all(prefix[k] for k in ('initial_parameters_exact', 'initial_sampler_rng_exact', 'actual_first1000_batches_exact'))
    # Region-specific sums explain how much of a method's all-point MAE change lies in R.
    contributions = {}
    for method in ('071B_linear', '072B_lstm4000', '072C_transformer4000'):
        controls = ('fixed_prior095',) if method == '071B_linear' else ('071B_linear', 'fixed_prior095')
        for control in controls:
            contributions[method+'__minus__'+control] = {
                r: (actual[r][method]['point_equal']['mae']-actual[r][control]['point_equal']['mae'])*
                    actual[r][method]['point_equal']['unit_count']/442263.
                for r in REGIONS[1:] if actual[r][method]['point_equal']['unit_count']}
            equal(sum(contributions[method+'__minus__'+control].values()),
                  actual['all'][method]['point_equal']['mae']-actual['all'][control]['point_equal']['mae'], 'region_MAE_contribution')
    result = dict(status='COMPLETED_LOCAL_AGGREGATE_AUDIT', scope=__doc__, model_forwards=0,
        optimizer_updates=0, archive_reads=0, test_opened=False, summary_execution=summary_launch,
        checks=CHECKS, population=s['population'], position_population=s['position_population'],
        constant_fit=fit, constant_comparisons=comparisons, benchmark_four_weights=actual['all'],
        region_four_weights={r: actual[r] for r in REGIONS[1:]}, pairwise_comparisons=results,
        point_tails_from_server_arrays={r: {m: s['position_weightings'][r][m]['point_equal'] for m in FOCUS}
                                       for r in ('all', 'anchor_outside_helper_supported')},
        largest_component=dict(id=largest, scene_count=130, metrics=components[largest]['all']),
        remaining_components=dict(scene_count=170, metrics=remaining['all']),
        MAE_difference_contribution_to_all_points=contributions,
        server_extended_pairing={k: v for k, v in pair.items() if k != 'exposure'},
        limitations=['No local raw NPZ/checkpoint/training JSONL or fit-sampling audit; raw pairing/reconstruction is server evidence.',
            'Global P95/P99 are copied from server array evaluation, not re-derived from scene quantiles.',
            'R is anchor_outside_helper_supported; anchor_inside_support is outside R. Sequence residuals may change both regions.',
            'T grid has its unique minimum at 0.028, not a globally monotonic preference for zero. V was not searched for an optimal constant.',
            'Beating this finite T-fitted constant supports only this comparator-specific increment; it does not exclude all global constants or isolate extra features/time dependence as a unique cause.',
            'T synthetic groups, repeated-anchor weighting and V deployed full-population weighting differ. Returned outputs do not identify the cause of historical gradients or isolate distribution shift.',
            '071B beats the 4000-step Transformer on RMSE/P99 but loses MAE/P95. No overall superiority claim passes the declared criteria.',
            '072B/C have the same initial tensors and first1000 samples as prior fits per server audit, not an in-run step1000 checkpoint or guaranteed identical optimizer trajectory.',
            'V is repeatedly reused development data, not independent test confirmation; no significance claim.'])
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    print(json.dumps({'status': result['status'], 'checks': CHECKS,
        'constant_gate': fit['selected_gate'], 'grid_descending_edges': fit['curve_decreasing_edges'],
        'headlines': {m: actual['all'][m]['point_equal'] for m in FOCUS},
        'transformer_vs_gate': results['072C_transformer4000__minus__071B_linear']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
