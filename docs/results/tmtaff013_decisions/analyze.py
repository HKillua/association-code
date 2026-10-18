"""Recompute 013 decision diagnostics from returned JSON only; no model/data reads.

Run from any directory with Python + numpy. Writes diagnostics.json beside this file.
The source JSON files and code remain unchanged. Times below are arithmetic scenarios,
not forecasts, confidence intervals, GPU utilization measurements or budget guarantees.
"""
from pathlib import Path
import json
import math
import time

import numpy as np


def quantiles(values):
    return dict(zip(('min', 'p05', 'p25', 'median', 'p75', 'p95', 'max'),
                    map(float, np.quantile(values, (0, .05, .25, .5, .75, .95, 1)))))


def percent(numerator, denominator):
    return 100 * numerator / denominator


def main():
    started = time.monotonic()
    directory = Path(__file__).resolve().parent
    root = directory.parents[2]
    source = root / 'HandOver/results/TMTAFF-TIME-COST-20261018-013'
    run, history, coverage = [json.loads((source / 'files' / name).read_text())
                              for name in ('run.json', 'history.json', 'coverage.json')]
    calibration = run['margin_calibration']
    samples = calibration['samples']
    distances = np.asarray([d for b in samples for d in b['distances']], dtype=np.float32)
    labels = np.asarray([y for b in samples for y in b['labels']])
    pairs = [(g['scene_index'], i) for b in samples for g in b['groups'] for i in g['pair_indices']]
    assert len(distances) == len(labels) == len(pairs) == 4096
    positive = distances[labels == 1]
    negative = distances[labels == 0]
    assert len(positive) == len(negative) == 2048
    median = float(np.median(negative.astype(np.float64)))
    assert median == calibration['effective_margin'] == run['effective_margin']

    def hinge_report(margin):
        # Match scalar conversion and subtraction in the actual FP32 training loss.
        hinge = np.maximum(np.float32(margin) - negative, np.float32(0)).astype(np.float64)
        p = positive.astype(np.float64)
        return dict(margin=margin, training_dtype_margin=float(np.float32(margin)),
                    negative_strict_active_count=int((hinge > 0).sum()),
                    negative_strict_active_fraction=float((hinge > 0).mean()),
                    negative_ties_count=int((negative == np.float32(margin)).sum()),
                    positive_class_loss_mean=float(.5 * p.mean()),
                    negative_class_loss_mean=float(.5 * np.square(hinge).mean()),
                    positive_loss_contribution=float(.25 * p.mean()),
                    negative_loss_contribution=float(.25 * np.square(hinge).mean()),
                    total_loss_mean=float(.25 * (p.mean() + np.square(hinge).mean())),
                    positive_dloss_dd_mean=.5,
                    negative_dloss_dd_mean=float(-hinge.mean()),
                    active_negative_dloss_dd_mean=float(-hinge[hinge > 0].mean()) if (hinge > 0).any() else None,
                    negative_abs_to_positive_abs_derivative_ratio=float(hinge.mean() / .5),
                    derivative_scope='per-example d loss / d distance, not parameter gradient; batch mean divides by 64')

    before = hinge_report(run['requested_model_config']['margin'])
    after = hinge_report(median)
    assert after['negative_strict_active_count'] == calibration['negative_hinge']['strict_active_count']
    assert math.isclose(after['total_loss_mean'], calibration['loss']['total'], abs_tol=1e-10)
    margin_report = dict(source_fields=['run.margin_calibration.samples', 'run.requested_model_config.margin'],
                        scope=calibration['scope'], before_requested_margin=before, after_calibrated_margin=after,
                        distance_quantiles={name: quantiles(values) for name, values in (('positive', positive), ('negative', negative))},
                        presentations=4096, unique_pairs=len(set(pairs)), duplicate_presentations=4096-len(set(pairs)),
                        unique_scenes=len({s for s, _ in pairs}),
                        positive_below_margin_fraction=float((positive < np.float32(median)).mean()),
                        negative_below_positive_median_fraction=float((negative < np.median(positive)).mean()),
                        interpretation='same pre-TAM-update distances under two margins; no before/after representation-performance comparison')

    population = run['sampling']['association']
    node_count = 0
    for scene in population['scenes']:
        discriminant = 1 + 8 * scene['pair_count']
        square_root = math.isqrt(discriminant)
        assert square_root ** 2 == discriminant
        nodes = (1 + square_root) // 2
        assert nodes * (nodes - 1) // 2 == scene['pair_count']
        node_count += nodes
    assert node_count == 54792
    index_to_id = {s['scene_index']: s['scene_id'] for s in population['scenes']}

    def union(fields, stages):
        return set().union(*(set(map(tuple, coverage[s][fields])) for s in stages))

    def positive_pairs(stage):
        item = coverage[stage]
        pos, neg = set(item['positive_scene_ids']), set(item['negative_scene_ids'])
        # These observed stage samples have disjoint positive/negative scenes, so
        # coverage pair keys can be classified without opening any labels or ZIP.
        assert not pos & neg
        return {(s, i) for s, i in item['pairs'] if index_to_id[s] in pos}

    stages = tuple(run['config']['updates'])
    observed_coverage = {}
    for stage in stages:
        c = coverage[stage]
        pos_count = len(positive_pairs(stage))
        observed_coverage[stage] = dict(
            unique_scenes=c['unique_scene_count'], scene_percent=percent(c['unique_scene_count'], population['scene_count']),
            unique_tracks=c['unique_track_count'], track_percent_all_valid=percent(c['unique_track_count'], node_count),
            unique_pairs=c['unique_pair_count'], all_candidate_percent=percent(c['unique_pair_count'], population['pair_count']),
            unique_positive_pairs=pos_count, positive_pool_percent=percent(pos_count, population['positive_pair_count']),
            unique_negative_pairs=c['unique_pair_count']-pos_count,
            track_presentations=c['track_presentations'], pair_presentations=c['pair_presentations'],
            positive_pair_presentations=c['positive_pair_presentations'])
    all_tracks = union('tracks', stages)
    representation_tracks = union('tracks', ('TCM', 'JOINT'))
    association_pairs = union('pairs', ('TAM_FROZEN', 'JOINT'))
    fusion_positive_pairs = positive_pairs('TFM_FROZEN') | positive_pairs('JOINT')
    combined_coverage = dict(
        any_stage_unique_scenes=len(set().union(*(set(coverage[s]['scenes']) for s in stages))),
        any_stage_unique_tracks=len(all_tracks), any_stage_track_percent=percent(len(all_tracks), node_count),
        tcm_or_joint_gradient_unique_tracks=len(representation_tracks),
        tcm_or_joint_gradient_track_percent=percent(len(representation_tracks), node_count),
        association_gradient_unique_pairs=len(association_pairs),
        association_gradient_all_candidate_percent=percent(len(association_pairs), population['pair_count']),
        fusion_gradient_unique_positive_pairs=len(fusion_positive_pairs),
        fusion_gradient_positive_pool_percent=percent(len(fusion_positive_pairs), population['positive_pair_count']),
        all_stage_pair_presentations=sum(coverage[s]['pair_presentations'] for s in stages),
        fusion_positive_pair_presentations=sum(coverage[s]['positive_pair_presentations'] for s in ('TFM_FROZEN','JOINT')))

    stage_diagnostics = []
    for row in history:
        norm, total = row['gradient_norm'], row['loss']
        item = dict(stage=row['stage'], update=row['update'], loss=total, gradient_norm_pre_clip=norm,
                    gradient_clip=run['config']['gradient_clip'],
                    nominal_clip_factor=min(1., run['config']['gradient_clip'] / (norm + 1e-6)),
                    seconds=row['seconds'], learning_rate=row['learning_rate'])
        if 'fusion_valid_counts' in row:
            denominator = row['positive_pairs'] * run['requested_model_config']['length']
            item.update(positive_pair_presentations=row['positive_pairs'],
                        query_positions_per_feature=denominator,
                        valid_query_positions_per_feature=row['fusion_valid_counts'],
                        supervision_percent_per_feature=[percent(v, denominator) for v in row['fusion_valid_counts']],
                        unique_endpoints=row['unique_endpoints'],
                        completion_target_elements=row['completion_target_elements'],
                        fusion_component_losses=row['fusion_component_losses'])
            pieces = dict(association=row['association_loss'],
                          weighted_completion=run['config']['completion_weight'] * row['completion_loss'],
                          fusion=row['fusion_loss'])
            assert math.isclose(sum(pieces.values()), total, rel_tol=1e-10)
            item.update(loss_terms=pieces, scalar_loss_percent={k: percent(v, total) for k, v in pieces.items()},
                        raw_completion_loss=row['completion_loss'])
        if 'selection' in row:
            item['selection'] = row['selection']
        stage_diagnostics.append(item)

    stage_times = {stage: [r['seconds'] for r in history if r['stage'] == stage] for stage in stages}
    validation_times = [r['seconds'] for r in run['validation_timings']]
    calibration_seconds = calibration['seconds']
    outer_seconds = float((source / 'elapsed_seconds.txt').read_text().strip())
    training_seconds = sum(r['seconds'] for r in history)
    validation_seconds = sum(validation_times)
    final_incomplete_seconds = run['final_evaluation_seconds']
    timing = dict(outer_seconds=outer_seconds, run_seconds=run['elapsed_seconds'],
                  calibration_seconds=calibration_seconds, training_seconds=training_seconds,
                  validation_seconds=validation_seconds, failed_final_seconds=final_incomplete_seconds,
                  run_unitemized_seconds=run['elapsed_seconds']-calibration_seconds-training_seconds-validation_seconds-final_incomplete_seconds,
                  outside_run_seconds=outer_seconds-run['elapsed_seconds'],
                  calibration_plus_validation_percent_outer=percent(calibration_seconds+validation_seconds, outer_seconds),
                  measured_seconds_by_stage=stage_times,
                  training_percent_outer=percent(training_seconds, outer_seconds))

    plans = {'100_100_100_20': dict(zip(stages, (100,100,100,20))),
             '90pct_expected_access_ladder': dict(zip(stages, (1971,1178,1178,1178)))}
    scenarios = {}
    for name, updates in plans.items():
        components = {s: [updates[s]*min(stage_times[s]), updates[s]*max(stage_times[s])] for s in stages}
        step_bounds = [sum(v[i] for v in components.values()) for i in (0,1)]
        scenarios[name] = dict(updates=updates, training_seconds_by_stage=components,
                               training_only_hours=[t/3600 for t in step_bounds],
                               training_plus_single_measured_calibration_hours=[(t+calibration_seconds)/3600 for t in step_bounds],
                               calibration_seconds_per_planned_update=calibration_seconds/sum(updates.values()),
                               expected_access_percent=dict(
                                   TCM_tracks=100*(1-(1-run['config']['batch_tracks']/node_count)**updates['TCM']),
                                   **{s+'_per_class_scenes':100*(1-(1-1/population['scene_count'])**updates[s])
                                      for s in ('TAM_FROZEN','TFM_FROZEN','JOINT')}),
                               validation_scenarios={})
        for frequency in (2,20,100):
            count = sum(math.ceil(updates[s]/frequency) for s in ('TAM_FROZEN','JOINT'))
            val_bounds = [count*min(validation_times), count*max(validation_times)]
            scenarios[name]['validation_scenarios'][str(frequency)] = dict(
                frequency=frequency, validation_passes=count, validation_only_hours=[v/3600 for v in val_bounds],
                training_calibration_validation_hours=[(step_bounds[i]+calibration_seconds+val_bounds[i])/3600 for i in (0,1)])

    result = dict(
        schema='tmtaff013_decision_diagnostics_v1',
        provenance=dict(source_directory=str(source.relative_to(root)),
                        execution_version_record=(source/'actual_commit.txt').read_text().strip(),
                        source_fields=['run.json', 'history.json', 'coverage.json', 'elapsed_seconds.txt', 'exit_code.txt'],
                        procedure='NumPy/stdlib arithmetic only; no model, ZIP, reference, checkpoint or T_test reads',
                        source_status=run['status'], source_exit_code=int((source/'exit_code.txt').read_text().strip()),
                        test_opened=run['test_opened']),
        calibration=margin_report,
        population=dict(scenes=population['scene_count'], valid_tracks=node_count,
                        track_count_derivation='sum n from each full canonical candidate count n*(n-1)/2 in run.sampling.association.scenes',
                        all_candidate_pairs=population['pair_count'], positive_pairs=population['positive_pair_count'],
                        negative_pairs=population['negative_pair_count']),
        stage_coverage=observed_coverage, combined_coverage=combined_coverage,
        stage_diagnostics=stage_diagnostics, timing=timing, arithmetic_scenarios=scenarios,
        scenario_limitations=[
            'Two observed, different-batch step times provide arithmetic endpoints, not confidence/prediction intervals.',
            'Uses the original 013 implementation, hardware/load and microbatch configuration; no claimed speedup from future caches.',
            'One measured calibration is amortized once; recalibration is not multiplied by training updates.',
            'Final complete fusion/association evaluation cost is unknown; failed 79-second partial pass is not its total cost.',
            'Preparation, IO/checkpoints and resource contention are excluded; arithmetic totals are not approved budgets.',
            'Changing update counts or validation frequency changes training/selection protocol and needs an explicit plan.',
            '90% ladder concerns expected TCM-track or per-class-scene access, not 90% pair coverage or convergence.'],
        script_elapsed_seconds=time.monotonic()-started)
    path = directory / 'diagnostics.json'
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(output=str(path), script_elapsed_seconds=result['script_elapsed_seconds'],
                         population=result['population'], combined_coverage=combined_coverage,
                         timing=timing, scenarios=scenarios), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
