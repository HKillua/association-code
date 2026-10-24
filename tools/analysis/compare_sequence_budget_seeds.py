"""097 CPU table: two within-seed budget changes, no inference or training.

Read completed original 090C/097A reference tables and their complete scene
tables. This verifies returned aggregate bindings, not checkpoint/NPZ replay.
Different seeds need not share parameters or ordered training samples.
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
from pathlib import Path
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

ROOT = Path(__file__).resolve().parents[2]
METRICS = ('mae', 'rmse', 'p95', 'p99')


def _json(path):
    path = Path(path)
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt', encoding='utf-8') as handle:
        return json.load(handle)


def _require(value, message):
    if not value:
        raise ValueError(message)


def _contract(seed):
    filename = ('frozen-position-adapter-full-finetune-level1.json' if seed == 20261023
                else 'sequence-budget-replication-level1.json')
    return _json(ROOT/'configs'/filename), _json(ROOT/'configs/data/mtad-level1-scene3000.json')


def _bind(path, seed):
    path = Path(path).resolve()
    run, produced = _json(path/'run.json'), _json(path/'summary.json')
    config, manifest = _contract(seed)
    steps = config['training']['updates']
    for item in (run, produced):
        _require(item.get('status') == 'COMPLETED' and item.get('test_opened') is False
            and item.get('configuration') == config and item.get('optimizer_updates') == steps,
            'budget arm must be completed at its own declared seed/configuration/endpoint')
    _require(run.get('phase') == 'completed' and run.get('completed_endpoints') == ['initial', 'final']
        and run.get('validation_selection') is False, 'budget arm has incomplete fixed endpoints')
    _require(produced.get('source_teacher_updates') == steps
        and produced.get('additional_updates') == steps and produced.get('total_stage_updates') == 2*steps,
        'budget source/additional update accounting differs')
    if seed == 20261021:
        _require(run.get('scope') == '097-sequence-added-budget-seed-sensitivity'
            and run.get('source_identity') == '073C' and run.get('experiment_id') == '097A',
            'second seed must be declared097A from073C')
    table_dir = path/'reference-comparison'
    table_run, table = _json(table_dir/'run.json'), _json(table_dir/'summary.json.gz')
    rows = _json(table_dir/'scene-summary.json.gz')
    for item in (table_run, table):
        _require(item.get('status') == 'COMPLETED' and item.get('model_forwards') == 0
            and item.get('optimizer_updates') == 0 and item.get('test_opened') is False,
            'reference comparison must be completed and inference-free')
    _require(table.get('source_system') == run.get('source_system') == produced.get('source_system')
        and Path(table['inputs']['finetune']).resolve() == path
        and Path(table['inputs']['linear']).resolve() == Path(run['paired_control']).resolve()
        and run['paired_control'] == produced['paired_control'], 'reference table provenance differs')
    ids = manifest['splits']['V_select']
    _require(table.get('scene_ids') == ids and [row['scene_ids'][0] for row in rows] == ids
        and all(row.get('scene_count') == 1 and len(row['scene_ids']) == 1
                and row['component_id'] == manifest['components'][row['scene_ids'][0]] for row in rows),
        'reference scene population/order/components differ')
    pair = table['pairing']
    arm = pair['arms']['finetune']
    _require(pair.get('status') == 'PAIR_VERIFIED' and pair.get('matched_updates') == steps
        and arm.get('matched_updates') == steps and arm.get('matched_anchors') == steps*config['training']['batch_size']
        and all(arm.get(key) is True for key in ('source_final_parameters_exact', 'fresh_adam',
            'actual_successful_batches_exact', 'sampling_exposure_exact', 'successful_route_exposure_recomputed'))
        and arm.get('teacher_unchanged') is False and table.get('initial_output_source_final_exact') is True,
        'within-seed source/fresh-Adam/actual-sample verification is missing')
    for method in ('linear', 'finetune'):
        for weight, overall in table['position_weightings']['all'][method].items():
            values = [row['position_weightings']['all'][method][weight] for row in rows]
            n = sum(row['unit_count'] for row in values)
            _require(n == overall['unit_count'], 'scene weighting population does not reaggregate')
            for metric in ('mae', 'mse'):
                pooled = sum(row['unit_count']*row[metric] for row in values if row['unit_count'])/n if n else None
                _require((pooled is None and overall[metric] is None)
                    or (pooled is not None and math.isclose(pooled, overall[metric], rel_tol=1e-9, abs_tol=1e-7)),
                    'scene weighting metric does not reaggregate')
    baseline, final = [table['position_weightings']['all'][name]['point_equal'] for name in ('linear', 'finetune')]
    def wins(items):
        deltas = [x['position_weightings']['all']['finetune']['point_equal']['mae']
                  -x['position_weightings']['all']['linear']['point_equal']['mae'] for x in items]
        return dict(improved=sum(x < -1e-6 for x in deltas), worsened=sum(x > 1e-6 for x in deltas),
            tied=sum(abs(x) <= 1e-6 for x in deltas), total=len(deltas))
    rest = table['remaining_components']
    outside = None if rest is None else {name: rest['position_weightings']['all'][name]['point_equal']
                                        for name in ('linear', 'finetune')}
    result = dict(seed=seed, source_identity='072C' if seed == 20261023 else '073C',
        added_arm='090C' if seed == 20261023 else '097A', source_run=str(path),
        source_updates=steps, added_updates=steps, total_updates=2*steps,
        baseline=baseline, final=final, delta={key: final[key]-baseline[key] for key in METRICS},
        relative_mae_reduction=1-final['mae']/baseline['mae'],
        full_population=table['position_population'], group_population=table['population'],
        weightings={name: table['position_weightings']['all'][name] for name in ('linear', 'finetune')},
        scene_directions=wins(rows), component_directions=wins(table['components'].values()),
        outside_largest_component=outside, source_system=table['source_system'],
        within_seed_pair_verified=True)
    scene_rows = [dict(seed=seed, scene_id=row['scene_ids'][0], component_id=row['component_id'],
        baseline=row['position_weightings']['all']['linear']['point_equal'],
        final=row['position_weightings']['all']['finetune']['point_equal']) for row in rows]
    return result, scene_rows


def run(args):
    if args.cpu_threads != 4:
        raise ValueError('097 CPU comparison fixes CPU4')
    output = Path(args.output).resolve()
    paths = [Path(args.seed23_run).resolve(), Path(args.seed21_run).resolve()]
    _require(paths[0] != paths[1], 'two distinct seed runs are required')
    for path in paths:
        _require(output != path and output not in path.parents and path not in output.parents,
            'two-seed table must be separate from immutable source runs')
    if output.exists():
        raise FileExistsError('comparison output already exists; no overwrite or retry')
    arms, scenes = [], []
    for path, seed in zip(paths, (20261023, 20261021)):
        arm, rows = _bind(path, seed); arms.append(arm); scenes.extend(rows)
    _require(arms[0]['source_system'] == arms[1]['source_system']
        and arms[0]['full_population'] == arms[1]['full_population']
        and arms[0]['group_population'] == arms[1]['group_population'], 'frozen056/full evaluation populations differ')
    result = dict(status='COMPLETED', scope='097-two-within-seed-added-budget-changes',
        arms=arms, model_forwards=0, optimizer_updates=0, test_opened=False,
        checkpoint_or_raw_prediction_replay=False, quantiles_averaged=False,
        cross_seed_parameter_or_sample_equality_required=False, default_promoted=False,
        interpretation='Each seed compares its own4000 endpoint with a fresh-Adam4000 replay. Two development seeds are not independent generalization, optimizer continuation or convergence evidence.')
    output.mkdir(parents=True, exist_ok=False)
    for name, value in [('summary.json', result), ('scene-summary.json', scenes),
                        ('run.json', {key: result[key] for key in ('status', 'scope', 'model_forwards', 'optimizer_updates', 'test_opened')})]:
        (output/name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    report = ['# 两seed追加预算敏感性', '', '| seed | MAE 初→末 m | RMSE Δ m | P95 Δ m | P99 Δ m | 改善场景/组件 |',
              '|---|---:|---:|---:|---:|---|']
    for arm in arms:
        d = arm['delta']
        report.append(f"| {arm['seed']} | {arm['baseline']['mae']:.4f}→{arm['final']['mae']:.4f} | {d['rmse']:+.4f} | {d['p95']:+.4f} | {d['p99']:+.4f} | {arm['scene_directions']['improved']}/{arm['scene_directions']['total']}；{arm['component_directions']['improved']}/{arm['component_directions']['total']} |")
    report += ['', 'fresh Adam与原批次重放的固定预算干预；不自动晋级、不判断充分收敛，不要求跨seed初值/采样相同。']
    (output/'report.md').write_text('\n'.join(report)+'\n')
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('seed23-run', 'seed21-run', 'output'):
        parser.add_argument('--'+key, required=True)
    parser.add_argument('--cpu-threads', type=int, default=4)
    result = run(parser.parse_args(argv)); print(result['status'], flush=True); return 0


if __name__ == '__main__':
    raise SystemExit(main())
