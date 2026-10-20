"""Recompute the returned TCM endpoint evidence, without model or dataset access.

Compare TCM1000 and TCM4000 only on the same 029 fragments. Report the new
full-V endpoint separately. No threshold selection or new GPU work.
"""
from pathlib import Path
import gzip
import json
import math
import statistics

ROOT = Path(__file__).resolve().parents[3]
RESULTS = ROOT / 'HandOver/results'
NEW = RESULTS / 'BASELINE-TCM-COVERAGE-20261020-033'
OLD = RESULTS / 'COMPLETION-CHECK-20261019-029'
TRAIN_OLD = RESULTS / 'STAGEWISE-COMPLETION-20261019-028/020'


def read(path):
    return json.loads(gzip.decompress(path.read_bytes())) if path.suffix == '.gz' else json.loads(path.read_text())


def metric_paths():
    for method in ('model', 'state_hold'):
        for direction in ('all', 'forward', 'reverse'):
            yield ('one_step', method, direction)
        for group, keys in [('regions', ('all', 'inside_support', 'before_support', 'after_support')),
                            ('input_distance_metrics', ('le_300s', 'gt_300_le_600s',
                             'gt_600_le_1800s', 'gt_1800s', 'no_input'))]:
            for key in keys:
                yield ('query', method, group, key)


def at(row, path):
    for key in path:
        row = row[key]
    return row


def pooled(rows, path):
    items = [at(row, path) for row in rows]
    totals = {key: sum(item[key] for item in items) for key in
              ('query_count', 'evaluable_query_count', 'missing_reference_query_count',
               'feature_elements', 'squared_error_sum')}
    totals['feature_mse'] = (totals['squared_error_sum'] / totals['feature_elements']
                             if totals['feature_elements'] else None)
    return totals


def compare(rows):
    output = {}
    for path in metric_paths():
        if path[1] != 'model':
            continue
        model = pooled(rows, path)
        control = pooled(rows, (path[0], 'state_hold', *path[2:]))
        output['/'.join((path[0], *path[2:]))] = dict(model=model, state_hold=control,
            model_vs_hold_reduction=(1 - model['feature_mse'] / control['feature_mse'])
            if control['feature_mse'] else None)
    outside = {}
    for method in ('model', 'state_hold'):
        parts = [output['query/regions/'+region][method] for region in ('before_support', 'after_support')]
        total = {key: sum(part[key] for part in parts) for key in parts[0] if key != 'feature_mse'}
        total['feature_mse'] = total['squared_error_sum'] / total['feature_elements'] if total['feature_elements'] else None
        outside[method] = total
    outside['model_vs_hold_reduction'] = (1-outside['model']['feature_mse']/outside['state_hold']['feature_mse']) if outside['state_hold']['feature_mse'] else None
    output['query/regions/outside_support'] = outside
    return output


def main():
    report, old = read(NEW/'completion-evaluation.json.gz'), read(OLD/'reference.json.gz')
    run, summary, history = (read(NEW/name) for name in ('run.json.gz', 'summary.json.gz', 'history.json.gz'))
    old_run, old_history = read(TRAIN_OLD/'run.json.gz'), read(TRAIN_OLD/'history.json.gz')
    launcher, request = read(NEW/'launcher-status.json'), read(NEW/'request.json')
    assert launcher['status'] == run['status'] == 'COMPLETED'
    assert launcher['returncode'] == int((NEW/'exit_code.txt').read_text()) == 0
    assert run['phase'] == 'tcm' and run['training_stages'] == ['TCM']
    assert [(x['stage'], x['update']) for x in history] == [('TCM', i) for i in range(1, 4001)]
    assert run['config']['updates'] == dict(TCM=4000, TAM_FROZEN=0, TFM_FROZEN=0, JOINT=0)
    assert not run['selection_performed'] and not run['test_opened']
    assert run['margin_calibration'] is None and run['effective_margin'] == 1.0
    assert all(math.isfinite(x[k]) for x in history for k in ('loss', 'gradient_norm', 'seconds'))
    assert summary['normalizer'] == read(OLD/'run.json.gz')['reference_normalizer']
    paths = list(metric_paths())
    for path in paths:
        for key, value in pooled(report['per_fragment'], path).items():
            expected = at(report, path)[key]
            assert value == expected or (value is not None and expected is not None
                                         and math.isclose(value, expected, rel_tol=1e-12, abs_tol=1e-10))
    old_rows = {x['node_id']: x for x in old['per_fragment']}
    new_rows = {x['node_id']: x for x in report['per_fragment']}
    common = [new_rows[key] for key in old_rows]
    assert len(old_rows) == len(common) == 86 and len(new_rows) == 1635
    for key, row in old_rows.items():
        other = new_rows[key]
        for field in ('scene_id', 'node_id', 'node_index', 'identity', 'native_count',
                      'reference_declared', 'has_complete_reference'):
            assert row[field] == other[field]
        for path in paths:
            for field in ('query_count', 'evaluable_query_count', 'missing_reference_query_count', 'feature_elements'):
                assert at(row, path)[field] == at(other, path)[field]
            if path[1] == 'state_hold':
                assert at(row, path) == at(other, path)
    baseline = compare(list(old_rows.values()))
    endpoint = compare(common)
    changes = {key: dict(tcm1000=baseline[key], tcm4000=endpoint[key],
                 endpoint_mse_reduction=1-endpoint[key]['model']['feature_mse']/baseline[key]['model']['feature_mse']
                 if baseline[key]['model']['feature_mse'] else None) for key in baseline}
    scene_ids = [x['scene_id'] for x in report['per_scene']]
    per_scene = {sid: compare([x for x in report['per_fragment'] if x['scene_id'] == sid]) for sid in scene_ids}
    old_scene = {sid: compare([x for x in old['per_fragment'] if x['scene_id'] == sid])
                 for sid in [x['scene_id'] for x in old['per_scene']]}
    windows = {}
    for start, end in [(1, 100), (1, 1000), (901, 1000), (1001, 2000), (2001, 3000), (3001, 4000), (3901, 4000)]:
        part = history[start-1:end]
        windows[f'{start}-{end}'] = dict(mean_loss=statistics.mean(x['loss'] for x in part),
            median_loss=statistics.median(x['loss'] for x in part),
            min_loss=min(x['loss'] for x in part), max_loss=max(x['loss'] for x in part))
    old_tcm = [x for x in old_history if x['stage'] == 'TCM']
    coverage = summary['coverage']['TCM']
    old_coverage = read(TRAIN_OLD/'coverage.json.gz')['TCM']
    config_differences = {key: dict(old=old_run['config'].get(key), new=run['config'].get(key))
                          for key in old_run['config'].keys() | run['config'].keys()
                          if old_run['config'].get(key) != run['config'].get(key)}
    out = dict(status='COMPLETED', source_directory=str(NEW.relative_to(ROOT)),
        source_029=str(OLD.relative_to(ROOT)), source_020=str(TRAIN_OLD.relative_to(ROOT)),
        scope='Returned frozen feature metrics only; common four-scene endpoint comparison and separate full V.',
        verification=dict(raw_command=request['argv'], phase=run['phase'], cursor=run['cursor'],
            launcher_returncode=launcher['returncode'], history_rows=len(history), finite_history=True,
            all_report_metrics_recomputed=True, common_fragments=86, common_metadata_and_denominators_equal=True,
            common_state_hold_equal=True, normalizer_equal=True, config_differences=config_differences,
            actual_weights_loaded_locally=False, test_opened=False, inference_calls=0, optimizer_updates=0),
        timing=dict(launcher_seconds=launcher['elapsed_seconds'], trainer_seconds=run['elapsed_seconds'],
            update_seconds=sum(x['seconds'] for x in history), final_evaluation_seconds=run['final_evaluation_seconds'],
            other_trainer_seconds=run['elapsed_seconds']-sum(x['seconds'] for x in history)-run['final_evaluation_seconds'],
            advisory_issued=launcher['time_advisory_issued']),
        training=dict(windows=windows, gradient_max=max(x['gradient_norm'] for x in history),
            clipped_steps=sum(x['gradient_norm'] > run['config']['gradient_clip'] for x in history),
            track_presentations=coverage['track_presentations'], unique_tracks=len(coverage['tracks']),
            old_unique_tracks=len(old_coverage['tracks']),
            first1000_loss_max_difference=max(abs(a['loss']-b['loss']) for a,b in zip(old_tcm,history)),
            first1000_loss_identical_count=sum(a['loss']==b['loss'] for a,b in zip(old_tcm,history))),
        full_V=compare(report['per_fragment']), common_four=changes,
        full_V_scene_better_than_hold={key: sum(values[key]['model']['feature_mse'] < values[key]['state_hold']['feature_mse']
            for values in per_scene.values()) for key in ('one_step/all', 'query/regions/all')},
        common_fragment_changes={label: {kind: sum(op(at(new_rows[key], path)['feature_mse'], at(row, path)['feature_mse'])
            for key,row in old_rows.items()) for kind,op in [('better', lambda a,b:a<b),
                                                           ('equal', lambda a,b:a==b), ('worse', lambda a,b:a>b)]}
            for label,path in [('one_step', ('one_step','model','all')),
                               ('query', ('query','model','regions','all'))]},
        per_scene_tcm4000={sid: {key: dict(
            model_mse=value['model']['feature_mse'],
            state_hold_mse=value['state_hold']['feature_mse'],
            evaluable_queries=value['model']['evaluable_query_count'])
            for key,value in metrics.items() if key in
            ('one_step/all', 'query/regions/all', 'query/regions/before_support', 'query/regions/after_support')}
            for sid,metrics in per_scene.items()},
        common_per_scene_tcm1000={sid: {key: value['model']['feature_mse']
            for key,value in metrics.items() if key in ('one_step/all', 'query/regions/all')}
            for sid,metrics in old_scene.items()},
        limits=['Feature MSE is not physical position/speed/course error or downstream association.',
                'Single seed and no independent repeat; first 1000 optimization histories are not bitwise identical.',
                'No old TCM1000 full-V evaluation; never compare full V to the old four-scene mean.',
                'Visible-support output is interpolation, not learned completion.',
                'No raw predictions or local checkpoint load; aggregated returned SSE and counts were recomputed.'])
    target = Path(__file__).with_name('033-audit.json')
    target.write_text(json.dumps(out, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(status=out['status'], common_query_reduction=changes['query/regions/all']['endpoint_mse_reduction'],
                          common_one_step_reduction=changes['one_step/all']['endpoint_mse_reduction'],
                          unique_training_tracks=out['training']['unique_tracks']), ensure_ascii=False))


if __name__ == '__main__':
    main()
