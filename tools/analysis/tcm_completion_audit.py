"""095: aggregate complete 088 fragment diagnostics; no models or forward passes."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gzip
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
METHODS = ('model', 'state_hold')
REGIONS = ('all', 'inside_support', 'before_support', 'after_support')
DISTANCES = ('le_300s', 'gt_300_le_600s', 'gt_600_le_1800s', 'gt_1800s', 'no_input')
PATHS = tuple(('one_step', key) for key in ('all', 'forward', 'reverse')) + tuple(
    ('query', family, key) for family, keys in (('regions', REGIONS),
                                              ('input_distance_metrics', DISTANCES)) for key in keys)
COUNTS = ('query_count', 'evaluable_query_count', 'missing_reference_query_count', 'feature_elements')
OUTSIDE = 'query/regions/outside_support'
CONCENTRATION = ('one_step/all', 'query/regions/all', OUTSIDE,
                 'query/regions/before_support', 'query/regions/after_support')
LENGTHS = ('2-16', '17-32', '33-64', '65-96', '97-128')
LIMITS = [
    'Feature MSE uses the frozen 128-dimensional representation, not physical units.',
    'Inside-support equality follows shared observation interpolation, not learned completion.',
    'Native one-step targets are next observations; natural-query targets require four reference attributes.',
    'Only returned original metrics are aggregated; weights, raw inputs and predictions are not re-evaluated.',
    'Per-fragment metrics contain separate region and distance marginals, not joint region-by-distance data.',
    'Single seed and repeated-development V with shared identities; no independence or generalization CI claim.',
    'Posthoc subgroups and concentration do not select a deployable directional rule or change official population.',
]


def read(path):
    with Path(path).open(encoding='utf-8') as handle:
        return json.load(handle)


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def profile():
    config = read(ROOT / 'configs/tmtaff-tcm-level1.json')
    config['training'].setdefault('fusion_monitor_updates', [])
    return dict(manifest=read(ROOT / 'configs/data/mtad-level1-scene3000.json'),
                training=config['training'], model=config['model'], train_fragments=155134, validation_fragments=24599,
                dimension=config['model']['d_model'], queries=config['model']['length'])


def metric(row, method, path):
    value = row[path[0]][method]
    for key in path[1:]:
        value = value[key]
    return value


def empty():
    return dict.fromkeys((*COUNTS, 'squared_error_sum'), 0)


def accumulate(total, value):
    for key in (*COUNTS, 'squared_error_sum'):
        total[key] += value[key]


def finish(value):
    result = dict(value)
    result['feature_mse'] = value['squared_error_sum'] / value['feature_elements'] if value['feature_elements'] else None
    result['reference_coverage'] = value['evaluable_query_count'] / value['query_count'] if value['query_count'] else None
    return result


def pool(values):
    result = empty()
    for value in values:
        accumulate(result, value)
    return finish(result)


def metric_equal(left, right, label):
    for key in COUNTS:
        require(left[key] == right[key], 'population mismatch: ' + label + '/' + key)
    for key in ('squared_error_sum', 'feature_mse', 'reference_coverage'):
        x, y = left[key], right[key]
        require((x is None and y is None) or (x is not None and y is not None
                and math.isclose(x, y, rel_tol=1e-10, abs_tol=1e-7)), 'metric mismatch: ' + label + '/' + key)


def validate_metric(value, dimension, label):
    require(all(type(value[key]) is int and value[key] >= 0 for key in COUNTS), 'invalid count: ' + label)
    require(value['query_count'] == value['evaluable_query_count'] + value['missing_reference_query_count']
            and value['feature_elements'] == dimension * value['evaluable_query_count'], 'invalid denominator: ' + label)
    require(isinstance(value['squared_error_sum'], (int, float))
            and math.isfinite(value['squared_error_sum']) and value['squared_error_sum'] >= 0,
            'invalid squared error: ' + label)
    require(value['feature_elements'] > 0 or value['squared_error_sum'] == 0,
            'nonzero squared error without evaluable elements: ' + label)
    metric_equal(value, finish({key: value[key] for key in (*COUNTS, 'squared_error_sum')}), label)


def flat_metrics(row):
    result = {'/'.join(path): {method: metric(row, method, path) for method in METHODS} for path in PATHS}
    result[OUTSIDE] = {method: pool(metric(row, method, ('query', 'regions', region))
                                    for region in ('before_support', 'after_support')) for method in METHODS}
    return result


def group(rows):
    rows = list(rows)
    return {key: {method: pool(row['metrics'][key][method] for row in rows) for method in METHODS}
            for key in (*('/'.join(path) for path in PATHS), OUTSIDE)}


def contrast(pair):
    model, held = pair['model'], pair['state_hold']
    mm, hm = model['feature_mse'], held['feature_mse']
    return dict(evaluable_query_count=model['evaluable_query_count'],
                mse_difference=None if mm is None else mm - hm,
                relative_mse_change=None if mm is None or hm == 0 else mm / hm - 1,
                squared_error_difference=model['squared_error_sum'] - held['squared_error_sum'])


def summarize_units(rows, key, label):
    """Counts and concentration use every recorded unit; null units stay visible."""
    rows = list(rows)
    records = [dict(unit=str(row[label]), **contrast(row['metrics'][key])) for row in rows]
    valid = [item for item in records if item['mse_difference'] is not None]
    tolerance = 1e-12  # arithmetic tie guard, not statistical or engineering significance
    positive = sorted((item for item in valid if item['squared_error_difference'] > 0),
                      key=lambda item: (-item['squared_error_difference'], item['unit']))
    negative = sorted((item for item in valid if item['squared_error_difference'] < 0),
                      key=lambda item: (item['squared_error_difference'], item['unit']))
    worse_sum = sum(item['squared_error_difference'] for item in positive)
    better_sum = sum(item['squared_error_difference'] for item in negative)
    net = worse_sum + better_sum
    concentration = {}
    for n in (1, 5, 10):
        total = sum(item['squared_error_difference'] for item in positive[:n])
        concentration[str(n)] = dict(positive_sse_difference=total,
            share_of_all_positive_difference=total / worse_sum if worse_sum else None,
            share_of_net_difference=total / net if net else None)
    means = {method: (sum(row['metrics'][key][method]['feature_mse'] for row in rows
                              if row['metrics'][key][method]['feature_mse'] is not None) / len(valid)
                      if valid else None) for method in METHODS}
    return dict(unit_count=len(rows), evaluable_units=len(valid), missing_units=len(rows) - len(valid),
        model_better=sum(item['mse_difference'] < -tolerance for item in valid),
        model_worse=sum(item['mse_difference'] > tolerance for item in valid),
        equal_within_arithmetic_guard=sum(abs(item['mse_difference']) <= tolerance for item in valid),
        arithmetic_guard=tolerance, unit_equal_mean_mse=means,
        positive_sse_difference=worse_sum, negative_sse_difference=better_sum, net_sse_difference=net,
        top_positive=positive[:10], top_negative=negative[:10], concentration=concentration,
        interpretation='Signed net shares may exceed one; ordering is descriptive and does not remove any unit.')


def length_bucket(count):
    return LENGTHS[0 if count <= 16 else 1 if count <= 32 else 2 if count <= 64 else 3 if count <= 96 else 4]


def audit(source, manifest, expected, check=lambda: None):
    source = Path(source)
    run, summary, coverage, history = (read(source / name) for name in
                                      ('run.json', 'summary.json', 'coverage.json', 'history.json'))
    require(manifest == expected['manifest'], '095 requires the frozen Level1 manifest')
    updates = expected['training']['updates']['TCM']
    require(run.get('status') == 'COMPLETED' and run.get('phase') == 'tcm'
            and run.get('final_evaluation_status') == 'COMPLETED'
            and run.get('training_stages') == ['TCM'] and run.get('test_opened') is False
            and run.get('input_adapter') == 'native_time_v1' and run.get('split_unit') == 'scene'
            and run.get('config') == expected['training'] and run.get('selection_performed') is False,
            'source is not the fixed completed088 TCM endpoint')
    require(run['cursor'] == dict(stage=1, update=0, completed_updates=updates)
            and run['checkpoint_cursor'] == dict(stage=0, update=updates, completed_updates=updates)
            and all(run['requested_model_config'].get(key) == value for key, value in expected['model'].items()),
            '088 cursor or model configuration differs')
    require(summary.get('phase') == 'tcm' and summary.get('test_opened') is False
            and summary.get('selection_performed') is False
            and summary.get('completion_evaluation') == 'completion-evaluation.json'
            and summary['coverage'] == coverage
            and summary['normalizer']['training_splits'] == ['T_core']
            and summary['normalizer']['training_scene_ids'] == manifest['splits']['T_core'],
            '088 source summary, coverage or T-only normalizer differs')
    require(len(history) == updates and [(row['stage'], row['update']) for row in history]
            == [('TCM', index) for index in range(1, updates + 1)]
            and sum(row['track_presentations'] for row in history) == expected['train_fragments']
            and coverage['TCM']['track_presentations'] == expected['train_fragments']
            and coverage['TCM']['unique_track_count'] == expected['train_fragments']
            and coverage['TCM']['completed_epochs'] == 1
            and coverage['TCM']['scene_ids'] == manifest['splits']['T_core'],
            '088 successful updates or one-pass training population differs')
    batch_size = expected['training']['batch_tracks']
    require(updates == math.ceil(expected['train_fragments'] / batch_size),
            '088 ordered-fragment update budget differs from one pass')
    for index, row in enumerate(history):
        start = index * batch_size
        stop = min(start + batch_size, expected['train_fragments'])
        require(row['track_presentations'] == stop - start and row.get('tcm_sampling') == dict(
            strategy='ordered_fragment', epoch=1, batch_in_epoch=index + 1,
            batches_per_epoch=updates, eligible_tracks=expected['train_fragments'],
            pool_start=start, pool_stop=stop, completed_epochs=int(index + 1 == updates)),
            '088 ordered-fragment ranges or final short-tail differ')
    for stage in ('TAM_FROZEN', 'TFM_FROZEN', 'JOINT'):
        require(coverage[stage]['track_presentations'] == coverage[stage]['pair_presentations'] == 0,
                '088 source trained a downstream stage')
    check()
    complete = read(source / 'completion-evaluation.json')
    require(complete.get('not_a_replay_input') is not True and isinstance(complete.get('per_fragment'), list),
            '095 requires original full per_fragment; extracted aggregate is not a replay input')
    scene_ids = manifest['splits']['V_select']
    require(complete.get('test_opened') is False and complete['scene_count'] == len(scene_ids)
            and [row['scene_id'] for row in complete['per_scene']] == scene_ids
            and complete['fragment_count'] == len(complete['per_fragment']) == expected['validation_fragments'],
            '088 completion evaluation omits full validation population')
    dimension, queries = expected['dimension'], expected['queries']
    by_scene = {sid: [] for sid in scene_ids}
    fragments, seen = [], set()
    order = {sid: index for index, sid in enumerate(scene_ids)}
    last_order = -1
    for index, row in enumerate(complete['per_fragment']):
        if index % 100 == 0:
            check()
        sid, node, count = row['scene_id'], row['node_id'], row['native_count']
        require(sid in by_scene and order[sid] >= last_order and (sid, node) not in seen
                and row['node_index'] == len(by_scene[sid]) and type(count) is int and 2 <= count <= 128,
                'fragment identity, order, index or native length differs')
        seen.add((sid, node)); last_order = order[sid]
        for path in PATHS:
            for method in METHODS:
                validate_metric(metric(row, method, path), dimension, f'{sid}/{node}/{method}/{path}')
            for key in COUNTS:
                require(metric(row, 'model', path)[key] == metric(row, 'state_hold', path)[key],
                        'model/hold fragment population differs')
        for method in METHODS:
            for direction in ('forward', 'reverse'):
                item = metric(row, method, ('one_step', direction))
                require(item['query_count'] == item['evaluable_query_count'] == count - 1,
                        'native one-step count differs from native fragment length')
            metric_equal(metric(row, method, ('one_step', 'all')),
                         pool(metric(row, method, ('one_step', key)) for key in ('forward', 'reverse')), 'native directions')
            all_query = metric(row, method, ('query', 'regions', 'all'))
            require(all_query['query_count'] == queries, 'fragment query grid population differs')
            metric_equal(all_query, pool(metric(row, method, ('query', 'regions', key))
                                         for key in REGIONS[1:]), 'query regions')
            metric_equal(all_query, pool(metric(row, method, ('query', 'input_distance_metrics', key))
                                         for key in DISTANCES), 'query distances')
        metric_equal(metric(row, 'model', ('query', 'regions', 'inside_support')),
                     metric(row, 'state_hold', ('query', 'regions', 'inside_support')), 'shared interpolation')
        valid = row['query']['model']['regions']['all']['evaluable_query_count']
        require(type(row['has_complete_reference']) is bool and row['has_complete_reference'] == (valid > 0)
                and type(row['reference_declared']) is bool and (row['reference_declared'] or valid == 0),
                'fragment reference declaration differs')
        compact = dict(fragment=f'{sid}/{node}', scene_id=sid, native_count=count,
                       has_complete_reference=row['has_complete_reference'], metrics=flat_metrics(row))
        fragments.append(compact); by_scene[sid].append(compact)
    all_metrics = group(fragments)
    for path in PATHS:
        for method in METHODS:
            metric_equal(all_metrics['/'.join(path)][method], metric(complete, method, path),
                         'complete original aggregate/' + method + '/' + '/'.join(path))
    scenes = []
    for original in complete['per_scene']:
        sid = original['scene_id']; rows = by_scene[sid]
        require(original['fragment_count'] == len(rows)
                and original['no_complete_reference_fragment_count'] == sum(not row['has_complete_reference'] for row in rows),
                'source per_scene population differs from full fragments')
        scenes.append(dict(scene_id=sid, component_id=str(manifest['components'][sid]),
                           fragment_count=len(rows), no_complete_reference_fragment_count=original['no_complete_reference_fragment_count'],
                           reference_targets_without_input=original['reference_targets_without_input'], metrics=group(rows)))
    require(complete['no_complete_reference_fragment_count'] == sum(row['no_complete_reference_fragment_count'] for row in scenes),
            'global missing-reference fragment count differs')
    component_members = defaultdict(list)
    for scene in scenes:
        component_members[scene['component_id']].append(scene)
    components = [dict(component_id=cid, scene_ids=[row['scene_id'] for row in rows],
                       fragment_count=sum(row['fragment_count'] for row in rows), metrics=group(rows))
                  for cid, rows in sorted(component_members.items())]
    largest = min(component_members, key=lambda key: (-len(component_members[key]), key))
    outside = [row for row in scenes if row['component_id'] != largest]
    strata = {key: dict(fragment_count=sum(length_bucket(row['native_count']) == key for row in fragments),
                        metrics=group(row for row in fragments if length_bucket(row['native_count']) == key)) for key in LENGTHS}
    for collection in (scenes, components):
        regrouped = group(collection)
        for key in all_metrics:
            for method in METHODS:
                metric_equal(regrouped[key][method], all_metrics[key][method], 'scene/component regroup/' + key)
    stratum_regroup = group(strata.values())
    for key in all_metrics:
        for method in METHODS:
            metric_equal(stratum_regroup[key][method], all_metrics[key][method], 'length regroup/' + key)
    result = dict(status='COMPLETED', scope='095-original-TCM-feature-audit', model_forwards=0,
        optimizer_updates=0, checkpoint_opened=False, raw_archive_opened=False, test_opened=False,
        source_run=str(source.resolve()), source_evaluation=str((source / 'completion-evaluation.json').resolve()),
        scene_count=len(scenes), component_count=len(components), fragment_count=len(fragments),
        feature_dimension=dimension, query_count=queries * len(fragments),
        no_complete_reference_fragment_count=complete['no_complete_reference_fragment_count'],
        reference_targets_without_input=sum(len(row['reference_targets_without_input']) for row in scenes),
        metrics=all_metrics, contrasts={key: contrast(value) for key, value in all_metrics.items()},
        length_strata=strata,
        largest_component=dict(component_id=largest, scene_count=len(component_members[largest]),
                               metrics=group(component_members[largest])),
        outside_largest_component=dict(scene_count=len(outside), metrics=group(outside)),
        unit_diagnostics={kind: {key: summarize_units(rows, key, label) for key in CONCENTRATION}
                          for kind, rows, label in (('fragments', fragments, 'fragment'), ('scenes', scenes, 'scene_id'),
                                                    ('components', components, 'component_id'))},
        region_distance_joint='unavailable in original per_fragment schema; marginal tables cannot reconstruct it',
        aggregation_verified=True, limitations=LIMITS)
    return result, scenes, components


def run(args, *, expected=None):
    require(args.cpu_threads == 4, '095 fixes CPU4')
    source, output = Path(args.source_run).resolve(), Path(args.output).resolve()
    require(source != output and source not in output.parents and output not in source.parents,
            '095 output must preserve immutable source')
    if output.exists() or Path(args.output).is_symlink():
        raise FileExistsError('095 output exists; no overwrite or automatic retry')
    expected = profile() if expected is None else expected
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    state = dict(status='RUNNING', scope='095-original-TCM-feature-audit', cpu_threads=4, model_forwards=0,
                 optimizer_updates=0, checkpoint_opened=False, raw_archive_opened=False, test_opened=False,
                 time_policy='advisory', advisory_seconds=3600, advisory_reached=False)
    write(output / 'run.json', state)
    write(output / 'execution.json', dict(arguments=vars(args), source_run=str(source), source_files=[
        str(source / name) for name in ('run.json', 'summary.json', 'coverage.json', 'history.json', 'completion-evaluation.json')]))
    try:
        def check():
            if time.monotonic() - started >= 3600 and not state['advisory_reached']:
                state['advisory_reached'] = True
                write(output / 'run.json', state)
                print('ADVISORY:095 continues the same fixed original-file aggregation.', flush=True)
        result, scenes, components = audit(source, read(args.manifest), expected, check)
        for name, values in (('scene-summary.json.gz', scenes), ('component-summary.json.gz', components)):
            with gzip.open(output / name, 'wt', encoding='utf-8') as handle:
                json.dump(values, handle, ensure_ascii=False, allow_nan=False)
        result.update(scene_summary='scene-summary.json.gz', component_summary='component-summary.json.gz')
        write(output / 'summary.json', result)
        state.update(status='COMPLETED', elapsed_seconds=time.monotonic() - started,
                     scene_count=result['scene_count'], fragment_count=result['fragment_count'])
        write(output / 'run.json', state)
        return result
    except BaseException as error:
        state.update(status='FAILED', elapsed_seconds=time.monotonic() - started,
                     error=f'{type(error).__name__}: {error}')
        write(output / 'run.json', state)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source-run', 'manifest', 'output'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--cpu-threads', type=int, default=4)
    args = parser.parse_args(argv)
    try:
        result = run(args)
        print(f"COMPLETED095: {result['scene_count']} scenes / {result['fragment_count']} fragments", flush=True)
        return 0
    except (Exception, KeyboardInterrupt) as error:
        print(f'FAILED095 {type(error).__name__}: {error}', file=sys.stderr, flush=True)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
