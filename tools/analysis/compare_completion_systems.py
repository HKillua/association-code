"""096 learned/hold system comparison from complete saved physical predictions.

This invocation reads JSON/NPZ only: no checkpoint, model, raw archive, forward,
training, threshold selection or correspondence selection. Original population
coverage and duplicate outputs remain separate from common-point diagnostics.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import time

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from association.evaluation.artifacts import load_scene_prediction
from association.evaluation.association import evaluate_scenes, compare_predictions
from association.evaluation.grouping import group_predicted_tracks, group_tracks_by_identity
from association.evaluation.physical import group_membership_diagnostics
from association.evaluation.system_attribution import ATTRIBUTES
from association.utils import runtime
from tools.analysis.compare_physical_predictions import _completed, _equal, _validate_pair
from tools.analysis.compare_completion_policies import (
    _metrics, _verify_metrics, _pooled, _delta, _group_totals, _numbers,
)


SIDES = ('learned', 'state_hold')
METHODS = ('predicted', 'predicted_mean', 'predicted_anchor', 'oracle')
CONTROLS = ('predicted_mean', 'predicted_anchor')
MODE = 'completion_system_evaluation_only'


def _require(value, message):
    if not value:
        raise ValueError(message)


def _expected_source_updates():
    """Public 096 endpoints; real tiny-chain tests replace only this helper."""
    return dict(tam=4000, tfm=1000)


def _bind(directory, policy):
    directory = Path(directory).resolve()
    run = runtime.read_json(directory / 'run.json')
    _require(run.get('status') == 'COMPLETED' and run.get('mode') == MODE
             and run.get('completion_policy') == policy
             and run.get('optimizer_updates') == 0
             and run.get('threshold_reselected') is False
             and run.get('test_opened') is False,
             'only a completed, fixed-policy zero-training 096 system is accepted')
    summary = runtime.read_json(directory / 'summary.json')
    binding = runtime.read_json(directory / 'source-binding.json')
    _require(binding == run.get('source_binding') == summary.get('source_binding'),
             'system source binding differs across original records')
    _require(binding.get('completion_policy') == policy
             and binding.get('normalizer_equal') is True
             and binding.get('frozen_completion_equal') is True
             and binding.get('full_data_contract_equal') is True,
             'system omits verified same-policy source assembly evidence')
    updates = _expected_source_updates()
    _require(binding.get('source_tam_update') == updates['tam']
             and binding.get('source_tfm_update') == updates['tfm'],
             'system sources differ from the predeclared 094/093 endpoints')
    sources = binding.get('source_runs')
    _require(isinstance(sources, dict) and set(sources) >= {'tcm', 'tam', 'tfm'}
             and all(isinstance(sources[key], str) and sources[key] for key in ('tcm', 'tam', 'tfm')),
             'system must identify the three actual source runs')
    _require(run.get('source_tam_run') == sources['tam'] and run.get('source_tfm_run') == sources['tfm']
             and isinstance(binding.get('normalizer'), dict) and binding['normalizer'],
             'system run/source metadata or saved normalizer is incomplete')
    evaluation = summary['evaluation']
    _require(evaluation.get('oracle_only') is False
             and evaluation.get('selection') is None
             and evaluation.get('test_opened') is False
             and evaluation.get('threshold') == binding.get('threshold')
             and 'threshold' in binding,
             'system evaluation changed the original TAM threshold or is oracle-only')
    entries, paths = _completed(directory / 'predictions')
    ids = [entry['scene_id'] for entry in entries]
    _require(ids == run.get('validation_scene_ids') == binding.get('validation_scene_ids'),
             'run and original-source full prediction population differ')
    _require(isinstance(binding.get('validation_component_ids'), list)
             and len(binding['validation_component_ids']) == len(ids),
             'system binding omits original validation component population')
    records = evaluation.get('records')
    _require(isinstance(records, list) and [str(row['scene_id']) for row in records] == ids,
             'full original association records must cover every scene in order')
    records = [dict(row, pairs=np.asarray(row['pairs'], dtype=np.int64).reshape(-1, 2),
                    intervals=np.asarray(row['intervals'], dtype=float).reshape(-1, 2))
               for row in records]
    association = evaluate_scenes(records, evaluation['threshold'])
    _numbers(association, evaluation['association'], policy + '/saved association')
    return dict(directory=directory, run=run, binding=binding, evaluation=evaluation,
                entries=entries, paths=paths, records=records, association=association)


def _groups(artifact, node_count):
    offsets = artifact['group_offsets']
    nodes = artifact['group_node_indices']
    _require(offsets.ndim == nodes.ndim == 1 and offsets.dtype.kind in 'iu'
             and nodes.dtype.kind in 'iu' and len(offsets) > 0 and offsets[0] == 0
             and offsets[-1] == len(nodes) == node_count
             and (np.diff(offsets) > 0).all(), 'invalid saved group partition')
    groups = [nodes[a:b].tolist() for a, b in zip(offsets[:-1], offsets[1:])]
    _require(sorted(node for group in groups for node in group) == list(range(node_count)),
             'saved groups do not partition all nodes exactly once')
    memberships = np.empty(node_count, dtype=np.int64)
    for index, group in enumerate(groups):
        memberships[group] = index
    _require(_equal(memberships, artifact['node_group_indices']), 'saved membership arrays disagree')
    return groups


def _alignment(artifact, scene, groups):
    """Check the fixed first-anchor mapping, not a new best-error matching."""
    targets = artifact['target_ids'].tolist()
    target_index = {identity: index for index, identity in enumerate(targets)}
    _require(len(target_index) == len(targets), 'duplicate declared targets')
    anchors = [target_index[str(scene['identities'][group[0]])] for group in groups]
    first, duplicates = {}, []
    for index, anchor in enumerate(anchors):
        if anchor in first:
            duplicates.append(index)
        else:
            first[anchor] = index
    expected = np.full(len(targets) + len(duplicates), -1, dtype=np.int64)
    for anchor, group in first.items():
        expected[anchor] = group
    expected[len(targets):] = duplicates
    _require(_equal(artifact['group_anchor_target_indices'], anchors)
             and _equal(artifact['aligned_group_indices'], expected), 'first-anchor mapping differs')
    _require(not artifact['aligned_truth_valid'][len(targets):].any(),
             'duplicate predictions must never receive assigned truth')
    for row, group in enumerate(expected):
        if group < 0:
            _require(not artifact['aligned_prediction_valid'][row].any(),
                     'unmatched target has a prediction')
        else:
            _require(_equal(artifact['aligned_prediction'][row], artifact['group_states'][group])
                     and _equal(artifact['aligned_prediction_valid'][row], artifact['group_valid'][group]),
                     'aligned prediction does not match its original group output')
    return dict(scene_id=str(scene['scene_id']), component_id=str(scene['component_id']),
        groups=groups, group_diagnostics=group_membership_diagnostics(artifact['node_group_indices'], scene['identities']),
        declared_target_count=len(targets), anchor_matched_target_count=len(first),
        unmatched_target_count=len(targets)-len(first), duplicate_anchor_group_count=len(duplicates))


def _targets(artifact):
    n = len(artifact['target_ids'])
    return {key: value[:n] if key.startswith('aligned_') else value for key, value in artifact.items()}


def _common_regions(artifact):
    # Group-derived support differs between providers. These strata use only
    # the identical target observation mask, not either system's chosen groups.
    observed = artifact['aligned_observed']
    return dict(all=np.ones(observed.shape, bool), observed=observed, missing=~observed)


def _method_pair(left, right, *, same_groups):
    for key in ('target_ids', 'split', 'correspondence_policy'):
        _require(_equal(left[key], right[key]), 'within-system target protocol differs: ' + key)
    n = len(left['target_ids'])
    for key in ('aligned_truth', 'aligned_truth_valid', 'aligned_observed'):
        _require(_equal(left[key][:n], right[key][:n]), 'within-system target evidence differs: ' + key)
    if same_groups:
        for key in ('grouping_kind', 'group_offsets', 'group_node_indices', 'node_group_indices',
                    'group_anchor_target_indices', 'aligned_group_indices', 'aligned_target_ids'):
            _require(_equal(left[key], right[key]), 'ordinary control changed predicted groups: ' + key)
    a, b = _targets(left), _targets(right)
    common = a['aligned_prediction_valid'] & b['aligned_prediction_valid'] & a['aligned_truth_valid']
    return a, b, common


def _scene(arms, index):
    loaded = {side: load_scene_prediction(arms[side]['paths'][index]) for side in SIDES}
    a, b, n = _validate_pair(loaded['learned'], loaded['state_hold'],
                           arms['learned']['entries'][index], arms['state_hold']['entries'][index], 'predicted')
    sid = arms['learned']['entries'][index]['scene_id']
    result = dict(scene_id=sid, component_id=str(loaded['learned']['scene']['component_id']),
                  full={}, common={}, controls={}, oracle_common={}, grouping={})
    chunks = {}
    for side in SIDES:
        current, arm = loaded[side], arms[side]
        scene, methods, record = current['scene'], current['methods'], arm['records'][index]
        _require(set(METHODS).issubset(methods), 'required system or fixed-group control artifact missing')
        _require(scene['input_adapter'] == 'native_time_v1', '096 requires native-time inputs')
        _require(str(scene['component_id']) == str(arm['binding']['validation_component_ids'][index]),
                 'artifact component differs from the original source population')
        for key in ('scene_id', 'component_id', 'node_ids', 'identities', 'sources', 'intervals', 'pairs', 'labels'):
            _require(_equal(record[key], scene[key]), 'actual score/input population differs: ' + key)
        expected_pairs = np.column_stack(np.triu_indices(len(scene['node_ids']), 1))
        _require(_equal(scene['pairs'], expected_pairs), 'score population omits canonical candidate pairs')
        expected = group_predicted_tracks(len(scene['node_ids']), record['pairs'], record['scores'],
            threshold=arm['evaluation']['threshold'], grouping_policy='anchor_direct', node_order='input')
        result['full'][side], result['controls'][side], result['oracle_common'][side] = {}, {}, {}
        for method in METHODS:
            artifact = methods[method]
            _require(artifact['grouping_kind'] == ('oracle' if method == 'oracle' else 'predicted'),
                     'system and oracle grouping kinds cannot be interchanged')
            groups = _groups(artifact, len(scene['node_ids']))
            _require(groups == (group_tracks_by_identity(scene['identities'], node_order='input')
                                if method == 'oracle' else expected),
                     'saved groups differ from original frozen-threshold candidates')
            grouping = _alignment(artifact, scene, groups)
            if method == 'predicted':
                result['grouping'][side] = grouping
            metrics, errors = _metrics(artifact)
            _verify_metrics(metrics, arm['evaluation']['per_scene'][index]['methods'][method],
                            side + '/' + sid + '/' + method)
            result['full'][side][method] = metrics
            chunks[('full', side, method)] = metrics, errors
        for control in CONTROLS:
            system, ordinary, mask = _method_pair(methods['predicted'], methods[control], same_groups=True)
            result['controls'][side][control] = {}
            for method, artifact in (('predicted', system), (control, ordinary)):
                metrics, errors = _metrics(artifact, valid=mask)
                result['controls'][side][control][method] = metrics
                chunks[('control:' + control, side, method)] = metrics, errors
        system, oracle, mask = _method_pair(methods['predicted'], methods['oracle'], same_groups=False)
        for method, artifact in (('predicted', system), ('oracle', oracle)):
            metrics, errors = _metrics(artifact, valid=mask, regions=_common_regions(artifact))
            result['oracle_common'][side][method] = metrics
            chunks[('oracle_common', side, method)] = metrics, errors
    a, b = _targets(a), _targets(b)
    common = a['aligned_prediction_valid'] & b['aligned_prediction_valid'] & a['aligned_truth_valid']
    for side, artifact in (('learned', a), ('state_hold', b)):
        metrics, errors = _metrics(artifact, valid=common, regions=_common_regions(artifact))
        result['common'][side] = metrics
        chunks[('common', side, 'predicted')] = metrics, errors
    return result, chunks


def _aggregate(acc, rows, arms, indices):
    if not indices:
        return None
    def pooled(scope, side, method):
        item = acc[(scope, side, method)]
        return _pooled(item['metrics'], item['errors'], indices)
    result = dict(scene_count=len(indices),
        full={side: {method: pooled('full', side, method) for method in METHODS} for side in SIDES},
        common={side: pooled('common', side, 'predicted') for side in SIDES}, controls={}, oracle_common={},
        grouping={side: _group_totals([rows[i]['grouping'][side] for i in indices]) for side in SIDES},
        association={side: evaluate_scenes([arms[side]['records'][i] for i in indices],
                     arms[side]['evaluation']['threshold']) for side in SIDES})
    for side in SIDES:
        result['controls'][side] = {}
        for control in CONTROLS:
            pair = {method: pooled('control:' + control, side, method) for method in ('predicted', control)}
            pair['system_minus_control'] = _delta(pair['predicted'], pair[control])
            result['controls'][side][control] = pair
        pair = {method: pooled('oracle_common', side, method) for method in ('predicted', 'oracle')}
        pair['predicted_minus_oracle'] = _delta(pair['predicted'], pair['oracle'])
        result['oracle_common'][side] = pair
    result['learned_minus_hold'] = _delta(result['common']['learned'], result['common']['state_hold'])
    result['paired_edges'] = compare_predictions(
        [arms['state_hold']['records'][i] for i in indices], [arms['learned']['records'][i] for i in indices],
        arms['state_hold']['evaluation']['threshold'], arms['learned']['evaluation']['threshold'])
    return result


def _wins(rows):
    values = [(row['common']['learned']['all']['position']['mae'],
               row['common']['state_hold']['all']['position']['mae']) for row in rows]
    valid = [(a, b) for a, b in values if a is not None and b is not None]
    return dict(all_units=len(rows), evaluable_units=len(valid),
        learned_better=sum(a < b for a, b in valid), state_hold_better=sum(a > b for a, b in valid),
        tied=sum(a == b for a, b in valid),
        equal_unit_mean_delta=float(np.mean([a-b for a, b in valid])) if valid else None)


def _report(result):
    lines = ['# 096 complete-system comparison', '',
        'Fixed-final native-time systems; repeated V development, one seed, one-pass 088 source. No promotion.', '',
        '| System | Own MAE m | Own RMSE m | Own P95 m | Own P99 m | Position coverage | Common MAE m |',
        '|---|---:|---:|---:|---:|---:|---:|']
    def display(value):
        return 'null' if value is None else f'{value:.6f}'
    for side in SIDES:
        own = result['all']['full'][side]['predicted']['all']['position']
        common = result['all']['common'][side]['all']['position']
        values = [own[key] for key in ('mae', 'rmse', 'p95', 'p99', 'coverage')] + [common['mae']]
        lines.append('| ' + side + ' | ' + ' | '.join(display(value) for value in values) + ' |')
    lines += ['', 'Own-population errors have different availability. The main paired error comparison uses common points;',
        'both original full-population coverage and unmatched/duplicate outputs are retained.', '',
        'Ordinary controls use the same predicted groups, observed inputs and explicit masks. Their paired errors',
        'use each within-system intersection. Oracle comparisons are diagnostics, not deployable system rankings',
        'or a causal error decomposition. No 072C cross-schema ranking is performed by this comparator.', '',
        'Exact all-region P95/P99 are recomputed from pooled valid error arrays. Scene quantiles are not averaged.', '',
        f"Scene wins: {json.dumps(result['scene_wins'])}",
        f"Component wins: {json.dumps(result['component_wins'])}", '']
    return '\n'.join(lines)


def compare(learned_run, hold_run, output):
    directories = dict(learned=Path(learned_run).resolve(), state_hold=Path(hold_run).resolve())
    output = Path(output).resolve()
    _require(directories['learned'] != directories['state_hold'], 'system arms must be distinct')
    _require(not any(output.is_relative_to(path) or path.is_relative_to(output) for path in directories.values()),
             'comparison output must be separate from both original runs')
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    state = dict(status='RUNNING', mode='completion_system_cache_comparison',
        source_runs={side: str(path) for side, path in directories.items()}, optimizer_updates=0,
        model_forwards=0, checkpoint_loads=0, threshold_reselected=False, test_opened=False)
    runtime.write_json(output / 'run.json', state)
    try:
        arms = {side: _bind(directories[side], side) for side in SIDES}
        _require(arms['learned']['binding']['source_runs']['tcm'] == arms['state_hold']['binding']['source_runs']['tcm'],
                 'paired systems were not assembled from the same 088 source')
        _require(arms['learned']['run']['validation_scene_ids'] == arms['state_hold']['run']['validation_scene_ids'],
                 'system validation populations differ')
        _require(arms['learned']['binding']['normalizer'] == arms['state_hold']['binding']['normalizer'],
                 'paired system normalizers differ')
        rows, acc = [], {}
        for index in range(len(arms['learned']['entries'])):
            row, chunks = _scene(arms, index)
            rows.append(row)
            for key, (metrics, errors) in chunks.items():
                item = acc.setdefault(key, dict(metrics=[], errors=[]))
                item['metrics'].append(metrics)
                item['errors'].append(errors)
        counts = Counter(row['component_id'] for row in rows)
        largest = sorted(counts, key=lambda key: (-counts[key], key))[0]
        all_report = _aggregate(acc, rows, arms, list(range(len(rows))))
        for side in SIDES:
            for method in METHODS:
                saved = arms[side]['evaluation']['fusion'][method]
                _verify_metrics(all_report['full'][side][method],
                    dict(full_target_metrics={key: saved[key] for key in ('all', 'observed', 'missing')},
                         **{key: saved[key] for key in ('input_support_metrics', 'input_distance_metrics') if key in saved}),
                    side + '/pooled/' + method)
        components = [dict(component_id=component,
            **_aggregate(acc, rows, arms, [i for i, row in enumerate(rows) if row['component_id'] == component]))
            for component in sorted(counts)]
        result = dict(status='COMPLETED', mode=state['mode'], sources=state['source_runs'],
            source_bindings={side: arms[side]['binding'] for side in SIDES},
            scene_count=len(rows), component_count=len(counts),
            largest_component=dict(component_id=largest, scene_count=counts[largest],
                selection='largest declared component by scene count, lexical tie-break; not chosen by error'),
            thresholds={side: arms[side]['evaluation']['threshold'] for side in SIDES},
            all=all_report, per_component=components,
            outside_largest_component=_aggregate(acc, rows, arms,
                [i for i, row in enumerate(rows) if row['component_id'] != largest]),
            scene_wins=_wins(rows), component_wins=_wins(components),
            validation_arrays_equal=True, original_threshold_grouping_verified=True,
            original_metrics_reaggregated=True, ordinary_controls_same_predicted_groups=True,
            paired_training_audit='093/094 original CPU comparisons remain separate; this invocation reads no weights or training logs',
            primary_estimand='predicted-system position MAE on the common valid reference/prediction intersection; retain both original coverage ledgers',
            support_strata='full metrics use each system own predicted-group support; cross-provider common strata use identical target observed/missing only',
            quantile_policy='exact pooled valid point errors, all-region P95/P99; never average scene quantiles',
            oracle_interpretation='same trained TFM with true groups is diagnostic; common-point contrast is not a causal decomposition',
            external_baseline_comparison='not performed: 072C requires an explicit cross-schema input/population bridge',
            limitations='single seed; repeated V; shared identities; one-pass 088 source; no author-protocol reproduction or significance claim',
            promotion_performed=False, optimizer_updates=0, model_forwards=0, checkpoint_loads=0,
            threshold_reselected=False, test_opened=False, elapsed_seconds=time.monotonic()-started)
        runtime.write_json_gzip(output / 'scene-summary.json.gz', rows)
        runtime.write_json_gzip(output / 'summary.json.gz', result)
        (output / 'report.md').write_text(_report(result), encoding='utf-8')
        state.update(status='COMPLETED', completed_scenes=len(rows), elapsed_seconds=time.monotonic()-started)
        runtime.write_json(output / 'run.json', state)
        return result
    except (Exception, KeyboardInterrupt) as error:
        state.update(status='FAILED', failure_type=type(error).__name__, failure=str(error),
                     elapsed_seconds=time.monotonic()-started)
        runtime.write_json(output / 'run.json', state)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('learned-run', 'hold-run', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--cpu-threads', type=int, default=4, choices=range(1, 5))
    args = parser.parse_args(argv)
    import torch
    torch.set_num_threads(args.cpu_threads)
    result = compare(args.learned_run, args.hold_run, args.output)
    print(json.dumps(dict(status=result['status'], scene_count=result['scene_count'], output=str(args.output.resolve()))))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
