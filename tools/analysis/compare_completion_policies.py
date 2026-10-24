"""Paired 093/094 evidence from original completed artifacts, without inference.

The existing V thresholds/correspondences are retained. No checkpoint selection,
training, raw-archive read, policy search, or T_test access occurs here.
"""
from __future__ import annotations

import argparse
from collections import Counter
import gzip
import json
import math
from pathlib import Path
import sys
import time

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from association.evaluation.artifacts import load_scene_prediction
from association.evaluation.physical import evaluate_aligned_fusion, group_membership_diagnostics
from association.evaluation.fusion_controls import attribute_errors
from association.evaluation.scene import INPUT_SUPPORT_REGIONS, INPUT_DISTANCE_REGIONS
from association.evaluation.system_attribution import ATTRIBUTES, assert_metrics, pool_metrics
from association.utils import runtime
from tools.analysis.compare_physical_predictions import _equal, _validate_pair
from tools.experiments import completion_value_fit as arm_checks

ROOT = Path(__file__).resolve().parents[2]
SIDES = ('learned', 'state_hold')


def _require(value, message):
    if not value:
        raise ValueError(message)


def _expected_configuration(phase, policy):
    """Fixed public profile; real tiny-chain tests replace only this helper."""
    suffix = 'hold' if policy == 'state_hold' else 'learned'
    return runtime.read_json(ROOT / f'configs/tmtaff-{phase}-{suffix}-level1.json')


def _same(left, right, label):
    _require(arm_checks._same_tree(left, right), 'paired evidence differs: ' + label)


def _numbers(left, right, label):
    """Saved/recomputed metric identity, with tolerance only for arithmetic."""
    if isinstance(left, dict) and isinstance(right, dict):
        _require(left.keys() == right.keys(), 'metric inventory differs: ' + label)
        for key in left:
            _numbers(left[key], right[key], label + '/' + key)
    elif isinstance(left, list) and isinstance(right, list):
        _require(len(left) == len(right), 'metric length differs: ' + label)
        for index, (a, b) in enumerate(zip(left, right)):
            _numbers(a, b, label + '/' + str(index))
    elif isinstance(left, float) or isinstance(right, float):
        _require(left is not None and right is not None
                 and math.isclose(left, right, rel_tol=1e-10, abs_tol=1e-9),
                 'metric differs: ' + label)
    else:
        _require(left == right, 'metric differs: ' + label)


def _bind(directory, phase, policy):
    import torch
    directory = Path(directory).resolve()
    expected = arm_checks.source_checks._canonical(_expected_configuration(phase, policy), phase)
    preparation = runtime.read_json(directory / 'preparation.json')
    _same(arm_checks.source_checks._canonical(preparation['configuration'], phase), expected,
          policy + '/fixed_configuration')
    verified = arm_checks.validate_completed_run(directory, phase, policy)
    previous = runtime.read_json(directory / 'completion-value-verification.json')
    _same(previous, verified, policy + '/original_arm_verification')
    final = torch.load(directory / 'checkpoint.pt', map_location='cpu', weights_only=True)
    initial = torch.load(directory / 'pairing-initial.pt', map_location='cpu', weights_only=True)
    return dict(directory=directory, verified=verified, final=final, initial=initial,
                run=runtime.read_json(directory / 'run.json'),
                summary=runtime.read_json(directory / 'summary.json'))


def _pairing(arms, phase):
    a, b = (arms[side] for side in SIDES)
    for key in ('states', 'rng', 'optimizer', 'normalizer', 'data_contract',
                'config', 'fusion_config', 'training_stages', 'cursor'):
        _same(a['initial'][key], b['initial'][key], 'initial/' + key)
    for arm, policy in ((a, 'learned'), (b, 'state_hold')):
        _require(arm['initial']['initialization'].get('target_completion_policy') == policy,
                 'source transfer omits actual target completion policy')
    _same({key: value for key, value in a['initial']['initialization'].items() if key != 'target_completion_policy'},
          {key: value for key, value in b['initial']['initialization'].items() if key != 'target_completion_policy'},
          'initial/source_initialization')
    for key in ('rng', 'normalizer', 'data_contract', 'coverage', 'config', 'fusion_config'):
        _same(a['final'][key], b['final'][key], 'final/' + key)
    for left, right in zip(a['final']['history'], b['final']['history']):
        for key in ('stage', 'update', 'actual_samples', 'supervision_counts'):
            _same(left[key], right[key], 'actual_update/' + str(left['update']) + '/' + key)
    _require(len(a['final']['history']) == len(b['final']['history']), 'successful update lengths differ')
    if phase == 'tam':
        ca, cb = (arm['final']['margin_calibration'] for arm in (a, b))
        _require(isinstance(ca, dict) and isinstance(cb, dict), 'T-only calibration evidence missing')
        _require(len(ca['samples']) == len(cb['samples']), 'calibration sample lengths differ')
        for left, right in zip(ca['samples'], cb['samples']):
            for key in ('batch', 'groups', 'labels', 'duplicate_pair_count'):
                _same(left[key], right[key], 'T-only margin calibration/' + key)
    return dict(all_initial_parameters_equal=True, fresh_optimizers_equal=True,
        initial_and_final_rng_equal=True, ordered_successful_samples_equal=True,
        task_supervision_counts_equal=True, semantic_train_and_validation_contract_equal=True,
        frozen_source_weights_checked=True, initial_predictions_required_equal=False,
        training_float_arrays_and_query_masks_replayed=False,
        training_audit_scope='actual ordered scene/pair endpoints, labels, counts, semantic inputs and RNG',
        validation_audit_scope=('actual saved input/truth/availability arrays at both endpoints'
            if phase == 'tfm' else 'all canonical candidates, node identities/sources/intervals and labels'),
        actual_samples={side: arms[side]['verified']['actual_samples'] for side in SIDES})


def _path_inside(path, root):
    path = Path(path).resolve()
    _require(path.is_relative_to(root), 'endpoint artifact is outside its original run: ' + str(path))
    return path


def _tfm_endpoints(arm):
    import torch
    rows = arm['summary']['fusion_evaluations']
    expected = [0, arm['final']['config']['updates']['TFM_FROZEN']]
    _same([row['update'] for row in rows], expected, 'initial/final endpoint updates')
    ids = [row['scene_id'] for row in arm['final']['data_contract'][1]]
    result = []
    for endpoint_index, row in enumerate(rows):
        snapshot = torch.load(arm['directory'] / 'snapshots' / f"update-{row['update']:06d}.pt",
                              map_location='cpu', weights_only=True)
        bound_snapshot = arm['initial'] if endpoint_index == 0 else arm['final']
        for key in ('states', 'config', 'model_config', 'fusion_config', 'normalizer', 'data_contract', 'initialization'):
            _same(snapshot[key], bound_snapshot[key], 'TFM endpoint snapshot/' + key)
        _require(snapshot['cursor']['completed_updates'] == row['update'], 'TFM endpoint snapshot update differs')
        del snapshot
        evaluation = runtime.read_json(_path_inside(row['evaluation'], arm['directory']))
        path = _path_inside(row['predictions'], arm['directory'])
        index = runtime.read_json(path)
        _require(row['status'] == 'COMPLETED' and evaluation['oracle_only'] is True
                 and evaluation['association'] is None and evaluation['selection'] is None
                 and 'predicted' not in evaluation['fusion']
                 and index.get('format') == 'scene_physical_predictions_v1'
                 and index.get('test_opened') is False, 'incomplete or non-oracle TFM endpoint')
        _same([entry['scene_id'] for entry in index['scenes']], ids, 'full V prediction population')
        _same([entry['scene_index'] for entry in index['scenes']], list(range(len(ids))), 'V scene order')
        paths = []
        for entry in index['scenes']:
            _require(Path(entry['metadata']).name == entry['metadata'], 'scene path leaves artifact directory')
            paths.append(path.parent / entry['metadata'])
        result.append(dict(update=row['update'], evaluation=evaluation, entries=index['scenes'], paths=paths))
    return result


def _regions(artifact):
    observed = artifact['aligned_observed']
    result = dict(all=np.ones(observed.shape, bool), observed=observed, missing=~observed)
    if 'aligned_input_support_kind' in artifact:
        kind, nearest = artifact['aligned_input_support_kind'], artifact['aligned_nearest_input_seconds']
        result.update({'support:' + name: kind == index for index, name in enumerate(INPUT_SUPPORT_REGIONS)})
        masks = (nearest <= 300, (nearest > 300) & (nearest <= 600),
                 (nearest > 600) & (nearest <= 1800),
                 (nearest > 1800) & np.isfinite(nearest), ~np.isfinite(nearest))
        result.update({'distance:' + name: mask for name, mask in zip(INPUT_DISTANCE_REGIONS, masks)})
    return result


def _metrics(artifact, *, valid=None, regions=None):
    prediction = artifact['aligned_prediction']
    truth, truth_valid = artifact['aligned_truth'], artifact['aligned_truth_valid']
    valid = artifact['aligned_prediction_valid'] if valid is None else valid
    regions = _regions(artifact) if regions is None else regions
    metrics = {name: evaluate_aligned_fusion(prediction, truth, truth_valid,
                    prediction_valid=valid, region_mask=mask) for name, mask in regions.items()}
    raw = attribute_errors(prediction, truth)
    errors = {}
    for attr, indices in ATTRIBUTES.items():
        mask = valid[..., indices].all(axis=-1) & truth_valid[..., indices].all(axis=-1)
        errors[attr] = raw[attr][mask]
        _require(np.isfinite(errors[attr]).all(), 'nonfinite error on common valid population')
    return metrics, errors


def _verify_metrics(metrics, saved, name):
    assert_metrics({k: metrics[k] for k in ('all', 'observed', 'missing')},
                   saved['full_target_metrics'], name)
    for prefix, key, regions in (('support:', 'input_support_metrics', INPUT_SUPPORT_REGIONS),
                                 ('distance:', 'input_distance_metrics', INPUT_DISTANCE_REGIONS)):
        if key in saved:
            assert_metrics({region: metrics[prefix + region] for region in regions}, saved[key], name + '/' + key)


def _pooled(rows, errors, indices):
    if not indices:
        return None
    result = pool_metrics([rows[index] for index in indices])
    for attr in ATTRIBUTES:
        values = np.concatenate([errors[index][attr] for index in indices])
        _require(len(values) == result['all'][attr]['intersection_count'], 'pooled error denominator differs')
        result['all'][attr].update(p95=float(np.quantile(values, .95)) if len(values) else None,
                                   p99=float(np.quantile(values, .99)) if len(values) else None)
    return result


def _delta(learned, hold):
    if learned is None or hold is None:
        return None
    return {attr: {key: (learned['all'][attr][key] - hold['all'][attr][key]
                if learned['all'][attr].get(key) is not None and hold['all'][attr].get(key) is not None else None)
            for key in ('mae', 'rmse', 'p95', 'p99', 'coverage', 'scene_macro_mae')}
            for attr in ATTRIBUTES}


def _compare_tfm(arms):
    endpoints = {side: _tfm_endpoints(arms[side]) for side in SIDES}
    population = arms['learned']['final']['data_contract'][1]
    components = Counter(str(row['component_id']) for row in population)
    largest = sorted(components, key=lambda value: (-components[value], value))[0]
    component_indices = {key: [i for i, row in enumerate(population) if str(row['component_id']) == key]
                         for key in sorted(components)}
    outside = [i for i, row in enumerate(population) if str(row['component_id']) != largest]
    reports = []
    # Read four artifacts for one scene at a time; retain only metrics and
    # valid error vectors, not all scenes' full input/prediction arrays.
    accumulators = [dict(full={side: [] for side in SIDES}, common={side: [] for side in SIDES},
        ordinary_common={side: [] for side in (*SIDES, 'simple_fusion_oracle')}, controls={},
        errors={kind: {side: [] for side in (SIDES if kind != 'ordinary_common' else (*SIDES, 'simple_fusion_oracle'))}
                for kind in ('full', 'common', 'ordinary_common')}, per_scene=[]) for _ in range(2)]
    for index, contract in enumerate(population):
        loaded = [[load_scene_prediction(endpoints[side][endpoint]['paths'][index]) for side in SIDES]
                  for endpoint in range(2)]
        reference = loaded[0][0]
        for endpoint in range(2):
            for side_index, side in enumerate(SIDES):
                current = loaded[endpoint][side_index]
                _validate_pair(reference, current, endpoints['learned'][0]['entries'][index],
                               endpoints[side][endpoint]['entries'][index], 'oracle')
                _require(str(current['scene']['component_id']) == str(contract['component_id']), 'component differs')
                for key in ('node_ids', 'identities', 'sources', 'intervals', 'grid'):
                    expected = np.asarray(contract[key]).reshape(-1, 2) if key == 'intervals' else contract[key]
                    _require(_equal(current['scene'][key], expected), 'actual V input differs from source: ' + key)
                _require(_equal(current['methods']['oracle']['target_ids'], contract['target_ids']), 'target population differs')
                _require(current['methods'].keys() == reference['methods'].keys(), 'physical method inventories differ')
                for method, artifact in current['methods'].items():
                    original = reference['methods'][method]
                    _require(artifact.keys() == original.keys(), 'method artifact inventory differs')
                    for key in original:
                        # Only learned TFM prediction values may differ. Its
                        # availability/groups and all ordinary controls must match.
                        if method == 'oracle' and key in ('aligned_prediction', 'group_states'):
                            continue
                        _require(_equal(artifact[key], original[key]), 'fixed validation evidence differs: ' + method + '/' + key)
        for endpoint in range(2):
            accumulator = accumulators[endpoint]
            artifacts = {side: loaded[endpoint][i]['methods']['oracle'] for i, side in enumerate(SIDES)}
            n = len(artifacts['learned']['target_ids'])
            _require(all(len(artifact['aligned_truth']) == n for artifact in artifacts.values()),
                     'oracle partition unexpectedly contains duplicate target rows')
            common = (artifacts['learned']['aligned_prediction_valid']
                      & artifacts['state_hold']['aligned_prediction_valid']
                      & artifacts['learned']['aligned_truth_valid'])
            ordinary = loaded[endpoint][0]['methods']['simple_fusion_oracle']
            _require(_equal(ordinary['aligned_truth'], artifacts['learned']['aligned_truth'])
                     and _equal(ordinary['aligned_truth_valid'], artifacts['learned']['aligned_truth_valid']),
                     'ordinary-control truth differs')
            triple = common & ordinary['aligned_prediction_valid']
            row = dict(scene_id=contract['scene_id'], component_id=str(contract['component_id']),
                       full={}, common={}, ordinary_common={})
            for side in SIDES:
                artifact = artifacts[side]
                for kind, valid in (('full', None), ('common', common), ('ordinary_common', triple)):
                    metrics, errors = _metrics(artifact, valid=valid)
                    accumulator[kind][side].append(metrics)
                    accumulator['errors'][kind][side].append(errors)
                    row[kind][side] = metrics
                saved = endpoints[side][endpoint]['evaluation']['per_scene'][index]['methods']['oracle']
                _verify_metrics(row['full'][side], saved, side + '/scene/' + contract['scene_id'])
            metrics, errors = _metrics(ordinary, valid=triple, regions=_regions(artifacts['learned']))
            accumulator['ordinary_common']['simple_fusion_oracle'].append(metrics)
            accumulator['errors']['ordinary_common']['simple_fusion_oracle'].append(errors)
            row['ordinary_common']['simple_fusion_oracle'] = metrics
            for method in reference['methods']:
                if method == 'oracle':
                    continue
                metrics, _ = _metrics(loaded[endpoint][0]['methods'][method])
                accumulator['controls'].setdefault(method, []).append(metrics)
                for side in SIDES:
                    saved = endpoints[side][endpoint]['evaluation']['per_scene'][index]['methods'][method]
                    _verify_metrics(metrics, saved, method + '/scene/' + contract['scene_id'])
            accumulator['per_scene'].append(row)
    all_indices = list(range(len(population)))
    for endpoint, accumulator in enumerate(accumulators):
        report = dict(update=endpoints['learned'][endpoint]['update'], full={}, common={}, ordinary_common={},
                      controls={name: pool_metrics(rows) for name, rows in accumulator['controls'].items()},
                      per_scene=accumulator['per_scene'], per_component=[], outside_largest_component={})
        for kind in ('full', 'common', 'ordinary_common'):
            for side, rows in accumulator[kind].items():
                report[kind][side] = _pooled(rows, accumulator['errors'][kind][side], all_indices)
        for side in SIDES:
            saved = endpoints[side][endpoint]['evaluation']['fusion']['oracle']
            _verify_metrics(report['full'][side], dict(full_target_metrics={k: saved[k] for k in ('all', 'observed', 'missing')},
                **{key: saved[key] for key in ('input_support_metrics', 'input_distance_metrics') if key in saved}), 'pooled/' + side)
        for key, indices in component_indices.items():
            row = dict(component_id=key, scene_count=len(indices), full={}, common={})
            for kind in ('full', 'common'):
                for side in SIDES:
                    row[kind][side] = _pooled(accumulator[kind][side], accumulator['errors'][kind][side], indices)
            row['learned_minus_hold'] = _delta(row['common']['learned'], row['common']['state_hold'])
            report['per_component'].append(row)
        for kind in ('full', 'common'):
            report['outside_largest_component'][kind] = {side: _pooled(accumulator[kind][side],
                accumulator['errors'][kind][side], outside) for side in SIDES}
        report['learned_minus_hold'] = _delta(report['common']['learned'], report['common']['state_hold'])
        for weighting, rows in (('scene', report['per_scene']), ('component', report['per_component'])):
            values = [(row['common']['learned']['all']['position']['mae'],
                       row['common']['state_hold']['all']['position']['mae']) for row in rows]
            valid = [(a, b) for a, b in values if a is not None and b is not None]
            report[weighting + '_wins'] = dict(all_units=len(rows), evaluable_units=len(valid),
                learned_better=sum(a < b for a, b in valid), state_hold_better=sum(a > b for a, b in valid),
                tied=sum(a == b for a, b in valid),
                equal_unit_mean_delta=float(np.mean([a-b for a, b in valid])) if valid else None)
        reports.append(report)
    return dict(scene_count=len(population), component_count=len(components),
        largest_component=dict(component_id=largest, scene_count=components[largest],
            selection='largest declared V component by scene count, lexical tie-break; no error selection'),
        actual_validation_arrays_equal=True, prediction_availability_equal=True,
        ordinary_controls_equal_at_all_endpoints=True, endpoints=reports,
        final_minus_initial={side: _delta(reports[1]['full'][side], reports[0]['full'][side]) for side in SIDES},
        quantile_policy='exact pooled valid point errors for all-region P95/P99; never an average of scene quantiles',
        ordinary_comparison='three-way intersection: learned/hold/simple_fusion_oracle prediction and reference',
        interpretation='oracle-only fixed-final pilot; no predicted-group system score, no independent-test or multiseed claim')


def _compare_tam(arms):
    from association.evaluation.association import evaluate_scenes, compare_predictions
    from association.evaluation.grouping import group_predicted_tracks
    records, thresholds, evaluations, grouping = {}, {}, {}, {}
    population = arms['learned']['final']['data_contract'][1]
    for side in SIDES:
        arm = arms[side]
        selected, last = arm['final']['selected'], arm['final']['history'][-1]
        _same({key: value for key, value in selected.items() if key != 'states'},
              arm['summary']['selection'], side + '/saved final selection')
        _same(selected['states'], arm['final']['states'], side + '/fixed-final selected states')
        _require(selected['update'] == last['update'] == arm['final']['cursor']['completed_updates']
                 and selected['stage'] == 'TAM_FROZEN', 'TAM selected endpoint is not the fixed final')
        for threshold in (last['selection']['threshold'], arm['summary']['evaluation']['threshold']):
            _same(selected['threshold'], threshold, side + '/original final threshold')
        for path in (last['score_records'], arm['summary']['endpoint_score_records'],
                     arm['summary']['evaluation']['score_records']):
            _same(selected['score_records'], path, side + '/original final score records')
        path = _path_inside(arm['summary']['endpoint_score_records'], arm['directory'])
        records[side] = runtime.load_records(path)
        thresholds[side] = selected['threshold']
        evaluations[side] = evaluate_scenes(records[side], thresholds[side])
        _numbers(evaluations[side], arm['summary']['evaluation']['association'], side + '/association')
        grouping[side] = []
        for record, contract in zip(records[side], population):
            for key in ('scene_id', 'component_id', 'node_ids', 'identities', 'sources', 'intervals'):
                expected = np.asarray(contract[key]).reshape(-1, 2) if key == 'intervals' else contract[key]
                _require(_equal(record[key], expected), 'saved candidate source differs: ' + key)
            groups = group_predicted_tracks(len(record['node_ids']), record['pairs'], record['scores'],
                threshold=thresholds[side], grouping_policy='anchor_direct', node_order='input')
            memberships = np.empty(len(record['node_ids']), dtype=int)
            for index, nodes in enumerate(groups):
                memberships[nodes] = index
            anchors = [str(record['identities'][nodes[0]]) for nodes in groups]
            grouping[side].append(dict(scene_id=record['scene_id'], component_id=record['component_id'],
                groups=groups, group_diagnostics=group_membership_diagnostics(memberships, record['identities']),
                declared_target_count=len(contract['target_ids']), anchor_matched_target_count=len(set(anchors)),
                unmatched_target_count=len(contract['target_ids'])-len(set(anchors)),
                duplicate_anchor_group_count=len(anchors)-len(set(anchors))))
    for a, b in zip(records['learned'], records['state_hold']):
        for key in ('scene_id', 'component_id', 'node_ids', 'identities', 'sources', 'intervals', 'pairs', 'labels'):
            _require(_equal(a[key], b[key]), 'actual paired score population differs: ' + key)
    counts = Counter(str(row['component_id']) for row in population)
    largest = sorted(counts, key=lambda key: (-counts[key], key))[0]
    components = []
    for component in sorted(counts):
        rows = {side: [row for row in records[side] if str(row['component_id']) == component] for side in SIDES}
        components.append(dict(component_id=component, scene_count=counts[component],
            association={side: evaluate_scenes(rows[side], thresholds[side]) for side in SIDES},
            paired_edges=compare_predictions(rows['state_hold'], rows['learned'], thresholds['state_hold'], thresholds['learned'])))
    outside = {side: [row for row in records[side] if str(row['component_id']) != largest] for side in SIDES}
    return dict(scene_count=len(population), component_count=len(counts), thresholds=thresholds,
        margins={side: arms[side]['final']['effective_margin'] for side in SIDES},
        association=evaluations, grouping=grouping,
        grouping_policy='actual anchor_direct/input-order groups; no transitive connected-component replacement',
        per_component=components, largest_component=dict(component_id=largest, scene_count=counts[largest]),
        outside_largest_component={side: evaluate_scenes(outside[side], thresholds[side]) for side in SIDES},
        paired_edges=compare_predictions(records['state_hold'], records['learned'], thresholds['state_hold'], thresholds['learned']),
        interpretation='completion with the same T-only margin-calibration rule and V-threshold rule; '
            'margins/thresholds may differ; V-selected pipeline comparison, not pure representation causality',
        threshold_reselected=False, physical_fusion_evaluated=False)


def compare(learned_run, hold_run, output, phase):
    _require(phase in ('tam', 'tfm'), 'phase must be tam or tfm')
    directories = dict(learned=Path(learned_run).resolve(), state_hold=Path(hold_run).resolve())
    output = Path(output).resolve()
    _require(directories['learned'] != directories['state_hold'], 'the paired arms must be distinct')
    _require(not any(output.is_relative_to(path) or path.is_relative_to(output) for path in directories.values()),
             'comparison output must be separate from both original runs')
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    state = dict(status='RUNNING', phase=phase, source_runs={key: str(value) for key, value in directories.items()},
                 training_updates=0, model_forward_count=0, test_opened=False)
    runtime.write_json(output / 'run.json', state)
    try:
        arms = {side: _bind(directories[side], phase, side) for side in SIDES}
        pairing = _pairing(arms, phase)
        result = dict(status='COMPLETED', phase=phase, sources=state['source_runs'], pairing=pairing,
            comparison=_compare_tfm(arms) if phase == 'tfm' else _compare_tam(arms),
            promotion_performed=False, training_updates=0, model_forward_count=0, test_opened=False,
            elapsed_seconds=time.monotonic()-started)
        with gzip.open(output / 'comparison.json.gz', 'wt', encoding='utf-8') as stream:
            json.dump(result, stream, ensure_ascii=False, allow_nan=False)
            stream.write('\n')
        runtime.write_json(output / 'summary.json', {key: value for key, value in result.items() if key != 'comparison'}
            | dict(full_comparison='comparison.json.gz', scene_count=result['comparison']['scene_count'],
                   component_count=result['comparison']['component_count']))
        state.update(status='COMPLETED', elapsed_seconds=result['elapsed_seconds'])
        runtime.write_json(output / 'run.json', state)
        return result
    except (Exception, KeyboardInterrupt) as error:
        state.update(status='FAILED', elapsed_seconds=time.monotonic()-started,
                     failure_type=type(error).__name__, failure=str(error))
        runtime.write_json(output / 'run.json', state)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('tam', 'tfm'), required=True)
    for name in ('learned-run', 'hold-run', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--cpu-threads', type=int, default=4, choices=range(1, 5))
    args = parser.parse_args(argv)
    import torch
    torch.set_num_threads(args.cpu_threads)
    result = compare(args.learned_run, args.hold_run, args.output, args.phase)
    print(json.dumps(dict(status=result['status'], phase=result['phase'], output=str(args.output.resolve()))))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
