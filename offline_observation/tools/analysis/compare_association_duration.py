"""060: paired 1000/3000-update system comparison using frozen V caches only."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch

from association.evaluation.fusion_controls import ordinary_fusion
from association.evaluation.oracle_diagnostics import POPULATION_KEYS, _comparison
from association.evaluation.scene import evaluate_full_scene
from association.evaluation.system_attribution import (
    ATTRIBUTES, _metrics, _regions, _same, assert_metrics, pool_metrics,
)
from association.utils.runtime import write_json
from association.utils.runtime import read_json_or_gzip as read, write_json_gzip as write_gzip

ARMS = ('anchor', 'mean_supported', 'attribute_hybrid')
METHODS = sorted(group + '_' + arm for group in ('ols', 'learned_assoc') for arm in ARMS)
INPUT_KEYS = ('node_ids', 'identities', 'sources', 'target_identity_ids', 'query_seconds',
              'target', 'target_valid', 'native_states', 'support', 'nearest_seconds', 'exact_observation')


def _load_metadata(source, manifest, endpoint_step):
    """Discard large per-group trees before the next source summary is loaded."""
    run, summary = read(source/'run.json'), read(source/'summary.json')
    ids = manifest['splits']['V_select']
    for record in (run, summary):
        if (record.get('status') != 'COMPLETED' or record.get('dataset_level') != 'Level1'
                or record.get('split') != 'V_select' or record.get('test_opened') is not False
                or record.get('optimizer_updates') != 0 or record.get('association_endpoint') != 'final'
                or record.get('ordinary_controls') != list(ARMS) or record.get('method_names') != METHODS
                or record.get('fusion_checkpoint_step') is not None):
            raise ValueError('completed zero-training six-arm Level1 final system required')
    wp = summary.get('ols_working_point', {})
    if (run.get('motion_observation_policy') != 'canonical_all'
            or run.get('threshold_reselected') is not False or run.get('normalizer_refitted') is not False
            or run.get('completed_scenes') != len(ids) or run.get('total_scenes') != len(ids)
            or summary.get('scene_ids') != ids or summary.get('scene_count') != len(ids)
            or summary.get('association_endpoint_step') != endpoint_step
            or wp.get('motion_observation_policy') != 'canonical_all'
            or wp.get('validation_scene_ids') != ids):
        raise ValueError('complete V population, canonical policy and fixed endpoint step required')
    inputs = run.get('inputs')
    if (not isinstance(inputs, dict) or inputs != summary.get('inputs')
            or not isinstance(inputs.get('association_run'), str) or not inputs['association_run']):
        raise ValueError('system run and summary must bind the same association training source')
    training_run = read(inputs['association_run'])
    if (training_run.get('status') != 'COMPLETED'
            or training_run.get('optimizer_updates') != endpoint_step
            or training_run.get('config', {}).get('training', {}).get('updates') != endpoint_step):
        raise ValueError('association training source must be completed at the configured final endpoint')
    expected = {}
    for row in summary['per_scene']:
        sid = row['scene_id']
        if sid in expected or set(row['unique_target']) != set(METHODS):
            raise ValueError('source per-scene methods/population differ')
        expected[sid] = {name: dict(
            population={key: item[key] for key in POPULATION_KEYS},
            full_target_metrics=item['full_target_metrics'])
            for name, item in row['unique_target'].items()}
    if list(expected) != ids:
        raise ValueError('source per-scene population/order differs from manifest')
    names = {f'states-{sid}.npz' for sid in ids}
    for directory in ('predictions', 'system-predictions'):
        if {p.name for p in (source/directory).glob('*.npz')} != names:
            raise ValueError(f'incomplete or extra cached scenes in {directory}')
    return dict(scene_ids=ids, fragment_count=summary['fragment_count'],
        endpoint_step=endpoint_step, ols_working_point=wp,
        association_run=inputs['association_run'],
        association_threshold=summary['association_threshold'], association=summary['association'],
        expected=expected, pooled={name: {region: summary['unique_target'][name][region]
            for region in ('all', 'observed', 'missing')} for name in METHODS})


def _prefix_status(metadata):
    """Carry the separate training-prefix check, never infer it from system success."""
    paths = {side: values.get('association_run') for side, values in metadata.items()}
    if not all(paths.values()):
        return dict(comparison_status='NOT_AVAILABLE', causal_attribution_valid=False,
                    reason='system provenance does not locate both association runs')
    path = Path(paths['candidate']).resolve().parent/'prefix-comparison.json'
    if not path.exists():
        return dict(comparison_status='NOT_AVAILABLE', causal_attribution_valid=False,
                    artifact=str(path), reason='training prefix comparison artifact is absent')
    record = read(path)
    status = record.get('comparison_status')
    valid = record.get('causal_attribution_valid')
    if (status not in ('EXACT_MATCH', 'TOLERANT_MATCH', 'MISMATCH') or type(valid) is not bool
            or valid != (status != 'MISMATCH') or record.get('protocol_valid') is not True
            or record.get('prefix_update') != 1000 or record.get('source_endpoint') != 'final'
            or Path(record.get('source_directory', '')).resolve() != Path(paths['reference']).resolve().parent):
        raise ValueError('prefix comparison does not bind this 1000-update reference run')
    return dict(artifact=str(path), **{key: record[key] for key in (
        'comparison_status', 'causal_attribution_valid', 'prefix_update', 'source_directory', 'protocol_valid')},
        interpretation=('fixed-protocol prefix agreement supports this single-seed duration comparison; not independent replication'
            if valid else 'descriptive final-system comparison only; pure training-duration attribution is invalid'),
        bitwise_state_and_scores_equal=record.get('bitwise_state_and_scores_equal'),
        own_threshold_changed_decisions=record.get('own_threshold_changed_decisions'),
        scores={key: value for key, value in record.get('scores', {}).items() if key != 'per_scene'},
        thresholds=record.get('thresholds'))


def _load_arrays(source, sid):
    result = []
    for directory in ('predictions', 'system-predictions'):
        with np.load(source/directory/f'states-{sid}.npz', allow_pickle=False) as saved:
            result.append({key: saved[key] for key in saved.files})
    return result


def _replay(fragment, saved, expected):
    """Re-evaluate saved groups, never regenerate scores or choose representatives."""
    nodes, targets = fragment['identities'], fragment['target_identity_ids']
    n, t = len(nodes), len(targets)
    if (fragment['query_seconds'].shape != (64,) or fragment['native_states'].shape != (n, 64, 4)
            or len(set(fragment['node_ids'])) != n):
        raise ValueError('complete original node order and Q64 native states required')
    base = {key[len('ols_anchor_'):]: value for key, value in saved.items() if key.startswith('ols_anchor_')}
    truth, valid = base['aligned_truth'][:t], base['aligned_truth_valid'][:t]
    _same(base['target_ids'], targets, 'complete target registry')
    registry = {str(identity): row for row, identity in enumerate(targets)}
    if len(registry) != t:
        raise ValueError('target registry must be unique')
    target_support, target_exact = np.zeros((t, 64), bool), np.zeros((t, 64), bool)
    target_nearest = np.full((t, 64), np.inf)
    for node, identity in enumerate(nodes):
        row = registry[str(identity)]
        _same(fragment['target'][node], truth[row], 'fragment reference')
        _same(fragment['target_valid'][node], valid[row], 'fragment reference mask')
        target_support[row] |= fragment['support'][node]
        target_exact[row] |= fragment['exact_observation'][node]
        target_nearest[row] = np.minimum(target_nearest[row], fragment['nearest_seconds'][node])
    regions = _regions(target_support, target_nearest, target_exact)
    artifacts, representatives, methods = {}, {}, {}
    for grouping in ('ols', 'learned_assoc'):
        offsets, flat = (fragment[grouping + '_' + key] for key in ('group_offsets', 'group_node_indices'))
        if (offsets.ndim != 1 or len(offsets) < 2 or offsets[0] != 0 or offsets[-1] != n
                or np.any(np.diff(offsets) <= 0) or not np.array_equal(np.sort(flat), np.arange(n))):
            raise ValueError('saved groups must partition the complete input population')
        groups = [flat[a:b].tolist() for a, b in zip(offsets[:-1], offsets[1:])]
        anchors = np.asarray([members[0] for members in groups], np.int64)
        controls = ordinary_fusion(fragment['native_states'], fragment['nearest_seconds'], fragment['support'], groups)
        _same(controls['membership'], fragment[grouping + '_membership'], 'group membership')
        predictions = controls['predictions']
        hybrid = predictions['mean_supported'].copy()
        hybrid[..., :2] = predictions['anchor'][..., :2]
        predictions = dict(predictions, attribute_hybrid=hybrid)
        for arm in ARMS:
            name = grouping + '_' + arm
            _same(predictions[arm], fragment[name], name + ' cached ordinary prediction')
            outputs = [dict(node_indices=members, states=fragment[name][members[0]]) for members in groups]
            report = evaluate_full_scene(outputs, node_identities=nodes, target_identities=targets,
                target_states=truth, target_valid=valid, node_observed=fragment['support'], split='V_select',
                grouping_kind='predicted', correspondence_policy='anchor_identity_first', return_artifact=True,
                node_input_support=dict(support=fragment['support'], nearest_seconds=fragment['nearest_seconds']))
            artifact = report.pop('artifact')
            cached = {key[len(name)+1:]: value for key, value in saved.items() if key.startswith(name + '_')}
            if set(artifact) != set(cached):
                raise ValueError('saved system artifact fields changed')
            for key in artifact:
                _same(artifact[key], cached[key], name + '/' + key)
            population = {key: report[key] for key in POPULATION_KEYS}
            if population != expected[name]['population']:
                raise ValueError('cached population differs from source report')
            assert_metrics(report['full_target_metrics'], expected[name]['full_target_metrics'], name)
            methods[name] = dict(population=population, full_target_metrics=report['full_target_metrics'],
                fixed_target_strata=_metrics(artifact['aligned_prediction'][:t], truth, valid,
                                            artifact['aligned_prediction_valid'][:t], regions))
            selected = cached['aligned_group_indices'][:t]
            rep = np.full(t, -1, np.int64)
            rep[selected >= 0] = anchors[selected[selected >= 0]]
            representatives[name], artifacts[name] = rep, cached
    return methods, artifacts, representatives, truth, valid, regions


def compare_scene(reference, candidate, reference_saved, candidate_saved, reference_expected, candidate_expected):
    for key in INPUT_KEYS:
        _same(reference[key], candidate[key], 'cross-run input/' + key)
    for key in reference:
        if key.startswith('ols_') or key.startswith('region_'):
            _same(reference[key], candidate[key], 'unchanged OLS/' + key)
    for key in reference_saved:
        if key.startswith('ols_'):
            _same(reference_saved[key], candidate_saved[key], 'unchanged OLS system/' + key)
    first = _replay(reference, reference_saved, reference_expected)
    second = _replay(candidate, candidate_saved, candidate_expected)
    for index, label in ((3, 'complete truth'), (4, 'complete truth mask')):
        _same(first[index], second[index], label)
    for name in METHODS:
        if name.startswith('ols_'):
            assert_metrics(first[0][name]['full_target_metrics'], second[0][name]['full_target_metrics'], name)
    pairs = {}
    for arm in ARMS:
        name = 'learned_assoc_' + arm
        artifacts = dict(reference=first[1][name], candidate=second[1][name])
        representatives = dict(reference=first[2][name], candidate=second[2][name])
        pairs[arm] = _comparison('reference', 'candidate', artifacts, representatives,
                                first[3], first[4], first[5], {})
    # Same representative means exactly the same native Anchor output. Any
    # improvement from changed representatives is a selection/grouping effect.
    unchanged = (first[2]['learned_assoc_anchor'] >= 0) & (
        first[2]['learned_assoc_anchor'] == second[2]['learned_assoc_anchor'])
    t = len(unchanged)
    _same(first[1]['learned_assoc_anchor']['aligned_prediction'][:t][unchanged],
          second[1]['learned_assoc_anchor']['aligned_prediction'][:t][unchanged], 'same representative Anchor')
    return dict(fragment_count=len(reference['node_ids']), source_replay_verified=True,
        unchanged_ols_controls_verified=True, methods=dict(reference=first[0], candidate=second[0]), pairwise=pairs)


def aggregate(rows):
    result = dict(scene_count=len(rows), methods={}, pairwise={})
    for source in ('reference', 'candidate'):
        result['methods'][source] = {}
        for name in METHODS:
            items = [row['methods'][source][name] for row in rows]
            result['methods'][source][name] = dict(
                population={key: sum(item['population'][key] for item in items) for key in POPULATION_KEYS},
                **{key: pool_metrics([item[key] for item in items]) for key in ('full_target_metrics', 'fixed_target_strata')})
    for arm in ARMS:
        items = [row['pairwise'][arm] for row in rows]
        first = items[0]
        pooled = dict(methods={side: pool_metrics([item['methods'][side] for item in items])
                              for side in ('reference', 'candidate')},
            coverage_transitions={attr: {key: sum(item['coverage_transitions'][attr][key] for item in items)
                for key in first['coverage_transitions'][attr]} for attr in ATTRIBUTES},
            error_change_second_minus_first={},
            **{key: sum(item[key] for item in items) for key in
               ('same_representative_target_count', 'changed_representative_target_count')})
        for key in ('newly_covered', 'lost_coverage'):
            pooled[key] = pool_metrics([item[key] for item in items])
        for region, attrs in first['error_change_second_minus_first'].items():
            pooled['error_change_second_minus_first'][region] = {}
            for attr, values in attrs.items():
                changes = [item['error_change_second_minus_first'][region][attr] for item in items]
                totals = {key: sum(change[key] for change in changes) for key in values if key != 'mean_error_change'}
                count = totals['common_point_count']
                eligible = [change['mean_error_change'] for change in changes if change['mean_error_change'] is not None]
                totals.update(mean_error_change=totals['signed_error_change_sum']/count if count else None,
                    scene_macro_error_change=float(np.mean(eligible)) if eligible else None,
                    eligible_scene_count=len(eligible),
                    second_better_scene_count=sum(value < -1e-9 for value in eligible),
                    second_worse_scene_count=sum(value > 1e-9 for value in eligible),
                    tied_scene_count=sum(abs(value) <= 1e-9 for value in eligible))
                pooled['error_change_second_minus_first'][region][attr] = totals
        result['pairwise'][arm] = pooled
    return result


def _compact_scene(row):
    """Full stratified totals are in summary; source NPZs retain every point."""
    regions = ('all', 'same_representative', 'changed_representative')
    return dict(scene_id=row['scene_id'], fragment_count=row['fragment_count'],
        source_replay_verified=True, unchanged_ols_controls_verified=True,
        methods={side: {name: {key: item[key] for key in ('population', 'full_target_metrics')}
                  for name, item in methods.items()} for side, methods in row['methods'].items()},
        pairwise={arm: dict(
            methods={side: {region: metrics[region] for region in regions}
                     for side, metrics in pair['methods'].items()},
            error_change_second_minus_first={region: pair['error_change_second_minus_first'][region] for region in regions},
            newly_covered=pair['newly_covered']['all'], lost_coverage=pair['lost_coverage']['all'],
            **{key: pair[key] for key in ('coverage_transitions', 'same_representative_target_count',
                                          'changed_representative_target_count')})
            for arm, pair in row['pairwise'].items()})


def run(args):
    sources = {name: Path(getattr(args, name + '_system')).resolve() for name in ('reference', 'candidate')}
    output = Path(args.output).resolve()
    if sources['reference'] == sources['candidate'] or any(
            output == source or output in source.parents or source in output.parents for source in sources.values()):
        raise ValueError('distinct sources and a separate new output directory required')
    if type(args.cpu_threads) is not int or args.cpu_threads < 1:
        raise ValueError('cpu_threads must be a positive integer')
    torch.set_num_threads(args.cpu_threads)
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    state = dict(status='RUNNING', scope='060-association-duration-cached-comparison', dataset_level='Level1',
        split='V_select', sources=sources, optimizer_updates=0, gpu_used=False, test_opened=False,
        archive_members_read=0, reference_members_read=0, cached_reference_used=True,
        threshold_reselected=False, ols_refitted=False, cpu_threads=args.cpu_threads, completed_scenes=0)
    write_json(output/'run.json', state)
    try:
        manifest = read(args.manifest)
        ids = manifest.get('splits', {}).get('V_select', [])
        if (manifest.get('dataset_level') != 'Level1' or manifest.get('split_unit') != 'scene'
                or not ids or len(ids) != len(set(ids))
                or any(not isinstance(sid, str) or Path(sid).name != sid or sid in ('.', '..') for sid in ids)):
            raise ValueError('complete Level1 scene-level V manifest required')
        metadata = {name: _load_metadata(sources[name], manifest, step)
                    for name, step in (('reference', 1000), ('candidate', 3000))}
        if (metadata['reference']['ols_working_point'] != metadata['candidate']['ols_working_point']
                or metadata['reference']['fragment_count'] != metadata['candidate']['fragment_count']):
            raise ValueError('OLS working point and complete population must remain fixed')
        prefix = _prefix_status(metadata)
        rows = []
        for sid in ids:
            reference, reference_saved = _load_arrays(sources['reference'], sid)
            candidate, candidate_saved = _load_arrays(sources['candidate'], sid)
            row = compare_scene(reference, candidate, reference_saved, candidate_saved,
                metadata['reference']['expected'][sid], metadata['candidate']['expected'][sid])
            rows.append(dict(scene_id=sid, **row))
            state.update(completed_scenes=len(rows), total_scenes=len(ids), elapsed_seconds=time.monotonic()-started)
            write_json(output/'run.json', state)
        fragment_count = sum(row['fragment_count'] for row in rows)
        if fragment_count != metadata['reference']['fragment_count']:
            raise ValueError('fragment population differs from source summary')
        pooled = aggregate(rows)
        for source in sources:
            for name in METHODS:
                assert_metrics(pooled['methods'][source][name]['full_target_metrics'],
                               metadata[source]['pooled'][name], source + '/' + name)
        write_gzip(output/'scene-summary.json.gz', [_compact_scene(row) for row in rows])
        pooled.update(state)
        pooled.update(status='COMPLETED', scene_ids=ids, fragment_count=fragment_count,
            prefix_comparison=prefix, causal_attribution_valid=prefix['causal_attribution_valid'],
            endpoints={name: {key: metadata[name][key] for key in ('endpoint_step', 'association_threshold', 'association')}
                       for name in sources},
            unchanged_ols_controls_verified=True, source_replay_verified=True,
            definitions=dict(population='G=predicted groups; U=anchor-matched targets; D=duplicate groups; G=U+D',
                common='same registered target and Q64 point, reference-valid and prediction-valid in both runs',
                representatives='saved aligned_group_indices and original first-node group anchors; no truth-error reselection',
                strata='fixed true-identity union of input support and minimum nearest-input distance; evaluation only'),
            limitations=['repeated V development population and one seed; not independent generalization',
                'conditional MAE must be read with coverage, duplicates and gained/lost target points',
                'same-representative Anchor outputs are identical; changed-Anchor gains are selection effects, not a better native estimator',
                'time points within scenes/identities are correlated; scene macros are reported separately',
                'MISMATCH or absent prefix comparison permits descriptive results only, not pure training-duration attribution'])
        write_json(output/'summary.json', pooled)
        state.update(status='COMPLETED', elapsed_seconds=time.monotonic()-started)
        write_json(output/'run.json', state)
        return pooled
    except BaseException as error:
        state.update(status='INTERRUPTED' if isinstance(error, (KeyboardInterrupt, InterruptedError)) else 'FAILED',
                     error=f'{type(error).__name__}: {error}', elapsed_seconds=time.monotonic()-started)
        write_json(output/'run.json', state)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('reference-system', 'candidate-system', 'manifest', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--cpu-threads', type=int, default=4)
    args = parser.parse_args()
    run(args)
    print(f'COMPLETED: {args.output}/summary.json')


if __name__ == '__main__':
    main()
