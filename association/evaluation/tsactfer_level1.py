"""074D TSA scores to full-system outputs; frozen056 physical inputs, new groups.

References are used for frozen-cache replay and metrics, never to decide the new
association scores, predicted groups, representatives or physical outputs.
The original online TSA assignment policy is not implemented or claimed here.
"""
from __future__ import annotations

from pathlib import Path
import time

import numpy as np

from association.data.timed import prepare_timed_scene
from association.data.motion_evidence import scene_motion_evidence
from association.evaluation.association import evaluate_scenes
from association.evaluation.evidence_fusion_combo import read_scores
from association.evaluation.fusion_controls import ordinary_fusion, attribute_errors
from association.evaluation.grouping import group_predicted_tracks
from association.evaluation.system_attribution import ATTRIBUTES, _same
from association.evaluation.oracle_diagnostics import POPULATION_KEYS
from association.evaluation.tmtaff import aggregate_fusion_reports
from association.utils.runtime import read_json, write_json, write_json_gzip
from tools.analysis.compare_association_duration import ARMS, _load_arrays, _replay
from tools.analysis.frozen_fusion_transfer import groups_from_cache
from tools.experiments.level1_system_evaluate import (
    ordinary_predictions, evaluate_unique_population, common_target_metrics, _fragment_regions,
    _pool_fragment, metrics, _verify_association_metrics)

METHODS = tuple(group+'_'+arm for group in ('ols', 'tsa') for arm in ARMS)


def verify_native_source(native, source):
    """Verify current raw V inputs match frozen056 visible physical evidence.

    This reads no physical reference, never builds a target or changes a score.
    Called before the full canonical observations are released by the loader.
    """
    with np.load(Path(source)/'predictions'/f'states-{native["scene_id"]}.npz', allow_pickle=False) as saved:
        for key in ('node_ids', 'sources', 'identities'):
            _same(native[key], saved[key], '074D raw/frozen V '+key)
        n = len(native['node_ids'])
        _same(native['pairs'], np.column_stack(np.triu_indices(n, 1)), '074D all native pairs')
        prepared = prepare_timed_scene(native, length=64, max_observations=128, time_unit_seconds=20.)
        motion = scene_motion_evidence(prepared, motion_observation_policy='canonical_all')
        for key, old in (('states', 'native_states'), ('query_seconds', 'query_seconds'),
                         ('support', 'support'), ('nearest_seconds', 'nearest_seconds'),
                         ('exact_observation', 'exact_observation')):
            _same(motion[key], saved[old], '074D raw/frozen motion '+key)


def paper_masks(record):
    pairs, intervals = np.asarray(record['pairs']), np.asarray(record['intervals'])
    a, b = intervals[pairs[:, 0]], intervals[pairs[:, 1]]
    same = np.asarray(record['sources'])[pairs[:, 0]] == np.asarray(record['sources'])[pairs[:, 1]]
    gap = np.maximum(a[:, 0], b[:, 0])-np.minimum(a[:, 1], b[:, 1])
    return dict(itsa=same & (gap >= 0) & (gap < 1200) &
                    ((a[:, 1]-a[:, 0]) > 120) & ((b[:, 1]-b[:, 0]) > 120),
                mstsa=~same & (-gap > 120))


def scope_metrics(records, threshold):
    """Same original paper eligibility, with the full-population working point."""
    result = {}
    for task in ('itsa', 'mstsa'):
        subset, count, positives = [], 0, 0
        for row in records:
            mask = paper_masks(row)[task]
            subset.append(dict(row, **{key: np.asarray(row[key])[mask] for key in ('pairs', 'labels', 'scores')}))
            count += int(mask.sum()); positives += int(np.asarray(row['labels'])[mask].sum())
        result[task] = dict(metrics=evaluate_scenes(subset, threshold), candidate_count=count,
            positive_count=positives, threshold=threshold, threshold_reselected=False)
    return result


def _records_for(cache, templates):
    if set(cache) != {str(row['scene_id']) for row in templates}:
        raise ValueError('source score population differs from complete V')
    records = []
    for row in templates:
        item = cache[str(row['scene_id'])]
        for key in ('node_ids', 'pairs', 'sources'):
            _same(item[key], row[key], '074D source candidate '+key)
        for key in ('identities', 'labels'):
            if key in item:
                _same(item[key], row[key], '074D source candidate '+key)
        scores = np.asarray(item['scores'])
        if scores.shape != (len(row['pairs']),) or not np.isfinite(scores).all():
            raise ValueError('074D source scores must be finite and one per candidate')
        records.append(dict(row, scores=scores))
    return records


def _source_records(records, templates):
    """Test injections and file caches pass the identical population contract."""
    if len({str(row['scene_id']) for row in records}) != len(records):
        raise ValueError('074D source scores contain duplicate scenes')
    return _records_for({str(row['scene_id']): row for row in records}, templates)


def _replay_score_groups(row, threshold, groups, n, name):
    actual = group_predicted_tracks(n, row['pairs'], row['scores'], threshold=threshold,
        grouping_policy='anchor_direct', node_order='input')
    if [list(group) for group in actual] != [list(group) for group in groups]:
        raise ValueError('074D '+name+' source scores/threshold do not reproduce frozen groups')


def _scene(fragment, truth, valid, sid, component):
    targets = fragment['target_identity_ids']
    return dict(scene_id=sid, component_id=component, split='V_select',
        identities=fragment['identities'], target_identity_ids=targets,
        target=fragment['target'], valid=fragment['target_valid'],
        fusion_targets={str(identity): dict(states=truth[i], valid=valid[i]) for i, identity in enumerate(targets)},
        motion=dict(states=fragment['native_states'], support=fragment['support'],
                    nearest_seconds=fragment['nearest_seconds'], query_seconds=fragment['query_seconds']))


def evaluate_system(records, threshold, source056, metadata, output, *, check=None,
                    ols_records=None, reference_records=None):
    """OLS/TSA×3 plus paired056 comparisons, without any neural forward."""
    source056, output = Path(source056).resolve(), Path(output).resolve()
    if output == source056.parent or source056.parent in output.parents or output in source056.parent.parents:
        raise ValueError('074D output cannot overlap immutable056')
    ids = [str(row['scene_id']) for row in records]
    if ids != metadata['scene_ids'] or len(set(ids)) != len(ids):
        raise ValueError('074D complete V score order required')
    for row, sid in zip(records, ids):
        scores = np.asarray(row['scores'])
        if scores.shape != (len(row['pairs']),) or not np.isfinite(scores).all():
            raise ValueError('074D final scores must be finite and one per candidate')
        if str(row['component_id']) != str(metadata['component_ids'][sid]):
            raise ValueError('074D score component differs from frozen manifest')
    if ols_records is None:
        ols_records = _records_for(read_scores(source056.parent/'association/baseline-scores.npz'), records)
    else:
        ols_records = _source_records(ols_records, records)
    if reference_records is None:
        reference_records = _records_for(read_scores(source056.parent/'association/final-scores.npz'), records)
    else:
        reference_records = _source_records(reference_records, records)
    baseline = read_json(source056.parent/'association/baseline.json')
    if (baseline['selection']['threshold'] != metadata['ols_working_point']['threshold']
            or baseline.get('motion_observation_policy') != 'canonical_all'):
        raise ValueError('074D OLS source threshold/policy differs from frozen056')
    _verify_association_metrics(evaluate_scenes(ols_records, metadata['ols_working_point']['threshold']),
                               baseline['selected_threshold']['metrics'])
    _verify_association_metrics(evaluate_scenes(reference_records, metadata['association_threshold']),
                               metadata['association'])
    output.mkdir(parents=True, exist_ok=False)
    for name in ('predictions', 'system-predictions'):
        (output/name).mkdir()
    started = time.monotonic()
    state = dict(status='RUNNING', scope='074D-tsa-full-system', source_system=str(source056),
        model_forwards=0, optimizer_updates=0, test_opened=False, threshold_reselected=False,
        completed_scenes=0, total_scenes=len(ids), method_names=list(METHODS))
    write_json(output/'run.json', state)
    rows, comembership, errors = [], {'ols': [], 'tsa': [], 'reference056': []}, {name: {a: [] for a in ATTRIBUTES} for name in METHODS}
    try:
        for row, ols_row, reference_row, sid in zip(records, ols_records, reference_records, ids):
            if check:
                check()
            fragment, cached = _load_arrays(source056, sid)
            replay = _replay(fragment, cached, metadata['expected'][sid])
            for key in ('node_ids', 'sources', 'identities'):
                _same(row[key], fragment[key], '074D score/frozen input '+key)
            n = len(fragment['node_ids'])
            pairs = np.column_stack(np.triu_indices(n, 1))
            _same(row['pairs'], pairs, '074D complete V candidate population')
            _same(row['labels'], (fragment['identities'][pairs[:, 0]] == fragment['identities'][pairs[:, 1]]).astype(np.float32),
                  '074D unchanged evaluation labels')
            partitions = dict(ols=groups_from_cache(fragment, 'ols'),
                tsa=group_predicted_tracks(n, row['pairs'], row['scores'], threshold=threshold,
                                          grouping_policy='anchor_direct', node_order='input'))
            _replay_score_groups(ols_row, metadata['ols_working_point']['threshold'], partitions['ols'], n, 'OLS')
            _replay_score_groups(reference_row, metadata['association_threshold'],
                                groups_from_cache(fragment, 'learned_assoc'), n, '056')
            motion = dict(states=fragment['native_states'], support=fragment['support'], nearest_seconds=fragment['nearest_seconds'])
            by_group, flat = {}, {}
            for name, groups in partitions.items():
                controls = ordinary_fusion(motion['states'], motion['nearest_seconds'], motion['support'], groups)
                by_group[name] = ordinary_predictions(controls, ARMS)
                flat.update({name+'_'+arm: values for arm, values in by_group[name].items()})
                if name == 'ols':
                    for arm in ARMS:
                        _same(flat[name+'_'+arm], fragment[name+'_'+arm], '074D exact OLS source replay')
            # References validate the old replay above and score these new
            # outputs here; they never enter the preceding prediction decisions.
            scene = _scene(fragment, replay[3], replay[4], sid, row['component_id'])
            reports, artifacts = evaluate_unique_population(scene, partitions, by_group)
            common, common_mask = common_target_metrics(artifacts)
            paired, pair_masks = {}, {}
            for arm in ARMS:
                pair, mask = common_target_metrics({
                    'tsa': artifacts['tsa_'+arm], 'reference056': replay[1]['learned_assoc_'+arm]})
                paired[arm] = pair; pair_masks[arm] = mask
            for name, groups in dict(partitions, reference056=groups_from_cache(fragment, 'learned_assoc')).items():
                membership = np.empty(n, np.int64)
                for group_index, nodes in enumerate(groups):
                    membership[nodes] = group_index
                scores = (membership[pairs[:, 0]] == membership[pairs[:, 1]]).astype(np.float64)
                comembership[name].append(dict(row, scores=scores))
            regions = _fragment_regions(scene['motion'])
            fragment_metrics = {name: {region: metrics(values, scene['target'], scene['valid'], mask)
                                      for region, mask in regions.items()} for name, values in flat.items()}
            per_scene = dict(scene_id=sid, component_id=str(row['component_id']), unique_target=reports,
                common_valid_target_points=common, comparisons_to_056=paired, fragment=fragment_metrics)
            rows.append(per_scene)
            target_count = len(fragment['target_identity_ids'])
            for name, artifact in artifacts.items():
                current = attribute_errors(artifact['aligned_prediction'][:target_count], artifact['aligned_truth'][:target_count])
                valid = artifact['aligned_prediction_valid'][:target_count] & artifact['aligned_truth_valid'][:target_count]
                for attr, columns in ATTRIBUTES.items():
                    errors[name][attr].append(current[attr][valid[..., list(columns)].all(-1)])
            payload = dict(node_ids=fragment['node_ids'], identities=fragment['identities'], sources=fragment['sources'],
                target_identity_ids=fragment['target_identity_ids'], pairs=row['pairs'], tsa_scores=row['scores'],
                query_seconds=fragment['query_seconds'], native_states=fragment['native_states'],
                support=fragment['support'], nearest_seconds=fragment['nearest_seconds'], **flat)
            for name, groups in partitions.items():
                payload[name+'_group_offsets'] = np.r_[0, np.cumsum([len(g) for g in groups])]
                payload[name+'_group_node_indices'] = np.array([node for group in groups for node in group], np.int64)
            np.savez_compressed(output/'predictions'/f'states-{sid}.npz', **payload)
            saved = {'common_prediction_valid_on_target_rows': common_mask}
            for name, artifact in artifacts.items():
                saved.update({name+'_'+key: value for key, value in artifact.items()})
            for arm in ARMS:
                saved['comparison056_'+arm+'_common_prediction_valid'] = pair_masks[arm]
                saved.update({'reference056_'+arm+'_'+key: value for key, value in replay[1]['learned_assoc_'+arm].items()})
            np.savez_compressed(output/'system-predictions'/f'states-{sid}.npz', **saved)
            state['completed_scenes'] = len(rows)
            write_json(output/'run.json', state)
        def pool(items):
            return dict(scene_count=len(items), methods={name: dict(
                population={key: sum(r['unique_target'][name][key] for r in items) for key in POPULATION_KEYS},
                full_target_metrics=aggregate_fusion_reports([r['unique_target'][name] for r in items]),
                common_valid_target_points=aggregate_fusion_reports([r['common_valid_target_points'][name] for r in items]))
                for name in METHODS}, comparisons_to_056={arm: {
                    side: aggregate_fusion_reports([r['comparisons_to_056'][arm][side] for r in items])
                    for side in ('tsa', 'reference056')} for arm in ARMS})
        summary = pool(rows)
        summary.update(status='COMPLETED', scene_ids=ids, source_system=str(source056),
            method_names=list(METHODS), endpoint='final', threshold=threshold,
            model_forwards=0, optimizer_updates=0, test_opened=False,
            association={name: dict(metrics=evaluate_scenes(value, point),
                                   paper_scope=scope_metrics(value, point)) for name, value, point in (
                ('ols', ols_records, metadata['ols_working_point']['threshold']),
                ('tsa', records, threshold), ('reference056', reference_records, metadata['association_threshold']))},
            group_comembership={name: evaluate_scenes(value, .5) for name, value in comembership.items()},
            fragment=_pool_fragment(rows, METHODS),
            components={component: pool([r for r in rows if r['component_id'] == component])
                        for component in sorted({r['component_id'] for r in rows})},
            limitations=['Offline native128 TSA adaptation; not author online assignment or convergence.',
                'Full V threshold selection uses the same Macro-F1 rule as056; paper subsets retain that working point.',
                'Different grouping can change coverage and first representative; conditional MAE is not a complete ranking.',
                'Frozen056 physical inputs and reference population; references never enter scoring/grouping.',
                'V is reused development data, not sealed test or unseen-identity evidence.'])
        for name in METHODS:
            summary['methods'][name]['attribute_quantiles'] = {}
            for attr, chunks in errors[name].items():
                values = np.concatenate(chunks)
                summary['methods'][name]['attribute_quantiles'][attr] = dict(count=len(values),
                    p95=float(np.quantile(values, .95)) if len(values) else None,
                    p99=float(np.quantile(values, .99)) if len(values) else None)
        write_json(output/'summary.json', summary)
        write_json_gzip(output/'scene-summary.json.gz', rows)
        state.update(status='COMPLETED', elapsed_seconds=time.monotonic()-started)
        write_json(output/'run.json', state)
        return summary
    except BaseException as error:
        state.update(status='FAILED', error=f'{type(error).__name__}: {error}')
        write_json(output/'run.json', state)
        raise
