"""Evaluate a new-unit fixed TFM endpoint on frozen 056 groups; no training.

The result is a same-task adaptation, not an author reproduction. A separate
research contract must authorize any actual trained endpoint and server run.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
import signal
import sys
import time

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch

from association.data.dataset import load_timed_dataset
from association.data.physical_units import SPEED_UNIT_POLICY, normalizer_speed_policy
from association.evaluation.artifacts import ScenePredictionWriter
from association.evaluation.evidence_fusion_combo import read_scores
from association.evaluation.fixed056_controls import Fixed056Controls, NAMES
from association.evaluation.grouping import group_predicted_tracks
from association.evaluation.tsactfer_level1 import _records_for
from association.evaluation.tsactfer_level1 import verify_native_source
from association.evaluation.association import evaluate_scenes
from association.evaluation.fusion_controls import attribute_errors
from association.evaluation.tmtaff import evaluate_complete
from association.models.tmtaff import ModelConfig
from association.models.fusion import PairFusionConfig
from association.training.tmtaff import _scenes, load_complete_snapshot
from association.utils import runtime
from tools.analysis.compare_association_duration import _load_metadata, _load_arrays, _replay
from tools.analysis.compare_physical_predictions import _equal
from tools.analysis.frozen_fusion_transfer import groups_from_cache
from tools.analysis.diagnose_three_scenes import source_metadata as reference090c_metadata
from tools.analysis.coordinate_metrics import _artifact as old_artifact
from tools.experiments.completion_value_fit import _same_tree
from tools.experiments.level1_system_evaluate import common_target_metrics


def bind_endpoint(directory):
    """Require a completed new-unit TFM's own fixed final checkpoint."""
    directory = Path(directory)
    run = runtime.read_json(directory/'run.json')
    prep = runtime.read_json(directory/'preparation.json')
    payload = torch.load(directory/'checkpoint.pt', map_location='cpu', weights_only=True)
    cfg = prep['configuration']
    updates = run['config']['updates']['TFM_FROZEN']
    cursor = payload['cursor']
    if (run.get('status') != 'COMPLETED' or run.get('phase') != 'tfm'
            or run.get('final_evaluation_status') != 'COMPLETED' or run.get('test_opened') is not False
            or payload.get('format_version') != 1 or payload.get('phase') != 'tfm'
            or payload.get('input_adapter') != 'native_time_v1'
            or payload['config'] != run['config'] or cfg['training'] != run['config']
            or cursor['completed_updates'] != updates or run['cursor']['completed_updates'] != updates
            or not _same_tree(run['checkpoint_cursor'], cursor) or run.get('selection_performed') is not False
            or payload['model_config'] != asdict(ModelConfig(**cfg['model']))
            or payload['fusion_config'] != asdict(PairFusionConfig(**cfg['fusion']))
            or normalizer_speed_policy(payload['normalizer']) != SPEED_UNIT_POLICY
            or cfg['data'].get('speed_unit_policy') != SPEED_UNIT_POLICY):
        raise ValueError('TFM source is not a completed fixed final new-unit endpoint')
    if (ModelConfig(**cfg['model']).length != 64 or cfg['data']['max_observations'] != 128
            or cfg['data']['time_unit_seconds'] != 20.
            or cfg['data'].get('reference_longitude_policy') != 'shortest_arc'
            or ModelConfig(**cfg['model']).completion_policy != 'learned'
            or run['config']['singleton_policy'] != 'observed' or cfg['train_splits'] != ['T_core']):
        raise ValueError('fixed056 comparison requires Q64, cap128, 20s, shortest_arc, learned completion and observed singletons')
    return run, cfg, payload


def source_exposure(run, payload):
    """Report actual source budgets; do not infer missing sampling journals."""
    init = payload['initialization']
    source = Path(init['source_run'])
    tcm_run, tcm_summary = (runtime.read_json(source/name) for name in ('run.json', 'summary.json'))
    if (init.get('source_phase') != 'tcm' or tcm_run.get('status') != 'COMPLETED'
            or tcm_run.get('phase') != 'tcm' or tcm_run.get('test_opened') is not False
            or tcm_run['cursor']['completed_updates'] != init['source_tcm_updates']):
        raise ValueError('TCM initialization source does not match completed endpoint')
    if normalizer_speed_policy(tcm_summary['normalizer']) != SPEED_UNIT_POLICY:
        raise ValueError('TCM initialization source does not use the new speed unit')
    if not _same_tree(tcm_summary['normalizer'], payload['normalizer']):
        raise ValueError('TCM and TFM physical normalizers differ')
    tcm = torch.load(source/'checkpoint.pt', map_location='cpu', weights_only=True)
    frozen = [key for key in tcm['states'][0] if key in ('positions', 'heading_mean_std', 'speed_unit_contract')
              or key.startswith(('input_embedding.', 'tcm_encoder.', 'tcm_decoder.'))]
    if (tcm.get('phase') != 'tcm' or not frozen or not _same_tree(tcm['data_contract'], payload['data_contract'])
            or any(not _same_tree(tcm['states'][0][key], payload['states'][0].get(key)) for key in frozen)):
        raise ValueError('TFM frozen completion differs from its TCM source')
    fields = ('unique_scene_count', 'unique_track_count', 'unique_pair_count',
              'track_presentations', 'transition_presentations', 'pair_presentations',
              'positive_pair_presentations', 'completed_epochs')
    def compact(item):
        return {key: item.get(key) for key in fields}
    tfm_history = [row for row in payload['history'] if row.get('stage') == 'TFM_FROZEN']
    if len(tfm_history) != payload['cursor']['completed_updates']:
        raise ValueError('TFM history does not cover the actual completed updates')
    counts = [row.get('fusion_valid_counts') for row in tfm_history]
    return dict(tcm=dict(source_run=str(source), completed_updates=init['source_tcm_updates'],
                         elapsed_seconds=tcm_run.get('elapsed_seconds'),
                         device=tcm_run.get('device'),
                         cuda_peak_allocated_bytes=tcm_run.get('cuda_process_peak_allocated_bytes'),
                         cuda_peak_reserved_bytes=tcm_run.get('cuda_process_peak_reserved_bytes'),
                         exposure=compact(tcm_summary['coverage']['TCM'])),
                tfm=dict(completed_updates=payload['cursor']['completed_updates'],
                         elapsed_seconds=run.get('elapsed_seconds'),
                         device=run.get('device'),
                         cuda_peak_allocated_bytes=run.get('cuda_process_peak_allocated_bytes'),
                         cuda_peak_reserved_bytes=run.get('cuda_process_peak_reserved_bytes'),
                         exposure=compact(payload['coverage']['TFM_FROZEN']),
                         valid_attribute_presentations=(np.sum(counts, axis=0).tolist()
                             if all(isinstance(x, (list, tuple)) and len(x) == 4 for x in counts) else None),
                         ordered_pair_draws_saved=bool(payload.get('pairing_audit'))),
                note='Missing journal fields are null; unique exposures are not independent observations.')


def _partition_membership(fragment, *, name):
    """Validate an ordered, complete partition and derive its membership.

    Historical 090C fragments save offsets/indices but no membership array.
    Derivation is from those actual arrays, never a new grouping decision.
    """
    n = len(fragment['node_ids'])
    offsets = np.asarray(fragment['learned_assoc_group_offsets'])
    flat = np.asarray(fragment['learned_assoc_group_node_indices'])
    if (offsets.ndim != 1 or not len(offsets) or offsets.dtype.kind not in 'iu'
            or flat.shape != (n,) or flat.dtype.kind not in 'iu'
            or offsets[0] != 0 or offsets[-1] != n
            or np.any(offsets[1:] <= offsets[:-1])
            or not np.array_equal(np.sort(flat), np.arange(n))):
        raise ValueError(name+' ordered groups do not partition the complete nodes')
    membership = np.empty(n, dtype=np.int64)
    for group, (start, stop) in enumerate(zip(offsets[:-1], offsets[1:])):
        membership[flat[start:stop]] = group
    if ('learned_assoc_membership' in fragment
            and not _equal(fragment['learned_assoc_membership'], membership)):
        raise ValueError(name+' cached membership differs from its ordered groups')
    if ('group_anchor_node_indices' in fragment
            and not _equal(fragment['group_anchor_node_indices'], flat[offsets[:-1]])):
        raise ValueError(name+' cached anchors differ from its ordered groups')
    return membership


def _bind_reference_fragment(fragment, old, record):
    """Bind 090C's real legacy schema to the verified056 score population.

    The caller must first bind completed090C metadata to this056 source.
    Missing legacy sources are established transitively through exact node
    order and the056 score cache. Missing membership is derived from the actual
    complete partition. Neither missing field is inserted into the old cache.
    """
    for key in ('node_ids', 'sources'):
        if not _equal(fragment[key], record[key]):
            raise ValueError('056 score population differs from cached input: '+key)
    for key in ('node_ids', 'identities', 'target_identity_ids', 'query_seconds',
                'learned_assoc_group_offsets', 'learned_assoc_group_node_indices',
                'native_states', 'support', 'nearest_seconds', 'exact_observation'):
        if not _equal(fragment[key], old[key]):
            raise ValueError('090C and 056 cached input/groups differ: '+key)
    if 'sources' in old and not _equal(fragment['sources'], old['sources']):
        raise ValueError('090C and 056 cached input/groups differ: sources')
    if not _equal(_partition_membership(fragment, name='056'),
                  _partition_membership(old, name='090C')):
        raise ValueError('090C and 056 derived group memberships differ')
    return dict(sources=('explicit_cache' if 'sources' in old else
                         'verified056_scores_via_same_source_system_and_exact_node_order'),
                membership=('explicit_cache_verified_against_ordered_partition'
                            if 'learned_assoc_membership' in old else
                            'derived_from_complete_ordered_partition'))


def bind_scenes(scenes, records, source, metadata, reference090c=None):
    """Validate all saved groups and reference arrays before model inference."""
    if [str(s['scene_id']) for s in scenes] != metadata['scene_ids']:
        raise ValueError('prepared V scene order differs from fixed056')
    if len(records) != len(scenes):
        raise ValueError('fixed056 score records do not cover every prepared scene')
    reference_bindings = []
    if reference090c is not None:
        reference_run = runtime.read_json(Path(reference090c)/'run.json')
        if (reference_run.get('status') != 'COMPLETED'
                or not reference_run.get('source_system')
                or Path(reference_run['source_system']).resolve() != Path(source).resolve()):
            raise ValueError('completed090C metadata does not bind this frozen056 source')
    for scene, record in zip(scenes, records):
        sid = str(scene['scene_id'])
        fragment, saved = _load_arrays(source, sid)
        if reference090c is not None:
            with np.load(Path(reference090c)/'final/predictions'/f'states-{sid}.npz', allow_pickle=False) as old:
                reference_bindings.append(_bind_reference_fragment(fragment, old, record))
        _, _, _, truth, valid, _ = _replay(fragment, saved, metadata['expected'][sid])
        for key in ('node_ids', 'identities', 'sources'):
            if not np.array_equal(np.asarray(scene[key]), fragment[key]):
                raise ValueError('fixed056 node population differs: '+key)
        if not np.array_equal(np.asarray(scene['grid']), fragment['query_seconds']):
            raise ValueError('fixed056 query grid differs')
        targets = scene['fusion_targets']
        if list(targets) != fragment['target_identity_ids'].tolist():
            raise ValueError('fixed056 target registry differs')
        for index, target in enumerate(targets.values()):
            if (not np.array_equal(target['states'], truth[index], equal_nan=True)
                    or not np.array_equal(target['valid'], valid[index])):
                raise ValueError('fixed056 aligned reference or mask differs')
        groups = group_predicted_tracks(len(scene['node_ids']), record['pairs'], record['scores'],
            threshold=metadata['association_threshold'], grouping_policy='anchor_direct', node_order='input')
        if [list(group) for group in groups] != groups_from_cache(fragment, 'learned_assoc'):
            raise ValueError('fixed056 scores/threshold do not reproduce original ordered groups')
    return dict(scene_count=len(scenes), fixed_groups_exact=True,
                aligned_reference_exact=True, scores_and_threshold_fixed=True,
                reference090c_input_binding=(dict(scene_count=len(reference_bindings),
                    source_system_metadata_exact=True,
                    legacy_sources_bound_via056_scores=sum(row['sources'] != 'explicit_cache'
                        for row in reference_bindings),
                    legacy_membership_derived_from_ordered_partition=sum(
                        row['membership'] == 'derived_from_complete_ordered_partition'
                        for row in reference_bindings),
                    exact_observation_checked=True) if reference090c is not None else None))


def verify_056_association(records, threshold, expected):
    """Bind all original candidates and both paper scopes, scene by scene.

    ITSA/MSTSA here mean same-source/cross-source across the complete frozen
    candidate population. A separate short-window protocol must not be silently
    introduced into the original 056/102 eight metrics.
    """
    if expected.get('threshold') != threshold:
        raise ValueError('fixed056 original association threshold differs')
    actual = evaluate_scenes(records, threshold)
    for scope in ('all', 'itsa', 'mstsa'):
        a = actual if scope == 'all' else actual[scope]
        b = expected if scope == 'all' else expected[scope]
        for key in ('tp', 'fp', 'fn', 'tn', 'pair_count'):
            if a[key] != b[key]:
                raise ValueError('fixed056 association aggregate differs: '+scope+'/'+key)
        if len(a['per_scene']) != len(b['per_scene']):
            raise ValueError('fixed056 association scene population differs: '+scope)
        for left, right in zip(a['per_scene'], b['per_scene']):
            for key in ('scene_id', 'component_id', 'pair_count', 'tp', 'fp', 'fn', 'tn'):
                if left[key] != right[key]:
                    raise ValueError('fixed056 association per-scene counts differ: '+scope+'/'+key)
        for key in ('m_pca', 'mw_pca', 'm_pfa', 'mw_pfa'):
            x, y = a['paper_metrics'][key], b['paper_metrics'][key]
            if (x is None) != (y is None) or (x is not None and not np.isclose(x, y, rtol=1e-12, atol=1e-14)):
                raise ValueError('fixed056 paper association metric differs: '+scope+'/'+key)
    return actual


class ComparisonWriter:
    """Save complete artifacts and per-pair per-attribute intersections."""
    def __init__(self, directory, reference090c=None):
        self.writer = ScenePredictionWriter(directory)
        self.rows = []
        self.reference090c = Path(reference090c) if reference090c else None
        self.names = (*NAMES, *(['internal_090c'] if self.reference090c else []))
        self.errors = {name: {arm: {attr: [] for attr in ('position', 'speed', 'heading')}
                              for arm in ('tmtaff', name)} for name in self.names}
        self.component_errors = {}

    def __call__(self, index, scene, artifacts):
        self.writer(index, scene, artifacts)
        component = str(scene['component_id'])
        if component not in self.component_errors:
            self.component_errors[component] = {name: {arm: {attr: [] for attr in ('position', 'speed', 'heading')}
                for arm in ('tmtaff', name)} for name in self.names}
        pairs = {}
        for name in NAMES:
            paired, mask = common_target_metrics({'tmtaff': artifacts['predicted'], name: artifacts[name]})
            count = len(artifacts['predicted']['target_ids'])
            truth = artifacts['predicted']['aligned_truth'][:count]
            truth_valid = artifacts['predicted']['aligned_truth_valid'][:count]
            pairs[name] = dict(per_attribute_common_prediction_counts=mask.sum(axis=(0, 1)).tolist(),
                per_attribute_evaluable_common_counts=(mask & truth_valid).sum(axis=(0, 1)).tolist(), methods=paired)
            for arm, key in (('tmtaff', 'predicted'), (name, name)):
                values = attribute_errors(artifacts[key]['aligned_prediction'][:count], truth)
                for attr, cols in (('position', (0, 1)), ('speed', (2,)), ('heading', (3,))):
                    valid = mask[..., list(cols)].all(-1) & truth_valid[..., list(cols)].all(-1)
                    errors = values[attr][valid]
                    self.errors[name][arm][attr].append(errors)
                    self.component_errors[component][name][arm][attr].append(errors)
        if self.reference090c:
            with np.load(self.reference090c/'final/system-predictions'/f"states-{scene['scene_id']}.npz",
                         allow_pickle=False) as saved:
                old = old_artifact(saved, 'learned_sequence')
            current = artifacts['predicted']
            count = len(current['target_ids'])
            for key in ('target_ids', 'aligned_group_indices'):
                if not np.array_equal(old[key][:count], current[key][:count]):
                    raise ValueError('090C and TMTAFF fixed target/group binding differ: '+key)
            for key in ('aligned_truth', 'aligned_truth_valid'):
                if not np.array_equal(old[key][:count], current[key][:count], equal_nan=True):
                    raise ValueError('090C and TMTAFF reference/mask differ: '+key)
            truth = current['aligned_truth'][:count]
            valid = (current['aligned_truth_valid'][:count, :, :2].all(-1)
                     & current['aligned_prediction_valid'][:count, :, :2].all(-1)
                     & old['aligned_prediction_valid'][:count, :, :2].all(-1))
            for arm, prediction in (('tmtaff', current['aligned_prediction'][:count]),
                                    ('internal_090c', old['aligned_prediction'][:count])):
                errors = attribute_errors(prediction, truth)['position'][valid]
                self.errors['internal_090c'][arm]['position'].append(errors)
                self.component_errors[component]['internal_090c'][arm]['position'].append(errors)
            pairs['internal_090c'] = dict(common_position_points=int(valid.sum()),
                scope='position only; 090C speed/heading inherited old Mean and are not compared here')
        self.rows.append(dict(scene_id=str(scene['scene_id']), component_id=str(scene['component_id']), pairs=pairs))

    def pooled(self, data=None):
        result = {}
        for name, arms in (self.errors if data is None else data).items():
            result[name] = {}
            for arm, attrs in arms.items():
                result[name][arm] = {}
                for attr, parts in attrs.items():
                    values = np.concatenate(parts) if parts else np.empty(0)
                    result[name][arm][attr] = dict(count=len(values),
                        mae=float(values.mean()) if len(values) else None,
                        rmse=float(np.sqrt(np.mean(values**2))) if len(values) else None,
                        p95=float(np.percentile(values, 95)) if len(values) else None,
                        p99=float(np.percentile(values, 99)) if len(values) else None,
                        unit={'position': 'm', 'speed': 'mps', 'heading': 'degrees'}[attr])
        return result

    def largest_component_excluded(self):
        counts = {key: sum(row['component_id'] == key for row in self.rows) for key in self.component_errors}
        if not counts:
            return None
        largest = sorted(counts, key=lambda key: (-counts[key], key))[0]
        merged = {name: {arm: {attr: [] for attr in ('position', 'speed', 'heading')}
                         for arm in ('tmtaff', name)} for name in self.names}
        for component, data in self.component_errors.items():
            if component == largest:
                continue
            for name, arms in data.items():
                for arm, attrs in arms.items():
                    for attr, parts in attrs.items():
                        merged[name][arm][attr].extend(parts)
        return dict(excluded_component=largest, excluded_scene_count=counts[largest],
                    remaining_scene_count=len(self.rows)-counts[largest], pairwise_common=self.pooled(merged))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('tfm-run', 'source-system', 'reference090c-run', 'manifest', 'alignment', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--archive', type=Path, nargs='+', required=True)
    parser.add_argument('--device', required=True)
    parser.add_argument('--cpu-threads', type=int, default=4)
    args = parser.parse_args(argv)
    started = time.monotonic()
    if args.cpu_threads < 1:
        raise ValueError('CPU threads must be positive')
    output = args.output.resolve()
    for source in (args.tfm_run.resolve(), args.source_system.resolve(), args.reference090c_run.resolve()):
        if output == source or source in output.parents or output in source.parents:
            raise ValueError('output must not overlap a source run')
    output.mkdir(parents=True, exist_ok=False)
    state = dict(status='RUNNING', optimizer_updates=0, tam_scoring_forwards=0,
        threshold_reselected=False, test_opened=False, source_system=str(args.source_system.resolve()),
        tfm_run=str(args.tfm_run.resolve()), device=args.device)
    runtime.write_json(output/'run.json', state)
    def terminated(signum, frame):
        raise InterruptedError('SIGTERM: preserve incomplete evaluation; no retry')
    previous_signal = signal.signal(signal.SIGTERM, terminated)
    try:
        torch.set_num_threads(args.cpu_threads)
        run, cfg, payload = bind_endpoint(args.tfm_run)
        exposure = source_exposure(run, payload)
        manifest, alignment = runtime.read_json(args.manifest), runtime.read_json(args.alignment)
        metadata = _load_metadata(args.source_system, manifest, endpoint_step=1000)
        reference = reference090c_metadata(args.reference090c_run, manifest)
        reference_run = runtime.read_json(args.reference090c_run/'run.json')
        if Path(reference_run['source_system']).resolve() != args.source_system.resolve():
            raise ValueError('090C does not bind this same frozen056 source system')
        def verify_visible(native):
            if native['tracks']:
                verify_native_source(native, args.source_system)
            else:
                path = args.source_system/'predictions'/f"states-{native['scene_id']}.npz"
                with np.load(path, allow_pickle=False) as saved:
                    if len(saved['node_ids']) or len(native['node_ids']):
                        raise ValueError('empty native scene differs from frozen056')
        dataset = load_timed_dataset(args.archive, manifest, train_splits=cfg['train_splits'],
            data_config={**cfg['data'], 'length': 64}, alignment_by_scene=alignment,
            validation_native_verifier=verify_visible)
        if not _same_tree(_scenes(dataset['train'], dataset['validation'], 64), payload['data_contract']):
            raise ValueError('prepared scene contract differs from TFM training endpoint')
        scenes = dataset['validation']
        records = _records_for(read_scores(args.source_system.parent/'association/final-scores.npz'), scenes)
        verification = bind_scenes(scenes, records, args.source_system, metadata, args.reference090c_run)
        verification['raw_visible_to_frozen056_motion_exact'] = True
        original_association = verify_056_association(records, metadata['association_threshold'], metadata['association'])
        verification['association_scopes_and_scene_counts_exact'] = True
        model = load_complete_snapshot(args.tfm_run/'checkpoint.pt', device=args.device, selected=False)
        writer = ComparisonWriter(output/'predictions', args.reference090c_run)
        device = runtime.device_from_string(args.device)
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device)
        result = evaluate_complete(model['model'], model['fusion'], model['normalizer'], scenes,
            device=args.device, batch_tracks=run['config']['validation_batches']['tracks'],
            pair_batch_size=run['config']['validation_batches']['pairs'], select_on_validation=False,
            threshold=metadata['association_threshold'], singleton_policy=run['config']['singleton_policy'],
            association_records=records, predicted_controls=True,
            extra_predicted_controls=Fixed056Controls(args.source_system), prediction_writer=writer)
        result.pop('records', None)
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        cost = dict(device=str(device), evaluation_seconds=result['elapsed_seconds'],
            model_parameters=sum(p.numel() for p in model['model'].parameters()),
            fusion_parameters=sum(p.numel() for p in model['fusion'].parameters()),
            peak_cuda_allocated_bytes=(torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else None),
            peak_cuda_reserved_bytes=(torch.cuda.max_memory_reserved(device) if device.type == 'cuda' else None),
            cuda_peak_scope='evaluation after model load; training peak is in each source run')
        runtime.write_json(output/'summary.json', dict(scope='TMTAFF same-task time adaptation, fixed056',
            evaluation=result, source_verification=verification, common_by_scene=writer.rows,
            internal_090c_source=reference,
            pairwise_common_pooled=writer.pooled(),
            component_common={key: writer.pooled(value) for key, value in writer.component_errors.items()},
            excluding_largest_component=writer.largest_component_excluded(),
            inherited_056_association=original_association,
            evaluation_cost=cost,
            training_exposure=exposure, tfm_config=run['config'],
            tfm_cursor=payload['cursor'], test_opened=False))
        state.update(status='COMPLETED', scenes=len(scenes), source_verification=verification)
    except BaseException as error:
        state.update(status='FAILED', error_type=type(error).__name__, error=str(error))
        raise
    finally:
        signal.signal(signal.SIGTERM, previous_signal)
        state['elapsed_seconds'] = time.monotonic()-started
        runtime.write_json(output/'run.json', state)


if __name__ == '__main__':
    main()
