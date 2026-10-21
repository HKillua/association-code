"""Assemble the frozen MTAD development split for the explicit time adaptation.

No split creation, legacy scene64 resampling, test access, clock inference or
model construction. Reference CSVs are consumed one scene at a time; only the
aligned targets remain in the returned model data.
"""
from __future__ import annotations

from contextlib import closing
from numbers import Real
import time

import numpy as np

from association.data.archive import DataError, MTADArchive, validate_manifest
from association.data.targets import build_scene_fusion_targets
from association.data.native import iter_native_split_from_archive
from association.data.reference_states import iter_reference_states
from association.data.timed import _settings, prepare_timed_scene
from association.data.motion_evidence import scene_motion_evidence, validate_motion_observation_policy


def _request(manifest, train_splits, data_config, alignment_by_scene):
    validate_manifest(manifest)
    if not isinstance(train_splits, (tuple, list)) or tuple(train_splits) not in (
            ('T_core',), ('T_core', 'T_head')):
        raise DataError('train_splits must explicitly be T_core or T_core+T_head in that order')
    required = {'length', 'max_observations', 'time_unit_seconds'}
    if not required.issubset(data_config) or set(data_config) - required - {'reference_longitude_policy'}:
        raise DataError('data_config requires length/max_observations/time_unit_seconds; '
                        'only reference_longitude_policy is optional')
    _settings(**{key: data_config[key] for key in required})
    if data_config.get('reference_longitude_policy', 'linear') not in ('linear', 'shortest_arc'):
        raise DataError('reference_longitude_policy must be linear or shortest_arc')
    splits = [*train_splits, 'V_select']
    requested = [sid for split in splits for sid in manifest['splits'][split]]
    if not manifest['splits']['T_core'] or not manifest['splits']['V_select']:
        raise DataError('Frozen T_core and V_select selections must be nonempty')
    if not isinstance(alignment_by_scene, dict) or not set(requested).issubset(alignment_by_scene):
        raise DataError('Explicit reference alignment must cover every requested scene before reference access')
    policies = {}
    for sid in requested:
        item = alignment_by_scene[sid]
        if not isinstance(item, dict) or set(item) != {'time_offset_seconds', 'max_gap_seconds'}:
            raise DataError(f'Scene {sid}: provide explicit time_offset_seconds and max_gap_seconds')
        if any(isinstance(v, (bool, np.bool_)) or not isinstance(v, Real) or not np.isfinite(v)
               for v in item.values()) or item['max_gap_seconds'] < 0:
            raise DataError(f'Scene {sid}: alignment values must be finite; maximum gap nonnegative')
        policies[sid] = {key: float(value) for key, value in item.items()}
    return splits, policies


def load_timed_dataset(archive_path, manifest, *, train_splits, data_config,
                       alignment_by_scene, check=None, motion_observation_policy=None):
    """Return train/validation scene lists and a JSON-compatible preparation report.

    Every label-derived A-B identity is a target, including targets whose nodes
    were quarantined. An empty scene's query grid uses its explicit label time
    extent; absent or zero-duration bounds fail. Nonempty scenes use native
    observation support. Clock mapping is mandatory and never estimated from
    reference values. Optional check bounds preparation between scene reads.
    """
    started = time.monotonic()
    if motion_observation_policy is not None:
        validate_motion_observation_policy(motion_observation_policy)
    splits, policies = _request(manifest, train_splits, data_config, alignment_by_scene)
    input_config = {key: data_config[key] for key in ('length', 'max_observations', 'time_unit_seconds')}
    longitude_policy = data_config.get('reference_longitude_policy', 'linear')
    scenes, populations = {}, {}
    if check:
        check()
    with MTADArchive(archive_path) as archive:
        for split in splits:
            populations[split] = []
            native_scenes = iter_native_split_from_archive(archive, manifest, split,
                max_points=data_config['max_observations'], check=check)
            for native in native_scenes:
                if check:
                    check()
                sid = native['scene_id']
                labels = archive.read_labels(sid)
                target_ids = list(dict.fromkeys('-'.join(row['mmsi'].split('-')[:2]) for row in labels))
                query = None
                if not native['tracks']:
                    if not labels:
                        raise DataError(f'Scene {sid}: no valid nodes or label time range; query grid cannot be invented')
                    start, stop = min(row['t_s'] for row in labels), max(row['t_e'] for row in labels)
                    if stop <= start:
                        raise DataError(f'Scene {sid}: empty scene label range must have positive duration')
                    query = np.linspace(start, stop, data_config['length'])
                prepared = prepare_timed_scene(native, **input_config, query_times=query)
                if motion_observation_policy is not None:
                    # Compute offline candidate evidence while this scene's
                    # dense tracks are in memory, then retain only N-by-Q arrays.
                    prepared['motion'] = scene_motion_evidence(prepared,
                        motion_observation_policy=motion_observation_policy)
                    prepared['motion_observation_policy'] = motion_observation_policy
                # The capped native arrays are sufficient for training. Keep
                # canonical point/selection diagnostics, not duplicate raw tracks.
                prepared.pop('tracks', None)
                prepared['target_identity_ids'] = target_ids
                prepared['target_population_policy'] = 'all A-B identities in original scene labels before quarantine'
                prepared['reference_alignment'] = {
                    **policies[sid], 'policy': 'aligned_reference_time = raw_reference_time - time_offset_seconds',
                    'query_grid_policy': prepared['timed_input_audit']['grid_policy'],
                    'clock_mapping_source': 'explicit caller per-scene alignment',
                }
                # Absence meant linear in old checkpoints. Preserve those exact
                # data contracts so evaluation-only and resume remain usable.
                if longitude_policy != 'linear':
                    prepared['reference_alignment']['longitude_policy'] = longitude_policy
                populations[split].append(prepared)
                scenes[sid] = prepared
                # Timed arrays and audits own everything retained for training.
                # Release this scene's full canonical tracks before loading next.
                del native, labels
                archive._label_cache.pop(sid, None)
            del native_scenes
        archive_members = list(archive.members_read)
    references_read = []
    for split in splits:
        scene_ids = manifest['splits'][split]
        if not scene_ids:
            continue
        if check:
            check()
        with closing(iter_reference_states(archive_path, manifest, split=split, scene_ids=scene_ids)) as stream:
            for sid in scene_ids:
                # Check before advancing the generator, not after it has read
                # another potentially large reference CSV.
                if check:
                    check()
                reference = next(stream)
                if reference['scene_id'] != sid:
                    raise DataError('Reference iterator changed the requested scene order')
                scene = scenes[sid]
                scene['fusion_targets'] = build_scene_fusion_targets(scene, reference,
                    target_identity_ids=scene['target_identity_ids'], longitude_policy=longitude_policy,
                    **policies[sid])
                references_read.append({'scene_id': sid, 'split': split, 'member': reference['member'],
                                        'raw_row_count': len(reference['rows']), 'issue_count': len(reference['issues'])})
                # Do not retain raw reference rows in the scene, report or closure.
                del reference
    if check:
        check()
    training = [scene for split in train_splits for scene in populations[split]]
    validation = populations['V_select']
    scene_reports = []
    for scene in [*training, *validation]:
        targets = scene['fusion_targets']
        counts = np.zeros(4, dtype=np.int64)
        for item in targets.values():
            counts += item['valid'].sum(axis=0)
        scene_reports.append(dict(scene_id=scene['scene_id'], split=scene['split'],
            component_id=scene['component_id'], node_count=len(scene['native_values']),
            native_point_count=int(scene['native_observed'].sum()), candidate_count=len(scene['pairs']),
            positive_count=int(scene['labels'].sum()), target_count=len(targets),
            target_without_input_count=len(set(targets) - set(scene['identities'])),
            target_without_reference_count=sum(item['report']['missing_identity'] for item in targets.values()),
            reference_valid_points_per_feature=counts.tolist(), reference_alignment=scene['reference_alignment'],
            input_audit=scene['timed_input_audit'], quarantine_audit=scene['audit']))
        if scene.get('split_unit') == 'scene':
            scene_reports[-1]['split_unit'] = 'scene'
        if motion_observation_policy is not None:
            scene_reports[-1]['motion_native_point_count'] = int(scene['motion']['native_count'].sum())
    report = dict(scope='TMTAFF-time-adapted-v1', training_splits=list(train_splits), selection_split='V_select',
        training_scene_ids=[s['scene_id'] for s in training], validation_scene_ids=[s['scene_id'] for s in validation],
        data_config=dict(data_config), reference_longitude_policy=longitude_policy,
        scenes=scene_reports, observation_members_read=archive_members,
        reference_members_read=references_read, reference_loading='stream by scene; retain aligned L-by-4 targets only',
        test_opened=False, resampled_with_legacy_scene64=False, elapsed_seconds=time.monotonic()-started)
    if manifest.get('split_unit') == 'scene':
        report['split_unit'] = 'scene'
    if motion_observation_policy is not None:
        report['motion_observation_policy'] = motion_observation_policy
    return dict(train=training, validation=validation, report=report)
