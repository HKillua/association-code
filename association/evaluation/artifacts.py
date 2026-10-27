"""Optional scene-local physical prediction evidence, without pickle or hashes.

An index lists only fully written scene artifacts; it is not a declaration that
the overall evaluation completed. The enclosing run/summary remains authoritative.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from association.utils import runtime


ALIGNED_ARRAYS = ('aligned_prediction', 'aligned_prediction_valid', 'aligned_truth',
                  'aligned_truth_valid', 'aligned_observed')


def load_scene_prediction(path):
    """Load one scene metadata JSON and its NPZ with pickle disabled.

    Returns ``metadata``, ``scene`` and ``methods`` (the original evaluator
    artifact dictionaries). This does not choose a common mask or recompute
    correspondences; consumers can explicitly intersect the saved valid masks.
    """
    path = Path(path)
    metadata = runtime.read_json(path)
    if metadata.get('format') != 'scene_physical_predictions_v1':
        raise ValueError('unsupported prediction artifact format')
    filename = metadata['npz']
    if not isinstance(filename, str) or Path(filename).name != filename:
        raise ValueError('artifact NPZ must be in the scene metadata directory')
    with np.load(path.parent / filename, allow_pickle=False) as archive:
        if set(archive.files) != set(metadata['arrays']):
            raise ValueError('artifact array inventory differs from metadata')
        arrays = {key: archive[key] for key in archive.files}
    scene = {key.removeprefix('scene__'): value for key, value in arrays.items() if key.startswith('scene__')}
    scene.update({key: metadata[key] for key in ('scene_id', 'component_id', 'split', 'input_adapter')})
    if 'physical_units' in metadata:
        scene['physical_units'] = dict(metadata['physical_units'])
    grid = scene['grid']
    if grid.ndim != 1 or not np.isfinite(grid).all() or (np.diff(grid) <= 0).any():
        raise ValueError('artifact grid must be a finite increasing vector')
    methods = {}
    prefixes = set()
    for name, item in metadata['methods'].items():
        prefix = item['prefix']
        if not isinstance(prefix, str) or not prefix.startswith('m') or not prefix.endswith('__') or prefix in prefixes:
            raise ValueError('artifact method prefixes must be distinct')
        prefixes.add(prefix)
        artifact = {key[len(prefix):]: value for key, value in arrays.items() if key.startswith(prefix)}
        artifact.update({key: item[key] for key in ('split', 'grouping_kind', 'correspondence_policy')})
        if not set(ALIGNED_ARRAYS).issubset(artifact):
            raise ValueError('artifact is missing aligned metric arrays')
        shape = artifact['aligned_prediction'].shape
        if len(shape) != 3 or shape[1:] != (len(grid), 4):
            raise ValueError('artifact aligned predictions must be [rows, queries, 4]')
        for key in ALIGNED_ARRAYS[1:]:
            expected = shape[:2] if key == 'aligned_observed' else shape
            if artifact[key].shape != expected:
                raise ValueError('artifact aligned metric shapes disagree')
            if key != 'aligned_truth' and artifact[key].dtype != np.bool_:
                raise ValueError('artifact validity and observed masks must be boolean')
        targets, groups = item['target_count'], item['group_count']
        if (artifact['target_ids'].shape != (targets,)
                or artifact['aligned_target_ids'].shape != (shape[0],)
                or artifact['aligned_group_indices'].shape != (shape[0],)
                or shape[0] < targets
                or not np.array_equal(artifact['aligned_target_ids'][:targets], artifact['target_ids'])
                or artifact['group_states'].shape != (groups, len(grid), 4)
                or artifact['group_valid'].shape != artifact['group_states'].shape
                or artifact['group_valid'].dtype != np.bool_
                or artifact['node_group_indices'].shape != scene['node_ids'].shape):
            raise ValueError('artifact target/group metadata disagrees with arrays')
        support_keys = {'aligned_input_support_kind', 'aligned_nearest_input_seconds',
                        'input_support_region_names', 'input_distance_region_names'}
        if support_keys.intersection(artifact):
            from association.evaluation.scene import INPUT_SUPPORT_REGIONS, INPUT_DISTANCE_REGIONS
            if not support_keys.issubset(artifact):
                raise ValueError('incomplete input-support artifact arrays')
            kinds, nearest = artifact['aligned_input_support_kind'], artifact['aligned_nearest_input_seconds']
            if (kinds.shape != shape[:2] or kinds.dtype.kind not in 'iu'
                    or ((kinds < 0) | (kinds >= len(INPUT_SUPPORT_REGIONS))).any()
                    or nearest.shape != shape[:2] or nearest.dtype.kind != 'f'
                    or np.isnan(nearest).any() or (nearest < 0).any()
                    or artifact['input_support_region_names'].tolist() != list(INPUT_SUPPORT_REGIONS)
                    or artifact['input_distance_region_names'].tolist() != list(INPUT_DISTANCE_REGIONS)):
                raise ValueError('invalid input-support artifact arrays')
        methods[name] = artifact
    return dict(metadata=metadata, scene=scene, methods=methods)


class ScenePredictionWriter:
    """Write one NPZ plus small metadata JSON per scene in a new directory.

    Values/masks are the evaluator's actual aligned arrays, not recomputed
    predictions. Numeric file names and method prefixes avoid using scene or
    source IDs as paths. Files from an earlier invocation are never overwritten.
    """

    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.entries = []
        self._scene_ids = set()
        self._indices = set()
        self._write_index()

    def _write_index(self):
        runtime.write_json(self.directory / 'index.json', dict(
            format='scene_physical_predictions_v1', scenes=self.entries,
            completion_scope='listed scenes only; consult enclosing run and summary for evaluation completion',
            row_semantics='all declared targets first; duplicate predictions appended without assigned truth',
            test_opened=False))

    def __call__(self, scene_index, scene, method_artifacts):
        if isinstance(scene_index, bool) or not isinstance(scene_index, int) or scene_index < 0:
            raise ValueError('scene_index must be a nonnegative integer')
        scene_id = str(scene['scene_id'])
        if scene_index in self._indices or scene_id in self._scene_ids:
            raise FileExistsError('scene prediction artifact already written')
        if scene.get('split') not in ('T_core', 'T_head', 'V_select'):
            raise ValueError('prediction artifacts require an explicit development split; T_test is sealed')
        if not isinstance(method_artifacts, dict) or not method_artifacts:
            raise ValueError('at least one method artifact is required')
        arrays = {}

        def add(key, value, *, text=False):
            value = np.asarray(value, dtype=str if text else None)
            if value.dtype.hasobject:
                raise ValueError(f'object/pickle arrays are forbidden: {key}')
            arrays[key] = value

        for key in ('grid', 'values', 'observed', 'pairs', 'labels', 'intervals'):
            add('scene__' + key, scene[key])
        for key in ('node_ids', 'identities', 'sources'):
            add('scene__' + key, scene[key], text=True)
        # Native observations make saved case plots distinguish input points
        # from support interpolation, without reading the archive again.
        for key in ('observation_times', 'native_values', 'native_times', 'native_observed'):
            if key in scene:
                add('scene__' + key, scene[key])
        methods = {}
        for index, (name, artifact) in enumerate(method_artifacts.items()):
            if not isinstance(name, str) or not name or not isinstance(artifact, dict):
                raise ValueError('methods must map nonempty names to evaluator artifacts')
            prefix = f'm{index:03d}__'
            metadata = {key: artifact[key] for key in ('split', 'grouping_kind', 'correspondence_policy')}
            if metadata['split'] != scene['split']:
                raise ValueError('artifact split differs from its scene')
            for key, value in artifact.items():
                if key not in metadata:
                    add(prefix + key, value)
            methods[name] = dict(prefix=prefix, **metadata,
                target_count=len(artifact['target_ids']),
                group_count=len(artifact['group_anchor_target_indices']))
        stem = f'{scene_index:06d}'
        entry = dict(scene_index=scene_index, scene_id=scene_id,
                     npz=stem + '.npz', metadata=stem + '.json')
        metadata = dict(format='scene_physical_predictions_v1', **entry,
            split=scene['split'], component_id=str(scene['component_id']),
            input_adapter=scene.get('input_adapter', 'registered_v1'),
            arrays=list(arrays), methods=methods, test_opened=False)
        if 'physical_units' in scene:
            metadata['physical_units'] = dict(scene['physical_units'])
        # Validate small metadata before claiming files. An interrupted NPZ may
        # remain on disk, but is never advertised in index.json or overwritten.
        payload = json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
        with (self.directory / entry['npz']).open('xb') as stream:
            np.savez_compressed(stream, **arrays)
        with (self.directory / entry['metadata']).open('x', encoding='utf-8') as stream:
            stream.write(payload)
        self.entries.append(entry)
        self._scene_ids.add(scene_id)
        self._indices.add(scene_index)
        self._write_index()
        return entry.copy()
