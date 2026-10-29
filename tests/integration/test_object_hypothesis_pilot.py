"""129 finite hypotheses/T fitting/output binding; fixtures are software only."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from tests.integration.test_query_contribution_control import cache_fixture
from tests.integration.test_source_preference_control import saved_scene
from tools.analysis import source_preference_control as prior
from tools.analysis import object_hypothesis_pilot as p


def native(nodes, sources, intervals, positions=None):
    if positions is None:
        positions = [np.array([[0., .001*i], [0., .001*i+.0001]]) for i in range(len(nodes))]
    events = np.concatenate([np.column_stack((v, np.ones((len(v), 2)))) for v in positions])
    count = len(events)
    return dict(node_ids=np.asarray(nodes), sources=np.asarray(sources), query_seconds=np.arange(64)*20.,
        event_offsets=np.arange(0, count+1, 2, dtype=np.int64),
        event_times_seconds=np.asarray(intervals, float).reshape(-1), event_values=events,
        event_valid=np.ones((count, 4), bool), event_node_ids=np.repeat(np.asarray(nodes), 2),
        event_canonical_indices=np.tile(np.arange(2), len(nodes)))


class ObjectHypothesisTests(unittest.TestCase):
    def test_native_seconds_dates_line_and_top2_preserve_full_temporal_population(self):
        a = native(['a', 'b', 'c', 'd'], ['x', 'x', 'x', 'y'], [[0., 1.], [2., 3.], [.5, 1.5], [4., 5.]],
            [np.array([[0., 179.999], [0., -179.999]]), np.array([[0., -179.997], [0., -179.995]]),
             np.array([[1., 1.], [1., 1.001]]), np.array([[0., 0.], [0., .001]])])
        f = p.native_features(a)
        self.assertIn([0, 1], f['candidate_edges'].tolist())
        index = f['candidate_edges'].tolist().index([0, 1])
        self.assertLess(f['candidate_distance_m'][index], 1e-5)
        self.assertTrue(f['native_overlap_conflict'][0, 2])
        self.assertFalse(f['native_overlap_conflict'][0, 3])
        self.assertTrue(np.all(a['sources'][f['candidate_edges'][:, 0]] == a['sources'][f['candidate_edges'][:, 1]]))
        self.assertEqual(len(f['singleton_node_indices']), 4)
        changed = deepcopy(a)
        changed['event_node_ids'][2] = 'a'; changed['event_canonical_indices'][2] = 0
        with self.assertRaisesRegex(ValueError, 'same original event ID'):
            p.native_features(changed)

    def test_T_threshold_ties_and_no_positive_use_declared_all_reject(self):
        model, curve = p.fit_rejection([1., 2., 3., 4.], [True, True, False, False], [True]*4)
        self.assertEqual((model['mode'], model['threshold_metres']), ('threshold', 2.))
        self.assertEqual(model['balanced_accuracy'], 1.)
        for labels in ([False]*4, [True]*4):
            model, _ = p.fit_rejection([1., 2., 3., 4.], labels, [True]*4)
            self.assertEqual(model['mode'], 'reject_all')
            self.assertIsNone(model['threshold_metres'])
            json.dumps(model, allow_nan=False)
        tied, _ = p.fit_rejection([1., 1.], [True, False], [True, True])
        self.assertEqual(tied['mode'], 'reject_all')

    def test_union_is_truth_blind_and_preserves_original_conflicts_and_all_nodes(self):
        features = dict(candidate_edges=np.array([[0, 1], [1, 2], [2, 3]]),
            candidate_distance_m=np.array([1., 2., 3.]), candidate_gap_seconds=np.ones(3),
            native_overlap_conflict=np.zeros((4, 4), bool), repeated_event_conflict=np.zeros((4, 4), bool))
        features['native_overlap_conflict'][0, 2] = features['native_overlap_conflict'][2, 0] = True
        model = dict(mode='threshold', threshold_metres=1.5)
        result, audit, trace = p.merge_partition(np.arange(4), features, model, reject=False)
        np.testing.assert_array_equal(result, [0, 0, 1, 1])
        self.assertEqual((audit['accepted_merge_count'], audit['working_conflict_rejected_count']), (2, 1))
        rejected, audit, _ = p.merge_partition(np.arange(4), features, model, reject=True)
        np.testing.assert_array_equal(rejected, [0, 0, 1, 2])
        self.assertEqual(audit['threshold_rejected_count'], 2)
        features['native_overlap_conflict'][0, 1] = features['native_overlap_conflict'][1, 0] = True
        result, audit, _ = p.merge_partition(np.array([0, 0, 1, 2]), features, model, reject=False)
        self.assertEqual(audit['initial_internal_working_conflict_pairs'], 1)
        self.assertEqual(audit['final_internal_working_conflict_pairs'], 1)
        self.assertEqual(result[0], result[1])

    def test_six_outputs_replay_original_WS_mean_and_LOW_exact(self):
        fragment, membership, retained, calibration, public = saved_scene()
        row, _, original, _ = prior.analyze_scene(fragment, membership, retained, calibration, public)
        membership_dict = {name: membership for name in p.PARTITIONS}
        outputs, _ = p.construct_predictions(fragment, membership_dict)
        np.testing.assert_array_equal(outputs['WS_MEAN'], original['WS_MEAN']['group_states'])
        np.testing.assert_array_equal(outputs['WS_LOW_SOURCE_SUPPORTED'], original['WS_LOW_SOURCE_SUPPORTED']['group_states'])
        self.assertEqual(set(outputs), set(p.METHODS))
        for name in p.METHODS:
            np.testing.assert_array_equal(outputs[name][..., 2:], outputs['WS_MEAN'][..., 2:])

    def test_cached_runner_replays_controls_and_preserves_post_prediction_failure(self):
        for fail in (False, True):
            with self.subTest(fail=fail), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary).resolve()
                old_args, meta = cache_fixture(root)
                control, native_run = root/p.CONTROL_RUN, root/'native127'
                (control/'private').mkdir(parents=True)
                training_ids = ['t1', 't2']
                for split, ids in [('T_core', training_ids), ('V_select', meta['ids'])]:
                    for kind in ('native-events', 'native-evaluation'):
                        (native_run/'private'/kind/split).mkdir(parents=True)
                    for sid in ids:
                        source_sid = sid if split == 'V_select' else '1'
                        with np.load(old_args.source_system/'predictions'/f'states-{source_sid}.npz', allow_pickle=False) as saved:
                            physical = {key: saved[key].copy() for key in p.PHYSICAL_KEYS}
                            identity = saved['identities'].copy()
                        data = native(physical['node_ids'], physical['sources'], [[0., 1.]]*len(identity))
                        data['query_seconds'] = physical['query_seconds']
                        np.savez_compressed(native_run/'private/native-events'/split/f'scene-{sid}.npz', **data)
                        if fail and split == 'V_select' and sid == '2':
                            identity[0] = 'corrupt_identity'
                        np.savez_compressed(native_run/'private/native-evaluation'/split/f'scene-{sid}.npz',
                            node_ids=physical['node_ids'], identities=identity)
                        if split == 'V_select':
                            f, membership, retained, calibration, public = saved_scene(sid)
                            _, _, artifacts, _ = prior.analyze_scene(f, membership, retained, calibration, public)
                            payload = {name+'_'+key: value for name in ('WS_MEAN', 'WS_LOW_SOURCE_SUPPORTED')
                                       for key, value in artifacts[name].items()}
                            np.savez_compressed(control/'private'/f'states-{sid}.npz', **payload)
                p.old.frozen.write(native_run/'native-cache.json', dict(status='CACHE_COMPLETED',
                    scene_ids=dict(T_core=training_ids, V_select=meta['ids'])))
                run = dict(status='COMPLETED', exit_code=0, idea_id=prior.IDEA, plan_version='1.0',
                    scene_ids=meta['ids'], source_system=str(old_args.source_system), partition_run=str(old_args.partition_run))
                p.old.frozen.write(control/'run.json', run); p.old.frozen.write(control/'summary.json', run)
                manifest = root/'manifest129.json'; p.old.frozen.write(manifest, dict(splits=dict(T_core=training_ids)))
                args = SimpleNamespace(source_system=old_args.source_system, partition_run=old_args.partition_run,
                    control_run=control, native_run=native_run, manifest=manifest, output=root/'object-out', cpu_threads=1)
                with patch.object(p.old, 'metadata', return_value=meta):
                    if fail:
                        with self.assertRaisesRegex(ValueError, 'native/056 identities'):
                            p.run(args)
                        state = p.old.frozen.read(args.output/'run.json')
                        self.assertEqual((state['status'], state['completed_scenes']), ('FAILED', 1))
                        self.assertTrue((args.output/'private/predictions-2.npz').exists())
                        self.assertFalse((args.output/'summary.json').exists())
                    else:
                        result = p.run(args)
                        self.assertEqual(result['scene_ids'], ['1', '2'])
                        self.assertEqual(len(result['pairwise']), 15)
                        self.assertEqual(len(result['methods']), 6)
                        self.assertEqual(result['rejection_model']['mode'], 'reject_all')
                        self.assertTrue(result['original121_two_artifacts_exact'])
                with self.assertRaises(FileExistsError):
                    p.run(args)


if __name__ == '__main__':
    unittest.main()
