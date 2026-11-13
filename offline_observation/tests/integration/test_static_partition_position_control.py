"""115 software checks; tiny synthetic caches are not MTAD evidence."""
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from tools.analysis import static_partition_position_control as p


def fixture(sid='1'):
    n, q = 3, 64
    pairs = np.array([[0, 1], [0, 2], [1, 2]], np.int64)
    identities = np.array(['t0', 't0', 't1'])
    targets = np.array(['t0', 't1', 'no-input'])
    states = np.zeros((n, q, 4)); states[:, :, 0] = np.array([0., .002, 1.])[:, None]
    states[..., 2:] = [5., 10.]
    support = np.ones((n, q), bool); support[:, 0] = False
    nearest = np.ones((n, q)); nearest[:, 0] = 100.
    truth = np.zeros((3, q, 4)); truth[..., 0] = np.array([.001, 1., 2.])[:, None]
    valid = np.zeros(truth.shape, bool); valid[..., :2] = True
    record = dict(scene_id=sid, component_id='c', node_ids=np.array(['n0', 'n1', 'n2']),
        sources=np.array(['a', 'b', 'b']), pairs=pairs,
        scores=p.frozen.THRESHOLD+np.array([3., -1., 2.]), labels=np.array([1, 0, 0], bool),
        joint_support_queries=np.full(3, 63, np.int64))
    memberships = dict(A0=np.array([0, 0, 1]), WS=np.array([0, 0, 0]),
                       CL=np.array([0, 0, 1]), CC=np.array([0, 0, 0]))
    fragment = dict(node_ids=record['node_ids'].copy(), sources=record['sources'].copy(), identities=identities,
        target_identity_ids=targets, native_states=states, support=support, nearest_seconds=nearest,
        exact_observation=np.zeros((n, q), bool), query_seconds=np.arange(q, dtype=float),
        target=truth[[0, 0, 1]], target_valid=valid[[0, 0, 1]],
        learned_assoc_membership=memberships['A0'], learned_assoc_group_offsets=np.array([0, 2, 3]),
        learned_assoc_group_node_indices=np.array([0, 1, 2]))
    groups = [[0, 1], [2]]
    mean = p.ordinary_fusion(states, nearest, support, groups)['predictions']['mean_supported']
    fragment[p.BASE] = mean
    report = p.evaluate_full_scene([dict(node_indices=g, states=mean[g[0]]) for g in groups],
        node_identities=identities, target_identities=targets, target_states=truth, target_valid=valid,
        node_observed=support, split='V_select', grouping_kind='predicted',
        correspondence_policy='anchor_identity_first', return_artifact=True,
        node_input_support=dict(support=support, nearest_seconds=nearest))
    artifact = report.pop('artifact')
    saved = {p.BASE+'_'+key: value for key, value in artifact.items()}
    saved.update(ols_anchor_target_ids=artifact['target_ids'],
        ols_anchor_aligned_truth=artifact['aligned_truth'].copy(),
        ols_anchor_aligned_truth_valid=artifact['aligned_truth_valid'].copy())
    public = dict(scene_id=sid, component_id='c', raw=p.frozen.evaluate_scenes([record], p.frozen.THRESHOLD),
        raw_invariance_verified=True, original_A0_membership_verified=True, arms={})
    for name, membership in memberships.items():
        co = membership[pairs[:, 0]] == membership[pairs[:, 1]]
        public['arms'][name] = dict(decoded=p.frozen.evaluate_scenes([dict(record, scores=co.astype(float))], .5))
    expected = dict(population={key: report[key] for key in p.POPULATION_KEYS}, full_target_metrics=report['full_target_metrics'])
    return record, fragment, saved, memberships, public, expected


class PositionControlTests(unittest.TestCase):
    def test_new_groups_recompute_mean_and_keep_full_targets_and_pairwise_intersections(self):
        args = fixture()
        row, private, arrays = p.analyze_scene(*args)
        self.assertEqual(row['methods']['A0']['position']['all']['intersection_count'], 128)
        self.assertEqual(row['methods']['A0']['position']['all']['available_target_count'], 192)
        self.assertEqual(row['methods']['WS']['position']['all']['intersection_count'], 64)
        self.assertEqual(row['pairwise']['WS']['coverage_transitions']['lost_target_points'], 64)
        self.assertEqual(row['pairwise']['WS']['coverage_transitions']['lost_target_count'], 1)
        # CC's smaller target set must not shrink the separate A0/CL comparison.
        self.assertEqual(row['pairwise']['CL']['common_valid_position']['A0']['intersection_count'], 128)
        self.assertEqual(row['pairwise']['CC']['common_valid_position']['A0']['intersection_count'], 64)
        self.assertAlmostEqual(private['artifacts']['WS']['group_states'][0, 1, 0], (0.+.002+1.)/3)
        self.assertNotEqual(private['artifacts']['WS']['group_states'][0, 1, 0], args[1][p.BASE][0, 1, 0])
        # Speed/heading reference invalidity never removes valid lat/lon points.
        self.assertFalse(args[2][p.BASE+'_aligned_truth_valid'][..., 2:].any())
        self.assertEqual(len(arrays['methods']['A0']['all']), 128)

    def test_prediction_uses_only_saved_physical_inputs_and_preserves_both_fallbacks(self):
        _, fragment, _, membership, _, _ = fixture()
        fragment['native_states'][1, :, 1] = 180.
        groups, before = p.construct_states(fragment, membership)
        self.assertEqual(before['A0'][0, 0, 0], fragment['native_states'][0, 0, 0])
        self.assertEqual(before['A0'][0, 1, 1], fragment['native_states'][0, 1, 1])
        changed = dict(fragment, identities=np.array(['unknown']*3), target=np.full_like(fragment['target'], 99.))
        _, after = p.construct_states(changed, membership)
        for name in p.METHODS:
            np.testing.assert_array_equal(before[name], after[name])
        self.assertEqual(groups['A0'], [[0, 1], [2]])
        with self.assertRaisesRegex(ValueError, 'canonical'):
            p.groups_from_membership(np.array([1, 1, 0]), 3)

    def test_exact_A0_and_identity_registry_bindings_fail_instead_of_relaxing_tolerance(self):
        for corrupt in ('node', 'source', 'prediction', 'reference', 'offset', 'label'):
            args = list(fixture())
            if corrupt == 'node': args[1]['node_ids'][0] = 'xx'
            elif corrupt == 'source': args[1]['sources'][0] = 'z'
            elif corrupt == 'prediction': args[1][p.BASE][0, 1, 0] += 1e-12
            elif corrupt == 'reference': args[1]['target'][0, 1, 0] += 1e-12
            elif corrupt == 'offset': args[1]['learned_assoc_group_offsets'][1] = 1
            else: args[0]['labels'][1] = True
            with self.subTest(corrupt=corrupt), self.assertRaises(ValueError):
                p.analyze_scene(*args)

    def test_pooled_rmse_uses_points_and_null_intersections_remain_null(self):
        def metrics(values):
            n = len(values)
            counts = dict(available_target_count=n+1, available_prediction_count=n, intersection_count=n,
                missing_prediction_count=1, prediction_without_target_count=0,
                available_target_trajectory_count=1, covered_target_trajectory_count=int(n>0),
                missing_target_trajectory_count=int(n==0))
            return dict(counts, **{k:v for k,v in p.stats(values).items() if k != 'point_count'})
        first, second = np.array([3.]), np.array([4., 5., 12.])
        pooled = p.pooled_position([metrics(first), metrics(second)], [first, second])
        self.assertAlmostEqual(pooled['rmse'], np.sqrt(194./4))
        self.assertEqual(pooled['intersection_count'], 4)
        empty = p.pooled_position([metrics(np.array([]))], [np.array([])])
        self.assertIsNone(empty['mae']); self.assertIsNone(empty['p99'])
        delta = p.delta_stats(np.array([-3., 1., 0.]))
        self.assertAlmostEqual(delta['mean_error_change'], -2/3)
        self.assertEqual((delta['better_point_count'], delta['worse_point_count'], delta['tied_point_count']), (1, 1, 1))

    def cache_fixture(self, root):
        source = root/'original056'/'system'; source.mkdir(parents=True)
        association = source.parent/'association'; association.mkdir()
        previous = root/'original114'; previous.mkdir()
        for base in (source, previous):
            (base/'scenes').mkdir()
        (source/'predictions').mkdir(); (source/'system-predictions').mkdir(); (previous/'private').mkdir()
        ids = ['1', '2']; payload = {'scene_ids': np.array(ids)}
        expected, original, raw = {}, {}, {scope: [] for scope in p.frozen.SCOPES}
        for sid in ids:
            record, fragment, saved, memberships, public, exp = fixture(sid)
            payload.update({sid+'_'+key: record[key] for key in p.frozen.FIELDS})
            np.savez_compressed(source/'predictions'/f'states-{sid}.npz', **fragment)
            np.savez_compressed(source/'system-predictions'/f'states-{sid}.npz', **saved)
            p.frozen.write(previous/'scenes'/f'scene-{sid}.json.gz', public)
            p.frozen.write(previous/'private'/f'scene-{sid}.json.gz', dict(memberships={k:v.tolist() for k,v in memberships.items()}))
            expected[sid] = {p.BASE: exp}; original[sid] = memberships['A0']
            for scope, row in p.frozen.metric_rows(public['raw']).items(): raw[scope].append(row)
        np.savez_compressed(association/'final-scores.npz', **payload)
        p.frozen.write(root/'manifest.json', {})
        meta = dict(ids=ids, all_ids=ids, components={sid:'c' for sid in ids}, physical=dict(expected=expected),
                    original=original, expected_raw=raw, association=association)
        return source, previous, meta

    def test_original114_pilot_metadata_cannot_be_reselected_or_rebound(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); source = root/'original056/system'; source.mkdir(parents=True)
            association = source.parent/'association'; association.mkdir()
            previous = root/p.PARTITION_RUN; previous.mkdir()
            ids = [str(i) for i in range(300)]
            chosen = p.partition.pilot_selection(ids)
            run = dict(status='COMPLETED', idea_id='IDEA-STATIC-PARTITION-PILOT-20261029', plan_version='1.0',
                threshold=p.frozen.THRESHOLD, seed=p.partition.SEED, methods=list(p.partition.METHODS),
                pilot_scene_ids=chosen, source_association=str(association), model_forwards=0, models_constructed=0,
                optimizer_updates=0, **{k:False for k in ('gpu_used', 'scores_recomputed', 'threshold_reselected',
                    'normalizer_refitted', 'native_read', 'reference_read', 'test_opened')})
            summary = dict(status='COMPLETED', plan_version='1.0', expanded=True, completed_scene_ids=ids,
                analyzed_bindings_scene_count=300, pilot=dict(scene_ids=chosen), expansion_gate=dict(expand=True),
                source_cache=dict(path=str(association/'final-scores.npz')))
            p.frozen.write(previous/'run.json', run)
            p.frozen.write(previous/'summary.json', summary)
            p.frozen.write(previous/'pilot-summary.json', dict(pilot=summary['pilot'], expansion_gate=summary['expansion_gate']))
            with patch.object(p, '_bind_source'), patch.object(p, '_load_metadata', return_value={}), \
                 patch.object(p.frozen, 'metadata', return_value=(ids, {}, {}, {})):
                actual = p.metadata(source, previous, {})
                self.assertEqual(actual['ids'], chosen)
                for corrupt in ('source', 'pilot'):
                    changed = dict(run)
                    if corrupt == 'source': changed['source_association'] = str(root/'other/association')
                    else: changed['pilot_scene_ids'] = list(reversed(chosen))
                    p.frozen.write(previous/'run.json', changed)
                    with self.subTest(corrupt=corrupt), self.assertRaisesRegex(ValueError, 'original114'):
                        p.metadata(source, previous, {})

    def test_fixed_cached_flow_no_expansion_private_arrays_and_failure_preservation(self):
        for fail in (False, True):
            with self.subTest(fail=fail), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary); source, previous, meta = self.cache_fixture(root)
                if fail:
                    path = source/'predictions/states-1.npz'
                    with np.load(path, allow_pickle=False) as saved:
                        bad = {key: saved[key] for key in saved.files}
                    bad[p.BASE][0, 1, 0] += 1e-12
                    np.savez_compressed(path, **bad)
                args = SimpleNamespace(source_system=source, partition_run=previous, output=root/'out',
                                       manifest=root/'manifest.json', cpu_threads=1)
                with patch.object(p, 'metadata', return_value=meta):
                    if fail:
                        with self.assertRaisesRegex(ValueError, 'A0 exact ordinary'):
                            p.run(args)
                        state = p.frozen.read(root/'out/run.json')
                        self.assertEqual(state['status'], 'FAILED')
                        self.assertEqual(state['completed_scenes'], 0)
                        self.assertFalse((root/'out/summary.json').exists())
                    else:
                        result = p.run(args)
                        self.assertTrue(result['no_expansion']); self.assertEqual(result['scene_count'], 2)
                        self.assertEqual(set(result['methods']), set(p.METHODS))
                        self.assertEqual(result['model_forwards'], 0)
                        self.assertTrue(result['cached_reference_read'])
                        self.assertEqual(result['pairwise']['CL']['common_valid_position']['A0']['intersection_count'], 256)
                        with np.load(root/'out/private/states-1.npz', allow_pickle=False) as private:
                            self.assertIn('WS_group_states', private.files)
                        with self.assertRaises(FileExistsError): p.run(args)
