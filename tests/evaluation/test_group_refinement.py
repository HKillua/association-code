"""055 contracts: label-free clique refinement and unchanged physical population."""
from copy import deepcopy
import inspect
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import numpy as np

from association.evaluation.fusion_controls import metrics, ordinary_fusion, rms_haversine_scores
from association.evaluation.association import evaluate_scenes
from association.evaluation.grouping import group_predicted_tracks
from association.evaluation.group_refinement import reconstruct_graph, refine_groups
from association.evaluation.scene import evaluate_full_scene
from association.evaluation.system_attribution import ANCHOR, BASE, INDEPENDENT
from association.evaluation.refinement_diagnostics import (
    NEW_ANCHOR, NEW_MEAN, METHODS, analyze_refinement, aggregate_refinement,
)
from association.evaluation.tmtaff import aggregate_fusion_reports
from association.utils.runtime import read_json, write_json
from tools.experiments.level1_system_evaluate import _pool_fragment


class GroupRefinementTests(unittest.TestCase):
    @staticmethod
    def fixture():
        """Two anchor stars: a mixed star and a pure star; a singleton remains.

        Splitting recovers c while adding a duplicate b. New groups from the
        first parent must be globally interleaved with the second parent's.
        """
        q, n = 64, 8
        identities = np.asarray(['a', 'b', 'c', 'b', 'c', 'b', 'c', 'd'])
        targets = np.asarray(['a', 'b', 'c', 'd', 'reference-unavailable'])
        native = np.zeros((n, q, 4), np.float64)
        native[:, :, 0] = (30. + np.asarray([0., .1, .005, .105, -.005, .095, -.006, .2]))[:, None]
        native[:, :, 1] = 120. + np.linspace(0., .001, q)
        native[:, :, 2] = np.asarray([5., 12., 6., 8., 9., 11., 7., 4.])[:, None]
        native[:, :, 3] = np.asarray([359., 90., 1., 95., 2., 85., 358., 180.])[:, None]
        query = np.arange(q, dtype=np.float64) * 60.
        support = np.ones((n, q), bool)
        support[:, 48:] = False  # No helper can contribute here.
        support[2, 16:32] = False
        nearest = np.broadcast_to(np.minimum(query, 3780.-query), (n, q)).copy()
        exact = np.zeros_like(support)
        exact[:, 0] = True
        pairs = np.column_stack(np.triu_indices(n, 1))
        scores = -np.log1p(-rms_haversine_scores(native[..., :2], pairs) / 1000.)
        threshold = -np.log1p(800. / 1000.)
        groups = group_predicted_tracks(n, pairs, scores, threshold=threshold,
            grouping_policy='anchor_direct', node_order='input')
        assert groups == [[0, 2, 4, 6], [1, 3, 5], [7]]
        refined = [[0, 2], [1, 3], [4, 6], [5], [7]]
        truth = np.array([native[0], native[1], native[4], native[7], native[0]])
        truth[0, :, 0] += .001
        truth[1, :, 2] = 9.
        truth[2, :, 3] = 0.
        truth[4] = np.nan
        valid = np.ones_like(truth, bool)
        valid[4] = False
        valid[1, :8, 2] = False  # Attribute masks must not be conflated.
        valid[2, -4:, 3] = False
        truth_indices = np.asarray([0, 1, 2, 1, 2, 1, 2, 3])
        ordinary = ordinary_fusion(native, nearest, support, groups)
        baseline = ordinary['predictions']['mean_supported']
        fragment = dict(node_ids=np.asarray([f'node-{i}' for i in range(n)]),
            identities=identities, sources=np.asarray(['1', '2', '1', '2', '3', '3', '2', '1']),
            target_identity_ids=targets, query_seconds=query,
            target=truth[truth_indices], target_valid=valid[truth_indices],
            native_states=native, support=support, nearest_seconds=nearest,
            exact_observation=exact, ols_mean_supported=baseline,
            ols_membership=ordinary['membership'], ols_group_offsets=np.asarray([0, 4, 7, 8]),
            ols_group_node_indices=np.asarray([0, 2, 4, 6, 1, 3, 5, 7]))
        regions = dict(all=np.ones_like(support), inside_support=support,
                       outside_support=~support, far_gt_1800s=nearest > 1800.)
        fragment.update({'region_'+name: value for name, value in regions.items()})
        scene = evaluate_full_scene([
            dict(node_indices=g, states=baseline[g[0]], prediction_valid=np.isfinite(baseline[g[0]]))
            for g in groups], node_identities=identities, target_identities=targets,
            target_states=truth, target_valid=valid, node_observed=support,
            split='V_select', grouping_kind='predicted', correspondence_policy='anchor_identity_first',
            return_artifact=True, node_input_support=dict(support=support, nearest_seconds=nearest))
        saved = scene.pop('artifact')
        expected = dict(scene_id='1', component_id='1', unique_target={BASE: scene},
            group_count=dict(ols=len(groups)), fragment={BASE: {
                region: metrics(baseline, fragment['target'], fragment['target_valid'], mask)
                for region, mask in regions.items()}})
        return fragment, saved, expected, threshold, groups, refined

    @classmethod
    def cached_source(cls, root):
        """Create genuine 053 arrays/summary, then run the existing 054 entry."""
        from tools.analysis.system_attribution import run as run054
        fragment, saved, expected, threshold, _, _ = cls.fixture()
        source = root / 'original053'
        (source/'predictions').mkdir(parents=True)
        (source/'system-predictions').mkdir()
        np.savez_compressed(source/'predictions'/'states-1.npz', **fragment)
        np.savez_compressed(source/'system-predictions'/'states-1.npz',
            common_prediction_valid_on_target_rows=saved['aligned_prediction_valid'][:5],
            **{BASE+'_'+key: value for key, value in saved.items()})
        summary = dict(status='COMPLETED', dataset_level='Level1', split='V_select',
            scene_ids=['1'], scene_count=1, fragment_count=8, method_names=[BASE],
            test_opened=False, optimizer_updates=0, per_scene=[expected],
            ols_working_point=dict(threshold=float(threshold), selection_split='V_select',
                motion_observation_policy='canonical_all', validation_scene_ids=['1']),
            fragment=_pool_fragment([expected], [BASE]),
            unique_target={BASE: aggregate_fusion_reports([expected['unique_target'][BASE]])})
        summary['ols_working_point'].update(scope='ols-working-point-v1', dataset_level='Level1',
            synthetic=False, score_semantics='-log1p(rms_haversine_m/1000)',
            training_scene_ids=['training-unread'],
            data=dict(length=64, max_observations=128, time_unit_seconds=20.0))
        ols = source.parent/'ols'
        ols.mkdir()
        pairs, scores, _, _ = reconstruct_graph(fragment, threshold)
        association = evaluate_scenes([dict(scene_id='1', component_id='1', pairs=pairs,
            scores=scores, labels=fragment['identities'][pairs[:, 0]] == fragment['identities'][pairs[:, 1]],
            sources=fragment['sources'])], threshold)
        write_json(ols/'baseline.json', dict(selection=dict(threshold=float(threshold)),
            evaluation=association, scene_ids=['1'], test_opened=False,
            reference_read=False, optimizer_updates=0))
        write_json(ols/'ols-working-point.json', summary['ols_working_point'])
        write_json(source/'summary.json', summary)
        write_json(source/'run.json', dict(status='COMPLETED', test_opened=False,
            optimizer_updates=0, completed_scenes=1, total_scenes=1,
            motion_observation_policy='canonical_all', threshold_reselected=False,
            normalizer_refitted=False))
        manifest = root/'manifest.json'
        write_json(manifest, dict(dataset_level='Level1', split_unit='scene',
            splits=dict(T_core=['training-unread'], V_select=['1'], T_test=['test-unread'])))
        attr = root/'attribution054'
        run054(SimpleNamespace(source_system=source, manifest=manifest, output=attr, cpu_threads=1))
        return SimpleNamespace(source_system=source, source_attribution=attr,
            manifest=manifest, output=root/'refinement055', cpu_threads=1)

    @staticmethod
    def rebuild_saved(fragment, saved):
        """Refresh a physical fixture after explicitly changing inputs or labels."""
        offsets, flat = fragment['ols_group_offsets'], fragment['ols_group_node_indices']
        groups = [flat[a:b].tolist() for a, b in zip(offsets[:-1], offsets[1:])]
        f = ordinary_fusion(fragment['native_states'], fragment['nearest_seconds'], fragment['support'], groups)
        fragment[BASE] = f['predictions']['mean_supported']
        count = len(fragment['target_identity_ids'])
        report = evaluate_full_scene([dict(node_indices=g, states=fragment[BASE][g[0]],
            prediction_valid=np.isfinite(fragment[BASE][g[0]])) for g in groups],
            node_identities=fragment['identities'], target_identities=fragment['target_identity_ids'],
            target_states=saved['aligned_truth'][:count], target_valid=saved['aligned_truth_valid'][:count],
            node_observed=fragment['support'], split='V_select', grouping_kind='predicted',
            correspondence_policy='anchor_identity_first', return_artifact=True,
            node_input_support=dict(support=fragment['support'], nearest_seconds=fragment['nearest_seconds']))
        return report['artifact']

    def test_reconstruction_matches_original_scores_and_partition(self):
        fragment, _, _, threshold, original, expected = self.fixture()
        pairs, scores, groups, refined = reconstruct_graph(fragment, threshold)
        np.testing.assert_array_equal(pairs, np.column_stack(np.triu_indices(8, 1)))
        direct = -np.log1p(-rms_haversine_scores(fragment['native_states'][..., :2], pairs) / 1000.)
        np.testing.assert_array_equal(scores, direct)
        self.assertEqual(groups, original)
        self.assertEqual(refined, expected)

    def test_threshold_ties_are_included_without_epsilon(self):
        pairs = np.asarray([[0, 1], [0, 2], [1, 2]])
        threshold = -1.
        scores = np.asarray([threshold, threshold, np.nextafter(threshold, -np.inf)])
        self.assertEqual(refine_groups(3, [[0, 1, 2]], pairs, scores, threshold=threshold),
                         [[0, 1], [2]])
        scores[-1] = threshold
        self.assertEqual(refine_groups(3, [[0, 1, 2]], pairs, scores, threshold=threshold),
                         [[0, 1, 2]])

    def test_first_compatible_subgroup_and_global_order_are_fixed(self):
        # Node 4 can join either [0,2] or [3]; it must join the first.
        pairs = np.column_stack(np.triu_indices(6, 1))
        rejected = {(2, 3), (2, 5), (3, 5)}
        scores = np.asarray([-2. if tuple(p) in rejected else 0. for p in pairs])
        self.assertEqual(refine_groups(6, [[0, 2, 3, 4, 5], [1]], pairs, scores, threshold=-1.),
                         [[0, 2, 4], [1], [3], [5]])
        # A compatible cross-parent edge is never permission to merge parents.
        self.assertEqual(refine_groups(3, [[0, 2], [1]],
            np.asarray([[0, 1], [0, 2], [1, 2]]), np.zeros(3), threshold=0.), [[0, 2], [1]])

    def test_refinement_accepts_no_identity_or_reference_input(self):
        self.assertEqual(set(inspect.signature(refine_groups).parameters),
                         {'node_count', 'groups', 'pairs', 'scores', 'threshold'})
        fragment, _, _, threshold, _, _ = self.fixture()
        original = reconstruct_graph(fragment, threshold)
        # Reconstruction itself only needs prediction arrays/group records.
        minimal = {key: fragment[key] for key in ('native_states', 'ols_group_offsets',
                    'ols_group_node_indices', 'ols_membership')}
        minimal_result = reconstruct_graph(minimal, threshold)
        for index in (0, 1):
            np.testing.assert_array_equal(original[index], minimal_result[index])
        self.assertEqual(original[2:], minimal_result[2:])

    def test_cached_replay_and_complete_graph_contract_fail_closed(self):
        fragment, _, _, threshold, _, _ = self.fixture()
        for key, replacement in (
            ('native_states', fragment['native_states'][:, :63]),
            ('ols_membership', np.zeros(8, dtype=np.int64)),
            ('ols_group_node_indices', np.arange(8)),
        ):
            changed = dict(fragment, **{key: replacement})
            with self.subTest(key=key), self.assertRaises(ValueError):
                reconstruct_graph(changed, threshold)
        changed = deepcopy(fragment)
        changed['native_states'][2, :, 0] += 1.
        with self.assertRaises(ValueError):
            reconstruct_graph(changed, threshold)
        pairs = np.asarray([[0, 1], [0, 2], [1, 2]])
        for malformed in (pairs[:2], np.asarray([[0, 1], [0, 1], [1, 2]])):
            with self.subTest(pairs=malformed.tolist()), self.assertRaises(ValueError):
                refine_groups(3, [[0, 1, 2]], malformed, np.zeros(len(malformed)), threshold=-1.)

    def test_population_conservation_and_removed_correct_and_wrong_pairs(self):
        fragment, saved, expected, _, old, new = self.fixture()
        report, arrays = analyze_refinement(fragment, saved, old, new, expected)
        self.assertEqual(set(report['methods']), set(METHODS))
        account = report['refinement_accounting']
        self.assertEqual((account['delta_group_count'], account['delta_represented_identity_count'],
                          account['delta_duplicate_group_count']), (2, 1, 1))
        self.assertEqual((account['removed_tp'], account['removed_fp']), (4, 2))
        self.assertEqual(account['attributes']['position']['rescued_target_count'], 1)
        self.assertEqual(account['attributes']['position']['recovered_targets_per_added_group'], .5)
        self.assertEqual(account['attributes']['position']['added_duplicates_per_added_group'], .5)
        for label, counts in (('original', (6, 3, 0, 19)), ('refined', (2, 1, 4, 21)),
                              ('singleton', (0, 0, 6, 22))):
            self.assertEqual(tuple(report['grouping'][label]['co_membership'][k]
                for k in ('tp', 'fp', 'fn', 'tn')), counts)
        self.assertIsNone(report['grouping']['singleton']['co_membership']['precision'])
        self.assertEqual(report['grouping']['original']['capacity']['position']['group_count_deficit'], 1)
        self.assertEqual(report['grouping']['refined']['capacity']['position']['group_count_deficit'], 0)
        for method in METHODS:
            system = report['methods'][method]['system']
            self.assertEqual(system['target_identity_count'], 5)
            self.assertEqual(system['reference_unavailable_target_count'], 1)
            self.assertEqual(system['full_target_metrics']['all']['position']['available_target_count'], 256)
        self.assertEqual(report['methods'][NEW_ANCHOR]['system']['duplicate_prediction_group_count'], 1)
        self.assertEqual(report['methods'][INDEPENDENT]['system']['duplicate_prediction_group_count'], 4)
        np.testing.assert_array_equal(arrays[NEW_ANCHOR+'_selected_anchor_node_index'], [0, 1, 4, 7, -1])

    def test_same_anchor_helper_and_purity_masks_exclude_singleton_dilution(self):
        fragment, saved, expected, _, old, new = self.fixture()
        report, arrays = analyze_refinement(fragment, saved, old, new, expected)
        expected_points = dict(all=192, same_anchor_both_multi=128,
            same_anchor_both_multi_old_helper=96, same_anchor_both_multi_new_helper=80,
            same_anchor_both_multi_both_helpers=80, pure_multi_same_anchor_both_helpers=48)
        for method, regions in report['joint_four_arm']['methods'].items():
            for region, count in expected_points.items():
                with self.subTest(method=method, region=region):
                    self.assertEqual(regions[region]['position']['intersection_count'], count)
            self.assertEqual(regions['all']['speed']['intersection_count'], 184)
        np.testing.assert_array_equal(arrays[NEW_MEAN+'_selected_group_class'], [3, 2, 2, 1, 0])
        np.testing.assert_array_equal(arrays[NEW_MEAN+'_selected_parent_size'], [4, 3, 4, 1, 0])
        self.assertEqual(report['grouping']['refined']['group_class_counts'],
                         dict(singleton=2, pure_multi=2, mixed_multi=1))
        pair = '__vs__'.join(sorted((ANCHOR, NEW_ANCHOR)))
        self.assertEqual(report['pairwise'][pair]['methods'][NEW_ANCHOR]['all']['position']['intersection_count'], 192)
        # The duplicated b output is retained only in the separate all-group view.
        all_groups = report['all_group_fusion']['refined']['anchor']['all']['position']
        unique = report['methods'][NEW_ANCHOR]['system']['full_target_metrics']['all']['position']
        self.assertEqual((all_groups['intersection_count'], unique['intersection_count']), (320, 256))

    def test_no_supported_helpers_keeps_null_evidence_and_singleton_equivalence(self):
        fragment, saved, _, _, old, new = self.fixture()
        fragment['support'][:] = False
        fragment['exact_observation'][:] = False
        saved = self.rebuild_saved(fragment, saved)
        report, arrays = analyze_refinement(fragment, saved, old, new)
        self.assertFalse(arrays['joint_region_same_anchor_both_multi_both_helpers'].any())
        for row in report['joint_four_arm']['methods'].values():
            for attr in ('position', 'speed', 'heading'):
                self.assertEqual(row['same_anchor_both_multi_both_helpers'][attr]['intersection_count'], 0)
                self.assertIsNone(row['same_anchor_both_multi_both_helpers'][attr]['mae'])
        for label in ('original', 'refined', 'singleton'):
            for attr in ('position', 'speed', 'heading'):
                self.assertAlmostEqual(report['all_group_fusion'][label]['fusion_effect']['all'][attr]
                                       ['mean_minus_anchor_mae'], 0.)

    def test_antipodal_heading_fallback_is_not_effective_helper_evidence(self):
        fragment, saved, _, _, old, new = self.fixture()
        fragment['native_states'][0, :, 3] = 0.
        fragment['native_states'][2, :, 3] = 180.
        saved = self.rebuild_saved(fragment, saved)
        report, arrays = analyze_refinement(fragment, saved, old, new)
        # The new [0,2] pair has an exactly opposed heading while its position
        # and speed are still fused. Other helpers prevent the old group's tie.
        self.assertTrue(arrays['joint_region_same_anchor_both_multi_both_helpers'][0, :16].all())
        self.assertFalse(arrays['joint_effective_helpers_valid'][0, :16, 3].any())
        self.assertTrue(arrays['joint_effective_helpers_valid'][0, :16, :3].all())
        np.testing.assert_array_equal(arrays[NEW_MEAN+'_group_states'][0, :16, 3], np.zeros(16))
        ordinary = report['joint_four_arm']['methods'][NEW_MEAN]['same_anchor_both_multi_both_helpers']
        effective = report['joint_effective_helpers']['methods'][NEW_MEAN]['same_anchor_both_multi_both_helpers']
        self.assertEqual(ordinary['heading']['intersection_count'], 80)
        self.assertEqual(effective['heading']['intersection_count'], 48)
        self.assertEqual(effective['position']['intersection_count'], 80)

    def test_changed_anchor_is_separate_from_the_unchanged_anchor_claim(self):
        fragment, saved, _, _, old, new = self.fixture()
        # Before refinement c's representative is node 7; the recovered node 4
        # becomes its earlier representative afterwards. Truth never chooses it.
        fragment['identities'][7] = 'c'
        fragment['target'][7] = saved['aligned_truth'][2]
        fragment['target_valid'][7] = saved['aligned_truth_valid'][2]
        saved = self.rebuild_saved(fragment, saved)
        report, arrays = analyze_refinement(fragment, saved, old, new)
        self.assertEqual(arrays[ANCHOR+'_selected_anchor_node_index'][2], 7)
        self.assertEqual(arrays[NEW_ANCHOR+'_selected_anchor_node_index'][2], 4)
        self.assertFalse(arrays['joint_region_same_anchor'][2].any())
        self.assertTrue(arrays['joint_region_changed_anchor'][2].all())
        key = '__vs__'.join(sorted((ANCHOR, NEW_ANCHOR)))
        self.assertEqual(report['pairwise'][key]['changed_representative_target_count'], 1)

    def test_aggregation_uses_per_scene_capacity_not_pooled_minimum(self):
        fragment, saved, expected, _, old, new = self.fixture()
        first, _ = analyze_refinement(fragment, saved, old, new, expected)
        # A second actual scene has three outputs but only one evaluable target.
        # Its surplus outputs must not compensate for the first scene's deficit.
        saved['aligned_truth_valid'][1:4] = False
        for i, identity in enumerate(fragment['identities']):
            if identity != 'a':
                fragment['target_valid'][i] = False
        saved = self.rebuild_saved(fragment, saved)
        second, _ = analyze_refinement(fragment, saved, old, new)
        pooled = aggregate_refinement([first, second])
        capacity = pooled['grouping']['original']['capacity']['position']
        self.assertEqual(capacity['group_capacity_target_count'], 4)
        self.assertEqual(capacity['group_count_deficit'], 1)
        self.assertEqual(capacity['trajectory_coverage_upper_bound'], 4/5)

    def test_cached_entry_preserves_source_and_writes_recomputable_outputs(self):
        from tools.analysis.group_refinement import run as run055
        with TemporaryDirectory() as directory:
            args = self.cached_source(Path(directory))
            source_before = (args.source_system/'predictions'/'states-1.npz').read_bytes()
            result = run055(args)
            self.assertEqual(result['status'], 'COMPLETED')
            self.assertEqual(result['scene_ids'], ['1'])
            self.assertEqual(set(result['methods']), set(METHODS))
            state = read_json(args.output/'run.json')
            self.assertEqual(state['status'], 'COMPLETED')
            self.assertEqual(state['completed_scenes'], 1)
            self.assertEqual(state['optimizer_updates'], 0)
            self.assertFalse(state['test_opened'])
            self.assertFalse(state['threshold_reselected'])
            self.assertFalse(state['gpu_used'])
            self.assertEqual((args.source_system/'predictions'/'states-1.npz').read_bytes(), source_before)
            with np.load(args.output/'predictions'/'states-1.npz', allow_pickle=False) as arrays:
                np.testing.assert_array_equal(arrays[NEW_ANCHOR+'_selected_anchor_node_index'], [0, 1, 4, 7, -1])
                self.assertIn('joint_four_arm_valid', arrays.files)
                self.assertEqual(int(arrays['joint_region_same_anchor_both_multi_both_helpers'].sum()), 80)

    def test_entry_rejects_drift_in_previously_completed_054_metrics(self):
        from tools.analysis.group_refinement import run as run055
        with TemporaryDirectory() as directory:
            args = self.cached_source(Path(directory))
            path = args.source_attribution/'summary.json'
            summary = read_json(path)
            summary['methods'][BASE]['system']['full_target_metrics']['all']['position']['mae'] += 1.
            write_json(path, summary)
            with self.assertRaises(ValueError):
                run055(args)
            if (args.output/'run.json').exists():
                self.assertEqual(read_json(args.output/'run.json')['status'], 'FAILED')

    def test_entry_rejects_drift_in_original_edge_statistics(self):
        from tools.analysis.group_refinement import run as run055
        with TemporaryDirectory() as directory:
            args = self.cached_source(Path(directory))
            path = args.source_system.parent/'ols'/'baseline.json'
            baseline = read_json(path)
            baseline['evaluation']['per_scene'][0]['fp'] += 1
            write_json(path, baseline)
            with self.assertRaises(ValueError):
                run055(args)
            if (args.output/'run.json').exists():
                self.assertEqual(read_json(args.output/'run.json')['status'], 'FAILED')


if __name__ == '__main__':
    unittest.main()
