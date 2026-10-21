"""058 cache-to-report contracts, including missing targets and truth-only selectors."""
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from association.evaluation.fusion_controls import ordinary_fusion
from association.evaluation.oracle_diagnostics import METHODS, aggregate_oracles, analyze_oracles, select_candidates
from association.evaluation.scene import evaluate_full_scene
from association.evaluation.system_attribution import BASE
from association.evaluation.tmtaff import aggregate_fusion_reports
from association.utils.runtime import read_json, write_json
from tools.analysis.oracle_diagnostics import run
from tools.analysis.system_attribution import read


class OracleDiagnosticsTests(unittest.TestCase):
    @staticmethod
    def fixture():
        """A's first fragment is swallowed by X; A has later duplicate anchors.

        B has input but no predicted representative; C has no input. D has an
        output without valid reference. Truth grouping switches A's anchor.
        """
        q = 4
        nodes = np.asarray(['x', 'a', 'a', 'b', 'a', 'd'])
        targets = np.asarray(['x', 'a', 'b', 'c', 'd'])
        truth = np.broadcast_to(np.array([
            [30., 120., 5., 1.], [30.01, 120.01, 6., 359.],
            [30.02, 120.02, 7., 2.], [30.03, 120.03, 8., 180.],
            [30.04, 120.04, 9., 90.],
        ])[:, None], (5, q, 4)).copy()
        valid = np.ones_like(truth, bool)
        indices = np.array([0, 1, 1, 2, 1, 4])
        native = truth[indices].copy()
        native[0, :, 0] += .003
        native[1, :, 0] += .001
        native[2, :, 0] += .006
        native[4, :, 0] -= .001
        native[1, :, 3] = 1.
        native[2, :, 3] = 355.
        native[4, :, 3] = 359.
        native[2, :, 2] += 4.
        valid[1, 1, :2] = False
        valid[2, 2, 3] = False
        valid[4] = False
        truth[~valid] = np.nan
        support = np.ones((6, q), bool)
        support[:, -1] = False
        nearest = np.broadcast_to(np.array([0., 60., 600., 2000.]), (6, q)).copy()
        groups = [[0, 1], [2, 3], [4], [5]]
        ordinary = ordinary_fusion(native, nearest, support, groups)
        mean = ordinary['predictions']['mean_supported']
        fragment = dict(node_ids=np.asarray([f'node-{i}' for i in range(6)]),
            identities=nodes, sources=np.asarray(['1', '2', '1', '2', '3', '1']),
            target_identity_ids=targets, query_seconds=np.array([0., 60., 600., 2000.]),
            target=truth[indices], target_valid=valid[indices], native_states=native,
            nearest_seconds=nearest, support=support, exact_observation=nearest == 0,
            ols_mean_supported=mean, ols_membership=ordinary['membership'],
            ols_group_offsets=np.asarray([0, 2, 4, 5, 6]),
            ols_group_node_indices=np.asarray([0, 1, 2, 3, 4, 5]))
        system = evaluate_full_scene([dict(node_indices=g, states=mean[g[0]]) for g in groups],
            node_identities=nodes, target_identities=targets, target_states=truth, target_valid=valid,
            node_observed=support, split='V_select', grouping_kind='predicted',
            correspondence_policy='anchor_identity_first', return_artifact=True,
            node_input_support=dict(support=support, nearest_seconds=nearest))
        saved = system.pop('artifact')
        expected = dict(scene_id='1', component_id='1', unique_target={BASE: system})
        return fragment, saved, expected

    @classmethod
    def cache(cls, root):
        source = root/'source053'
        (source/'predictions').mkdir(parents=True)
        (source/'system-predictions').mkdir()
        fragment, saved, expected = cls.fixture()
        np.savez_compressed(source/'predictions'/'states-1.npz', **fragment)
        np.savez_compressed(source/'system-predictions'/'states-1.npz',
                            **{BASE+'_'+key: value for key, value in saved.items()})
        summary = dict(status='COMPLETED', dataset_level='Level1', split='V_select',
            scene_ids=['1'], scene_count=1, fragment_count=6, method_names=[BASE],
            test_opened=False, optimizer_updates=0, per_scene=[expected],
            ols_working_point=dict(threshold=-1.95, motion_observation_policy='canonical_all',
                                   validation_scene_ids=['1']),
            unique_target={BASE: aggregate_fusion_reports([expected['unique_target'][BASE]])})
        write_json(source/'summary.json', summary)
        write_json(source/'run.json', dict(status='COMPLETED', test_opened=False, optimizer_updates=0,
            completed_scenes=1, total_scenes=1, motion_observation_policy='canonical_all',
            threshold_reselected=False, normalizer_refitted=False))
        manifest = root/'manifest.json'
        write_json(manifest, dict(dataset_level='Level1', split_unit='scene',
            splits=dict(T_core=['unread'], V_select=['1'], T_test=['sealed'])))
        return SimpleNamespace(source_system=source, manifest=manifest, output=root/'oracle058', cpu_threads=1)

    def test_selector_joint_position_circular_heading_ties_and_invalid_truth(self):
        anchor = np.array([[[.1, 0., 10., 359.], [0., .2, 5., 4.], [1., 2., 3., 4.]]])
        mean = np.array([[[0., .2, 0., 2.], [.01, 0., 5., 4.], [0., 0., 0., 0.]]])
        truth = np.array([[[0., 0., 0., 1.], [0., 0., 5., 4.], [np.nan]*4]])
        truth_valid = np.isfinite(truth)
        prediction_valid = np.ones_like(truth_valid)
        prediction_valid[0, 1, 2] = False
        result, valid, choice = select_candidates(anchor, mean, truth, truth_valid, prediction_valid)
        np.testing.assert_array_equal(result[0, 0, :2], anchor[0, 0, :2])
        np.testing.assert_array_equal(result[0, 1, :2], mean[0, 1, :2])
        self.assertEqual(result[0, 0, 3], 2.)  # Circular errors: 1 degree vs 2.
        np.testing.assert_array_equal(choice[0, 1], [True, False, False])
        np.testing.assert_array_equal(result[0, 2], anchor[0, 2])
        np.testing.assert_array_equal(valid, prediction_valid)
        self.assertFalse(choice[0, 2].any())

    def test_full_population_representatives_and_common_point_bounds(self):
        fragment, saved, expected = self.fixture()
        report, arrays = analyze_oracles(fragment, saved, expected)
        self.assertEqual(set(report['methods']), set(METHODS))
        pred = report['methods']['predicted_anchor']['system']
        oracle = report['methods']['oracle_identity_anchor']['system']
        self.assertEqual(pred['duplicate_prediction_group_count'], 1)
        self.assertEqual(oracle['duplicate_prediction_group_count'], 0)
        self.assertEqual(oracle['target_identity_count'], 5)
        self.assertEqual(oracle['unmatched_target_ids'], ['c'])
        self.assertEqual(oracle['reference_unavailable_target_count'], 1)
        np.testing.assert_array_equal(arrays['predicted_anchor_selected_anchor_node_index'], [0, 2, -1, -1, 5])
        np.testing.assert_array_equal(arrays['oracle_identity_anchor_selected_anchor_node_index'], [0, 1, 3, -1, 5])
        pair = report['pairwise']['predicted_anchor__vs__oracle_identity_anchor']
        self.assertEqual(pair['changed_representative_target_count'], 1)
        self.assertEqual(pair['coverage_transitions']['position']['rescued_target_count'], 1)
        self.assertEqual(pair['coverage_transitions']['position']['gained_target_points'], 4)
        self.assertEqual(pair['methods']['predicted_anchor']['changed_representative']['position']['intersection_count'], 3)
        self.assertEqual(pair['methods']['predicted_anchor']['same_representative']['position']['intersection_count'], 4)
        for group in ('predicted', 'oracle_identity'):
            for arm in ('anchor', 'mean_supported', 'attribute_hybrid'):
                comparison = report['pairwise'][group+'_'+arm+'__vs__'+group+'_oracle_selector']
                for attr in ('position', 'speed', 'heading'):
                    delta = comparison['error_change_second_minus_first']['all'][attr]
                    self.assertEqual(delta['second_worse_point_count'], 0)
                    self.assertEqual(comparison['coverage_transitions'][attr]['gained_target_points'], 0)
                    self.assertEqual(comparison['coverage_transitions'][attr]['lost_target_points'], 0)
            self.assertFalse(arrays[group+'_oracle_selector_aligned_prediction_valid'][3].any())
        self.assertTrue(arrays['oracle_identity_anchor_diagnostic_only'])
        self.assertFalse(arrays['predicted_anchor_diagnostic_only'])
        self.assertTrue(arrays['predicted_oracle_selector_uses_reference_for_selection'])
        np.testing.assert_array_equal(arrays['predicted_mean_supported_group_states'], saved['group_states'])

    def test_reference_changes_do_not_modify_ordinary_predictions(self):
        fragment, saved, _ = self.fixture()
        original, arrays = analyze_oracles(fragment, saved)
        altered_fragment, altered_saved = deepcopy(fragment), deepcopy(saved)
        altered_fragment['target'][..., 2] += 20.
        altered_saved['aligned_truth'][..., 2] += 20.
        # The cache itself still describes identical fixed predictions, so
        # change reference arrays only and evaluate without stale metric trees.
        changed, other = analyze_oracles(altered_fragment, altered_saved)
        for group in ('predicted', 'oracle_identity'):
            for arm in ('anchor', 'mean_supported', 'attribute_hybrid'):
                np.testing.assert_array_equal(arrays[group+'_'+arm+'_group_states'], other[group+'_'+arm+'_group_states'])
        self.assertNotEqual(original['methods']['predicted_anchor']['system']['full_target_metrics']['all']['speed']['mae'],
                            changed['methods']['predicted_anchor']['system']['full_target_metrics']['all']['speed']['mae'])

    def test_point_weighted_errors_and_scene_macros_remain_distinct(self):
        fragment, saved, _ = self.fixture()
        first, _ = analyze_oracles(fragment, saved)
        fragment['target_valid'][:, 1:] = False
        saved['aligned_truth_valid'][:, 1:] = False
        fragment['target'][..., 0] += .05
        saved['aligned_truth'][..., 0] += .05
        second, _ = analyze_oracles(fragment, saved)
        pooled = aggregate_oracles([first, second])
        metrics = [r['methods']['predicted_anchor']['system']['full_target_metrics']['all']['position']
                   for r in (first, second)]
        expected = sum(r['mae']*r['intersection_count'] for r in metrics)/sum(r['intersection_count'] for r in metrics)
        macro = sum(r['mae'] for r in metrics)/2
        actual = pooled['methods']['predicted_anchor']['full_target_metrics']['all']['position']
        self.assertAlmostEqual(actual['mae'], expected)
        self.assertAlmostEqual(actual['scene_macro_mae'], macro)
        self.assertNotAlmostEqual(expected, macro)
        key = 'predicted_anchor__vs__oracle_identity_anchor'
        deltas = [r['pairwise'][key]['error_change_second_minus_first']['all']['position'] for r in (first, second)]
        actual_delta = pooled['pairwise'][key]['error_change_second_minus_first']['all']['position']
        self.assertAlmostEqual(actual_delta['mean_error_change'],
            sum(r['signed_error_change_sum'] for r in deltas)/sum(r['common_point_count'] for r in deltas))
        self.assertAlmostEqual(actual_delta['scene_macro_error_change'], sum(r['mean_error_change'] for r in deltas)/2)
        self.assertNotAlmostEqual(actual_delta['mean_error_change'], actual_delta['scene_macro_error_change'])

    def test_cache_entry_outputs_complete_and_failure_is_terminal(self):
        with TemporaryDirectory() as directory:
            args = self.cache(Path(directory))
            result = run(args)
            self.assertEqual(result['status'], 'COMPLETED')
            self.assertEqual(result['scene_count'], 1)
            self.assertFalse(result['gpu_used'])
            self.assertEqual(result['optimizer_updates'], 0)
            self.assertFalse(result['test_opened'])
            self.assertTrue(result['all_source_scenes_replayed'])
            self.assertTrue((args.output/'predictions'/'states-1.npz').exists())
            self.assertEqual(len(read(args.output/'scene-summary.json.gz')), 1)
            with np.load(args.output/'predictions'/'states-1.npz', allow_pickle=False) as archive:
                self.assertEqual(archive['oracle_identity_anchor_grouping_kind'], 'oracle')
            with self.assertRaises(FileExistsError):
                run(args)
            self.assertEqual(read_json(args.output/'run.json')['status'], 'COMPLETED')
            # Fail source replay after creating a new attempt directory; never
            # publish a successful summary or silently repair changed states.
            path = args.source_system/'predictions'/'states-1.npz'
            with np.load(path, allow_pickle=False) as archive:
                fragment = {key: archive[key] for key in archive.files}
            fragment['native_states'][0, 0, 0] += .1
            np.savez_compressed(path, **fragment)
            args.output = Path(directory)/'failed058'
            with self.assertRaisesRegex(ValueError, 'ordinary mean cache'):
                run(args)
            self.assertEqual(read_json(args.output/'run.json')['status'], 'FAILED')
            self.assertFalse((args.output/'summary.json').exists())
            args.output = Path(directory)/'interrupted058'
            with patch('tools.analysis.oracle_diagnostics.analyze_oracles', side_effect=KeyboardInterrupt):
                with self.assertRaises(KeyboardInterrupt):
                    run(args)
            self.assertEqual(read_json(args.output/'run.json')['status'], 'INTERRUPTED')


if __name__ == '__main__':
    unittest.main()
