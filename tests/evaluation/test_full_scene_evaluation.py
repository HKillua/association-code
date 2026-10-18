"""Full target denominators, erroneous groups and reference-independent pairing."""
import json
import unittest

import numpy as np

from association.evaluation.scene import evaluate_full_scene, simple_fusion_outputs, single_source_outputs


def states(speeds):
    speeds = np.asarray(speeds, dtype=float)
    result = np.zeros((*speeds.shape, 4), dtype=float)
    result[..., 2] = speeds
    return result


def evaluate(outputs, nodes, targets, truth, observed=None, valid=None, kind='predicted', split='V_select'):
    if observed is None:
        observed = np.ones((len(nodes), truth.shape[1]), bool)
    if valid is None:
        valid = np.ones(truth.shape, bool)
    return evaluate_full_scene(outputs, node_identities=nodes, target_identities=targets,
                               target_states=truth, target_valid=valid, node_observed=observed,
                               split=split, grouping_kind=kind,
                               correspondence_policy='anchor_identity_first')


class FullSceneEvaluationTests(unittest.TestCase):
    def test_mixed_group_keeps_error_and_merged_target_in_denominator(self):
        truth = states([[1, 1], [10, 10]])
        # Wrongly merges A and B; it is closer to B but anchor remains A.
        outputs = [{'node_indices': [0, 1], 'states': states([9, 9])}]
        result = evaluate(outputs, ['A', 'B'], ['A', 'B'], truth)
        speed = result['full_target_metrics']['all']['speed']
        self.assertEqual(speed['available_target_count'], 4)
        self.assertEqual(speed['intersection_count'], 2)
        self.assertEqual(speed['missing_prediction_count'], 2)
        self.assertEqual(speed['coverage'], .5)
        self.assertEqual(speed['mae'], 8)
        self.assertEqual(result['unmatched_target_ids'], ['B'])
        self.assertEqual(result['group_diagnostics']['merge_pairs'], 1)
        self.assertEqual(result['all_predicted_groups'][0]['member_target_ids'], ['A', 'B'])
        self.assertEqual(result['all_predicted_groups'][0]['metrics']['all']['speed']['mae'], 8)

    def test_split_duplicate_first_group_is_not_selected_by_lowest_error(self):
        truth = states([[1, 1]])
        outputs = [{'node_indices': [0], 'states': states([9, 9])},
                   {'node_indices': [1], 'states': states([1, 1])}]
        result = evaluate(outputs, ['A', 'A'], ['A'], truth)
        self.assertEqual(result['selected_group_by_target'], [0])
        self.assertEqual(result['duplicate_prediction_group_count'], 1)
        metric = result['full_target_metrics']['all']['speed']
        self.assertEqual(metric['available_target_count'], 2)
        self.assertEqual(metric['prediction_without_target_count'], 2)
        self.assertEqual(metric['mae'], 8)
        self.assertEqual(result['group_diagnostics']['split_pairs'], 1)
        self.assertEqual([x['metrics']['all']['speed']['mae'] for x in result['all_predicted_groups']], [8, 0])
        self.assertTrue(result['all_predicted_groups'][1]['duplicate_for_anchor_target'])
        outputs[0]['states'] = None
        missing_first = evaluate(outputs, ['A', 'A'], ['A'], truth)
        self.assertEqual(missing_first['full_target_metrics']['all']['speed']['coverage'], 0)
        self.assertEqual(missing_first['all_predicted_groups'][1]['metrics']['all']['speed']['coverage'], 1)

    def test_target_without_input_nodes_and_missing_singleton_are_retained(self):
        truth = states([[1, 1], [2, 2], [3, 3]])
        outputs = [{'node_indices': [0], 'states': states([1, 1])},
                   {'node_indices': [1], 'states': None}]
        result = evaluate(outputs, ['A', 'B'], ['A', 'B', 'C'], truth)
        self.assertEqual(result['target_identity_count'], 3)
        self.assertEqual(result['anchor_matched_target_count'], 2)
        self.assertEqual(result['selected_group_by_target'], [0, 1, None])
        self.assertEqual(result['unmatched_target_ids'], ['C'])
        metric = result['full_target_metrics']['all']['position']
        self.assertEqual(metric['available_target_trajectory_count'], 3)
        self.assertEqual(metric['missing_target_trajectory_count'], 2)
        self.assertEqual(metric['trajectory_coverage'], 1/3)
        self.assertFalse(result['all_predicted_groups'][1]['has_position_output'])

    def test_observed_region_uses_union_of_original_nodes_not_prediction_availability(self):
        truth = states([[1, 1, 1, 1], [2, 2, 2, 2]])
        outputs = [{'node_indices': [0, 1], 'states': states([2, 4, 6, 8])},
                   {'node_indices': [2], 'states': None}]
        observed = np.array([[True, False, False, False], [False, True, False, False],
                             [False, False, True, False]])
        result = evaluate(outputs, ['A', 'A', 'B'], ['A', 'B'], truth, observed)
        metrics = result['full_target_metrics']
        self.assertEqual(metrics['observed']['speed']['available_target_count'], 3)
        self.assertEqual(metrics['missing']['speed']['available_target_count'], 5)
        self.assertEqual(metrics['observed']['speed']['mae'], 2)
        self.assertEqual(metrics['missing']['speed']['mae'], 6)
        self.assertEqual(metrics['missing']['speed']['missing_prediction_count'], 3)

    def test_oracle_partition_is_explicit_and_cannot_hide_mixed_or_split_groups(self):
        truth = states([[1], [2]])
        outputs = [{'node_indices': [0, 2], 'states': states([1])},
                   {'node_indices': [1], 'states': states([2])}]
        result = evaluate(outputs, ['A', 'B', 'A'], ['A', 'B'], truth, kind='oracle')
        self.assertEqual(result['group_diagnostics']['split_pairs'], 0)
        self.assertEqual(result['group_diagnostics']['merge_pairs'], 0)
        self.assertEqual(result['full_target_metrics']['all']['speed']['mae'], 0)
        for bad in ([{'node_indices': [0, 1, 2], 'states': states([1])}],
                    [{'node_indices': [0], 'states': states([1])},
                     {'node_indices': [1], 'states': states([2])},
                     {'node_indices': [2], 'states': states([1])}]):
            with self.assertRaises(ValueError):
                evaluate(bad, ['A', 'B', 'A'], ['A', 'B'], truth, kind='oracle')

    def test_partial_reference_and_prediction_masks_remain_feature_specific(self):
        truth = states([[1, 2]])
        truth[0, 1, 3] = np.nan
        output = truth[0].copy(); output[0, 2] = np.nan
        result = evaluate([{'node_indices': [0], 'states': output,
                            'prediction_valid': np.isfinite(output)}],
                          ['A'], ['A'], truth, valid=np.isfinite(truth))
        metrics = result['full_target_metrics']['all']
        self.assertEqual(metrics['position']['coverage'], 1)
        self.assertEqual(metrics['speed']['coverage'], .5)
        self.assertEqual(metrics['heading']['available_target_count'], 1)
        json.dumps(result, allow_nan=False)

    def test_empty_scene_and_all_missing_reference_have_defined_null_metrics(self):
        empty = evaluate([], [], [], np.empty((0, 3, 4)))
        self.assertIsNone(empty['full_target_metrics']['all']['position']['coverage'])
        self.assertEqual(empty['target_identity_count'], 0)
        truth = np.full((1, 2, 4), np.nan)
        result = evaluate([{'node_indices': [0], 'states': states([1, 1])}],
                          ['A'], ['A'], truth, valid=np.zeros_like(truth, bool))
        self.assertEqual(result['reference_unavailable_target_count'], 1)
        self.assertEqual(result['full_target_metrics']['all']['position']['prediction_without_target_count'], 2)
        json.dumps(empty, allow_nan=False); json.dumps(result, allow_nan=False)


class SingleSourceEvaluationTests(unittest.TestCase):
    def test_union_uses_first_source_fragment_and_retains_source_absent_targets(self):
        values = states([[1, 1, 1], [4, 4, 4], [9, 9, 9], [2, 2, 2]])
        valid = np.ones_like(values, bool)
        valid[0, 2] = False; values[0, 2] = np.nan
        identities = ['A', 'A', 'A', 'B']
        sources = ['9001', '9001', '9002', '9002']
        before_values, before_valid = values.copy(), valid.copy()
        outputs = single_source_outputs(values, valid, identities, sources, source='9001')
        self.assertEqual([r['node_indices'] for r in outputs], [[0, 1, 2], [3]])
        np.testing.assert_array_equal(outputs[0]['states'][:, 2], [1, 1, 4])
        self.assertEqual(outputs[0]['source_node_indices'], [0, 1])
        self.assertFalse(outputs[1]['prediction_valid'].any())
        result = evaluate(outputs, identities, ['A', 'B'], states([[1, 1, 1], [2, 2, 2]]), kind='oracle')
        self.assertEqual(result['full_target_metrics']['all']['position']['coverage'], .5)
        self.assertEqual(result['full_target_metrics']['all']['speed']['mae'], 1)
        np.testing.assert_equal(values, before_values); np.testing.assert_equal(valid, before_valid)

    def test_coordinates_are_not_combined_from_different_fragments(self):
        values = np.array([[[10., np.nan, 1, 0]], [[np.nan, 20., 2, 180]]])
        valid = np.isfinite(values)
        outputs = single_source_outputs(values, valid, ['A', 'A'], ['S', 'S'], source='S')
        self.assertFalse(outputs[0]['prediction_valid'][0, :2].any())
        self.assertEqual(outputs[0]['states'][0, 2], 1)
        self.assertEqual(outputs[0]['states'][0, 3], 0)
        none = single_source_outputs(values, valid, ['A', 'A'], ['S', 'S'], source='absent')
        self.assertFalse(none[0]['prediction_valid'].any())


class SimpleFusionEvaluationTests(unittest.TestCase):
    def test_equal_source_weight_after_fixed_fragment_selection(self):
        values = states([[2, 2, 2], [100, 100, 100], [6, 6, 6]])
        valid = np.ones_like(values, bool)
        valid[0, 1] = False; values[0, 1] = np.nan
        valid[:, 2] = False; values[:, 2] = np.nan
        before = values.copy()
        outputs = simple_fusion_outputs(values, valid, ['A'] * 3, ['9001', '9001', '9002'])
        np.testing.assert_equal(outputs[0]['states'][:, 2], [4, 53, np.nan])
        self.assertFalse(outputs[0]['prediction_valid'][2].any())
        # Repeating an already selected source fragment cannot add weight.
        duplicated = simple_fusion_outputs(np.concatenate([values, values[1:2]]),
            np.concatenate([valid, valid[1:2]]), ['A'] * 4, ['9001', '9001', '9002', '9001'])
        np.testing.assert_equal(outputs[0]['states'], duplicated[0]['states'])
        np.testing.assert_equal(values, before)
        result = evaluate(outputs, ['A'] * 3, ['A'], states([[4, 53, 7]]),
                          observed=valid.all(axis=-1), kind='oracle')
        self.assertEqual(result['full_target_metrics']['all']['position']['coverage'], 2/3)
        self.assertEqual(result['full_target_metrics']['missing']['speed']['missing_prediction_count'], 1)

    def test_circular_heading_and_longitude_wrap_and_cancellation(self):
        values = np.array([[[10, 179, 2, 359], [10, 20, 2, 0], [10, 0, 2, 90]],
                           [[20, -179, 6, 1], [20, 20, 6, 180], [20, 180, 6, 90]]], float)
        output = simple_fusion_outputs(values, np.ones_like(values, bool), ['A', 'A'], ['S1', 'S2'])[0]
        self.assertAlmostEqual(output['states'][0, 0], 15)
        self.assertAlmostEqual(abs(output['states'][0, 1]), 180)
        self.assertAlmostEqual(output['states'][0, 3], 0)
        self.assertFalse(output['prediction_valid'][1, 3])
        self.assertTrue(output['prediction_valid'][1, :3].all())
        self.assertFalse(output['prediction_valid'][2, :2].any())
        self.assertTrue(output['prediction_valid'][2, 2:].all())

    def test_features_only_average_valid_sources_and_coordinates_stay_paired(self):
        values = np.array([[[10, np.nan, 2, np.nan]], [[np.nan, 20, np.nan, 90]]], float)
        output = simple_fusion_outputs(values, np.isfinite(values), ['A', 'A'], ['S1', 'S2'])[0]
        self.assertFalse(output['prediction_valid'][0, :2].any())
        np.testing.assert_equal(output['states'][0, 2:], [2, 90])
        self.assertEqual(simple_fusion_outputs(np.empty((0, 3, 4)), np.empty((0, 3, 4), bool), [], []), [])
        with self.assertRaises(ValueError):
            simple_fusion_outputs(values, np.ones_like(values, bool), ['A', 'A'], ['S1', 'S2'])


if __name__ == '__main__':
    unittest.main()
