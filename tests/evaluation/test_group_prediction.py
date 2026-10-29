"""Prediction/emission compatibility on synthetic arrays, not MTAD results."""
from copy import deepcopy
import inspect
import unittest

import numpy as np

from association.evaluation.fusion_controls import ordinary_fusion
from association.evaluation.group_prediction import (
    PREDICTION_METHODS, emit_group_outputs, predict_group_outputs,
)
from association.evaluation.query_contribution import query_contribution
from association.evaluation.scene import evaluate_full_scene


def inputs():
    values = np.array([[10., 175., 2., 359.], [20., 180., 4., 360.],
                       [30., 0., 6., 0.], [40., 180., 8., 180.],
                       [50., -180., 10., 360.]])
    states = np.repeat(values[:, None, :], 4, axis=1)
    states[2, 2, 1:] = [180., 6., 360.]
    nearest = np.array([[0., 0., 0., 0.], [1., 1., 1., 1.],
                        [2., 2., 2., 2.], [3., 3., 3., 3.], [0., 0., 0., 0.]])
    support = np.array([[False, False, False, False], [False, False, True, True],
                        [False, True, True, True], [False, True, False, True],
                        [False, False, False, False]])
    return states, nearest, support, np.array(['A', 'B', 'A', 'C', 'D']), [[0, 2, 3], [1], [4]]


class GroupPredictionTests(unittest.TestCase):
    def test_common_entry_keeps_all_existing_controls_exact(self):
        states, nearest, support, sources, groups = inputs()
        ordinary = ordinary_fusion(states, nearest, support, groups)
        anchors = [group[0] for group in groups]
        for method in PREDICTION_METHODS:
            with self.subTest(method=method):
                result = predict_group_outputs(states, nearest, support, sources, groups, method=method)
                if method in ordinary['predictions']:
                    expected = ordinary['predictions'][method][anchors]
                    self.assertIsNone(result['weights'])
                else:
                    previous = query_contribution(states, nearest, support, sources, groups, method=method)
                    expected = previous['group_states']
                    np.testing.assert_array_equal(result['weights'], previous['weights'])
                    self.assertEqual(result['audit'], previous['audit'])
                np.testing.assert_array_equal(result['group_states'], expected)
                self.assertEqual([row['node_indices'] for row in result['group_outputs']], groups)
                np.testing.assert_array_equal(np.stack([row['states'] for row in result['group_outputs']]), expected)
        # Historical mean circular cancellation still falls back to this group's
        # first anchor, and its all-unsupported queries retain that whole state.
        mean = predict_group_outputs(*inputs(), method='mean_supported')['group_states']
        np.testing.assert_array_equal(mean[0, 0], states[0, 0])
        np.testing.assert_array_equal(mean[0, 1], [35., 175., 7., 359.])
        # The strong anchor is exact; old nearest may choose an unsupported node.
        for method in ('anchor', 'nearest_observation'):
            np.testing.assert_array_equal(predict_group_outputs(*inputs(), method=method)['group_states'],
                                          states[anchors])
        supported = predict_group_outputs(*inputs(), method='nearest_supported')['group_states']
        np.testing.assert_array_equal(supported[0, 2], states[2, 2])
        # Valid integer distances must keep legacy ordering even above the
        # float64 exact-integer range; converting these first would create a tie.
        distant = np.array([[2**53 + 1], [2**53]], dtype=np.int64)
        unchanged = predict_group_outputs(states[:2, :1], distant, support[:2, :1],
                                          sources[:2], [[0, 1]], method='nearest_observation')
        np.testing.assert_array_equal(unchanged['group_states'][0], states[1, :1])

    def test_emission_can_follow_an_independent_recovery_and_never_aliases_inputs(self):
        data = inputs()
        copied = deepcopy(data)
        result = predict_group_outputs(*data, method='source_balanced_supported')
        replacement = result['group_states'] + 1.
        emitted = emit_group_outputs(replacement, data[-1], node_count=len(data[0]))
        np.testing.assert_array_equal(emitted[0]['states'], replacement[0])
        emitted[0]['states'][:] = -1.
        emitted[0]['node_indices'].append(99)
        result['group_outputs'][0]['states'][:] = -2.
        for current, original in zip(data[:4], copied[:4]):
            np.testing.assert_array_equal(current, original)
        self.assertEqual(data[-1], copied[-1])
        np.testing.assert_array_equal(replacement, result['group_states'] + 1.)

    def test_all_emitted_groups_reach_existing_evaluation_without_target_selection(self):
        states, nearest, support, sources, groups = inputs()
        result = predict_group_outputs(states, nearest, support, sources, groups, method='mean_supported')
        old_states = ordinary_fusion(states, nearest, support, groups)['predictions']['mean_supported']
        previous = [dict(node_indices=g, states=old_states[g[0]]) for g in groups]
        kwargs = dict(node_identities=['X', 'Y', 'X', 'X', 'X'], target_identities=['X', 'Y', 'Z'],
                      target_states=states[:3], target_valid=np.ones((3, 4, 4), bool),
                      node_observed=support, split='V_select', grouping_kind='predicted',
                      correspondence_policy='anchor_identity_first', return_artifact=True)
        before = evaluate_full_scene(previous, **kwargs)
        after = evaluate_full_scene(result['group_outputs'], **kwargs)
        old_artifact, new_artifact = before.pop('artifact'), after.pop('artifact')
        self.assertEqual(after, before)
        for key in old_artifact:
            np.testing.assert_array_equal(new_artifact[key], old_artifact[key])
        self.assertEqual(after['selected_group_by_target'], [0, 1, None])
        self.assertEqual(after['duplicate_prediction_group_count'], 1)
        self.assertEqual(after['predicted_group_count'], 3)
        self.assertEqual(after['full_target_metrics']['all']['position']['missing_prediction_count'], 4)
        self.assertEqual(after['full_target_metrics']['all']['position']['prediction_without_target_count'], 4)

    def test_emission_rejects_incomplete_rows_partition_and_invalid_physical_values(self):
        values = np.ones((2, 3, 4))
        for states, groups, count in (
                (values[:1], [[0, 1], [2]], 3), (values, [[0, 1]], 3),
                (values, [[0, 1], [1, 2]], 3), (values, [[1, 0], [2]], 3),
                (values, [[0, 1], [2]], True), (values, [[0, 1], [2]], 3.),
                (np.ones((2, 0, 4)), [[0, 1], [2]], 3),
                (np.ones((2, 3, 3)), [[0, 1], [2]], 3),
                (np.full((2, 3, 4), np.nan), [[0, 1], [2]], 3)):
            with self.subTest(shape=states.shape, groups=groups, count=count), self.assertRaises(ValueError):
                emit_group_outputs(states, groups, node_count=count)

    def test_common_input_boundary_rejects_bad_arrays_and_reference_arguments(self):
        for index, value in ((0, np.full((5, 4, 4), np.inf)), (0, np.ones((5, 4, 3))),
                             (1, np.full((5, 4), -1.)), (2, np.ones((5, 4), int)),
                             (3, ['A']), (4, [[0, 1]])):
            data = list(inputs()); data[index] = value
            for method in ('anchor', 'nearest_supported'):
                with self.subTest(index=index, method=method), self.assertRaises(ValueError):
                    predict_group_outputs(*data, method=method)
        with self.assertRaises(ValueError):
            predict_group_outputs(*inputs(), method='unregistered')
        for func in (predict_group_outputs, emit_group_outputs):
            self.assertNotIn('fragment', inspect.signature(func).parameters)
            self.assertNotIn('reference', inspect.signature(func).parameters)
        with self.assertRaises(TypeError):
            predict_group_outputs(*inputs(), method='anchor', reference=np.zeros((1, 4, 4)))

    def test_empty_scene_emits_no_groups_and_retains_query_shape(self):
        for method in PREDICTION_METHODS:
            result = predict_group_outputs(np.empty((0, 3, 4)), np.empty((0, 3)),
                                           np.empty((0, 3), bool), [], [], method=method)
            self.assertEqual(result['group_states'].shape, (0, 3, 4))
            self.assertEqual(result['group_outputs'], [])


if __name__ == '__main__':
    unittest.main()
