"""Guard R4's circular math, no-target selection, and retained anchor population."""
import unittest

import numpy as np

from association.evaluation.grouping import group_predicted_tracks
from tools.analysis.fixed_group_fusion_control import complete_native_attributes, metrics, ordinary_fusion


class FixedGroupFusionTests(unittest.TestCase):
    def test_native_attribute_interpolation_padding_and_validity(self):
        values = np.array([[[0., 179., 2., 350.], [0., -179., 6., 10.],
                            [np.nan, np.nan, -1., -1.]]])
        times = np.array([[0., 2., np.nan]])
        visible = np.array([[True, True, False]])
        query = np.array([-1., 0., 1., 2., 3.])
        positions = np.zeros((1, 5, 2))
        result = complete_native_attributes(values, times, visible, query, positions)
        np.testing.assert_array_equal(result[0, :, 2], [2., 2., 4., 6., 6.])
        np.testing.assert_array_equal(result[0, :, 3], [350., 350., 0., 10., 10.])
        values[0, 0, 3] = -1.
        with self.assertRaises(ValueError):
            complete_native_attributes(values, times, visible, query, positions)

    def test_fixed_fusion_circular_means_selection_fallback_and_common_masks(self):
        states = np.array([[[0., 179., 2., 350.], [0., 179., 2., 0.]],
                           [[0., -179., 6., 10.], [0., -179., 6., 180.]],
                           [[10., 20., 30., 40.], [10., 20., 30., 40.]]])
        nearest = np.array([[1., 1.], [1., 0.], [2., 2.]])
        support = np.array([[True, False], [False, False], [False, True]])
        result = ordinary_fusion(states, nearest, support, [[0, 1], [2]])
        self.assertEqual(result['selected_nodes'].tolist(), [[0, 1], [0, 1], [2, 2]])
        predictions = result['predictions']
        np.testing.assert_array_equal(predictions['anchor'], states)
        np.testing.assert_array_equal(predictions['mean_supported'][:, 1], states[:, 1])
        np.testing.assert_array_equal(predictions['mean_supported'][:2, 0], np.repeat(states[:1, 0], 2, axis=0))
        np.testing.assert_allclose(predictions['mean_all'][:2, 0, 1], -180.)
        np.testing.assert_allclose(predictions['mean_all'][:2, 0, 3] % 360., 0., atol=1e-12)
        # Antipodal headings are not averaged to an arbitrary 90 degrees.
        np.testing.assert_array_equal(predictions['mean_all'][:2, 1, 3], [0., 180.])
        self.assertEqual(result['audit']['mean_all']['heading_zero_resultant'], 2)
        self.assertEqual(result['audit']['mean_supported']['no_supported_member_fallback'], 3)
        self.assertEqual(result['helper_support'].tolist(), [[False, False], [True, False], [False, False]])
        valid = np.ones(states.shape, bool)
        valid[0, 0, 2] = False
        target = states.copy()
        target[0, 0, 2] = np.nan
        for prediction in predictions.values():
            result_metrics = metrics(prediction, target, valid, np.ones((3, 2), bool))
            self.assertEqual(result_metrics['position']['intersection_count'], 6)
            self.assertEqual(result_metrics['speed']['intersection_count'], 5)
            self.assertEqual(result_metrics['heading']['intersection_count'], 6)
        # Changing evaluation truth cannot change the already fixed predictions.
        second = ordinary_fusion(states, nearest, support, [[0, 1], [2]])
        for arm in predictions:
            np.testing.assert_array_equal(second['predictions'][arm], predictions[arm])

    def test_predicted_groups_remain_direct_and_singletons_survive(self):
        pairs = np.array([[0, 1], [0, 2], [1, 2], [0, 3], [1, 3], [2, 3]])
        groups = group_predicted_tracks(4, pairs, np.array([1., 0., 1., 0., 0., 0.]),
                                       threshold=.5, grouping_policy='anchor_direct', node_order='input')
        self.assertEqual(groups, [[0, 1], [2], [3]])
        states = np.zeros((4, 2, 4))
        result = ordinary_fusion(states, np.ones((4, 2)), np.ones((4, 2), bool), groups)
        self.assertEqual(result['membership'].tolist(), [0, 0, 1, 2])
        for prediction in result['predictions'].values():
            self.assertEqual(prediction.shape, states.shape)
            self.assertTrue(np.isfinite(prediction).all())


if __name__ == '__main__':
    unittest.main()
