"""Truth-blind support control and exact candidate-copy contracts."""
import unittest

import numpy as np

from association.evaluation.fusion_controls import ordinary_fusion
from association.evaluation.support_control import support_switch_prediction


class SupportControlTests(unittest.TestCase):
    def test_all_support_cases_and_every_group_processed_without_mutation(self):
        anchor = np.arange(5 * 4 * 4, dtype=float).reshape(5, 4, 4)
        mean = anchor + 100.
        support = np.array([[True, False, False, False], [True, True, False, True],
                            [False, True, False, False], [True, True, False, True],
                            [False, True, False, True]])
        groups = [[0, 1], [2, 3], [4]]
        originals = [array.copy() for array in (anchor, mean, support)]
        prediction, switched = support_switch_prediction(anchor, mean, support, groups)
        expected = np.array([[False, True, False, True], [False, False, False, False],
                             [True, False, False, True], [False, False, False, False],
                             [False, False, False, False]])
        np.testing.assert_array_equal(switched, expected)
        np.testing.assert_array_equal(prediction[..., :2][expected], mean[..., :2][expected])
        np.testing.assert_array_equal(prediction[..., :2][~expected], anchor[..., :2][~expected])
        np.testing.assert_array_equal(prediction[..., 2:], mean[..., 2:])
        # Group [2,3] is still processed even if a later evaluator discards it as a
        # duplicate. No list of retained targets can be supplied to this function.
        self.assertTrue(switched[2, 0])
        for current, original in zip((anchor, mean, support), originals):
            np.testing.assert_array_equal(current, original)
        prediction[:] = -1.
        np.testing.assert_array_equal(mean, originals[1])

    def test_exact_pair_copy_preserves_circular_fallback_and_singleton(self):
        states = np.array([[[20., 175., 5., 359.], [20., 175., 5., 359.]],
                           [[30., 0., 6., 0.], [30., -179., 6., 1.]],
                           [[40., 180., 7., 180.], [40., 179., 7., 359.]],
                           [[50., -178., 8., 180.], [50., -178., 8., 180.]]])
        support = np.array([[False, True], [True, True], [True, True], [False, False]])
        groups = [[0, 1, 2], [3]]
        ordinary = ordinary_fusion(states, np.zeros((4, 2)), support, groups)['predictions']
        prediction, switched = support_switch_prediction(
            ordinary['anchor'], ordinary['mean_supported'], support, groups)
        self.assertTrue(switched[0, 0])
        # Longitude and heading use the existing zero-resultant fallback; the
        # switch does not independently average or normalize either coordinate.
        np.testing.assert_array_equal(prediction[0, 0], [35., 175., 6.5, 359.])
        np.testing.assert_array_equal(prediction[0, 1, :2], states[0, 1, :2])
        np.testing.assert_array_equal(prediction[3], states[3])
        np.testing.assert_array_equal(prediction[..., 2:], ordinary['mean_supported'][..., 2:])

    def test_switch_is_action_not_assertion_of_changed_states(self):
        states = np.ones((2, 2, 4))
        prediction, switched = support_switch_prediction(states, states, np.array(
            [[False, False], [True, False]]), [[0, 1]])
        self.assertTrue(switched[0, 0])
        self.assertFalse(switched[:, 1].any())
        np.testing.assert_array_equal(prediction, states)

    def test_reject_invalid_states_support_and_partition(self):
        states = np.ones((3, 2, 4))
        support = np.zeros((3, 2), bool)
        for groups in ([[0, 1], [1, 2]], [[0, 1]], [[1, 0], [2]],
                       [[0, 1, 2], []], [[0., 1., 2.]], [[False, 1, 2]]):
            with self.subTest(groups=groups), self.assertRaises(ValueError):
                support_switch_prediction(states, states, support, groups)
        for invalid_support in (None, support.astype(int), support[:, :1]):
            with self.subTest(support=invalid_support), self.assertRaises(ValueError):
                support_switch_prediction(states, states, invalid_support, [[0, 1, 2]])
        for malformed in (np.ones((3, 2)), np.ones((3, 2, 3)),
                          np.full((3, 2, 4), np.nan)):
            with self.subTest(shape=malformed.shape), self.assertRaises(ValueError):
                support_switch_prediction(malformed, states, support, [[0, 1, 2]])


if __name__ == '__main__':
    unittest.main()
