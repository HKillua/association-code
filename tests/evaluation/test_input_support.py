"""Actual-input support uses native seconds and supplied groups, never labels."""
import copy
import unittest

import numpy as np

from association.evaluation.support import build_input_support, group_input_support


class InputSupportTests(unittest.TestCase):
    def test_native_seconds_and_nearest_real_point_inside_support(self):
        query = np.array([99., 100.25, 104., 111.75, 120.])
        prepared = build_input_support(query, [[100.25, 111.75, np.nan]], [[True, True, False]])
        np.testing.assert_array_equal(prepared['support'], [[False, True, True, True, False]])
        np.testing.assert_allclose(prepared['nearest_seconds'], [[1.25, 0., 3.75, 0., 8.25]])
        # Being inside the endpoint interval does not imply zero point distance.
        self.assertGreater(prepared['nearest_seconds'][0, 2], 0.)

    def test_hidden_points_and_nan_padding_do_not_influence_support_or_distance(self):
        query = np.array([1., 4., 8., 10., 15.])
        times = np.array([[0., 4., 8., 10., np.nan], [1., 2., 3., 4., 5.]])
        visible = np.array([[False, True, True, True, False], [False]*5])
        before_times, before_visible = times.copy(), visible.copy()
        first = build_input_support(query, times, visible)
        changed = times.copy(); changed[~visible] = np.nan
        second = build_input_support(query, changed, visible)
        for key in first:
            np.testing.assert_array_equal(first[key], second[key])
        np.testing.assert_array_equal(first['support'][0], [False, True, True, True, False])
        self.assertEqual(first['nearest_seconds'][0, 0], 3.)
        self.assertFalse(first['support'][1].any())
        self.assertTrue(np.isinf(first['nearest_seconds'][1]).all())
        np.testing.assert_array_equal(times, before_times)
        np.testing.assert_array_equal(visible, before_visible)

    def test_aligned_support_override_is_copied_and_cannot_invent_input(self):
        support = np.array([[True, False, True], [False, False, False]])
        prepared = build_input_support([0., 5., 10.], [[0., 10.], [np.nan, np.nan]],
                                       [[True, True], [False, False]], node_support=support)
        np.testing.assert_array_equal(prepared['support'], support)
        np.testing.assert_array_equal(prepared['nearest_seconds'][0], [0., 5., 0.])
        support[0, 0] = False
        self.assertTrue(prepared['support'][0, 0])
        support[1, 0] = True
        with self.assertRaisesRegex(ValueError, 'without visible input'):
            build_input_support([0., 5., 10.], [[0., 10.], [np.nan, np.nan]],
                                [[True, True], [False, False]], node_support=support)

    def test_mixed_group_keeps_actual_helper_and_does_not_repair_membership(self):
        prepared = build_input_support([0., 4., 8., 12.],
            [[0., 4.], [8., 12.], [np.nan, np.nan]],
            [[True, True], [True, True], [False, False]])
        # Node 1 may be the wrong identity; no labels enter this API.
        groups = [[0, 1], [2]]
        before_groups, before = copy.deepcopy(groups), copy.deepcopy(prepared)
        result = group_input_support(prepared, groups)
        np.testing.assert_array_equal(result['anchor_support'][0], [True, True, False, False])
        np.testing.assert_array_equal(result['helper_only_support'][0], [False, False, True, True])
        self.assertFalse(result['unsupported'][0].any())
        self.assertTrue(result['unsupported'][1].all())
        self.assertTrue(np.isinf(result['nearest_seconds'][1]).all())
        self.assertEqual(groups, before_groups)
        for key in before:
            np.testing.assert_array_equal(prepared[key], before[key])

    def test_eligible_source_subset_never_promotes_helper_to_anchor(self):
        prepared = build_input_support([0., 2., 4., 6.], [[0., 4.], [2., 6.]],
                                       [[True, True], [True, True]])
        group = [[0, 1]]
        both = group_input_support(prepared, group)
        np.testing.assert_array_equal(both['anchor_support'], [[True, True, True, False]])
        np.testing.assert_array_equal(both['helper_only_support'], [[False, False, False, True]])
        np.testing.assert_array_equal(both['nearest_seconds'], [[0., 0., 0., 0.]])
        source = group_input_support(prepared, group, eligible_groups=[[1]])
        self.assertFalse(source['anchor_support'].any())
        np.testing.assert_array_equal(source['helper_only_support'], [[False, True, True, True]])
        np.testing.assert_array_equal(source['unsupported'], [[True, False, False, False]])
        np.testing.assert_array_equal(source['nearest_seconds'], [[2., 0., 2., 0.]])
        none = group_input_support(prepared, group, eligible_groups=[[]])
        self.assertTrue(none['unsupported'].all())
        self.assertTrue(np.isinf(none['nearest_seconds']).all())

    def test_empty_populations_and_single_visible_point(self):
        prepared = build_input_support([1., 2., 3.], np.empty((0, 0)), np.empty((0, 0), dtype=bool))
        result = group_input_support(prepared, [])
        self.assertEqual(result['anchor_support'].shape, (0, 3))
        result = group_input_support(prepared, [[]])
        self.assertTrue(result['unsupported'].all())
        self.assertTrue(np.isinf(result['nearest_seconds']).all())
        prepared = build_input_support([3., 2., 1.], [[2., np.nan]], [[True, False]])
        np.testing.assert_array_equal(prepared['support'], [[False, True, False]])
        np.testing.assert_array_equal(prepared['nearest_seconds'], [[1., 0., 1.]])
        empty_queries = build_input_support([], [[2.]], [[True]])
        self.assertEqual(group_input_support(empty_queries, [[0]])['unsupported'].shape, (1, 0))


if __name__ == '__main__':
    unittest.main()
