"""Query-weight contracts on synthetic inputs; these are not MTAD results."""
import inspect
import json
import unittest

import numpy as np

from association.evaluation.query_contribution import (
    query_contribution, recover_group_states, supported_query_weights,
)


def inputs():
    values = np.array([[10., 179., 2., 359.], [20., 179., 4., 359.],
                       [40., -179., 8., 1.], [60., 20., 10., 90.]])
    states = np.repeat(values[:, None, :], 3, axis=1)
    support = np.array([[False, True, True], [False, True, False],
                        [False, True, True], [True, True, True]])
    return states, np.ones((4, 3)), support, np.array(['A', 'A', 'B', 'C']), [[0, 1, 2], [3]]


class QueryContributionTests(unittest.TestCase):
    def test_source_mass_is_equal_despite_unequal_supported_fragment_counts(self):
        data = inputs()
        result = query_contribution(*data, method='source_balanced_supported')
        weights, recovered, audit = result['weights'], result['group_states'], result['audit']
        np.testing.assert_array_equal(weights[:3, 0], [0., 0., 0.])
        np.testing.assert_array_equal(weights[:3, 1], [.25, .25, .5])
        np.testing.assert_array_equal(weights[:3, 2], [.5, 0., .5])
        np.testing.assert_array_equal(recovered[0, 0], data[0][0, 0])
        self.assertEqual(recovered[0, 1, 0], 27.5)
        self.assertEqual(recovered[0, 1, 2], 5.5)
        self.assertEqual(recovered[0, 2, 0], 25.)
        self.assertEqual(audit['supported_source_count'][0], [0, 2, 2])
        self.assertEqual(audit['supported_member_count'][0], [0, 3, 2])
        self.assertEqual(audit['unequal_source_member_counts'][0], [False, True, False])
        self.assertEqual(audit['zero_weight_fallback_count'], [1, 0])
        # Public audit is directly serializable and retains counts, not names.
        json.dumps(audit, allow_nan=False)
        self.assertNotIn('sources', audit)
        self.assertNotIn('states', audit)

    def test_nearest_ignores_unsupported_members_and_ties_use_original_index(self):
        data = list(inputs())
        data[1][:3] = [[0., 0., 0.], [5., 1., 0.], [5., 0., 100.]]
        data[2][:3] = [[False, False, False], [True, False, False], [True, False, True]]
        # Preserve the exact existing angular representation when copying.
        data[0][1, 0, 1:] = [180., 4., 360.]
        result = query_contribution(*data, method='nearest_supported')
        np.testing.assert_array_equal(result['weights'][:3], [[0., 0., 0.], [1., 0., 0.], [0., 0., 1.]])
        np.testing.assert_array_equal(result['group_states'][0],
                                      [data[0][1, 0], data[0][0, 1], data[0][2, 2]])
        self.assertEqual(result['audit']['single_member_copy_count'][0], 2)

    def test_recovery_is_separate_and_handles_circular_wrap_and_anchor_fallback(self):
        states = np.array([[[10., 100., 4., 70.]] * 3,
                           [[20., 179., 6., 359.], [20., 0., 6., 0.], [20., 0., 6., 0.]],
                           [[40., -179., 10., 1.], [40., 180., 10., 180.], [40., 0., 10., 0.]]])
        weights = np.array([[0., 0., 0.], [1., 1., 0.], [1., 1., 0.]])
        result = recover_group_states(states, weights, [[0, 1, 2]])
        output = result['group_states'][0]
        self.assertAlmostEqual(abs(output[0, 1]), 180.)
        self.assertAlmostEqual((output[0, 3] + 180.) % 360. - 180., 0.)
        np.testing.assert_allclose(output[1], [30., 100., 8., 70.], atol=1e-12, rtol=0.)
        np.testing.assert_array_equal(output[2], states[0, 2])
        self.assertEqual(result['audit']['longitude_zero_resultant_count'], [1])
        self.assertEqual(result['audit']['heading_zero_resultant_count'], [1])
        # Replacing W does not require nearest/support/source, and is normalized.
        weights[:, 0] = [2., 1., 0.]
        alternate = recover_group_states(states, weights, [[0, 1, 2]])['group_states'][0]
        self.assertAlmostEqual(alternate[0, 0], 40. / 3.)
        self.assertAlmostEqual(alternate[0, 2], 14. / 3.)

    def test_uniform_source_replication_and_source_renaming_preserve_recovery(self):
        states, nearest, support, sources, groups = inputs()
        original = query_contribution(states, nearest, support, sources, groups,
                                      method='source_balanced_supported')['group_states']
        # Replicate the full A source equally, preserving all its support states.
        take = np.array([0, 1, 0, 1, 2, 3])
        repeated = query_contribution(states[take], nearest[take], support[take], sources[take],
                                      [[0, 1, 2, 3, 4], [5]],
                                      method='source_balanced_supported')['group_states']
        np.testing.assert_allclose(repeated, original, atol=1e-12, rtol=1e-14)
        renamed = query_contribution(states, nearest, support, np.array(['z', 'z', 'a', 'x']), groups,
                                     method='source_balanced_supported')['group_states']
        np.testing.assert_array_equal(renamed, original)

    def test_groups_are_isolated_and_inputs_remain_unmodified(self):
        data = inputs()
        copies = [value.copy() for value in data[:4]]
        for method in ('source_balanced_supported', 'nearest_supported'):
            before = query_contribution(*data, method=method)
            changed = [value.copy() for value in data[:4]] + [data[4]]
            changed[0][3] = [80., 120., 100., 270.]
            changed[1][3] = 900.
            changed[2][3] = False
            changed[3][3] = 'A'
            after = query_contribution(*changed, method=method)
            np.testing.assert_array_equal(before['group_states'][0], after['group_states'][0])
            np.testing.assert_array_equal(before['weights'][:3], after['weights'][:3])
        for original, copied in zip(data[:4], copies):
            np.testing.assert_array_equal(original, copied)
        signature = inspect.signature(query_contribution)
        self.assertEqual(list(signature.parameters), ['states', 'nearest_seconds', 'support', 'sources', 'groups', 'method'])
        with self.assertRaises(TypeError):
            query_contribution(*data, method='nearest_supported', identities=['forbidden'])

    def test_invalid_partition_and_observable_inputs_fail(self):
        for groups in ([[0, 1], [1, 2, 3]], [[0, 1]], [[1, 0], [2, 3]],
                       [[2, 3], [0, 1]], [[0, 1, 2, 3], []], [[0, 1, 2, 4]],
                       [[0., 1., 2., 3.]], [[False, 1, 2, 3]]):
            data = list(inputs()); data[-1] = groups
            with self.subTest(groups=groups), self.assertRaises(ValueError):
                query_contribution(*data, method='source_balanced_supported')
        for index, value in ((0, np.ones((4, 3, 3))), (0, np.full((4, 3, 4), np.nan)),
                             (1, np.full((4, 3), -1.)), (1, np.full((4, 3), np.inf)),
                             (1, np.ones((4, 2))), (2, np.ones((4, 3), int)),
                             (3, ['a', 'b']), (3, ['a', '', 'b', 'c']), (3, [1, 2, 3, 4])):
            data = list(inputs()); data[index] = value
            with self.subTest(index=index, value=value), self.assertRaises(ValueError):
                query_contribution(*data, method='nearest_supported')
        with self.assertRaises(ValueError):
            query_contribution(*inputs(), method='mean_supported')
        for weights in (np.full((4, 3), -1.), np.full((4, 3), np.nan), np.ones((3, 3))):
            with self.subTest(weights=weights), self.assertRaises(ValueError):
                recover_group_states(inputs()[0], weights, inputs()[-1])

    def test_empty_nodes_keep_query_dimension(self):
        states = np.empty((0, 3, 4))
        nearest, support = np.empty((0, 3)), np.empty((0, 3), bool)
        for method in ('source_balanced_supported', 'nearest_supported'):
            result = query_contribution(states, nearest, support, [], [], method=method)
            self.assertEqual(result['group_states'].shape, (0, 3, 4))
            self.assertEqual(result['weights'].shape, (0, 3))
        with self.assertRaises(ValueError):
            supported_query_weights(np.empty((0, 0)), np.empty((0, 0), bool), [], [], method='nearest_supported')


if __name__ == '__main__':
    unittest.main()
