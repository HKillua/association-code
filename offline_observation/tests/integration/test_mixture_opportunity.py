"""Software-only oracle checks; synthetic cases are not MTAD evidence."""
from copy import deepcopy
import unittest

import numpy as np

from association.evaluation import mixture_opportunity as m
from association.evaluation.query_contribution import recover_group_states


def pairs_from_positions(values):
    values = np.asarray(values, np.float64)
    longitude = np.deg2rad(values[..., 1])
    return np.stack((values[..., 0], np.sin(longitude), np.cos(longitude)), axis=-1)


def physical_fixture():
    states = np.zeros((5, 3, 4))
    states[..., 0] = np.arange(5)[:, None]*.01
    states[..., 1] = np.array([0., 120., 90., 179., -179.])[:, None]
    support = np.ones((5, 3), bool)
    support[1, 1] = False
    support[:3, 2] = False
    groups = [[0, 1, 2], [3, 4]]
    counts = np.array([[2, 1, 0], [1, 1, 0], [1, 1, 1], [1, 1, 1]])
    pack = dict(node_to_source_row=np.array([0, 0, 1, 2, 3]), supported_counts=counts,
        group_offsets=np.array([0, 2, 4]), source_group_indices=np.array([0, 0, 1, 1]),
        available=counts > 0)
    return dict(states=states, support=support, groups=groups, pack=pack)


class MixtureOpportunityTests(unittest.TestCase):
    def test_source_moments_support_binding_and_no_mutation(self):
        data = physical_fixture()
        original = deepcopy(data)
        moments = m.source_position_moments(**data)
        self.assertEqual(moments.shape, (4, 3, 3))
        np.testing.assert_allclose(moments[0, 0], [.005, np.sqrt(3.)/4., .25], atol=1e-15)
        np.testing.assert_array_equal(moments[0, 1], [0., 0., 1.])
        np.testing.assert_array_equal(moments[:2, 2], 0.)
        for key in ('states', 'support'):
            np.testing.assert_array_equal(data[key], original[key])
        for key in data['pack']:
            np.testing.assert_array_equal(data['pack'][key], original['pack'][key])
        data['pack']['supported_counts'][0, 0] = 1
        with self.assertRaisesRegex(ValueError, 'binding'):
            m.source_position_moments(**data)
        data = physical_fixture()
        data['pack']['node_to_source_row'][0] = 3
        with self.assertRaisesRegex(ValueError, 'belong'):
            m.source_position_moments(**data)

    def test_moments_recover_raw_nodes_and_expose_nonassociative_angles(self):
        data = physical_fixture()
        moments = m.source_position_moments(**data)
        pairs = np.stack((moments[:2, 0], moments[2:, 0]))
        anchor = data['states'][[0, 3], 0, :2]
        for alpha in (0., .125, .5, .75, 1.):
            weights = np.zeros((5, 3))
            weights[[0, 1], 0] = alpha/2.
            weights[2, 0] = 1.-alpha
            weights[3, 0], weights[4, 0] = alpha, 1.-alpha
            actual = recover_group_states(data['states'], weights, data['groups'])['group_states'][:, 0, :2]
            sufficient = m.mixture_positions(pairs, alpha, anchor)
            np.testing.assert_allclose(sufficient, actual, atol=1e-12, rtol=0.)
            swapped = m.mixture_positions(pairs[:, ::-1], 1.-alpha, anchor)
            np.testing.assert_allclose(sufficient, swapped, atol=1e-12, rtol=0.)
        # Source A has longitude 60 degrees but resultant magnitude .5;
        # replacing it with its unit angle before mixing would incorrectly give 75.
        position = m.mixture_positions(pairs[:1], .5, anchor[:1])
        self.assertGreater(abs(position[0, 1]-75.), 5.)

    def test_symmetric_error_cancellation_and_same_direction_no_gain(self):
        pairs = pairs_from_positions([[[-.01, 0.], [.01, 0.]], [[.01, 0.], [.02, 0.]]])
        truth = np.zeros((2, 2))
        result = m.grid_mixture_oracle(pairs, truth, truth)
        self.assertEqual(result['alpha'][0], .5)
        self.assertAlmostEqual(result['grid_error_m'][0], 0., places=10)
        self.assertAlmostEqual(result['gain_m'][0], m.EARTH_RADIUS_M*np.pi/180.*.01, places=7)
        self.assertEqual(result['alpha'][1], 1.)
        self.assertAlmostEqual(result['gain_m'][1], 0., places=10)
        self.assertEqual(result['audit']['grid_size'], 4097)
        self.assertEqual(result['audit']['grid_candidate_count'], 8194)
        np.testing.assert_allclose(result['endpoint_errors_m'][1],
            m.EARTH_RADIUS_M*np.pi/180.*np.array([.01, .02]), atol=1e-8)
        self.assertTrue(np.all(result['gain_m'] >= -1e-7))

    def test_endpoints_and_date_line(self):
        pairs = pairs_from_positions([[[20., 120.], [21., 121.]], [[0., 179.], [0., -179.]]])
        truth = np.array([[20., 120.], [0., 180.]])
        result = m.grid_mixture_oracle(pairs, truth, truth, chunk_size=1)
        np.testing.assert_array_equal(result['alpha'], [1., .5])
        np.testing.assert_allclose(result['grid_error_m'], 0., atol=1e-7)
        self.assertAlmostEqual(abs(result['position'][1, 1]), 180., places=12)
        self.assertAlmostEqual(result['min_resultant'][1], np.cos(np.pi/180.), places=14)
        self.assertTrue(np.isfinite(result['gap_bound_m']).all())

    def test_exact_grid_ties_midpoint_then_original_index_and_chunks(self):
        pairs = pairs_from_positions([[[3., 10.], [3., 10.]], [[0., 0.], [0., 0.]]])
        anchors = np.zeros((2, 2))
        truth = np.array([[4., 10.], [1., 0.]])
        odd = m.grid_mixture_oracle(pairs, anchors, truth, grid_size=9, chunk_size=1)
        even = m.grid_mixture_oracle(pairs, anchors, truth, grid_size=4, chunk_size=2)
        np.testing.assert_array_equal(odd['alpha'], [.5, .5])
        np.testing.assert_array_equal(odd['tied_grid_count'], [9, 9])
        np.testing.assert_array_equal(even['alpha'], [1./3., 1./3.])
        np.testing.assert_array_equal(even['grid_index'], [1, 1])
        self.assertEqual(odd['audit']['grid_tie_query_count'], 2)
        together = m.grid_mixture_oracle(pairs, anchors, truth, grid_size=9, chunk_size=2)
        for key in odd:
            if key != 'audit':
                np.testing.assert_array_equal(odd[key], together[key])

    def test_continuous_gap_bounds_known_offgrid_optimum(self):
        pairs = pairs_from_positions([[[-.02, 0.], [.03, 0.]]])
        truth = np.zeros((1, 2))
        result = m.grid_mixture_oracle(pairs, truth, truth, grid_size=16)
        # Continuous optimum is alpha=.6; a distinct reference moves it off grid.
        truth[0, 0] = .00123
        result = m.grid_mixture_oracle(pairs, truth*0., truth, grid_size=16)
        expected = m.EARTH_RADIUS_M*np.deg2rad(.05)/(2.*15)+1e-6
        self.assertAlmostEqual(result['gap_bound_m'][0], expected, places=9)
        self.assertGreater(result['grid_error_m'][0], 0.)
        self.assertLessEqual(result['grid_error_m'][0], result['gap_bound_m'][0])
        self.assertEqual(result['audit']['certified_gap_count'], 1)

    def test_undefined_longitude_keeps_point_and_marks_gap_uncertified(self):
        pairs = pairs_from_positions([[[0., 0.], [0., 180.]]])
        anchor = np.array([[0., 37.]])
        result = m.grid_mixture_oracle(pairs, anchor, anchor)
        self.assertEqual(result['alpha'][0], .5)
        np.testing.assert_array_equal(result['position'], anchor)
        self.assertEqual(result['grid_error_m'][0], 0.)
        self.assertLessEqual(result['min_resultant'][0], m.RESULTANT_EPSILON)
        self.assertTrue(np.isnan(result['gap_bound_m'][0]))
        self.assertEqual(result['audit']['uncertified_gap_count'], 1)
        self.assertEqual(result['audit']['degenerate_selected_count'], 1)
        self.assertGreaterEqual(result['audit']['degenerate_grid_candidate_count'], 1)

    def test_reference_only_oracle_does_not_modify_moments_or_physical_recovery(self):
        pairs = pairs_from_positions([[[-.02, 4.], [.03, 4.]]])
        original = pairs.copy()
        anchor = np.array([[0., 4.]])
        physical_before = m.mixture_positions(pairs, .3, anchor)
        first = m.grid_mixture_oracle(pairs, anchor, np.array([[-.02, 4.]]))
        second = m.grid_mixture_oracle(pairs, anchor, np.array([[.03, 4.]]))
        self.assertEqual(first['alpha'][0], 1.)
        self.assertEqual(second['alpha'][0], 0.)
        np.testing.assert_array_equal(pairs, original)
        np.testing.assert_array_equal(m.mixture_positions(pairs, .3, anchor), physical_before)
        self.assertTrue(first['audit']['reference_only'])

    def test_invalid_inputs_and_empty_output(self):
        pairs = pairs_from_positions([[[0., 0.], [1., 1.]]])
        anchor = np.zeros((1, 2))
        for alpha in (-.1, 1.1, np.nan, [0., 1.]):
            with self.assertRaises(ValueError):
                m.mixture_positions(pairs, alpha, anchor)
        for kwargs in ({'grid_size': 1}, {'grid_size': 4098}, {'grid_size': True},
                       {'chunk_size': 0}, {'chunk_size': 129}, {'chunk_size': 1.5}):
            with self.assertRaises(ValueError):
                m.grid_mixture_oracle(pairs, anchor, anchor, **kwargs)
        malformed = pairs.copy(); malformed[0, 0, 0] = np.inf
        with self.assertRaises(ValueError):
            m.grid_mixture_oracle(malformed, anchor, anchor)
        with self.assertRaises(ValueError):
            m.grid_mixture_oracle(pairs, anchor, np.full((1, 2), np.nan))
        empty = m.grid_mixture_oracle(np.empty((0, 2, 3)), np.empty((0, 2)), np.empty((0, 2)))
        self.assertEqual(empty['position'].shape, (0, 2))
        self.assertEqual(empty['audit']['query_count'], 0)
        self.assertEqual(empty['audit']['grid_candidate_count'], 0)
        data = dict(states=np.empty((0, 3, 4)), support=np.empty((0, 3), bool), groups=[],
            pack=dict(node_to_source_row=np.empty(0, np.int64), supported_counts=np.empty((0, 3), np.int64),
                group_offsets=np.array([0]), source_group_indices=np.empty(0, np.int64),
                available=np.empty((0, 3), bool)))
        self.assertEqual(m.source_position_moments(**data).shape, (0, 3, 3))


if __name__ == '__main__':
    unittest.main()
