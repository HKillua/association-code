"""Synthetic checks of the one-interval course readout and its C149 boundary."""

import unittest

import numpy as np

from association.data.motion_evidence import scene_motion_evidence
from tools.analysis import heading_fallback_fit as c149
from tools.analysis import heading_interval_readout as readout


def track(name, times, longitude):
    times = np.asarray(times, dtype=np.float64)
    values = np.column_stack((np.zeros(len(times)), np.broadcast_to(longitude, times.shape),
                              np.ones(len(times)), np.full(len(times), 13.)))
    return dict(node_id=name, source=name, canonical_times=times, canonical_values=values)


def frozen_scene(tracks):
    nodes = np.array([item['node_id'] for item in tracks])
    sources = np.array([item['source'] for item in tracks])
    query = np.arange(64.) * 10.
    native_time = np.concatenate([item['canonical_times'] for item in tracks])
    native_position = np.concatenate([item['canonical_values'][:, :2] for item in tracks])
    offsets = np.r_[0, np.cumsum([len(item['canonical_times']) for item in tracks])]
    motion = scene_motion_evidence(dict(tracks=tracks, node_ids=nodes, sources=sources, grid=query),
                                   motion_observation_policy='canonical_all')
    motion['node_ids'] = nodes
    native = query, native_time, native_position, offsets
    baseline = c149.freeze_scene(motion, tracks, native, np.zeros(len(nodes), dtype=np.int64))
    return motion, tracks, native, baseline


class HeadingIntervalReadoutTest(unittest.TestCase):
    def test_spherical_course_and_date_line(self):
        intervals = np.array([
            [0., 1., 0., 0., 1., 0.],
            [0., 1., 0., 0., 0., 1.],
            [0., 1., 0., 0., -1., 0.],
            [0., 1., 0., 0., 0., -1.],
            [0., 1., 0., 179.9, 0., -179.9],
            [0., 1., 0., -179.9, 0., 179.9],
            [0., 1., 0., 0., 0., 0.],
            [0., 1., 0., 0., 0., 180.],
        ])
        bearing = readout.initial_bearing(intervals)
        np.testing.assert_allclose(bearing[:6], [0., 90., 180., 270., 90., 270.], atol=1e-9)
        self.assertTrue(np.isnan(bearing[6:]).all())

    def test_low_speed_geometry_tie_copies_and_conservative_exclusion(self):
        time = np.arange(81.)
        east, west = 12. + time * .0001, 12. - time * .0001
        args = frozen_scene([track('east', time, east), track('west', time, west),
                             track('east_copy', time, east), track('west_copy', time, west)])
        baseline = args[-1]
        changed = readout.freeze_scene(*args)
        q = 5
        self.assertTrue(changed['enabled'][0, q])
        self.assertEqual(changed['fallback_reason'][0, q], readout.ENABLED)
        self.assertEqual(changed['selected_time_side'][0, q], readout.AFTER_QUERY)
        self.assertEqual(changed['selected_provider_node'][0, q], 1)
        self.assertEqual(changed['selected_provider_native_row'][0, q], 51)
        np.testing.assert_allclose(changed['selected_interval'][0, q], [51., 52., 0., west[51], 0., west[52]])
        self.assertAlmostEqual(changed['new_states'][0, q, 3], 270.)
        self.assertEqual(changed['selected_midpoint_age_seconds'][0, q], 1.5)
        self.assertEqual(changed['selected_max_age_seconds'][0, q], 2.)
        winner = changed['selected_interval_index'][0, q]
        k0, k1 = changed['candidate_provider_offsets'][winner:winner+2]
        np.testing.assert_array_equal(changed['candidate_provider_nodes'][k0:k1], [1, 3])
        np.testing.assert_array_equal(changed['candidate_provider_native_rows'][k0:k1], [51, 51])
        a, b = changed['pool_query_offsets'][q:q+2]
        pool = changed['candidate_keys'][changed['pool_candidate_indices'][a:b]]
        self.assertEqual(len(pool), baseline['nonreuse_count'][0, q])
        self.assertEqual(len(pool), len(np.unique(pool, axis=0)))
        self.assertIn(winner, changed['pool_candidate_indices'][a:b])
        # Both of the native position reports around each supported Q are protected.
        self.assertFalse(np.isin(pool[:, :2], [49., 50.]).any())
        self.assertTrue(((pool[:, 0] == 51.) & (pool[:, 1] == 52.)
                         & (pool[:, 3] == east[51])).any())
        np.testing.assert_array_equal(changed['new_states'][..., :3], baseline['states'][..., :3])
        np.testing.assert_array_equal(changed['new_states'][~changed['enabled']], baseline['states'][~changed['enabled']])
        self.assertEqual(changed['fallback_reason'][0, 30], readout.NOT_LOW_SPEED)
        for key in baseline:
            np.testing.assert_array_equal(changed[key], baseline[key])

    def test_degenerate_nearest_candidate_keeps_original_angle(self):
        query = np.arange(64.) * 10.
        time = np.r_[query[:7], [61. - 1e-8, 61. + 1e-8, 62., 63.], query[7:]]
        longitude = np.full(len(time), 12.)
        longitude[8:11] += [1e-11, .0001, .0002]
        args = frozen_scene([track('single', time, longitude)])
        changed = readout.freeze_scene(*args)
        q = 6
        self.assertEqual(changed['fallback_reason'][0, q], readout.DEGENERATE)
        self.assertFalse(changed['enabled'][0, q])
        self.assertEqual(changed['new_states'][0, q, 3], args[-1]['states'][0, q, 3])
        np.testing.assert_array_equal(changed['selected_interval'][0, q, :2], time[7:9])
        self.assertEqual(changed['selected_provider_native_row'][0, q], 7)
        a, b = changed['pool_query_offsets'][q:q+2]
        pool = changed['candidate_keys'][changed['pool_candidate_indices'][a:b]]
        self.assertTrue(np.isfinite(readout.initial_bearing(pool)).any())

    def test_absent_geometry_and_archive_drift(self):
        args = frozen_scene([track('still', np.arange(64.) * 10., 12.)])
        changed = readout.freeze_scene(*args)
        self.assertFalse(changed['enabled'].any())
        self.assertTrue((changed['fallback_reason'] == readout.NO_CANDIDATE).all())
        self.assertEqual(changed['candidate_keys'].shape, (0, 6))
        self.assertTrue(np.isnan(changed['selected_interval']).all())
        np.testing.assert_array_equal(changed['new_states'], args[-1]['states'])
        corrupt = dict(args[-1], interval_count=args[-1]['interval_count'] + 1)
        with self.assertRaisesRegex(AssertionError, 'C149 archived interval_count differs'):
            readout.freeze_scene(*args[:3], corrupt)
