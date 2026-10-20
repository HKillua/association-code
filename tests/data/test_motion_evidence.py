"""Motion evidence preserves native predictions and excludes reference inputs."""
import unittest
from unittest.mock import patch
import zipfile

import numpy as np

from association.data.archive import DataError
from association.data.motion_evidence import (
    EARTH_RADIUS_M, build_motion_evidence, prepare_motion_scenes)
from tests.fixtures.native_zip import make_native_zip_fixture


class MotionEvidenceTests(unittest.TestCase):
    def test_two_point_short_track_units_padding_and_exact_observation(self):
        times = np.array([[11., 12., np.nan], [0., 10., 20.]])
        values = np.array([[[20., 120., 4., 350.], [20.001, 120.002, 6., 10.],
                            [np.nan]*4],
                           [[0., 0., 2., 0.], [0., .001, 2., 0.], [0., .002, 2., 0.]]])
        visible = np.array([[True, True, False], [True, True, True]])
        query = np.array([0., 10., 20.])
        result = build_motion_evidence(values, times, visible, query, ['9001', '9002'])
        self.assertEqual(result['states'].dtype, np.float64)
        np.testing.assert_array_equal(result['native_count'], [2, 3])
        np.testing.assert_array_equal(result['span_seconds'], [1., 20.])
        self.assertFalse(result['support'][0].any())
        self.assertFalse(result['exact_observation'][0].any())
        self.assertTrue(result['exact_observation'][1].all())
        np.testing.assert_allclose(result['states'][0, :, 0], 20.+(query-11.)*.001)
        np.testing.assert_array_equal(result['states'][0, :, 2:], [[4., 350.], [4., 350.], [6., 10.]])
        expected_scale = EARTH_RADIUS_M*np.pi/180.
        np.testing.assert_allclose(result['ols_velocity_mps'][0],
            expected_scale*np.array([.001, .002*np.cos(np.deg2rad(20.0005))]))
        np.testing.assert_allclose(result['ols_velocity_mps'], result['endpoint_velocity_mps'], atol=1e-9)
        np.testing.assert_allclose(result['fit_rmse_m'], 0., atol=1e-8)
        # Source names describe evidence, but cannot change states or velocities.
        renamed = build_motion_evidence(values, times, visible, query, ['x', 'y'])
        for name in result:
            if name != 'sources':
                np.testing.assert_array_equal(result[name], renamed[name])

    def test_dateline_unwrap_fit_residual_and_interpolated_support(self):
        values = np.array([[[10., 179., 4., 350.],
                            [10.002, 180., 6., 0.],
                            [10., -179., 8., 10.]]])
        times = np.array([[0., 10., 20.]])
        query = np.array([-10., 0., 5., 10., 20., 30.])
        result = build_motion_evidence(values, times, np.ones((1, 3), bool), query, ['9001'])
        np.testing.assert_allclose(result['states'][0, :, 1], [178., 179., 179.5, 180., -179., -178.])
        np.testing.assert_array_equal(result['support'][0], [False, True, True, True, True, False])
        np.testing.assert_array_equal(result['exact_observation'][0], [False, True, False, True, True, False])
        self.assertEqual(result['states'][0, 2, 3], 355.)
        # Independent OLS with intercept, in the declared fixed native-mean-latitude plane.
        xy = values[0, :, :2].copy()
        xy[:, 1] = [179., 180., 181.]
        design = np.column_stack([np.ones(3), times[0]])
        fitted = design @ np.linalg.lstsq(design, xy, rcond=None)[0]
        scale = EARTH_RADIUS_M*np.pi/180.*np.array([1., np.cos(np.deg2rad(xy[:, 0].mean()))])
        expected = np.sqrt(np.mean(np.sum(((xy-fitted)*scale)**2, axis=-1)))
        self.assertAlmostEqual(result['fit_rmse_m'][0], expected, places=7)

    def test_loader_retains_empty_scene_all_pairs_and_never_reads_reference_or_test(self):
        fixture = make_native_zip_fixture(self)
        original_read = zipfile.ZipFile.read
        members = []
        def read(archive, member, *args, **kwargs):
            name = member.filename if hasattr(member, 'filename') else str(member)
            members.append(name)
            if '真实场景' in name or name.endswith('-3.csv'):
                raise AssertionError('reference and sealed test must not be read')
            return original_read(archive, member, *args, **kwargs)
        checks = []
        with patch.object(zipfile.ZipFile, 'read', read):
            scenes = prepare_motion_scenes(fixture.path, fixture.manifest, 'V_select',
                                          check=lambda: checks.append(True))
        self.assertEqual([s['scene_id'] for s in scenes], ['2', '4'])
        self.assertEqual(len(checks), 4)
        self.assertEqual(len(scenes[0]['pairs']), 6)
        self.assertEqual(scenes[1]['motion']['states'].shape, (0, 64, 4))
        np.testing.assert_array_equal(scenes[1]['motion']['query_seconds'], np.linspace(5., 25., 64))
        self.assertEqual(set(scenes[0]), {
            'scene_id', 'component_id', 'split', 'node_ids', 'identities', 'sources',
            'pairs', 'labels', 'motion'})
        self.assertFalse(any('真实场景' in name for name in members))
        with patch('association.data.motion_evidence.MTADArchive',
                   side_effect=AssertionError('split rejected before I/O')):
            with self.assertRaises(DataError):
                prepare_motion_scenes(fixture.path, fixture.manifest, 'T_test')


if __name__ == '__main__':
    unittest.main()
