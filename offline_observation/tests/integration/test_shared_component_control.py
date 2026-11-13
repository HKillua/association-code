"""Necessary behavior checks; synthetic caches do not count as MTAD evidence."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from tests.integration.test_query_contribution_control import cache_fixture, scene_fixture
from tools.analysis import shared_component_control as p


def offset_fixture():
    states = np.zeros((12, 64, 4))
    groups = [[2*g, 2*g+1] for g in range(6)]
    for gid, (a, b) in enumerate(groups):
        states[a, :, 0] = 30.+gid*.01+np.arange(64)*.00001
        states[a, :, 1] = 120.+gid*.01
        states[b] = states[a]
        states[b, :, :2] += [.01, .02]
    return states, np.zeros((12, 64)), np.ones((12, 64), bool), np.asarray(['a', 'b']*6), groups


class SharedComponentControlTests(unittest.TestCase):
    def test_crossfit_excludes_all_current_fold_inputs_including_chart(self):
        states, near, support, sources, groups = offset_fixture()
        centers, _, audit, private = p.heldout_centers(states, support, sources, groups)
        self.assertEqual(audit['fold_training_group_counts'], [4, 4, 4])
        np.testing.assert_allclose(centers[:, 0], p.RADIUS*np.deg2rad(.01), atol=1e-7)
        changed = states.copy()
        # Every node in fold 0 changes, both its chart latitude and source gap.
        for gid in (0, 3):
            changed[groups[gid], :, 0] += 5.
            changed[groups[gid][1], :, 1] += 2.
        other, _, _, other_private = p.heldout_centers(changed, support, sources, groups)
        np.testing.assert_array_equal(centers[0], other[0])
        self.assertEqual(private['fold_chart_latitude'][0], other_private['fold_chart_latitude'][0])

    def test_spherical_shift_has_correct_length_and_preserves_zero_and_other_attributes(self):
        state = np.asarray([[[0., 179.999, 7., 359.]]])
        np.testing.assert_array_equal(p.spherical_shift(state, [0., 0.]), state)
        moved = p.spherical_shift(state, [0., 1000.])
        self.assertLess(moved[0, 0, 1], -179.)
        np.testing.assert_allclose(p.attribute_errors(moved, state)['position'], 1000., atol=1e-6)
        np.testing.assert_array_equal(moved[..., 2:], state[..., 2:])

    def test_insufficient_fit_does_not_report_equatorial_relative_residual(self):
        states, _, support, sources, groups = offset_fixture()
        centers, _, audit, _ = p.heldout_centers(states[:2], support[:2], sources[:2], groups[:1])
        np.testing.assert_array_equal(centers, np.zeros((3, 2)))
        self.assertIsNone(audit['input_only_group_equal_relative_residual_before'])
        self.assertIsNone(audit['input_only_group_equal_relative_residual_after'])
        self.assertEqual(audit['relative_residual_defined_group_count'], 0)
        self.assertEqual(audit['relative_residual_undefined_group_count'], 1)
        self.assertEqual(audit['relative_residual_undefined_pair_queries'], 64)

    def test_huber_reduces_one_extreme_member_without_changing_speed_or_heading(self):
        states = np.zeros((3, 64, 4))
        states[..., 1] = np.array([0., .0001, 1.])[:, None]
        states[..., 2] = np.array([1., 3., 8.])[:, None]
        states[..., 3] = np.array([359., 1., 90.])[:, None]
        support = np.ones((3, 64), bool)
        near = np.zeros((3, 64))
        original = states.copy()
        result, weights = p.reweighted_mean(states, near, support, [[0, 1, 2]])
        mean = p.ordinary_fusion(states, near, support, [[0, 1, 2]])['predictions']['mean_supported'][[0]]
        self.assertLess(abs(result[0, 10, 1]), abs(mean[0, 10, 1])/10.)
        self.assertLess(weights[2, 10], weights[0, 10])
        np.testing.assert_array_equal(result[..., 2:], mean[..., 2:])
        np.testing.assert_array_equal(states, original)

    def test_single_source_and_no_fit_fallback_keep_mean_anchor_and_empty_support(self):
        fragment, memberships, retained, public, _ = scene_fixture()
        groups = p.old.groups_from_membership(memberships['WS'], len(fragment['sources']))
        prediction, audit, _ = p.construct_predictions(fragment['native_states'], fragment['nearest_seconds'],
            fragment['support'], np.asarray(['same']*len(fragment['sources'])), groups)
        self.assertEqual(audit['fold_training_group_counts'], [0, 0, 0])
        for name in ('WS_FIXED_MEAN', 'WS_SYMMETRIC_MEAN'):
            np.testing.assert_array_equal(prediction[name], prediction['WS_MEAN'])
        for name in ('WS_HUBER', 'WS_FIXED_HUBER', 'WS_SYMMETRIC_HUBER'):
            np.testing.assert_array_equal(prediction[name][:, 0], prediction['WS_ANCHOR'][:, 0])

    def test_complete_evaluation_keeps_fixed_population_and_reference_only_affects_evaluation(self):
        fragment, memberships, retained, public, _ = scene_fixture()
        row, arrays, artifacts, private = p.analyze_scene(fragment, memberships, retained, public)
        self.assertEqual(len(row['pairwise']), 22)
        for name in p.METHODS:
            self.assertEqual(row['methods'][name]['population'], row['methods']['WS_MEAN']['population'])
            np.testing.assert_array_equal(artifacts[name]['aligned_prediction_valid'],
                                          artifacts['WS_MEAN']['aligned_prediction_valid'])
        retained['WS']['group_states'] = retained['WS']['group_states'].copy()
        retained['WS']['group_states'][0, 1, 0] += 1e-12
        with self.assertRaisesRegex(ValueError, '115 exact WS Mean'):
            p.analyze_scene(fragment, memberships, retained, public)

    def test_complete_cached_run_and_failure_preservation(self):
        for failure in (False, True):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as temporary:
                args, meta = cache_fixture(Path(temporary))
                if failure:
                    path = args.control_run/'private/states-2.npz'
                    with np.load(path, allow_pickle=False) as saved:
                        changed = {key: saved[key] for key in saved.files}
                    changed['WS_group_states'][0, 1, 0] += 1e-12
                    np.savez_compressed(path, **changed)
                with patch.object(p.qc, 'metadata', return_value=meta):
                    if failure:
                        with self.assertRaisesRegex(ValueError, '115 exact WS Mean'):
                            p.run(args)
                        state = p.old.frozen.read(args.output/'run.json')
                        self.assertEqual((state['status'], state['completed_scenes']), ('FAILED', 1))
                        self.assertTrue((args.output/'private/states-1.npz').exists())
                        self.assertFalse((args.output/'summary.json').exists())
                    else:
                        result = p.run(args)
                        self.assertEqual(result['scene_ids'], ['1', '2'])
                        self.assertEqual(result['methods']['WS_MEAN']['position']['intersection_count'], 128)
                        self.assertTrue(result['fixed_R_population_invariants_verified'])
                        with np.load(args.output/'private/states-1.npz', allow_pickle=False) as saved:
                            self.assertIn('fold_center_ne_m', saved.files)
                            self.assertIn('WS_FIXED_HUBER_query_weights', saved.files)
                        with self.assertRaises(FileExistsError):
                            p.run(args)


if __name__ == '__main__':
    unittest.main()
