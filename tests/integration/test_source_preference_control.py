"""121 behavior checks; synthetic fixtures are software evidence only."""
from copy import deepcopy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from tests.integration.test_query_contribution_control import cache_fixture, scene_fixture
from tools.analysis import shared_component_control as shared
from tools.analysis import source_preference_control as p


def saved_scene(sid='1'):
    fragment, memberships, retained, public, _ = scene_fixture(sid)
    row, _, artifacts, calibration = shared.analyze_scene(fragment, memberships, retained, public)
    return fragment, memberships['WS'], artifacts, calibration, row


def cached118(root):
    old_args, meta = cache_fixture(root)
    shared_run = root/p.SHARED_RUN
    (shared_run/'scenes').mkdir(parents=True)
    (shared_run/'private').mkdir()
    for sid in meta['ids']:
        _, _, artifacts, calibration, row = saved_scene(sid)
        p.old.frozen.write(shared_run/'scenes'/f'scene-{sid}.json.gz', row)
        payload = {name+'_'+key: value for name, artifact in artifacts.items() for key, value in artifact.items()}
        payload.update(calibration)
        np.savez_compressed(shared_run/'private'/f'states-{sid}.npz', **payload)
    run = dict(status='COMPLETED', idea_id=p.ORIGINAL_IDEA, plan_version='1.0',
        source_system=str(old_args.source_system), partition_run=str(old_args.partition_run),
        scene_ids=meta['ids'], completed_scenes=len(meta['ids']), methods=list(p.LEGACY_METHODS))
    summary = dict(status='COMPLETED', idea_id=p.ORIGINAL_IDEA, plan_version='1.0',
        scene_ids=meta['ids'], fixed_R_population_invariants_verified=True, no_expansion=True)
    p.old.frozen.write(shared_run/'run.json', run)
    p.old.frozen.write(shared_run/'summary.json', summary)
    return SimpleNamespace(source_system=old_args.source_system, partition_run=old_args.partition_run,
        shared_run=shared_run, output=root/'source-preference-out', manifest=old_args.manifest, cpu_threads=1), meta


class SourcePreferenceTests(unittest.TestCase):
    def test_priority_each_query_source_mean_and_corresponding_anchor_fallback(self):
        states = np.zeros((3, 64, 4))
        states[:, :, 0] = np.array([1., 2., 4.])[:, None]
        states[..., 1] = 120.
        states[..., 2:] = [7., 359.]
        sources = np.asarray(['b', 'a', 'b'])
        support = np.ones((3, 64), bool)
        support[:, 0] = False
        support[1, 2] = False
        support[[0, 2], 3] = False
        calibration = dict(source_order=np.array(['a', 'b']), group_fold=np.array([0]),
                           fold_center_ne_m=np.array([[1000., 0.], [0., 0.], [0., 0.]]))
        mean = np.ones((1, 64, 4)); mean[..., 2:] = [9., 45.]
        original = states.copy()
        outputs, audit, weights = p.construct_predictions(states, np.zeros((3, 64)), support,
            sources, [[0, 1, 2]], calibration, mean)
        low, fixed = (outputs[name] for name in p.NEW_METHODS)
        self.assertEqual(low[0, 1, 0], 2.)
        self.assertEqual(low[0, 2, 0], 2.5)
        self.assertEqual(low[0, 3, 0], 2.)
        np.testing.assert_array_equal(low[0, 0, :2], states[0, 0, :2])
        moved = p.spherical_shift(states, [-1000., 0.])
        np.testing.assert_array_equal(fixed[0, 0, :2], moved[0, 0, :2])
        np.testing.assert_array_equal(fixed[0, 1, :2], states[1, 1, :2])
        np.testing.assert_allclose(fixed[0, 2, 0], moved[[0, 2], 2, 0].mean(), atol=1e-14)
        for output in outputs.values():
            np.testing.assert_array_equal(output[..., 2:], mean[..., 2:])
        self.assertEqual(audit['supported_query_types'],
                         dict(both_sources=61, low_only=1, high_only=1, no_support=1))
        np.testing.assert_array_equal(states, original)
        np.testing.assert_array_equal(weights['query_weights'][:, 2], [.5, 0., .5])

    def test_single_source_date_line_and_saved_fold_order_rejection(self):
        states = np.zeros((2, 64, 4))
        states[:, :, 1] = np.array([179., -179.])[:, None]
        calibration = dict(source_order=np.array(['only']), group_fold=np.array([0]),
                           fold_center_ne_m=np.ones((3, 2))*999.)
        output, audit, _ = p.construct_predictions(states, np.zeros((2, 64)), np.ones((2, 64), bool),
            np.array(['only', 'only']), [[0, 1]], calibration, np.zeros((1, 64, 4)))
        self.assertTrue(np.all(np.abs(output[p.NEW_METHODS[0]][..., 1]) > 179.99))
        np.testing.assert_array_equal(output[p.NEW_METHODS[0]], output[p.NEW_METHODS[1]])
        self.assertEqual(audit['supported_query_types']['low_only'], 64)
        for bad in (dict(calibration, source_order=np.array(['other'])),
                    dict(calibration, group_fold=np.array([1])),
                    dict(calibration, fold_center_ne_m=np.full((3, 2), np.nan))):
            with self.subTest(bad=list(bad)), self.assertRaises(ValueError):
                p.construct_predictions(states, np.zeros((2, 64)), np.ones((2, 64), bool),
                    np.array(['only', 'only']), [[0, 1]], bad, np.zeros((1, 64, 4)))

    def test_imported_artifacts_never_recompute_legacy_predictions_or_fit_centers(self):
        args = saved_scene()
        with patch.object(shared, 'heldout_centers', side_effect=AssertionError('no refit')), \
                patch.object(shared, 'construct_predictions', side_effect=AssertionError('no old arms')), \
                patch.object(p.old, 'ordinary_fusion', side_effect=AssertionError('no ordinary recalculation')):
            row, arrays, artifacts, private = p.analyze_scene(*args)
        self.assertEqual(len(row['methods']), 10)
        self.assertEqual(len(row['pairwise']), 45)
        for name in p.LEGACY_METHODS:
            for key in args[2][name]:
                np.testing.assert_array_equal(artifacts[name][key], args[2][name][key])
        for name in p.METHODS:
            self.assertEqual(row['methods'][name]['population'], row['methods']['WS_MEAN']['population'])
            np.testing.assert_array_equal(artifacts[name]['aligned_prediction_valid'],
                                          artifacts['WS_MEAN']['aligned_prediction_valid'])
        self.assertIn('predicted_group_both_sources', row['methods']['WS_MEAN']['fixed_target_strata'])

    def test_saved_prediction_truth_membership_and_public_binding_fail(self):
        for kind in ('prediction', 'reference', 'membership', 'public'):
            fragment, membership, retained, calibration, public = saved_scene()
            if kind == 'prediction':
                retained['WS_FIXED_MEAN']['aligned_prediction'][0, 1, 0] += .01
            elif kind == 'reference':
                fragment['target'][0, 1, 0] += .01
            elif kind == 'membership':
                membership = np.array([0, 1, 1])
            else:
                public['methods']['WS_HUBER']['position']['all']['p99'] += .1
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                p.analyze_scene(fragment, membership, retained, calibration, public)

    def test_metadata_and_full_cached_run_preserve_partial_failure(self):
        for failure in (False, True):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as temporary:
                args, meta = cached118(Path(temporary).resolve())
                with patch.object(p.old, 'metadata', return_value=meta):
                    self.assertEqual(p.metadata(args.source_system, args.partition_run, args.shared_run, {})['ids'], ['1', '2'])
                    original = p.old.frozen.read(args.shared_run/'run.json')
                    p.old.frozen.write(args.shared_run/'run.json', dict(original, status='FAILED'))
                    with self.assertRaisesRegex(ValueError, 'original118'):
                        p.metadata(args.source_system, args.partition_run, args.shared_run, {})
                    p.old.frozen.write(args.shared_run/'run.json', original)
                if failure:
                    file = args.shared_run/'private/states-2.npz'
                    with np.load(file, allow_pickle=False) as saved:
                        payload = {key: saved[key] for key in saved.files}
                    payload['WS_FIXED_HUBER_group_states'][0, 1, 0] += .1
                    np.savez_compressed(file, **payload)
                with patch.object(p, 'metadata', return_value=meta):
                    if failure:
                        with self.assertRaisesRegex(ValueError, '118 exact'):
                            p.run(args)
                        state = p.old.frozen.read(args.output/'run.json')
                        self.assertEqual((state['status'], state['completed_scenes']), ('FAILED', 1))
                        self.assertTrue((args.output/'private/states-1.npz').exists())
                        self.assertFalse((args.output/'summary.json').exists())
                    else:
                        result = p.run(args)
                        self.assertEqual(result['scene_ids'], ['1', '2'])
                        self.assertEqual((result['imported_scene_method_count'], result['new_scene_method_count']), (16, 4))
                        self.assertFalse(result['centers_refitted'])
                        self.assertFalse(result['huber_recomputed'])
                        self.assertTrue(result['original118_artifacts_exact'])
                        self.assertEqual(result['methods']['WS_MEAN']['position']['intersection_count'], 128)
                        with self.assertRaises(FileExistsError):
                            p.run(args)


if __name__ == '__main__':
    unittest.main()
