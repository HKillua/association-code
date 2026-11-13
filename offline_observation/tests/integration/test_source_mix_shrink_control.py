"""124 controls: analytical fixtures establish software behavior only."""
from copy import deepcopy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from tests.integration.test_source_preference_control import cached118, saved_scene
from tools.analysis import shared_component_control as shared
from tools.analysis import source_preference_control as prior
from tools.analysis import source_mix_shrink_control as p
from tools.analysis import audit_shared_control_arrays as audit


def saved121(sid='1'):
    fragment, membership, legacy, calibration, public118 = saved_scene(sid)
    row, _, artifacts, _ = prior.analyze_scene(fragment, membership, legacy, calibration, public118)
    return fragment, membership, artifacts, calibration, row


class SourceMixShrinkTests(unittest.TestCase):
    def test_mass_center_factor_and_mutually_exclusive_support_domains(self):
        states = np.zeros((6, 64, 4))
        states[:, :, 0] = np.tile([2., 1.], 3)[:, None]
        states[..., 1] = 120.
        states[..., 2:] = [8., 359.]
        sources = np.array(['b', 'a']*3)
        groups = [[0, 1], [2, 3], [4, 5]]
        support = np.ones((6, 64), bool)
        support[:2, 0] = False
        support[1, 2] = False
        support[0, 3] = False
        calibration = dict(source_order=np.array(['a', 'b']), group_fold=np.arange(3),
            fold_center_ne_m=np.array([[600., 0.], [300., 0.], [900., 0.]]),
            group_fit_eligible=np.ones(3, bool), group_mean_delta_degrees=np.ones((3, 2)))
        mean = np.zeros((3, 64, 4)); mean[..., 2:] = [9., 45.]
        original = states.copy()
        output, audit, private = p.construct_predictions(states, np.zeros((6, 64)), support,
            sources, groups, calibration, mean)
        self.assertEqual(audit['fold_training_group_counts'], [2, 2, 2])
        np.testing.assert_allclose(audit['fold_center_scale'], np.repeat(1./3, 3))
        np.testing.assert_array_equal(private['soft75_query_weights'][:2, 1], [.25, .75])
        np.testing.assert_array_equal(private['equal_query_weights'][:2, 1], [.5, .5])
        np.testing.assert_array_equal(private['soft75_query_weights'][:2, 2], [1., 0.])
        np.testing.assert_array_equal(private['hard_query_weights'][:2, 3], [0., 1.])
        np.testing.assert_array_equal(private['soft75_query_weights'][:2, 0], [0., 0.])
        shifted = prior.spherical_shift(states[:2], [-600., 0.])
        shrunk = prior.spherical_shift(states[:2], [-200., 0.])
        self.assertAlmostEqual(output[p.NEW_METHODS[0]][0, 1, 0], 1.25, places=14)
        self.assertAlmostEqual(output[p.NEW_METHODS[1]][0, 1, 0],
                               .75*states[1, 1, 0]+.25*shifted[0, 1, 0], places=14)
        self.assertAlmostEqual(output[p.NEW_METHODS[4]][0, 1, 0],
                               .5*states[1, 1, 0]+.5*shifted[0, 1, 0], places=14)
        np.testing.assert_array_equal(output[p.NEW_METHODS[2]][0, 2, :2], shrunk[0, 2, :2])
        np.testing.assert_array_equal(output[p.NEW_METHODS[2]][0, 0, :2], shrunk[0, 0, :2])
        np.testing.assert_array_equal(output[p.NEW_METHODS[1]][0, 0, :2], shifted[0, 0, :2])
        np.testing.assert_array_equal(output[p.NEW_METHODS[0]][0, 0, :2], states[0, 0, :2])
        outside = ~private['both_sources']
        np.testing.assert_array_equal(output[p.NEW_METHODS[2]][outside], output[p.NEW_METHODS[3]][outside])
        for value in output.values():
            np.testing.assert_array_equal(value[..., 2:], mean[..., 2:])
        np.testing.assert_array_equal(states, original)
        self.assertFalse(audit['empirical_bayes_prior_fitted'])

    def test_single_source_date_line_and_bad_saved_eligibility_are_explicit(self):
        states = np.zeros((2, 64, 4)); states[:, :, 1] = np.array([179., -179.])[:, None]
        support = np.ones((2, 64), bool)
        calibration = dict(source_order=np.array(['only']), group_fold=np.array([0]),
            fold_center_ne_m=np.zeros((3, 2)), group_fit_eligible=np.array([False]),
            group_mean_delta_degrees=np.full((1, 2), np.nan))
        args = (states, np.zeros((2, 64)), support, np.array(['only', 'only']), [[0, 1]])
        output, audit, _ = p.construct_predictions(*args, calibration, np.zeros((1, 64, 4)))
        for name in p.NEW_METHODS:
            self.assertTrue(np.all(np.abs(output[name][..., 1]) > 179.99))
            np.testing.assert_array_equal(output[name], output[p.NEW_METHODS[0]])
        self.assertEqual(audit['fold_training_group_counts'], [0, 0, 0])
        for key, value in (('group_fit_eligible', np.array([True])),
                           ('group_fold', np.array([1])),
                           ('fold_center_ne_m', np.ones((3, 2)))):
            with self.subTest(key=key), self.assertRaises(ValueError):
                p.construct_predictions(*args, dict(calibration, **{key: value}), np.zeros((1, 64, 4)))

    def test_saved_ten_arms_import_exact_without_old_prediction_or_center_fit(self):
        args = saved121()
        with patch.object(shared, 'heldout_centers', side_effect=AssertionError('no refit')), \
                patch.object(shared, 'construct_predictions', side_effect=AssertionError('no old predictions')), \
                patch.object(prior, 'construct_predictions', side_effect=AssertionError('no old priority')):
            row, _, artifacts, _ = p.analyze_scene(*args)
        self.assertEqual((len(row['methods']), len(row['pairwise'])), (15, 105))
        for name in p.LEGACY_METHODS:
            for key in args[2][name]:
                np.testing.assert_array_equal(artifacts[name][key], args[2][name][key])
        for name in p.METHODS:
            self.assertEqual(row['methods'][name]['population'], row['methods']['WS_MEAN']['population'])
            np.testing.assert_array_equal(artifacts[name]['aligned_prediction_valid'],
                                          artifacts['WS_MEAN']['aligned_prediction_valid'])
        broken = list(deepcopy(args))
        broken[2]['WS_LOW_SOURCE_SUPPORTED']['aligned_prediction'][0, 1, 0] += .01
        with self.assertRaisesRegex(ValueError, '121 exact'):
            p.analyze_scene(*broken)

    def test_cached_cli_entrypoint_binding_and_partial_failure_preserve_outputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            legacy, meta = cached118(root)
            control = root/p.CONTROL_RUN
            (control/'private').mkdir(parents=True); (control/'scenes').mkdir()
            for sid in meta['ids']:
                _, _, artifacts, calibration, row = saved121(sid)
                payload = {name+'_'+key: value for name, artifact in artifacts.items() for key, value in artifact.items()}
                payload.update(calibration)
                np.savez_compressed(control/'private'/f'states-{sid}.npz', **payload)
                p.old.frozen.write(control/'scenes'/f'scene-{sid}.json.gz', row)
            full = root/p.FULL_RUN
            full.mkdir()
            for path, idea in ((control, prior.IDEA), (full, p.FULL_IDEA)):
                run = dict(status='COMPLETED', idea_id=idea, plan_version='1.0',
                    exit_code=0, source_system=str(legacy.source_system), partition_run=str(legacy.partition_run),
                    pilot_run=str(control),
                    scene_ids=meta['ids'], original32_ids=meta['ids'], completed_scenes=len(meta['ids']),
                    methods=list(p.LEGACY_METHODS))
                summary = dict(run, fixed_R_population_invariants_verified=True,
                               original118_artifacts_exact=True, no_expansion=True)
                p.old.frozen.write(path/'run.json', run); p.old.frozen.write(path/'summary.json', summary)
            args = SimpleNamespace(source_system=legacy.source_system, partition_run=legacy.partition_run,
                source_preference_run=control, source_full_run=full, manifest=legacy.manifest,
                output=root/'mix-out', cpu_threads=1)
            with patch.object(p.old, 'metadata', return_value=meta):
                self.assertEqual(p.metadata(args.source_system, args.partition_run, control, full, {})['ids'], meta['ids'])
                original = p.old.frozen.read(full/'run.json')
                p.old.frozen.write(full/'run.json', dict(original, status='FAILED'))
                with self.assertRaisesRegex(ValueError, 'completed original121'):
                    p.metadata(args.source_system, args.partition_run, control, full, {})
                p.old.frozen.write(full/'run.json', original)
            with patch.object(p, 'metadata', return_value=meta):
                result = p.run(args)
            self.assertEqual((result['imported_scene_method_count'], result['new_scene_method_count']), (20, 10))
            self.assertEqual(result['scene_ids'], meta['ids'])
            self.assertEqual(p.old.frozen.read(args.output/'run.json')['status'], 'COMPLETED')
            self.assertEqual(len(result['pairwise']), 105)
            verified = audit.audit_run(args.output, audit.Checks())
            self.assertEqual(verified['method_count'], 15)
            args.output = root/'mix-failed-out'
            # Corrupt only the second scene after creating legitimate saved data.
            file = control/'private/states-2.npz'
            with np.load(file, allow_pickle=False) as saved:
                payload = {key: saved[key] for key in saved.files}
            payload['WS_FIXED_LOW_SOURCE_SUPPORTED_aligned_prediction'][0, 1, 0] += .1
            np.savez_compressed(file, **payload)
            with patch.object(p, 'metadata', return_value=meta):
                with self.assertRaisesRegex(ValueError, '121 exact'):
                    p.run(args)
            self.assertEqual(p.old.frozen.read(args.output/'run.json')['status'], 'FAILED')
            self.assertEqual(p.old.frozen.read(args.output/'run.json')['completed_scenes'], 1)
            self.assertTrue((args.output/'private/states-1.npz').exists())
            self.assertFalse((args.output/'summary.json').exists())
            with self.assertRaises(FileExistsError):
                p.run(args)


if __name__ == '__main__':
    unittest.main()
