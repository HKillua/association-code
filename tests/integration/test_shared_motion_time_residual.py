"""122 mathematical/schema checks. Fixtures are software evidence, not MTAD results."""
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from tests.integration.test_query_contribution_control import cache_fixture, scene_fixture
from tools.analysis import shared_component_control as shared
from tools.analysis import shared_motion_time_residual as p


def motion_fixture(count=12, tau=2.5, b=(120., -60.)):
    query = np.arange(64, dtype=float)*20.
    velocity = np.column_stack((np.arange(count)*.3+1., (np.arange(count) % 4)*.7-1.))
    states = np.zeros((2*count, 64, 4))
    scale = p.RADIUS*np.pi/180.
    for gid, v in enumerate(velocity):
        low = np.array([0., 120.])+query[:, None]*v/scale
        high = low+(np.asarray(b)-v*tau)/scale
        states[2*gid, :, :2], states[2*gid+1, :, :2] = low, high
    physical = dict(native_states=states, support=np.ones(states.shape[:2], bool),
                    query_seconds=query, sources=np.tile(np.array(['a', 'b']), count),
                    node_ids=np.array([f'n-{i}' for i in range(2*count)]))
    membership = np.repeat(np.arange(count), 2)
    calibration = dict(source_order=np.array(['a', 'b']), group_fold=np.arange(count) % 3)
    return physical, membership, calibration


def cached118(root):
    args, meta = cache_fixture(root)
    shared_run = root/p.SHARED_RUN
    (shared_run/'scenes').mkdir(parents=True)
    (shared_run/'private').mkdir()
    for sid in meta['ids']:
        fragment, memberships, retained, public, _ = scene_fixture(sid)
        row, _, artifacts, calibration = shared.analyze_scene(fragment, memberships, retained, public)
        p.old.frozen.write(shared_run/'scenes'/f'scene-{sid}.json.gz', row)
        payload = {name+'_'+key: value for name, artifact in artifacts.items() for key, value in artifact.items()}
        payload.update(calibration)
        np.savez_compressed(shared_run/'private'/f'states-{sid}.npz', **payload)
    run = dict(status='COMPLETED', idea_id=p.ORIGINAL_IDEA, plan_version='1.0',
        source_system=str(args.source_system), partition_run=str(args.partition_run),
        scene_ids=meta['ids'], completed_scenes=len(meta['ids']), methods=list(p.SHARED_METHODS))
    summary = dict(status='COMPLETED', idea_id=p.ORIGINAL_IDEA, plan_version='1.0',
        scene_ids=meta['ids'], fixed_R_population_invariants_verified=True, no_expansion=True)
    p.old.frozen.write(shared_run/'run.json', run)
    p.old.frozen.write(shared_run/'summary.json', summary)
    return SimpleNamespace(source_system=args.source_system, partition_run=args.partition_run,
        shared_run=shared_run, output=root/'motion-residual-out', manifest=args.manifest, cpu_threads=1), meta


class MotionResidualTests(unittest.TestCase):
    def test_known_spatial_and_time_coefficients_and_degenerate_constant_velocity(self):
        v = np.array([[1., 0.], [0., 1.], [2., 2.], [-1., 3.]])
        bias, tau = np.array([120., -60.]), 2.5
        d = bias-v*tau
        model = p.fit_models(d, v)['JOINT_LS']
        np.testing.assert_allclose(model['b'], bias, atol=1e-12)
        self.assertAlmostEqual(model['tau'], tau, places=12)
        self.assertEqual((model['rank'], model['status']), (3, 'fitted'))
        v = np.tile([2., 3.], (6, 1))
        d = bias-v*tau
        fits = p.fit_models(d, v)
        self.assertEqual(fits['JOINT_LS']['status'], 'static_fallback')
        self.assertIsNone(fits['JOINT_LS']['tau'])
        np.testing.assert_array_equal(fits['JOINT_LS']['b'], fits['STATIC_LS']['b'])
        # TIME is identifiable only under its additional assumption b=0.
        self.assertEqual(fits['TIME_LS']['status'], 'fitted')
        unavailable = p.fit_models(np.empty((0, 2)), np.empty((0, 2)))
        self.assertTrue(all(m['status'] == 'unavailable' for m in unavailable.values()))

    def test_query_seconds_short_arc_and_three_point_support_are_used(self):
        query = np.arange(64)*20.
        scale = p.RADIUS*np.pi/180.
        states = np.zeros((2, 64, 4))
        states[0, :, 1] = (179.9+30.*query/scale+180.) % 360.-180.
        states[1, :, 1] = (states[0, :, 1]+20./scale+180.) % 360.-180.
        support = np.ones((2, 64), bool)
        support[:, 0] = False
        support[1, 10] = False
        rows = p.group_motion_rows(states, support, query, np.array(['a', 'b']), [[0, 1]])
        np.testing.assert_allclose(rows['velocity_degrees_per_second']*scale, [[0., 30.]], atol=1e-7)
        np.testing.assert_allclose(rows['delta_degrees']*scale, [[0., 20.]], atol=1e-7)
        self.assertEqual(rows['shared_pair_query_count'][0], 62)
        self.assertEqual(rows['derivative_pair_query_count'][0], 58)
        for bad in (query[::-1], np.zeros(64)):
            with self.assertRaises(ValueError):
                p.group_motion_rows(states, support, bad, np.array(['a', 'b']), [[0, 1]])

    def test_heldout_known_signal_and_no_current_fold_parameter_or_chart_access(self):
        physical, membership, calibration = motion_fixture()
        original = physical['native_states'].copy()
        report, private, errors = p.analyze_scene(physical, membership, calibration, scene_id='1', component_id='c')
        self.assertEqual(report['evaluated_group_count'], 12)
        self.assertLess(report['methods']['JOINT_LS']['mean_m'], 1e-6)
        self.assertGreater(report['methods']['STATIC_LS']['mean_m'], 1.)
        self.assertGreater(report['methods']['JOINT_PERMUTED_LS']['mean_m'], 1.)
        np.testing.assert_allclose(private['fold_parameters_JOINT_LS'][:, 2], 2.5, atol=1e-6)
        np.testing.assert_array_equal(physical['native_states'], original)
        changed = {key: value.copy() for key, value in physical.items()}
        for node in np.flatnonzero((membership % 3) == 0):
            changed['native_states'][node, :, 0] += np.linspace(0., .2, 64)
        _, updated, _ = p.analyze_scene(changed, membership, calibration, scene_id='1', component_id='c')
        self.assertEqual(private['fold_chart_latitude'][0], updated['fold_chart_latitude'][0])
        for name in p.METHODS:
            np.testing.assert_array_equal(private['fold_parameters_'+name][0], updated['fold_parameters_'+name][0])
        _, replay, _ = p.analyze_scene(physical, membership, calibration, scene_id='1', component_id='c')
        for fold in range(3):
            training, heldout = np.flatnonzero(membership[::2] % 3 != fold), np.flatnonzero(membership[::2] % 3 == fold)
            for part, expected in (('training', training), ('heldout', heldout)):
                key = f'fold_{fold}_{part}_velocity_groups'
                np.testing.assert_array_equal(np.sort(private[key]), expected)
                np.testing.assert_array_equal(private[key], replay[key])

    def test_complete_undefined_groups_and_public_privacy(self):
        physical, membership, calibration = motion_fixture()
        physical['support'][:2] = False
        physical['support'][2:4] = False
        physical['support'][2:4, 20] = True
        # Extra private/truth fields must have no path to fit or selection.
        physical.update(target=object(), identities=object(), target_valid=object())
        report, private, errors = p.analyze_scene(physical, membership, calibration, scene_id='1', component_id='c')
        self.assertEqual((report['group_count'], report['evaluated_group_count'], report['undefined_group_count']), (12, 10, 2))
        self.assertEqual(report['no_shared_support_group_count'], 1)
        self.assertEqual(report['shared_without_derivative_group_count'], 1)
        for name in p.METHODS:
            self.assertTrue(np.isnan(errors[name][:2]).all())
            self.assertEqual(report['methods'][name]['group_count'], 10)
        public_text = str(report)
        for forbidden in ('fold_chart_latitude', 'group_delta_degrees', 'group_velocity_degrees', "'tau':", "'b':", 'node_ids', 'identities'):
            self.assertNotIn(forbidden, public_text)
        self.assertIn('fold_parameters_JOINT_LS', private)
        aggregate = p.aggregate([report, report], [errors, errors])
        self.assertEqual((aggregate['group_count'], aggregate['evaluated_group_count'], aggregate['undefined_group_count']), (24, 20, 4))

    def test_original118_metadata_and_real_schema_runner_failure_preservation(self):
        for failure in (False, True):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as temporary:
                args, meta = cached118(Path(temporary).resolve())
                with patch.object(p.old, 'metadata', return_value=meta):
                    self.assertEqual(p.metadata(args.source_system, args.partition_run, args.shared_run, {})['ids'], ['1', '2'])
                    original = p.old.frozen.read(args.shared_run/'run.json')
                    p.old.frozen.write(args.shared_run/'run.json', dict(original, methods=['wrong']))
                    with self.assertRaisesRegex(ValueError, 'original118'):
                        p.metadata(args.source_system, args.partition_run, args.shared_run, {})
                    p.old.frozen.write(args.shared_run/'run.json', original)
                # These object arrays cannot be read with allow_pickle=False.
                # A successful run proves that the physical allowlist is used.
                for sid in meta['ids']:
                    file = args.source_system/'predictions'/f'states-{sid}.npz'
                    with np.load(file, allow_pickle=False) as saved:
                        values = {key: saved[key] for key in p.PHYSICAL_KEYS}
                    values.update(target=np.array([{'forbidden': 'reference'}], object),
                                  identities=np.array([{'forbidden': 'identity'}], object))
                    np.savez_compressed(file, **values)
                if failure:
                    file = args.shared_run/'private/states-2.npz'
                    with np.load(file, allow_pickle=False) as saved:
                        values = {key: saved[key] for key in saved.files}
                    values['WS_MEAN_node_group_indices'] = np.array([0, 1, 1])
                    np.savez_compressed(file, **values)
                with patch.object(p, 'metadata', return_value=meta):
                    if failure:
                        with self.assertRaisesRegex(ValueError, '118 WS membership'):
                            p.run(args)
                        state = p.old.frozen.read(args.output/'run.json')
                        self.assertEqual((state['status'], state['completed_scenes']), ('FAILED', 1))
                        self.assertTrue((args.output/'private/residuals-1.npz').exists())
                        self.assertFalse((args.output/'summary.json').exists())
                    else:
                        result = p.run(args)
                        self.assertEqual(result['scene_ids'], ['1', '2'])
                        self.assertFalse(result['reference_array_fields_loaded'])
                        self.assertFalse(result['identity_array_fields_loaded'])
                        self.assertFalse(result['predictions_changed'])
                        self.assertEqual(result['group_count'], 2)
                        self.assertEqual(result['undefined_group_count'], 2)
                        self.assertEqual(result['evaluated_group_count'], 0)
                        with np.load(args.output/'private/residuals-1.npz', allow_pickle=False) as private:
                            self.assertIn('fold_parameters_JOINT_LS', private.files)
                            self.assertNotIn('aligned_truth', private.files)
                        with self.assertRaises(FileExistsError):
                            p.run(args)


if __name__ == '__main__':
    unittest.main()
