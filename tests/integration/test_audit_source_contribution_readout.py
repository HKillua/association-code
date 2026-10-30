"""Independent140 software audit, including deliberate saved-result corruption."""
import ast
import copy
from pathlib import Path
import tempfile
import unittest

import numpy as np

from tools.analysis import audit_source_contribution_readout as audit


class ContributionAuditUnitTests(unittest.TestCase):
    def test_response_and_weighted_fit(self):
        errors = np.array([[1000., 3000., 9000.]])
        row = dict(features=np.zeros((1, 3, 10)), labeled=np.ones((1, 3), bool),
                   errors_m=errors, sample_weight=np.array([[1., 2., 1.]]))
        model = audit.fit([row])
        expected = np.dot(audit.chord(errors).ravel(), [.25, .5, .25])
        self.assertAlmostEqual(model['beta'][0], expected)
        np.testing.assert_array_equal(model['beta'][1:], 0.)
        np.testing.assert_array_equal(model['std'], 1.)
        self.assertTrue(audit.fit([])['no_labels_fallback'])
        # Exact sphere diameter response at half-circumference; not radial squared.
        self.assertAlmostEqual(audit.chord(np.array([np.pi*audit.RADIUS_M]))[0], (2*audit.RADIUS_M/1000.)**2)

    def test_independent_k2_solution_and_actual_scope(self):
        states = np.zeros((2, 2, 4)); states[1, :, 1] = .01
        risk = np.zeros((2, 2)); available = np.ones((2, 2), bool); available[:, 1] = False
        result = audit.policies(risk, available, [0, 2], states)
        for name in audit.NEW:
            np.testing.assert_array_equal(result[name]['source_weights'][:, 0], [.5, .5])
            np.testing.assert_array_equal(result[name]['source_weights'][:, 1], 0.)
            self.assertTrue(np.isnan(result[name]['objective_km2'][0, 1]))
        self.assertLess(result[audit.NEW[2]]['objective_km2'][0, 0], 0.)
        self.assertEqual(result[audit.NEW[2]]['duality_gap_km2'][0, 0], 0.)
        self.assertEqual(result[audit.NEW[1]]['floor_hit'].sum(), 2)
        with self.assertRaisesRegex(ValueError, 'K<=2'):
            audit.policies(np.zeros((3, 1)), np.ones((3, 1), bool), [0, 3], np.zeros((3, 1, 4)))

    def test_import_tree_is_independent(self):
        for module in (audit, audit.prior, audit.original, audit.physical, audit.common, audit.core):
            for node in ast.walk(ast.parse(Path(module.__file__).read_text())):
                if isinstance(node, ast.ImportFrom):
                    self.assertFalse((node.module or '').startswith('association'))
                    if node.module == 'tools.analysis':
                        self.assertTrue(all(a.name.startswith('audit_') for a in node.names))
                elif isinstance(node, ast.Import):
                    self.assertTrue(all(not a.name.startswith('association') for a in node.names))


class ContributionAuditIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from tests.integration.test_source_contribution_readout_pilot import fixture
        from tools.analysis import source_contribution_readout_pilot as pilot
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name).resolve()
        cls.args = fixture(cls.root)
        pilot.run(cls.args)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def context(self):
        args = self.args
        state = audit.core.read(args.output/'run.json')
        paths = {key: Path(state[key]) for key in ('native_run', 'value_run', 'partition_run', 'source_risk_run')}
        return state, paths, audit.core.read(args.manifest)

    def test_complete_audit(self):
        args = self.args
        report = audit.audit(args.output, args.native_run, args.source_risk_run, args.manifest,
                             self.root/'independent140-audit.json')
        self.assertEqual(report['status'], 'COMPLETED', report.get('error'))
        self.assertEqual(report['audited_model_count'], 13)
        self.assertEqual(report['metric_scope']['T_core']['pairs_per_scene'], 15)
        self.assertEqual(report['metric_scope']['V_select']['pairs_per_scene'], 78)
        for scope in report['metric_scope'].values():
            self.assertEqual(scope['primary_bootstrap_count'], 4)

    def test_chord_label_and_fit_corruption_rejected(self):
        state, paths, manifest = self.context(); args = self.args
        path = args.output/'private/training-0.npz'; original = audit.load(path)
        changed = {k: v.copy() for k, v in original.items()}
        index = tuple(np.argwhere(changed['labeled'])[0]); changed['y_chord_km2'][index] += .1
        try:
            np.savez_compressed(path, **changed)
            with self.assertRaisesRegex(ValueError, 'chord label'):
                audit.inspect_training(args.output, paths, manifest, state['training_scene_ids'], audit.core.Checks())
        finally:
            np.savez_compressed(path, **original)
        rows = [audit.load(args.output/'private'/f'training-{sid}.npz') for sid in state['training_scene_ids']]
        models = audit.load(args.output/'private/models.npz'); models['full__beta'][0] += 1.
        with self.assertRaisesRegex(ValueError, 'independent beta'):
            audit.inspect_fit(models, 'full', rows, audit.load(args.source_risk_run/'private/models.npz'), audit.core.Checks())

    def test_score_weight_objective_and_metric_corruption_rejected(self):
        state, paths, manifest = self.context(); args = self.args
        _, predictions, full = audit.inspect_training(args.output, paths, manifest, state['training_scene_ids'], audit.core.Checks())
        row = audit.core.read(args.output/'T-scene-summary.json.gz')[0]
        path = args.output/'private'/f'policy-T_core-{row["scene_id"]}.npz'; original = audit.load(path)
        for key, index in (('risk_raw', tuple(np.argwhere(original['available'])[0])),
                           (audit.NEW[2]+'__source_weights', (0, 1)),
                           (audit.NEW[2]+'__objective_km2', (0, 1))):
            with self.subTest(key=key):
                changed = {k: v.copy() for k, v in original.items()}; changed[key][index] += .1
                try:
                    np.savez_compressed(path, **changed)
                    with self.assertRaises(ValueError):
                        audit.inspect_policy(args.output, paths, 'T_core', row, predictions, full, audit.core.Checks())
                finally:
                    np.savez_compressed(path, **original)
        saved = audit.core.artifacts(args.output/'private'/f'states-T_core-{row["scene_id"]}.npz',
                                     audit.T_METHODS, template=audit.T_METHODS[0])
        altered = copy.deepcopy(row)
        altered['methods'][audit.NEW[2]]['position']['all']['mae'] += 1.
        with self.assertRaisesRegex(ValueError, 'mae'):
            audit.core.inspect_scene(saved, altered, audit.T_METHODS, audit.core.Checks(), expected_split='T_core')


if __name__ == '__main__':
    unittest.main()
