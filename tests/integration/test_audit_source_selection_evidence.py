"""139 auditor software checks; artificial caches are not scientific evidence."""
import ast
from pathlib import Path
import tempfile
import unittest

import numpy as np

from tools.analysis import audit_source_selection_evidence as audit


class SourceSelectionAuditUnitTests(unittest.TestCase):
    def test_weighted_meter_response_and_unpenalized_intercept(self):
        row = dict(features=np.zeros((1, 3, 10)), labeled=np.ones((1, 3), bool),
                   errors_m=np.array([[1000., 3000., 9000.]]), sample_weight=np.array([[1., 2., 1.]]))
        fitted = audit.meter_fit([row])
        self.assertAlmostEqual(fitted['beta'][0], 4.)
        np.testing.assert_array_equal(fitted['beta'][1:], 0.)
        np.testing.assert_array_equal(fitted['std'], 1.)
        self.assertEqual(fitted['training_label_count'], 3)
        empty = audit.meter_fit([])
        self.assertTrue(empty['no_labels_fallback'])
        np.testing.assert_array_equal(empty['beta'], np.zeros(66))

    def test_imports_are_independent(self):
        for module in (audit, audit.original, audit.physical, audit.common, audit.core):
            for node in ast.walk(ast.parse(Path(module.__file__).read_text())):
                if isinstance(node, ast.ImportFrom):
                    self.assertFalse((node.module or '').startswith('association'))
                    if node.module == 'tools.analysis':
                        self.assertTrue(all(alias.name.startswith('audit_') for alias in node.names))
                elif isinstance(node, ast.Import):
                    self.assertTrue(all(not alias.name.startswith('association') for alias in node.names))


class SourceSelectionAuditIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from tests.integration.test_source_selection_evidence_pilot import fixture
        from tools.analysis import source_selection_evidence_pilot as pilot
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
        paths = {k: Path(state[k]) for k in ('native_run', 'value_run', 'partition_run', 'source_risk_run', 'quality_run', 'mixture_run')}
        manifest = audit.core.read(args.manifest)
        return state, paths, manifest

    def test_full_saved_array_audit(self):
        args = self.args
        report = audit.audit(args.output, args.native_run, args.source_risk_run, args.quality_run,
                             args.mixture_run, args.manifest, self.root/'independent139-audit.json')
        self.assertEqual(report['status'], 'COMPLETED', report.get('error'))
        self.assertEqual(report['audited_model_count'], 13)
        self.assertEqual(report['metric_scope']['T_core']['pairs_per_scene'], 6)
        self.assertEqual(report['metric_scope']['V_select']['pairs_per_scene'], 66)
        self.assertEqual(report['metric_scope']['V_select']['primary_bootstrap_count'], 1)
        self.assertGreater(report['exact_checks'], 10000)

    def test_changed_meter_label_or_coefficient_is_rejected(self):
        state, paths, manifest = self.context()
        args = self.args
        path = args.output/'private/training-0.npz'
        original = audit.load(path)
        changed = {k: v.copy() for k, v in original.items()}
        index = tuple(np.argwhere(changed['labeled'])[0])
        changed['y_meter'][index] += .1
        try:
            np.savez_compressed(path, **changed)
            with self.assertRaisesRegex(ValueError, 'meter labels exactly km'):
                audit.inspect_training(args.output, args.source_risk_run, args.native_run, args.value_run,
                                       manifest, state['training_scene_ids'], audit.core.Checks())
        finally:
            np.savez_compressed(path, **original)
        rows = [audit.load(args.output/'private'/f'training-{sid}.npz') for sid in state['training_scene_ids']]
        model = audit.load(args.output/'private/models.npz')
        model['full__QUADRATIC__beta'][0] += 1.
        with self.assertRaisesRegex(ValueError, 'independent beta'):
            audit.inspect_fit(model, 'full', rows, audit.load(args.source_risk_run/'private/models.npz'), audit.core.Checks())

    def test_policy_and_diagnostic_tampering_is_rejected(self):
        state, paths, manifest = self.context()
        args = self.args
        training, predictions, full = audit.inspect_training(args.output, args.source_risk_run, args.native_run,
            args.value_run, manifest, state['training_scene_ids'], audit.core.Checks())
        row = audit.core.read(args.output/'T-scene-summary.json.gz')[0]
        path = args.output/'private'/f'policy-T_core-{row["scene_id"]}.npz'
        original = audit.load(path)
        changed = {k: v.copy() for k, v in original.items()}
        changed[audit.METER_T+'__source_weights'].flat[0] += .25
        try:
            np.savez_compressed(path, **changed)
            with self.assertRaisesRegex(ValueError, 'source_weights'):
                audit.inspect_policy(args.output, paths, 'T_core', row, predictions, full, audit.core.Checks())
        finally:
            np.savez_compressed(path, **original)
        _, policy, scores = audit.inspect_policy(args.output, paths, 'T_core', row, predictions, full, audit.core.Checks())
        path = args.output/'private'/f'diagnostic-{row["scene_id"]}.npz'
        original = audit.load(path)
        for key in ('METER__selection_regret_m', 'METER__actual_error_m', 'role_best'):
            with self.subTest(key=key):
                changed = {k: v.copy() for k, v in original.items()}
                changed[key].flat[0] += 1
                try:
                    np.savez_compressed(path, **changed)
                    with self.assertRaises(ValueError):
                        audit.inspect_diagnostic(args.output, paths, row, policy, scores, training[0], audit.core.Checks())
                finally:
                    np.savez_compressed(path, **original)


if __name__ == '__main__':
    unittest.main()
