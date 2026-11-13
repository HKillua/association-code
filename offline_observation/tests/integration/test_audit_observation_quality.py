"""Independent-auditor tampering tests; the cache fixture is not MTAD evidence."""
import ast
from pathlib import Path
import tempfile
import unittest

import numpy as np

from tests.integration.test_observation_quality_pilot import fixture
from tools.analysis import audit_observation_quality as audit
from tools.analysis import observation_quality_pilot as pilot


class ObservationQualityAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name).resolve()
        cls.args = fixture(cls.root)
        pilot.run(cls.args)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_full_independent_audit_and_complete_posthoc_scope(self):
        args = self.args
        result = audit.audit(args.output, args.native_run, args.source_risk_run, args.manifest, self.root/'audit.json')
        self.assertEqual(result['status'], 'COMPLETED', result.get('error'))
        self.assertEqual(result['metric_scope']['T_core']['pairs_per_scene'], 15)
        self.assertEqual(result['metric_scope']['V_select']['pairs_per_scene'], 105)
        self.assertEqual(result['metric_scope']['V_select']['primary_bootstrap_count'], 4)
        self.assertEqual(set(result['posthoc_harm']['T_core']['all_components']), set(audit.NEW))
        self.assertEqual(len(result['posthoc_harm']['T_core']['by_component']), 12)
        counts = result['contribution_counts']['T_core']['WS_OBS_LOO_F100']
        self.assertGreater(counts['uninformative_fallback_group_queries'], 0)
        self.assertGreaterEqual(counts['potential_floor_hit_source_queries'], counts['floor_hit_source_queries'])
        self.assertGreater(result['exact_checks'], 10000)

    def test_independent_native_rebuild_detects_tampered_saved_values(self):
        args = self.args
        row = audit.core.read(args.output/'scene-summary.json.gz')[0]
        sid = row['scene_id']
        policy_path = args.output/'private'/f'policy-V_select-{sid}.npz'
        state_path = args.output/'private'/f'states-V_select-{sid}.npz'
        cases = [(policy_path, 'native_loo_proxy_m2', 'native proxy loo_proxy_m2'),
                 (policy_path, 'WS_OBS_OLS_F1__source_weights', 'WS_OBS_OLS_F1 source_weights'),
                 (policy_path, 'WS_OBS_LOO_F1__group_states', 'WS_OBS_LOO_F1 frozen physical prediction'),
                 (state_path, 'WS_SHARED_Q_HARD_group_states', 'exact original133 WS_SHARED_Q_HARD'),
                 (state_path, 'WS_MEAN_aligned_truth', 'original target states')]
        for path, key, reason in cases:
            with self.subTest(key=key):
                original = audit.npz(path)
                changed = {k: v.copy() for k, v in original.items()}
                changed[key].flat[0] += 1.
                try:
                    np.savez_compressed(path, **changed)
                    with self.assertRaisesRegex(ValueError, reason):
                        audit.inspect_physical(args.output, args.native_run, args.source_risk_run,
                            args.value_run, args.partition_run, 'V_select', row, audit.core.Checks())
                finally:
                    np.savez_compressed(path, **original)

    def test_harm_denominators_are_explicit(self):
        # Largest1% of ALL points and largest1% of HARMED points are different.
        base = np.ones(200)
        candidate = base.copy()
        candidate[:4] += [1., 2., 3., 4.]
        result = {audit.EQUAL: {'errors': base}}
        result.update({name: {'errors': candidate} for name in audit.NEW})
        report = audit.harm_summary([result], [{'component_id': 'fixture'}])['all_components'][audit.NEW[0]]
        self.assertEqual(report['positive_harm_points'], 4)
        self.assertAlmostEqual(report['top_one_percent_all_points_harm_fraction'], .7)
        self.assertAlmostEqual(report['top_one_percent_harmed_points_harm_fraction'], .4)

    def test_auditor_has_no_production_imports(self):
        for module in (audit, audit.common, audit.core):
            tree = ast.parse(Path(module.__file__).read_text())
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    self.assertFalse((node.module or '').startswith('association'))
                    if (node.module or '').startswith('tools.analysis'):
                        self.assertTrue(all(alias.name.startswith('audit_') for alias in node.names))
                elif isinstance(node, ast.Import):
                    self.assertTrue(all(not alias.name.startswith('association') for alias in node.names))


if __name__ == '__main__':
    unittest.main()
