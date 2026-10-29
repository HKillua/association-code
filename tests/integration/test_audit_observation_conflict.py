"""Auditor software contracts; all examples are artificial, not MTAD evidence."""
import ast
import json
import tempfile
from unittest.mock import patch
from pathlib import Path
import unittest

import numpy as np

from tools.analysis import audit_observation_conflict as audit


def diagnostics():
    desired = np.array([[0, 0, 0, 0], [0, 1, 0, 1], [0, 0, 1, 0],
                        [0, 1, 1, 1], [-1, 0, 0, 1], [-2, -2, 1, 0]])
    values = np.ones((6, 4, 2))
    known = desired != -2
    for i in range(6):
        for j in range(4):
            if desired[i, j] >= 0:
                values[i, j, desired[i, j]] = 0.
    endpoint = np.tile([3., 7.], (6, 1))
    roles = np.tile([0, 1], (6, 1))
    result = audit.preference_diagnostics(values, known, endpoint, roles)
    return values, known, endpoint, roles, result


class ObservationConflictAuditTests(unittest.TestCase):
    def test_hand_partition_choice_credit_and_unknown(self):
        _, _, _, _, result = diagnostics()
        np.testing.assert_array_equal(result['cell'], np.arange(6))
        np.testing.assert_array_equal(result['ols_subcell'], [-1, -1, 0, 1, -1, -1])
        np.testing.assert_array_equal(result['selection_regret_m'][2], [0., 0., 4., 0.])
        self.assertEqual(result['selection_cost_m'][4, 0], 5.)
        self.assertEqual(result['selection_regret_m'][4, 0], 2.)
        self.assertEqual(result['selection_credit'][4, 0], .5)
        self.assertTrue(np.isnan(result['selection_cost_m'][5, 0]))
        np.testing.assert_array_equal(result['preference_mass'][5, 0], 0.)
        # These equalities are imposed by C/D support definitions, not evidence.
        self.assertEqual(result['selection_regret_m'][2, 3], result['selection_regret_m'][2, 0])
        self.assertEqual(result['selection_regret_m'][3, 3], result['selection_regret_m'][3, 2])

    def test_source_column_permutation_preserves_diagnosis(self):
        values, known, endpoint, roles, before = diagnostics()
        after = audit.preference_diagnostics(values[:, :, ::-1], known, endpoint[:, ::-1], roles[:, ::-1])
        for key in ('cell', 'ols_subcell', 'role_best', 'selection_cost_m', 'selection_regret_m', 'selection_credit'):
            np.testing.assert_array_equal(before[key], after[key])
        np.testing.assert_array_equal(before['preference_mass'], after['preference_mass'][:, :, ::-1])

    def test_endpoint_reference_tie_credit_and_signed_damage(self):
        values, known, endpoint, roles, _ = diagnostics()
        endpoint[0] = [3., 3.]
        result = audit.preference_diagnostics(values, known, endpoint, roles)
        np.testing.assert_array_equal(result['selection_credit'][0], 1.)
        self.assertEqual(result['role_best'][0], 2)
        stats = audit.delta_stats([-3., 0., 1000., 1001.])
        self.assertEqual(stats['positive_gt1000_count'], 1)
        self.assertEqual(stats['positive_0_to_1000_count'], 1)
        self.assertEqual(stats['benefit_sum'], 3.)
        self.assertEqual(stats['min'], -3.)
        self.assertIsNone(audit.stats([])['mean'])

    def test_no_production_imports(self):
        tree = ast.parse(Path(audit.__file__).read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                self.assertFalse((node.module or '').startswith('association'))
                if node.module == 'tools.analysis':
                    self.assertTrue(all(alias.name.startswith('audit_') for alias in node.names))
                if (node.module or '').startswith('tools.analysis.'):
                    self.assertTrue(node.module.split('.')[-1].startswith('audit_'))
            elif isinstance(node, ast.Import):
                self.assertTrue(all(not alias.name.startswith('association') for alias in node.names))


class ObservationConflictAuditIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from tests.integration.test_observation_conflict_probe import fixture
        from tools.analysis import observation_conflict_probe as probe
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name).resolve()
        cls.args = fixture(cls.root)
        probe.run(cls.args)
        cls.paths = {key: getattr(cls.args, key) for key in ('quality_run', 'source_run', 'mixture_run', 'native_run')}

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_full_independent_audit_with_T_only_read_guard(self):
        original_load, original_read = audit.np.load, audit.core.read
        accesses = []
        def guard(path):
            value = str(path)
            self.assertFalse(any(part in value for part in ('V_select', 'T_test', 'native-events', 'models.npz')))
            self.assertNotIn(Path(path), (self.args.quality_run/'summary.json', self.args.source_run/'summary.json'))
            accesses.append(value)
        def load(path, *args, **kwargs):
            guard(path)
            return original_load(path, *args, **kwargs)
        def read(path):
            guard(path)
            return original_read(path)
        with patch.object(audit.np, 'load', load), patch.object(audit.core, 'read', read):
            report = audit.audit(self.args.output, self.args.quality_run, self.args.source_run, self.args.mixture_run,
                self.args.native_run, self.args.manifest, self.root/'audit.json')
        self.assertEqual(report['status'], 'COMPLETED', report.get('error'))
        self.assertEqual(report['D2_query_count'], 1952)
        self.assertEqual(report['full_T_position_points'], 4064)
        self.assertEqual(report['full_group_queries'], 6144)
        self.assertEqual(report['primary_bootstraps'], 4)
        self.assertGreater(report['exact_checks'], 100000)
        self.assertTrue(accesses)

    def test_saved_preference_oracle_and_actual_error_tampering_rejected(self):
        path = self.args.output/'private/aligned-0.npz'
        original = audit.npz(path)
        for key in ('preference_winner', 'grid_error_m', 'actual_error_m', 'source_rows'):
            with self.subTest(key=key):
                changed = {k: v.copy() for k, v in original.items()}
                changed[key].flat[0] += 1
                try:
                    np.savez_compressed(path, **changed)
                    with self.assertRaisesRegex(ValueError, key):
                        audit.audit_run(self.args.output, self.paths, self.args.manifest, audit.core.Checks())
                finally:
                    np.savez_compressed(path, **original)

    def test_original_member_join_and_oof_fold_corruption_rejected(self):
        path = self.args.quality_run/'private/policy-T_core-0.npz'
        original = audit.npz(path)
        changed = {k: v.copy() for k, v in original.items()}
        changed['node_to_source_row'][0] = 1
        try:
            np.savez_compressed(path, **changed)
            with self.assertRaisesRegex(ValueError, 'source membership equivalence'):
                audit.inspect_join(self.paths, '0', 'T0', audit.core.Checks())
        finally:
            np.savez_compressed(path, **original)
        path = self.args.source_run/'T-fold-index.json'
        original_text = path.read_text()
        folds = json.loads(original_text)
        folds[0]['training_components'].append(folds[0]['heldout_component'])
        try:
            path.write_text(json.dumps(folds))
            with self.assertRaisesRegex(ValueError, 'QOOF isolation'):
                audit.metadata(self.paths, self.args.manifest, audit.core.Checks())
        finally:
            path.write_text(original_text)

    def test_bootstrap_schema_and_component_degeneracy(self):
        data = audit.npz(self.args.output/'private/aligned-0.npz')
        first = {key: value.copy() for key, value in data.items()}
        second = {key: value.copy() for key, value in data.items()}
        second['cell'][:] = 0
        expected = audit.bootstraps([first, second], ['a', 'b'])
        self.assertEqual(set(expected), {'actual_LOO_minus_NEAREST', 'selection_L_minus_N',
                                        'selection_O_minus_N', 'selection_Q_minus_N'})
        for result in expected.values():
            self.assertIsNone(result['percentile95_m'])
            self.assertEqual(result['contributing_component_count'], 1)
            self.assertGreater(result['empty_resamples'], 0)
            self.assertEqual(result['empty_resamples']+result['valid_resamples'], 2000)
        broken = dict(expected)
        del broken['selection_O_minus_N']
        with self.assertRaisesRegex(ValueError, 'schema'):
            audit.compare_tree(broken, expected, audit.core.Checks(), 'four primary bootstraps')

    def test_original_full_population_corruption_rejected(self):
        row = audit.core.read(self.args.quality_run/'training-scene-summary.json.gz')[0]
        row['methods']['WS_SOURCE_EQUAL']['population']['target_identity_count'] += 1
        with self.assertRaisesRegex(ValueError, 'original population'):
            audit.inspect_join(self.paths, '0', 'T0', audit.core.Checks(), prior_quality=row)


if __name__ == '__main__':
    unittest.main()
