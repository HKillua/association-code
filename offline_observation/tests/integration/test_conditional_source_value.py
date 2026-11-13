"""125 actual isolation/routing software regressions, not MTAD effect evidence."""
from copy import deepcopy
import unittest

import numpy as np

from tools.analysis import conditional_source_value as p
from tests.integration.test_source_mix_shrink_control import saved121


class ConditionalValueTests(unittest.TestCase):
    def test_gain_sign_first_anchor_only_and_unknown_population(self):
        hard = np.zeros((3, 64, 4)); hard[..., 0] = 1.
        soft = np.zeros_like(hard)
        truth = np.zeros((3, 64, 4)); valid = np.ones_like(truth, bool); valid[0, 0, :2] = False
        gain, labeled, audit = p.gain_labels(hard, soft, np.ones((3, 64), bool),
            [[0, 1], [2], [3]], np.array(['a', 'a', 'a', 'b']), np.array(['a', 'b', 'c']), truth, valid)
        self.assertFalse(labeled[0, 0]); self.assertTrue(labeled[0, 1])
        self.assertGreater(gain[0, 1], 100000.)
        self.assertTrue(np.isnan(gain[1]).all())
        self.assertEqual(audit['duplicate_anchor_groups_without_training_labels'], 1)
        self.assertEqual(audit['target_without_group_anchor_count'], 1)
        self.assertEqual(audit['labeled_both_query_count'], 127)
        self.assertEqual(audit['all_matched_reference_valid_query_count'], 127)
        both = np.ones((3, 64), bool); both[2] = False
        _, eligible, reduced = p.gain_labels(hard, soft, both, [[0, 1], [2], [3]],
            np.array(['a', 'a', 'a', 'b']), np.array(['a', 'b', 'c']), truth, valid)
        self.assertEqual(int(eligible.sum()), 63)
        self.assertEqual(reduced['all_matched_reference_valid_query_count'], 127)

    def test_T_only_component_isolation_and_no_label_fallback(self):
        rng = np.random.default_rng(20)
        rows = [dict(scene_id=str(i), component_id='a' if i < 2 else 'b', split='T_core',
                     x=rng.normal(size=(8, 15)), y=rng.normal(size=8)) for i in range(3)]
        original, train = p.fit_component_fold(rows, 'a')
        changed = deepcopy(rows)
        for row in changed[:2]:
            row['x'][:] = 1e9; row['y'][:] = -1e12
        replay, _ = p.fit_component_fold(changed, 'a')
        self.assertEqual([row['scene_id'] for row in train], ['2'])
        for name in p.HEADS:
            for key in ('mean', 'std', 'beta', 'permutation'):
                np.testing.assert_array_equal(original[name][key], replay[name][key])
        with self.assertRaisesRegex(ValueError, 'T_core'):
            p.fit_component_fold([dict(rows[0], split='V_select')], 'a')
        empty = p.fit_heads(np.empty((0, 15)), np.empty(0))
        for model in empty.values():
            np.testing.assert_array_equal(p.predict_head(model, np.ones((3, 15))*1e3), np.zeros(3))
            self.assertTrue(model['no_labels_fallback'])

    def test_mean_scaled_ridge_normal_equations_and_permutation(self):
        rng = np.random.default_rng(21); x = rng.normal(size=(120, 15)); y = 3.+x[:, 0]*7.
        models = p.fit_heads(x, y)
        self.assertEqual(models['TMEAN']['beta'].shape, (1,))
        for name in ('LINEAR', 'QUADRATIC', 'SHUFFLED'):
            model = models[name]
            design = p.phi(x, model['mean'], model['std'], name != 'LINEAR')
            penalty = model['beta'].copy(); penalty[0] = 0.
            residual = design.T @ (design @ model['beta']-y[model['permutation']])/len(y)+penalty
            np.testing.assert_allclose(residual, 0., atol=1e-12)
            self.assertEqual(len(model['beta']), 16 if name == 'LINEAR' else 136)
        self.assertFalse(np.array_equal(models['SHUFFLED']['permutation'], np.arange(len(y))))
        np.testing.assert_array_equal(models['SHUFFLED']['mean'], models['QUADRATIC']['mean'])
        extreme = p.phi(np.ones((1, 15))*1e9, np.zeros(15), np.ones(15), True)
        np.testing.assert_array_equal(extreme[0, 1:16], np.ones(15)*8.)
        np.testing.assert_array_equal(extreme[0, 16:], np.ones(120)*64.)

    def test_gate_outside_both_exact_and_regret_definition(self):
        models = p.fit_heads(np.zeros((4, 15)), np.ones(4))
        hard = np.zeros((1, 64, 4)); soft = np.ones_like(hard)
        both = np.zeros((1, 64), bool); both[0, 2] = True
        outputs, gains, gates = p.apply_heads(models, np.zeros((1, 64, 15)), both, hard, soft)
        for value in outputs.values():
            np.testing.assert_array_equal(value[~both], hard[~both])
            np.testing.assert_array_equal(value[both], soft[both])
        metrics = p.gain_metrics(np.array([3., -5.]), np.array([1., 1.]), full_denominator=100)
        self.assertEqual(metrics['total_realized_gain_m'], -2.)
        self.assertEqual(metrics['mean_regret_m'], 2.5)
        self.assertEqual(metrics['full_common_point_mae_change_m'], .02)
        self.assertEqual(metrics['mean_realized_gain_m'], -1.)

    def test_saved_candidates_import_and_all_eleven_artifacts(self):
        fragment, membership, retained, calibration, public = saved121()
        mix_row, _, mixed_artifacts, _ = p.mix.analyze_scene(fragment, membership, retained, calibration, public)
        groups = p.old.groups_from_membership(membership, len(fragment['native_states']))
        n = len(fragment['native_states']); pairs = np.column_stack(np.triu_indices(n, 1))
        features, both, hard, soft = p.make_actions_features(states=fragment['native_states'],
            support=fragment['support'], nearest_seconds=fragment['nearest_seconds'], sources=fragment['sources'],
            groups=groups, calibration=calibration, native_count=np.ones(n)*5., fit_rmse_m=np.zeros(n),
            ols_velocity_mps=np.zeros((n, 2)), pairs=pairs, scores=np.ones(len(pairs))*2.,
            mean_states=retained['WS_MEAN']['group_states'],
            imported_hard=retained['WS_FIXED_LOW_SOURCE_SUPPORTED']['group_states'],
            imported_soft=mixed_artifacts[p.SOFT]['group_states'])
        _, _, constructed_hard, constructed_soft = p.make_actions_features(states=fragment['native_states'],
            support=fragment['support'], nearest_seconds=fragment['nearest_seconds'], sources=fragment['sources'],
            groups=groups, calibration=calibration, native_count=np.ones(n)*5., fit_rmse_m=np.zeros(n),
            ols_velocity_mps=np.zeros((n, 2)), pairs=pairs, scores=np.ones(len(pairs))*2.,
            mean_states=retained['WS_MEAN']['group_states'])
        np.testing.assert_array_equal(constructed_hard, hard)
        np.testing.assert_array_equal(constructed_soft, soft)
        models = p.fit_heads(np.empty((0, 15)), np.empty(0))
        added, _, _ = p.apply_heads(models, features, both, hard, soft)
        controls = {name: retained[name] for name in p.BASES}; controls[p.SOFT] = mixed_artifacts[p.SOFT]
        predictions = {name: artifact['group_states'] for name, artifact in controls.items()}; predictions.update(added)
        row, _, artifacts = p.evaluate_v(fragment, groups, predictions, controls, both)
        self.assertEqual((len(row['methods']), len(row['pairwise'])), (11, 55))
        for name in p.HEAD_METHODS:
            for key in artifacts[name]:
                np.testing.assert_array_equal(artifacts[name][key], artifacts['WS_FIXED_LOW_SOURCE_SUPPORTED'][key])
        self.assertEqual(features.shape, (len(groups), 64, 15))


if __name__ == '__main__':
    unittest.main()
