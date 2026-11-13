"""137 isolation, selection and recovery contracts; no MTAD effect claims."""
from copy import deepcopy
import unittest
from unittest.mock import patch

import numpy as np

from association.evaluation import observation_scale as scale
from association.evaluation import source_risk as risk
from association.evaluation.fusion_controls import ordinary_fusion
from tests.integration.test_source_risk import inputs
from tools.analysis import observation_scale_pilot as pilot
from tools.analysis import native_cv_state_control as cv


class ObservationScaleTests(unittest.TestCase):
    def test_split_endpoints_pattern_and_missing_case(self):
        for n, expected in ((2, []), (3, [1]), (6, [1, 4]), (8, [1, 4])):
            t = np.cumsum(np.arange(n)+1.)
            xy = np.column_stack((np.zeros(n), np.linspace(179., 179.2, n)))
            original = xy.copy()
            result = scale.split_native_observations(t, xy)
            np.testing.assert_array_equal(result['heldout_indices'], expected)
            self.assertIn(0, result['retained_indices'])
            self.assertIn(n-1, result['retained_indices'])
            self.assertEqual(result['eligible'], bool(expected))
            np.testing.assert_array_equal(xy, original)
        none = scale.heldout_rho_errors(times=[0., 1.], latlon=[[0., 0.], [0., .1]])
        self.assertFalse(none['eligible'])
        self.assertEqual(none['analytic_filter_smoother_fits'], 0)
        self.assertEqual(none['rts_positions'].shape, (9, 0, 2))
        self.assertTrue(np.isfinite(none['mse_m2']).all())

    def test_heldout_positions_absent_from_chart_and_prediction(self):
        t = np.array([0., 3., 9., 17., 28., 42.])
        xy = np.array([[10., 179.8], [10.01, -179.9], [10.02, -179.7],
                       [10.03, -179.5], [10.04, -179.3], [10.05, -179.1]])
        changed = xy.copy()
        changed[[1, 4]] = [[70., 80.], [-30., -100.]]
        one = scale.heldout_rho_errors(times=t, latlon=xy)
        two = scale.heldout_rho_errors(times=t, latlon=changed)
        for key in ('forward_positions', 'rts_positions', 'retained_latlon'):
            np.testing.assert_array_equal(one[key], two[key])
        self.assertFalse(np.array_equal(one['mse_m2'], two['mse_m2']))
        self.assertEqual(one['analytic_filter_smoother_fits'], 9)
        self.assertEqual(one['analytic_work']['kalman_observation_updates'], 9*3)
        # Direct retained-only calls bind the reused kernel including dates.
        expected, _ = cv.cv_native_positions(t[[0, 2, 3, 5]], xy[[0, 2, 3, 5]], t[[1, 4]], scale.RHOS[0])
        np.testing.assert_array_equal(one['rts_positions'][0], expected['rts'])

    def test_scene_then_fragment_equal_not_point_or_global_fragment_weighting(self):
        # Candidate0 wins by equal scenes (5 vs 6) although concatenating all
        # fragments gives candidate1 (20/3 vs 4). Missing nodes never count 0.
        one = np.tile(np.array([0., 12.]+[100.]*7), (1, 1))
        two = np.tile(np.array([10., 0.]+[100.]*7), (2, 1))
        rows = [dict(scene_id='a', mse_m2=np.vstack([one, np.zeros((1, 9))]), eligible=np.array([True, False])),
                dict(scene_id='b', mse_m2=two, eligible=np.ones(2, bool)),
                dict(scene_id='c', mse_m2=np.zeros((1, 9)), eligible=np.zeros(1, bool))]
        result = scale.select_observation_rho(rows)
        np.testing.assert_array_equal(result['objective_by_rho'][:2], [5., 6.])
        self.assertEqual(result['selected_index'], 0)
        self.assertEqual(result['eligible_scene_count'], 2)
        self.assertIsNone(result['per_scene'][2]['objective_by_rho'])
        rows[0]['mse_m2'][:] = 1.
        tie = scale.select_observation_rho(rows[:1])
        self.assertEqual(tie['tie_count'], 9)
        self.assertEqual(tie['selected_rho'], scale.RHOS[0])
        with self.assertRaisesRegex(ValueError, 'no eligible'):
            scale.select_observation_rho(rows[2:])

    def test_contract_rejects_bad_times_and_grid(self):
        with self.assertRaises(ValueError):
            scale.split_native_observations([0., 0.], [[0., 0.], [0., 0.]])
        with self.assertRaises(ValueError):
            scale.heldout_rho_errors(times=[0., 1., 2.], latlon=[[0., 0.]]*3, rhos=[1e-8])
        with self.assertRaises(TypeError):
            scale.heldout_rho_errors(times=[0., 1., 2.], latlon=[[0., 0.]]*3, reference=np.zeros((3, 2)))

    def test_equal_recovery_zero_support_and_general_k_rename(self):
        for k in (1, 2, 3, 4):
            data = inputs(tuple(f's{i}' for i in range(k)), q=4)
            data['support'][:, 0] = False
            groups = data.pop('groups')
            first = [g[0] for g in groups]
            mean = ordinary_fusion(data['states'], data['nearest_seconds'], data['support'], groups)[
                'predictions']['mean_supported'][first]
            pack = risk.build_source_features(**data, groups=groups)
            positions = data['states'][..., :2].copy()
            positions[..., 0] += .01
            recovered, _, _ = pilot.recover_equal(data, groups, mean, pack, positions)
            np.testing.assert_array_equal(recovered['group_states'][0, 0, :2], positions[0, 0])
            np.testing.assert_array_equal(recovered['group_states'][..., 2:], mean[..., 2:])
            renamed = deepcopy(data)
            renamed['sources'] = np.array([f'z{99-i}' for i in range(k)])
            p2 = risk.build_source_features(**renamed, groups=groups)
            other, _, _ = pilot.recover_equal(renamed, groups, mean, p2, positions)
            np.testing.assert_array_equal(other['group_states'], recovered['group_states'])
            # Supported outputs are invariant to physical row permutations.
            order = np.arange(k)[::-1]
            permuted = {key: value[order] for key, value in data.items()}
            pp = risk.build_source_features(**permuted, groups=groups)
            pm = ordinary_fusion(permuted['states'], permuted['nearest_seconds'], permuted['support'], groups)[
                'predictions']['mean_supported'][first]
            out, _, _ = pilot.recover_equal(permuted, groups, pm, pp, positions[order])
            np.testing.assert_allclose(out['group_states'][:, 1:, :2], recovered['group_states'][:, 1:, :2], atol=1e-12)
            # Fixed group-first no-support fallback is an explicit ordering
            # boundary inherited from127; no permutation claim there.


if __name__ == '__main__':
    unittest.main()
