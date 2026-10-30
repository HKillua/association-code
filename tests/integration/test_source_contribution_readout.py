"""Small mathematical/software checks only; no scientific performance claims."""
import unittest

import numpy as np

from association.evaluation import source_contribution_readout as c
from association.evaluation import source_risk as original
from tools.analysis import observation_common as common
from tests.integration.test_source_risk_pilot import scene_arrays


class SourceContributionReadoutTests(unittest.TestCase):
    def test_chord_labels_and_invalid_values(self):
        e = np.asarray([0., np.pi*c.RADIUS_M/2, np.pi*c.RADIUS_M])
        np.testing.assert_allclose(c.chord_targets(e),
            [0., 2*(c.RADIUS_M/1000.)**2, 4*(c.RADIUS_M/1000.)**2])
        for bad in ([np.nan], [-1.], [np.pi*c.RADIUS_M+1.]):
            with self.assertRaises(ValueError):
                c.chord_targets(bad)
        # Date-line endpoints: compare a separate great-circle expression.
        states = np.zeros((2, 4)); states[:, 1] = [179.9, -179.9]
        chord = np.sum(np.diff(c.embedding_km(states), axis=0)**2)
        self.assertAlmostEqual(chord, c.chord_targets([np.deg2rad(.2)*c.RADIUS_M])[0], places=7)

    def test_one_weighted_ridge_and_normal_equation(self):
        x = np.arange(80, dtype=float).reshape(8, 10)/40.
        y = np.asarray([1., 2., 3., 2., 6., 8., 4., 10.])
        weight = np.arange(1., 9.)
        model = c.fit_risk_model(x, y, weight)
        self.assertEqual(model['beta'].shape, (66,))
        self.assertEqual(model['training_label_count'], 8)
        a = weight/weight.sum()
        mu = (a[:, None]*x).sum(0)
        sigma = np.sqrt((a[:, None]*(x-mu)**2).sum(0))
        z = np.clip((x-mu)/sigma, -8., 8.)
        phi = np.column_stack([np.ones(8), z]+[z[:, i]*z[:, j] for i in range(10) for j in range(i, 10)])
        penalty = model['beta'].copy(); penalty[0] = 0.
        np.testing.assert_allclose(phi.T@(a*(phi@model['beta']-y))+penalty, 0., atol=1e-12)
        np.testing.assert_allclose(c.predict_risk(model, x), phi@model['beta'])
        constant = c.fit_risk_model(np.zeros((3, 10)), np.asarray([2., 4., 8.]), np.asarray([1., 2., 1.]))
        self.assertAlmostEqual(constant['beta'][0], 4.5)
        np.testing.assert_array_equal(constant['beta'][1:], np.zeros(65))

    def test_two_source_solution_includes_endpoints_and_ties(self):
        states = np.zeros((2, 3, 4)); states[1, :, 1] = .01
        xyz = c.embedding_km(states[:, 0]); d = np.sum((xyz[0]-xyz[1])**2)
        r = np.asarray([[1., 0., 4.*d], [1., 4.*d, 0.]])
        out = c.contribution_weights(r, np.ones_like(r, bool), np.asarray([0, 2]), states, kind='geometry')
        np.testing.assert_allclose(out['source_weights'], [[.5, 1., 0.], [.5, 0., 1.]])
        np.testing.assert_allclose(out['duality_gap_km2'], 0., atol=1e-12)
        # Equal wrong risks can produce a negative surrogate; do not clip it.
        bad = c.contribution_weights(np.zeros_like(r), np.ones_like(r, bool), np.asarray([0, 2]), states, kind='geometry')
        self.assertTrue((bad['objective_km2'] < 0.).all())
        same = np.zeros_like(states)
        out = c.contribution_weights(r, np.ones_like(r, bool), np.asarray([0, 2]), same, kind='geometry')
        np.testing.assert_allclose(out['source_weights'][:, 0], .5)

    def test_general_k_geometry_certificate_and_permutation(self):
        states = np.zeros((4, 1, 4))
        states[:, 0, :2] = [[0., 0.], [.01, 0.], [0., .015], [.012, .013]]
        xyz = c.embedding_km(states)[:, 0]
        target = np.asarray([.5, .1, .3, .1])@xyz
        r = np.sum((xyz-target)**2, axis=1)[:, None]
        out = c.contribution_weights(r, np.ones_like(r, bool), np.asarray([0, 4]), states, kind='geometry')
        w = out['source_weights'][:, 0]
        true = np.sum((w@xyz-target)**2)
        self.assertAlmostEqual(out['objective_km2'][0, 0], true, places=10)
        self.assertGreaterEqual(true, 0.)
        self.assertLessEqual(true, out['duality_gap_km2'][0, 0]+1e-10)
        self.assertLessEqual(out['solver_iterations'][0, 0], 256)
        order = np.asarray([2, 0, 3, 1])
        perm = c.contribution_weights(r[order], np.ones_like(r, bool), np.asarray([0, 4]), states[order], kind='geometry')
        np.testing.assert_allclose(perm['source_weights'], out['source_weights'][order], atol=1e-10, rtol=0.)
        # A fixed nonzero reference bias shared by all sources cannot be removed
        # by convex combination of identical endpoints.
        same = np.zeros((3, 1, 4)); same[:, 0, 0] = 1.
        shared = c.contribution_weights(np.ones((3, 1)), np.ones((3, 1), bool), np.asarray([0, 3]), same, kind='geometry')
        np.testing.assert_allclose(shared['source_weights'], 1./3.)

    def test_masks_floor_and_recovery_reuses_original_nodes(self):
        motion, ev, member, artifact = scene_arrays('0')
        groups = [[0, 1], [2], [3]]
        pack = original.build_source_features(**{key: motion[key] for key in common.source.MOTION_KEYS}, groups=groups)
        mean = artifact['group_states']
        r = np.zeros_like(pack['available'], float)
        for kind in ('hard', 'inverse', 'geometry'):
            policy = c.contribution_weights(r, pack['available'], pack['group_offsets'], pack['source_states'], kind=kind)
            recovered = c.recover_contributions(states=motion['states'], support=motion['support'], groups=groups,
                pack=pack, source_weights=policy['source_weights'], mean_states=mean)
            np.testing.assert_array_equal(recovered['group_states'][:, 0], mean[:, 0])
            np.testing.assert_array_equal(recovered['group_states'][..., 2:], mean[..., 2:])
            np.testing.assert_array_equal(recovered['node_weights'][~motion['support']], 0.)
            if kind == 'inverse':
                np.testing.assert_array_equal(policy['floor_hit'], pack['available'])
        equal = original.recover_source_policy(states=motion['states'], support=motion['support'], groups=groups,
            pack=pack, scores=r, kind='equal', mean_states=mean)
        custom = c.recover_contributions(states=motion['states'], support=motion['support'], groups=groups,
            pack=pack, source_weights=equal['source_weights'], mean_states=mean)
        np.testing.assert_array_equal(custom['group_states'], equal['group_states'])
        bad = dict(pack); bad['supported_counts'] = pack['supported_counts']+1
        with self.assertRaises(ValueError):
            c.recover_contributions(states=motion['states'], support=motion['support'], groups=groups,
                pack=bad, source_weights=equal['source_weights'], mean_states=mean)

    def test_same_source_fragments_do_not_renormalise_circular_moments(self):
        states = np.zeros((3, 2, 4)); states[:, :, 1] = np.asarray([-80., 80., 180.])[:, None]
        states[..., 2] = 4.; states[..., 3] = 359.
        support = np.ones((3, 2), bool); support[:, 1] = False
        groups = [[0, 1, 2]]
        nearest = np.ones((3, 2))
        pack = original.build_source_features(states=states, support=support, nearest_seconds=nearest,
            sources=np.asarray(['A', 'A', 'B']), groups=groups, native_count=np.full(3, 4),
            span_seconds=np.full(3, 10.), fit_rmse_m=np.ones(3), ols_velocity_mps=np.zeros((3, 2)))
        mean = common.source.ordinary_fusion(states, nearest, support, groups)['predictions']['mean_supported'][[0]]
        weights = np.where(pack['available'], .5, 0.)
        result = c.recover_contributions(states=states, support=support, groups=groups, pack=pack,
                                         source_weights=weights, mean_states=mean)
        np.testing.assert_allclose(result['node_weights'][:, 0], [.25, .25, .5])
        # The original moments give longitude ±180, whereas first normalising
        # source A's resultant to longitude0 and then averaging with180 is degenerate.
        self.assertAlmostEqual(abs(result['group_states'][0, 0, 1]), 180.)
        np.testing.assert_array_equal(result['group_states'][:, 1], mean[:, 1])


if __name__ == '__main__':
    unittest.main()
