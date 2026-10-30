import unittest
import numpy as np

from association.evaluation import relative_source_decision as pair
from association.evaluation import actual_mixture_gain as mixture
from association.evaluation import source_risk as risk


class DecisionFollowupTests(unittest.TestCase):
    def test_cost_changes_optimal_preference_and_common_error_offset_does_not(self):
        x = np.zeros((2, 11, 10)); x[0, :, 0] = 1.; x[1, :, 0] = -1.
        errors = np.zeros((2, 11)); errors[0, :10] = 100.; errors[1, -1] = 10000.
        pack = dict(features=x, labeled=np.ones((2, 11), bool), errors_m=errors, group_offsets=np.array([0, 2]))
        args = ([pack], np.zeros(10), np.ones(10))
        simple = pair.fit_pair_model(*args, regret_weighted=False)
        weighted = pair.fit_pair_model(*args, regret_weighted=True)
        self.assertGreater(simple['beta'][1], 0.)
        self.assertLess(weighted['beta'][1], 0.)
        for model in (simple, weighted):
            self.assertLessEqual(model['gradient_infinity_norm'], 1e-8)
            self.assertLess(model['objective'], model['initial_objective'])
        shifted = dict(pack, errors_m=errors+90000.)
        same = pair.fit_pair_model([shifted], args[1], args[2], regret_weighted=True)
        np.testing.assert_allclose(weighted['beta'], same['beta'], atol=1e-12)
        swapped = dict(pack, features=x[::-1], errors_m=errors[::-1])
        other = pair.fit_pair_model([swapped], args[1], args[2], regret_weighted=True)
        np.testing.assert_allclose(weighted['beta'], other['beta'], atol=1e-12)

    def test_continuous_action_recovers_known_gain_curve(self):
        x = np.zeros((12, 24))
        gains = np.repeat(((mixture.ALPHAS-.25)**2-.25**2)[None], 12, axis=0)
        model = mixture.fit_gain_model(x, gains)
        alpha, gain, curvature = mixture.predict_alpha(model, x)
        np.testing.assert_allclose(alpha, .25, atol=1e-12)
        self.assertEqual(model['global_alpha'], .25)
        self.assertTrue((curvature > 0).all())
        np.testing.assert_allclose(gain, -.0625, atol=1e-12)
        bad = gains.copy(); bad[0, 0] = 1.
        with self.assertRaises(ValueError):
            mixture.fit_gain_model(x, bad)

    def test_context_and_mass_are_source_permutation_equivariant(self):
        rng = np.random.default_rng(20)
        pack = dict(features=rng.normal(size=(3, 4, 10)), available=np.ones((3, 4), bool),
                    group_offsets=np.array([0, 3]), source_states=np.zeros((3, 4, 4)))
        pack['source_states'][:, :, 1] = np.arange(3)[:, None]
        scores = np.repeat(np.array([[.3], [.1], [.2]]), 4, axis=1)
        x, hard, equal = mixture.context(pack, scores)
        order = np.array([2, 0, 1])
        perm = {k: v[order] if k != 'group_offsets' else v for k, v in pack.items()}
        xp, hp, ep = mixture.context(perm, scores[order])
        np.testing.assert_allclose(x, xp, atol=1e-12)
        np.testing.assert_array_equal(hard[order], hp)
        np.testing.assert_array_equal(equal[order], ep)
        for a in (0., .37, 1.):
            np.testing.assert_allclose(((1-a)*hard+a*equal).sum(0), 1.)

    def test_actual_recovery_endpoints_and_no_support_preserved(self):
        states = np.zeros((2, 2, 4)); states[1, :, 0] = 2.
        motion = dict(states=states, support=np.array([[True, False]]*2), nearest_seconds=np.ones((2, 2)),
            sources=np.array(['b', 'a']), native_count=np.array([2, 2]), span_seconds=np.ones(2),
            fit_rmse_m=np.zeros(2), ols_velocity_mps=np.zeros((2, 2)))
        groups = [[0, 1]]
        pack = risk.build_source_features(**motion, groups=groups)
        scores = np.array([[0., 0.], [1., 1.]])
        _, hard, equal = mixture.context(pack, scores)
        mean = np.zeros((1, 2, 4)); mean[0, 1, 0] = 7.
        a0 = mixture.recover(motion, groups, mean, pack, hard, equal, np.zeros((1, 2)))['group_states']
        a1 = mixture.recover(motion, groups, mean, pack, hard, equal, np.ones((1, 2)))['group_states']
        self.assertEqual(a0[0, 0, 0], 0.)
        self.assertEqual(a1[0, 0, 0], 1.)
        self.assertEqual(a0[0, 1, 0], 7.)
        self.assertEqual(a1[0, 1, 0], 7.)


if __name__ == '__main__':
    unittest.main()
