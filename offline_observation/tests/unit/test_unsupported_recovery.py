import unittest
import numpy as np

from association.evaluation import unsupported_recovery as method


class UnsupportedRecoveryTests(unittest.TestCase):
    def test_endpoint_velocity_factors_and_longitude(self):
        t, q = np.array([0., 10.]), np.array([-5., 0., 5., 10., 20.])
        trace = dict(RTS=dict(endpoints=np.array([[1., 2., 3., 4.], [5., 6., 7., 8.]]),
                             positions=np.zeros((5, 2))),
                     native_local=np.array([[0., 0.], [2., 4.]]), chart_origin_latlon=np.array([0., 179.]),
                     meters_per_degree=np.array([10., 10.]))
        out = method.outside_positions(t, q, trace, np.array([.2, .4]))
        np.testing.assert_allclose(out['CVP_OLSV'][0], [0., 179.])
        np.testing.assert_allclose(out['NATIVEP_OLSV'][-1], [.4, 179.8])
        np.testing.assert_allclose(out['HOLD'][-1], [.5, 179.6])
        np.testing.assert_allclose(out['NATIVEP_CVV'][-1], [7.2, -172.6])
        for value in out.values():
            np.testing.assert_array_equal(value[1:4], 0.)
        with self.assertRaises(FloatingPointError):
            method.outside_positions(t, q*1e6, trace, np.ones(2))

    def test_source_balancing_support_in_and_renaming(self):
        motion = dict(states=np.zeros((3, 2, 4)), support=np.array([[True, False]]*3),
                      sources=np.array(['z', 'z', 'a']))
        groups = [[0, 1, 2]]  # Actual groups_from_membership returns lists.
        base, hard = np.ones((1, 2, 4))*2, np.ones((1, 2, 4))*3
        positions = {i: {k: np.array([[0., 0.], [v, 0.]])
                     for k in ('CVP_OLSV', 'NATIVEP_CVV', 'NATIVEP_OLSV', 'HOLD')}
                     for i, v in enumerate((0., 4., 10.))}
        out, zero = method.assemble(motion, groups, base, hard, positions)
        self.assertEqual(out[method.NEW[-1]][0, 1, 0], 6.)
        np.testing.assert_array_equal(out[method.NEW[-1]][~zero], hard[~zero])
        renamed = dict(motion, sources=np.array(['beta', 'beta', 'alpha']))
        other, _ = method.assemble(renamed, groups, base, hard, positions)
        np.testing.assert_array_equal(out[method.NEW[-1]], other[method.NEW[-1]])
        order = np.array([2, 0, 1])
        permuted = {k: v[order] for k, v in motion.items()}
        perm_out, _ = method.assemble(permuted, groups, base, hard, {j: positions[int(i)] for j, i in enumerate(order)})
        np.testing.assert_allclose(out[method.NEW[-1]], perm_out[method.NEW[-1]])


if __name__ == '__main__':
    unittest.main()
