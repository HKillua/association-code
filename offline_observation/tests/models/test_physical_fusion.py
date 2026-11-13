"""Mathematical correctness only; these checks are not MTAD performance evidence."""
import unittest

import numpy as np

from association.models.physical_fusion import (
    EARTH_RADIUS_M, geographic_exp, geographic_log,
    risk_simplex_weights, smooth_bounded_positions,
)


class PhysicalFusionTests(unittest.TestCase):
    def test_sphere_roundtrip_dateline_pole_and_antipode(self):
        for origin in ([72., 179.8], [90., 25.], [-40., -120.]):
            xy = np.array([[0., 0.], [125000., -72000.], [-3300., 420.]], np.float64)
            positions = geographic_exp(xy, origin)
            np.testing.assert_allclose(geographic_log(positions, origin), xy, atol=1e-7, rtol=0)
        points = np.array([[72.2, -179.7], [71.5, 179.1]])
        np.testing.assert_allclose(geographic_exp(geographic_log(points, [72., 179.8]), [72., 179.8]), points, atol=1e-10)
        with self.assertRaises(ValueError):
            geographic_log([[0., 180.]], [0., 0.])
        with self.assertRaises(ValueError):
            geographic_exp([[EARTH_RADIUS_M * np.pi, 0.]], [0., 0.])

    def test_risk_identity_and_projection_of_conditional_mean(self):
        points = np.array([[0., 0.], [3., 0.], [0., 4.], [4., 4.]])
        mean = np.array([1.2, 1.3])
        risks = np.sum((points - mean) ** 2, axis=1) + 2.7
        w, diagnostic = risk_simplex_weights(points, risks)
        self.assertTrue(diagnostic['converged'])
        self.assertLess(diagnostic['relative_kkt_residual'], 1e-10)
        self.assertAlmostEqual(w.sum(), 1.)
        self.assertTrue((w >= 0).all())
        np.testing.assert_allclose(w @ points, mean, atol=1e-8)
        self.assertAlmostEqual(diagnostic['objective'], 2.7, places=8)
        # Inconsistent estimated risks can give a negative objective. The QP
        # is still convex on the simplex; do not silently force a PSD matrix.
        inconsistent, estimate = risk_simplex_weights([[-1., 0.], [1., 0.]], [0., 0.])
        np.testing.assert_array_equal(inconsistent, [.5, .5])
        self.assertEqual(estimate['objective'], -1.)
        arbitrary = np.array([.1, .2, .3, .4])
        distances = np.sum((points[:, None] - points[None, :]) ** 2, -1)
        self.assertAlmostEqual(risks @ arbitrary - .5 * arbitrary @ distances @ arbitrary,
                               np.sum((arbitrary @ points - mean) ** 2) + 2.7)
        # R2P is exactly this risk construction, including outside-hull targets.
        target = np.array([5., -2.])
        tri = points[:3]
        weights, diagnostic = risk_simplex_weights(tri, np.sum((tri - target) ** 2, 1))
        self.assertTrue(diagnostic['converged'])
        np.testing.assert_allclose(weights @ tri, [3., 0.], atol=1e-10)

    def test_risk_permutation_translation_invalid_and_degenerate_candidates(self):
        points = np.array([[0., 0.], [1., 0.], [0., 2.], [2., 1.]])
        risks = np.array([4., 2., 3., 1.])
        w, diagnostic = risk_simplex_weights(points, risks)
        order = [2, 0, 3, 1]
        v, permuted = risk_simplex_weights(points[order] + 10000., risks[order])
        self.assertTrue(diagnostic['converged'] and permuted['converged'])
        np.testing.assert_allclose(w @ points, v @ points[order], atol=1e-9)
        self.assertAlmostEqual(diagnostic['objective'], permuted['objective'])
        offset, shifted = risk_simplex_weights(points, risks + 1e12)
        np.testing.assert_array_equal(w, offset)
        self.assertEqual(diagnostic['iterations'], shifted['iterations'])
        repeat, tied = risk_simplex_weights(np.zeros((4, 2)), np.ones(4))
        np.testing.assert_array_equal(repeat, [.25] * 4)
        self.assertEqual(tied['iterations'], 0)
        repeated, _ = risk_simplex_weights(np.zeros((3, 2)), [3., 1., 2.])
        np.testing.assert_array_equal(repeated, [0., 1., 0.])
        masked, _ = risk_simplex_weights([[np.nan, 0.], [2., 4.], [1., 1.]], [1., 2., 0.], np.array([True, True, False]))
        np.testing.assert_array_equal(masked, [0., 1., 0.])
        empty, failed = risk_simplex_weights([[np.nan, 0.]], [0.])
        self.assertFalse(failed['converged'])
        np.testing.assert_array_equal(empty, [0.])
        _, limited = risk_simplex_weights(points, risks, max_iterations=0)
        self.assertFalse(limited['converged'])

    def test_irregular_constant_velocity_and_exact_identity(self):
        times = np.array([0., 2., 7., 25., 40.])
        base = np.column_stack((200. + 3. * times, -100. - 7. * times))
        result, diagnostic = smooth_bounded_positions(base, times, 2., 50.)
        self.assertTrue(diagnostic['converged'])
        np.testing.assert_allclose(result, base, atol=1e-8, rtol=0)
        for strength, radius in ((0., 20.), (3., 0.)):
            result, diagnostic = smooth_bounded_positions(base, times, strength, radius)
            np.testing.assert_array_equal(result, base)
            self.assertEqual(diagnostic['reason'], 'identity')

    def test_box_optimum_not_clipped_unconstrained_and_failure_fallback(self):
        base = np.column_stack(([0., 1., 3.], [0., 0., 0.]))
        result, diagnostic = smooth_bounded_positions(base, [0., 1., 2.], 1., .2, tau=1.)
        self.assertTrue(diagnostic['converged'])
        np.testing.assert_allclose(result[:, 0], [-.2, 1.2, 2.8], atol=1e-9)
        self.assertLessEqual(np.max(np.abs(result - base)), .2 + 1e-14)
        self.assertLess(diagnostic['kkt_residual'], 1e-9)
        d = np.array([[1., -2., 1.]])
        unconstrained = np.linalg.solve(np.eye(3) + d.T @ d, base)
        clipped = base + np.clip(unconstrained - base, -.2, .2)
        objective = lambda x: np.sum((x - base) ** 2) + np.sum((d @ x) ** 2)
        self.assertLess(objective(result), objective(clipped) - 1e-3)
        failed, diagnostic = smooth_bounded_positions(base, [0., 1., 2.], 1., .2, tau=1., max_iterations=0)
        self.assertTrue(diagnostic['fallback'])
        np.testing.assert_array_equal(failed, base)


if __name__ == '__main__':
    unittest.main()
