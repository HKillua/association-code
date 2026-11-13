"""Small operator checks only; no real archive, reference, model or MTAD result."""
from types import SimpleNamespace
import unittest

import numpy as np

from tools.analysis.diagnose_pair_representation_dependency import (
    _envelope_update, _has_literal_collision, _version_variation,
    interpolate_candidate, pair_representations, VERSIONS,
)


class PairRepresentationDependencyTests(unittest.TestCase):
    def test_same_mean_different_shapes_distinguish_the_four_interfaces(self):
        t = np.arange(16, dtype=np.float64)
        a = np.column_stack((t, t*t, np.full(16, 5.), 350. + t/2))
        b = np.column_stack((2*t, 3*t, np.full(16, 2.), 340. + t/2))
        changed = b.copy()
        changed[:, 0] = b[::-1, 0]  # Same B mean; different aligned temporal shape.
        first = pair_representations(a, b)
        second = pair_representations(a, changed)
        self.assertLess(first['literal_error'], 1e-8)
        self.assertTrue(first['zero_range'][2])
        np.testing.assert_allclose(first['reconstruction_absolute'], 0., atol=1e-12)
        envelopes = {version: {} for version in VERSIONS}
        _envelope_update(envelopes, True, first['payloads'])
        _envelope_update(envelopes, False, second['payloads'])
        variation = _version_variation(envelopes)
        self.assertFalse(variation['literal']['cross_label_any_change_gt_tolerance'])
        self.assertFalse(variation['stats']['cross_label_any_change_gt_tolerance'])
        self.assertTrue(variation['pointwise']['cross_label_any_change_gt_tolerance'])
        self.assertTrue(variation['rawB']['cross_label_any_change_gt_tolerance'])
        self.assertTrue(_has_literal_collision([first['payloads']['literal']],
                                               [second['payloads']['literal']]))
        # A native change smaller than normalized epsilon can still exceed its
        # attribute-scaled float64 numerical tolerance.
        shifted = b.copy()
        shifted[:, 0] += 1e-10
        tiny = pair_representations(a, shifted)
        native_envelopes = {version: {} for version in VERSIONS}
        _envelope_update(native_envelopes, True, first['payloads'])
        _envelope_update(native_envelopes, False, tiny['payloads'])
        native_variation = _version_variation(native_envelopes)
        self.assertFalse(native_variation['literal']['cross_label_any_change_gt_tolerance'])
        self.assertTrue(native_variation['stats']['cross_label_any_change_gt_tolerance'])
        self.assertTrue(native_variation['rawB']['cross_label_any_change_gt_tolerance'])
        self.assertGreater(native_variation['rawB']['cross_label_native_max_difference_by_attribute']['lat'], 0.)
        self.assertIsNone(native_variation['pointwise']['cross_label_native_max_difference_by_attribute'])
        # Conversely, one ULP at a large native magnitude is exact float64
        # variation but is below its 64-epsilon componentwise numerical bound.
        large = b.copy()
        large[:, 2] = 1e9
        one_ulp = large.copy()
        one_ulp[:, 2] = np.nextafter(1e9, np.inf)
        large_envelopes = {version: {} for version in VERSIONS}
        _envelope_update(large_envelopes, True, pair_representations(a, large)['payloads'])
        _envelope_update(large_envelopes, False, pair_representations(a, one_ulp)['payloads'])
        large_variation = _version_variation(large_envelopes)
        self.assertTrue(large_variation['rawB']['cross_label_any_exact_content_change'])
        self.assertFalse(large_variation['rawB']['cross_label_any_change_gt_tolerance'])
        self.assertFalse(large_variation['stats']['cross_label_any_change_gt_tolerance'])
        # Each side being epsilon-close to a common template does not imply
        # that their direct cross-label distance is at most epsilon.
        self.assertFalse(_has_literal_collision([np.full(64, .5 - .75e-8)],
                                                [np.full(64, .5 + .75e-8)]))
        track = SimpleNamespace(times=np.array([0., 15.]),
            values=np.array([[0., 0., 2., 350.], [1., 1., 2., 10.]]))
        aligned = interpolate_candidate(track, t)
        self.assertTrue(aligned[7, 3] > 359. and aligned[8, 3] < 1.)
        self.assertIsNone(interpolate_candidate(track, t - 1.))
        with self.assertRaises(ValueError):
            pair_representations(a, changed * np.nan)


if __name__ == '__main__':
    unittest.main()
