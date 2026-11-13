"""Only the new target/fitter is tested here; original risk tests cover its readout."""
import unittest
import numpy as np

from association.evaluation import source_risk as original
from association.evaluation import source_selection_evidence as new


class SourceSelectionEvidenceTests(unittest.TestCase):
    def test_same_target_reproduces_original_quadratic_and_raw_target_is_single_fit(self):
        rng = np.random.default_rng(8)
        x = rng.normal(size=(71, 10))
        x[:, 2] = 3.  # fixed std floor must be unchanged
        errors = np.exp(rng.normal(size=71))*300.
        weights = rng.choice([.25, .5], size=71)
        y = np.log1p(errors/1000.)
        old = original.fit_score_models(x, y, weights)['QUADRATIC']
        current = new.fit_quadratic_target(x, y, weights)
        for key in old:
            np.testing.assert_array_equal(current[key], old[key])
        meter = new.fit_meter_model(x, errors, weights)
        self.assertEqual(meter['target_kind'], 'radial_error_km')
        self.assertEqual(meter['beta'].shape, (66,))
        self.assertFalse(np.array_equal(meter['beta'], old['beta']))
        # Explicit normal equations are a separate formulation of the requested
        # minimizer, including 1/K-weight normalization and unpenalized bias.
        z = np.clip((x-meter['mean'])/meter['std'], -8., 8.)
        basis = np.column_stack([np.ones(len(x))]+[z[:, j] for j in range(10)]+
                               [z[:, i]*z[:, j] for i in range(10) for j in range(i, 10)])
        a = weights/weights.sum()
        residual = basis.T @ (a*(basis@meter['beta']-errors/1000.))
        residual[1:] += meter['beta'][1:]
        np.testing.assert_allclose(residual, 0., atol=2e-14)
        self.assertEqual(meter['std'][2], 1.)

    def test_empty_fit_and_bad_labels_are_explicit(self):
        empty = new.fit_meter_model(np.empty((0, 10)), np.empty(0), np.empty(0))
        self.assertTrue(empty['no_labels_fallback'])
        self.assertEqual(empty['training_label_count'], 0)
        np.testing.assert_array_equal(empty['beta'], np.zeros(66))
        for errors in ([-1.], [np.inf]):
            with self.assertRaisesRegex(ValueError, 'nonnegative'):
                new.fit_meter_model(np.zeros((1, 10)), errors, [1.])

    def test_negative_scores_are_ranked_without_clipping(self):
        model = new.fit_meter_model(np.zeros((1, 10)), [1000.], [1.])
        model['beta'][:] = 0.
        model['beta'][0], model['beta'][1] = -2., 1.
        x = np.zeros((3, 10)); x[:, 0] = [-1., 0., 1.]
        scores = original.predict_score(model, x)[:, None]
        np.testing.assert_array_equal(scores[:, 0], [-3., -2., -1.])
        weights = original.source_weights(scores, np.ones((3, 1), bool), np.array([0, 3]), kind='hard')
        np.testing.assert_array_equal(weights[:, 0], [1., 0., 0.])


if __name__ == '__main__':
    unittest.main()
