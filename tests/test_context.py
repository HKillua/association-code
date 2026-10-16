import unittest

import numpy as np
import torch

from association.context import (FeatureScaler, MonotoneCalibrator, ResidualHead,
                                 context_features, fixed_rule, head_features,
                                 scene_pool_features)


def graph(probabilities):
    q = np.asarray(probabilities, dtype=float)
    mask = ~np.eye(len(q), dtype=bool)
    z = np.zeros_like(q)
    z[mask] = np.log(q[mask] / (1 - q[mask]))
    return z, mask


class ContextTests(unittest.TestCase):
    def test_analytic_support_and_missing_is_not_negative(self):
        z, mask = graph([[0, .5, .8, .9], [.5, 0, .6, .7],
                         [.8, .6, 0, .4], [.9, .7, .4, 0]])
        mask[1, 3] = mask[3, 1] = False
        z[1, 3] = z[3, 1] = np.nan
        result = context_features(z, mask)
        pair = np.flatnonzero(np.all(result["pairs"] == [0, 1], axis=1))[0]
        self.assertEqual(result["counts"][pair], 1)
        self.assertAlmostEqual(result["support"][pair], .48)
        self.assertAlmostEqual(result["contradiction"][pair], .44)
        np.testing.assert_allclose(result["features"][pair],
                                   [0, .48, .44, 1.4, .2, np.log(2), 0, 0], rtol=1e-6)

    def test_two_negative_edges_are_not_high_support(self):
        z, mask = graph([[0, .5, .01], [.5, 0, .01], [.01, .01, 0]])
        result = context_features(z, mask)
        self.assertAlmostEqual(result["support"][0], .0001)
        self.assertAlmostEqual(result["contradiction"][0], .0198)

    def test_zero_neighbors_and_zero_initial_head_are_identity(self):
        z, mask = graph([[0, .3], [.3, 0]])
        result = context_features(z, mask)
        self.assertEqual(result["counts"].tolist(), [0])
        self.assertEqual(fixed_rule(result).tolist(), [0])
        head = ResidualHead("B", 2)
        inputs = torch.from_numpy(result["features"])
        self.assertEqual(head(inputs, result["counts"]).item(), 0)
        with torch.no_grad():
            head.network[-1].bias.fill_(100)
        self.assertEqual(head(inputs, result["counts"]).item(), 0)
        self.assertLessEqual(head(inputs, [1]).abs().item(), 2)

    def test_base_node_permutation_equivariance(self):
        rng = np.random.default_rng(10)
        raw = rng.normal(size=(7, 7))
        z = (raw + raw.T) / 2
        mask = ~np.eye(7, dtype=bool)
        mask[1, 5] = mask[5, 1] = False
        original = context_features(z, mask)
        permutation = np.array([3, 6, 0, 5, 2, 1, 4])
        changed = context_features(z[np.ix_(permutation, permutation)],
                                   mask[np.ix_(permutation, permutation)])
        expected = {tuple(pair): f for pair, f in zip(original["pairs"], original["features"])}
        for pair, feature in zip(changed["pairs"], changed["features"]):
            key = tuple(sorted(permutation[pair]))
            np.testing.assert_allclose(feature, expected[key], rtol=1e-6, atol=1e-7)

    def test_shuffle_changes_correspondence_only(self):
        z, mask = graph([[0, .5, .2, .8], [.5, 0, .2, .8],
                         [.2, .2, 0, .4], [.8, .8, .4, 0]])
        original = context_features(z, mask)
        shuffled = context_features(z, mask, shuffle=True, seed=6)
        repeated = context_features(z, mask, shuffle=True, seed=6)
        np.testing.assert_array_equal(shuffled["features"], repeated["features"])
        np.testing.assert_array_equal(original["features"][:, [0, 3, 4, 5, 6, 7]],
                                      shuffled["features"][:, [0, 3, 4, 5, 6, 7]])
        self.assertAlmostEqual(original["support"][0], .34)
        self.assertAlmostEqual(shuffled["support"][0], .16)
        self.assertTrue(shuffled["shuffleable"][0])
        self.assertTrue(shuffled["shuffle_effective"][0])

    def test_controls_use_allowed_information_and_real_matched_capacity(self):
        z, mask = graph([[0, .5, .2], [.5, 0, .8], [.2, .8, 0]])
        context = context_features(z, mask)
        pooled = scene_pool_features(z, mask)
        np.testing.assert_allclose(pooled[:, 3:].sum(axis=1), 1)
        self.assertEqual(pooled.shape, (3, 13))
        marginal = head_features("EndpointMarginal", context)
        context["features"][:, 1:3] += 100
        np.testing.assert_array_equal(marginal, head_features("EndpointMarginal", context))
        for kind in ["B", "B_shuffle", "PairMLP", "EndpointMarginal", "ScenePool"]:
            head = ResidualHead(kind)
            self.assertLess(abs(head.parameter_count - 833) / 833, .05)
            feature = torch.from_numpy(head_features(kind, context, pooled))
            output = head(feature, context["counts"])
            torch.testing.assert_close(output, torch.zeros(len(feature)))
            output.sum().backward()
            self.assertIsNotNone(head.network[-1].weight.grad)

    def test_calibration_monotonic_identity_and_checkpoint(self):
        module = MonotoneCalibrator()
        values = torch.tensor([-10., -1., 0., 1., 10.])
        torch.testing.assert_close(module(values), values)
        with torch.no_grad():
            module.raw_a.fill_(-30)
            module.bias.fill_(2)
        self.assertGreater(module.slope.item(), 0)
        self.assertTrue(torch.all(torch.diff(module(values)) >= 0))
        clone = MonotoneCalibrator()
        clone.load_state_dict(module.state_dict())
        torch.testing.assert_close(clone(values), module(values))

    def test_scaler_fit_is_training_only_and_frozen(self):
        inputs = torch.tensor([[1., 10.], [3., 10.]])
        scaler = FeatureScaler(2)
        with self.assertRaises(ValueError):
            scaler.fit(inputs, split="T_test")
        scaler.fit(inputs)
        torch.testing.assert_close(scaler(inputs), torch.tensor([[-1., 0.], [1., 0.]]))
        with self.assertRaises(ValueError):
            scaler.fit(inputs)
        clone = FeatureScaler(2)
        clone.load_state_dict(scaler.state_dict())
        torch.testing.assert_close(clone(inputs), scaler(inputs))

    def test_corrupt_graph_and_nonfinite_features_fail(self):
        with self.assertRaises(ValueError):
            context_features(np.array([[0., 1.], [2., 0.]]), ~np.eye(2, dtype=bool))
        with self.assertRaises(ValueError):
            context_features(np.array([[0., np.inf], [np.inf, 0.]]), ~np.eye(2, dtype=bool))
        with self.assertRaises(ValueError):
            context_features(np.zeros((2, 2)), np.ones((2, 2), dtype=bool))
        with self.assertRaises(ValueError):
            ResidualHead("B")(torch.full((1, 8), float("nan")), [1])


if __name__ == "__main__":
    unittest.main()
