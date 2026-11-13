import importlib.util
import unittest


if importlib.util.find_spec("torch") is not None:
    import torch

    from online_observation.nemot_motion import MotionBatch, NEMOTMotion, fit_motion, predict_residuals


def inputs(lengths, horizons, *, own=None, neighbors=None, weights=None, target=None):
    count = len(lengths)
    if own is None:
        own = torch.zeros(count, 6, 5)
    if neighbors is None:
        neighbors = torch.zeros(count, 6, 4, 5)
    if weights is None:
        weights = torch.zeros(count, 6, 4)
    return MotionBatch(own, neighbors, weights, torch.tensor(lengths), torch.tensor(horizons),
                       torch.zeros(count, 2, dtype=torch.float64), target)


@unittest.skipUnless(importlib.util.find_spec("torch") is not None, "NEMOT motion requires PyTorch")
class MotionTests(unittest.TestCase):
    def test_short_history_and_current_reports_keep_exact_kf_center(self):
        model = NEMOTMotion(use_neighbors=True)
        residuals = predict_residuals(model, inputs([1, 2, 3], [15.0, 0.0, 60.0]))
        self.assertTrue(torch.equal(residuals[:2], torch.zeros(2, 2)))
        self.assertTrue(torch.isfinite(residuals).all())
        self.assertLessEqual(residuals[2].abs().max().item(), 720.0)

    def test_zero_neighbor_weights_and_padding_preserve_individual_output(self):
        isolated = NEMOTMotion(use_neighbors=False)
        contextual = NEMOTMotion(use_neighbors=True)
        contextual.load_state_dict(isolated.state_dict())
        own = torch.zeros(1, 6, 5)
        own[:, 2:] = 10000.0
        neighbors = torch.full((1, 6, 4, 5), 2000.0)
        baseline = predict_residuals(isolated, inputs([2], [20.0]))
        unseen = inputs([2], [20.0], own=own, neighbors=neighbors)
        self.assertTrue(torch.equal(baseline, predict_residuals(contextual, unseen)))

        weights = torch.zeros(1, 6, 4)
        weights[:, :2, 0] = 1.0
        observed = inputs([2], [20.0], own=own, neighbors=neighbors, weights=weights)
        self.assertTrue(torch.equal(baseline, predict_residuals(isolated, observed)))
        self.assertFalse(torch.equal(baseline, predict_residuals(contextual, observed)))

    def test_full_training_budget_and_earliest_tied_selection(self):
        batch = inputs([1, 1, 1], [10.0, 20.0, 30.0], target=torch.zeros(3, 2, dtype=torch.float64))
        model, history = fit_motion(batch, batch, use_neighbors=False, epochs=2, batch_size=2)
        self.assertFalse(model.training)
        self.assertEqual(history["selected_epoch"], 1)
        self.assertEqual(history["selected_mae_m"], 0)
        self.assertEqual([epoch["train_samples"] for epoch in history["epochs"]], [3, 3])
        self.assertEqual([epoch["updates"] for epoch in history["epochs"]], [2, 2])


if __name__ == "__main__":
    unittest.main()
