import unittest

import numpy as np
import torch

from association.models.evidence_fusion import (
    EvidenceFusion, evidence_fusion_loss, fusion_inputs, physical_errors)
from association.models.cooperative import CooperativeConfig
from association.training.evidence_fusion import prepare_evidence_scenes, training_members
from tools.experiments.cooperative_fit import synthetic_scenes


class EvidenceFusionTests(unittest.TestCase):
    def test_exact_rejection_circular_convexity_and_optional_native_lock(self):
        model = EvidenceFusion()
        states = torch.tensor([[[30., 179., 10., 359.]], [[31., -179., 20., 1.]]],
                              dtype=torch.float64)
        features = torch.zeros((1, 1, 27))
        exact = torch.ones(1, dtype=torch.bool)
        result = model(states, features, exact)
        self.assertTrue(torch.equal(result['states'], states[0]))
        self.assertTrue(torch.equal(result['weights'][1], torch.zeros_like(result['weights'][1])))
        singleton = model(states[:1], features[:0], exact)
        self.assertTrue(torch.equal(singleton['states'], states[0]))
        self.assertTrue(torch.equal(singleton['weights'], torch.ones_like(singleton['weights'])))
        with torch.no_grad():
            model.utility[-1].bias.fill_(2.)
        result = model(states, features, exact)
        np.testing.assert_allclose(result['states'].detach(),
            [[30.+2/3, -179.-2/3, 10.+20/3, 1/3]], atol=1e-12)
        np.testing.assert_allclose(result['weights'].detach().sum(0), 1.)
        model.exact_observation_lock = True
        self.assertTrue(torch.equal(model(states, features, exact)['states'], states[0]))

    def test_anchor_truth_masked_utility_starts_zero_relu_and_updates(self):
        model = EvidenceFusion()
        states = torch.tensor([[[30., 120., 10., 359.]],
                               [[30.01, 120., 20., 1.]]], dtype=torch.float64)
        features = torch.ones((1, 1, 27))
        exact = torch.zeros(1, dtype=torch.bool)
        valid = torch.ones((1, 4), dtype=torch.bool)
        wrong = evidence_fusion_loss(model(states, features, exact), states, states[0], valid)
        self.assertTrue((wrong['gain_target'] < 0).all())
        # Only speed supervised; nonfinite masked targets must be safely ignored.
        target = torch.tensor([[float('nan'), float('nan'), 20., float('nan')]], dtype=torch.float64)
        valid = torch.tensor([[False, False, True, False]])
        initial = model.utility[-1].bias.detach().clone()
        loss = evidence_fusion_loss(model(states, features, exact), states, target, valid)
        loss['total'].backward()
        self.assertTrue(torch.isfinite(loss['total']))
        self.assertLess(model.utility[-1].bias.grad[1].item(), 0.)
        self.assertEqual(model.utility[-1].bias.grad[0].item(), 0.)
        optimizer = torch.optim.Adam(model.parameters(), lr=.001)
        optimizer.step()
        self.assertFalse(torch.equal(initial, model.utility[-1].bias))
        self.assertGreater(model(states, features, exact)['weights'][1, 0, 1].item(), 0.)

    def test_millimetre_haversine_and_zero_error_gradients(self):
        prediction = torch.tensor([[1e-8, 0., 0., 0.], [0., 0., 0., 0.]],
                                  dtype=torch.float64, requires_grad=True)
        error = physical_errors(prediction, torch.zeros_like(prediction))
        self.assertAlmostEqual(error[0, 0].item(), 6371000.*np.deg2rad(1e-8), places=12)
        self.assertEqual(error[1, 0].item(), 0.)
        error.sum().backward()
        self.assertTrue(torch.isfinite(prediction.grad).all())

    def test_visible_features_and_hard_wrong_helper_ignore_reference(self):
        train, _ = synthetic_scenes(CooperativeConfig(length=64, max_observations=4))
        scene = prepare_evidence_scenes(train)[0]
        members = training_members(scene, 0, 'clean_plus_wrong')
        self.assertEqual(members[:2], [0, 1])
        self.assertNotEqual(scene['identities'][members[-1]], scene['identities'][0])
        before = fusion_inputs(scene['motion'], members)
        scene['target'][:] = -12345.
        scene['valid'][:] = False
        scene['motion']['sources'][:] = 'changed_source'
        after = fusion_inputs(scene['motion'], members)
        for first, second in zip(before, after):
            self.assertTrue(torch.equal(first, second))
        self.assertEqual(members, training_members(scene, 0, 'clean_plus_wrong'))


if __name__ == '__main__':
    unittest.main()
