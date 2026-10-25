"""Visible-input invariance and physical endpoints used by position models."""
import unittest
import numpy as np
import torch
from association.evaluation.fusion_controls import ordinary_fusion
from association.models.support_gate import SupportGate, gate_inputs, mix_gate_states
from tests.fixtures.learning import fixture_motion

class PositionInputTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)


    def test_visible_features_mean_endpoint_and_helper_order(self):
        motion = fixture_motion()
        class VisibleOnly(dict):
            def __getitem__(self, key):
                if key in ('target', 'reference', 'identities', 'fusion_targets'):
                    raise AssertionError('truth must never enter forward input preparation')
                return super().__getitem__(key)
        visible = VisibleOnly(motion, identities=object(), reference=object())
        first = gate_inputs(visible, [0, 1, 2])
        second = gate_inputs(visible, [0, 2, 1])
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(first, second)))
        expected = ordinary_fusion(motion['states'], motion['nearest_seconds'], motion['support'], [[0, 1, 2]])
        self.assertTrue(np.array_equal(first[1].numpy(), expected['predictions']['mean_supported'][0]))
        self.assertEqual(first[2].shape, (3, 56))
        self.assertEqual(first[3].tolist(), [False, True, False])
        self.assertTrue(torch.equal(first[2][2], torch.zeros(56)))
        result = SupportGate()(*first)
        self.assertTrue(torch.allclose(result['gate'], torch.full((3,), .95), atol=1e-7, rtol=0))
        self.assertTrue(torch.equal(result['states'][[0, 2], :2], first[0][[0, 2], :2]))
        self.assertTrue(torch.equal(result['states'][:, 2:], first[1][:, 2:]))
        # An alternative anchor remains first; its helpers are sorted separately.
        third = gate_inputs(motion, [2, 1, 0])
        self.assertTrue(torch.equal(third[0], torch.tensor(motion['states'][2])))
        with self.assertRaises(ValueError):
            gate_inputs(motion, [-1, 1])
        with self.assertRaises(ValueError):
            gate_inputs(motion, [0., 1.])


    def test_exact_endpoints_dateline_and_singleton(self):
        anchor = torch.tensor([[10., 179., 1., 20.], [10., -179., 1., 20.]], dtype=torch.float64)
        mean = torch.tensor([[12., -179., 7., 30.], [12., 179., 7., 30.]], dtype=torch.float64)
        eligible = torch.ones(2, dtype=torch.bool)
        for value, expected in ((0., anchor[:, :2]), (1., mean[:, :2])):
            result = mix_gate_states(anchor, mean, torch.full((2,), value), eligible)
            self.assertTrue(torch.equal(result[:, :2], expected))
            self.assertTrue(torch.equal(result[:, 2:], mean[:, 2:]))
        midpoint = mix_gate_states(anchor, mean, torch.full((2,), .5), eligible)
        self.assertTrue(torch.equal(midpoint[:, 1], torch.full((2,), -180., dtype=torch.float64)))
        motion = fixture_motion()
        arguments = gate_inputs(motion, [0])
        result = SupportGate()(*arguments)
        self.assertFalse(result['eligible'].any())
        self.assertTrue(torch.equal(result['states'], arguments[0]))
        self.assertTrue(torch.equal(arguments[2], torch.zeros((3, 56))))


