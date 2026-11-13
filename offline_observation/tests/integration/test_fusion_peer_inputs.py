"""Semantic peer interface checks; these are not MTAD performance evidence."""
import unittest

import numpy as np
import torch

from association.models.fusion_peer_inputs import sensor_segments, ptfm_segment_inputs, deepaf_visible_inputs
from association.models.ptfm import PTFM, local_states
from tests.integration.test_source_risk_pilot import scene_arrays
from tools.experiments.fusion_peer_training_probe import ptfm_loss


class PeerInputTests(unittest.TestCase):
    def test_two_sensor_streams_no_cross_pool_and_gap_segmentation(self):
        motion, _, _, _ = scene_arrays('1')
        motion['support'][:] = True
        motion['support'][1, 10] = False
        blocks = sensor_segments(motion, [0, 1])
        self.assertEqual([b['query_indices'].tolist() for b in blocks], [list(range(10)), list(range(11, 64))])
        np.testing.assert_array_equal(blocks[0]['left'], motion['states'][0, :10])
        np.testing.assert_array_equal(blocks[0]['right'], motion['states'][1, :10])
        a, b, context, origin = ptfm_segment_inputs(blocks[0])
        swapped = ptfm_segment_inputs(blocks[0], swapped=True)
        torch.testing.assert_close(swapped[0], b)
        torch.testing.assert_close(swapped[1], a)
        torch.testing.assert_close(swapped[3], origin)
        motion['sources'] = np.where(motion['sources'] == 'source-a', 'new-X', 'new-Y')
        renamed = sensor_segments(motion, [0, 1])
        np.testing.assert_array_equal(renamed[0]['left'], blocks[0]['left'])
        self.assertEqual(sensor_segments(motion, [0, 2]), [])

    def test_new_ptfm_branches_train_and_deepaf_uses_anchor_bootstrap(self):
        motion, ev, _, _ = scene_arrays('1')
        block = max(sensor_segments(motion, [0, 1]), key=lambda b: len(b['query_indices']))
        row = dict(segment=block, target=ev['target_states'][0], valid=ev['target_valid'][0])
        normalizer = dict(mean=[0]*4, scale=[1]*4, split='T_core', truth_used=False)
        model = PTFM(normalizer, sensor_branches=True)
        self.assertIsNone(model.context_gate)
        loss, detail = ptfm_loss(model, [row], 'cpu')
        loss.backward()
        self.assertTrue(any(p.grad is not None and (p.grad != 0).any() for p in model.anchor.parameters()))
        self.assertTrue(any(p.grad is not None and (p.grad != 0).any() for p in model.helper.parameters()))
        inputs = deepaf_visible_inputs(motion, [0, 1])
        np.testing.assert_array_equal(inputs['origin_latlon'].numpy(), motion['states'][0, 0, :2])
        np.testing.assert_array_equal(inputs['base_states'][:, :2].numpy(), motion['states'][0, :, :2])
        self.assertTrue(inputs['candidate_valid'][0].all())
        self.assertNotIn('target', inputs)
