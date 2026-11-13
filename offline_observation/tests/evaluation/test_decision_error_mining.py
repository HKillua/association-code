from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import numpy as np
import torch
from association.models.association_context import EdgeResidual, context_cache, fit_context_normalizer
from association.training.decision_error_mining import matched_masks, selected_bce, train
from tools.analysis.structured_association_pilot import predict


class DecisionMiningTests(unittest.TestCase):
    def test_partition_error_differs_from_single_edge_and_counts_match(self):
        pairs = np.column_stack(np.triu_indices(3, 1))
        labels = torch.tensor([1., 0., 0.], dtype=torch.float64)
        # A spurious positive edge is rejected by grouping: no decision error.
        logits = torch.tensor([5., 4., -.5], dtype=torch.float64, requires_grad=True)
        masks = matched_masks(logits, logits, pairs, labels, 3, threshold=2.)
        self.assertGreater(logits[1].item(), 2.)
        self.assertFalse(masks['DEC'].any())
        zero = selected_bce(logits, labels, masks['DEC'])
        zero.backward()
        torch.testing.assert_close(logits.grad, torch.zeros_like(logits))
        # A wrong high edge prevents the true pair; HARD uses the matched quotas.
        dec = torch.tensor([3., 6., -7.], dtype=torch.float64, requires_grad=True)
        hard = torch.tensor([3., -8., 7.], dtype=torch.float64, requires_grad=True)
        masks = matched_masks(dec, hard, pairs, labels, 3, threshold=2.)
        np.testing.assert_array_equal(masks['DEC'], [True, True, False])
        np.testing.assert_array_equal(masks['HARD'], [True, False, True])
        for positive in (False, True):
            self.assertEqual(sum(masks['DEC'][labels.numpy().astype(bool)==positive]), sum(masks['HARD'][labels.numpy().astype(bool)==positive]))
        self.assertIsNone(dec.grad)
        selected_bce(hard, labels, masks['HARD']).backward()
        self.assertIsNone(dec.grad)
        self.assertEqual(hard.grad[1].item(), 0.)

    def test_complete_exposure_real_optimization_saved_masks_and_label_free_inference(self):
        torch.set_num_threads(1); torch.manual_seed(20)
        pairs = np.column_stack(np.triu_indices(4, 1)); identities = np.array([0, 0, 1, 1]); caches = []
        for i in range(2):
            scores = np.array([1., 8., 3., 2., 7., 4.])
            cache = context_cache(np.random.default_rng(i).normal(size=(6,64)).astype(np.float32), pairs, scores, 4, 1.7)
            cache.update(scores=scores, scene_id=str(i), identities=identities, labels=identities[pairs[:,0]]==identities[pairs[:,1]])
            caches.append(cache)
        normalizer = fit_context_normalizer(caches, 'CTX'); initial = EdgeResidual().state_dict()
        heads, curve, sampling, masks = train(caches, initial, normalizer, 'cpu', lambda **kw:None)
        self.assertEqual(sampling['per_scene_passes'], [4,4]); self.assertEqual(sampling['pair_presentations_per_head'],48)
        self.assertEqual(sampling['updates_per_head'], 8)
        for row in curve:
            labels = caches[int(row['scene_id'])]['labels']
            for key in heads:
                selected = np.unpackbits(masks[f"{row['step']:04d}_{key}"])[:row['pairs']].astype(bool)
                self.assertEqual(int((selected & labels).sum()), row['heads'][key]['selected_positive'])
                self.assertEqual(int((selected & ~labels).sum()), row['heads'][key]['selected_negative'])
            for field in ('selected_positive','selected_negative'):
                self.assertEqual(row['heads']['HARD'][field], row['heads']['DEC'][field])
        for head in heads.values():
            self.assertTrue(any(not torch.equal(head.state_dict()[k], initial[k]) for k in initial))
        cache = {k:v for k,v in caches[0].items() if k not in ('identities','labels')}
        before = predict(cache, heads, normalizer, 'cpu')
        with TemporaryDirectory() as tmp:
            path = Path(tmp)/'heads.pt'; torch.save({k:h.state_dict() for k,h in heads.items()},path)
            states = torch.load(path,weights_only=True); restored = {k:EdgeResidual() for k in states}
            for k,head in restored.items(): head.load_state_dict(states[k])
            after = predict(cache,restored,normalizer,'cpu')
        for key in before: np.testing.assert_array_equal(before[key],after[key])

if __name__=='__main__': unittest.main()
