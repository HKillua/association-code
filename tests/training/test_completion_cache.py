import unittest

import numpy as np
import torch

from association.models.tmtaff import AssociationModel, ModelConfig
from association.training.batches import FrozenCompletionCache


class CompletionCacheTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(31)
        self.model = AssociationModel(ModelConfig(d_model=8,nhead=2,num_layers=1,
                                     dim_feedforward=16,length=6,dropout=0.1))
        self.model.set_completion_trainable(False,include_embedding=True)
        self.model.train()
        values = np.random.default_rng(4).normal(size=(3,6,4)).astype(np.float32)
        observed = np.zeros((3,6),dtype=bool)
        observed[:,1:5] = True
        self.normalized = [(values,observed)]
        self.keys = [(0,0),(0,1),(0,2)]

    def test_cached_completion_matches_uncached_tam_outputs_and_gradients(self):
        cache = FrozenCompletionCache(self.model,self.normalized,'cpu',4096,2)
        with torch.no_grad():
            direct = self.model.complete(torch.as_tensor(self.normalized[0][0]),
                                         torch.as_tensor(self.normalized[0][1]))
        cached = cache.get(self.keys)
        torch.testing.assert_close(cached,direct,rtol=1e-6,atol=1e-6)
        torch.testing.assert_close(cache.get(self.keys),cached,rtol=0,atol=0)
        gradients, outputs = [], []
        for values in (direct,cached):
            self.model.zero_grad(set_to_none=True)
            torch.manual_seed(99)
            output = self.model.tam_encoder(values)
            output.square().sum().backward()
            gradients.append([p.grad.clone() for p in self.model.tam_encoder.parameters()])
            outputs.append(output.detach())
        torch.testing.assert_close(outputs[0],outputs[1],rtol=1e-6,atol=1e-6)
        for a,b in zip(*gradients):
            torch.testing.assert_close(a,b,rtol=1e-5,atol=1e-5)
        self.assertEqual(cache.hits,3)
        self.assertEqual(cache.misses,3)

    def test_capacity_eviction_and_no_reuse_after_unfreeze_or_weight_change(self):
        track_bytes = 6*8*4
        cache = FrozenCompletionCache(self.model,self.normalized,'cpu',track_bytes,2)
        cache.get(self.keys)
        self.assertEqual(cache.bytes,track_bytes)
        self.assertEqual(len(cache.entries),1)
        self.assertEqual(cache.evictions,2)
        self.model.set_completion_trainable(True,include_embedding=True)
        with self.assertRaisesRegex(ValueError,'frozen'):
            cache.get(self.keys)
        self.model.set_completion_trainable(False,include_embedding=True)
        with torch.no_grad():
            self.model.input_embedding.weight.add_(1)
        with self.assertRaisesRegex(ValueError,'stale'):
            cache.get(self.keys)
        cache.clear()
        self.assertEqual(cache.bytes,0)


if __name__ == '__main__':
    unittest.main()
