"""Train-only manual time scaling, compatibility and matched-arm regressions.

All inputs are tiny synthetic native scenes. No accelerator, MTAD archive,
reference fitting, or validation-based scale selection is performed.
"""
import copy
from dataclasses import asdict, replace
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from association.models.cooperative import CooperativeConfig, CooperativeModel
from association.training import cooperative as training
from association.training.common import fit_normalizer, normalize_scene
from tools.experiments.cooperative_fit import synthetic_scenes


class CooperativeTimeScalingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def setUp(self):
        self.config = CooperativeConfig(length=4, max_observations=4, d_model=8,
            nhead=2, num_layers=1, dim_feedforward=16, head_hidden=8, dropout=0.)
        self.scenes, self.validation = synthetic_scenes(self.config)
        self.normalizer = fit_normalizer(self.scenes)
        self.packed = [normalize_scene(s, self.normalizer) for s in self.scenes]
        self.scaled_config = replace(self.config, support_time_policy='train_absmax_v1')

    def model(self, config=None, scale=None):
        torch.manual_seed(20)
        return CooperativeModel(config or self.config, self.normalizer,
                                support_time_scale=scale).eval()

    def encoded(self, model):
        values, observed = self.packed[0]
        return model.encode(torch.from_numpy(values), torch.from_numpy(observed))

    def test_legacy_preserves_manual_feature_formula_and_old_state_keys(self):
        model = self.model()
        self.assertEqual(model.config.support_time_policy, 'legacy')
        self.assertNotIn('support_time_scale', model.state_dict())
        encoded = self.encoded(model)
        pairs = torch.tensor([[0, 1], [0, 2]])
        first, last, count = model._support(encoded)
        support = torch.stack((first, last, last-first, count.log1p()), -1)
        left, right = encoded['pooled'][pairs[:, 0]], encoded['pooled'][pairs[:, 1]]
        sl, sr = support[pairs[:, 0]], support[pairs[:, 1]]
        explicit_input = torch.cat(((left-right).abs(), left*right, (sl-sr).abs(),
                                    sl*sr, left.new_zeros((len(pairs), 4))), -1)
        expected = model.identity_base(explicit_input).squeeze(-1)
        torch.testing.assert_close(model.score_pairs(encoded, pairs), expected, rtol=0, atol=0)
        q = encoded['queries']
        expected_support = torch.stack((q-first[:, None], last[:, None]-q,
            (last-first)[:, None].expand_as(q), count.log1p()[:, None].expand_as(q)), -1)
        torch.testing.assert_close(model._query_support(encoded), expected_support, rtol=0, atol=0)


    def test_scale_fits_only_valid_native_and_all_query_times(self):
        packed = copy.deepcopy(self.packed)
        values, observed = packed[0]
        m = self.config.max_observations
        # Max queries must be included even though query observed flags are false.
        values[:, -1, 4] = 200.
        # Valid native times can exceed queries; fit uses absolute magnitude.
        values[0, 0, 4] = -700.
        # All physical-state channels and reference contents are irrelevant to S.
        values[..., :4] = np.nan
        native_tau = values[:, :m, 4]
        native_tau[~observed[:, :m]] = np.inf
        scenes = copy.deepcopy(self.scenes)
        scenes[0]['fusion_targets'] = None
        result = training.fit_support_time_scaling(scenes, packed, m)
        self.assertEqual(result['policy'], 'train_absmax_v1')
        self.assertEqual(result['scale_tau'], 700.)
        values[0, 0, 4] = 0.
        result = training.fit_support_time_scaling(scenes, packed, m)
        self.assertEqual(result['scale_tau'], 200.)
        values[..., 4] = 0.25
        result = training.fit_support_time_scaling(scenes, packed, m)
        self.assertEqual(result['scale_tau'], 1.)


    def test_only_manual_mlp_paths_change_not_encoding_physical_support_or_plain_decode(self):
        legacy, scaled = self.model(), self.model(self.scaled_config, 10.)
        for name, value in legacy.named_parameters():
            torch.testing.assert_close(value, dict(scaled.named_parameters())[name], rtol=0, atol=0)
        self.assertEqual(sum(p.numel() for p in legacy.parameters()),
                         sum(p.numel() for p in scaled.parameters()))
        a, b = self.encoded(legacy), self.encoded(scaled)
        for key in a:
            torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)
        torch.testing.assert_close(legacy.summarize(a), scaled.summarize(b), rtol=0, atol=0)
        for left, right in zip(legacy._support(a), scaled._support(b)):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
        for route in ('plain', 'support'):
            torch.testing.assert_close(legacy.decode_group(a, [0, 1], routing=route),
                                       scaled.decode_group(b, [0, 1], routing=route), rtol=0, atol=0)
        pairs = torch.tensor([[0, 1], [0, 2]])
        # Both identity heads, utility and ordinary gate must use the new rule.
        for use_utility in (False, True):
            self.assertFalse(torch.equal(legacy.score_pairs(a, pairs, use_utility),
                                         scaled.score_pairs(b, pairs, use_utility)))
        self.assertFalse(torch.equal(legacy.utility(a, pairs), scaled.utility(b, pairs)))
        members = torch.tensor([0, 1])
        self.assertFalse(torch.equal(legacy._route_weights(a, members, 'learned', None, None),
                                     scaled._route_weights(b, members, 'learned', None, None)))
        # Unscaled count features must not silently change with time scaling.
        torch.testing.assert_close(legacy._query_support(a)[..., 3],
                                   scaled._query_support(b)[..., 3], rtol=0, atol=0)

    def test_historical_checkpoint_without_policy_or_scale_loads_identically(self):
        model = self.model()
        config = asdict(self.config)
        config.pop('support_time_policy')
        payload = dict(format='cooperative-physical-v1', model_config=config,
                       model_state=model.state_dict(), normalizer=self.normalizer)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'historical.pt'
            torch.save(payload, path)
            restored, _ = training.load_cooperative_checkpoint(path, device='cpu')
        self.assertEqual(restored.config.support_time_policy, 'legacy')
        self.assertEqual(set(restored.state_dict()), set(model.state_dict()))
        for key in model.state_dict():
            torch.testing.assert_close(restored.state_dict()[key], model.state_dict()[key], rtol=0, atol=0)
        a, b = self.encoded(model), self.encoded(restored)
        torch.testing.assert_close(model.decode_group(a, [0, 1]), restored.decode_group(b, [0, 1]), rtol=0, atol=0)
        torch.testing.assert_close(model.score_pairs(a, [[0, 1]]), restored.score_pairs(b, [[0, 1]]), rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
