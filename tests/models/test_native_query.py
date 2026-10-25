import copy
import unittest

import numpy as np
import torch

from association.models.native_query import NativeQueryPosition


def inputs():
    query = np.array([1000., 1020., 1040., 1060., 1080.])
    base = torch.tensor([[30. + .001 * i, 120. + .002 * i, 10., 45.]
                         for i in range(len(query))], dtype=torch.float64)
    sources = []
    for source, shift in enumerate((0., 7.)):
        sources.append(dict(source_id=source, event_ids=[0, 1, 2],
            times=np.array([990., 1030., 1090.]) + shift,
            values=np.array([[30., 120., 9., 40.], [30.003, 120.005, 12., 60.],
                             [30.008, 120.009, 15., 80.]]) + np.array([source * .002, 0., 0., 0.]),
            valid=np.ones((3, 4), dtype=bool)))
    return sources, base, query


class NativeQueryTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(20261026)

    def model(self, mode, active=True):
        model = NativeQueryPosition(pooling=mode)
        if active:
            with torch.no_grad():
                model.output.weight.normal_(0., .01)
                model.output.bias.copy_(torch.tensor([.02, -.01]))
        return model

    def test_zero_head_and_all_empty_return_exact_frozen_base(self):
        events, base, query = inputs()
        base.requires_grad_()
        for mode in ('mean', 'query'):
            model = self.model(mode, active=False)
            result = model(events, base, query)
            self.assertTrue(torch.equal(result['states'], base))
            self.assertFalse(result['base_states'].requires_grad)
            self.assertEqual(torch.count_nonzero(result['residual_m']), 0)
            model.output.bias.data.fill_(1.)
            for empty in ([], [dict(times=np.array([np.nan]), values=np.full((1, 4), np.nan),
                                    valid=np.zeros((1, 4), dtype=bool))]):
                result = model(empty, base, query)
                self.assertTrue(torch.equal(result['states'], base))
                self.assertEqual(torch.count_nonzero(result['residual_m']), 0)
                result['states'][:, :2].sum().backward()
                self.assertIsNone(base.grad)

    def test_padding_invalid_values_and_empty_sources_cannot_change_result(self):
        events, base, query = inputs()
        events[0]['valid'][1, 2:] = False
        modified = copy.deepcopy(events)
        modified[0]['values'][1, 2:] = np.nan
        for source in modified:
            source['times'] = np.r_[source['times'], np.nan, -1.e30]
            source['values'] = np.vstack((source['values'], np.full((2, 4), np.inf)))
            source['valid'] = np.vstack((source['valid'], np.zeros((2, 4), dtype=bool)))
        modified.append(dict(times=np.array([np.nan]), values=np.full((1, 4), np.nan),
                             valid=np.zeros((1, 4), dtype=bool)))
        for mode in ('mean', 'query'):
            model = self.model(mode)
            a, b = model(events, base, query), model(modified, base, query)
            torch.testing.assert_close(a['residual_m'], b['residual_m'], rtol=0., atol=1e-5)
            self.assertEqual(a['diagnostics']['event_count'], b['diagnostics']['event_count'])
            self.assertEqual(b['diagnostics']['active_source_count'], 2)
            self.assertFalse(b['diagnostics']['patch_valid'][-1].any())

    def test_source_reordering_renaming_and_event_reordering_are_symmetric(self):
        events, base, query = inputs()
        modified = copy.deepcopy(events[::-1])
        for source in modified:
            source['source_id'] = 'arbitrary-unseen-name'
            source['event_ids'] = ['a', 'b', 'c']
            for key in ('times', 'values', 'valid'):
                source[key] = source[key][::-1].copy()
        for mode in ('mean', 'query'):
            model = self.model(mode)
            a, b = model(events, base, query), model(modified, base, query)
            torch.testing.assert_close(a['states'], b['states'], rtol=0., atol=1e-8)
            torch.testing.assert_close(a['residual_m'], b['residual_m'], rtol=0., atol=2e-5)
            torch.testing.assert_close(a['diagnostics']['patch_counts'].flip(0), b['diagnostics']['patch_counts'])

    def test_real_time_is_used_and_outside_events_are_kept(self):
        events, base, query = inputs()
        changed = copy.deepcopy(events)
        changed[0]['times'][1] += 1.  # Same patch, different true event time.
        for mode in ('mean', 'query'):
            model = self.model(mode)
            a, b = model(events, base, query), model(changed, base, query)
            self.assertFalse(torch.allclose(a['residual_m'], b['residual_m']))
            self.assertEqual(a['diagnostics']['event_count'], 6)
            self.assertEqual(a['diagnostics']['patch_counts'][:, 0].sum(), 2)
            self.assertEqual(a['diagnostics']['patch_counts'][:, -1].sum(), 2)
            translated = copy.deepcopy(events)
            for source in translated:
                source['times'] += 1.e9
            c = model(translated, base, query + 1.e9)
            self.assertTrue(torch.equal(a['states'], c['states']))

    def test_query_summary_masks_key_columns_including_padding(self):
        model = self.model('query')
        q = torch.randn(1, 1, 64)
        tokens = torch.cat((q, torch.randn(1, 2, 64)), 1)
        valid = torch.tensor([[True, True, False]])
        a = model.pooler(q, tokens, tokens, key_padding_mask=~valid, need_weights=False)[0]
        tokens[:, 2] = 1.e6
        b = model.pooler(q, tokens, tokens, key_padding_mask=~valid, need_weights=False)[0]
        self.assertTrue(torch.equal(a, b))
        c = model.pooler(q, tokens[:, :2], tokens[:, :2], need_weights=False)[0]
        torch.testing.assert_close(a, c, rtol=0., atol=1e-7)

    def test_parameter_match_common_initialization_and_active_gradients(self):
        events, base, query = inputs()
        models = []
        for mode in ('mean', 'query'):
            torch.manual_seed(7)
            models.append(self.model(mode, active=False))
        counts = [sum(p.numel() for p in model.parameters()) for model in models]
        self.assertLess(abs(counts[0] - counts[1]) / max(counts), .05)
        self.assertEqual(abs(counts[0] - counts[1]), 64)
        for name, value in models[0].state_dict().items():
            if not name.startswith('pooler.'):
                self.assertTrue(torch.equal(value, models[1].state_dict()[name]), name)
        for model in models:
            # Nonzero output isolates whether the body has a live gradient;
            # zero initialization necessarily blocks the body's first update.
            with torch.no_grad():
                model.output.weight.normal_(0., .01)
            result = model(events, base, query)
            result['residual_m'].square().mean().backward()
            for name, parameter in model.named_parameters():
                self.assertIsNotNone(parameter.grad, name)
                self.assertTrue(torch.isfinite(parameter.grad).all(), name)
                self.assertGreater(float(parameter.grad.abs().sum()), 0., name)
            self.assertTrue(torch.equal(result['states'][:, 2:], base[:, 2:]))


if __name__ == '__main__':
    unittest.main()
