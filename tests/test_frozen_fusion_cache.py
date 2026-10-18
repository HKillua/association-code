"""Frozen completion reuse against the uncached per-pair training computation."""
import copy
import unittest
from unittest.mock import patch

import torch

from association.fusion import FusionLoss
from association.fusion_training import fusion_training_parameters, fusion_training_step
from association.reference import fit_normalizer, normalize_scene
from association.timed_models import TimedAssociationModel
from tests import test_fusion_training as fixtures
from tests.test_timed_pipeline import timed_scene


def uncached_frozen_step(modules, optimizer, batch, microbatch, clip):
    """Reference computation from the prior frozen loop: recompute every chunk.

    Keep fusion call ordering (including dropout), global feature denominators
    and once-only variance regularization; no cache or optimized helper used.
    """
    model, fusion, objective = modules
    parameters = fusion_training_parameters(*modules, stage='TFM_FROZEN')
    optimizer.zero_grad(set_to_none=True)
    total = 0.
    counts = batch['valid'].sum((0, 1))
    for start in range(0, len(batch['pairs']), microbatch):
        stop = start + microbatch
        nodes, inverse = torch.unique(batch['pairs'][start:stop], return_inverse=True)
        with torch.no_grad():
            completed = model.complete(batch['values'][nodes], batch['observed'][nodes])
        states = fusion.forward_pairs(completed, inverse.reshape(-1, 2))['states']
        loss = objective(states, batch['target'][start:stop], batch['valid'][start:stop],
                         normalization_counts=counts, regularizer_weight=float(start == 0))['total']
        loss.backward()
        total += float(loss.detach())
    norm = torch.nn.utils.clip_grad_norm_(parameters, clip, error_if_nonfinite=True)
    optimizer.step()
    return total, float(norm)


class FrozenFusionCacheTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original_num_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.original_num_threads)

    def fixture(self, *, native=False, dtype=torch.float64, pairs=None):
        fixture = fixtures.FusionTrainingTests()
        model, fusion, _ = fixture.models(dropout=.2)
        objective = FusionLoss(position_loss_mode='standard_smooth_l1')
        original = fixture.batch()
        if native:
            torch.manual_seed(18)
            model = TimedAssociationModel(model.config, max_observations=3)
            scene = timed_scene('synthetic', 'T_core')
            x, mask = normalize_scene(scene, fit_normalizer([scene]))
            original['values'], original['observed'] = torch.tensor(x), torch.tensor(mask)
        modules = tuple(module.to(dtype=dtype) for module in (model, fusion, objective))
        pairs = [[0, 1], [0, 1], [2, 3]] if pairs is None else pairs
        count = len(pairs)
        target = torch.tensor([30., 120., 5., 359.], dtype=dtype).expand(count, 4, 4).clone()
        valid = torch.ones_like(target, dtype=torch.bool)
        # Unequal masks exercise the unchanged effective-batch denominators.
        valid[0, :, 0] = False
        valid[0, :2, 1] = False
        valid[-1, :2, 2:] = False
        target[~valid] = float('nan')
        batch = dict(values=original['values'].to(dtype=dtype), observed=original['observed'],
            pairs=torch.tensor(pairs), labels=torch.ones(count, dtype=dtype), target=target, valid=valid)
        optimizer = torch.optim.Adam(fusion_training_parameters(*modules, stage='TFM_FROZEN'), lr=.001)
        return modules, optimizer, batch

    def run_step(self, modules, optimizer, batch, **kwargs):
        return fusion_training_step(*modules, optimizer, stage='TFM_FROZEN', **batch,
            pair_microbatch=kwargs.pop('pair_microbatch', 1), track_microbatch=1,
            completion_weight=0, gradient_clip=1., **kwargs)

    def test_repeated_pair_reduces_actual_completion_calls_and_tracks(self):
        modules, optimizer, batch = self.fixture()
        with patch.object(modules[0], 'complete', wraps=modules[0].complete) as complete:
            report = self.run_step(modules, optimizer, batch)
        self.assertEqual(complete.call_count, 2)  # Old loop: 3 calls / 6 tracks.
        self.assertEqual([len(call.args[0]) for call in complete.call_args_list], [2, 2])
        self.assertEqual(report['completion_cache'], dict(scope='step_local_cpu', hits=2,
            misses=4, forward_calls=2, max_forward_tracks=2, peak_entries=4, peak_bytes=4 * 4 * 8 * 8))
        self.assertEqual(report['microbatches'], 3)
        self.assertEqual(report['pair_count'], 3)  # Repeated training draws remain.

    def test_partial_hits_keep_microbatch_bound_and_only_complete_missing_nodes(self):
        pairs = [[0, 1], [1, 2], [2, 3], [0, 3], [0, 1]]
        modules, optimizer, batch = self.fixture(native=True, pairs=pairs)
        with patch.object(modules[0], 'complete', wraps=modules[0].complete) as complete:
            report = self.run_step(modules, optimizer, batch)
        self.assertEqual([len(call.args[0]) for call in complete.call_args_list], [2, 1, 1])
        self.assertEqual(report['completion_cache']['misses'], 4)
        self.assertEqual(report['completion_cache']['hits'], 6)
        self.assertEqual(report['completion_cache']['max_forward_tracks'], 2)

    def test_dropout_loss_gradients_adam_updates_and_rng_match_uncached_computation(self):
        for native in (False, True):
            for dtype in (torch.float64, torch.float32):
                for microbatch in (1, 2):
                    with self.subTest(native=native, dtype=dtype, microbatch=microbatch):
                        pairs = [[0, 1], [1, 2], [2, 3], [0, 3], [0, 1]]
                        modules, optimizer, batch = self.fixture(native=native, dtype=dtype, pairs=pairs)
                        reference = copy.deepcopy(modules)
                        ref_optimizer = torch.optim.Adam(
                            fusion_training_parameters(*reference, stage='TFM_FROZEN'), lr=.001)
                        torch.manual_seed(711)
                        report = self.run_step(modules, optimizer, batch, pair_microbatch=microbatch)
                        actual_rng = torch.get_rng_state()
                        torch.manual_seed(711)
                        loss, norm = uncached_frozen_step(reference, ref_optimizer, batch, microbatch, 1.)
                        self.assertTrue(torch.equal(actual_rng, torch.get_rng_state()))
                        tolerance = dict(atol=1e-10, rtol=1e-9) if dtype == torch.float64 else dict(atol=2e-6, rtol=2e-5)
                        torch.testing.assert_close(torch.tensor(report['loss'], dtype=dtype),
                                                   torch.tensor(loss, dtype=dtype), **tolerance)
                        torch.testing.assert_close(torch.tensor(report['gradient_norm'], dtype=dtype),
                                                   torch.tensor(norm, dtype=dtype), **tolerance)
                        for actual_module, expected_module in zip(modules, reference):
                            for actual, expected in zip(actual_module.parameters(), expected_module.parameters()):
                                torch.testing.assert_close(actual, expected, **tolerance)
                                if actual.grad is None or expected.grad is None:
                                    self.assertIsNone(actual.grad)
                                    self.assertIsNone(expected.grad)
                                else:
                                    torch.testing.assert_close(actual.grad, expected.grad, **tolerance)
                        for actual, expected in zip(optimizer.state.values(), ref_optimizer.state.values()):
                            for key in actual:
                                torch.testing.assert_close(actual[key], expected[key], **tolerance)
                        self.assertFalse(modules[0].training)
                        self.assertTrue(modules[1].training)  # Fusion dropout was on.

    def test_cache_is_discarded_between_steps_and_input_changes(self):
        modules, optimizer, batch = self.fixture(native=True)
        completed = []
        original = modules[0].complete
        def capture(values, observed):
            result = original(values, observed)
            completed.append(result.detach().clone())
            return result
        with patch.object(modules[0], 'complete', side_effect=capture):
            self.run_step(modules, optimizer, batch)
            batch['values'][0, :2, :4] += .2
            report = self.run_step(modules, optimizer, batch)
        self.assertEqual(len(completed), 4)
        self.assertEqual(report['completion_cache']['misses'], 4)
        self.assertFalse(torch.equal(completed[0][0], completed[2][0]))

    def test_cache_hits_do_not_skip_deadline_or_partially_step(self):
        modules, optimizer, batch = self.fixture()
        before = [p.detach().clone() for module in modules for p in module.parameters()]
        calls = []
        def check():
            calls.append(1)
            if len(calls) == 2:  # The second chunk would be a complete cache hit.
                raise TimeoutError('deadline')
        with patch.object(optimizer, 'step', wraps=optimizer.step) as step, \
             patch.object(modules[0], 'complete', wraps=modules[0].complete) as complete:
            with self.assertRaisesRegex(TimeoutError, 'deadline'):
                self.run_step(modules, optimizer, batch, check=check)
            self.assertEqual(step.call_count, 0)
            self.assertEqual(complete.call_count, 1)
        for parameter, prior in zip((p for m in modules for p in m.parameters()), before):
            torch.testing.assert_close(parameter, prior, rtol=0, atol=0)
            self.assertIsNone(parameter.grad)
        # A later independent call cannot see the aborted call's cache.
        with patch.object(modules[0], 'complete', wraps=modules[0].complete) as complete:
            self.run_step(modules, optimizer, batch)
        self.assertEqual(complete.call_count, 2)

    def test_joint_keeps_recomputation_and_trainable_completion(self):
        modules, _, batch = self.fixture()
        optimizer = torch.optim.Adam(fusion_training_parameters(*modules, stage='JOINT'), lr=.001)
        with patch.object(modules[0], 'complete', wraps=modules[0].complete) as complete:
            report = fusion_training_step(*modules, optimizer, stage='JOINT', **batch,
                pair_microbatch=1, track_microbatch=1, completion_weight=0, gradient_clip=1.)
        self.assertEqual(complete.call_count, 3)
        self.assertNotIn('completion_cache', report)
        self.assertTrue(any(p.grad is not None for p in modules[0].tcm_encoder.parameters()))


if __name__ == '__main__':
    unittest.main()
