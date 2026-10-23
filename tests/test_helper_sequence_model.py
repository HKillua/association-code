import copy
import unittest

import numpy as np
import torch

from association.models.helper_sequence import (
    AGGREGATIONS, FEATURE_WIDTH, RAW_FEATURE_WIDTH, ROUTING_SEMANTICS,
    HelperSequencePosition, helper_sequence_inputs, initialize_from_reference,
)
from association.models.sequence_position import SequencePosition, sequence_inputs, sequence_objectives
from tests.test_sequence_position import motion_fixture


class HelperSequenceModelTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(20261023)
        self.reference = SequencePosition(backbone='transformer').state_dict()

    def model(self, mode):
        model = HelperSequencePosition(aggregation=mode)
        initialize_from_reference(model, self.reference, branch_seed=2026102301)
        return model

    def test_parameter_counts_common_initialization_and_independent_rng(self):
        self.assertEqual(sum(v.numel() for v in self.reference.values()), 105154)
        models = [self.model(mode) for mode in AGGREGATIONS]
        self.assertEqual([sum(p.numel() for p in model.parameters()) for model in models],
                         [107602, 107602, 107618])
        for model in models:
            before = torch.random.get_rng_state().clone()
            initialize_from_reference(model, self.reference, 19)
            self.assertTrue(torch.equal(torch.random.get_rng_state(), before))
            state = model.state_dict()
            for name, old in self.reference.items():
                current = state[name]
                if name == 'input_projection.0.weight':
                    self.assertEqual(current.shape, (64, FEATURE_WIDTH))
                    self.assertEqual(torch.count_nonzero(current[:, RAW_FEATURE_WIDTH:]), 0)
                    current = current[:, :RAW_FEATURE_WIDTH]
                self.assertTrue(torch.equal(old, current), name)
            self.assertGreater(torch.count_nonzero(model.phi[-1].weight), 0)
            self.assertEqual(model.routing_semantics, ROUTING_SEMANTICS)
        for name, value in models[0].state_dict().items():
            self.assertTrue(torch.equal(value, models[1].state_dict()[name]), name)
            self.assertTrue(torch.equal(value, models[2].state_dict()[name]), name)
        self.assertEqual(torch.count_nonzero(models[2].scorer.weight), 0)

    def test_visible_inputs_canonical_helpers_and_exact_fixed095(self):
        class VisibleOnly(dict):
            def __getitem__(self, key):
                if key in ('target', 'reference', 'identities', 'fusion_targets', 'source'):
                    raise AssertionError('truth, identity and source cannot enter inference')
                return super().__getitem__(key)
        motion = VisibleOnly(motion_fixture(), reference=object(), identities=object())
        args = helper_sequence_inputs(motion, [0, 2, 1])
        canonical = helper_sequence_inputs(motion, [0, 1, 2])
        historical = sequence_inputs(motion, [0, 1, 2])
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(args, canonical)))
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(args[:3], historical)))
        self.assertEqual(args[3].shape, (2, 3, 27))
        for mode in AGGREGATIONS:
            result = self.model(mode)(*args)
            self.assertTrue(torch.equal(result['states'], args[0]))
            self.assertTrue(torch.equal(result['states'][:, 2:], args[2][:, 2:]))
            self.assertTrue(torch.equal(result['helper_counts'], args[-1].sum(0)))
        for members in ([0., 1.], [0, 0], [0, 5], []):
            with self.assertRaises(ValueError):
                helper_sequence_inputs(motion, members)

    def test_single_supported_helper_and_empty_set_equivalence(self):
        features = torch.randn(3, 5, 27)
        support = torch.zeros((3, 5), dtype=torch.bool)
        support[0, 0] = support[1, 1] = support[2, 2] = True
        representations = [self.model(mode).helper_representation(features, support)
                           for mode in AGGREGATIONS]
        for branch, weights, count in representations:
            self.assertTrue(torch.equal(branch[3:], torch.zeros_like(branch[3:])))
            self.assertTrue(torch.equal(weights.sum(0), (count > 0).double()))
        for branch, weights, _ in representations[1:]:
            torch.testing.assert_close(branch, representations[0][0], rtol=1e-6, atol=1e-6)
            self.assertTrue(torch.equal(weights, representations[0][1]))
        for mode in AGGREGATIONS:
            branch, weights, count = self.model(mode).helper_representation(
                torch.empty((0, 5, 27)), torch.empty((0, 5), dtype=torch.bool))
            self.assertTrue(torch.equal(branch, torch.zeros((5, 16))))
            self.assertEqual(weights.shape, (0, 5))
            self.assertEqual(torch.count_nonzero(count), 0)

    def test_multiple_helpers_distinguish_joint_encoding_and_uniform_c_equals_b(self):
        features = torch.randn(4, 7, 27) * 4.
        support = torch.ones((4, 7), dtype=torch.bool)
        a, b, c = [self.model(mode) for mode in AGGREGATIONS]
        outputs = [model.helper_representation(features, support) for model in (a, b, c)]
        self.assertFalse(torch.allclose(outputs[0][0], outputs[1][0]))
        self.assertTrue(torch.equal(outputs[1][0], outputs[2][0]))
        self.assertTrue(torch.equal(outputs[1][1], outputs[2][1]))
        for model in (a, b, c):
            perm = torch.tensor([3, 1, 0, 2])
            reverse = model.helper_representation(features[perm], support[perm])
            expected = model.helper_representation(features, support)
            torch.testing.assert_close(reverse[0], expected[0], rtol=1e-6, atol=1e-6)
            torch.testing.assert_close(reverse[1], expected[1][perm], rtol=0., atol=0.)
        with torch.no_grad():
            c.scorer.weight.fill_(.1)
        branch, weights, _ = c.helper_representation(features, support)
        self.assertFalse(torch.equal(weights, outputs[1][1]))
        self.assertFalse(torch.equal(branch, outputs[1][0]))

    def test_unsupported_values_cannot_change_output_or_softmax(self):
        args = helper_sequence_inputs(motion_fixture(), [0, 1, 2])
        changed = list(args)
        changed[3] = args[3].clone()
        changed[3][~args[4]] = torch.nan
        for mode in AGGREGATIONS:
            model = self.model(mode)
            with torch.no_grad():
                # Exercise the branch, rather than merely the zero output head.
                model.input_projection[0].weight[:, RAW_FEATURE_WIDTH:].fill_(.01)
                model.output.weight.fill_(.02)
                if model.scorer is not None:
                    model.scorer.weight.fill_(.1)
            first, second = model(*args), model(*changed)
            self.assertTrue(torch.equal(first['states'], second['states']))
            self.assertTrue(torch.equal(first['helper_embedding'], second['helper_embedding']))
            weights = first['helper_weights']
            self.assertEqual(torch.count_nonzero(weights[~args[4]]), 0)
            torch.testing.assert_close(weights.sum(0), args[4].any(0).double(), rtol=0., atol=1e-15)
        changed[3][args[4]] = torch.inf
        with self.assertRaises(ValueError):
            self.model('pre_pool')(*changed)

    def test_empty_helpers_initialize_all_adam_state_and_zero_branch_gradients(self):
        for mode in AGGREGATIONS:
            for members in ([0], [0, 1, 2]):
                motion = motion_fixture()
                motion['support'][:] = False
                args = helper_sequence_inputs(motion, members)
                model = self.model(mode)
                optimizer = torch.optim.Adam(model.parameters(), lr=.001)
                result = model(*args)
                target = args[0].clone()
                target[:, 0] += .01
                valid = torch.ones_like(target, dtype=torch.bool)
                loss = sequence_objectives(result, target, valid)[0]['linear']
                loss.backward()
                for name, parameter in model.named_parameters():
                    self.assertIsNotNone(parameter.grad, name)
                    self.assertTrue(torch.isfinite(parameter.grad).all(), name)
                    if name.startswith(('phi.', 'scorer.')):
                        self.assertEqual(torch.count_nonzero(parameter.grad), 0, name)
                optimizer.step()
                self.assertEqual(len(optimizer.state), len(list(model.parameters())))

    def test_new_paths_receive_gradients_after_multiple_updates_with_partial_truth(self):
        args = helper_sequence_inputs(motion_fixture(), [0, 1, 2])
        for mode in AGGREGATIONS:
            model = self.model(mode)
            optimizer = torch.optim.Adam(model.parameters(), lr=.001)
            target = args[0].clone()
            target[:2, 0] += torch.tensor([.02, -.03])
            target[:, 2:] = torch.nan
            target[2, :2] = torch.nan
            valid = torch.zeros_like(target, dtype=torch.bool)
            valid[:2, :2] = True
            seen_phi = seen_projection = seen_scorer = False
            for _ in range(8):
                optimizer.zero_grad(set_to_none=True)
                result = model(*args)
                loss = sequence_objectives(result, target, valid)[0]['linear']
                loss.backward()
                self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()
                                    for p in model.parameters()))
                seen_phi |= any(bool(torch.count_nonzero(p.grad)) for p in model.phi.parameters())
                seen_projection |= bool(torch.count_nonzero(
                    model.input_projection[0].weight.grad[:, RAW_FEATURE_WIDTH:]))
                if model.scorer is not None:
                    seen_scorer |= bool(torch.count_nonzero(model.scorer.weight.grad))
                optimizer.step()
            self.assertTrue(seen_phi and seen_projection)
            self.assertTrue(seen_scorer or model.scorer is None)

    def test_rejects_noninitial_or_incompatible_reference_and_noncontract_model(self):
        model = self.model('pre_pool')
        for mutate in (
            lambda state: state.pop('output.bias'),
            lambda state: state.update(extra=torch.zeros(1)),
            lambda state: state.update({'output.bias': torch.ones(2)}),
            lambda state: state.update({'input_projection.0.weight': torch.zeros(64, 92)}),
            lambda state: state.update({'output.weight': state['output.weight'].double()}),
        ):
            invalid = copy.deepcopy(self.reference)
            mutate(invalid)
            with self.assertRaises(ValueError):
                initialize_from_reference(model, invalid, 13)
        for kwargs in (dict(aggregation='max_pool'), dict(backbone='lstm'),
                       dict(hidden_size=128), dict(initial_gate=.9)):
            with self.assertRaises(ValueError):
                HelperSequencePosition(**kwargs)


if __name__ == '__main__':
    unittest.main()
