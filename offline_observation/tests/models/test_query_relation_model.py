import copy
import unittest

import torch

from association.models.query_relation import (
    BRANCH_PREFIXES, MODES, QueryRelationPosition, helper_sequence_inputs,
    initialize_from_reference, observable_relation_bias,
)
from association.models.sequence_position import SequencePosition, sequence_objectives
from tests.training.test_sequence_position import motion_fixture


class QueryRelationModelTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(20261023)
        self.teacher = SequencePosition(backbone='transformer')
        with torch.no_grad():
            self.teacher.output.weight.normal_(0., .02)
            self.teacher.output.bias.copy_(torch.tensor([.12, -.08]))
        self.reference = copy.deepcopy(self.teacher.state_dict())

    def model(self, mode, *, active=False):
        model = QueryRelationPosition(mode=mode)
        initialize_from_reference(model, self.reference, 20359023)
        if active:
            with torch.no_grad():
                # Activate the new branch without depending on a training run.
                values = torch.arange(64 * 16, dtype=torch.float32).reshape(64, 16)
                model.branch_projection.weight.copy_(.02 * torch.sin(values))
        return model

    def random_inputs(self, helpers=4, queries=5):
        features = torch.randn(queries, 76)
        helper = torch.randn(helpers, queries, 27)
        support = torch.ones((helpers, queries), dtype=torch.bool)
        base = torch.tensor([[20., 179.7, 5., 350.]] * queries, dtype=torch.float64)
        return base, features, base.clone(), helper, support

    def test_source_final_is_exact_both_modes_share_all_parameters_and_rng_is_preserved(self):
        arguments = helper_sequence_inputs(motion_fixture(), [0, 1, 2])
        models = [self.model(mode) for mode in MODES]
        self.assertGreater(torch.count_nonzero(self.reference['output.weight']), 0)
        for model in models:
            before = torch.get_rng_state().clone()
            report = initialize_from_reference(model, self.reference, 20359023)
            self.assertTrue(torch.equal(before, torch.get_rng_state()))
            self.assertTrue(report['source_final_parameters_exact'])
            self.assertEqual(model.input_projection[0].weight.shape, (64, 76))
            self.assertEqual(torch.count_nonzero(model.branch_projection.weight), 0)
            self.assertTrue(all(parameter.requires_grad for parameter in model.parameters()))
            for name, value in self.reference.items():
                self.assertTrue(torch.equal(value, model.state_dict()[name]), name)
            for evaluation in (False, True):
                model.train(not evaluation)
                self.teacher.train(not evaluation)
                with torch.no_grad():
                    expected = self.teacher(*arguments[:3])
                    observed = model(*arguments)
                self.assertTrue(torch.equal(observed['states'], expected['states']))
                self.assertTrue(torch.equal(observed['residual_m'], expected['residual_m']))
        for name, value in models[0].state_dict().items():
            self.assertTrue(torch.equal(value, models[1].state_dict()[name]), name)
        self.assertEqual(sum(p.numel() for p in models[0].parameters()),
                         sum(p.numel() for p in models[1].parameters()))

    def test_fixed_relation_bias_uses_only_declared_columns_and_is_bounded(self):
        helper = torch.zeros((3, 2, 27))
        helper[1, :, 16] = 2.
        helper[2, :, 20] = 1000.
        bias = observable_relation_bias(helper)
        self.assertEqual(bias.shape, (2, 3, 3))
        self.assertTrue(torch.equal(bias.diagonal(dim1=1, dim2=2), torch.zeros((2, 3))))
        self.assertTrue(torch.equal(bias[:, 0, 1], torch.full((2,), -1.)))
        self.assertTrue(torch.equal(bias[:, 0, 2], torch.full((2,), -4.)))
        self.assertTrue(torch.equal(bias, bias.transpose(1, 2)))
        self.assertTrue(((bias >= -4.) & (bias <= 0.)).all())
        helper[:, :, 0] = 900.
        self.assertTrue(torch.equal(bias, observable_relation_bias(helper)))

    def test_single_helper_per_query_and_empty_group_have_no_relation_effect(self):
        args = list(self.random_inputs(3, 5))
        args[4].zero_()
        args[4][0, 0] = args[4][1, 1] = args[4][2, 2] = True
        generic, relation = [self.model(mode, active=True) for mode in MODES]
        for inputs in (args, self.random_inputs(0, 5)):
            first, second = generic(*inputs), relation(*inputs)
            for key in ('states', 'residual_m', 'helper_embedding', 'helper_weights', 'helper_counts'):
                self.assertTrue(torch.equal(first[key], second[key]), key)
            empty = first['helper_counts'] == 0
            self.assertTrue(torch.equal(first['helper_embedding'][empty], torch.zeros((int(empty.sum()), 16))))
            self.assertTrue(torch.equal(first['helper_weights'].sum(0), (first['helper_counts'] > 0).double()))

    def test_relation_changes_multi_helper_branch_and_same_weight_off_reconstructs_generic(self):
        inputs = self.random_inputs()
        generic, relation = [self.model(mode, active=True) for mode in MODES]
        before = copy.deepcopy(relation.state_dict())
        generic_result = generic(*inputs)
        result = relation(*inputs)
        self.assertFalse(torch.allclose(result['helper_embedding'], generic_result['helper_embedding']))
        self.assertFalse(torch.equal(result['states'], generic_result['states']))
        disabled = relation(*inputs, relation_enabled=False)
        self.assertTrue(torch.equal(disabled['states'], generic_result['states']))
        self.assertTrue(torch.equal(disabled['helper_weights'], generic_result['helper_weights']))
        relation.set_relation_enabled(False)
        self.assertTrue(torch.equal(relation(*inputs)['states'], generic_result['states']))
        relation.set_relation_enabled(True)
        self.assertTrue(torch.equal(relation(*inputs)['states'], result['states']))
        for name, value in before.items():
            self.assertTrue(torch.equal(value, relation.state_dict()[name]))
        with self.assertRaises(ValueError):
            generic.set_relation_enabled(True)
        with self.assertRaises(ValueError):
            generic(*inputs, relation_enabled=True)
        with self.assertRaises(ValueError):
            relation.set_relation_enabled(1)

    def test_helper_permutation_invariance_and_equivariant_pool_weights(self):
        args = list(self.random_inputs())
        args[4][0, 1] = args[4][2, 3] = False
        permutation = torch.tensor([3, 1, 0, 2])
        changed = [*args[:3], args[3][permutation], args[4][permutation]]
        for mode in MODES:
            model = self.model(mode, active=True)
            first, second = model(*args), model(*changed)
            torch.testing.assert_close(first['helper_embedding'], second['helper_embedding'], rtol=2e-6, atol=2e-6)
            torch.testing.assert_close(first['helper_weights'][permutation], second['helper_weights'], rtol=2e-6, atol=2e-7)
            torch.testing.assert_close(first['states'], second['states'], rtol=0., atol=1e-8)
            self.assertTrue(torch.equal(first['helper_counts'], second['helper_counts']))
            self.assertTrue(torch.equal(first['states'][:, 2:], args[2][:, 2:]))

    def test_unsupported_nan_cannot_change_attention_or_prediction(self):
        args = list(self.random_inputs())
        args[4][0, 1] = args[4][2, 3] = False
        args[4][:, 4] = False
        altered = [*args[:3], args[3].clone(), args[4]]
        altered[3][~args[4]] = torch.nan
        for mode in MODES:
            model = self.model(mode, active=True)
            first, second = model(*args), model(*altered)
            for key in ('states', 'helper_embedding', 'helper_weights'):
                self.assertTrue(torch.equal(first[key], second[key]), key)
            self.assertEqual(torch.count_nonzero(second['helper_weights'][~args[4]]), 0)
            torch.testing.assert_close(second['helper_weights'].sum(0), args[4].any(0).double(), rtol=0., atol=1e-7)
            altered[3][0, 0, 1] = torch.inf
            with self.assertRaises(ValueError):
                model(*altered)
            altered[3][0, 0, 1] = args[3][0, 0, 1]

    def test_all_masked_and_zero_helper_groups_have_finite_zero_branch_gradients(self):
        for helpers in (0, 3):
            inputs = list(self.random_inputs(helpers))
            inputs[4].zero_()
            inputs[3].fill_(torch.nan)
            for mode in MODES:
                model = self.model(mode, active=True)
                result = model(*inputs)
                self.assertTrue(torch.isfinite(result['states']).all())
                self.assertEqual(torch.count_nonzero(result['helper_embedding']), 0)
                result['residual_m'].square().mean().backward()
                for name, parameter in model.named_parameters():
                    self.assertIsNotNone(parameter.grad, name)
                    self.assertTrue(torch.isfinite(parameter.grad).all(), name)
                    if name.startswith(BRANCH_PREFIXES):
                        self.assertEqual(torch.count_nonzero(parameter.grad), 0, name)

    def test_position_objective_updates_inherited_and_branch_parameters_without_nonfinite_gradients(self):
        args = self.random_inputs()
        target = args[0].clone()
        target[:, 0] += .02
        target[:, 1] -= .01
        target[:, 2:] = torch.nan
        valid = torch.ones_like(target, dtype=torch.bool)
        for mode in MODES:
            model = self.model(mode)
            optimizer = torch.optim.Adam(model.parameters(), lr=.001)
            old = copy.deepcopy(model.state_dict())
            for step in range(2):
                optimizer.zero_grad(set_to_none=True)
                result = model(*args)
                losses, _ = sequence_objectives(result, target, valid)
                self.assertTrue(torch.isfinite(losses['linear']))
                losses['linear'].backward()
                for name, parameter in model.named_parameters():
                    self.assertIsNotNone(parameter.grad, name)
                    self.assertTrue(torch.isfinite(parameter.grad).all(), name)
                self.assertGreater(float(model.branch_projection.weight.grad.abs().sum()), 0.)
                if step == 1:
                    self.assertGreater(float(model.phi[0].weight.grad.abs().sum()), 0.)
                    self.assertGreater(float(model.set_value.weight.grad.abs().sum()), 0.)
                    self.assertGreater(float(model.pool_query.weight.grad.abs().sum()), 0.)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                optimizer.step()
            self.assertFalse(torch.equal(old['input_projection.0.weight'], model.input_projection[0].weight))
            self.assertFalse(torch.equal(old['branch_projection.weight'], model.branch_projection.weight))
            self.assertEqual(len(optimizer.state), len(list(model.parameters())))
            self.assertTrue(all(float(entry['step']) == 2. for entry in optimizer.state.values()))
            self.assertTrue(torch.isfinite(model(*args)['states']).all())

    def test_bad_source_schema_dtype_and_undeclared_modes_are_rejected(self):
        for mutate in (
            lambda value: value.pop('output.bias'),
            lambda value: value.update(extra=torch.zeros(1)),
            lambda value: value.update({'input_projection.0.weight': torch.zeros(64, 92)}),
            lambda value: value.update({'output.weight': value['output.weight'].double()}),
            lambda value: value.update({'output.bias': torch.tensor([torch.nan, 0.])}),
        ):
            invalid = copy.deepcopy(self.reference)
            mutate(invalid)
            with self.assertRaises(ValueError):
                initialize_from_reference(self.model('generic'), invalid, 1)
        for kwargs in (dict(mode='weighted'), dict(hidden_size=128), dict(num_layers=1), dict(initial_gate=.9)):
            with self.assertRaises(ValueError):
                QueryRelationPosition(**kwargs)
        args = list(self.random_inputs())
        for index, replacement in ((1, args[1].double()), (3, args[3][..., :26]),
                                   (4, args[4].float()), (3, args[3][:, :-1])):
            invalid = args.copy()
            invalid[index] = replacement
            with self.assertRaises(ValueError):
                self.model('relation')(*invalid)


if __name__ == '__main__':
    unittest.main()
