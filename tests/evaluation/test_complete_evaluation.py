from tests.fixtures.models import CompleteEvaluationFixture
"""Real CPU three-module inference and full-population evaluation contracts."""
import copy
import math
import unittest
from unittest.mock import patch

import numpy as np
import torch

from association.evaluation.tmtaff import aggregate_fusion_reports
from association.evaluation.scene import evaluate_full_scene, simple_fusion_outputs
from association.evaluation.grouping import group_predicted_tracks
from association.evaluation.association import evaluate_scenes, select_threshold


from tests.fixtures.scenes import scene, empty_scene


class TCMCompletionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def setUp(self):
        from association.data.timed import fit_timed_normalizer
        from association.models.timed import TimedAssociationModel
        from association.models.tmtaff import ModelConfig
        from tests.fixtures.scenes import timed_scene
        self.scene = timed_scene('completion', 'V_select')
        self.normalizer = fit_timed_normalizer([timed_scene('train', 'T_core')])
        self.model = TimedAssociationModel(ModelConfig(length=4, d_model=8, nhead=2,
            num_layers=1, dim_feedforward=16, dropout=0.), 'contrastive', max_observations=3)

    def evaluate(self, scenes=None):
        from association.evaluation.tmtaff import evaluate_tcm_completion
        return evaluate_tcm_completion(self.model, self.normalizer,
            [self.scene] if scenes is None else scenes)

    def test_one_step_matches_training_denominator_and_reference_cannot_change_forward(self):
        from association.training.common import normalize_scene
        changed = copy.deepcopy(self.scene)
        for target in changed['fusion_targets'].values():
            target['states'][:, 0] += .5
        outputs = []
        complete = self.model.complete
        def capture(values, observed):
            self.assertEqual(len(values), 1)
            result = complete(values, observed)
            outputs.append(result.clone())
            return result
        with patch.object(self.model, 'complete', side_effect=capture), \
             patch.object(self.model.tam_encoder, 'forward', side_effect=AssertionError('no association')):
            first, second = self.evaluate(), self.evaluate([changed])
        for original, altered in zip(outputs[:4], outputs[4:]):
            torch.testing.assert_close(original, altered, atol=0, rtol=0)
        self.assertEqual(first['one_step'], second['one_step'])
        self.assertNotEqual(first['query']['model']['regions']['all']['feature_mse'],
                            second['query']['model']['regions']['all']['feature_mse'])
        values, observed = normalize_scene(self.scene, self.normalizer)
        with torch.inference_mode():
            self.model.requires_grad_(False)
            expected = self.model.completion_loss(torch.from_numpy(values), torch.from_numpy(observed)).item()
        total = first['one_step']['model']['all']
        self.assertEqual(total['feature_elements'], 2 * (self.scene['native_observed'].sum()-4) * 8)
        self.assertAlmostEqual(total['feature_mse'], expected, places=6)
        self.assertEqual(first['fragment_count'], 4)  # Two repeated identities retain both fragments.

    def test_endpoint_control_uses_directional_state_with_query_time(self):
        from association.models.timed import continuous_time_encoding
        from association.training.common import normalize_scene
        result = self.evaluate()
        values, _ = normalize_scene(self.scene, self.normalizer)
        row = result['per_fragment'][0]  # Two native points, no query inside support.
        target = self.scene['fusion_targets'][self.scene['identities'][0]]['states']
        target = torch.tensor((target-np.array(self.normalizer['mean']))/np.array(self.normalizer['std']),
                              dtype=torch.float32)
        query = torch.from_numpy(values[0, 3:, 4])
        with torch.inference_mode():
            expected_target = torch.relu(self.model.input_embedding(target + continuous_time_encoding(query, 4)))
            for region, indices, endpoint in (('before_support', [0, 1], 0), ('after_support', [2, 3], 1)):
                state = torch.from_numpy(values[0, endpoint, :4])
                held = torch.relu(self.model.input_embedding(state + continuous_time_encoding(query[indices], 4)))
                expected_sum = (held.double()-expected_target[indices].double()).square().sum().item()
                actual = row['query']['state_hold']['regions'][region]
                self.assertEqual(actual['query_count'], 2)
                self.assertTrue(math.isclose(actual['squared_error_sum'], expected_sum, rel_tol=1e-6, abs_tol=1e-7))
        self.assertIsNone(row['query']['model']['regions']['inside_support']['feature_mse'])
        for name in ('model', 'state_hold'):
            query_report = result['query'][name]
            self.assertEqual(sum(query_report['regions'][region]['query_count'] for region in
                ('inside_support', 'before_support', 'after_support')), 16)
            self.assertEqual(sum(row['query_count'] for row in query_report['input_distance_metrics'].values()), 16)
        self.assertAlmostEqual(result['query']['model']['regions']['inside_support']['feature_mse'],
                               result['query']['state_hold']['regions']['inside_support']['feature_mse'], places=6)

    def test_partial_reference_and_empty_scene_keep_population_without_filling_targets(self):
        from tests.fixtures.scenes import empty_timed_scene
        del self.scene['fusion_targets']['completion-B']
        self.scene['fusion_targets']['completion-A']['valid'][0, 1] = False
        result = self.evaluate([self.scene, empty_timed_scene()])
        self.assertEqual(result['fragment_count'], 4)
        self.assertEqual(result['scene_count'], 2)
        self.assertEqual(result['no_complete_reference_fragment_count'], 2)
        self.assertEqual(result['per_scene'][1]['reference_targets_without_input'], ['empty-unobserved'])
        for name in ('model', 'state_hold'):
            row = result['query'][name]['regions']['all']
            self.assertEqual(row['query_count'], 16)
            self.assertEqual(row['evaluable_query_count'], 6)
            self.assertEqual(row['missing_reference_query_count'], 10)
            self.assertEqual(row['feature_elements'], 6*8)
        empty = self.evaluate([empty_timed_scene()])
        self.assertIsNone(empty['one_step']['model']['all']['feature_mse'])
        self.assertIsNone(empty['query']['model']['regions']['all']['feature_mse'])


class CompleteEvaluationTests(CompleteEvaluationFixture, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)


    def test_actual_tcm_tam_tfm_chain_reuses_completion_and_retains_wrong_groups(self):
        example = scene(include_missing_target=True)
        seen = {'encoder': 0, 'decoder': 0, 'tam': 0, 'tfm': 0}
        def hook(name):
            def called(*args):
                seen[name] += 1
            return called
        handles = [module.register_forward_hook(hook(name)) for name, module in
                   [('encoder', self.model.tcm_encoder), ('decoder', self.model.tcm_decoder),
                    ('tam', self.model.tam_encoder), ('tfm', self.fusion)]]
        checks = []
        try:
            with patch.object(self.model, 'complete', wraps=self.model.complete) as complete:
                result = self.evaluate([example], check=lambda: checks.append(True))
                self.assertEqual(complete.call_count, 2)  # Two node batches, reused for both groupings.
        finally:
            for handle in handles:
                handle.remove()
        self.assertGreater(seen['encoder'], 0)
        self.assertGreater(seen['decoder'], 0)
        self.assertEqual(seen['tam'], 2)
        self.assertEqual(seen['tfm'], 5)  # Predicted four-node group plus two oracle pairs.
        self.assertEqual(result['fusion_forward_count'], 5)
        self.assertGreater(len(checks), 5)
        predicted = result['per_scene'][0]['methods']['predicted']
        self.assertEqual(predicted['all_predicted_groups'][0]['node_indices'], [0, 1, 2, 3])
        self.assertEqual(predicted['group_diagnostics']['merge_pairs'], 4)
        self.assertEqual(predicted['unmatched_target_ids'], ['B', 'C'])
        self.assertEqual(result['fusion']['predicted']['all']['position']['available_target_count'], 12)
        self.assertEqual(result['fusion']['predicted']['all']['position']['coverage'], 1/3)
        self.assertFalse(self.model.training); self.assertFalse(self.fusion.training)
        self.assertTrue(all(p.grad is None for p in list(self.model.parameters()) + list(self.fusion.parameters())))

    def test_reference_values_cannot_change_predicted_or_oracle_model_outputs(self):
        original, changed = scene(), scene()
        for target in changed['fusion_targets'].values():
            target['states'][:, 0] += 1
            target['states'][:, 2] += 5
        captured = []
        handle = self.fusion.register_forward_hook(
            lambda module, args, result: captured.append(result['states'].detach().clone()))
        try:
            first = self.evaluate([original]); first_predictions = captured[:]
            captured.clear()
            second = self.evaluate([changed]); second_predictions = captured[:]
        finally:
            handle.remove()
        np.testing.assert_array_equal(first['records'][0]['scores'], second['records'][0]['scores'])
        self.assertEqual(first['threshold'], second['threshold'])
        self.assertEqual(len(first_predictions), len(second_predictions))
        for left, right in zip(first_predictions, second_predictions):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
        self.assertNotEqual(first['fusion']['predicted']['all']['position']['mae'],
                            second['fusion']['predicted']['all']['position']['mae'])

    def test_one_validation_threshold_is_reused_for_every_scene_and_association_metrics(self):
        examples = [scene('one'), scene('two')]
        examples[1]['values'][examples[1]['observed'], 0] += .03
        result = self.evaluate(examples, threshold=None, select=True)
        expected = select_threshold(result['records'])
        self.assertEqual(result['threshold'], expected['threshold'])
        self.assertEqual(result['association'], evaluate_scenes(result['records'], result['threshold']))
        for record, report in zip(result['records'], result['per_scene']):
            groups = group_predicted_tracks(4, record['pairs'], record['scores'], threshold=result['threshold'],
                                            grouping_policy='anchor_direct', node_order='input')
            actual = [group['node_indices'] for group in report['methods']['predicted']['all_predicted_groups']]
            self.assertEqual(actual, groups)

    def test_singleton_policy_is_explicit_and_observed_never_decodes_missing_latents(self):
        example = scene()
        missing = self.evaluate([example], threshold=None, singleton='missing')
        passthrough = self.evaluate([example], threshold=None, singleton='observed')
        self.assertEqual(missing['fusion']['predicted']['all']['position']['coverage'], 0)
        row = passthrough['per_scene'][0]['methods']['predicted']
        # First A and first B fragments occupy target rows, later fragments remain duplicates.
        self.assertEqual(row['selected_group_by_target'], [0, 2])
        self.assertEqual(row['duplicate_prediction_group_count'], 2)
        self.assertEqual(passthrough['fusion']['predicted']['all']['position']['intersection_count'], 6)
        self.assertEqual(passthrough['fusion']['predicted']['missing']['position']['intersection_count'], 0)
        self.assertEqual(missing['fusion_forward_count'], 2)
        self.assertEqual(passthrough['fusion_forward_count'], 2)  # Only the two oracle pairs were fused.
        with self.assertRaises(ValueError):
            self.evaluate([example], singleton='complete')

    def test_all_sources_remain_in_every_scene_including_absent_source(self):
        examples = [scene('one'), scene('two')]
        examples[1]['sources'] = ['9001'] * 4
        result = self.evaluate(examples)
        absent = result['per_scene'][1]['methods']['single_source:9002']
        self.assertEqual(absent['full_target_metrics']['all']['position']['coverage'], 0)
        self.assertEqual(absent['target_identity_count'], 2)
        self.assertIn('single_source:9001', result['fusion'])
        self.assertIn('single_source:9002', result['fusion'])

    def test_simple_oracle_uses_observations_even_when_9001_copies_reference(self):
        original = scene(include_missing_target=True)
        # Source 9001 exactly copies reference, but is still just one source.
        for node, identity in enumerate(original['identities']):
            observed = original['observed'][node]
            original['values'][node, observed] = original['fusion_targets'][identity]['states'][observed]
            if original['sources'][node] == '9002':
                original['values'][node, observed, 2] += 6
        changed = copy.deepcopy(original)
        for target in changed['fusion_targets'].values():
            target['states'][:, 2] += 100
        captured = []
        def capture(*args):
            outputs = simple_fusion_outputs(*args)
            captured.append(copy.deepcopy(outputs))
            return outputs
        with patch('association.evaluation.tmtaff.simple_fusion_outputs', side_effect=capture):
            first, second = self.evaluate([original]), self.evaluate([changed])
        self.assertEqual(len(captured), 2)
        for left, right in zip(*captured):
            np.testing.assert_equal(left['states'], right['states'])
            np.testing.assert_equal(left['prediction_valid'], right['prediction_valid'])
        self.assertEqual(captured[0][0]['states'][1, 2], 4)  # 1 and 7, equal source mean.
        metric = first['fusion']['simple_fusion_oracle']['all']['position']
        self.assertEqual(metric['available_target_count'], 12)
        self.assertEqual(metric['intersection_count'], 7)
        self.assertEqual(metric['coverage'], 7/12)
        missing = first['fusion']['simple_fusion_oracle']['missing']['position']
        self.assertEqual(missing['intersection_count'], 0)
        self.assertEqual(missing['missing_prediction_count'], 5)
        self.assertNotEqual(first['fusion']['simple_fusion_oracle']['all']['speed']['mae'],
                            second['fusion']['simple_fusion_oracle']['all']['speed']['mae'])

    def test_wrong_clock_test_split_and_empty_observed_fragment_are_rejected_before_inference(self):
        changes = [lambda x: x.update(split='T_test'),
                   lambda x: x['observation_times'].__setitem__((0, 0), .01),
                   lambda x: x['observed'].__setitem__(0, False)]
        for change in changes:
            example = scene(); change(example)
            with patch.object(self.model, 'complete', wraps=self.model.complete) as complete:
                with self.assertRaises(ValueError):
                    self.evaluate([example])
                complete.assert_not_called()

    def test_reference_grid_must_match_exactly_before_any_scene_inference(self):
        changes = [lambda item: item.pop('query_times'),
                   lambda item: item['query_times'].__setitem__(1, 1.000000001),
                   lambda item: item.update(query_times=np.arange(5.)),
                   lambda item: item['query_times'].__setitem__(0, np.nan)]
        for change in changes:
            invalid = scene('invalid'); change(invalid['fusion_targets']['A'])
            with patch.object(self.model, 'complete', wraps=self.model.complete) as complete:
                with self.assertRaisesRegex(ValueError, 'query_times'):
                    self.evaluate([scene(), invalid])
                complete.assert_not_called()

    def test_full_candidate_population_and_labels_are_required_before_inference(self):
        changes = [lambda item: item.update(pairs=item['pairs'][:-1], labels=item['labels'][:-1]),
                   lambda item: item['pairs'].__setitem__(1, item['pairs'][0]),
                   lambda item: item.update(pairs=item['pairs'][::-1], labels=item['labels'][::-1]),
                   lambda item: item['labels'].__setitem__(0, 0),
                   lambda item: item.update(identities=item['identities'][:-1])]
        for change in changes:
            invalid = scene('invalid'); change(invalid)
            with patch.object(self.model, 'complete', wraps=self.model.complete) as complete:
                with self.assertRaises(ValueError):
                    self.evaluate([scene(), invalid])
                complete.assert_not_called()


    def test_empty_scenes_and_reference_only_targets_are_not_dropped(self):
        result = self.evaluate([scene(), empty_scene(), empty_scene('reference-only', with_target=True)])
        self.assertEqual(result['association']['coverage']['all_scenes'], 3)
        self.assertEqual(result['association']['coverage']['scenes_without_pairs'], 2)
        self.assertEqual(len(result['per_scene']), 3)
        self.assertEqual(result['per_scene'][1]['methods']['predicted']['target_identity_count'], 0)
        missed = result['per_scene'][2]['methods']['predicted']['full_target_metrics']['all']['position']
        self.assertEqual(missed['available_target_count'], 4)
        self.assertEqual(missed['missing_prediction_count'], 4)


class AggregateFusionReportTests(unittest.TestCase):
    def test_pool_error_numerators_before_sqrt_and_count_missing_target(self):
        def report(speed_errors, predicted=True):
            errors = np.asarray(speed_errors, float)
            truth = np.zeros((1, len(errors), 4), float)
            prediction = truth[0].copy(); prediction[:, 2] = errors
            return evaluate_full_scene([{'node_indices': [0], 'states': prediction if predicted else None}],
                node_identities=['A'], target_identities=['A'], target_states=truth,
                target_valid=np.ones_like(truth, bool), node_observed=np.ones(truth.shape[:2], bool),
                split='V_select', grouping_kind='predicted', correspondence_policy='anchor_identity_first')
        result = aggregate_fusion_reports([report([1]), report([3, 3, 3]), report([0, 0], predicted=False)])
        speed = result['all']['speed']
        self.assertEqual(speed['available_target_count'], 6)
        self.assertEqual(speed['intersection_count'], 4)
        self.assertEqual(speed['coverage'], 2/3)
        self.assertEqual(speed['mae'], 2.5)
        self.assertEqual(speed['mse'], 7)
        self.assertEqual(speed['rmse'], math.sqrt(7))
        self.assertNotEqual(speed['rmse'], 2)  # Mean of scene RMSEs would be wrong.
        self.assertEqual(speed['trajectory_coverage'], 2/3)
        self.assertIsNone(result['missing']['speed']['rmse'])
        empty = aggregate_fusion_reports([])
        self.assertIsNone(empty['all']['speed']['coverage'])


if __name__ == '__main__':
    unittest.main()
