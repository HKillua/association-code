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
