"""Real CPU three-module inference and full-population evaluation contracts."""
import copy
import math
import unittest
from unittest.mock import patch

import numpy as np
import torch

from association.complete_evaluation import _inference_device, aggregate_fusion_reports, evaluate_complete
from association.full_scene_evaluation import evaluate_full_scene, simple_fusion_outputs
from association.fusion import PairFusion, PairFusionConfig
from association.fusion_inference import group_predicted_tracks
from association.metrics import evaluate_scenes, select_threshold
from association.models import AssociationModel, ModelConfig
from association.reference import fit_normalizer


def scene(scene_id='one', include_missing_target=False):
    grid = np.arange(4, dtype=float)
    observed = np.array([[1, 1, 0, 0], [0, 1, 1, 0], [1, 1, 1, 1], [1, 1, 0, 0]], bool)
    values = np.zeros((4, 4, 4), dtype=np.float32)
    for node in range(4):
        values[node, :, 0] = .001 * np.arange(4) + .01 * (node // 2)
        values[node, :, 1] = .002 * np.arange(4)
        values[node, :, 2] = 1 + node // 2
        values[node, :, 3] = 15 + node
    values[~observed] = np.nan
    identities = ['A', 'A', 'B', 'B']
    pairs = np.array([[i, j] for i in range(4) for j in range(i+1, 4)], dtype=np.int64)
    targets = {}
    for index, identity in enumerate(['A', 'B'] + (['C'] if include_missing_target else [])):
        states = np.zeros((4, 4), dtype=float)
        states[:, 0] = .001 * grid + .01 * index
        states[:, 1] = .002 * grid
        states[:, 2] = index + 1
        states[:, 3] = 15
        targets[identity] = {'states': states, 'valid': np.ones((4, 4), bool), 'query_times': grid.copy()}
    return dict(scene_id=scene_id, component_id='component-' + scene_id, split='V_select',
                grid=grid, observation_times=np.where(observed, grid[None], np.nan),
                values=values, observed=observed, node_ids=[scene_id + ':' + str(i) for i in range(4)],
                identities=identities, sources=['9001', '9002', '9001', '9002'],
                intervals=np.array([[0, 1], [1, 2], [0, 3], [0, 1]], float), pairs=pairs,
                labels=np.array([identities[i] == identities[j] for i, j in pairs], dtype=np.int64),
                fusion_targets=targets)


def empty_scene(scene_id='empty', with_target=False):
    result = scene(scene_id)
    for key in ('values', 'observed', 'observation_times', 'intervals'):
        result[key] = result[key][:0]
    result.update(node_ids=[], identities=[], sources=[], pairs=np.empty((0, 2), dtype=np.int64),
                  labels=np.empty(0, dtype=np.int64), fusion_targets={})
    if with_target:
        result['fusion_targets']['Z'] = {'states': np.zeros((4, 4)), 'valid': np.ones((4, 4), bool),
                                         'query_times': result['grid'].copy()}
    return result


class CompleteEvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def setUp(self):
        torch.manual_seed(42)
        self.model = AssociationModel(ModelConfig(length=4, d_model=8, nhead=2,
                                     num_layers=2, dim_feedforward=16, dropout=.1), 'ce')
        self.fusion = PairFusion(PairFusionConfig(fc_hidden=8, length=4, d_model=8,
                                 nhead=2, num_layers=2, dim_feedforward=16, dropout=.1))
        self.normalizer = fit_normalizer([scene()])

    def evaluate(self, scenes, threshold=-1e6, singleton='missing', select=False, **kwargs):
        return evaluate_complete(self.model, self.fusion, self.normalizer, scenes, device='cpu',
                                 batch_tracks=2, pair_batch_size=2, select_on_validation=select,
                                 threshold=threshold, singleton_policy=singleton, **kwargs)

    def test_progress_separates_scoring_and_fusion_without_changing_results(self):
        examples = [scene(), empty_scene(with_target=True)]
        baseline = self.evaluate(examples)
        events = []
        actual = self.evaluate(examples, progress=events.append)
        self.assertEqual(actual['association'], baseline['association'])
        self.assertEqual(actual['fusion'], baseline['fusion'])
        for left, right in zip(actual['records'], baseline['records']):
            np.testing.assert_array_equal(left['scores'], right['scores'])
        self.assertEqual([e['stage'] for e in events],
                         ['association_scoring']*4 + ['physical_fusion']*4 + ['completed'])
        self.assertEqual(events[-1]['completed_scenes'], 2)
        self.assertEqual(events[-1]['fusion_forward_count'], actual['fusion_forward_count'])
        self.assertTrue(all(e['elapsed_seconds'] >= 0 for e in events))

    def test_progress_retains_stage_when_fusion_is_interrupted(self):
        events = []
        def progress(item):
            events.append(item)
            if item['stage'] == 'physical_fusion':
                raise TimeoutError('external evaluation deadline')
        with patch.object(self.fusion, 'forward', wraps=self.fusion.forward) as forward:
            with self.assertRaisesRegex(TimeoutError, 'external evaluation deadline'):
                self.evaluate([scene()], progress=progress)
            forward.assert_not_called()
        self.assertEqual(events[-1]['stage'], 'physical_fusion')
        self.assertEqual(events[-1]['fusion_forward_count'], 0)
        self.assertNotIn('completed', [event['stage'] for event in events])

    def test_failed_metric_aggregation_does_not_report_completed(self):
        events = []
        with patch('association.complete_evaluation.aggregate_fusion_reports',
                   side_effect=TimeoutError('aggregation deadline')):
            with self.assertRaisesRegex(TimeoutError, 'aggregation deadline'):
                self.evaluate([scene()], progress=events.append)
        self.assertEqual(events[-1]['stage'], 'physical_fusion')
        self.assertNotIn('completed', [event['stage'] for event in events])

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
        with patch('association.complete_evaluation.simple_fusion_outputs', side_effect=capture):
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

    def test_generic_cuda_device_resolution_is_tested_without_gpu_calls(self):
        self.assertEqual(_inference_device('cuda', 'cuda:0', 'cuda:0'), torch.device('cuda:0'))
        self.assertEqual(_inference_device('cuda', 'cuda:3', 'cuda:3'), torch.device('cuda:3'))
        self.assertEqual(_inference_device('cpu', 'cpu', 'cpu'), torch.device('cpu'))
        for requested, model_device, fusion_device in [('cuda:0', 'cuda:1', 'cuda:1'),
                ('cuda', 'cuda:1', 'cuda:2'), ('cuda', 'cpu', 'cpu')]:
            with self.assertRaises(ValueError):
                _inference_device(requested, model_device, fusion_device)

    def test_empty_scenes_and_reference_only_targets_are_not_dropped(self):
        result = self.evaluate([scene(), empty_scene(), empty_scene('reference-only', with_target=True)])
        self.assertEqual(result['association']['coverage']['all_scenes'], 3)
        self.assertEqual(result['association']['coverage']['scenes_without_pairs'], 2)
        self.assertEqual(len(result['per_scene']), 3)
        self.assertEqual(result['per_scene'][1]['methods']['predicted']['target_identity_count'], 0)
        missed = result['per_scene'][2]['methods']['predicted']['full_target_metrics']['all']['position']
        self.assertEqual(missed['available_target_count'], 4)
        self.assertEqual(missed['missing_prediction_count'], 4)

    def test_input_scenes_and_parameters_are_not_modified(self):
        example = scene(); before = copy.deepcopy(example)
        model_before = {key: value.clone() for key, value in self.model.state_dict().items()}
        self.evaluate([example])
        for key in ('values', 'observed', 'grid', 'observation_times', 'pairs', 'labels'):
            np.testing.assert_equal(example[key], before[key])
        for identity in example['fusion_targets']:
            np.testing.assert_equal(example['fusion_targets'][identity]['states'], before['fusion_targets'][identity]['states'])
        for key, value in self.model.state_dict().items():
            torch.testing.assert_close(value, model_before[key], rtol=0, atol=0)


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
