"""068 CPU system contracts: fixed cache, all groups, common masks and tails."""
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import torch

from association.evaluation.support_gate import METHODS, evaluate_support_gate, compare_endpoints, _sensitivity
from association.evaluation.system_attribution import pool_metrics
from association.models.support_gate import SupportGate
from association.utils.runtime import read_json
from tools.analysis.compare_association_duration import _replay
from tests.evaluation import test_learned_group_oracle as cache_fixture


def support_gate_cache_fixture(root, *, sparse=False, scene_id='3035'):
    """Actual six-arm Q64 cache with swallowed/missing targets and a duplicate.

    Motion reconstruction alone is substituted by callers: the existing fixture
    has cached physical evidence but no source observations. All ordinary replay,
    model forward, group alignment and metrics use production implementations.
    """
    fragment, saved, expected = cache_fixture.LearnedGroupOracleTests.fixture(sparse)
    source = root/'source056'/'system'
    source.mkdir(parents=True)
    for directory, values in (('predictions', fragment), ('system-predictions', saved)):
        (source/directory).mkdir()
        np.savez_compressed(source/directory/f'states-{scene_id}.npz', **values)
    replay = _replay(fragment, saved, expected)
    artifact = replay[1]['learned_assoc_anchor']
    n = len(fragment['node_ids'])
    motion = dict(states=fragment['native_states'], support=fragment['support'],
        nearest_seconds=fragment['nearest_seconds'], query_seconds=fragment['query_seconds'],
        exact_observation=fragment['exact_observation'], native_count=np.full(n, 8),
        span_seconds=np.full(n, 100.), fit_rmse_m=np.ones(n),
        ols_velocity_mps=np.ones((n, 2)), endpoint_velocity_mps=np.ones((n, 2)))
    raw = dict(scene_id=scene_id, split='V_select', component_id='shared', motion=motion,
        identities=fragment['identities'], target_identity_ids=fragment['target_identity_ids'],
        target=fragment['target'], valid=fragment['target_valid'],
        fusion_targets={identity: dict(states=artifact['aligned_truth'][i], valid=artifact['aligned_truth_valid'][i])
                        for i, identity in enumerate(fragment['target_identity_ids'])})
    metadata = dict(scene_ids=[scene_id], component_ids={scene_id: 'shared'},
        expected={scene_id: expected}, fragment_count=n,
        pooled={method: pool_metrics([row['full_target_metrics']]) for method, row in expected.items()})
    return source, metadata, [raw]


class SupportGateEvaluationTests(unittest.TestCase):
    def evaluate(self, root, model=None, *, sparse=False, mutate=None):
        source, metadata, validation = support_gate_cache_fixture(root, sparse=sparse)
        model = SupportGate() if model is None else model
        if mutate:
            mutate(source, metadata, validation)
        with patch('association.evaluation.support_gate.frozen_motion', side_effect=lambda raw, *_: raw):
            result = evaluate_support_gate(model, validation, source, metadata, root/'result')
        return result, source, metadata, validation

    def test_all_output_groups_forwarded_before_fixed_unique_population(self):
        model = SupportGate()
        calls = []
        handle = model.register_forward_hook(lambda module, inputs, result: calls.append(result['states'].clone()))
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            summary, source, metadata, raw = self.evaluate(root, model)
            self.assertEqual(len(calls), 4)  # Includes duplicate singleton and no-reference target.
            self.assertEqual(summary['model_forwards'], 4)
            self.assertEqual(summary['method_names'], list(METHODS))
            population = summary['methods']['anchor']['population']
            self.assertEqual(population['target_identity_count'], 5)
            self.assertEqual(population['predicted_group_count'], 4)
            self.assertEqual(population['duplicate_prediction_group_count'], 1)
            for method in METHODS:
                item = summary['methods'][method]
                self.assertEqual(item['population'], population)
                self.assertEqual(item['full_target_metrics']['all']['position']['intersection_count'], 127)
                self.assertEqual(item['attribute_quantiles']['position']['all']['point_count'], 127)
            self.assertEqual(summary['routing']['all_output_groups']['output_query_count'], 256)
            self.assertEqual(summary['routing']['unique_selected_groups']['output_query_count'], 192)
            self.assertEqual(summary['routing']['unique_selected_groups']['eligible_query_count'], 1)
            self.assertEqual(summary['component_count'], 1)
            self.assertFalse(summary['test_opened'])
            with np.load(root/'result/predictions/states-3035.npz', allow_pickle=False) as payload:
                self.assertEqual(payload['group_gate'].shape, (4, 64))
                self.assertEqual(payload['group_eligible'].dtype, bool)
                np.testing.assert_array_equal(payload['group_anchor_node_indices'], [0, 2, 4, 5])
                self.assertIn('first representative', str(payload['inference_scope']))
            with np.load(root/'result/system-predictions/states-3035.npz', allow_pickle=False) as saved:
                for method in METHODS:
                    np.testing.assert_array_equal(saved[method+'_aligned_group_indices'],
                                                  saved['anchor_aligned_group_indices'])
                    np.testing.assert_array_equal(saved[method+'_aligned_prediction_valid'],
                                                  saved['anchor_aligned_prediction_valid'])
                # A's later perfect duplicate cannot replace its original group.
                self.assertGreater(summary['methods']['learned_gate']['full_target_metrics']['all']['position']['mae'], 0.)
            with self.assertRaises(FileExistsError):
                evaluate_support_gate(model, raw, source, metadata, root/'result')
        handle.remove()

    def test_reference_masks_and_empty_strata_do_not_filter_prediction_population(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            summary, _, _, _ = self.evaluate(root, sparse=True)
            for attr in ('position', 'speed', 'heading'):
                self.assertEqual(summary['methods']['learned_gate']['attribute_quantiles'][attr]['all']['point_count'], 1)
                self.assertIsNone(summary['methods']['learned_gate']['attribute_quantiles'][attr]['no_predicted_group']['p99'])
            changes = summary['pairwise']['learned_gate__minus__support_switch']['all']['position']
            self.assertEqual(changes['common_point_count'], 1)
            self.assertEqual(changes['tied_point_count'], 1)
            self.assertEqual(summary['routing']['all_output_groups']['output_query_count'], 256)

    def test_pooled_attribute_quantiles_and_endpoint_comparison_use_same_points(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            first_source, first_meta, first_raw = support_gate_cache_fixture(root/'first', scene_id='3035')
            second_source, second_meta, second_raw = support_gate_cache_fixture(root/'second', sparse=True, scene_id='3036')
            for directory in ('predictions', 'system-predictions'):
                (second_source/directory/'states-3036.npz').rename(first_source/directory/'states-3036.npz')
            metadata = dict(first_meta, scene_ids=['3035', '3036'], component_ids={'3035': 'shared', '3036': 'shared'},
                expected=dict(first_meta['expected'], **second_meta['expected']), fragment_count=12,
                pooled={method: pool_metrics([first_meta['expected']['3035'][method]['full_target_metrics'],
                                              second_meta['expected']['3036'][method]['full_target_metrics']])
                        for method in first_meta['pooled']})
            model = SupportGate()
            with patch('association.evaluation.support_gate.frozen_motion', side_effect=lambda raw, *_: raw):
                summary = evaluate_support_gate(model, first_raw+second_raw, first_source, metadata, root/'initial')
                # A fixed changed gate, with identical population, tests physical endpoint pairing.
                with torch.no_grad():
                    for name, parameter in model.named_parameters():
                        if name.endswith('bias') and parameter.numel() == 1:
                            parameter.fill_(0.)
                evaluate_support_gate(model, first_raw+second_raw, first_source, metadata, root/'final')
            values = []
            for sid in metadata['scene_ids']:
                with np.load(root/'initial/predictions'/f'states-{sid}.npz', allow_pickle=False) as arrays:
                    errors = arrays['learned_gate_position_error']
                    values.append(errors[np.isfinite(errors)])
            quantiles = summary['methods']['learned_gate']['attribute_quantiles']['position']['all']
            self.assertEqual(quantiles['point_count'], 128)
            self.assertEqual(quantiles['p99'], np.quantile(np.concatenate(values), .99, method='linear'))
            self.assertNotAlmostEqual(quantiles['p95'], np.mean([np.quantile(v, .95) for v in values]))
            paired = compare_endpoints(root/'initial', root/'final')
            self.assertTrue(paired['population_and_masks_exact'])
            self.assertEqual(paired['paired']['all']['position']['common_point_count'], 128)
            self.assertEqual(paired['paired']['all']['speed']['signed_error_change_sum'], 0.)
            self.assertEqual(paired['paired']['all']['heading']['signed_error_change_sum'], 0.)
            self.assertEqual(paired['components']['shared']['all']['position']['common_point_count'], 128)
            path = root/'final/predictions/states-3035.npz'
            with np.load(path, allow_pickle=False) as archive:
                corrupted = dict(archive)
            corrupted['group_eligible'][0, 0] = ~corrupted['group_eligible'][0, 0]
            np.savez_compressed(path, **corrupted)
            with self.assertRaisesRegex(ValueError, 'endpoint fixed input'):
                compare_endpoints(root/'initial', root/'final')

    def test_corrupted_ordinary_cache_rejected_before_any_forward(self):
        def corrupt(source, metadata, raw):
            path = source/'predictions/states-3035.npz'
            with np.load(path, allow_pickle=False) as cache:
                values = dict(cache)
            values['learned_assoc_anchor'][0, 0, 0] += .001
            np.savez_compressed(path, **values)
        model = SupportGate()
        with TemporaryDirectory() as tmp, patch.object(model, 'forward', wraps=model.forward) as forward:
            root = Path(tmp)
            with self.assertRaisesRegex(ValueError, 'cached ordinary prediction'):
                self.evaluate(root, model, mutate=corrupt)
            forward.assert_not_called()
            self.assertEqual(read_json(root/'result/run.json')['status'], 'FAILED')
            self.assertFalse((root/'result/summary.json').exists())

    def test_nonfinite_or_wrong_gate_mask_or_state_rejected_without_success(self):
        for case in ('nan', 'scalar', 'mask', 'mean', 'speed', 'outside_position', 'gate_range'):
            with self.subTest(case=case), TemporaryDirectory() as tmp:
                model = SupportGate()
                original = model.forward
                def bad(*args):
                    result = {key: value.clone() for key, value in original(*args).items()}
                    if case == 'nan':
                        result['logits'][0] = torch.nan
                    elif case == 'scalar':
                        result['gate'] = result['gate'][0]
                    elif case == 'mask':
                        result['eligible'][0] = ~result['eligible'][0]
                    elif case == 'mean':
                        result['mean_states'][0, 0] += .01
                    elif case == 'speed':
                        result['states'][0, 2] += 1.
                    elif case == 'outside_position':
                        result['states'][0, 0] += .01
                    else:
                        result['gate'][0] = 2.
                    return result
                with patch.object(model, 'forward', side_effect=bad), self.assertRaises(ValueError):
                    self.evaluate(Path(tmp), model)
                self.assertEqual(read_json(Path(tmp)/'result/run.json')['status'], 'FAILED')
                self.assertFalse((Path(tmp)/'result/summary.json').exists())

    def test_missing_population_and_component_or_protected_path_rejected(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, metadata, raw = support_gate_cache_fixture(root)
            model = SupportGate()
            with self.assertRaisesRegex(ValueError, 'immutable056'):
                evaluate_support_gate(model, raw, source, metadata, source.parent/'new')
            with self.assertRaisesRegex(ValueError, 'complete V'):
                evaluate_support_gate(model, [], source, metadata, root/'empty')
            changed = deepcopy(metadata)
            changed['component_ids'] = {}
            with self.assertRaisesRegex(ValueError, 'component registry'):
                evaluate_support_gate(model, raw, source, changed, root/'component')

    def test_fixed_leave_one_target_sensitivity_uses_reduced_denominator(self):
        # Deliberately distinct full and leave-one denominators; no method rows removed.
        arrays = dict(sensitivity_target=np.array([[True, False, False]]))
        arrays.update({method+'_position_error': np.array([[10., 2., 4.]]) for method in METHODS})
        arrays['support_switch_position_error'] = np.array([[1., 1., 1.]])
        result = _sensitivity([arrays])['comparisons']['learned_gate__minus__support_switch']
        self.assertEqual(result['full_point_count'], 3)
        self.assertEqual(result['omitted_point_count'], 1)
        self.assertEqual(result['remaining_point_count'], 2)
        self.assertEqual(result['leave_one_mean_error_difference_m'], 2.)
        self.assertEqual(result['leave_one_mse_difference_m2'], 9.)


if __name__ == '__main__':
    unittest.main()
