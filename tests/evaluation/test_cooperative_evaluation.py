from tests.fixtures.models import CooperativeEvaluationFixture
"""Native cooperative inference, fixed identity and full-population contracts."""
import copy
import unittest
from unittest.mock import patch

import numpy as np
import torch

from association.evaluation.association import evaluate_scenes


from tests.fixtures.scenes import empty_timed_scene

class CooperativeEvaluationTests(CooperativeEvaluationFixture, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)


    def test_short_native_fragment_singletons_and_missing_target_population(self):
        self.assertFalse(self.scene['observed'][0].any())
        self.assertEqual(self.scene['native_observed'][0].sum(), 2)
        target = copy.deepcopy(next(iter(self.scene['fusion_targets'].values())))
        self.scene['fusion_targets']['unobserved-target'] = target
        with patch.object(self.model, 'encode', wraps=self.model.encode) as encode, \
             patch.object(self.model, 'score_pairs', wraps=self.model.score_pairs) as score, \
             patch.object(self.model, 'decode_group', wraps=self.model.decode_group) as decode, \
             patch.object(self.model, 'utility', side_effect=AssertionError('plain must not run utility')), \
             patch.object(self.model.query_pool, 'forward', side_effect=AssertionError('base must not pool queries')):
            result = self.evaluate()
        self.assertEqual(encode.call_count, 2)
        self.assertEqual(score.call_count, 3)
        self.assertTrue(all(len(call.args[0]['memory']) <= 4 for call in score.call_args_list))
        self.assertEqual(decode.call_count, 6)  # Four predicted singletons and two oracle pairs.
        self.assertEqual(result['fusion_forward_count'], 6)
        self.assertEqual(result['per_scene'][0]['predicted_groups'], [[0], [1], [2], [3]])
        self.assertEqual(result['singleton_policy'], 'model_decode')
        np.testing.assert_equal(result['records'][0]['pairs'], self.scene['pairs'])
        self.assertEqual(result['fusion']['oracle']['all']['position']['available_target_count'], 12)
        self.assertEqual(result['fusion']['oracle']['all']['position']['intersection_count'], 8)
        self.assertEqual(result['fusion']['predicted']['all']['position']['intersection_count'], 8)
        self.assertEqual(result['fusion']['oracle']['missing']['position']['intersection_count'], 2)
        self.assertEqual(result['fusion']['simple_fusion_oracle']['missing']['position']['intersection_count'], 0)
        self.assertIn('single_source:9002', result['fusion'])
        self.assertFalse(result['test_opened'])

    def test_fixed_records_keep_scores_groups_and_avoid_new_identity_calls(self):
        fixed = [self.fixed()]
        with patch.object(self.model, 'score_pairs', side_effect=AssertionError('fixed identity changed')):
            plain = self.evaluate(fixed_records=fixed, threshold=0.)
            utility = self.evaluate(fixed_records=fixed, threshold=0., routing='utility')
        self.assertEqual(plain['association'], utility['association'])
        self.assertEqual(plain['association'], evaluate_scenes(fixed, 0.))
        self.assertEqual(plain['per_scene'][0]['predicted_groups'], [[0, 1], [2, 3]])
        self.assertEqual(plain['per_scene'][0]['predicted_groups'], utility['per_scene'][0]['predicted_groups'])
        np.testing.assert_equal(plain['records'][0]['scores'], fixed[0]['scores'])
        self.assertEqual(utility['identity_score_source'], 'fixed_records')
        self.assertIsNone(utility['selection'])


    def test_references_cannot_change_native_scores_or_any_decoded_states(self):
        changed = copy.deepcopy(self.scene)
        for target in changed['fusion_targets'].values():
            target['states'][:, 0] += .1
            target['states'][:, 2] += 10
        before, after = [], []
        decode = self.model.decode_group
        def capture(destination):
            def call(*args, **kwargs):
                result = decode(*args, **kwargs)
                destination.append(result.detach().clone())
                return result
            return call
        with patch.object(self.model, 'decode_group', side_effect=capture(before)):
            first = self.evaluate(routing='utility')
        with patch.object(self.model, 'decode_group', side_effect=capture(after)):
            second = self.evaluate([changed], routing='utility')
        np.testing.assert_array_equal(first['records'][0]['scores'], second['records'][0]['scores'])
        self.assertEqual(len(before), len(after))
        for left, right in zip(before, after):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
        self.assertNotEqual(first['fusion']['oracle']['all']['position']['mae'],
                            second['fusion']['oracle']['all']['position']['mae'])

    def test_empty_scene_and_no_input_target_remain_with_shared_source_rows(self):
        empty = empty_timed_scene()
        result = self.evaluate([self.scene, empty])
        self.assertEqual(result['association']['coverage']['all_scenes'], 2)
        self.assertEqual(result['association']['coverage']['scenes_without_pairs'], 1)
        self.assertEqual(result['fusion']['oracle']['all']['position']['available_target_count'], 12)
        self.assertEqual(result['fusion']['oracle']['all']['position']['intersection_count'], 8)
        self.assertEqual(result['per_scene'][1]['predicted_groups'], [])
        self.assertEqual(result['per_scene'][1]['methods']['single_source:9001']['target_identity_count'], 1)
        fixed = self.fixed(empty)
        fixed = {key: value.tolist() if isinstance(value, np.ndarray) else value for key, value in fixed.items()}
        only_empty = self.evaluate([empty], fixed_records=[fixed])
        self.assertEqual(only_empty['fusion_forward_count'], 0)
        self.assertEqual(only_empty['fusion']['predicted']['all']['position']['coverage'], 0.)
        no_targets = self.evaluate([empty_timed_scene('nothing', with_target=False)])
        self.assertIsNone(no_targets['fusion']['oracle']['all']['position']['coverage'])

    def test_utility_route_uses_anchor_direction_in_bounded_pair_batches(self):
        seen = []
        def utility(encoded, pairs, summaries=None):
            count = len(pairs)
            seen.append(count)
            result = torch.zeros((count, 2, 4, 4), device=encoded['memory'].device)
            result[:, 0] = .25
            result[:, 1] = -.75
            return result
        decode = self.model.decode_group
        passed = []
        def capture(encoded, members, **kwargs):
            passed.append(kwargs['utility'].detach().cpu().clone())
            return decode(encoded, members, **kwargs)
        fixed = self.fixed(); fixed['scores'][:] = 2.
        with patch.object(self.model, 'utility', side_effect=utility), \
             patch.object(self.model, 'decode_group', side_effect=capture):
            result = self.evaluate(fixed_records=[fixed], threshold=0., routing='utility', pair_batch_size=1)
        self.assertEqual(result['per_scene'][0]['predicted_groups'], [[0, 1, 2, 3]])
        self.assertTrue(seen and max(seen) == 1)
        self.assertEqual(tuple(passed[0].shape), (3, 4, 4))
        self.assertTrue(all(torch.all(value == .25) for value in passed))

    def test_identity_route_reads_frozen_scores_for_nonzero_anchor(self):
        fixed = self.fixed()
        received = []
        decode = self.model.decode_group
        def capture(encoded, members, **kwargs):
            received.append(kwargs['identity_logits'].detach().cpu().numpy())
            return decode(encoded, members, **kwargs)
        with patch.object(self.model, 'score_pairs', side_effect=AssertionError('fixed score changed')), \
             patch.object(self.model, 'utility', side_effect=AssertionError('identity needs no utility')), \
             patch.object(self.model, 'decode_group', side_effect=capture):
            self.evaluate(fixed_records=[fixed], threshold=0., routing='identity')
        self.assertEqual(len(received), 4)
        for scores in received:
            np.testing.assert_equal(scores, np.array([2.]))


class CooperativeCompletionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def setUp(self):
        from association.models.cooperative import CooperativeConfig, CooperativeModel
        from association.data.cooperative import fit_cooperative_normalizer
        from tests.fixtures.scenes import timed_scene
        self.scene = timed_scene('completion', 'V_select')
        config = CooperativeConfig(length=4, max_observations=3, d_model=8, nhead=2,
            num_layers=1, dim_feedforward=16, dropout=0., head_hidden=8,
            model_variant='support_residual_v2', support_time_policy='train_absmax_v1')
        self.normalizer = fit_cooperative_normalizer([timed_scene('train', 'T_core')], config)
        self.model = CooperativeModel(config, self.normalizer, support_time_scale=10.)

    def evaluate(self, scenes=None):
        from association.evaluation.cooperative import evaluate_cooperative_completion
        return evaluate_cooperative_completion(self.model, self.normalizer,
            [self.scene] if scenes is None else scenes)

    def test_every_fragment_kept_and_forward_ignores_helpers_and_reference(self):
        changed = copy.deepcopy(self.scene)
        for target in changed['fusion_targets'].values():
            target['states'][:, 0] += .2
            target['states'][:, 2] += 10
        outputs = []
        decode = self.model.decode_group
        def capture(encoded, members, **kwargs):
            self.assertEqual(len(encoded['memory']), 1)
            self.assertEqual(members, [0])
            self.assertEqual(kwargs, {'routing': 'plain'})
            result = decode(encoded, members, **kwargs)
            outputs.append(result.detach().clone())
            return result
        with patch.object(self.model, 'encode', wraps=self.model.encode) as encode, \
             patch.object(self.model, 'decode_group', side_effect=capture), \
             patch.object(self.model, 'score_pairs', side_effect=AssertionError('no identity scoring')), \
             patch.object(self.model, 'utility', side_effect=AssertionError('no helper utility')):
            first, second = self.evaluate(), self.evaluate([changed])
        self.assertEqual(encode.call_count, 8)
        self.assertTrue(all(len(call.args[0]) == 1 for call in encode.call_args_list))
        self.assertEqual(first['fragment_count'], 4)
        self.assertEqual(len(first['per_fragment']), 4)
        self.assertEqual([row['node_id'] for row in first['per_fragment']], self.scene['node_ids'])
        all_metrics = first['methods']['model']['regions']['all']
        self.assertEqual(all_metrics['position']['available_target_count'], 16)
        self.assertEqual(all_metrics['position']['available_target_trajectory_count'], 4)
        for before, after in zip(outputs[:4], outputs[4:]):
            torch.testing.assert_close(before, after, rtol=0, atol=0)
        self.assertNotEqual(all_metrics['position']['mae'],
            second['methods']['model']['regions']['all']['position']['mae'])

    def test_unsupervised_fragments_and_empty_scene_remain_without_invented_controls(self):
        del self.scene['fusion_targets']['completion-B']
        self.scene['fusion_targets']['completion-A']['valid'][0, 1] = False
        result = self.evaluate([self.scene, empty_timed_scene()])
        self.assertEqual(result['scene_count'], 2)
        self.assertEqual(result['fragment_count'], 4)
        self.assertEqual(result['reference_unavailable_fragment_count'], 2)
        self.assertEqual([row['has_supervision'] for row in result['per_fragment']], [True, True, False, False])
        self.assertEqual(result['per_scene'][1]['reference_targets_without_input'], ['empty-unobserved'])
        model = result['methods']['model']['regions']['all']
        base = result['methods']['position_base']['regions']['all']
        self.assertEqual(model['position']['intersection_count'], 6)
        self.assertEqual(base['position']['intersection_count'], 6)
        self.assertEqual(model['position']['prediction_without_target_count'], 10)
        self.assertEqual(base['position']['prediction_without_target_count'], 10)
        for field in ('speed', 'heading'):
            self.assertIsNone(base[field]['mae'])
            self.assertEqual(base[field]['available_prediction_count'], 0)
        # With the v2 residual head's zero initialization, the position control
        # exactly equals model positions on the same partially valid population.
        self.assertEqual(base['position']['mae'], model['position']['mae'])
        none = self.evaluate([empty_timed_scene()])
        self.assertEqual(none['completion_forward_count'], 0)
        self.assertIsNone(none['methods']['model']['regions']['all']['position']['mae'])

    def test_native_support_endpoints_and_distance_bins_partition_query_population(self):
        from tests.fixtures.scenes import native_scene
        from association.data.timed import prepare_timed_scene
        native = native_scene('completion', 'V_select')
        for track in native['tracks']:
            track['canonical_times'] *= 200
        native['intervals'] *= 200
        expanded = prepare_timed_scene(native, length=4, max_observations=3, time_unit_seconds=10.)
        expanded['fusion_targets'] = copy.deepcopy(self.scene['fusion_targets'])
        for target in expanded['fusion_targets'].values():
            target['query_times'] = expanded['grid'].copy()
        result = self.evaluate([expanded])
        for name in ('model', 'position_base'):
            regions = result['methods'][name]['regions']
            self.assertEqual(sum(regions[key]['position']['intersection_count'] for key in
                ('inside_support', 'before_support', 'after_support')), 16)
            self.assertEqual(regions['exact_observation']['position']['intersection_count'], 2)
            self.assertEqual(regions['inside_support']['position']['intersection_count'], 8)
            self.assertEqual(regions['inside_nonobserved']['position']['intersection_count'], 6)
            distances = result['methods'][name]['input_distance_metrics']
            self.assertEqual(sum(row['position']['intersection_count'] for row in distances.values()), 16)
            self.assertTrue(all(distances[key]['position']['intersection_count'] > 0 for key in
                ('le_300s', 'gt_300_le_600s', 'gt_600_le_1800s', 'gt_1800s')))
        # This short fragment has native input but no query inside its span.
        short = result['per_fragment'][0]
        self.assertEqual(short['region_query_counts']['inside_support'], 0)
        self.assertEqual(short['region_query_counts']['before_support'], 2)
        self.assertEqual(short['region_query_counts']['after_support'], 2)


if __name__ == '__main__':
    unittest.main()
