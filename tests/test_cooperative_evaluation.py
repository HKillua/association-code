"""Native cooperative inference, fixed identity and full-population contracts."""
import copy
import unittest
from unittest.mock import patch

import numpy as np
import torch

from association.cooperative import CooperativeConfig, CooperativeModel
from association.cooperative_evaluation import evaluate_cooperative, score_cooperative
from association.metrics import evaluate_scenes, select_threshold
from association.reference import fit_normalizer
from association.timed_inputs import prepare_timed_scene
from test_timed_pipeline import native_scene, timed_scene


def empty_timed_scene(name='empty', with_target=True):
    scene = native_scene(name, 'V_select')
    scene.update(tracks=[], node_ids=[], identities=np.array([], dtype=str), sources=[],
                 intervals=np.empty((0, 2)), pairs=np.empty((0, 2), dtype=np.int64),
                 labels=np.empty(0, dtype=np.float32))
    result = prepare_timed_scene(scene, length=4, max_observations=3,
                                 time_unit_seconds=10., query_times=np.array([0., 10., 20., 30.]))
    result['fusion_targets'] = ({name + '-unobserved': dict(states=np.zeros((4, 4)),
        valid=np.ones((4, 4), dtype=bool), query_times=result['grid'].copy())} if with_target else {})
    return result


class CooperativeEvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def setUp(self):
        torch.manual_seed(42)
        self.train = timed_scene('train', 'T_core')
        self.scene = timed_scene('validation', 'V_select')
        self.normalizer = fit_normalizer([self.train])
        self.model = CooperativeModel(CooperativeConfig(length=4, max_observations=3,
            d_model=8, nhead=2, num_layers=1, dim_feedforward=16, dropout=.1, head_hidden=8),
            self.normalizer)

    def evaluate(self, scenes=None, **kwargs):
        options = dict(device='cpu', batch_tracks=2, pair_batch_size=2,
                       select_on_validation=False, threshold=None, routing='plain')
        options.update(kwargs)
        return evaluate_cooperative(self.model, self.normalizer,
                                    [self.scene] if scenes is None else scenes, **options)

    def fixed(self, scene=None):
        scene = self.scene if scene is None else scene
        keys = ('scene_id', 'component_id', 'node_ids', 'identities', 'sources', 'intervals', 'labels', 'pairs')
        record = {key: copy.deepcopy(scene[key]) for key in keys}
        record['scores'] = np.where(scene['labels'], 2., -2.)
        return record

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

    def test_fixed_record_mismatch_or_missing_edges_fails_before_inference(self):
        for key in ('scene_id', 'component_id', 'node_ids', 'identities', 'sources', 'intervals', 'labels', 'pairs', 'scores'):
            with self.subTest(field=key):
                bad = self.fixed()
                if key in ('scene_id', 'component_id'):
                    bad[key] += '-wrong'
                elif key in ('node_ids', 'identities', 'sources'):
                    bad[key] = list(bad[key]); bad[key][0] += '-wrong'
                elif key == 'intervals':
                    bad[key][0, 0] += 1
                elif key == 'labels':
                    bad[key][0] = 1 - bad[key][0]
                elif key == 'pairs':
                    bad[key] = bad[key][:-1]
                else:
                    bad[key][0] = np.nan
                with patch.object(self.model, 'encode', side_effect=AssertionError('must validate first')):
                    with self.assertRaises(ValueError):
                        self.evaluate(fixed_records=[bad], threshold=0.)
        with self.assertRaisesRegex(ValueError, 'threshold refitting'):
            self.evaluate(fixed_records=[self.fixed()], select_on_validation=True)

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

    def test_score_api_selection_freeze_flags_and_v_only_boundary(self):
        next(self.model.parameters()).requires_grad_(False)
        before = [parameter.requires_grad for parameter in self.model.parameters()]
        records = score_cooperative(self.model, self.normalizer, [self.scene], 'cpu', 2, 2)
        result = self.evaluate(select_on_validation=True)
        np.testing.assert_array_equal(records[0]['scores'], result['records'][0]['scores'])
        self.assertEqual(result['selection'], select_threshold(records))
        self.assertEqual(before, [parameter.requires_grad for parameter in self.model.parameters()])
        self.assertFalse(self.model.training)
        self.assertTrue(all(parameter.grad is None for parameter in self.model.parameters()))
        for split in ('T_core', 'T_test'):
            with self.subTest(split=split):
                changed = copy.deepcopy(self.scene); changed['split'] = split
                with self.assertRaisesRegex(ValueError, 'V_select'):
                    self.evaluate([changed])
        with self.assertRaisesRegex(ValueError, 'distinct'):
            self.evaluate([self.scene, self.scene])
        with self.assertRaises(ValueError):
            self.evaluate(batch_tracks=0)
        self.normalizer = copy.deepcopy(self.normalizer)
        self.normalizer['mean'][0] += .1
        with self.assertRaisesRegex(ValueError, 'normalizer differs'):
            self.evaluate()


if __name__ == '__main__':
    unittest.main()
