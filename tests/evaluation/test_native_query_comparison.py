"""Small cached mathematical/parameter checks, not real-data research results."""
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import torch

from association.evaluation.fusion_controls import attribute_errors
from association.evaluation.sequence_position import evaluate_sequence_position
from association.models.sequence_position import SequencePosition, tangent_position
from association.training.native_query import NativeExperiment, ARMS
from tests.fixtures.cache import support_gate_cache_fixture
from tools.analysis import compare_native_query as compare


class NativeQueryComparisonTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_actual_common_parameters_and_frozen_budget_initialization(self):
        torch.manual_seed(71)
        base = SequencePosition(backbone='transformer')
        with torch.no_grad(): base.output.bias[0] = .2
        original = {key: value.clone() for key, value in base.state_dict().items()}
        initials = {}
        for arm in ARMS:
            torch.manual_seed(73)
            model = NativeExperiment(base, arm, dict(hidden_dim=64, patches=8, heads=4))
            initials[arm] = dict(base_state=model.base.state_dict(), branch_state=model.branch.state_dict())
        result = compare.verify_initial_states(initials, original)
        self.assertTrue(result['common_initial_tensors_exact'])
        self.assertTrue(result['budget_initial_equals_strong'])
        self.assertGreater(result['common_tensor_count'], 5)
        changed = deepcopy(initials)
        changed['query']['branch_state']['decoder.in_proj_weight'][0, 0] += .1
        with self.assertRaisesRegex(ValueError, 'outside pooler'):
            compare.verify_initial_states(changed, original)
        changed = deepcopy(initials)
        changed['mean']['base_state']['output.bias'][0] += .1
        with self.assertRaisesRegex(ValueError, 'frozen base tensors'):
            compare.verify_initial_states(changed, original)
        changed = deepcopy(initials)
        changed['budget']['branch_state']['output.bias'][0] += .1
        with self.assertRaisesRegex(ValueError, 'added-budget initial'):
            compare.verify_initial_states(changed, original)

    def test_screen_requires_both_strong_and_budget_and_retains_unscorable_scenes(self):
        def entry(query=95., empty=False):
            table = {}
            for arm, value in (('strong', 100.), ('budget', 120.), ('mean', 100.), ('query', query)):
                value = None if empty else value
                table[arm] = {weight: dict(mae=value, rmse=None if value is None else 1.5*value,
                                          p99=None if value is None else 2*value)
                              for weight in compare.WEIGHTINGS}
            return dict(position_weightings={'all': table})
        scenes = [entry(), entry(), entry(), entry(empty=True)]
        summary = dict(entry(), components={str(i): item for i, item in enumerate(scenes)}, remaining_components=entry())
        result = compare.scientific_screen(summary, scenes)
        self.assertEqual(result['query']['status'], 'PROMISING_SINGLE_SEED')
        self.assertFalse(result['query']['automatic_new_training'])
        self.assertEqual(result['query']['comparisons']['strong']['unevaluable_scenes'], 1)
        self.assertEqual(result['query']['comparisons']['strong']['total_scenes'], 4)
        self.assertEqual(result['mean']['status'], 'NO_CLEAR_SINGLE_SEED_ADVANTAGE')
        # A weak added-budget result cannot make a candidate that loses to b pass.
        summary['position_weightings'] = entry(query=105.)['position_weightings']
        result = compare.scientific_screen(summary, scenes)
        self.assertTrue(result['query']['comparisons']['budget']['screen_passed'])
        self.assertFalse(result['query']['comparisons']['strong']['screen_passed'])
        self.assertEqual(result['query']['status'], 'NO_CLEAR_SINGLE_SEED_ADVANTAGE')

    def test_complete_cached_population_metrics_mask_and_base_binding(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source, metadata, validation = support_gate_cache_fixture(root)
            base = SequencePosition(backbone='transformer').eval()
            with torch.no_grad(): base.output.bias[:] = torch.tensor([.1, -.05])
            with patch('association.evaluation.sequence_position.frozen_motion', side_effect=lambda raw, *_: raw):
                evaluate_sequence_position(base, validation, source, metadata, root/'strong/final')
            reference, original = compare._cache(root/'strong/final', '3035')
            arms = {name: dict(path=root/name) for name in ('strong', *ARMS)}
            for arm, increment in zip(ARMS, ([50., -20.], [10., 20.], [-5., 0.])):
                fragment, system = deepcopy(reference), deepcopy(original)
                teacher = original['learned_sequence_group_states']
                residual = np.broadcast_to(increment, (*teacher.shape[:2], 2)).copy()
                origin = teacher
                if arm == 'budget':
                    origin = original['fixed_prior095_group_states']
                    residual += reference['group_tangent_residual_m']
                states = tangent_position(torch.from_numpy(origin.reshape(-1, 4)),
                    torch.from_numpy(residual.reshape(-1, 2))).numpy().reshape(origin.shape)
                system['learned_sequence_group_states'] = states
                aligned = np.full_like(system['learned_sequence_aligned_prediction'], np.nan)
                for row, gid in enumerate(system['learned_sequence_aligned_group_indices']):
                    if gid >= 0: aligned[row] = states[gid]
                system['learned_sequence_aligned_prediction'] = aligned
                fragment.update(learned_sequence_group_states=states, group_frozen_base_states=teacher.copy(),
                                group_native_increment_m=residual)
                t = len(system['learned_sequence_target_ids'])
                error = attribute_errors(aligned[:t], system['learned_sequence_aligned_truth'][:t])['position']
                fragment['learned_sequence_position_error'] = np.where(
                    np.isfinite(reference['learned_sequence_position_error']), error, np.nan)
                for sub, values in (('predictions', fragment), ('system-predictions', system)):
                    path = root/arm/'final'/sub; path.mkdir(parents=True)
                    np.savez_compressed(path/'states-3035.npz', **values)
            row, data = compare.scene_pair(arms, '3035')
            row['component_id'] = 'fixture'
            summary = compare.summarize([row], [data])
            self.assertEqual(summary['population']['predicted_group_count'], 4)
            self.assertEqual(summary['population']['duplicate_prediction_group_count'], 1)
            self.assertEqual(summary['position_population']['covered_target_points'], 127)
            for arm in ('strong', *ARMS):
                error = data[arm+'_position_error']; values = error[np.isfinite(error)]
                actual = summary['position_weightings']['all'][arm]['point_equal']
                self.assertAlmostEqual(actual['mae'], float(values.mean()))
                self.assertAlmostEqual(actual['rmse'], float(np.sqrt(np.mean(values ** 2))))
                self.assertAlmostEqual(actual['p99'], float(np.quantile(values, .99)))
            self.assertEqual(summary['pairwise']['query__minus__mean']['all']['common_point_count'], 127)
            self.assertEqual(summary['pairwise']['query__minus__budget']['no_predicted_group']['common_point_count'], 0)
            path = root/'query/final/system-predictions/states-3035.npz'
            with np.load(path, allow_pickle=False) as saved: good = {key: saved[key].copy() for key in saved.files}
            changed = deepcopy(good); changed['learned_sequence_aligned_truth_valid'][0, 0, 0] = False
            np.savez_compressed(path, **changed)
            with self.assertRaisesRegex(ValueError, 'target/mask/group population'):
                compare.scene_pair(arms, '3035')
            np.savez_compressed(path, **good)
            path = root/'query/final/predictions/states-3035.npz'
            with np.load(path, allow_pickle=False) as saved: bad = {key: saved[key].copy() for key in saved.files}
            bad['group_frozen_base_states'][0, 0, 0] += .001
            np.savez_compressed(path, **bad)
            with self.assertRaisesRegex(ValueError, 'frozen base versus090C'):
                compare.scene_pair(arms, '3035')


if __name__ == '__main__': unittest.main()
