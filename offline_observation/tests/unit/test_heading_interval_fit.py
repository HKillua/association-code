"""Software-only tests for the C150 matched evaluation and execution boundary."""

from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from tools.analysis import heading_interval_fit as c150


def example():
    query = np.arange(64, dtype=float)
    states = np.zeros((3, 64, 4))
    states[..., 3] = 160.
    states[2, 0, 3] = 350.
    branch = np.zeros((3, 64), dtype=np.uint8)
    branch[[0, 2], 0] = c150.LOW
    enabled = branch == c150.LOW
    new = states.copy()
    new[0, 0, 3], new[2, 0, 3] = 10., 145.
    chosen = np.full(branch.shape, -1, dtype=int)
    chosen[enabled] = [0, 1]
    interval = np.full((*branch.shape, 6), np.nan)
    interval[enabled] = [[-1., 0., 0., 12., .01, 12.], [1., 2., 0., 12., .01, 12.]]
    provider = np.full(branch.shape, -1, dtype=int)
    provider[enabled] = [1, 3]
    side = np.zeros(branch.shape, np.uint8)
    side[enabled] = [1, 3]
    numeric = np.full(branch.shape, np.nan)
    numeric[enabled] = 1.
    return dict(states=states, new_states=new, branch=branch, enabled=enabled,
                fallback_reason=np.where(enabled, 3, 0).astype(np.uint8),
                node_ids=np.array(['n0', 'n1', 'n2', 'n3', 'n4']), query=query,
                sources=np.array(['9001', '9002', '9001', '9001', '9002']),
                membership=np.array([0, 0, 1, 2, 2]), support=np.ones((5, 64), bool), nearest=np.zeros((5, 64)),
                candidate_groups=np.array([0, 2]), candidate_provider_offsets=np.array([0, 1, 3]),
                candidate_provider_nodes=np.array([1, 3, 4]), selected_interval_index=chosen,
                selected_interval=interval, selected_provider_node=provider, selected_time_side=side,
                **{f'selected_{key}': numeric.copy() for key in c150.CHOSEN_NUMBERS})


def reference(path, data):
    ids = np.array(['ship-a', 'ship-b', 'ship-c'])
    truth = np.zeros((3, 64, 4))
    truth[1, 0, 3] = 340.
    valid = np.ones(truth.shape, bool)
    valid[2, 1, 3] = False
    np.savez(path, node_ids=data['node_ids'], query_seconds=data['query'],
             identities=np.array(['ship-a', 'ship-a', 'ship-a', 'ship-b', 'ship-c']),
             target_ids=ids, target_states=truth, target_valid=valid)


class HeadingIntervalFitTest(unittest.TestCase):
    def test_first_claim_and_duplicate_and_missing_coverage(self):
        with TemporaryDirectory() as tmp:
            file = Path(tmp)/'evaluation.npz'
            data = example()
            reference(file, data)
            result, vectors, selected = c150.measure_scene(data, file)
        self.assertEqual(result['population']['duplicate_prediction_group_count'], 1)
        self.assertEqual(result['population']['anchor_matched_target_count'], 2)
        self.assertEqual(result['population']['unclaimed_target_heading_reference_points'], 63)
        self.assertEqual(result['coverage']['heading']['intersection_count'], 128)
        self.assertEqual(result['coverage']['heading']['available_target_count'], 191)
        np.testing.assert_array_equal(vectors['old_error'][vectors['enabled']], [160., 10.])
        np.testing.assert_array_equal(vectors['new_error'][vectors['enabled']], [10., 165.])
        np.testing.assert_array_equal(selected['provider_identity'], [1, 3])
        np.testing.assert_array_equal(selected['time_side'], [1, 3])
        np.testing.assert_array_equal(selected['provider_copies'], [1, 2])
        enabled = result['stats']['enabled']
        self.assertEqual((enabled['rescued_tail'], enabled['new_tail']), (1, 1))
        mixed = result['stats']['mixed']['old_non_tail']
        self.assertEqual(mixed['count'], 1)
        self.assertEqual((mixed['harmed'], mixed['harmed_gt30'], mixed['harmed_gt90']), (1, 1, 1))
        self.assertEqual(mixed['mean_delta_absolute_error'], 155.)

    def test_promotion_uses_whole_original_non_tail_population_and_strict_tails(self):
        def pool(old, new, pure, enabled):
            vectors = dict(old_error=np.array(old), new_error=np.array(new), pure=np.array(pure),
                           enabled=np.array(enabled), branch=np.ones(len(old), int))
            return {'stats': c150.heading_statistics(vectors, detailed=False)}

        good = pool([155., 160., 15., 25.], [40., 30., 14., 25.], [True, False, True, False], [True, True, True, False])
        mixed_harm = pool([155., 160., 15., 25.], [40., 30., 14., 26.], [True, False, True, False], [True]*4)
        no_effect = pool([155., 25.], [155., 25.], [True, True], [False, False])
        components = {'largest': good, 'first': good, 'second': good, 'zero': no_effect}
        succeed = c150.promotion(good, good, components, 'largest')
        self.assertTrue(succeed['passes'])
        self.assertEqual(succeed['qualifying_components_without_largest'], ['first', 'second'])
        self.assertEqual(succeed['component_mae_direction']['zero_enabled'], 1)
        self.assertIsNone(c150.promotion(good, no_effect, components, 'largest')['population_gates']
                          ['without_largest']['original_non_tail']['mixed'])
        self.assertFalse(c150.promotion(good, no_effect, components, 'largest')['passes'])
        failure = c150.promotion(mixed_harm, good, components, 'largest')
        self.assertTrue(all(failure['population_gates']['full']['main']['all'].values()))
        self.assertFalse(failure['population_gates']['full']['original_non_tail']['mixed'])
        self.assertFalse(failure['passes'])

    def test_frozen_arms_reject_speed_and_non_enabled_angle_edits(self):
        data = example()
        with TemporaryDirectory() as tmp:
            baseline = Path(tmp)/'baseline.npz'
            np.savez(baseline, states=data['states'], membership=data['membership'])
            with np.load(baseline, allow_pickle=False) as saved:
                c150.verify_arms(data, saved)
                for col in (2, 3):
                    invalid = dict(data, new_states=data['new_states'].copy())
                    invalid['new_states'][0, 1, col] += 1.
                    with self.assertRaises(AssertionError):
                        c150.verify_arms(invalid, saved)

    def exercise_run(self, corrupt_after_freeze):
        class Archive:
            def __init__(self, paths):
                self.members_read = []

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            out = root/'out'
            out.mkdir()
            ids = ['002', '030']
            data = example()
            for sid in ids:
                reference(root/f'{sid}.npz', data)
            original_path = root/'summary.json'
            c150.write(original_path, {'largest_scene_component': 'big', 'full': {}, 'without_largest': {}})
            source_hash = dict(prediction_sha256='old', motion_sha256='motion', ws_sha256='ws')
            previous = dict(scene_order_sha256='order', scene_hashes=dict.fromkeys(ids, source_hash))
            binding = (ids, {'002': 'big', '030': 'small'}, {sid: {'evaluation': root/f'{sid}.npz'} for sid in ids},
                       root, [], root, previous, 'big')

            def freeze_one(archive, sid, row, baseline_path, ws_path, expected, path):
                self.assertFalse((out/'freeze.json').exists())
                self.assertEqual(expected, source_hash)
                archive.members_read.append({'scene_id': sid, 'kind': 'measurements'})
                np.savez(path, **data)
                return dict(prediction_sha256=c150.file_hash(path), source_c149=expected)

            measure_scene = c150.measure_scene

            def after_freeze(saved, evaluation):
                freeze = c150.read(out/'freeze.json')
                self.assertTrue(freeze['all_predictions_frozen_before_evaluation'])
                self.assertEqual(list(freeze['scene_hashes']), ids)
                for sid in ids:
                    self.assertEqual(freeze['scene_hashes'][sid]['prediction_sha256'],
                                     c150.file_hash(out/'private'/f'{sid}.npz'))
                return measure_scene(saved, evaluation)

            args = SimpleNamespace(runs=root, manifest=root/'manifest.json', output=out)
            trace = []
            evaluations = []

            def checkpoint(phase, **state):
                trace.append((phase, state))
                if corrupt_after_freeze and phase == 'evaluate' and state.get('scored') == 0:
                    (out/'private'/f'{ids[0]}.npz').write_bytes(b'corrupt')

            def score(saved, evaluation):
                evaluations.append(evaluation)
                return after_freeze(saved, evaluation)

            with (patch.object(c150, 'bind_c149', return_value=binding),
                  patch.object(c150, 'MTADArchive', Archive), patch.object(c150, 'validate_measurement_archive'),
                  patch.object(c150, 'freeze_one', side_effect=freeze_one),
                  patch.object(c150, 'measure_scene', side_effect=score),
                  patch.object(c150, 'verify_c149_controls') as verify,
                  patch.dict(c150.C149_SHA256, summary=c150.file_hash(original_path))):
                if corrupt_after_freeze:
                    with self.assertRaisesRegex(ValueError, 'frozen prediction digest differs'):
                        c150.run(args, checkpoint)
                    self.assertEqual(evaluations, [])
                    self.assertFalse((out/'summary.json').exists())
                    return
                c150.run(args, checkpoint)
            self.assertEqual(len(evaluations), 2)
            self.assertEqual(verify.call_count, 2)
            self.assertEqual(trace[-1][1]['status'], 'COMPLETED')
            completed = c150.read(out/'summary.json')
            self.assertEqual(completed['full']['stats']['enabled']['count'], 4)
            self.assertEqual(completed['without_largest']['stats']['enabled']['count'], 2)
            self.assertFalse(completed['decision']['passes'])

    def test_run_writes_all_prediction_hashes_before_any_evaluation(self):
        self.exercise_run(False)

    def test_corrupt_prediction_is_rejected_before_truth(self):
        self.exercise_run(True)


if __name__ == '__main__':
    unittest.main()
