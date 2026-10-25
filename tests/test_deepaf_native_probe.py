"""Native-style driver checks, not a production or original-paper result."""
from copy import deepcopy
from contextlib import redirect_stderr
import io
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import torch

from association.models.deepaf_peer import DeepAFPeer
from association.utils.runtime import read_json
from tools.analysis import deepaf_native_probe as probe


class NativeProbeTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(probe.SEED)

    def test_simulation_units_modes_and_label_distinction(self):
        scenes = probe.generate_split(2, probe.SEED, 'check')
        self.assertEqual({s['motion'] for s in scenes}, set(probe.MODES))
        for scene in scenes:
            speed = np.linalg.norm(scene['truth_velocity'], axis=-1)
            self.assertTrue(np.all(speed >= 50. - 1e-8))
            self.assertTrue(np.all(speed <= 300. + 1e-8))
            self.assertEqual(len(scene['measurements']), 25)
            self.assertEqual(scene['initial_observations'].shape, (6, 2))
        counts = probe.split_counts(scenes)
        self.assertEqual(counts['future_scans'], 150)
        self.assertEqual(counts['target_detections']+counts['missed_target_detections'], 150)
        self.assertGreater(counts['clutter_candidates'], 0)
        model = DeepAFPeer()
        chosen = [next(s for s in scenes if s['motion'] == 'CA')]
        result = probe.rollout(model, probe.observable_batch(chosen))
        pn, vn = probe.supervision(chosen, result, 'native_labels')
        pa, va = probe.supervision(chosen, result, 'adapted_labels')
        self.assertFalse(torch.equal(pn, pa))
        self.assertFalse(torch.equal(vn, va))
        torch.testing.assert_close(pn, va)

    def test_truth_and_motion_tags_do_not_enter_forward(self):
        scenes = probe.generate_split(1, 17, 'clean')
        changed = deepcopy(scenes)
        for scene in changed:
            scene['truth_position'][:] = np.nan
            scene['truth_velocity'][:] = 1e100
            scene['motion'], scene['scene_id'] = 'arbitrary', 'changed'
            scene['detected'][:] = False
        a, b = probe.observable_batch(scenes), probe.observable_batch(changed)
        self.assertEqual(set(a), probe.VISIBLE_KEYS)
        for key in a:
            self.assertTrue(torch.equal(a[key], b[key]))
        model = DeepAFPeer()
        self.assertTrue(torch.equal(probe.rollout(model, a)['position'], probe.rollout(model, b)['position']))
        a['truth_position'] = torch.zeros(1)
        with self.assertRaises(ValueError):
            probe.rollout(model, a)

    def test_carry_only_missing_frames_and_own_feedback(self):
        scene = probe.generate_split(1, 41, 'empty')[0]
        scene['measurements'] = [np.empty((0, 2)) for _ in scene['measurements']]
        result = probe.rollout(DeepAFPeer(), probe.observable_batch([scene]), cache=True)
        valid, candidate = result['cache']['valid'], result['cache']['candidate']
        self.assertTrue(valid[..., 0].all())
        self.assertFalse(valid[..., 1:].any())
        torch.testing.assert_close(candidate[:, 1:, 0].double() * probe.SCALE,
                                   result['velocity'][:, :-1], rtol=1e-6, atol=1e-5)
        self.assertTrue(torch.isfinite(result['position']).all())

    def test_real_two_update_two_arm_endpoint_and_preserved_output(self):
        with TemporaryDirectory() as directory:
            path = Path(directory)/'probe'
            result = probe.run_probe(path, updates=2, train_per_motion=1, test_per_motion=1)
            self.assertEqual(result['status'], 'COMPLETED')
            self.assertFalse(result['production_level1'])
            self.assertFalse(result['original_paper_reproduction'])
            self.assertEqual(result['actual_updates'], dict(native_labels=2, adapted_labels=2))
            logs = [read_json(path/(arm+'-training.json')) for arm in probe.ARMS]
            self.assertEqual([r['scene_indices'] for r in logs[0]], [r['scene_indices'] for r in logs[1]])
            self.assertEqual(set(result['final']['native_labels']), {'all', *probe.MODES})
            before = (path/'run.json').read_bytes()
            with self.assertRaises(FileExistsError):
                probe.run_probe(path, updates=1, train_per_motion=1, test_per_motion=1)
            self.assertEqual(before, (path/'run.json').read_bytes())

    def test_truth_reset_is_previous_position_only_without_feedback(self):
        scenes = probe.generate_split(1, 53, 'shadow')
        visible = probe.observable_batch(scenes)
        model = DeepAFPeer().eval()
        actual = probe.rollout(model, visible, cache=True)
        before = deepcopy(actual)
        truth = np.stack([s['truth_position'][probe.INITIAL_COUNT:] for s in scenes])
        shadow = probe.truth_reset_shadow(model, visible, actual, truth)
        torch.testing.assert_close(shadow['position'][:, 0], actual['position'][:, 0], rtol=0, atol=0)
        torch.testing.assert_close(shadow['velocity'][:, 0], actual['velocity'][:, 0], rtol=0, atol=0)
        # Independent single-query reconstruction uses actual V and both actual
        # history tensors, never shadow V or shadow P increments.
        k = 2
        p = torch.tensor(truth[:, k-1], dtype=torch.float64)
        live = actual['cache']['valid'][:, k]
        candidate = torch.cat((actual['velocity'][:, k-1, None],
                               visible['measurements'][:, k]-p[:, None]), 1)
        candidate = torch.where(live[..., None], candidate, torch.zeros_like(candidate)).float()/probe.SCALE
        with torch.no_grad():
            hv = torch.ones(actual['cache']['ph'][:, k].shape[:2], dtype=torch.bool)
            expected_p = p + probe.SCALE * model.position_network(
                candidate, live, actual['cache']['ph'][:, k], hv).double()
            expected_v = probe.SCALE * model.velocity_network(
                candidate, live, actual['cache']['vh'][:, k], hv).double()
        torch.testing.assert_close(shadow['position'][:, k], expected_p, rtol=0, atol=0)
        torch.testing.assert_close(shadow['velocity'][:, k], expected_v, rtol=0, atol=0)
        changed_truth = truth.copy()
        changed_truth[:, 0] += 10000.
        changed = probe.truth_reset_shadow(model, visible, actual, changed_truth)
        self.assertFalse(torch.equal(changed['position'][:, 1], shadow['position'][:, 1]))
        torch.testing.assert_close(changed['position'][:, 2:], shadow['position'][:, 2:], rtol=0, atol=0)
        for key in ('position', 'velocity', 'previous_position'):
            self.assertTrue(torch.equal(actual[key], before[key]))
        for key in actual['cache']:
            self.assertTrue(torch.equal(actual['cache'][key], before['cache'][key]))

    def test_qualification_actual_checkpoints_sampling_and_oracle_separation(self):
        with TemporaryDirectory() as directory:
            path = Path(directory)/'qualification'
            result = probe.run_probe(path, updates=2, midpoint=1, train_per_motion=1,
                                     test_per_motion=1, qualification=True)
            self.assertEqual(result['actual_updates'], {'adapted_labels': 2})
            self.assertEqual(result['scope'], probe.QUALIFICATION_SCOPE)
            self.assertEqual(result['endpoint_updates'], [0, 1, 2])
            self.assertFalse((path/'native_labels-training.json').exists())
            for step in (0, 1, 2):
                checkpoint = torch.load(path/f'adapted_labels-step{step}.pt', weights_only=False)
                self.assertEqual(checkpoint['completed_updates'], step)
                self.assertEqual(len(checkpoint['completed_sampling_records']), step)
                self.assertEqual(checkpoint['planned_scene_indices'].shape, (2, 8))
                self.assertEqual(bool(checkpoint['optimizer']['state']), bool(step))
                self.assertTrue(all(int(s['step']) == step for s in checkpoint['optimizer']['state'].values()))
                loaded = DeepAFPeer()
                loaded.load_state_dict(checkpoint['model'])
                evaluation = read_json(path/f'adapted_labels-step{step}-heldout.json')
                replay = probe.evaluate(loaded, probe.generate_split(1, probe.SEED+1, 'heldout'), truth_reset=True)
                self.assertEqual(evaluation['groups'], replay['groups'])
                diagnostic = evaluation['truth_reset_previous_position_only']
                self.assertTrue(diagnostic['oracle_diagnostic'])
                self.assertFalse(diagnostic['peer_score'])
                self.assertFalse(diagnostic['feedback_used'])
                self.assertEqual(diagnostic['reset_query_indices'], list(range(1, 25)))
                self.assertEqual(len(diagnostic['scenes']), 3)
                self.assertEqual({r['motion'] for r in diagnostic['scenes']}, set(probe.MODES))
                self.assertTrue(all(len(r['position_errors_m']) == 25 for r in diagnostic['scenes']))
            logs = read_json(path/'adapted_labels-training.json')
            checkpoint = torch.load(path/'adapted_labels-step2.pt', weights_only=False)
            self.assertEqual(logs, checkpoint['completed_sampling_records'])
            np.testing.assert_array_equal([row['scene_indices'] for row in logs], checkpoint['planned_scene_indices'])
            self.assertEqual(len((path/'adapted_labels-sampling.jsonl').read_text().splitlines()), 2)

    def test_qualification_cli_rejects_smaller_production_budget(self):
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
            probe.main(['--mode', 'qualification', '--output', 'not-created', '--updates', '2'])
        self.assertEqual(error.exception.code, 2)


if __name__ == '__main__':
    unittest.main()
