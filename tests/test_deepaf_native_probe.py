"""Native-style driver checks, not a production or original-paper result."""
from copy import deepcopy
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


if __name__ == '__main__':
    unittest.main()
