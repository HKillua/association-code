"""099 real source4 -> scratch8 replay, continuation and failure boundaries."""
from argparse import Namespace
import copy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import time
import unittest
from unittest.mock import patch

import torch

from association.training import deepaf_peer as training
from association.training import sequence_position as source_training
from association.utils.runtime import read_json
from tests.training.test_sequence_position import tiny_training
from tools.experiments import deepaf_peer_fit as entry


def tiny_configuration(*, learning_rate=.001):
    config = read_json(entry.ROOT/'configs/deepaf-peer-level1-lr1e3.json')
    config['training'].update(updates=8, source_updates=4, batch_size=2, learning_rate=learning_rate)
    return config


class DeepAFTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.directory = TemporaryDirectory()
        cls.root = Path(cls.directory.name)
        cls.train = tiny_training()
        cls.config = tiny_configuration()
        source_training.train_sequence_position(cls.train, training.source_configuration(cls.config), cls.root/'source')
        cls.source_bytes = (cls.root/'source/final.pt').read_bytes()
        started = time.monotonic()
        with patch.object(training, 'HEARTBEAT_INTERVAL', 2):
            cls.result = training.train_deepaf_peer(cls.train, cls.config, cls.root/'peer',
                paired_reference_fit=cls.root/'source')
        cls.elapsed_seconds = time.monotonic()-started

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def test_real_replay_resets_only_sampler_and_preserves_continuous_adam(self):
        result = self.result
        self.assertEqual(result['status'], 'COMPLETED')
        self.assertEqual(result['optimizer_updates'], 8)
        self.assertEqual(result['sampler_reset_count'], 1)
        self.assertEqual(result['completed_replay_cycles'], 2)
        self.assertFalse(result['source_weights_inherited'])
        self.assertFalse(result['test_opened'])
        self.assertFalse(result['validation_opened'])
        self.assertEqual(result['parameter_count'], result['trainable_parameter_count'])
        self.assertGreater(result['nonzero_gradient_updates'], 0)
        self.assertTrue(all(count > 0 for count in result['network_gradients'].values()))
        old_rows = [json.loads(line) for line in (self.root/'source/training.jsonl').read_text().splitlines()]
        rows = [json.loads(line) for line in (self.root/'peer/training.jsonl').read_text().splitlines()]
        self.assertEqual([row['batch'] for row in rows], [row['batch'] for row in old_rows]*2)
        self.assertEqual([row['mode'] for row in rows], [row['mode'] for row in old_rows]*2)
        self.assertEqual([row['source_step'] for row in rows], [1, 2, 3, 4]*2)
        self.assertEqual([row['replay_cycle'] for row in rows], [0]*4+[1]*4)
        self.assertEqual(rows[3]['mode'], rows[4]['mode'])  # Both are source-cycle clean.
        self.assertNotEqual(rows[4]['mode'], training.TRAIN_MODES[4 % 3])
        states = []
        for filename, step in (('step-0.pt', 0), ('step-4.pt', 4), ('final.pt', 8)):
            model, saved = training.load_checkpoint(self.root/'peer'/filename)
            self.assertEqual(saved['step'], step)
            self.assertEqual(saved['actual_updates'], step)
            self.assertEqual(saved['checkpoint_kind'], 'endpoint')
            self.assertTrue(all(parameter.requires_grad for parameter in model.parameters()))
            if step:
                self.assertTrue(all(float(item['step']) == step for item in saved['optimizer']['state'].values()))
            else:
                self.assertEqual(saved['optimizer']['state'], {})
            states.append(saved)
        self.assertEqual(states[1]['numpy_rng'], states[2]['numpy_rng'])
        self.assertNotEqual(states[0]['numpy_rng'], states[1]['numpy_rng'])
        for prefix in result['network_gradients']:
            name = prefix.removesuffix('_nonzero_updates')+'.'
            self.assertTrue(any(not torch.equal(states[0]['model'][key], value)
                                for key, value in states[2]['model'].items() if key.startswith(name)))
        self.assertEqual((self.root/'source/final.pt').read_bytes(), self.source_bytes)
        proof = training.verify_replay(self.root/'peer', self.root/'source')
        self.assertEqual(proof['matched_updates'], 8)
        self.assertEqual(proof['matched_anchors'], 16)
        self.assertEqual(proof['status'], 'PAIR_VERIFIED')
        self.assertEqual(read_json(self.root/'peer/heartbeat.json')['optimizer_updates'], 8)

    def test_fixed_training_probe_is_same_batch_with_zero_training_exposure(self):
        probe = read_json(self.root/'peer/training-probe.json')
        self.assertEqual(probe['status'], 'COMPLETED')
        self.assertFalse(probe['selection'])
        self.assertEqual(probe['optimizer_updates'], 0)
        endpoints = probe['endpoints']
        self.assertEqual([point['step'] for point in endpoints], [0, 4, 8])
        self.assertTrue(all(point['batch'] == endpoints[0]['batch'] for point in endpoints))
        self.assertTrue(all(not point['sampled'] and point['optimizer_updates'] == 0 for point in endpoints))
        for point in endpoints:
            for key in ('objective', 'position_mse', 'velocity_mse', 'position_mae_m'):
                self.assertTrue(torch.isfinite(torch.tensor(point[key])), key)
        self.assertEqual(self.result['training_diagnostics']['diagnostic_group_forwards'], 6)
        self.assertEqual(self.result['exposure']['sampled_anchors'], 16)

    def test_corrupt_checkpoint_and_replay_are_rejected(self):
        _, final = training.load_checkpoint(self.root/'peer/final.pt')
        changed = copy.deepcopy(final)
        next(iter(changed['optimizer']['state'].values()))['step'] -= 4
        torch.save(changed, self.root/'bad.pt')
        with self.assertRaisesRegex(ValueError, 'successful update'):
            training.load_checkpoint(self.root/'bad.pt')
        changed = copy.deepcopy(final)
        first = next(iter(changed['optimizer']['state']))
        changed['optimizer']['state'][first]['exp_avg'] = torch.zeros(1)
        torch.save(changed, self.root/'bad.pt')
        with self.assertRaisesRegex(ValueError, 'moment'):
            training.load_checkpoint(self.root/'bad.pt')
        _, initial = training.load_checkpoint(self.root/'peer/step-0.pt')
        key = next(key for key, tensor in initial['model'].items() if tensor.is_floating_point())
        initial['model'][key].flatten()[0] += .25
        torch.save(initial, self.root/'bad.pt')
        with self.assertRaisesRegex(ValueError, 'scratch initialization'):
            training.load_checkpoint(self.root/'bad.pt')
        changed = copy.deepcopy(final)
        changed['numpy_rng'] = changed['source_sampling']['initial_numpy_rng']
        torch.save(changed, self.root/'bad.pt')
        with self.assertRaisesRegex(ValueError, 'sampler RNG'):
            training.load_checkpoint(self.root/'bad.pt')
        history = self.root/'peer/training.jsonl'
        original = history.read_text()
        try:
            rows = [json.loads(line) for line in original.splitlines()]
            rows[4]['source_step'] = 2
            history.write_text(''.join(json.dumps(row)+'\n' for row in rows))
            with self.assertRaisesRegex(ValueError, 'source-step replay'):
                training.verify_replay(self.root/'peer')
        finally:
            history.write_text(original)

    def test_optimizer_failure_is_not_counted_and_duplicate_output_is_immutable(self):
        preserved = (self.root/'peer/run.json').read_bytes()
        with self.assertRaises(FileExistsError):
            training.train_deepaf_peer(self.train, self.config, self.root/'peer',
                paired_reference_fit=self.root/'source')
        self.assertEqual((self.root/'peer/run.json').read_bytes(), preserved)
        with patch.object(torch.optim.Adam, 'step', side_effect=RuntimeError('optimizer failure')):
            with self.assertRaisesRegex(RuntimeError, 'optimizer failure'):
                training.train_deepaf_peer(self.train, self.config, self.root/'failed',
                    paired_reference_fit=self.root/'source')
        failed = read_json(self.root/'failed/run.json')
        self.assertEqual(failed['status'], 'FAILED')
        self.assertEqual(failed['optimizer_updates'], 0)
        self.assertEqual(failed['exposure']['sampled_anchors'], 0)
        self.assertEqual((self.root/'failed/training.jsonl').read_text(), '')
        self.assertFalse((self.root/'failed/final.pt').exists())
        _, recovery = training.load_checkpoint(self.root/'failed/interrupted.pt')
        self.assertEqual(recovery['checkpoint_kind'], 'recovery')
        self.assertEqual(recovery['step'], 0)
        self.assertEqual(recovery['optimizer']['state'], {})

    def test_frozen_configs_and_learning_rate_selection_boundaries(self):
        manifest = read_json(entry.ROOT/'configs/data/mtad-level1-scene3000.json')
        alignment = read_json(entry.ROOT/'configs/data/mtad-level1-alignment.json')
        for key, filename in entry.CONFIGS.items():
            config = read_json(entry.ROOT/'configs'/filename)
            self.assertEqual(key, (config['training']['seed'], config['training']['learning_rate']))
            entry.production_contract(config, manifest, alignment)
            self.assertEqual(training.source_configuration(config)['training']['learning_rate'], .001)
            config['training']['updates'] += 1
            with self.assertRaises(ValueError):
                entry.production_contract(config, manifest, alignment)
        replica = read_json(entry.ROOT/'configs/deepaf-peer-replica-lr1e3.json')
        with self.assertRaisesRegex(ValueError, 'requires completed seed23'):
            entry._selection(Namespace(selection_record=None), replica)


if __name__ == '__main__':
    unittest.main()
