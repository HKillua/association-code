import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from association.config import load_config, requested_seeds, validate_config
from association.runtime import (Run, device_from_string, load_records, read_json,
                                 save_records, write_json)

ROOT = Path(__file__).resolve().parents[1]


class RuntimeConfigTests(unittest.TestCase):
    def test_explicit_cuda_never_falls_back(self):
        with patch('torch.cuda.is_available', return_value=False):
            with self.assertRaisesRegex(RuntimeError, 'fallback is forbidden'):
                device_from_string('cuda:0')
        self.assertEqual(str(device_from_string('cpu')), 'cpu')
        with self.assertRaises(ValueError):
            device_from_string('mps')

    def test_seed_prefix_and_budget_validation(self):
        config = load_config(ROOT/'configs/pilot.json')
        self.assertEqual(requested_seeds(config, [20]), [20])
        for seeds in ([], [21], [20, 22], [20, 21, 22, 23]):
            with self.assertRaises(ValueError):
                requested_seeds(config, seeds)
        config['reference']['max_hours'] = float('nan')
        with self.assertRaises(ValueError):
            validate_config(config)

    def test_one_epoch_zero_warmup_keeps_a_nonzero_training_rate(self):
        import torch
        from association.reference import _set_epoch_lr
        parameter = torch.nn.Parameter(torch.tensor(1.0))
        optimizer = torch.optim.Adam([parameter], lr=.002)
        lr = _set_epoch_lr(optimizer, {'warmup_epochs': 0, 'learning_rate': .002}, 0, 1)
        self.assertEqual(lr, .002)
        optimizer.zero_grad()
        parameter.square().backward()
        optimizer.step()
        self.assertLess(parameter.item(), 1.0)

    def test_cache_rejects_modified_payload_and_binding(self):
        record = dict(scores=np.array([0.3]), labels=np.array([1]), pairs=np.array([[0, 1]]),
                      sources=['a', 'b'], identities=['x', 'x'], intervals=np.array([[0., 1.], [1., 2.]]),
                      scene_id='one', component_id='one', node_ids=['a', 'b'])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'cache'
            save_records(path, [record], {'split': 'T_head'})
            self.assertEqual(load_records(path, {'split': 'T_head'})[0]['scores'][0], .3)
            with self.assertRaises(ValueError):
                load_records(path, {'split': 'T_test'})
            (path/'scene_000000.npz').write_bytes(b'corrupt')
            with self.assertRaisesRegex(ValueError, 'checksum'):
                load_records(path)

    def test_run_refuses_overwrite_and_preserves_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'run'
            run = Run(path, {'kind': 'test'}, .01)
            run.finish('FAILED', error='fixture')
            self.assertEqual(read_json(path/'run.json')['status'], 'FAILED')
            with self.assertRaises(FileExistsError):
                Run(path, {})
            with self.assertRaises(ValueError):
                write_json(path/'bad.json', {'bad': float('nan')})

    def test_cuda_telemetry_failure_preserves_terminal_failure_hash_and_previous_peaks(self):
        for failed_query in ('is_available', 'max_memory_allocated', 'max_memory_reserved'):
            with self.subTest(query=failed_query), tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp)/'run'
                run = Run(path, {'provenance': {'device': 'cuda:0'},
                                 'cuda_process_peak_allocated_bytes': 123,
                                 'cuda_process_peak_reserved_bytes': 456})
                with patch('torch.cuda.is_available', return_value=True), \
                        patch('torch.cuda.max_memory_allocated', return_value=789), \
                        patch('torch.cuda.max_memory_reserved', return_value=999), \
                        patch('torch.cuda.' + failed_query, side_effect=RuntimeError('CUDA telemetry failed')):
                    run.finish('FAILED', failure_type='ValueError', failure='original failure',
                               profile_sha256='saved-profile-hash', completed_steps=1)
                saved = read_json(path/'run.json')
                self.assertEqual(saved['status'], 'FAILED')
                self.assertEqual(saved['failure_type'], 'ValueError')
                self.assertEqual(saved['failure'], 'original failure')
                self.assertEqual(saved['profile_sha256'], 'saved-profile-hash')
                self.assertEqual(saved['completed_steps'], 1)
                self.assertIn('CUDA telemetry failed', saved['cuda_memory_measurement_error'])
                self.assertEqual(saved['cuda_process_peak_allocated_bytes'],
                                 789 if failed_query == 'max_memory_reserved' else 123)
                self.assertEqual(saved['cuda_process_peak_reserved_bytes'], 456)


if __name__ == '__main__':
    unittest.main()
