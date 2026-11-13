import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from association.utils.runtime import (Run, device_from_string, load_records, read_json,
                                 save_records, write_json)

ROOT = Path(__file__).resolve().parents[2]


class RuntimeConfigTests(unittest.TestCase):
    def test_explicit_cuda_never_falls_back(self):
        with patch('torch.cuda.is_available', return_value=False):
            with self.assertRaisesRegex(RuntimeError, 'fallback is forbidden'):
                device_from_string('cuda:0')
        self.assertEqual(str(device_from_string('cpu')), 'cpu')
        with self.assertRaises(ValueError):
            device_from_string('mps')


    def test_cache_checks_actual_data_and_split(self):
        record = dict(scores=np.array([0.3]), labels=np.array([1]), pairs=np.array([[0, 1]]),
                      sources=['a', 'b'], identities=['x', 'x'], intervals=np.array([[0., 1.], [1., 2.]]),
                      scene_id='one', component_id='one', node_ids=['a', 'b'])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'cache'
            save_records(path, [record], {'split': 'T_head'})
            self.assertEqual(load_records(path, {'split': 'T_head'})[0]['scores'][0], .3)
            with self.assertRaises(ValueError):
                load_records(path, {'split': 'T_test'})
            index = read_json(path/'index.json')
            index['binding']['legacy_note'] = 'extra metadata is not an execution gate'
            write_json(path/'index.json', index)
            self.assertEqual(load_records(path, {'split': 'T_head'})[0]['scores'][0], .3)
            for values, message in (({'labels': np.array([0])}, 'labels disagree'),
                                    ({'labels': np.array([2])}, 'labels binary'),
                                    ({'scores': np.array([.3, .4])}, 'record needs')):
                with self.subTest(message=message):
                    changed = {key: value for key, value in record.items()
                               if key in ('scores', 'labels', 'pairs', 'sources', 'identities', 'intervals')}
                    changed.update(values)
                    np.savez(path/'scene_000000.npz', **changed)
                    with self.assertRaisesRegex(ValueError, message):
                        load_records(path)
            (path/'scene_000000.npz').write_bytes(b'corrupt')
            with self.assertRaises(ValueError):
                load_records(path)
            index['records'][0]['file'] = '../elsewhere.npz'
            write_json(path/'index.json', index)
            with self.assertRaisesRegex(ValueError, 'path invalid'):
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


if __name__ == '__main__':
    unittest.main()
