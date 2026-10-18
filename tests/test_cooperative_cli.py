from dataclasses import asdict
import io
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from association import runtime
from association.cooperative import CooperativeConfig
from association.cooperative_training import load_cooperative_checkpoint
from tools.experiments.cooperative_fit import main, synthetic_configuration
import test_timed_dataset as native_fixture


class CooperativeCLITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def invoke(self, args):
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            return main(args)

    def test_normal_cli_runs_all_fixed_identity_arms_and_preserves_outputs(self):
        output = self.root/'check'
        args = ['check', '--device', 'cpu', '--output', str(output), '--cpu-threads', '1']
        self.assertEqual(self.invoke(args), 0)
        run = runtime.read_json(output/'run.json')
        self.assertEqual(run['status'], 'COMPLETED')
        self.assertEqual(run['completed_routes'], ['plain', 'identity', 'utility', 'learned'])
        original = (output/'run.json').read_bytes()
        self.assertEqual(self.invoke(args), 1)
        self.assertEqual((output/'run.json').read_bytes(), original)
        arrays = [np.load(output/(route+'-scores.npz')) for route in run['completed_routes']]
        self.addCleanup(lambda: [item.close() for item in arrays])
        for row in arrays[1:]:
            np.testing.assert_array_equal(row['scores_0'], arrays[0]['scores_0'])
            np.testing.assert_array_equal(row['pairs_0'], arrays[0]['pairs_0'])
        for route in run['completed_routes']:
            report = runtime.read_json(output/(route+'-evaluation.json'))
            self.assertFalse(report['test_opened'])
            self.assertEqual(report['threshold'], run['threshold'])
            model, payload = load_cooperative_checkpoint(output/('adapt-'+route+'.pt'), device='cpu')
            self.assertEqual(payload['completed_updates'], 1)
            self.assertFalse(model.training)
            self.assertFalse(payload['resume_supported'])

    def test_native_zip_cli_uses_full_population_and_sealed_test(self):
        fixture = native_fixture.TimedDatasetTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        fixture.config = dict(scope='cooperative-physical-v1',
            model=asdict(CooperativeConfig(length=4, max_observations=4, d_model=8,
                nhead=2, num_layers=1, dim_feedforward=16, head_hidden=8, dropout=0.)),
            data={'time_unit_seconds': 10.}, train_splits=['T_core'], training=synthetic_configuration())
        args = ['fit', '--device', 'cpu', *fixture.cli_args('candidate')]
        self.assertEqual(self.invoke(args), 0)
        output = fixture.root/'candidate'
        evidence = runtime.read_json(output/'execution.json')
        self.assertFalse(evidence['synthetic'])
        self.assertFalse(evidence['preparation']['test_opened'])
        report = runtime.read_json(output/'utility-evaluation.json')
        self.assertEqual(len(report['per_scene']), 2)
        # The empty native scene keeps a reference target in all denominators.
        empty = report['per_scene'][1]['methods']['predicted']['full_target_metrics']['all']['position']
        self.assertGreater(empty['available_target_count'], 0)
        self.assertEqual(empty['intersection_count'], 0)

    def test_synthetic_checks_cannot_silently_open_native_data(self):
        args = ['check', '--device', 'cpu', '--output', str(self.root/'bad'), '--archive', 'secret.zip']
        self.assertEqual(self.invoke(args), 1)
        self.assertFalse((self.root/'bad').exists())

    def test_missing_native_config_and_unavailable_cuda_fail_without_fallback(self):
        self.assertEqual(self.invoke(['fit', '--device', 'cpu', '--output', str(self.root/'fit')]), 1)
        with patch('torch.cuda.is_available', return_value=False):
            self.assertEqual(self.invoke(['check', '--device', 'cuda', '--output', str(self.root/'cuda')]), 1)
        self.assertFalse((self.root/'cuda').exists())

    def test_bad_native_budget_is_rejected_before_archive_access(self):
        config = dict(scope='cooperative-physical-v1', model=asdict(CooperativeConfig()),
                      data={'time_unit_seconds': 20.}, train_splits=['T_core'],
                      training=synthetic_configuration())
        config['training']['max_seconds'] = -1
        path = self.root/'bad.json'
        runtime.write_json(path, config)
        with patch('association.timed_dataset.load_timed_dataset', side_effect=AssertionError('archive must stay closed')):
            self.assertEqual(self.invoke(['fit', '--device', 'cpu', '--output', str(self.root/'bad-fit'),
                '--archive', 'unused.zip', '--manifest', 'unused.json', '--alignment', 'unused.json',
                '--config', str(path)]), 1)


if __name__ == '__main__':
    unittest.main()
