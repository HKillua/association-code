"""V2 public CLI: real control flow over synthetic native files, no real MTAD."""
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import asdict
import io
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


class CooperativeRevisionCLITests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.threads = torch.get_num_threads()
        self.addCleanup(torch.set_num_threads, self.threads)

    def invoke(self, args):
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            return main(args)

    def configuration(self):
        return dict(scope='cooperative-physical-v2',
            model=asdict(CooperativeConfig(length=4, max_observations=4, d_model=8,
                nhead=2, num_layers=1, dim_feedforward=16, head_hidden=8, dropout=0.,
                model_variant='support_residual_v2', support_time_policy='train_absmax_v1')),
            data={'time_unit_seconds': 10., 'reference_longitude_policy': 'shortest_arc'}, train_splits=['T_core'],
            training=synthetic_configuration('support_residual_v2'))

    def test_public_synthetic_v2_runs_and_reloads_all_arms_with_one_identity(self):
        output = self.root/'check'
        args = ['check', '--device', 'cpu', '--output', str(output), '--cpu-threads', '1',
                '--model-variant', 'support_residual_v2', '--save-predictions']
        self.assertEqual(self.invoke(args), 0)
        run = runtime.read_json(output/'run.json')
        self.assertEqual(run['completed_updates'], 6)
        self.assertEqual(run['completed_routes'], ['plain', 'scalar-u', 'attribute-u', 'learned-attribute'])
        self.assertEqual(run['identity_selection_phase'], 'teacher_endpoint_before_utility')
        self.assertEqual(run['training']['longitude_loss_mode'], 'shortest_arc')
        first = None
        for route in run['completed_routes']:
            model, payload = load_cooperative_checkpoint(output/('adapt-'+route+'.pt'), device='cpu')
            self.assertEqual(model.config.model_variant, 'support_residual_v2')
            self.assertEqual(payload['format'], 'cooperative-physical-v2')
            self.assertEqual(payload['completed_updates'], 1)
            self.assertEqual(payload['training']['longitude_loss_mode'], 'shortest_arc')
            with np.load(output/(route+'-scores.npz')) as arrays:
                scores = arrays['scores_0'].copy()
                if first is None:
                    first = scores
                np.testing.assert_array_equal(scores, first)
            report = runtime.read_json(output/(route+'-evaluation.json'))
            self.assertEqual(report['threshold'], run['threshold'])
            self.assertFalse(report['test_opened'])
            self.assertTrue((output/(route+'-predictions/index.json')).is_file())
        prior = (output/'run.json').read_bytes()
        self.assertEqual(self.invoke(args), 1)
        self.assertEqual((output/'run.json').read_bytes(), prior)

    def test_v2_native_zip_keeps_reference_only_scene_in_population(self):
        fixture = native_fixture.TimedDatasetTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        fixture.config = self.configuration()
        self.assertEqual(self.invoke(['fit', '--device', 'cpu', '--cpu-threads', '1',
                         '--save-predictions', *fixture.cli_args('v2')]), 0)
        output = fixture.root/'v2'
        run = runtime.read_json(output/'run.json')
        self.assertEqual(run['completed_updates'], 6)
        preparation = runtime.read_json(output/'execution.json')['preparation']
        self.assertFalse(preparation['test_opened'])
        self.assertEqual(preparation['reference_longitude_policy'], 'shortest_arc')
        self.assertTrue(all(scene['reference_alignment']['longitude_policy'] == 'shortest_arc'
                            for population in run['scene_contract'] for scene in population))
        for route in run['completed_routes']:
            report = runtime.read_json(output/(route+'-evaluation.json'))
            self.assertEqual(len(report['per_scene']), 2)
            empty = report['per_scene'][1]['methods']['predicted']['full_target_metrics']['all']['position']
            self.assertGreater(empty['available_target_count'], 0)
            self.assertEqual(empty['intersection_count'], 0)
            entries = runtime.read_json(output/(route+'-predictions/index.json'))['scenes']
            with np.load(output/(route+'-predictions')/entries[1]['npz']) as arrays:
                self.assertGreater(arrays['m000__aligned_truth_valid'].sum(), 0)
                self.assertEqual(arrays['m000__aligned_prediction_valid'].sum(), 0)

    def test_no_budget_or_inconsistent_variant_fails_before_opening_archive(self):
        cases = []
        value = self.configuration(); value['training']['max_seconds'] = None; cases.append(value)
        value = self.configuration(); value['scope'] = 'cooperative-physical-v1'; cases.append(value)
        value = self.configuration(); value['training']['routes'] = ['plain', 'utility']; cases.append(value)
        for index, config in enumerate(cases):
            with self.subTest(index=index):
                path = self.root/f'config{index}.json'
                runtime.write_json(path, config)
                output = self.root/f'blocked{index}'
                with patch('association.timed_dataset.load_timed_dataset', side_effect=AssertionError('archive must stay closed')) as loader:
                    self.assertEqual(self.invoke(['fit', '--device', 'cpu', '--cpu-threads', '1',
                        '--archive', 'not-opened.zip', '--manifest', 'not-opened.json',
                        '--alignment', 'not-opened.json', '--config', str(path), '--output', str(output)]), 1)
                    loader.assert_not_called()
                self.assertFalse(output.exists())

    def test_v2_synthetic_check_does_not_fallback_when_cuda_is_unavailable(self):
        with patch('torch.cuda.is_available', return_value=False):
            self.assertEqual(self.invoke(['check', '--device', 'cuda:0', '--output', str(self.root/'gpu'),
                    '--model-variant', 'support_residual_v2', '--cpu-threads', '1']), 1)
        self.assertFalse((self.root/'gpu').exists())


if __name__ == '__main__':
    unittest.main()
