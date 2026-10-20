"""Saved self-completion comparisons retain masks, empty scenes and all nodes."""
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import asdict
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from association.models.cooperative import CooperativeConfig
from association.utils import runtime
from tests.fixtures import native_zip
from tools.analysis.compare_completion_predictions import compare, main, REGIONS
from tools.experiments.cooperative_fit import main as fit, synthetic_configuration


class CompletionComparisonTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def write(self, name, *, corrected=False):
        directory = self.root/name
        prediction_dir = directory/'completion-predictions'
        prediction_dir.mkdir(parents=True)
        contracts, summaries, index = [], [], []
        for sid, n in (('observed', 3), ('empty', 0)):
            nodes = np.asarray(['n0', 'n1', 'n2'][:n], dtype=str)
            identities = np.asarray(['A', 'A', 'B'][:n], dtype=str)
            sources = np.asarray(['9001', '9002', '9002'][:n], dtype=str)
            grid = np.arange(3, dtype=float)
            truth = np.full((n, 3, 4), np.nan)
            valid = np.zeros((n, 3, 4), bool)
            prediction = np.broadcast_to([0., 0., 1., 1.], (n, 3, 4)).astype(np.float32).copy()
            if n:
                truth[0, 0, :2], valid[0, 0, :2] = [0., 0.], True
                truth[0, 1, 2], valid[0, 1, 2] = 3., True
                truth[1, 2, 3], valid[1, 2, 3] = 359., True
                prediction[0, 0, 0] = 0. if corrected else .001
                if corrected:
                    prediction[0, 1, 2], prediction[1, 2, 3] = 3., 359.
            masks = {key: np.zeros((n, 3), bool) for key in REGIONS}
            masks['all'][:] = True
            masks['before_support'][:, 0] = True
            masks['inside_support'][:, 1] = True
            masks['exact_observation'][:, 1] = True
            masks['after_support'][:, 2] = True
            base = np.broadcast_to([0., 0., np.nan, np.nan], (n, 3, 4)).copy()
            name_npz = sid+'.npz'
            np.savez_compressed(prediction_dir/name_npz, scene_id=np.asarray(sid), node_ids=nodes,
                identities=identities, sources=sources, query_seconds=grid, prediction=prediction,
                target=truth, target_valid=valid, position_base=base,
                nearest_seconds=np.broadcast_to([100., 0., 1900.], (n, 3)),
                **{'region_'+key: value for key, value in masks.items()})
            contracts.append(dict(scene_id=sid, split='V_select', node_ids=nodes.tolist(),
                identities=identities.tolist(), sources=sources.tolist(), grid=grid.tolist()))
            summaries.append(dict(scene_id=sid, fragment_count=n,
                reference_targets_without_input=[] if n else ['reference-only']))
            index.append(dict(scene_id=sid, path=name_npz, fragment_count=n))
        runtime.write_json(prediction_dir/'index.json', index)
        runtime.write_json(directory/'run.json', dict(status='COMPLETED', phase='teacher', test_opened=False,
            completed_updates=1, training=dict(teacher_updates=1), model_config=dict(length=3),
            phases=dict(teacher=dict(status='COMPLETED', completed_updates=1),
                        completion_evaluation=dict(status='COMPLETED')),
            scene_contract=[[], contracts]))
        runtime.write_json(directory/'completion-evaluation.json', dict(test_opened=False,
            scene_count=2, fragment_count=3, completion_forward_count=3, per_scene=summaries))
        return directory

    def test_numerical_errors_use_attribute_masks_and_preserve_empty_populations(self):
        left, right = self.write('left'), self.write('right', corrected=True)
        output = self.root/'comparison.json'
        result = compare(left, right, output)
        self.assertEqual(result['scene_count'], 2)
        self.assertEqual(result['fragment_count'], 3)  # Duplicate A identities stay separate.
        self.assertEqual(result['query_count'], 9)
        overall = result['overall']
        a, b = (overall[side]['regions']['all'] for side in ('left', 'right'))
        for feature in ('position', 'speed', 'heading'):
            self.assertEqual(a[feature]['intersection_count'], 1)
            self.assertEqual(a[feature]['available_prediction_count'], 9)
            self.assertEqual(a[feature]['prediction_without_target_count'], 8)
            self.assertEqual(b[feature]['mae'], 0)
        self.assertAlmostEqual(a['position']['mae'], 6371000*np.deg2rad(np.float32(.001)), places=4)
        self.assertEqual(a['speed']['mae'], 2)
        self.assertEqual(a['heading']['mae'], 2)  # 1 versus 359 is circular.
        expected = float((torch.log(torch.tensor(1.)+1e-6)-torch.log(torch.tensor(3.)+1e-6)).square())
        self.assertEqual(a['raw_log_speed_mse'], dict(count=1, mse=expected))
        self.assertEqual(b['raw_log_speed_mse']['mse'], 0)
        self.assertEqual(a['speed_statistics']['prediction_all'], dict(count=9, mean=1., std=0.))
        self.assertEqual(a['speed_statistics']['target_evaluable'], dict(count=1, mean=3., std=0.))
        delta = overall['right_minus_left']['regions']['all']
        self.assertEqual(delta['speed']['mae'], -2)
        self.assertEqual(delta['raw_log_speed_mse']['mse'], -expected)
        self.assertEqual(result['by_source']['9002']['fragment_count'], 2)
        self.assertIsNone(result['by_source']['9002']['left']['regions']['all']['speed']['mae'])
        self.assertEqual(result['per_scene'][1]['reference_targets_without_input'], ['reference-only'])
        self.assertEqual(result['per_scene'][1]['fragment_count'], 0)
        self.assertIsNone(result['per_scene'][1]['left']['regions']['all']['position']['mae'])
        self.assertIsNone(overall['left']['regions']['before_support']['raw_log_speed_mse']['mse'])
        self.assertEqual(overall['left']['input_distance_metrics']['gt_1800s']['heading']['mae'], 2)
        self.assertIsNone(overall['left']['input_distance_metrics']['gt_300_le_600s']['speed']['mae'])
        self.assertEqual(json.loads(output.read_text())['overall'], overall)
        with self.assertRaises(FileExistsError):
            compare(left, right, output)

    def test_partial_index_incomplete_run_or_different_reference_inputs_are_rejected(self):
        left = self.write('left')
        for key in ('target', 'target_valid', 'nearest_seconds', 'node_ids'):
            with self.subTest(key=key):
                right = self.write('right-'+key)
                path = right/'completion-predictions/observed.npz'
                with np.load(path) as loaded:
                    arrays = {name: loaded[name] for name in loaded.files}
                if key == 'node_ids':
                    arrays[key][0] = 'bad'
                elif key == 'target_valid':
                    arrays[key][0, 1, 2] = False
                else:
                    arrays[key].flat[0] += 1
                np.savez_compressed(path, **arrays)
                with self.assertRaises(ValueError):
                    compare(left, right, self.root/(key+'.json'))
        right = self.write('partial')
        index = runtime.read_json(right/'completion-predictions/index.json')
        runtime.write_json(right/'completion-predictions/index.json', index[:1])
        with self.assertRaises(ValueError):
            compare(left, right, self.root/'partial.json')
        right = self.write('unfinished')
        run = runtime.read_json(right/'run.json'); run['status'] = 'RUNNING'
        runtime.write_json(right/'run.json', run)
        with self.assertRaises(ValueError):
            compare(left, right, self.root/'unfinished.json')

    def test_different_bases_report_each_learning_increment_on_identical_population(self):
        left, right = self.write('hold'), self.write('ols', corrected=True)
        run = runtime.read_json(right/'run.json')
        run['model_config']['position_base_policy'] = 'native_ols'
        runtime.write_json(right/'run.json', run)
        path = right/'completion-predictions/observed.npz'
        with np.load(path) as saved:
            arrays = {key: saved[key] for key in saved.files}
        arrays['position_base'][0, 0, 0] = .002
        np.savez_compressed(path, **arrays)
        result = compare(left, right, self.root/'different-bases.json')
        overall = result['overall']
        for side in ('left', 'right'):
            base = overall['position_base'][side]['regions']['all']
            learned = overall[side]['regions']['all']['position']
            delta = overall['position_model_minus_own_base'][side]['regions']['all']
            self.assertEqual(base['intersection_count'], learned['intersection_count'])
            self.assertEqual(delta['mae'], learned['mae']-base['mae'])
        self.assertGreater(overall['position_model_minus_own_base']['left']['regions']['all']['mae'], 0)
        self.assertLess(overall['position_model_minus_own_base']['right']['regions']['all']['mae'], 0)
        self.assertIsNone(result['per_scene'][1]['position_model_minus_own_base']['right']['regions']['all']['mae'])
        self.assertEqual(result['fragment_count'], 3)
        # Different bases are allowed; a changed target still invalidates this comparison.
        arrays['target'][0, 0, 0] = .001
        np.savez_compressed(path, **arrays)
        with self.assertRaisesRegex(ValueError, 'reference differs: target'):
            compare(left, right, self.root/'changed-target.json')

    def test_two_tiny_native_teacher_runs_compare_through_public_cli(self):
        threads = torch.get_num_threads()
        self.addCleanup(torch.set_num_threads, threads)
        fixture = native_zip.make_native_zip_fixture(self)
        common = CooperativeConfig(length=4, max_observations=4, d_model=8, nhead=2,
            num_layers=1, dim_feedforward=16, head_hidden=8, dropout=0.,
            model_variant='support_residual_v2', support_time_policy='train_absmax_v1')
        paths = []
        for policy in ('physical_softplus', 'scaled_log'):
            fixture.config = dict(scope='cooperative-physical-v2', model={**asdict(common), 'speed_output_policy': policy},
                data=dict(time_unit_seconds=10., reference_longitude_policy='shortest_arc'), train_splits=['T_core'],
                training=synthetic_configuration('support_residual_v2', phase='teacher'))
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                self.assertEqual(fit(['fit', '--phase', 'teacher', '--save-predictions', '--device', 'cpu',
                    '--cpu-threads', '1', *fixture.cli_args(policy)]), 0)
            paths.append(fixture.root/policy)
        output = self.root/'native-comparison.json'
        with redirect_stdout(io.StringIO()):
            self.assertEqual(main(['--left', str(paths[0]), '--right', str(paths[1]), '--output', str(output)]), 0)
        result = runtime.read_json(output)
        self.assertEqual(result['scene_count'], 2)
        self.assertEqual(result['fragment_count'], 4)
        self.assertEqual(result['query_count'], 16)
        self.assertEqual(result['inference_calls'], 0)
        self.assertEqual(result['optimizer_updates'], 0)
        self.assertEqual(result['model_config']['right']['speed_output_policy'], 'scaled_log')
        self.assertEqual(result['per_scene'][1]['fragment_count'], 0)
        self.assertFalse(result['test_opened'])
        self.assertTrue(np.isfinite(result['overall']['right']['regions']['all']['raw_log_speed_mse']['mse']))


if __name__ == '__main__':
    unittest.main()
