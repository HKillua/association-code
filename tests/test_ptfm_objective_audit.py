"""085 objective parity, cache population and read-only source qualification."""
from argparse import Namespace
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
import json
import unittest
from unittest.mock import patch

import numpy as np
import torch

from association.evaluation.ptfm_objective_audit import group_objectives, summarize_objectives, bind_source, audit_training_log
from association.models.ptfm import PTFM, objectives
from association.utils.runtime import read_json, write_json
from tests.test_ptfm import normalizer
from tests.test_ptfm_residual import residual_config
from tests.test_ptfm_speed_diagnostics import completed_source
from tests.evaluation.test_support_gate import support_gate_cache_fixture
from tests.training.test_support_gate import tiny_train
from tools.experiments import ptfm_residual_fit
from tools.analysis import ptfm_objective_audit as entry


def completed_pair(root, sparse=False):
    direct, frozen = completed_source(root, sparse=sparse)
    fixture = root / 'raw-fixture'; fixture.mkdir()
    _, metadata, validation = support_gate_cache_fixture(fixture, sparse=sparse)
    cfg = residual_config(); write_json(frozen / 'configs/ptfm-residual-level1.json', cfg)
    write_json(root / 'residual-config.json', cfg)
    original = read_json(direct / 'run.json')
    args = Namespace(source_system=original['source_system'], output=str(root / 'source076b'),
        paired_control=original['paired_control'], paired_ptfm=str(direct), archive=['not-opened.zip'],
        manifest=str(root / 'manifest.json'), alignment=str(root / 'alignment.json'),
        config=str(root / 'residual-config.json'), device='cpu', cpu_threads=4, time_policy='advisory')
    with patch.object(ptfm_residual_fit, 'ROOT', frozen), patch.object(ptfm_residual_fit, 'production_contract'), \
            patch.object(ptfm_residual_fit, '_bind_source'), patch.object(ptfm_residual_fit, 'bind_paired_control'), \
            patch.object(ptfm_residual_fit, '_load_metadata', return_value=metadata), \
            patch.object(ptfm_residual_fit, 'load_timed_dataset',
                return_value=dict(train=tiny_train(), validation=validation, report={'fixture': True})), \
            patch('association.evaluation.ptfm.frozen_motion', side_effect=lambda raw, *_: raw):
        ptfm_residual_fit.run(args)
    return direct, root / 'source076b', frozen


class PTFMObjectiveAuditTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_numpy_objective_matches_original_masks_poles_angles_and_zero_labels(self):
        norm = normalizer(); model = PTFM(norm)
        states = np.array([[[89.9, 179.9, 7., 359.], [-89.9, -179.9, 5., 1.], [0., 0., 1e6, 180.]],
                           [[1., 2., 0., 0.], [3., 4., 0., 0.], [5., 6., 0., 0.]]])
        truth = states.copy(); truth[0, 0, [1, 3]] = [-179.9, 1.]
        truth[0, 1, [1, 3]] = [179.9, 359.]
        truth[0, 2, 2] = 1.
        valid = np.ones(states.shape, bool); valid[0, 1, :2] = False; valid[1] = False
        truth[~valid] = np.nan
        weights = np.array([[[.2, .3, .5], [.6, .1, .3], [0., 1., 0.]], [[1., 0., 0.]] * 3])
        actual = group_objectives(states, truth, valid, weights, norm['scale'])
        for group in range(2):
            value = dict(states=torch.tensor(states[group]), normalized_states=torch.zeros((3, 4), dtype=torch.float64),
                         task_weights=torch.tensor(weights[group]))
            loss, diag = objectives(model, value, torch.tensor(truth[group]), torch.tensor(valid[group]))
            self.assertAlmostEqual(actual['total'][group], float(loss), delta=1e-6)
            for i in range(3):
                if actual['count'][group, i]:
                    self.assertAlmostEqual(actual['loss'][group, i], diag['task_losses'][i], delta=1e-6)
                    self.assertAlmostEqual(actual['sigma'][group, i], diag['task_sigmas'][i], places=12)
        self.assertEqual(actual['total'][1], 0)
        self.assertTrue(np.isnan(actual['coefficient'][1]).all())
        actual['selected'] = np.array([True, False])
        summary = summarize_objectives([actual])
        self.assertEqual(summary['all_output_groups']['unsupervised_group_count'], 1)
        self.assertEqual(summary['unique_selected_groups']['group_count'], 1)
        self.assertAlmostEqual(summary['unique_selected_groups']['tasks']['heading']['loss_point_mean'], 4 / 3 / 30)

    def test_invalid_weight_shape_value_and_probability_sum_rejected(self):
        state = np.zeros((2, 3, 4)); valid = np.ones_like(state, bool)
        good = np.full((2, 3, 3), 1/3)
        for weights in (good[:, :, :2], good * 2, np.full_like(good, np.nan), good - 1):
            with self.assertRaisesRegex(ValueError, 'weights'):
                group_objectives(state, state, valid, weights, np.ones(4))
        with self.assertRaisesRegex(ValueError, 'scale'):
            group_objectives(state, state, valid, good, [1., 0., 1., 1.])

    def test_complete_real_sources_no_forward_and_duplicate_semantics(self):
        with TemporaryDirectory() as directory:
            root = Path(directory); direct, residual, frozen = completed_pair(root)
            watched = {p: p.read_bytes() for source in (direct, residual) for p in source.rglob('*') if p.is_file()}
            args = Namespace(direct=str(direct), residual=str(residual), output=str(root / 'audit'), cpu_threads=4)
            with patch.object(entry, 'ROOT', frozen), patch('torch.load', side_effect=AssertionError('checkpoint forbidden')), \
                    patch('association.models.ptfm.PTFM.forward', side_effect=AssertionError('forward forbidden')):
                result = entry.run(args)
            self.assertTrue(result['source_physical_population_replayed'])
            self.assertTrue(result['actual_training_batches_paired'])
            self.assertEqual(result['scene_count'], 1)
            for data in result['methods'].values():
                all_groups = data['endpoints']['final']['all_output_groups']
                unique = data['endpoints']['final']['unique_selected_groups']
                self.assertEqual(all_groups['group_count'] - unique['group_count'], 1)
                self.assertGreater(all_groups['tasks']['position']['valid_point_count'], unique['tasks']['position']['valid_point_count'])
                self.assertEqual(unique['tasks']['position']['valid_point_count'], 127)
                self.assertEqual(data['training_log']['fixed_100_step_windows'][0]['update_count'], 3)
            self.assertEqual(read_json(root / 'audit/run.json')['status'], 'COMPLETED')
            self.assertIn('UNKNOWN', (root / 'audit/report.md').read_text())
            for path, content in watched.items(): self.assertEqual(path.read_bytes(), content)
            with self.assertRaises(FileExistsError): entry.run(args)
            args.output = str(direct / 'wrong')
            with self.assertRaisesRegex(ValueError, 'preserve'): entry.run(args)

    def test_sparse_truth_and_source_physical_rejection_preserves_failure(self):
        with TemporaryDirectory() as directory:
            root = Path(directory); direct, residual, frozen = completed_pair(root, sparse=True)
            args = Namespace(direct=str(direct), residual=str(residual), output=str(root / 'ok'), cpu_threads=4)
            with patch.object(entry, 'ROOT', frozen): result = entry.run(args)
            self.assertEqual(result['methods']['direct074C']['endpoints']['final']['unique_selected_groups']['tasks']['position']['valid_point_count'], 1)
            path = residual / 'final/predictions/states-3035.npz'
            with np.load(path) as data: payload = {k: data[k].copy() for k in data.files}
            mask = np.isfinite(payload['ptfm_adapted_position_error'])
            payload['ptfm_adapted_position_error'][mask] += 1.
            np.savez_compressed(path, **payload)
            args.output = str(root / 'bad')
            with patch.object(entry, 'ROOT', frozen), self.assertRaisesRegex(ValueError, 'error'):
                entry.run(args)
            self.assertEqual(read_json(root / 'bad/run.json')['status'], 'FAILED')

    def test_source_updates_training_pairing_and_invalid_cached_weights_rejected(self):
        with TemporaryDirectory() as directory:
            root = Path(directory); direct, residual, frozen = completed_pair(root)
            cfg = read_json(frozen / 'configs/ptfm-adapted-level1.json')
            manifest = read_json(frozen / 'configs/data/mtad-level1-scene3000.json')
            path = direct / 'fit/run.json'; original = path.read_bytes(); changed = json.loads(original)
            changed['optimizer_updates'] += 1; write_json(path, changed)
            with self.assertRaisesRegex(ValueError, 'updates'): bind_source(direct, cfg, manifest)
            path.write_bytes(original)
            bound = bind_source(direct, cfg, manifest)
            logs = direct / 'fit/training.jsonl'; original = logs.read_bytes()
            rows = [json.loads(row) for row in original.splitlines()]
            rows[0]['batch'][0]['task_valid_counts'][0] += 1
            logs.write_text('\n'.join(json.dumps(row) for row in rows) + '\n')
            with self.assertRaisesRegex(ValueError, 'exposure'): audit_training_log(bound)
            logs.write_bytes(original)
            path = direct / 'initial/predictions/states-3035.npz'
            with np.load(path) as data: payload = {k: data[k].copy() for k in data.files}
            payload['group_task_weights'][0, 0] *= 2
            np.savez_compressed(path, **payload)
            args = Namespace(direct=str(direct), residual=str(residual), output=str(root / 'bad-weights'), cpu_threads=4)
            with patch.object(entry, 'ROOT', frozen), self.assertRaisesRegex(ValueError, 'weights'):
                entry.run(args)
            self.assertEqual(read_json(root / 'bad-weights/run.json')['status'], 'FAILED')


if __name__ == '__main__':
    unittest.main()
