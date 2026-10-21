"""Explicit physical output conditioning and historical snapshot compatibility."""
import copy
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch.nn import functional as F

from association.utils import runtime
from association.training.tmtaff import STAGES, load_complete_snapshot, train_complete
from association.models.fusion import PairFusion, PairFusionConfig
from association.models.tmtaff import ModelConfig
from tests.fixtures.scenes import timed_scene


class FusionOutputConditioningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def setUp(self):
        self.raw_config = PairFusionConfig(fc_hidden=8, length=4, d_model=8, nhead=2,
            num_layers=1, dim_feedforward=16, dropout=.1)
        self.affine_config = replace(self.raw_config, output_parameterization='train_position_affine')
        self.normalizer = dict(mean=[20., 131., 5., 180.], std=[.5, .6, 2., 90.])
        torch.manual_seed(71)
        self.left, self.right = torch.randn(2, 4, 8), torch.randn(2, 4, 8)

    def test_only_position_maps_to_physical_units_speed_and_heading_stay_raw(self):
        model = PairFusion(self.affine_config, self.normalizer).eval()
        with torch.no_grad():
            model.state_projection.weight.zero_()
            model.state_projection.bias.zero_()
            zero = model(self.left, self.right)['states']
            physical = torch.tensor([20., 131., F.softplus(torch.tensor(0.)).item(), 0.])
            torch.testing.assert_close(zero, physical.expand_as(zero), rtol=0, atol=0)
            model.state_projection.bias.copy_(torch.tensor([1., -2., .5, 3.]))
            actual = model(self.left, self.right)['states']
            expected = torch.tensor([20.5, 129.8, .5, 3.])
            expected[2] = F.softplus(expected[2])
            torch.testing.assert_close(actual, expected.expand_as(actual), rtol=0, atol=0)
            # No new heading wrap or geographic clamp is part of this adaptation.
            model.state_projection.bias[0] = 200.
            self.assertTrue((model(self.left, self.right)['states'][..., 0] > 90.).all())


class FusionOutputSnapshotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.train = [timed_scene('training', 'T_core')]
        self.validation = [timed_scene('validation', 'V_select')]
        self.model = ModelConfig(length=4, d_model=8, nhead=2, num_layers=1, dim_feedforward=16, dropout=.1)
        self.fusion = PairFusionConfig(length=4, d_model=8, nhead=2, num_layers=1,
            dim_feedforward=16, dropout=.1, fc_hidden=8, output_parameterization='train_position_affine')
        self.config = dict(updates=dict.fromkeys(STAGES, 2), learning_rates=dict.fromkeys(STAGES, .001),
            seed=13, branch='ce', batch_tracks=3, pair_batch_size=4, pair_microbatch=2,
            track_microbatch=2, completion_weight=.1, gradient_clip=1., max_hours=.1,
            validate_every_updates=1, completion_cache_mib=1,
            position_loss_mode='standard_smooth_l1', lr_schedule='constant',
            singleton_policy='observed', validation_batches=dict(tracks=2, pairs=3))

    def fit(self, name, **kwargs):
        return train_complete(self.train, self.validation, self.model, self.fusion,
                              self.config, self.root / name, **kwargs)

    def checkpoint(self, name):
        return torch.load(self.root / name / 'checkpoint.pt', weights_only=True)

    def assert_states_equal(self, left, right):
        for expected, actual in zip(left, right):
            self.assertEqual(set(expected), set(actual))
            for name, value in expected.items():
                torch.testing.assert_close(actual[name], value, rtol=0, atol=0)

    def test_affine_native_four_stages_selected_last_and_exact_resume(self):
        result = self.fit('full')
        self.assertEqual(result['status'], 'COMPLETED')
        self.assertEqual(result['fusion_output_parameterization'], 'train_position_affine')
        self.assertIn('output-scale-adapted', result['scope'])
        full = self.checkpoint('full')
        self.assertEqual(full['normalizer']['training_splits'], ['T_core'])
        self.assertEqual([row['stage'] for row in full['history']], [stage for stage in STAGES for _ in range(2)])
        self.assertTrue(all(np.isfinite(row['loss']) for row in full['history']))
        for selected in (True, False):
            loaded = load_complete_snapshot(self.root / 'full' / 'checkpoint.pt', selected=selected)
            expected = full['selected']['states'] if selected else full['states']
            self.assert_states_equal(expected, [loaded[key].state_dict() for key in ('model', 'fusion', 'objective')])
            torch.testing.assert_close(loaded['fusion'].position_mean,
                torch.tensor(full['normalizer']['mean'][:2], dtype=torch.float32), rtol=0, atol=0)
            torch.testing.assert_close(loaded['fusion'].position_std,
                torch.tensor(full['normalizer']['std'][:2], dtype=torch.float32), rtol=0, atol=0)
        self.assertEqual(self.fit('part', max_updates_this_call=5)['status'], 'INTERRUPTED')
        self.assertEqual(self.fit('resumed', resume_from=self.root/'part')['status'], 'COMPLETED')
        resumed = self.checkpoint('resumed')
        self.assert_states_equal(full['states'], resumed['states'])
        self.assert_states_equal(full['selected']['states'], resumed['selected']['states'])
        self.assertEqual([row['loss'] for row in full['history']], [row['loss'] for row in resumed['history']])
        self.assertEqual(runtime.read_json(self.root/'full'/'summary.json')['evaluation']['fusion'],
                         runtime.read_json(self.root/'resumed'/'summary.json')['evaluation']['fusion'])


    def test_snapshot_statistics_conflicts_fail_for_selected_last_and_resume(self):
        self.fit('part', max_updates_this_call=5)
        original = self.checkpoint('part')
        for target in ('normalizer', 'last', 'selected'):
            with self.subTest(target=target):
                saved = copy.deepcopy(original)
                if target == 'normalizer':
                    saved['normalizer']['mean'][0] += 1.
                elif target == 'last':
                    saved['states'][1]['position_mean'][0] += 1.
                else:
                    saved['selected']['states'][1]['position_std'][0] += 1.
                path = self.root/f'{target}.pt'
                torch.save(saved, path)
                selected = target != 'last'
                with self.assertRaisesRegex(ValueError, 'disagrees with its normalizer'):
                    load_complete_snapshot(path, selected=selected)
                if target != 'normalizer':
                    torch.save(saved, self.root/'part'/'checkpoint.pt')
                    with (patch.object(torch.optim.Adam, 'step', side_effect=AssertionError('must fail before optimizer update')),
                          self.assertRaisesRegex(ValueError, 'disagrees with its normalizer')):
                        self.fit(f'bad-{target}-resume', resume_from=self.root/'part')
        # Resume's input-moment semantic check rejects changed checkpoint metadata
        # before a new run directory exists.
        changed = copy.deepcopy(original)
        changed['normalizer']['mean'][0] += 1.
        torch.save(changed, self.root/'part'/'checkpoint.pt')
        with self.assertRaisesRegex(ValueError, 'resume configuration'):
            self.fit('bad-resume', resume_from=self.root/'part')
        self.assertFalse((self.root/'bad-resume').exists())


if __name__ == '__main__':
    unittest.main()
