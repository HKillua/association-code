"""Checkpoint reuse must preserve native time semantics and cumulative costs."""
from pathlib import Path
import tempfile
import unittest

import torch

from association.complete_training import STAGES, load_complete_snapshot, train_complete
from association.complete_evaluation import evaluate_complete
from association.fusion import PairFusionConfig
from association.models import ModelConfig
from association.reference import score_reference
from association.timed_models import TimedAssociationModel
from tests.test_timed_pipeline import timed_scene


class CompleteSnapshotTests(unittest.TestCase):
    def setUp(self):
        self.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, self.old_threads)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.train = [timed_scene('train', 'T_core')]
        self.validation = [timed_scene('validation', 'V_select')]
        for scene in [*self.train, *self.validation]:
            scene['reference_alignment'] = dict(time_offset_seconds=100., max_gap_seconds=20.)
        self.model = ModelConfig(length=4, d_model=8, nhead=2, num_layers=1,
                                 dim_feedforward=16, dropout=.1)
        self.fusion = PairFusionConfig(length=4, d_model=8, nhead=2, num_layers=1,
                                       dim_feedforward=16, dropout=.1, fc_hidden=8)
        self.config = dict(updates=dict.fromkeys(STAGES, 1), learning_rates=dict.fromkeys(STAGES, .001),
            seed=13, branch='ce', batch_tracks=3, pair_batch_size=4, pair_microbatch=2,
            track_microbatch=2, completion_weight=.1, gradient_clip=1., max_hours=.1,
            validate_every_updates=1, completion_cache_mib=1,
            position_loss_mode='standard_smooth_l1', lr_schedule='constant',
            singleton_policy='observed', validation_batches=dict(tracks=2, pairs=3))

    def fit(self, name, **kwargs):
        return train_complete(self.train, self.validation, self.model, self.fusion,
                              self.config, self.root / name, **kwargs)

    def test_selected_snapshot_restores_time_adapter_and_reproduces_validation(self):
        self.fit('full')
        path = self.root / 'full' / 'checkpoint.pt'
        saved = torch.load(path, weights_only=True)
        loaded = load_complete_snapshot(path)
        self.assertIsInstance(loaded['model'], TimedAssociationModel)
        self.assertEqual(loaded['model'].max_observations, 3)
        for module, expected in zip((loaded['model'], loaded['fusion'], loaded['objective']),
                                    saved['selected']['states']):
            self.assertFalse(module.training)
            for key, value in module.state_dict().items():
                torch.testing.assert_close(value, expected[key], rtol=0, atol=0)
        result = evaluate_complete(loaded['model'], loaded['fusion'], loaded['normalizer'],
            self.validation, device='cpu', batch_tracks=2, pair_batch_size=3,
            select_on_validation=True, threshold=None, singleton_policy='observed')
        self.assertEqual(result['threshold'], saved['selected']['threshold'])
        self.assertEqual(result['selection']['macro_f1'], saved['selected']['macro_f1'])
        baseline = result['records'][0]['scores']
        for frozen in (True, False):
            loaded['model'].set_completion_trainable(not frozen)
            flags = [p.requires_grad for p in loaded['model'].parameters()]
            scored = score_reference(loaded['model'], loaded['normalizer'], self.validation, 'cpu', 2, 3)
            self.assertEqual(baseline.tolist(), scored[0]['scores'].tolist())
            self.assertEqual(flags, [p.requires_grad for p in loaded['model'].parameters()])
        self.validation[0]['input_adapter'] = 'registered_v1'
        flags = [p.requires_grad for p in loaded['model'].parameters()]
        with self.assertRaisesRegex(ValueError, 'match the model input adapter'):
            evaluate_complete(loaded['model'], loaded['fusion'], loaded['normalizer'],
                self.validation, device='cpu', batch_tracks=2, pair_batch_size=3,
                select_on_validation=True, threshold=None, singleton_policy='observed')
        self.assertEqual(flags, [p.requires_grad for p in loaded['model'].parameters()])

    def test_native_resume_replays_updates_and_retains_preparation_cost(self):
        self.fit('full')
        first = self.fit('part', max_updates_this_call=2, initial_elapsed_seconds=10.)
        result = self.fit('resumed', resume_from=self.root / 'part', initial_elapsed_seconds=5.)
        expected = torch.load(self.root / 'full' / 'checkpoint.pt', weights_only=True)
        actual = torch.load(self.root / 'resumed' / 'checkpoint.pt', weights_only=True)
        for left, right in zip(expected['states'], actual['states']):
            for key in left:
                torch.testing.assert_close(left[key], right[key], rtol=0, atol=0)
        self.assertEqual([r['loss'] for r in expected['history']], [r['loss'] for r in actual['history']])
        self.assertGreaterEqual(result['elapsed_seconds'], first['elapsed_seconds'] + 5.)
        self.train[0]['reference_alignment']['max_gap_seconds'] = 21.
        with self.assertRaisesRegex(ValueError, 'resume configuration'):
            self.fit('changed', resume_from=self.root / 'part')
        self.assertFalse((self.root / 'changed').exists())

    def test_preparation_budget_exhaustion_prevents_new_run(self):
        with self.assertRaises(TimeoutError):
            self.fit('exhausted', initial_elapsed_seconds=360.)
        self.assertFalse((self.root / 'exhausted').exists())


if __name__ == '__main__':
    unittest.main()
