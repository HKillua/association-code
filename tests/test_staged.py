import copy
from pathlib import Path
import signal
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from association.dataset import load_split, make_synthetic_archive, prepare
from association.models import AssociationModel, ModelConfig
from association.reference import load_reference, normalize_scene, score_reference, train_reference
from association.runtime import BudgetExceeded, load_records, read_json
from association.staged import FrozenCompletionCache, train_staged_reference


class StagedTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config = read_json(Path(__file__).resolve().parents[1] / 'configs' / 'smoke.json')
        self.config['caps'] = dict.fromkeys(('T_core','T_head','V_select','T_test'), 1)
        ref = self.config['reference']
        for key in ('tcm_epochs','tam_epochs','warmup_epochs','joint_completion'):
            ref.pop(key)
        ref.update(schedule='staged_updates_v2', purpose='probe', tcm_updates=2, tam_updates=2,
                   joint_updates=1, validate_every_updates=1, completion_cache_mib=1,
                   batch_tracks=4, pair_batch_size=4)
        self.archive = make_synthetic_archive(self.root/'data.zip', n_scenes=12)
        self.manifest = prepare(self.archive, self.config, self.root/'prepared', synthetic=True)

    def run_fit(self, suffix='fit'):
        return train_staged_reference(self.archive, self.manifest, self.config, 'A0', 'ce',
                                      20, 'cpu', self.root/suffix)

    def test_probe_real_gradients_phase_freezes_and_no_validation_or_test(self):
        reads = []
        def reader(path, manifest, split, length):
            reads.append(split)
            self.assertEqual(split, 'T_core')
            return load_split(path, manifest, split, length)
        with patch('association.staged.load_split', side_effect=reader):
            summary = self.run_fit()
        self.assertEqual(reads, ['T_core'])
        self.assertFalse(summary['eligible_reference'])
        self.assertFalse((self.root/'fit'/'checkpoint.pt').exists())
        self.assertFalse((self.root/'fit'/'T_head').exists())
        progress = read_json(self.root/'fit'/'progress.json')
        self.assertEqual(progress['completed_updates'], 5)
        self.assertEqual(progress['status'], 'COMPLETED')
        stages = {s['phase']: s for s in progress['stages']}
        self.assertEqual(set(stages['TCM']['changed_modules']), {'tcm_encoder','tcm_decoder'})
        self.assertEqual(set(stages['TAM_FROZEN']['changed_modules']), {'tam_encoder','ce_classifier'})
        self.assertEqual(set(stages['JOINT']['changed_modules']),
                         {'input_embedding','tcm_encoder','tcm_decoder','tam_encoder','ce_classifier'})
        for row in progress['history']:
            self.assertTrue(np.isfinite(row['loss']))
            self.assertGreater(row['gradient_norm'], 0)
            if row['phase'] != 'TCM':
                self.assertEqual(row['positive_pairs'], 2)
                self.assertEqual(row['negative_pairs'], 2)
            if row['phase'] == 'JOINT':
                self.assertNotIn('completion_cache', row)
        with self.assertRaisesRegex(ValueError, 'probe'):
            load_reference(self.root/'fit', self.manifest, self.config, 'A0','ce',20,'cpu')

    def test_fit_full_validation_caches_reload_and_joint_starts_at_selected_tam(self):
        self.config['reference']['purpose'] = 'fit'
        validation_calls = []
        def scorer(model, normalizer, scenes, *args, **kwargs):
            validation_calls.append([s['scene_id'] for s in scenes])
            for scene in scenes:
                n = len(scene['values'])
                self.assertEqual(len(scene['pairs']), n*(n-1)//2)
            return score_reference(model, normalizer, scenes, *args, **kwargs)
        with patch('association.staged.score_reference', side_effect=scorer):
            summary = train_reference(self.archive,self.manifest,self.config,'A0','ce',20,'cpu',self.root/'fit')
        self.assertEqual(validation_calls, [self.manifest['splits']['V_select']]*3 +
                         [self.manifest['splits']['T_head'], self.manifest['splits']['V_select']])
        self.assertTrue(summary['eligible_reference'])
        self.assertFalse(summary['test_opened'])
        joint = summary['stages'][-1]
        self.assertEqual(joint['starts_from_selection']['stage'], 'TAM_FROZEN')
        model, normalizer, loaded = load_reference(self.root/'fit',self.manifest,self.config,'A0','ce',20,'cpu')
        records = load_records(self.root/'fit'/'V_select', summary['cache_bindings']['V_select'])
        scenes = load_split(self.archive,self.manifest,'V_select',self.config['model']['length'])
        for a,b in zip(records,score_reference(model,normalizer,scenes,'cpu',batch_tracks=4)):
            np.testing.assert_allclose(a['scores'],b['scores'],rtol=0,atol=0)
        self.assertEqual(loaded['selected_stage'], summary['selected_stage'])

    def test_budget_failure_and_non_overwrite(self):
        self.config['reference']['max_hours'] = 1e-12
        with self.assertRaises(BudgetExceeded):
            self.run_fit()
        self.assertEqual(read_json(self.root/'fit'/'run.json')['status'],'FAILED')
        self.assertFalse(read_json(self.root/'fit'/'progress.json')['eligible_reference'])
        with self.assertRaises(FileExistsError):
            self.run_fit()

    def test_sigterm_preserves_completed_progress_and_terminal_failure(self):
        from association.staged import _update
        calls = 0
        def update(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                signal.raise_signal(signal.SIGTERM)
            return _update(*args, **kwargs)
        previous = signal.getsignal(signal.SIGTERM)
        with patch('association.staged._update', side_effect=update):
            with self.assertRaisesRegex(InterruptedError, 'SIGTERM'):
                self.run_fit()
        self.assertEqual(signal.getsignal(signal.SIGTERM), previous)
        report = read_json(self.root/'fit'/'progress.json')
        self.assertEqual(report['completed_updates'], 1)
        self.assertEqual(report['status'],'FAILED')
        self.assertEqual(read_json(self.root/'fit'/'run.json')['status'],'FAILED')

    def test_progress_write_failure_keeps_original_error_and_terminal_run(self):
        from association.runtime import write_json
        def writer(path, value):
            if Path(path).name == 'progress.json' and value['status'] == 'FAILED':
                raise OSError('progress storage failure')
            return write_json(path, value)
        with patch('association.staged.load_split', side_effect=ValueError('original data error')):
            with patch('association.runtime.write_json', side_effect=writer):
                with self.assertRaisesRegex(ValueError, 'original data error'):
                    self.run_fit()
        terminal = read_json(self.root/'fit'/'run.json')
        self.assertEqual(terminal['status'],'FAILED')
        self.assertEqual(terminal['failure'],'original data error')
        self.assertIn('progress storage failure',terminal['evidence_write_errors'][0])

    def test_same_seed_probe_replays_pairs_and_losses(self):
        self.run_fit('a')
        self.run_fit('b')
        a,b = [read_json(self.root/name/'history.json') for name in ('a','b')]
        for x,y in zip(a,b):
            self.assertEqual(x['loss'],y['loss'])
            self.assertEqual(x.get('pair_indices'),y.get('pair_indices'))
            self.assertEqual(x.get('scene_ids'),y.get('scene_ids'))


class CompletionCacheTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(31)
        self.model = AssociationModel(ModelConfig(d_model=8,nhead=2,num_layers=1,
                                     dim_feedforward=16,length=6,dropout=0.1))
        self.model.set_completion_trainable(False,include_embedding=True)
        self.model.train()
        values = np.random.default_rng(4).normal(size=(3,6,4)).astype(np.float32)
        observed = np.zeros((3,6),dtype=bool)
        observed[:,1:5] = True
        self.normalized = [(values,observed)]
        self.keys = [(0,0),(0,1),(0,2)]

    def test_cached_completion_matches_uncached_tam_outputs_and_gradients(self):
        cache = FrozenCompletionCache(self.model,self.normalized,'cpu',4096,2)
        with torch.no_grad():
            direct = self.model.complete(torch.as_tensor(self.normalized[0][0]),
                                         torch.as_tensor(self.normalized[0][1]))
        cached = cache.get(self.keys)
        torch.testing.assert_close(cached,direct,rtol=1e-6,atol=1e-6)
        torch.testing.assert_close(cache.get(self.keys),cached,rtol=0,atol=0)
        gradients, outputs = [], []
        for values in (direct,cached):
            self.model.zero_grad(set_to_none=True)
            torch.manual_seed(99)
            output = self.model.tam_encoder(values)
            output.square().sum().backward()
            gradients.append([p.grad.clone() for p in self.model.tam_encoder.parameters()])
            outputs.append(output.detach())
        torch.testing.assert_close(outputs[0],outputs[1],rtol=1e-6,atol=1e-6)
        for a,b in zip(*gradients):
            torch.testing.assert_close(a,b,rtol=1e-5,atol=1e-5)
        self.assertEqual(cache.hits,3)
        self.assertEqual(cache.misses,3)

    def test_capacity_eviction_and_no_reuse_after_unfreeze_or_weight_change(self):
        track_bytes = 6*8*4
        cache = FrozenCompletionCache(self.model,self.normalized,'cpu',track_bytes,2)
        cache.get(self.keys)
        self.assertEqual(cache.bytes,track_bytes)
        self.assertEqual(len(cache.entries),1)
        self.assertEqual(cache.evictions,2)
        self.model.set_completion_trainable(True,include_embedding=True)
        with self.assertRaisesRegex(ValueError,'frozen'):
            cache.get(self.keys)
        self.model.set_completion_trainable(False,include_embedding=True)
        with torch.no_grad():
            self.model.input_embedding.weight.add_(1)
        with self.assertRaisesRegex(ValueError,'stale'):
            cache.get(self.keys)
        cache.clear()
        self.assertEqual(cache.bytes,0)

    def test_zero_cache_capacity_still_runs_and_does_not_store_tensors(self):
        cache = FrozenCompletionCache(self.model,self.normalized,'cpu',0,2)
        a = cache.get(self.keys)
        b = cache.get(self.keys)
        self.assertEqual(cache.bytes,0)
        self.assertEqual(cache.hits,0)
        self.assertEqual(cache.misses,6)
        torch.testing.assert_close(a,b,rtol=0,atol=0)


if __name__ == '__main__':
    unittest.main()
