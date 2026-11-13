"""082 paired crop planning, full training inputs and fixed-final system reuse."""
from argparse import Namespace
from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import torch

from association.utils.runtime import read_json, write_json, load_records
from baselines.tsactfer_data import build_positive_pools, sample_batch
from baselines.tsactfer_level1 import (FULL_SCOPE, configuration, train_native,
    final_model, full_native_batch, load_checkpoint, verify_exposure)
from tests.integration.test_tsactfer_level1 import (tiny_config, training_fixture, validation_fixture, source_scores)
from tools.experiments import tsactfer_level1_fit as shared
from tools.experiments import tsactfer_full_view_fit as entry


def full_config(base):
    config = deepcopy(base)
    config['scope'] = FULL_SCOPE
    config['data']['training_view'] = 'full_native'
    return config


class FullViewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_restore_full_inputs_without_changing_original_sampling_or_rng(self):
        base = tiny_config(); train, _ = training_fixture(base)
        pools, _ = build_positive_pools(train)
        left, right = np.random.default_rng(123), np.random.default_rng(123)
        for step in range(12):
            task = 'ITSA' if step % 2 else 'MSTSA'
            crop = sample_batch(train, pools, task, 2, left)
            planned = sample_batch(train, pools, task, 2, right)
            full = full_native_batch(train, planned)
            self.assertEqual(left.bit_generator.state, right.bit_generator.state)
            for key in ('identities', 'pairs'):
                np.testing.assert_array_equal(full[key], crop[key])
            self.assertEqual(full['crop_windows'], crop['crop_windows'])
            for slot, (si, i, j) in enumerate(full['pairs']):
                for key in ('positions', 'observed', 'time_ids', 'times'):
                    np.testing.assert_array_equal(full[key][slot], train[si][key][i])
                    np.testing.assert_array_equal(full[key][slot+2], train[si][key][j])
            self.assertEqual(full['augmentation']['output_points'], crop['augmentation']['input_points'])
            self.assertEqual(full['augmentation']['cropped_pairs'], 0)
            # Original batch is not mutated while it remains the audit source.
            self.assertEqual(planned['augmentation'], crop['augmentation'])

    def test_paired_fit_exact_init_rng_positive_batches_and_actual_full_exposure(self):
        base = tiny_config(); base['model']['dropout'] = .1
        config = full_config(base); train, normalizer = training_fixture(base)
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            old = train_native(train, normalizer, base, root/'original'/'fit')
            new = train_native(train, normalizer, config, root/'new'/'fit', reference_fit=root/'original'/'fit')
            self.assertEqual(new['scope'], FULL_SCOPE)
            self.assertTrue(new['pairing']['completed'])
            self.assertEqual(new['pairing']['successful_paired_updates'], 4)
            self.assertEqual(new['exposure']['nonzero_lr_updates'], 3)
            self.assertGreater(new['exposure']['token_exposures'], old['exposure']['token_exposures'])
            self.assertEqual(new['exposure']['planned_token_exposures'], old['exposure']['token_exposures'])
            self.assertEqual(new['exposure']['cropped_pairs'], 0)
            for key in ('scenes', 'identities', 'pairs', 'positive_pair_exposures'):
                self.assertEqual(new['exposure'][key], old['exposure'][key])
            initial = [torch.load(root/name/'fit/step-0.pt', weights_only=False) for name in ('original','new')]
            for key in initial[0]['model']:
                self.assertTrue(torch.equal(initial[0]['model'][key], initial[1]['model'][key]))
            self.assertTrue(torch.equal(initial[0]['torch_rng'], initial[1]['torch_rng']))
            self.assertEqual(initial[0]['numpy_rng'], initial[1]['numpy_rng'])
            original_rows = [json.loads(x) for x in (root/'original/fit/training.jsonl').read_text().splitlines()]
            new_rows = [json.loads(x) for x in (root/'new/fit/training.jsonl').read_text().splitlines()]
            for a,b in zip(original_rows,new_rows):
                self.assertEqual(a['batch'],b['batch'])
                self.assertEqual(a['augmentation'],b['planned_augmentation'])
            model, final = final_model(root/'new/fit', config, normalizer)
            self.assertEqual(final['scope'], FULL_SCOPE)
            self.assertEqual(verify_exposure(root/'new/fit', new), new['exposure'])
            wrong = deepcopy(final);wrong['scope'] = base['scope']
            torch.save(wrong, root/'wrong.pt')
            with self.assertRaisesRegex(ValueError, 'binding'):
                load_checkpoint(root/'wrong.pt')
            new_rows[0]['planned_augmentation']['cropped_pairs'] += 1
            (root/'new/fit/training.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in new_rows))
            with self.assertRaisesRegex(ValueError, 'exposure|batch'):
                final_model(root/'new/fit', config, normalizer)

    def test_reference_and_optimizer_failure_stop_before_exposure(self):
        base = tiny_config(); config = full_config(base);train, normalizer=training_fixture(base)
        with TemporaryDirectory() as tmp:
            root=Path(tmp); reference=root/'original/fit'
            train_native(train,normalizer,base,reference)
            with self.assertRaisesRegex(ValueError, 'requires074D'):
                train_native(train,normalizer,config,root/'missing')
            wrong=deepcopy(normalizer);wrong['mean'][0]+=1
            with self.assertRaisesRegex(ValueError, 'normalizer'):
                train_native(train,wrong,config,root/'wrong',reference_fit=reference)
            with patch('torch.optim.Adam.step',side_effect=RuntimeError('intentional failure')):
                with self.assertRaisesRegex(RuntimeError, 'intentional'):
                    train_native(train,normalizer,config,root/'failed',reference_fit=reference)
            run=read_json(root/'failed/run.json')
            self.assertEqual(run['status'],'FAILED')
            self.assertEqual(run['pairing']['successful_paired_updates'],0)
            self.assertEqual(run['exposure']['token_exposures'],0)
            self.assertFalse((root/'failed/final.pt').exists())
            self.assertEqual((root/'failed/training.jsonl').read_text(),'')

    def test_full_cli_pipeline_reuses_evaluation_and_binds_original_common_points(self):
        base=tiny_config();config=full_config(base);train,normalizer=training_fixture(base)
        with TemporaryDirectory() as tmp:
            root=Path(tmp)
            source,metadata,validation=validation_fixture(root,base,normalizer)
            manifest={'splits':{'T_core':['t1','t2'],'V_select':['3035']},'components':{'3035':'shared'}}
            write_json(root/'base.json',base);write_json(root/'config.json',config);write_json(root/'manifest.json',manifest)
            evaluate=shared.evaluate_system
            def evaluate_with_sources(*args,**kwargs):
                return evaluate(*args,**kwargs,**source_scores(source,args[0]))
            args=Namespace(source_system=str(source),manifest=str(root/'manifest.json'),config=str(root/'base.json'),
                archive=['not-read.zip'],output=str(root/'original'),device='cpu',cpu_threads=4,time_policy='advisory')
            with patch.object(shared,'production_contract'),patch.object(shared,'_bind_source'), \
                    patch.object(shared,'_load_metadata',return_value=metadata), \
                    patch.object(shared,'prepare_native_dataset',return_value=(train,validation,normalizer,{'test_opened':False})), \
                    patch.object(shared,'evaluate_system',side_effect=evaluate_with_sources):
                shared.run(args)
                args.config=str(root/'config.json');args.output=str(root/'full');args.reference_fit=str(root/'original/fit')
                result=entry.run(args)
            self.assertEqual(result['scope'],FULL_SCOPE)
            self.assertTrue(result['pairing']['completed'])
            self.assertFalse(result['test_opened'])
            system=read_json(root/'full/system/summary.json')
            self.assertEqual(system['scope'],'082-tsa-full-system')
            self.assertIn('reference074d',system)
            self.assertEqual(set(system['comparisons_to_074d']), {'anchor','mean_supported','attribute_hybrid'})
            for method in system['methods'].values():
                self.assertEqual(method['population']['target_identity_count'],5)
            with np.load(root/'full/system/system-predictions/states-3035.npz') as arrays:
                self.assertIn('comparison074d_anchor_common_prediction_valid',arrays.files)
            self.assertFalse((root/'full/fit/selected.pt').exists())
            self.assertTrue((root/'full/system/score-diagnostics.json.gz').exists())
            for name in ('tsa', 'reference074d'):
                self.assertIn('average_precision', system['ranking'][name]['pooled']['all'])
            self.assertEqual(system['ranking']['tsa']['pooled']['all']['pair_count'],
                             system['ranking']['reference074d']['pooled']['all']['pair_count'])
            binding = read_json(root/'full/final-endpoint.json')['score_binding']
            records = load_records(root/'full/final-scores', binding)
            for subdir, field in [('predictions', 'tsa_anchor'),
                                  ('system-predictions', 'tsa_anchor_aligned_prediction_valid')]:
                path = root/'original/system'/subdir/'states-3035.npz'
                with np.load(path) as cache:
                    original = {key: cache[key] for key in cache.files}
                changed = {key: value.copy() for key, value in original.items()}
                if changed[field].dtype == bool:
                    changed[field].flat[0] = not changed[field].flat[0]
                else:
                    changed[field].flat[0] += 1
                np.savez_compressed(path, **changed)
                with self.assertRaisesRegex(ValueError, 'cached replay mismatch'):
                    evaluate(records, binding['selection']['threshold'], source, metadata, root/('bad-'+subdir),
                        execution_scope='082-tsa-full-system', comparison_system=root/'original/system',
                        **source_scores(source, records))
                np.savez_compressed(path, **original)
            with self.assertRaises(FileExistsError):
                entry.run(args)

    def test_production_profile_is_explicit_and_not_accepted_as074d(self):
        root=Path(__file__).resolve().parents[2]
        base=read_json(root/'configs/tsactfer-level1.json');full=read_json(root/'configs/tsactfer-full-view-level1.json')
        manifest=read_json(root/'configs/data/mtad-level1-scene3000.json')
        shared.production_contract(full,manifest,profile='082')
        with self.assertRaises(ValueError):
            shared.production_contract(full,manifest)
        wrong=deepcopy(full);wrong['data']['training_view']='crop'
        with self.assertRaises(ValueError):configuration(wrong)
        wrong=deepcopy(base);wrong['data']['training_view']='full_native'
        with self.assertRaises(ValueError):configuration(wrong)


if __name__=='__main__':
    unittest.main()
