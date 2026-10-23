"""Bounded079 qualification: real model, exact crop path and paired populations."""
from argparse import Namespace
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import torch

from association.evaluation.tsactfer_input_probe import (DEFAULT, select_roster, bind_source,
    prepare_inputs, build_views, evaluate_views, probe)
from association.utils.runtime import read_json, write_json
from baselines.tsactfer_level1 import train_native, final_model
from tests.test_tsactfer_level1 import tiny_config, training_fixture


class TSAInputProbeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def config(self):
        return dict(DEFAULT, scene_count=2, pairs_per_head_scene=3, inference_batch_tracks=4)

    def test_real_training_to_frozen_paired_full_crop_probe(self):
        tsa = tiny_config(); scenes, normalizer = training_fixture(tsa)
        with TemporaryDirectory() as tmp:
            fit = Path(tmp)/'fit'
            train_native(scenes, normalizer, tsa, fit)
            model, _ = final_model(fit, tsa, normalizer)
            config = self.config()
            with patch('torch.optim.Adam', side_effect=AssertionError('no optimizer')):
                result, saved = probe(model, scenes, tsa, config)
            self.assertTrue(result['frozen_weights_unchanged'])
            self.assertEqual(result['optimizer_updates'], 0)
            self.assertFalse(model.training)
            for task in ('ITSA', 'MSTSA'):
                rows, arrays = saved[task]['rows'], saved[task]['arrays']
                self.assertEqual(len(rows), 6)
                self.assertEqual(result['tasks'][task]['paired_anchors'], 6)
                self.assertTrue(any(row['crop_tokens'] < row['full_tokens'] for row in rows))
                for i, same, cross in arrays['mapping']:
                    self.assertEqual(rows[i]['scene_id'], rows[same]['scene_id'])
                    self.assertNotEqual(rows[i]['scene_id'], rows[cross]['scene_id'])
                    self.assertNotEqual(rows[i]['identity'], rows[same]['identity'])
                    # Identities repeat across scenes: exclude globally, not by scene.
                    self.assertNotEqual(rows[i]['identity'], rows[cross]['identity'])
                for mode in ('full', 'train_crop'):
                    np.testing.assert_array_equal(arrays[f'same_scene__{mode}_positive'],
                                                  arrays[f'cross_scene__{mode}_positive'])
                    for domain in ('same_scene', 'cross_scene'):
                        metric = result['tasks'][task]['conditions'][domain+'__'+mode]['pooled']
                        self.assertEqual(metric['positive_count'], metric['negative_count'])
                        self.assertEqual(metric['prevalence'], .5)
                        self.assertTrue(0 <= metric['roc_auc'] <= 1)
                self.assertTrue(all(row['crop']['fraction'] >= .5 for row in rows))
            again, repeated = probe(model, scenes, tsa, config)
            self.assertEqual(result, again)
            for task in saved:
                self.assertEqual(saved[task]['rows'], repeated[task]['rows'])
                for key in saved[task]['arrays']:
                    np.testing.assert_array_equal(saved[task]['arrays'][key], repeated[task]['arrays'][key])

    def test_empty_head_and_missing_domains_do_not_invent_replacements(self):
        from baselines.tsactfer import TSActfer, TSAConfig
        tsa = tiny_config(); scenes, _ = training_fixture(tsa)
        model = TSActfer(TSAConfig(**tsa['model']))
        # Removing all positives must return unavailable metrics, not fake perfect scores.
        empty = deepcopy(scenes)
        for scene in empty:
            scene['labels'][:] = 0
        result, saved = probe(model, empty, tsa, self.config())
        for task in result['tasks'].values():
            self.assertFalse(task['available'])
            self.assertEqual(task['paired_anchors'], 0)
            for condition in task['conditions'].values():
                self.assertIsNone(condition['pooled']['roc_auc'])
                self.assertIsNone(condition['pooled']['average_precision'])
        views, _ = build_views(model, scenes[:1], tsa, self.config())
        report, _ = evaluate_views(views, self.config())
        self.assertEqual(report['ITSA']['paired_anchors'], 0)
        self.assertEqual(len(report['ITSA']['excluded_anchors']), 3)
        wrong = deepcopy(scenes); wrong[0]['split'] = 'V_select'
        with self.assertRaisesRegex(ValueError, 'T_core'):
            probe(model, wrong, tsa, self.config())

    def test_roster_is_fixed_without_validation_or_test(self):
        manifest = dict(splits={'T_core': list(map(str, range(100))), 'V_select': ['v'], 'T_test': ['test']})
        first = select_roster(manifest, DEFAULT)
        self.assertEqual(first, select_roster(manifest, DEFAULT))
        self.assertEqual(len(first), 32)
        self.assertEqual(len(set(first)), 32)
        bad = deepcopy(manifest); bad['splits']['T_core'][0] = 'test'
        with self.assertRaisesRegex(ValueError, 'disjoint'):
            select_roster(bad, DEFAULT)
        with self.assertRaisesRegex(ValueError, 'cap'):
            select_roster(manifest, dict(DEFAULT, scene_count=33))

    def test_final_checkpoint_status_normalizer_and_calls_binding(self):
        tsa = tiny_config(); scenes, normalizer = training_fixture(tsa)
        manifest = dict(splits={'T_core': ['t1', 't2'], 'V_select': ['v'], 'T_test': ['sealed']})
        with TemporaryDirectory() as tmp:
            source = Path(tmp)
            train_native(scenes, normalizer, tsa, source/'fit')
            common = dict(status='COMPLETED', scope=tsa['scope'], configuration=tsa,
                optimizer_updates=4, test_opened=False, endpoint='final')
            write_json(source/'summary.json', common)
            write_json(source/'run.json', dict(common, phase='completed', checkpoint_selection=False,
                retries=0, old066_acceptance='REJECTED'))
            endpoint = dict(status='COMPLETED', configuration=tsa, step=4, actual_updates=4,
                checkpoint=str(source/'fit/final.pt'), scene_ids=['v'], test_opened=False,
                scores_from_actual_final_checkpoint=True, normalizer=normalizer)
            write_json(source/'final-endpoint.json', endpoint)
            model, _, audit = bind_source(source, manifest, tsa)
            self.assertTrue(audit['source_final_bound'])
            self.assertTrue(all(not p.requires_grad for p in model.parameters()))
            bad = dict(endpoint, step=0)
            write_json(source/'final-endpoint.json', bad)
            with self.assertRaisesRegex(ValueError, 'binding'):
                bind_source(source, manifest, tsa)
            write_json(source/'final-endpoint.json', endpoint)
            bad = dict(common, status='FAILED')
            write_json(source/'summary.json', bad)
            with self.assertRaisesRegex(ValueError, 'completed'):
                bind_source(source, manifest, tsa)
            write_json(source/'summary.json', common)
            wrong = deepcopy(manifest); wrong['splits']['T_core'] = ['t1']
            with self.assertRaisesRegex(ValueError, 'normalizer'):
                bind_source(source, wrong, tsa)

    def test_cli_writes_recomputable_scores_and_preserves_failed_attempt(self):
        from tools.analysis import tsactfer_input_probe as entry
        tsa = tiny_config(); scenes, normalizer = training_fixture(tsa)
        manifest = dict(splits={'T_core': ['t1', 't2'], 'V_select': ['v'], 'T_test': ['sealed']})
        with TemporaryDirectory() as tmp:
            root = Path(tmp).resolve(); source = root/'source'; source.mkdir()
            train_native(scenes, normalizer, tsa, source/'fit')
            common = dict(status='COMPLETED', scope=tsa['scope'], configuration=tsa,
                optimizer_updates=4, test_opened=False, endpoint='final')
            write_json(source/'summary.json', common)
            write_json(source/'run.json', dict(common, phase='completed', checkpoint_selection=False,
                retries=0, old066_acceptance='REJECTED'))
            write_json(source/'final-endpoint.json', dict(status='COMPLETED', configuration=tsa,
                step=4, actual_updates=4, checkpoint=str(source/'fit/final.pt'), scene_ids=['v'],
                test_opened=False, scores_from_actual_final_checkpoint=True, normalizer=normalizer))
            write_json(root/'manifest.json', manifest); write_json(root/'config.json', self.config())
            args = Namespace(source_074d=str(source), output=str(root/'result'),
                manifest=str(root/'manifest.json'), config=str(root/'config.json'),
                archive=['fixture'], device='cpu', cpu_threads=4)
            def read(path):
                return tsa if str(path).endswith('configs/tsactfer-level1.json') else read_json(path)
            # Only large-data production inputs are replaced; source loader, final
            # weights, model, sampling, inference, metrics and artifact writes are real.
            with patch.object(entry, 'production_contract'), patch.object(entry, 'read_json', side_effect=read), \
                    patch.object(entry, 'prepare_inputs', return_value=(scenes, {'scene_ids': ['t1', 't2']})):
                result = entry.run(args)
                self.assertEqual(read_json(root/'result/run.json')['status'], 'COMPLETED')
                self.assertTrue((root/'result/report.md').is_file())
                with np.load(root/'result/scores/itsa.npz', allow_pickle=False) as arrays:
                    self.assertEqual(arrays['mapping'].shape, (6, 3))
                self.assertEqual(result['tasks']['ITSA']['paired_anchors'], 6)
                with self.assertRaises(FileExistsError):
                    entry.run(args)
                args.output = str(root/'failed')
                with patch.object(entry, 'probe', side_effect=RuntimeError('fixture failure')):
                    with self.assertRaisesRegex(RuntimeError, 'fixture failure'):
                        entry.run(args)
                self.assertEqual(read_json(root/'failed/run.json')['status'], 'FAILED')
                self.assertTrue((root/'failed/execution.json').is_file())
        with self.assertRaisesRegex(ValueError, 'frozen bounded'):
            entry.production_contract(dict(DEFAULT, seed=3), manifest, Path('source'))

    def test_archive_reader_reads_only_bounded_training_measurements_and_labels(self):
        from tests.integration.test_tsactfer_fit import long_track_archive
        from association.data.legacy_dataset import prepare
        from baselines.tsactfer_level1 import prepare_native_dataset
        root_repo = Path(__file__).resolve().parents[2]
        tsa = tiny_config()
        with TemporaryDirectory() as tmp:
            root = Path(tmp); archive = long_track_archive(root)
            protocol = read_json(root_repo/'configs/smoke.json')
            protocol['caps'] = {'T_core': 4, 'T_head': 1, 'V_select': 2, 'T_test': 1}
            manifest = prepare(archive, protocol, root/'prepared', synthetic=True)
            _, _, normalizer, _ = prepare_native_dataset([str(archive)], manifest, tsa)
            roster = select_roster(manifest, self.config())
            records, audit = prepare_inputs([str(archive)], manifest, roster, normalizer, tsa)
            self.assertEqual([r['scene_id'] for r in records], roster)
            self.assertEqual(len(audit['archive_members_read']), 4)
            self.assertEqual({r['kind'] for r in audit['archive_members_read']}, {'labels', 'measurements'})
            self.assertEqual({r['scene_id'] for r in audit['archive_members_read']}, set(roster))
            with self.assertRaisesRegex(ValueError, 'subset'):
                prepare_inputs([str(archive)], manifest, manifest['splits']['V_select'], normalizer, tsa)


if __name__ == '__main__':
    unittest.main()
