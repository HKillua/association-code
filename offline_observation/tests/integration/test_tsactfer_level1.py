"""074D native packing, actual two-head calls and fixed-final system contracts."""
from argparse import Namespace
from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import torch

from association.evaluation.tsactfer_level1 import evaluate_system, scope_metrics, paper_masks
from association.evaluation.association import evaluate_scenes
from association.utils.runtime import read_json, write_json
from baselines.tsactfer_data import fit_normalizer
from baselines.tsactfer_native import pack_native_scene
from baselines.tsactfer_level1 import (compact_native, pack_compact, train_native,
    final_model, load_checkpoint, score_scenes, verify_exposure, prepare_native_dataset)
from tools.experiments import tsactfer_level1_fit as entry
from tests.fixtures.cache import support_gate_cache_fixture
from tools.analysis.compare_association_duration import _load_arrays
from tools.analysis.frozen_fusion_transfer import groups_from_cache

ROOT = Path(__file__).resolve().parents[2]


def tiny_config():
    config = read_json(ROOT/'configs/tsactfer-level1.json')
    config['model'].update(length=8, d_model=8, nhead=2, num_layers=1, dim_feedforward=16, dropout=0.)
    config['data']['max_points'] = 8
    config['training'].update(updates=4, warmup_updates=1, pair_batch_size=2, inference_batch_tracks=8)
    return config


def native_scene(sid='t', split='T_core'):
    tracks = []
    for identity, offset in (('ship-a', 0.), ('ship-b', .1), ('ship-c', .2)):
        for part, source, start in ((0, 'ais', 0.), (1, 'ais', 600.), (2, 'radar', 0.)):
            times = np.linspace(start, start+240., 20)
            values = np.column_stack((30.+offset+times*.00001, 120.+offset+times*.00001,
                                      np.full(20, 8.), np.full(20, 45.)))
            tracks.append(dict(node_id=f'{sid}-{identity}-{part}', identity_id=identity, source=source,
                canonical_times=times, canonical_values=values, times=times, values=values))
    ids = np.array([t['identity_id'] for t in tracks])
    pairs = np.column_stack(np.triu_indices(len(tracks), 1))
    return dict(scene_id=sid, component_id='component-'+sid, split=split, tracks=tracks,
        node_ids=[t['node_id'] for t in tracks], identities=ids, sources=[t['source'] for t in tracks],
        intervals=np.array([[t['canonical_times'][0], t['canonical_times'][-1]] for t in tracks]),
        pairs=pairs, labels=(ids[pairs[:, 0]] == ids[pairs[:, 1]]).astype(np.float32))


def training_fixture(config):
    views = [compact_native(native_scene(sid), config['data']['max_points']) for sid in ('t1', 't2')]
    normalizer = fit_normalizer(views)
    from association.data.native import SELECTION_POLICY
    normalizer.update(training_splits=['T_core'], native_max_points=config['data']['max_points'],
                      native_selection_policy=SELECTION_POLICY)
    return [pack_compact(view, normalizer, config) for view in views], normalizer


def validation_fixture(root, config, normalizer):
    source, metadata, _ = support_gate_cache_fixture(root)
    fragment, _ = _load_arrays(source, '3035')
    n = len(fragment['node_ids'])
    length = config['model']['length']
    positions = fragment['native_states'][:, :length, :2][..., [1, 0]].astype(np.float32)
    positions = ((positions-np.array(normalizer['mean']))/np.array(normalizer['std'])).astype(np.float32)
    pairs = np.column_stack(np.triu_indices(n, 1))
    record = dict(scene_id='3035', component_id='shared', split='V_select',
        node_ids=fragment['node_ids'].tolist(), identities=fragment['identities'], sources=fragment['sources'],
        pairs=pairs, labels=(fragment['identities'][pairs[:, 0]] == fragment['identities'][pairs[:, 1]]).astype(np.float32),
        intervals=np.tile([0., 240.], (n, 1)), positions=positions,
        observed=np.ones((n, length), bool), time_ids=np.tile(np.arange(length), (n, 1)),
        times=np.tile(np.linspace(0., 240., length), (n, 1)))
    metadata.update(ols_working_point={'threshold': .5}, association_threshold=.5)
    old = source_scores(source, [record])
    metadata['association'] = evaluate_scenes(old['reference_records'], .5)
    (source.parent/'association').mkdir()
    write_json(source.parent/'association/baseline.json', dict(selection={'threshold': .5},
        motion_observation_policy='canonical_all',
        selected_threshold={'metrics': evaluate_scenes(old['ols_records'], .5)}))
    return source, metadata, [record]


def source_scores(source, records):
    """Full candidate fixtures whose decisions reproduce each frozen partition."""
    result = {}
    for name in ('ols', 'learned_assoc'):
        result[name] = []
        for row in records:
            fragment, _ = _load_arrays(source, row['scene_id'])
            membership = np.empty(len(row['node_ids']), np.int64)
            for index, nodes in enumerate(groups_from_cache(fragment, name)):
                membership[nodes] = index
            pairs = row['pairs']
            scores = (membership[pairs[:, 0]] == membership[pairs[:, 1]]).astype(float)
            result[name].append(dict(row, scores=scores))
    return dict(ols_records=result['ols'], reference_records=result['learned_assoc'])


class TSA074Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_compact_native_matches_adapter_without_retaining_canonical_arrays(self):
        config = tiny_config()
        train, normalizer = training_fixture(config)
        raw = native_scene('v', 'V_select')
        compact = compact_native(raw, 8)
        self.assertNotIn('tracks', compact)
        self.assertNotIn('audit', compact)
        self.assertEqual(compact['values'].shape, (9, 8, 4))
        actual = pack_compact(compact, normalizer, config)
        expected = pack_native_scene(raw, normalizer, max_points=8, length=8)
        for key in ('positions', 'observed', 'time_ids', 'times', 'pairs', 'labels'):
            np.testing.assert_array_equal(actual[key], expected[key])
        self.assertNotIn('values', actual)
        self.assertNotIn('tracks', actual)
        for split in ('V_select', 'T_test'):
            wrong = deepcopy(train)
            wrong[0]['split'] = split
            with TemporaryDirectory() as tmp:
                with self.assertRaisesRegex(ValueError, 'T_core'):
                    train_native(wrong, normalizer, config, Path(tmp)/'fit')
        long = deepcopy(compact)
        long['observation_times'][long['observed']] += 90000.
        with self.assertRaisesRegex(ValueError, 'vocabulary'):
            pack_compact(long, normalizer, config)

    def test_actual_fit_final_optimizer_heads_sampling_and_system(self):
        config = tiny_config(); train, normalizer = training_fixture(config)
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            fit = train_native(train, normalizer, config, root/'fit')
            self.assertEqual(fit['optimizer_updates'], 4)
            self.assertEqual(fit['exposure']['nonzero_lr_updates'], 3)
            self.assertEqual(fit['exposure']['task_calls'], {'ITSA': 2, 'MSTSA': 2})
            self.assertEqual(fit['exposure']['positive_pair_exposures'], 8)
            self.assertEqual(verify_exposure(root/'fit', fit), fit['exposure'])
            model, saved = final_model(root/'fit', config, normalizer)
            names = [name for name, _ in model.named_parameters()]
            for name, index in zip(names, saved['optimizer']['param_groups'][0]['params']):
                self.assertEqual(float(saved['optimizer']['state'][index]['step']), 2 if name.startswith('heads.') else 4)
            source, metadata, validation = validation_fixture(root, config, normalizer)
            records = score_scenes(model, validation, 'cpu', 8)
            from association.evaluation.association import select_threshold
            selection = select_threshold(records)
            result = evaluate_system(records, selection['threshold'], source, metadata, root/'system',
                **source_scores(source, records))
            self.assertEqual(result['status'], 'COMPLETED')
            self.assertEqual(len(result['methods']), 6)
            for method in result['methods'].values():
                self.assertEqual(method['population']['target_identity_count'], 5)
            self.assertEqual(result['association']['tsa']['paper_scope']['itsa']['threshold'], selection['threshold'])
            self.assertIn('reference056', result['comparisons_to_056']['anchor'])
            with np.load(root/'system/system-predictions/states-3035.npz', allow_pickle=False) as arrays:
                self.assertIn('tsa_anchor_aligned_prediction_valid', arrays.files)
                self.assertIn('comparison056_anchor_common_prediction_valid', arrays.files)
            changed = deepcopy(saved)
            head_index = next(index for name, index in zip(names, saved['optimizer']['param_groups'][0]['params']) if name.startswith('heads.'))
            changed['optimizer']['state'][head_index]['step'] = torch.tensor(4.)
            torch.save(changed, root/'bad.pt')
            with self.assertRaisesRegex(ValueError, 'call counts'):
                load_checkpoint(root/'bad.pt')
            changed = deepcopy(saved); changed['actual_updates'] = 2
            torch.save(changed, root/'bad.pt')
            with self.assertRaisesRegex(ValueError, 'binding'):
                load_checkpoint(root/'bad.pt')
            log = root/'fit/training.jsonl'
            log.write_text('\n'.join(log.read_text().splitlines()[:-1])+'\n')
            with self.assertRaisesRegex(ValueError, 'exposure'):
                final_model(root/'fit', config, normalizer)

    def test_score_population_component_and_old_decisions_are_bound(self):
        config = tiny_config(); _, normalizer = training_fixture(config)
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, metadata, records = validation_fixture(root, config, normalizer)
            records[0]['scores'] = np.zeros(len(records[0]['pairs']))
            originals = source_scores(source, records)
            for name, mutate, error in (
                ('nan', lambda r: r[0]['scores'].__setitem__(0, np.nan), 'finite'),
                ('shape', lambda r: r[0].update(scores=r[0]['scores'][:, None]), 'candidate'),
                ('component', lambda r: r[0].update(component_id='wrong'), 'component'),
                ('pairs', lambda r: r[0]['pairs'].__setitem__(0, [0, 0]), 'source candidate pairs'),
            ):
                wrong = deepcopy(records); mutate(wrong)
                with self.assertRaisesRegex(ValueError, error):
                    evaluate_system(wrong, .5, source, metadata, root/name, **originals)
            for name in ('ols_records', 'reference_records'):
                wrong = deepcopy(originals)
                wrong[name][0]['scores'][:] = 0
                adjusted = deepcopy(metadata)
                baseline = dict(selection={'threshold': .5}, motion_observation_policy='canonical_all',
                    selected_threshold={'metrics': evaluate_scenes(wrong['ols_records'], .5)})
                write_json(source.parent/'association/baseline.json', baseline)
                adjusted['association'] = evaluate_scenes(wrong['reference_records'], .5)
                with self.assertRaisesRegex(ValueError, 'do not reproduce frozen groups'):
                    evaluate_system(records, .5, source, adjusted, root/name, **wrong)
                self.assertEqual(read_json(root/name/'run.json')['status'], 'FAILED')
            write_json(source.parent/'association/baseline.json', dict(selection={'threshold': .5},
                motion_observation_policy='canonical_all',
                selected_threshold={'metrics': evaluate_scenes(originals['ols_records'], .5)}))
            # Edge(1,3) joins already-consumed helpers and changes no group, but
            # changes edge metrics. Both published source metrics must detect it.
            for name in ('ols_records', 'reference_records'):
                wrong = deepcopy(originals)
                row = wrong[name][0]
                row['scores'][np.all(row['pairs'] == [1, 3], axis=1)] = 1.
                with self.assertRaisesRegex(ValueError, 'endpoint summary'):
                    evaluate_system(records, .5, source, metadata, root/('edge-'+name), **wrong)

    def test_streamed_native_archive_reads_no_reference_or_sealed_split(self):
        from tests.integration.test_tsactfer_fit import long_track_archive
        from association.data.legacy_dataset import prepare
        config = tiny_config()
        with TemporaryDirectory() as tmp:
            root = Path(tmp); archive = long_track_archive(root)
            protocol = read_json(ROOT/'configs/smoke.json')
            protocol['caps'] = {'T_core': 4, 'T_head': 1, 'V_select': 2, 'T_test': 1}
            manifest = prepare(archive, protocol, root/'prepared', synthetic=True)
            seen = []
            train, validation, normalizer, audit = prepare_native_dataset([str(archive)], manifest,
                config, verify_validation=lambda raw: seen.append(raw['scene_id']))
            self.assertEqual(seen, manifest['splits']['V_select'])
            self.assertEqual(normalizer['training_scene_ids'], manifest['splits']['T_core'])
            self.assertEqual([row['scene_id'] for row in validation], manifest['splits']['V_select'])
            self.assertFalse(audit['reference_read'])
            self.assertFalse(audit['test_opened'])
            allowed = set(manifest['splits']['T_core']+manifest['splits']['V_select'])
            self.assertTrue(audit['archive_members_read'])
            for entry in audit['archive_members_read']:
                self.assertNotIn('参考', entry['member'])
                self.assertIn(Path(entry['member']).stem.split('-')[-1], allowed)
            for row in train+validation:
                self.assertNotIn('tracks', row)
                self.assertNotIn('values', row)
                self.assertLessEqual(row['positions'].shape[1], config['data']['max_points'])

    def test_failed_optimizer_does_not_commit_exposure(self):
        config = tiny_config(); train, normalizer = training_fixture(config)
        with TemporaryDirectory() as tmp:
            output = Path(tmp)/'fit'
            with patch('torch.optim.Adam.step', side_effect=RuntimeError('deliberate optimizer failure')):
                with self.assertRaisesRegex(RuntimeError, 'optimizer failure'):
                    train_native(train, normalizer, config, output)
            run = read_json(output/'run.json')
            self.assertEqual(run['status'], 'FAILED')
            self.assertEqual(run['optimizer_updates'], 0)
            self.assertEqual(run['exposure']['positive_pair_exposures'], 0)
            self.assertFalse((output/'final.pt').exists())
            self.assertEqual((output/'training.jsonl').read_text(), '')

    def test_paper_subset_keeps_one_full_population_threshold(self):
        raw = native_scene('v', 'V_select')
        row = {key: raw[key] for key in ('scene_id', 'component_id', 'node_ids', 'identities', 'sources', 'pairs', 'labels', 'intervals')}
        row['scores'] = np.linspace(-1., 1., len(row['pairs']))
        masks = paper_masks(row)
        self.assertGreater(masks['itsa'].sum(), 0)
        self.assertGreater(masks['mstsa'].sum(), 0)
        report = scope_metrics([row], .75)
        for item in report.values():
            self.assertEqual(item['threshold'], .75)
            self.assertFalse(item['threshold_reselected'])

    def test_real_cli_pipeline_final_cache_and_no_selected_checkpoint(self):
        config = tiny_config(); train, normalizer = training_fixture(config)
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, metadata, validation = validation_fixture(root, config, normalizer)
            manifest = {'splits': {'T_core': ['t1', 't2'], 'V_select': ['3035']}, 'components': {'3035': 'shared'}}
            write_json(root/'config.json', config); write_json(root/'manifest.json', manifest)
            original_evaluate = entry.evaluate_system
            def evaluate(*args, **kwargs):
                return original_evaluate(*args, **kwargs, **source_scores(source, args[0]))
            args = Namespace(source_system=str(source), manifest=str(root/'manifest.json'),
                config=str(root/'config.json'), archive=['not-read.zip'], output=str(root/'result'),
                device='cpu', cpu_threads=4, time_policy='advisory')
            with patch.object(entry, 'production_contract'), patch.object(entry, '_bind_source'), \
                    patch.object(entry, '_load_metadata', return_value=metadata), \
                    patch.object(entry, 'prepare_native_dataset', return_value=(train, validation, normalizer, {'test_opened': False})), \
                    patch.object(entry, 'evaluate_system', side_effect=evaluate):
                result = entry.run(args)
            self.assertEqual(result['status'], 'COMPLETED')
            self.assertEqual(result['endpoint'], 'final')
            self.assertEqual(result['optimizer_updates'], 4)
            self.assertTrue((root/'result/final-scores/index.json').is_file())
            self.assertFalse((root/'result/fit/selected.pt').exists())
            endpoint = read_json(root/'result/final-endpoint.json')
            self.assertEqual(endpoint['actual_updates'], 4)
            self.assertTrue(endpoint['scores_from_actual_final_checkpoint'])
            self.assertEqual(read_json(root/'result/system/run.json')['completed_scenes'], 1)
            with self.assertRaises(FileExistsError):
                entry.run(args)


if __name__ == '__main__':
    unittest.main()
