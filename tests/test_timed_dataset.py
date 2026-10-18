import copy
from contextlib import redirect_stderr, redirect_stdout
import csv
from dataclasses import asdict
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import numpy as np
import torch

from association import runtime
from association.data import DataError, MTADArchive
from association.models import ModelConfig
from association.fusion import PairFusionConfig
from association.timed_dataset import load_timed_dataset
from tools.experiments.tmtaff_complete_fit import main


def csv_bytes(header, rows):
    stream = io.StringIO(newline='')
    writer = csv.writer(stream)
    writer.writerow(header)
    writer.writerows(rows)
    return stream.getvalue().encode('utf-8-sig')


def frozen_manifest():
    splits = {'T_core': ['0'], 'T_head': ['1'], 'V_select': ['2', '4'], 'T_test': ['3']}
    ids = ['0', '1', '2', '3', '4']
    return dict(splits=splits, all_scenes=ids, components={sid: sid for sid in ids},
        component_members={sid: [sid] for sid in ids},
        component_splits={sid: split for split, group in splits.items() for sid in group},
        label_members={sid: {} for sid in ids}, caps={split: len(group) for split, group in splits.items()},
        development_scenes=['0'])


class TimedDatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.path = self.root / 'native.zip'
        self.manifest = frozen_manifest()
        self.data = dict(length=4, max_observations=4, time_unit_seconds=10.)
        self.alignment = {sid: dict(time_offset_seconds=1000., max_gap_seconds=10.)
                          for sid in ('0', '1', '2', '4')}
        self.config = dict(scope='TMTAFF-time-adapted-v1',
            model=asdict(ModelConfig(length=4, d_model=8, nhead=2, num_layers=1, dim_feedforward=16, dropout=0)),
            fusion=asdict(PairFusionConfig(fc_hidden=8, length=4, d_model=8, nhead=2,
                num_layers=1, dim_feedforward=16, dropout=0)),
            data={key: self.data[key] for key in ('max_observations', 'time_unit_seconds')},
            train_splits=['T_core'], training=dict(
                updates=dict.fromkeys(('TCM', 'TAM_FROZEN', 'TFM_FROZEN', 'JOINT'), 1),
                learning_rates=dict.fromkeys(('TCM', 'TAM_FROZEN', 'TFM_FROZEN', 'JOINT'), .001),
                seed=9, branch='ce', batch_tracks=4, pair_batch_size=4, pair_microbatch=2,
                track_microbatch=2, completion_weight=.1, gradient_clip=1., max_hours=.1,
                validate_every_updates=1, completion_cache_mib=1, position_loss_mode='standard_smooth_l1',
                lr_schedule='constant', singleton_policy='observed', validation_batches=dict(tracks=4, pairs=4)))
        self.write_zip()

    def write_zip(self, *, empty_labels=False, zero_label_duration=False, missing_reference=None):
        with zipfile.ZipFile(self.path, 'w') as archive:
            for sid in self.manifest['all_scenes']:
                labels, values, refs = [], [], []
                if sid == '4':
                    if not empty_labels:
                        labels = [[5, 5 if zero_label_duration else 25, 'target4a-0-0', '9001', '0']]
                    refs = [['target4a-0', t+1000, 20., 120., 5., 45.] for t in (5, 15, 25)]
                else:
                    for name, source, batch, times in (
                            ('a', '9001', '0', (0, 10, 20, 30)), ('a', '9002', '1', (12, 18)),
                            ('b', '9001', '2', (0, 10, 20, 30)), ('b', '9002', '3', (0, 10))):
                        labels.append([times[0], times[-1], f'target{sid}{name}-0-{batch}', source, batch])
                        values.extend([[batch, source, t, 20+int(sid)+(.1 if name == 'b' else 0)+t*.0001,
                                        120+t*.0002, 5, 45] for t in times])
                    labels.append([0, 30, f'target{sid}c-0-4', '9003', '4'])  # quarantined: no measurement
                    for name in ('a', 'b', 'c'):
                        if (sid, name) == missing_reference:
                            continue
                        refs.extend([[f'target{sid}{name}-0', t+1000, 20+int(sid)+t*.0001,
                                      120+t*.0002, 5, 45] for t in (0, 10, 20, 30)])
                for directory, stem, header, rows in (
                        ('关联表', '关联结果', ['t_s', 't_e', 'mmsi', 'source', 'batch'], labels),
                        ('量测场景', '场景', ['batch', 'source', 'time', 'lat', 'lon', 'vel', 'cou'], values),
                        ('真实场景', '场景', ['MMSI', 'time', 'lat', 'lon', 'vel', 'cou'], refs)):
                    content = b'SEALED TEST MUST NOT BE READ' if sid == '3' else csv_bytes(header, rows)
                    archive.writestr(f'数据集level2/{directory}/{stem}-{sid}.csv', content)

    def load(self, **kwargs):
        arguments = dict(train_splits=['T_core'], data_config=self.data, alignment_by_scene=self.alignment)
        arguments.update(kwargs)
        return load_timed_dataset(self.path, self.manifest, **arguments)

    def cli_args(self, output):
        for filename, value in [('manifest.json', self.manifest), ('alignment.json', self.alignment), ('config.json', self.config)]:
            runtime.write_json(self.root / filename, value)
        return ['--archive', str(self.path), '--manifest', str(self.root/'manifest.json'),
                '--alignment', str(self.root/'alignment.json'), '--config', str(self.root/'config.json'),
                '--output', str(self.root/output), '--cpu-threads', '1']

    def test_native_short_fragments_full_label_population_and_explicit_clock_alignment(self):
        result = self.load()
        scene = result['train'][0]
        self.assertEqual(len(scene['native_values']), 4)
        short = scene['node_ids'].index('0:9002:1')
        self.assertEqual(scene['observed'][short].sum(), 0)
        self.assertEqual(scene['native_observed'][short].sum(), 2)
        self.assertEqual(len(scene['pairs']), 6)
        self.assertEqual(set(scene['fusion_targets']), {'target0a-0', 'target0b-0', 'target0c-0'})
        self.assertNotIn('target0c-0', scene['identities'])
        self.assertNotIn('tracks', scene)
        for target in scene['fusion_targets'].values():
            np.testing.assert_array_equal(target['query_times'], scene['grid'])
            self.assertTrue(target['valid'].all())
        np.testing.assert_allclose(scene['fusion_targets']['target0a-0']['states'][:, 0], [20, 20.001, 20.002, 20.003])
        empty = result['validation'][1]
        self.assertEqual(empty['native_values'].shape, (0, 4, 4))
        np.testing.assert_array_equal(empty['grid'], np.linspace(5, 25, 4))
        self.assertEqual(list(empty['fusion_targets']), ['target4a-0'])
        self.assertTrue(empty['fusion_targets']['target4a-0']['valid'].all())
        self.assertEqual(scene['reference_alignment']['time_offset_seconds'], 1000.)
        self.assertFalse(result['report']['test_opened'])

    def test_missing_reference_preserves_the_declared_target_with_invalid_mask(self):
        self.write_zip(missing_reference=('0', 'c'))
        result = self.load()
        target = result['train'][0]['fusion_targets']['target0c-0']
        self.assertFalse(target['valid'].any())
        self.assertTrue(target['report']['missing_identity'])
        self.assertEqual(result['report']['scenes'][0]['target_without_reference_count'], 1)

    def test_split_and_alignment_validation_precede_any_archive_access(self):
        cases = [dict(train_splits=['T_test']), dict(train_splits=['T_head']),
                 dict(train_splits=['T_head', 'T_core']), dict(alignment_by_scene={}),
                 dict(alignment_by_scene={**self.alignment, '2': {'time_offset_seconds': 1000.}}),
                 dict(alignment_by_scene={**self.alignment, '2': dict(time_offset_seconds=1000., max_gap_seconds=-1.)}),
                 dict(data_config={**self.data, 'max_observations': 1})]
        for changes in cases:
            with self.subTest(changes=changes), \
                    patch('association.timed_dataset.MTADArchive', side_effect=AssertionError('No archive opened')):
                with self.assertRaises(DataError):
                    self.load(**changes)

    def test_only_declared_members_read_and_reference_consumed_before_next_scene(self):
        from association.timed_dataset import build_scene_fusion_targets
        events = []
        original_read = zipfile.ZipFile.read
        def read(archive, member, *args, **kwargs):
            events.append(('read', member.filename))
            return original_read(archive, member, *args, **kwargs)
        def build(scene, *args, **kwargs):
            events.append(('build', scene['scene_id']))
            return build_scene_fusion_targets(scene, *args, **kwargs)
        with patch.object(zipfile.ZipFile, 'read', read), \
                patch('association.timed_dataset.build_scene_fusion_targets', side_effect=build), \
                patch.object(MTADArchive, 'archive_sha256', side_effect=AssertionError('No hash gate')):
            self.load()
        reads = [name for kind, name in events if kind == 'read']
        self.assertFalse(any(name.endswith(('-1.csv', '-3.csv')) for name in reads))
        ref_events = [(kind, name) for kind, name in events if kind == 'build' or '真实场景' in name]
        self.assertEqual(ref_events, [('read', '数据集level2/真实场景/场景-0.csv'), ('build', '0'),
            ('read', '数据集level2/真实场景/场景-2.csv'), ('build', '2'),
            ('read', '数据集level2/真实场景/场景-4.csv'), ('build', '4')])
        result = self.load(train_splits=['T_core', 'T_head'])
        self.assertEqual([scene['scene_id'] for scene in result['train']], ['0', '1'])

    def test_empty_scene_without_valid_label_extent_cannot_invent_time_grid(self):
        for changes in (dict(empty_labels=True), dict(zero_label_duration=True)):
            self.write_zip(**changes)
            with self.subTest(changes=changes), \
                    patch('association.timed_dataset.iter_reference_states', side_effect=AssertionError('No reference read')):
                with self.assertRaisesRegex(DataError, 'query grid|positive duration'):
                    self.load()

    def test_loader_budget_check_can_stop_before_reading(self):
        def exhausted():
            raise TimeoutError('preparation exhausted')
        with patch('association.timed_dataset.MTADArchive', side_effect=AssertionError('No archive opened')):
            with self.assertRaisesRegex(TimeoutError, 'exhausted'):
                self.load(check=exhausted)

    def test_preparation_deadline_stops_before_next_native_or_reference_scene(self):
        self.manifest['splits']['T_core'] = ['0', '1']
        self.manifest['splits']['T_head'] = []
        self.manifest['caps'].update(T_core=2, T_head=0)
        self.manifest['component_splits']['1'] = 'T_core'
        loaded = []
        original_load = MTADArchive.load_scene
        def load(archive, sid):
            loaded.append(sid)
            return original_load(archive, sid)
        def native_deadline():
            if loaded:
                raise TimeoutError('native scene consumed remaining budget')
        with patch.object(MTADArchive, 'load_scene', load):
            with self.assertRaisesRegex(TimeoutError, 'native scene'):
                self.load(check=native_deadline)
        self.assertEqual(loaded, ['0'])

        from association.timed_dataset import build_scene_fusion_targets
        built, references_read = [], []
        original_read = zipfile.ZipFile.read
        def read(archive, member, *args, **kwargs):
            if '真实场景' in member.filename:
                references_read.append(member.filename)
            return original_read(archive, member, *args, **kwargs)
        def build(scene, *args, **kwargs):
            result = build_scene_fusion_targets(scene, *args, **kwargs)
            built.append(scene['scene_id'])
            return result
        def reference_deadline():
            if built:
                raise TimeoutError('reference scene consumed remaining budget')
        with patch.object(zipfile.ZipFile, 'read', read), \
                patch('association.timed_dataset.build_scene_fusion_targets', side_effect=build):
            with self.assertRaisesRegex(TimeoutError, 'reference scene'):
                self.load(check=reference_deadline)
        self.assertEqual(references_read, ['数据集level2/真实场景/场景-0.csv'])

    def test_real_zip_loader_to_four_training_stages_and_full_validation(self):
        from association.complete_training import train_complete
        result = self.load()
        output = self.root / 'fit'
        run = train_complete(result['train'], result['validation'], ModelConfig(**self.config['model']),
            PairFusionConfig(**self.config['fusion']), self.config['training'], output,
            device='cpu', initial_elapsed_seconds=.05)
        self.assertEqual(run['status'], 'COMPLETED')
        self.assertGreaterEqual(run['elapsed_seconds'], .05)
        checkpoint = torch.load(output/'checkpoint.pt', weights_only=True)
        self.assertEqual(checkpoint['cursor']['completed_updates'], 4)
        self.assertEqual(checkpoint['normalizer']['input_adapter'], 'native_time_v1')
        self.assertEqual(checkpoint['normalizer']['training_scene_ids'], ['0'])
        summary = runtime.read_json(output/'summary.json')
        self.assertEqual(summary['evaluation']['fusion']['oracle']['all']['position']['available_target_count'], 16)
        self.assertEqual(summary['evaluation']['fusion']['oracle']['all']['position']['missing_prediction_count'], 8)
        self.assertFalse(summary['test_opened'])

    def test_cli_inspect_only_builds_no_model_and_does_not_overwrite(self):
        args = self.cli_args('inspect') + ['--inspect-only']
        with patch('association.timed_models.TimedAssociationModel', side_effect=AssertionError('No model')), \
                patch('association.complete_training.train_complete', side_effect=AssertionError('No training')):
            self.assertEqual(main(args), 0)
        report = runtime.read_json(self.root/'inspect'/'preparation.json')
        self.assertEqual(report['status'], 'INSPECTED')
        self.assertFalse(report['model_constructed'])
        self.assertFalse(report['gpu_training_started'])
        self.assertEqual(main(args), 1)

    def test_cli_fit_requires_device_and_counts_preparation_cost(self):
        self.assertEqual(main(self.cli_args('no-device')), 1)
        self.assertFalse((self.root/'no-device').exists())
        args = self.cli_args('cli-fit') + ['--device', 'cpu']
        self.assertEqual(main(args), 0)
        preparation = runtime.read_json(self.root/'cli-fit'/'preparation.json')
        run = runtime.read_json(self.root/'cli-fit'/'run.json')
        self.assertGreater(preparation['elapsed_seconds'], 0)
        self.assertGreater(run['elapsed_seconds'], preparation['elapsed_seconds'])
        self.assertEqual(run['status'], 'COMPLETED')

    def test_cli_preparation_write_failure_keeps_original_training_error_and_notes(self):
        for original, exit_code, name in ((ValueError('original training error'), 1, 'failed'),
                                          (KeyboardInterrupt('original interruption'), 130, 'interrupted')):
            with self.subTest(name=name):
                args = self.cli_args(name) + ['--device', 'cpu']
                def fail(*args, **kwargs):
                    (self.root/name).mkdir()
                    raise original
                stderr = io.StringIO()
                with patch('association.complete_training.train_complete', side_effect=fail), \
                        patch('tools.experiments.tmtaff_complete_fit.runtime.write_json',
                              side_effect=OSError('secondary preparation storage error')), \
                        redirect_stderr(stderr), redirect_stdout(io.StringIO()):
                    self.assertEqual(main(args), exit_code)
                self.assertIn(f'FAILED {type(original).__name__}: {original}', stderr.getvalue())
                self.assertIn('preparation.json: OSError: secondary preparation storage error', stderr.getvalue())
                self.assertNotIn('FAILED OSError', stderr.getvalue())

    def test_cli_preparation_write_error_alone_still_fails(self):
        args = self.cli_args('write-failed') + ['--device', 'cpu']
        def complete(*args, **kwargs):
            (self.root/'write-failed').mkdir()
            return dict(status='COMPLETED')
        stderr, stdout = io.StringIO(), io.StringIO()
        with patch('association.complete_training.train_complete', side_effect=complete), \
                patch('tools.experiments.tmtaff_complete_fit.runtime.write_json',
                      side_effect=OSError('preparation storage error')), \
                redirect_stderr(stderr), redirect_stdout(stdout):
            self.assertEqual(main(args), 1)
        self.assertIn('FAILED OSError: preparation storage error', stderr.getvalue())
        self.assertNotIn('COMPLETED', stdout.getvalue())


if __name__ == '__main__':
    unittest.main()
