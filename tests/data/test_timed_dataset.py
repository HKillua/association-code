from tests.fixtures.native_zip import NativeZipFixture
import unittest
from unittest.mock import patch
import zipfile

import numpy as np
import torch

from association.utils import runtime
from association.data.archive import DataError, MTADArchive
from association.models.tmtaff import ModelConfig
from association.models.fusion import PairFusionConfig


class TimedDatasetTests(NativeZipFixture, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)


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
                    patch('association.data.dataset.MTADArchive', side_effect=AssertionError('No archive opened')):
                with self.assertRaises(DataError):
                    self.load(**changes)

    def test_only_declared_members_read_and_reference_consumed_before_next_scene(self):
        from association.data.dataset import build_scene_fusion_targets
        events = []
        original_read = zipfile.ZipFile.read
        def read(archive, member, *args, **kwargs):
            events.append(('read', member.filename))
            return original_read(archive, member, *args, **kwargs)
        def build(scene, *args, **kwargs):
            events.append(('build', scene['scene_id']))
            return build_scene_fusion_targets(scene, *args, **kwargs)
        with patch.object(zipfile.ZipFile, 'read', read), \
                patch('association.data.dataset.build_scene_fusion_targets', side_effect=build), \
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
                    patch('association.data.dataset.iter_reference_states', side_effect=AssertionError('No reference read')):
                with self.assertRaisesRegex(DataError, 'query grid|positive duration'):
                    self.load()


    def test_real_zip_loader_to_four_training_stages_and_full_validation(self):
        from association.training.tmtaff import train_complete
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


if __name__ == '__main__':
    unittest.main()
