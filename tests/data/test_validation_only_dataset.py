"""Validation-only preparation must preserve targets without opening train/test."""
from copy import deepcopy
from pathlib import PurePosixPath
import unittest
from unittest.mock import patch
import zipfile

import numpy as np

from association.data.archive import DataError
from tests.fixtures.native_zip import NativeZipFixture


class ValidationOnlyDatasetTests(NativeZipFixture, unittest.TestCase):
    def setUp(self):
        super().setUp()
        # Exercise the Level1 directory binding on the same real CSV fixture,
        # including one missing-input target and an entirely quarantined scene.
        with zipfile.ZipFile(self.path) as archive:
            members = [(info.filename.replace('level2', 'level1'), archive.read(info))
                       for info in archive.infolist()]
        with zipfile.ZipFile(self.path, 'w') as archive:
            for name, payload in members:
                archive.writestr(name, payload)
        self.manifest['dataset_level'] = 'Level1'

    def test_same_validation_inputs_and_full_target_population_as_default_loader(self):
        original_manifest = deepcopy(self.manifest)
        complete = self.load(motion_observation_policy='canonical_all')
        validation = self.load(validation_only=True,
            alignment_by_scene={sid: self.alignment[sid] for sid in ('2', '4')},
            motion_observation_policy='canonical_all')
        self.assertEqual(self.manifest, original_manifest)
        self.assertEqual(validation['train'], [])
        self.assertEqual(validation['report']['training_splits'], [])
        self.assertEqual(validation['report']['training_scene_ids'], [])
        self.assertEqual(validation['report']['validation_scene_ids'], ['2', '4'])
        self.assertEqual([scene['scene_id'] for scene in complete['train']], ['0'])
        for expected, actual in zip(complete['validation'], validation['validation']):
            self.assertEqual(expected['node_ids'], actual['node_ids'])
            self.assertEqual(expected['target_identity_ids'], actual['target_identity_ids'])
            for field in ('grid', 'native_values', 'native_observed', 'pairs', 'labels'):
                np.testing.assert_array_equal(expected[field], actual[field])
            np.testing.assert_array_equal(expected['motion']['states'], actual['motion']['states'])
            self.assertEqual(set(expected['fusion_targets']), set(actual['fusion_targets']))
            for identity, target in expected['fusion_targets'].items():
                for field in ('query_times', 'states', 'valid'):
                    np.testing.assert_array_equal(target[field], actual['fusion_targets'][identity][field])
                self.assertEqual(target['report'], actual['fusion_targets'][identity]['report'])
        nonempty, empty = validation['validation']
        self.assertIn('target2c-0', nonempty['fusion_targets'])
        self.assertNotIn('target2c-0', nonempty['identities'])
        self.assertEqual(len(empty['node_ids']), 0)
        self.assertEqual(list(empty['fusion_targets']), ['target4a-0'])
        self.assertTrue(empty['fusion_targets']['target4a-0']['valid'].all())
        self.assertEqual(validation['report']['scenes'],
                         [scene for scene in complete['report']['scenes'] if scene['split'] == 'V_select'])

    def test_unreadable_training_and_test_payloads_are_never_opened(self):
        with zipfile.ZipFile(self.path) as archive:
            members = [(info.filename, archive.read(info)) for info in archive.infolist()]
        with zipfile.ZipFile(self.path, 'w') as archive:
            for name, payload in members:
                sid = PurePosixPath(name).stem.rsplit('-', 1)[1]
                archive.writestr(name, payload if sid in ('2', '4') else b'FORBIDDEN PAYLOAD')
        reads = []
        original_read = zipfile.ZipFile.read

        def guarded_read(archive, member, *args, **kwargs):
            name = member.filename if isinstance(member, zipfile.ZipInfo) else member
            sid = PurePosixPath(name).stem.rsplit('-', 1)[1]
            self.assertIn(sid, ('2', '4'), 'training/test payload was opened')
            reads.append(name)
            return original_read(archive, member, *args, **kwargs)

        with patch.object(zipfile.ZipFile, 'read', guarded_read):
            result = self.load(validation_only=True, train_splits=['T_core', 'T_head'],
                alignment_by_scene={sid: self.alignment[sid] for sid in ('2', '4')})
        expected = {f'数据集level1/{directory}/{stem}-{sid}.csv'
                    for sid in ('2', '4') for directory, stem in
                    (('量测场景', '场景'), ('关联表', '关联结果'), ('真实场景', '场景'))}
        self.assertEqual(set(reads), expected)
        self.assertEqual(result['report']['training_splits'], [])
        self.assertFalse(result['report']['test_opened'])
        # The default still consumes training data; it must fail on this ZIP.
        with self.assertRaises(DataError):
            self.load()


if __name__ == '__main__':
    unittest.main()
