import copy
import unittest
from unittest.mock import patch
import zipfile

from association.data.archive import DataError, SPLIT_NAMES, build_scene_manifest, validate_manifest
from tests.fixtures.native_zip import NativeZipFixture


def metadata_fixture():
    ids = list(map(str, range(20)))
    groups = {str(i): [str(i), str(i + 10)] for i in range(10)}
    return dict(seed=1, protocol='synthetic historical metadata', all_scenes=ids,
        components={sid: cid for cid, members in groups.items() for sid in members},
        component_members=groups, component_splits={str(i): SPLIT_NAMES[i % 4] for i in range(10)},
        splits={split: [str(i)] for i, split in enumerate(SPLIT_NAMES)},
        caps=dict.fromkeys(SPLIT_NAMES, 1), label_members={sid: {} for sid in ids}, development_scenes=['10'])


class SceneManifestTests(unittest.TestCase):
    caps = dict(T_core=8, T_head=0, V_select=3, T_test=5)

    def test_metadata_only_reproducible_selection_preserves_exposure_and_real_components(self):
        source = metadata_fixture()
        before = copy.deepcopy(source)
        with patch('association.data.archive.MTADArchive', side_effect=AssertionError('No archive access')):
            result = build_scene_manifest(source, 20, self.caps)
            self.assertEqual(result, build_scene_manifest(source, 20, self.caps))
        self.assertEqual(source, before)
        self.assertEqual({split: len(scenes) for split, scenes in result['splits'].items()}, self.caps)
        self.assertTrue({'0', '1', '10'}.issubset(result['splits']['T_core']))
        self.assertIn('2', result['splits']['V_select'])
        self.assertTrue({'3', '7', '13', '17'}.issubset(result['splits']['T_test']))
        self.assertEqual(result['components'], source['components'])
        self.assertEqual(result['component_members'], source['component_members'])
        self.assertEqual(sum(role == 'unused' for role in result['scene_splits'].values()), 4)
        self.assertNotEqual(result['splits'], build_scene_manifest(source, 21, self.caps)['splits'])

    def test_historical_exposure_cannot_move_and_legacy_component_isolation_remains(self):
        source = metadata_fixture()
        result = build_scene_manifest(source, 20, self.caps)
        for first, sid, second in [('T_core', '0', 'T_test'), ('V_select', '2', 'T_core'), ('T_test', '7', 'V_select')]:
            changed = copy.deepcopy(result)
            other = changed['splits'][second][0]
            changed['splits'][first].remove(sid)
            changed['splits'][second].remove(other)
            changed['splits'][first].append(other)
            changed['splits'][second].append(sid)
            changed['scene_splits'].update({sid: second, other: first})
            with self.subTest(scene=sid), self.assertRaisesRegex(DataError, 'Historical scenes'):
                validate_manifest(changed)
        source['splits']['V_select'] = ['10']
        with self.assertRaisesRegex(DataError, 'Component crosses split boundary'):
            validate_manifest(source)


class SceneManifestLoaderTests(NativeZipFixture, unittest.TestCase):
    def test_native_loader_retains_shared_components_without_reading_sealed_test(self):
        source = copy.deepcopy(self.manifest)
        # Scene 4 was an unused member of the old training component. The new
        # scene protocol may place it in V while keeping the actual component.
        source['splits']['V_select'] = ['2']
        source['components']['4'] = '0'
        source['component_members']['0'].append('4')
        del source['component_members']['4'], source['component_splits']['4']
        self.manifest = build_scene_manifest(source, 20, dict(T_core=2, T_head=0, V_select=2, T_test=1))
        reads = []
        original = zipfile.ZipFile.read

        def read(archive, member, *args, **kwargs):
            reads.append(member.filename)
            return original(archive, member, *args, **kwargs)

        with patch.object(zipfile.ZipFile, 'read', read):
            result = self.load()
        self.assertEqual([scene['scene_id'] for scene in result['train']], ['0', '1'])
        self.assertEqual([scene['scene_id'] for scene in result['validation']], ['2', '4'])
        self.assertEqual(result['train'][0]['component_id'], result['validation'][1]['component_id'])
        self.assertFalse(any(name.endswith('-3.csv') for name in reads))
        self.assertFalse(result['report']['test_opened'])


if __name__ == '__main__':
    unittest.main()
