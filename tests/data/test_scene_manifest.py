import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from association.data.archive import DataError, MTADArchive, SPLIT_NAMES, build_scene_manifest, validate_manifest
from association.data.scene_manifest import build_fresh_scene_manifest
from association.data.motion_evidence import prepare_motion_scenes
from association.data.native import load_native_split
from association.data.reference_states import read_reference_states
from tests.fixtures.native_zip import NativeZipFixture, csv_bytes
from tools.experiments.make_scene_manifest import main


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


class FreshSceneManifestTests(unittest.TestCase):
    caps = dict(T_core=8, T_head=0, V_select=3, T_test=5)

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.archive_path = self.root / 'level1.zip'
        with zipfile.ZipFile(self.archive_path, 'w') as archive:
            for sid in range(20):
                # Every scene shares a source A; that must be disclosed, not
                # silently turned into identity-isolated selection.
                labels = csv_bytes(['t_s', 't_e', 'mmsi', 'source', 'batch'],
                                   [[0, 20, 'shared-0-' + str(sid), '9001', '0']])
                archive.writestr(f'Level1/关联表/关联结果-{sid}.csv', labels)
                for directory in ('量测场景', '真实场景'):
                    archive.writestr(f'Level1/{directory}/场景-{sid}.csv', b'MUST NOT BE READ')

    def build(self, **kwargs):
        with MTADArchive(self.archive_path) as archive:
            result = build_fresh_scene_manifest(archive, caps=self.caps, **kwargs)
            self.assertEqual({entry['kind'] for entry in archive.members_read}, {'labels'})
            self.assertEqual(len(archive.members_read), 20)
            self.assertEqual(archive._label_cache, {})
            return result

    def test_fresh_label_only_selection_retains_real_dependence_and_preview(self):
        original = zipfile.ZipFile.read
        reads = []

        def read(archive, member, *args, **kwargs):
            self.assertIn('关联表', member.filename)
            reads.append(member.filename)
            return original(archive, member, *args, **kwargs)

        with patch.object(zipfile.ZipFile, 'read', read):
            result = self.build()
        self.assertEqual(len(reads), 20)
        self.assertEqual(result, self.build())
        self.assertNotEqual(result['splits'], self.build(seed=21)['splits'])
        self.assertEqual({split: len(group) for split, group in result['splits'].items()}, self.caps)
        self.assertIn('0', result['splits']['T_core'])
        self.assertEqual(sum(role == 'unused' for role in result['scene_splits'].values()), 4)
        self.assertEqual(len(result['component_members']), 1)
        self.assertEqual(result['dependence']['selected_components_crossing_splits'], 1)
        self.assertEqual(set(result['dependence']['shared_A_counts'].values()), {1})
        self.assertFalse(result['dependence']['identity_independent'])
        self.assertEqual(result['dataset_level'], 'Level1')

    def test_fresh_preview_cannot_be_moved_to_validation(self):
        result = self.build()
        other = result['splits']['V_select'][0]
        result['splits']['T_core'].remove('0')
        result['splits']['T_core'].append(other)
        result['splits']['V_select'].remove(other)
        result['splits']['V_select'].append('0')
        result['scene_splits'].update({'0': 'V_select', other: 'T_core'})
        with self.assertRaisesRegex(DataError, 'Previewed development'):
            validate_manifest(result)

    def test_level_binding_uses_member_directories_and_rejects_before_payload(self):
        misleading_name = self.root/'level2.zip'
        self.archive_path.rename(misleading_name)
        self.archive_path = misleading_name
        # The actual internal Level1 declaration wins over the external name.
        manifest = self.build()
        with zipfile.ZipFile(self.archive_path) as archive:
            contents = [(info.filename, archive.read(info)) for info in archive.infolist()]
        wrong_archive = self.root/'claims-level1.zip'
        for case in ('Level2', 'unknown', 'mixed'):
            with zipfile.ZipFile(wrong_archive, 'w') as archive:
                for name, content in contents:
                    directory = ('Level2' if name.endswith('-0.csv') else 'Level1') if case == 'mixed' else case
                    archive.writestr(name.replace('Level1/', directory+'/', 1), content)
            with self.subTest(case=case), MTADArchive(wrong_archive) as archive, \
                    patch.object(zipfile.ZipFile, 'read', side_effect=AssertionError('No payload before level match')):
                self.assertEqual(archive.dataset_level, 'Level2' if case == 'Level2' else None)
                with self.assertRaisesRegex(DataError, 'Level1 member directories'):
                    build_fresh_scene_manifest(archive, caps=self.caps)
                with self.assertRaisesRegex(DataError, 'dataset level'):
                    load_native_split(wrong_archive, manifest, 'T_core')
                with self.assertRaisesRegex(DataError, 'dataset level'):
                    prepare_motion_scenes(wrong_archive, manifest, 'T_core')
                with self.assertRaisesRegex(DataError, 'dataset level'):
                    read_reference_states(wrong_archive, manifest, split='T_core',
                                          scene_ids=manifest['splits']['T_core'])

    def test_cli_accepts_fresh_archive_and_preserves_existing_output(self):
        output = self.root / 'manifest.json'
        args = ['--archive', str(self.archive_path), '--output', str(output),
                '--train-scenes', '8', '--validation-scenes', '3', '--test-scenes', '5']
        with patch('builtins.print'):
            main(args)
        result = json.loads(output.read_text())
        self.assertEqual(result['splits'], self.build()['splits'])
        before = output.read_bytes()
        with patch('sys.stderr'), self.assertRaises(SystemExit):
            main(args)
        self.assertEqual(output.read_bytes(), before)


if __name__ == '__main__':
    unittest.main()
