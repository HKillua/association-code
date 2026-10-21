import copy
import unittest
from unittest.mock import patch
import zipfile

import numpy as np

from association.data.alignment import derive_scene_alignment, prepare_scene_data
from association.data.archive import MTADArchive, build_scene_manifest
from association.data.motion_evidence import prepare_motion_scenes
from association.data.native import iter_native_split_from_archive, load_native_split_from_archive
from association.data.reference_states import read_reference_states
from tests.fixtures.native_zip import NativeZipFixture


class SceneAlignmentTests(NativeZipFixture, unittest.TestCase):
    def test_native_iterator_matches_list_without_reading_ahead(self):
        with MTADArchive(self.path) as archive:
            expected = load_native_split_from_archive(archive, self.manifest, 'V_select', max_points=4)
            expected_reads = list(archive.members_read)
        with MTADArchive(self.path) as archive:
            stream = iter_native_split_from_archive(archive, self.manifest, 'V_select', max_points=4)
            first = next(stream)
            self.assertEqual(first['scene_id'], '2')
            self.assertEqual({row['scene_id'] for row in archive.members_read}, {'2'})
            actual = [first, *stream]
            self.assertEqual(archive.members_read, expected_reads)
        np.testing.assert_equal(actual, expected)

    def test_streamed_population_and_old_clocks_agree_without_test_reads(self):
        self.manifest['splits']['V_select'] = ['2']  # empty scene 4 is not a clock fixture
        reads = []
        original = zipfile.ZipFile.read

        def read(archive, member, *args, **kwargs):
            reads.append(member.filename)
            return original(archive, member, *args, **kwargs)

        expected = {sid: dict(time_offset_seconds=1000., max_gap_seconds=600.) for sid in ('0', '2')}
        with patch.object(zipfile.ZipFile, 'read', read):
            alignment, summary = prepare_scene_data(self.path, self.manifest, expected)
        self.assertEqual(alignment, expected)
        self.assertEqual(summary['status'], 'COMPLETED')
        self.assertEqual(summary['historical_alignment_verified'], 2)
        self.assertEqual(summary['population']['T_core']['totals']['target_identities'], 3)
        self.assertEqual(summary['population']['T_core']['totals']['valid_fragments'], 4)
        self.assertEqual(summary['population']['T_core']['totals']['target_without_input_count'], 1)
        self.assertEqual(len(reads), 6)  # one label, measurement and reference per scene
        self.assertFalse(any(name.endswith(('-1.csv', '-3.csv', '-4.csv')) for name in reads))

    def test_contradictory_clock_and_changed_historical_offset_are_not_reselected(self):
        with MTADArchive(self.path) as archive:
            scene = archive.load_scene('0')
            labels = archive.read_labels('0')
            measurements = archive._rows('0', 'measurements')
        reference = read_reference_states(self.path, self.manifest, split='T_core', scene_ids=['0'])[0]
        extra = copy.deepcopy(reference['rows'][0])
        extra.update(time=999., state=[30., 130., 6., 90.])
        reference['rows'].append(extra)
        clock = derive_scene_alignment(scene, labels, measurements, reference)
        self.assertEqual(clock['time_offset_seconds'], 999.)
        self.assertEqual(clock['status'], 'CLOCK_UNRESOLVED')
        self.assertGreater(clock['matches']['unique_state_match_disagrees_fixed_offset'], 0)
        self.manifest['splits']['V_select'] = ['2']
        alignment, summary = prepare_scene_data(self.path, self.manifest,
            {'0': dict(time_offset_seconds=999., max_gap_seconds=600.)})
        self.assertEqual(summary['status'], 'FAILED')
        self.assertNotIn('0', alignment)
        self.assertTrue(summary['failures'][0]['historical_mismatch'])

    def test_explicit_scene_protocol_reaches_timed_and_motion_without_changing_legacy(self):
        legacy = self.load()
        self.assertNotIn('split_unit', legacy['train'][0])
        source = copy.deepcopy(self.manifest)
        source['splits']['V_select'] = ['2']
        source['components']['4'] = '0'
        source['component_members']['0'].append('4')
        del source['component_members']['4'], source['component_splits']['4']
        self.manifest = build_scene_manifest(source, 20, dict(T_core=2, T_head=0, V_select=2, T_test=1))
        timed = self.load()
        self.assertTrue(all(scene['split_unit'] == 'scene' for scene in timed['train']+timed['validation']))
        self.assertEqual(timed['train'][0]['component_id'], timed['validation'][1]['component_id'])
        for metadata in (False, True):
            motion = prepare_motion_scenes(self.path, self.manifest, 'T_core', include_evaluation_metadata=metadata)
            self.assertTrue(all(scene['split_unit'] == 'scene' for scene in motion))


if __name__ == '__main__':
    unittest.main()
