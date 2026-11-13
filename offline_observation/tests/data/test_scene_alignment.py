import copy
import unittest
from unittest.mock import patch
import zipfile
from types import SimpleNamespace

import numpy as np

from association.data.alignment import (
    TEMPORAL_CLOCK_POLICY, derive_scene_alignment, prepare_scene_data, observation_profile,
)
from association.data.archive import MTADArchive, build_scene_manifest
from association.data.motion_evidence import prepare_motion_scenes
from association.data.native import iter_native_split_from_archive, load_native_split_from_archive
from association.data.reference_states import read_reference_states
from tests.fixtures.native_zip import NativeZipFixture


class SceneAlignmentTests(NativeZipFixture, unittest.TestCase):
    def clock_fixture(self):
        with MTADArchive(self.path) as archive:
            scene = archive.load_scene('0')
            labels = archive.read_labels('0')
            measurements = archive._rows('0', 'measurements')
        reference = read_reference_states(self.path, self.manifest, split='T_core', scene_ids=['0'])[0]
        return scene, labels, measurements, reference

    def test_temporal_policy_handles_noisy_states_and_nonzero_level1_offsets(self):
        scene, labels, measurements, original = self.clock_fixture()
        before = [(track.times.copy(), track.values.copy()) for track in scene.tracks]
        for offset in (3., 22.):
            reference = copy.deepcopy(original)
            for row in reference['rows']:
                row['time'] += offset-1000.
                row['state'] = [value+.123 for value in row['state']]
            with self.subTest(offset=offset):
                exact = derive_scene_alignment(scene, labels, measurements, reference)
                self.assertEqual(exact['status'], 'CLOCK_UNRESOLVED')
                clock = derive_scene_alignment(scene, labels, measurements, reference,
                                               clock_policy=TEMPORAL_CLOCK_POLICY)
                self.assertEqual(clock['status'], 'CLOCK_HYPOTHESIS_SUPPORTED')
                self.assertEqual(clock['time_offset_seconds'], offset)
                self.assertEqual(clock['feasible_offset_range_seconds'], [offset, offset])
                self.assertEqual(clock['matches']['inside_reference_bounds_fragments'], len(scene.tracks))
                self.assertGreater(clock['matches']['reference_timestamp_endpoint_matches'], 0)
        for track, (times, values) in zip(scene.tracks, before):
            np.testing.assert_array_equal(track.times, times)
            np.testing.assert_array_equal(track.values, values)

    def test_temporal_conflicts_fail_without_reselecting_offset(self):
        scene, labels, measurements, reference = self.clock_fixture()
        changed = copy.deepcopy(labels)
        changed[0]['t_s'] += 1.
        clock = derive_scene_alignment(scene, changed, measurements, reference,
                                       clock_policy=TEMPORAL_CLOCK_POLICY)
        self.assertEqual(clock['status'], 'CLOCK_UNRESOLVED')
        self.assertGreater(clock['matches']['label_boundary_mismatch_fragments'], 0)
        extra = copy.deepcopy(reference['rows'][0])
        extra['time'] = 999.
        reference['rows'].append(extra)
        clock = derive_scene_alignment(scene, labels, measurements, reference,
                                       clock_policy=TEMPORAL_CLOCK_POLICY)
        self.assertEqual(clock['status'], 'CLOCK_UNRESOLVED')
        self.assertEqual(clock['time_offset_seconds'], 999.)
        self.assertEqual(clock['feasible_offset_range_seconds'], [1000., 1000.])
        self.assertGreater(clock['matches']['outside_reference_bounds_fragments'], 0)

    def test_temporal_preparation_records_evidence_keeps_alignment_schema_and_sealed_test(self):
        self.manifest['splits']['V_select'] = ['2']
        original_read = zipfile.ZipFile.read
        reads = []

        def read(archive, member, *args, **kwargs):
            reads.append(member.filename)
            self.assertFalse(member.filename.endswith(('-1.csv', '-3.csv', '-4.csv')))
            return original_read(archive, member, *args, **kwargs)

        with patch.object(zipfile.ZipFile, 'read', read):
            alignment, summary = prepare_scene_data(self.path, self.manifest,
                                                     clock_policy=TEMPORAL_CLOCK_POLICY)
        self.assertEqual(summary['status'], 'COMPLETED')
        self.assertEqual(summary['clock_policy'], TEMPORAL_CLOCK_POLICY)
        self.assertEqual(set(summary['scene_clocks']), {'0', '2'})
        self.assertEqual(summary['scene_clocks']['0']['feasible_offset_range_seconds'], [1000., 1000.])
        self.assertEqual(alignment, {sid: dict(time_offset_seconds=1000., max_gap_seconds=600.) for sid in ('0', '2')})
        self.assertEqual(len(reads), 6)
        self.assertFalse(summary['test_opened'])

    def test_dense_observation_profile_distinguishes_raw_from_capped_intervals(self):
        scene = SimpleNamespace(tracks=[SimpleNamespace(source='sensor', times=np.arange(300)*10.)])
        profile = observation_profile(scene)['sensor']
        self.assertEqual(profile['canonical_points'], 300)
        self.assertEqual(profile['selected_points'], 128)
        self.assertEqual(profile['canonical_interval_le_10s'], 299)
        self.assertEqual(profile['selected_interval_count'], 127)
        self.assertEqual(profile['canonical_interval_seconds_max'], 10.)
        self.assertEqual(profile['selected_interval_seconds_max'], 30.)
        self.assertEqual(profile['supported_queries'], 64)
        self.assertEqual(profile['selected_bidirectional_transitions'], 254)

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
        cold_alignment, cold_summary = prepare_scene_data(self.path, self.manifest)
        self.assertEqual(cold_alignment, expected)
        self.assertEqual(cold_summary['historical_alignment_scenes'], 0)
        self.assertEqual(cold_summary['population']['T_core']['tcm_ordered_batches_per_epoch_at_batch64'], 1)

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
