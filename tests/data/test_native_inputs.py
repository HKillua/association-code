import copy
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import numpy as np

from association.data.archive import DataError, MTADArchive
from association.data.native import load_native_split, load_native_split_from_archive


def frozen_fixture_manifest():
    splits = {'T_core': ['0'], 'T_head': ['1'], 'V_select': ['2'], 'T_test': ['3']}
    ids = ['0', '1', '2', '3']
    return {'splits': splits, 'all_scenes': ids, 'components': {sid: sid for sid in ids},
            'component_members': {sid: [sid] for sid in ids},
            'component_splits': {group[0]: split for split, group in splits.items()},
            'label_members': {sid: {} for sid in ids}, 'caps': {key: 1 for key in splits},
            'development_scenes': ['0']}


def csv_bytes(header, rows):
    stream = io.StringIO(newline='')
    writer = csv.writer(stream)
    writer.writerow(header)
    writer.writerows(rows)
    return stream.getvalue().encode('utf-8-sig')


def fragment(identity, source, batch, times):
    labels = [[times[0], times[-1], identity, source, batch]]
    values = [[batch, source, time, 20 + time / 100000, 130 - time / 200000, 5, 350]
              for time in times]
    return labels, values


class NativeInputsTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.path = Path(temp.name) / 'native.zip'
        self.manifest = frozen_fixture_manifest()

    def write_fixture(self, scene_zero):
        with zipfile.ZipFile(self.path, 'w') as archive:
            for sid in self.manifest['all_scenes']:
                labels, values = scene_zero if sid == '0' else fragment(
                    'other' + sid + '-0-0', '9001', '0', [0, 10])
                for folder, stem, header, rows in (
                    ('关联表', '关联结果', ['t_s', 't_e', 'mmsi', 'source', 'batch'], labels),
                    ('量测场景', '场景', ['batch', 'source', 'time', 'lat', 'lon', 'vel', 'cou'], values),
                ):
                    # A test-split parse would fail even if a read spy were absent.
                    content = b'SEALED TEST' if sid == '3' else csv_bytes(header, rows)
                    archive.writestr(f'数据集level2/{folder}/{stem}-{sid}.csv', content)
                archive.writestr(f'数据集level2/真实场景/场景-{sid}.csv', b'NEVER READ REFERENCE')

    def two_fragments(self):
        a, b = fragment('A-0-0', '9001', '0', [0, .1, .2, 2, 5, 100, 101, 200, 201, 400, 7200])
        c, d = fragment('A-0-1', '9002', '1', [6424, 6488])
        return a + c, b + d

    def test_short_fragment_survives_despite_empty_scene64_support(self):
        self.write_fixture(self.two_fragments())
        record = load_native_split(self.path, self.manifest, 'T_core', max_points=4)[0]
        short = next(track for track in record['tracks'] if track['source'] == '9002')
        grid = np.linspace(0, 7200, 64)
        self.assertEqual(int(((grid >= 6424) & (grid <= 6488)).sum()), 0)
        np.testing.assert_array_equal(short['times'], [6424, 6488])
        np.testing.assert_array_equal(short['selected_indices'], [0, 1])
        self.assertEqual(short['identity_id'], 'A-0')
        self.assertEqual(short['ancestor_id'], 'A')
        self.assertEqual(short['node_id'], '0:9002:1')
        np.testing.assert_array_equal(record['pairs'], [[0, 1]])
        np.testing.assert_array_equal(record['labels'], [1])
        self.assertEqual(record['audit']['native_inputs']['fragments_omitted_by_selection'], 0)
        self.assertNotIn('grid', record)

    def test_thinning_selects_exact_original_times_and_values_and_keeps_full_copy(self):
        self.write_fixture(self.two_fragments())
        original_archive = self.path.read_bytes()
        record = load_native_split(self.path, self.manifest, 'T_core', max_points=4)[0]
        track = record['tracks'][0]
        np.testing.assert_array_equal(track['selected_indices'], [0, 3, 6, 10])
        np.testing.assert_array_equal(track['times'], [0, 2, 101, 7200])
        np.testing.assert_array_equal(track['values'], track['canonical_values'][[0, 3, 6, 10]])
        self.assertEqual(len(track['canonical_times']), 11)
        self.assertEqual(track['values'].dtype, np.float64)
        self.assertEqual(track['times'].dtype, np.float64)
        self.assertEqual(self.path.read_bytes(), original_archive)
        audit = record['audit']['native_inputs']
        self.assertEqual((audit['canonical_points'], audit['selected_points'], audit['omitted_points']),
                         (13, 6, 7))
        self.assertEqual(audit['capped_fragments'], 1)
        self.assertGreater(audit['per_track'][0]['selected_max_gap_seconds'],
                           audit['per_track'][0]['canonical_max_gap_seconds'])
        json.dumps(record['audit'], allow_nan=False)


    def test_reads_only_selected_split_and_never_hashes_or_reads_reference(self):
        self.write_fixture(self.two_fragments())
        reads = []
        original_read = zipfile.ZipFile.read
        def read(archive, member, *args, **kwargs):
            reads.append(member.filename)
            return original_read(archive, member, *args, **kwargs)
        with patch.object(zipfile.ZipFile, 'read', read), \
                patch.object(MTADArchive, 'archive_sha256', side_effect=AssertionError('No hash gate')):
            records = load_native_split(self.path, self.manifest, 'V_select')
        self.assertEqual([row['scene_id'] for row in records], ['2'])
        self.assertEqual(reads, ['数据集level2/关联表/关联结果-2.csv',
                                 '数据集level2/量测场景/场景-2.csv'])
        self.assertEqual(records[0]['split'], 'V_select')
        self.assertEqual(records[0]['component_id'], '2')

    def test_sealed_split_and_invalid_cap_fail_before_archive_open(self):
        for split, cap in [('T_test', None), ('invalid', None), ('T_core', 1), ('T_core', True)]:
            with self.subTest(split=split, cap=cap), \
                    patch('association.data.native.MTADArchive', side_effect=AssertionError('No open')):
                with self.assertRaises(DataError):
                    load_native_split(self.path, self.manifest, split, max_points=cap)

    def test_manifest_and_real_inventory_are_validated_before_measurement_reads(self):
        self.write_fixture(self.two_fragments())
        bad = copy.deepcopy(self.manifest)
        bad['splits']['T_core'] = ['3']
        with MTADArchive(self.path) as archive:
            with patch.object(archive, '_rows', side_effect=AssertionError('No CSV read')):
                with self.assertRaises(DataError):
                    load_native_split_from_archive(archive, bad, 'T_core')
            with patch.object(archive, 'scene_ids', return_value=['0']):
                with self.assertRaisesRegex(DataError, 'Archive scenes'):
                    load_native_split_from_archive(archive, self.manifest, 'T_core')

    def test_existing_quarantine_is_reported_without_new_node_filtering(self):
        labels, points = self.two_fragments()
        more_labels, more_points = fragment('A-0-2', '9001', '2', [0, 10])
        more_points[0][3] = 'nan'
        self.write_fixture((labels + more_labels, points + more_points))
        record = load_native_split(self.path, self.manifest, 'T_core', max_points=2)[0]
        audit = record['audit']
        self.assertEqual((audit['candidate_nodes'], audit['valid_nodes'], audit['quarantined_nodes']), (3, 2, 1))
        self.assertEqual(audit['quarantine_reasons'],
                         {'fewer_than_two_distinct_times': 1, 'nonfinite_measurement': 1})
        self.assertEqual(audit['positive_edges_lost_to_quarantine'], 2)
        self.assertEqual(audit['native_inputs']['positive_pairs'], 1)
        self.assertEqual(audit['native_inputs']['fragments_preserved'], 2)

    def test_same_source_long_gap_positive_and_all_negative_candidates_survive(self):
        labels, points = [], []
        for identity, source, batch, times in [
                ('A-0-0', '9001', '0', [10, 20]),
                ('A-0-1', '9001', '1', [5000, 5010]),
                ('B-0-0', '9002', '2', [0, 7200])]:
            a, b = fragment(identity, source, batch, times)
            labels.extend(a)
            points.extend(b)
        self.write_fixture((labels, points))
        record = load_native_split(self.path, self.manifest, 'T_core', max_points=2)[0]
        np.testing.assert_array_equal(record['pairs'], [[0, 1], [0, 2], [1, 2]])
        np.testing.assert_array_equal(record['labels'], [1, 0, 0])
        self.assertEqual(record['intervals'][1, 0] - record['intervals'][0, 1], 4980)
        self.assertEqual(record['audit']['native_inputs']['fragments_preserved'], 3)

    def test_canonicalization_is_explicit_and_selection_does_not_crop_label_intervals(self):
        labels, points = fragment('A-0-0', '9001', '0', [0, 10])
        labels[0][:2] = [2, 8]
        points[0][-1] = 360
        self.write_fixture((labels, points + [points[0]]))
        record = load_native_split(self.path, self.manifest, 'T_core')[0]
        np.testing.assert_array_equal(record['tracks'][0]['canonical_times'], [0, 10])
        self.assertEqual(record['tracks'][0]['canonical_values'][0, 3], 0)
        self.assertEqual(record['audit']['exact_duplicate_rows_removed'], 1)
        self.assertEqual(record['audit']['measurement_rows_outside_label_interval_diagnostic_only'], 3)


    def test_all_quarantined_scene_keeps_empty_shapes_and_diagnostics(self):
        labels, points = fragment('A-0-0', '9001', '0', [0, 10])
        points[0][5] = -1
        self.write_fixture((labels, points))
        record = load_native_split(self.path, self.manifest, 'T_core')[0]
        self.assertEqual(record['tracks'], [])
        self.assertEqual(record['pairs'].shape, (0, 2))
        self.assertEqual(record['labels'].shape, (0,))
        self.assertEqual(record['intervals'].shape, (0, 2))
        self.assertIsNone(record['audit']['native_inputs']['point_retention_fraction'])
        self.assertEqual(record['audit']['quarantined_nodes'], 1)
        json.dumps(record['audit'], allow_nan=False)


if __name__ == '__main__':
    unittest.main()
