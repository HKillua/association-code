"""Software checks for 105; synthetic fixtures do not certify MTAD evidence."""
from argparse import Namespace
import csv
import io
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile

from association.data.archive import HEADERS
from association.data.reference_states import REFERENCE_HEADER, _read_rows
from tests.evaluation.test_three_case_chain import material
from tools.analysis.audit_three_case_chain import audit_case
from tools.analysis.audit_raw_case_chain import audit_scene, raw_rows, run
from tools.analysis.diagnose_three_scenes import read, write


def encode(header, rows):
    out = io.StringIO(newline='')
    writer = csv.DictWriter(out, fieldnames=header, lineterminator='\n')
    writer.writeheader(); writer.writerows(rows)
    return out.getvalue().encode('utf-8-sig')


def example(root):
    scene, public, private = material(root/'fixture')
    case = audit_case(scene, public, private, 0, 1)
    labels, measurements = [], []
    for track in private['input_tracks']:
        labels.append(dict(t_s='0', t_e='20', mmsi=track['identity_id']+'-segment',
                           source=track['source'], batch=track['batch']))
        for timestamp, state in zip(track['canonical_times'], track['canonical_values']):
            measurements.append(dict(batch=track['batch'], source=track['source'], time=str(timestamp),
                lat=str(state[0]), lon=str(state[1]), vel=str(state[2]), cou=str(state[3])))
    # Exact duplicate and reverse ordering must reconstruct the same canonical sequence.
    measurements = list(reversed(measurements)); measurements.append(dict(measurements[0]))
    refs = [row['raw'] for row in private['raw_reference']['rows']]
    payloads = dict(labels=encode(HEADERS['labels'], labels),
                    measurements=encode(HEADERS['measurements'], measurements),
                    reference=encode(REFERENCE_HEADER, refs))
    return private, case, payloads, measurements


class RawCaseChainTests(unittest.TestCase):
    def test_raw_rows_retain_original_strings_duplicate_and_order(self):
        rows, encoding = raw_rows(encode(HEADERS['measurements'], [dict(batch='1', source='2', time=' 5 ',
            lat='30', lon='120', vel='2', cou='90')]), HEADERS['measurements'])
        self.assertEqual(encoding, 'utf-8-sig')
        self.assertEqual(rows[0]['raw']['time'], ' 5 ')
        self.assertEqual(rows[0]['stripped']['time'], '5')
        self.assertEqual(rows[0]['line'], 2)

    def test_course_360_normalizes_only_after_same_time_conflict_detection(self):
        from tools.analysis.audit_raw_case_chain import node_check
        labels, _ = raw_rows(encode(HEADERS['labels'], [dict(t_s='0', t_e='10', mmsi='A-0-x',
                              source='9', batch='b')]), HEADERS['labels'])
        values = [dict(batch='b', source='9', time='0', lat='30', lon='120', vel='2', cou='360'),
                  dict(batch='b', source='9', time='10', lat='31', lon='120', vel='2', cou='90')]
        measurements, _ = raw_rows(encode(HEADERS['measurements'], values), HEADERS['measurements'])
        track = dict(node_index=0, node_id='3035:9:b', source='9', batch='b', identity_id='A-0',
                     canonical_times=[0., 10.], canonical_values=[[30., 120., 2., 0.], [31., 120., 2., 90.]])
        out, _ = node_check('3035', track, labels, measurements, [0, 1], 0., 'A-0-x')
        self.assertTrue(out['full_canonical_exact'])
        self.assertTrue(out['reader_would_keep_node'])
        values.append(dict(values[0], cou='0'))
        measurements, _ = raw_rows(encode(HEADERS['measurements'], values), HEADERS['measurements'])
        out, _ = node_check('3035', track, labels, measurements, [0, 1], 0., 'A-0-x')
        self.assertFalse(out['reader_would_keep_node'])
        self.assertEqual(out['conflicting_time_count'], 1)

    def test_103_104_schemas_reconstruct_selected_nodes_and_preserve_mismatch(self):
        with TemporaryDirectory() as temp:
            private, case, payloads, raw = example(Path(temp))
            public, secret = audit_scene('3035', 0, 1, payloads, private, case)
            self.assertEqual(public['mismatch_count'], 0)
            self.assertTrue(all(x['full_canonical_exact'] for x in public['nodes']))
            self.assertTrue(public['reference']['all_reference_raw_rows_match_103'])
            self.assertNotIn('A-0', str(public))
            self.assertNotIn('canonical_values', str(public))
            self.assertNotIn('source', public['nodes'][0])
            self.assertNotIn('batch', public['nodes'][0])
            self.assertIn('A-0', str(secret))
            # Alter a selected source's raw value. The mismatch survives as data,
            # rather than being silently repaired or dropping the case.
            raw[next(i for i, item in enumerate(raw) if item['batch'] == '0')]['lat'] = '31.0'
            payloads['measurements'] = encode(HEADERS['measurements'], raw)
            changed, _ = audit_scene('3035', 0, 1, payloads, private, case)
            self.assertGreater(changed['mismatch_count'], 0)
            self.assertEqual(len(changed['nodes']), 2)

    def test_source_1043_q55_reference_endpoints_and_neighbors_are_included(self):
        from tools.analysis.diagnose_three_scenes import brackets
        rows = [dict(MMSI='T-0', time=str(t), lat=str(30+t*.001), lon='120', vel='2', cou='90')
                for t in range(100, 171, 10)]
        payload = encode(REFERENCE_HEADER, rows)
        parsed, issues = _read_rows(payload, 'fixture')
        reference = dict(rows=parsed)
        ends = brackets(reference, 'T-0', list(range(64)), dict(time_offset_seconds=100., max_gap_seconds=600.))
        private = dict(reference_endpoints=[dict(target_index=3, target_identity='T-0', endpoints=ends,
                                                alignment_report=dict(time_offset_seconds=100.))],
                       raw_reference=reference)
        from tools.analysis.audit_raw_case_chain import reference_check
        public, secret = reference_check(payload, private, 3,
            dict(rows=[dict(query_index=54, query_seconds=54.)]), 54, '1043')
        self.assertEqual(public['queries'], [54, 55])
        self.assertTrue(all(x['matches'] for x in public['endpoint_checks']))
        self.assertGreaterEqual(len(public['selected_neighbor_times']), 4)
        self.assertTrue(all(row['previous_selected_is_immediate_original_time_neighbor'] is not None
                            for row in public['selected_neighbor_times'][1:]))
        self.assertNotIn('T-0', str(public))
        self.assertEqual(secret['target_identity'], 'T-0')

    def test_reference_step_is_only_between_immediate_original_times_and_marks_dateline(self):
        from tools.analysis.diagnose_three_scenes import brackets
        from tools.analysis.audit_raw_case_chain import reference_check
        rows = [dict(MMSI='T-0', time=str(t), lat='30', lon=('-179' if t >= 110 else '179'), vel='2', cou='90')
                for t in range(100, 171, 10)]
        payload = encode(REFERENCE_HEADER, rows)
        parsed, _ = _read_rows(payload, 'fixture')
        ref = dict(rows=parsed)
        ends = brackets(ref, 'T-0', list(range(64)), dict(time_offset_seconds=100., max_gap_seconds=600.))
        private = dict(reference_endpoints=[dict(target_index=3, target_identity='T-0', endpoints=ends,
                                                alignment_report=dict(time_offset_seconds=100.))], raw_reference=ref)
        public, _ = reference_check(payload, private, 3,
            dict(rows=[dict(query_index=1, query_seconds=1.)]), 1, '1043')
        steps = public['selected_neighbor_times']
        crossing = next(row for row in steps if row['previous_adjacent_lon_crosses_180_degrees'])
        self.assertEqual(crossing['previous_adjacent_delta_lon_raw_degrees'], -358.)
        self.assertEqual(crossing['previous_adjacent_delta_lon_short_arc_degrees'], 2.)
        gap = next(row for row in steps if row['previous_selected_is_immediate_original_time_neighbor'] is False)
        self.assertIsNone(gap['previous_selected_position_step_m'])
        self.assertIsNone(gap['previous_adjacent_delta_lat_degrees'])

    def test_one_scene_archive_reads_only_three_fixed_members_and_preserves_output(self):
        with TemporaryDirectory() as temp:
            root = Path(temp); private, case, payloads, _ = example(root)
            archive = root/'input.zip'
            with zipfile.ZipFile(archive, 'w') as z:
                for kind, name in (('labels', '关联表/关联结果'), ('measurements', '量测场景/场景'),
                                   ('reference', '真实场景/场景')):
                    z.writestr(f'数据集level1/{name}-3035.csv', payloads[kind])
                    z.writestr(f'数据集level1/{name}-3.csv', b'SEALED TEST PAYLOAD MUST NEVER BE READ')
            source103, source104 = root/'103', root/'104'; source103.mkdir(); source104.mkdir()
            write(source103/'run.json', dict(status='COMPLETED', test_opened=False,
                                              completed_scenes=['3035'], model_forwards=0))
            write(source104/'run.json', dict(status='COMPLETED', test_opened=False,
                                              completed_cases=['3035'], consistency_mismatch_count=0))
            write(source103/'private-evidence-3035.json.gz', private)
            write(source104/'case-3035.json.gz', case)
            write(root/'manifest.json', dict(splits=dict(V_select=['3035'])))
            args = Namespace(archive=[archive], manifest=root/'manifest.json', source103=source103,
                             source104=source104, output=root/'output')
            with patch('tools.analysis.audit_raw_case_chain.CASES', (('3035', 0, 1),)), \
                    patch('tools.analysis.audit_raw_case_chain.QUERY_COUNT', 3), \
                    patch('tools.analysis.audit_raw_case_chain.validate_manifest'):
                run(args)
                with self.assertRaises(FileExistsError):
                    run(args)
                with self.assertRaisesRegex(ValueError, 'overlap'):
                    run(Namespace(**{**vars(args), 'output': source103/'nested'}))
            result = read(root/'output'/'run.json')
            self.assertEqual(result['status'], 'COMPLETED')
            self.assertEqual(len(result['read_members']), 3)
            self.assertEqual({row['scene_id'] for row in result['read_members']}, {'3035'})
            self.assertEqual(result['optimizer_updates'], 0)
            self.assertEqual(read(root/'output'/'summary.json.gz')['mismatch_count'], 0)
            self.assertTrue((root/'output'/'private'/'case-3035.json.gz').is_file())

    def test_second_member_failure_preserves_first_member_read_journal(self):
        from tools.analysis.audit_raw_case_chain import read_members
        info = lambda name: SimpleNamespace(filename=name, compress_size=10, file_size=20)
        first, second, third = info('a'), info('b'), info('c')
        def read_member(item):
            if item is second:
                raise OSError('read failure')
            return b'payload'
        archive = SimpleNamespace(_members={'labels': {'3035': (first, 'labels')},
            'measurements': {'3035': (second, 'measurements')}}, _zip=SimpleNamespace(read=read_member))
        journal, durable = [], []
        with patch('tools.analysis.audit_raw_case_chain.reference_member', return_value=(third, 'reference')):
            with self.assertRaises(OSError):
                read_members(archive, '3035', journal, lambda: durable.append([x['status'] for x in journal]))
        self.assertEqual([row['status'] for row in journal], ['READ', 'PENDING'])
        self.assertEqual(durable, [['PENDING'], ['READ'], ['READ', 'PENDING']])
        self.assertEqual(journal[0]['actual_uncompressed_bytes'], 7)

    def test_linux_memory_limit_never_relaxes_inherited_process_limit(self):
        import resource
        from tools.analysis.audit_raw_case_chain import limit_address_space
        inherited = 2 * 1024**3
        with patch('tools.analysis.audit_raw_case_chain.sys.platform', 'linux'), \
             patch('tools.analysis.audit_raw_case_chain.resource.getrlimit', return_value=(inherited, resource.RLIM_INFINITY)), \
             patch('tools.analysis.audit_raw_case_chain.resource.setrlimit') as setter:
            result = limit_address_space()
        self.assertEqual(result['address_space_limit_bytes'], inherited)
        setter.assert_called_once_with(resource.RLIMIT_AS, (inherited, resource.RLIM_INFINITY))

    def test_oversized_member_is_journaled_before_payload_read(self):
        from tools.analysis.audit_raw_case_chain import read_members, MAX_UNCOMPRESSED_MEMBER_BYTES
        huge = SimpleNamespace(compress_size=10, file_size=MAX_UNCOMPRESSED_MEMBER_BYTES+1)
        normal = SimpleNamespace(compress_size=10, file_size=20)
        archive = SimpleNamespace(_members={'labels': {'3035': (huge, 'labels')},
            'measurements': {'3035': (normal, 'measurements')}}, _zip=SimpleNamespace(read=lambda x: self.fail('must not read')))
        journal, durable = [], []
        with patch('tools.analysis.audit_raw_case_chain.reference_member', return_value=(normal, 'reference')):
            with self.assertRaisesRegex(ValueError, 'byte ceiling'):
                read_members(archive, '3035', journal, lambda: durable.append(journal[-1]['status']))
        self.assertEqual(durable, ['PENDING', 'REJECTED_SIZE_LIMIT'])
        self.assertNotIn('actual_uncompressed_bytes', journal[0])


if __name__ == '__main__':
    unittest.main()
