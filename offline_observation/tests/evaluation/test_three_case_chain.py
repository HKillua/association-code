"""104 software checks only; synthetic data do not certify MTAD or cause."""
from argparse import Namespace
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from tests.evaluation.test_three_scene_diagnosis import fixture
from tools.analysis.diagnose_three_scenes import analyze_scene, clean, read, read_cache, write
from tools.analysis.audit_three_case_chain import audit_case, native_position, position_at, run


def material(root):
    prepared, targets, reference, policy = fixture(root)
    reference['scene_id'] = '3035'
    for row in reference['rows']:
        row['raw'].update(dict(zip(('lat', 'lon', 'vel', 'cou'), map(str, row['state']))))
    prepared['audit']['valid_raw_mmsi'] = {track['node_id']: track['identity_id']+'-segment'
                                         for track in prepared['tracks']}
    return analyze_scene(prepared, targets, reference, policy, read_cache(root, '3035'), {})


class ThreeCaseChainTests(unittest.TestCase):
    def test_independent_reference_preserves_duplicate_conflict_offset_and_gap(self):
        def row(line, time, latitude, longitude):
            return dict(line=line, time=time, identity_id='A-0', state=[latitude, longitude, 2., 90.])
        reference = dict(rows=[row(2, 100., 0., 179.), row(3, 110., 10., -179.)])
        policy = dict(time_offset_seconds=100., max_gap_seconds=10.)
        value, detail, _ = position_at(reference, 'A-0', 5., policy)
        np.testing.assert_allclose(value, [5., -180.])
        self.assertEqual(detail['alpha'], .5)
        self.assertEqual(detail['endpoints'][0]['raw_time_seconds'], 100.)
        conflict = deepcopy(reference)
        conflict['rows'].append(row(4, 100., 1., 179.))
        value, detail, _ = position_at(conflict, 'A-0', 5., policy)
        self.assertTrue(np.isnan(value[0]))
        self.assertEqual(value[1], -180.)
        self.assertEqual(detail['endpoints'][0]['conflicting_fields'], ['lat'])
        missing = deepcopy(reference)
        missing['rows'][0]['state'][0] = None
        _, detail, _ = position_at(missing, 'A-0', 0., policy)
        self.assertEqual(detail['endpoints'][0]['conflicting_fields'], [])
        self.assertEqual(detail['endpoints'][0]['unusable_fields'], ['lat'])
        value, detail, _ = position_at(reference, 'A-0', 5., dict(policy, max_gap_seconds=9.))
        self.assertTrue(np.isnan(value).all())
        self.assertEqual(detail['kind'], 'gap_exceeded')
        value, detail, _ = position_at(reference, 'A-0', -1., policy)
        self.assertTrue(np.isnan(value).all())
        self.assertEqual(detail['kind'], 'outside_support')

    def test_canonical_interpolation_and_endpoint_anchored_ols(self):
        times = np.array([0., 2., 4.])
        positions = np.array([[0., 179.], [2., -179.], [0., -177.]])
        np.testing.assert_allclose(native_position(times, positions, 1.), [1., -180.])
        # Mean latitude is not a valid extrapolation anchor: keep the last measured latitude.
        np.testing.assert_allclose(native_position(times, positions, 6.), [0., -175.], atol=1e-12)
        np.testing.assert_allclose(native_position(times, positions, 2.), positions[1])

    def test_real_103_schema_retains_neighbors_and_excludes_raw_identity_coordinates(self):
        with TemporaryDirectory() as temp:
            scene, public, private = material(Path(temp))
            case = audit_case(scene, public, private, 0, 1)
            self.assertEqual(case['query_indices'], [0, 1, 2])
            self.assertEqual(case['consistency_status'], 'MATCHES_WITHIN_TOLERANCE')
            self.assertEqual(case['failed_check_count'], 0)
            self.assertEqual(len(case['rows']), 3)
            self.assertEqual(len(case['rows'][1]['sources']), 2)
            self.assertTrue(case['rows'][1]['sources'][0]['identity_matches_bound_target'])
            self.assertEqual(case['rows'][1]['reference']['alpha'], 0.)
            serialized = str(clean(case))
            self.assertNotIn('A-0', serialized)
            self.assertNotIn('target_identity', serialized.replace('target_identity_parse_has_two_fields', ''))
            self.assertNotIn('latitude_origin_degrees', serialized)
            self.assertNotIn('canonical_values', serialized)

    def test_mismatches_are_recorded_and_do_not_swallow_case(self):
        with TemporaryDirectory() as temp:
            scene, public, private = material(Path(temp))
            public['query_sources'][0]['nearest_event_time'] += 1.
            private['raw_reference']['rows'][0]['raw']['lat'] = '39.0'
            case = audit_case(scene, public, private, 0, 1)
            self.assertEqual(case['consistency_status'], 'MISMATCH_FOUND')
            bad = {r['check'] for r in case['checks'] if not r['matches']}
            self.assertIn('reference_raw_numeric_parse', bad)
            self.assertIn('nearest_native_time', bad)
            self.assertEqual(len(case['rows']), 3)

    def test_run_preserves_numeric_outputs_when_plot_dependency_missing_and_never_overwrites(self):
        with TemporaryDirectory() as temp:
            root = Path(temp); source = root/'source'; source.mkdir()
            scene, public, private = material(root/'fixture')
            write(source/'run.json', dict(status='COMPLETED', completed_scenes=['3035'],
                test_opened=False, model_forwards=0, source_metadata=dict(config=dict(
                    data=dict(length=3, reference_longitude_policy='shortest_arc'),
                    motion_observation_policy='canonical_all'))))
            for name, obj in (('scene', scene), ('points', public), ('private-evidence', private)):
                write(source/f'{name}-3035.json.gz', obj)
            args = Namespace(source_run=str(source), output=str(root/'output'))
            with patch('tools.analysis.audit_three_case_chain.CASES', (('3035', 0, 1),)), \
                    patch('tools.analysis.audit_three_case_chain.QUERY_COUNT', 3), \
                    patch('tools.analysis.audit_three_case_chain.render_case', side_effect=ImportError('matplotlib')):
                run(args)
                with self.assertRaises(FileExistsError):
                    run(args)
            state = read(root/'output/run.json')
            self.assertEqual(state['status'], 'COMPLETED')
            self.assertFalse(state['plots_complete'])
            self.assertEqual(state['archive_files_read'], 0)
            self.assertEqual(state['model_forwards'], 0)
            self.assertEqual(len(read(root/'output/case-3035.json.gz')['rows']), 3)


if __name__ == '__main__':
    unittest.main()
