"""Native-only motion controls preserve short tracks, masks and data isolation."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import unittest
import zipfile

import numpy as np

from association.evaluation.native_motion import METHODS, predict_native_positions, position_metrics
from tests.fixtures.native_zip import make_native_zip_fixture
from tools.analysis.native_motion_control import main


class NativeMotionTests(unittest.TestCase):
    def test_constant_velocity_and_no_query_support_short_fragment(self):
        times = np.array([[0., 7., 20., 0.], [11., 12., 0., 0.]])
        visible = np.array([[True, True, True, False], [True, True, False, False]])
        states = np.zeros((2, 4, 4))
        states[..., 0] = 20.+times*.001
        states[..., 1] = 120.+times*.002
        query = np.array([-10., 0., 10., 20., 30., 40.])
        result = predict_native_positions(states, times, visible, query)
        expected = np.column_stack([20.+query*.001, 120.+query*.002])
        for name in METHODS[1:]:
            np.testing.assert_allclose(result['predictions'][name], np.broadcast_to(expected, (2, 6, 2)), atol=1e-12)
        self.assertFalse(result['regions']['inside_support'][1].any())
        self.assertEqual(result['native_point_counts'].tolist(), [3, 2])
        for name in METHODS[1:]:
            np.testing.assert_array_equal(result['predictions'][name][result['regions']['inside_support']],
                                          result['predictions']['native_hold'][result['regions']['inside_support']])
        self.assertGreater(abs(result['predictions']['native_ols'][1, 0, 0]
                               -result['predictions']['native_hold'][1, 0, 0]), .01)
        np.testing.assert_array_equal(result['regions']['all'],
            result['regions']['inside_support'] | result['regions']['before_support'] | result['regions']['after_support'])

    def test_dateline_slopes_exact_endpoints_and_latitude_clamp_are_reported(self):
        times = np.array([[0., 10., 20.], [0., 10., 20.]])
        positions = np.array([[[10., 179.], [10., 180.], [10., -179.]],
                              [[88., 0.], [89., 0.], [90., 0.]]])
        query = np.array([-10., 0., 5., 10., 20., 30.])
        result = predict_native_positions(positions, times, np.ones((2, 3), bool), query)
        for name in METHODS[1:]:
            np.testing.assert_allclose(result['predictions'][name][0, :, 1], [178., 179., 179.5, 180., -179., -178.])
            self.assertEqual(result['latitude_clamp_counts'][name].tolist(), [0, 1])
            self.assertEqual(result['predictions'][name][1, -1, 0], 90.)
        self.assertEqual(result['predictions']['native_hold'][0, 3, 1], 180.)
        # Exact half-turn direction follows the raw difference in both signs.
        half = predict_native_positions([[[0., 0.], [0., 180.]], [[0., 0.], [0., -180.]]],
            [[0., 10.], [0., 10.]], np.ones((2, 2), bool), [0., 5., 10.])
        np.testing.assert_array_equal(half['predictions']['native_hold'][:, 1, 1], [90., -90.])
        np.testing.assert_array_equal(half['native_half_turn_counts'], [1, 1])

    def test_hidden_padding_and_nonposition_states_do_not_change_predictions(self):
        values = np.array([[[20., 120., 5., 45.], [20.1, 120.1, 5., 45.], [0., 0., 0., 0.]]])
        times, visible = np.array([[0., 10., 0.]]), np.array([[True, True, False]])
        first = predict_native_positions(values, times, visible, [-5., 0., 5., 10., 15.])
        changed = values.copy()
        changed[..., 2:] = np.nan
        changed[:, 2] = np.nan
        times[:, 2] = np.nan
        second = predict_native_positions(changed, times, visible, [-5., 0., 5., 10., 15.])
        for name in METHODS:
            np.testing.assert_array_equal(first['predictions'][name], second['predictions'][name])
        target = np.zeros((1, 5, 4))
        target[..., :2] = first['predictions']['native_ols']
        valid = np.zeros_like(target, bool)
        valid[..., :2] = True
        score = position_metrics(first['predictions']['native_ols'], target, valid, first['regions']['all'])
        self.assertEqual(score['mae'], 0.)
        valid[..., 1] = False
        missing = position_metrics(first['predictions']['native_ols'], target, valid, first['regions']['all'])
        self.assertEqual(missing['intersection_count'], 0)
        self.assertIsNone(missing['mae'])


class NativeMotionCliTests(unittest.TestCase):
    def setUp(self):
        self.fixture = make_native_zip_fixture(self)

    def run_control(self, name):
        fixture = self.fixture
        for filename, value in [('manifest.json', fixture.manifest), ('alignment.json', fixture.alignment)]:
            (fixture.root/filename).write_text(json.dumps(value))
        output = fixture.root/name
        with redirect_stdout(io.StringIO()):
            self.assertEqual(main(['--archive', str(fixture.path), '--manifest', str(fixture.root/'manifest.json'),
                '--alignment', str(fixture.root/'alignment.json'), '--output', str(output)]), 0)
        return output, json.loads((output/'native-motion.json').read_text())

    def test_real_tiny_zip_complete_v_only_and_reference_cannot_change_predictions(self):
        output, report = self.run_control('first')
        self.assertEqual((report['scene_count'], report['fragment_count'], report['query_count']), (2, 4, 256))
        self.assertEqual(report['fit_fragment_count'], 4)
        self.assertEqual(report['fallback_fragment_count'], 0)
        self.assertFalse(report['test_opened'])
        self.assertEqual(len(report['per_fragment']), 4)
        self.assertEqual(report['per_scene'][1]['fragment_count'], 0)
        self.assertEqual(report['per_scene'][1]['reference_targets_without_input'], ['target4a-0'])
        self.assertEqual({row['scene_id'] for row in report['observation_members_read']}, {'2', '4'})
        self.assertEqual({row['scene_id'] for row in report['reference_members_read']}, {'2', '4'})
        # Two b fragments have the fixture's deliberate +0.1-degree latitude
        # source bias; native motion must not use the reference to remove it.
        self.assertAlmostEqual(report['overall']['native_ols']['regions']['all']['mae'],
                               6371000.*np.deg2rad(.1)/2., places=5)
        by_source = report['by_source']
        self.assertGreater(by_source['9002']['native_hold']['regions']['all']['mae'], 0.)
        self.assertLess(by_source['9002']['native_ols']['regions']['all']['mae'],
                        by_source['9002']['native_hold']['regions']['all']['mae'])
        # Poison only V reference latitude; observed ZIP members remain identical.
        with zipfile.ZipFile(self.fixture.path) as archive:
            members = {name: archive.read(name) for name in archive.namelist()}
        for name, payload in list(members.items()):
            if '/真实场景/' in name and name.endswith(('-2.csv', '-4.csv')):
                lines = payload.decode('utf-8-sig').splitlines()
                changed = [lines[0]]
                for line in lines[1:]:
                    cells = line.split(',')
                    cells[2] = str(float(cells[2])+1.)
                    changed.append(','.join(cells))
                members[name] = ('\n'.join(changed)+'\n').encode('utf-8-sig')
        with zipfile.ZipFile(self.fixture.path, 'w') as archive:
            for name, payload in members.items():
                archive.writestr(name, payload)
        poisoned, changed_report = self.run_control('poisoned')
        self.assertGreater(changed_report['overall']['native_ols']['regions']['all']['mae'], 10000.)
        for sid in ('2', '4'):
            with np.load(output/'predictions'/f'positions-{sid}.npz', allow_pickle=False) as first, \
                    np.load(poisoned/'predictions'/f'positions-{sid}.npz', allow_pickle=False) as second:
                for name in METHODS:
                    np.testing.assert_array_equal(first[name], second[name])
                for name in ('node_ids', 'query_seconds', 'native_times', 'native_observed'):
                    np.testing.assert_array_equal(first[name], second[name])
        with self.assertRaises(FileExistsError):
            self.run_control('first')


if __name__ == '__main__':
    unittest.main()
