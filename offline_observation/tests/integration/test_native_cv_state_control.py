"""127 math/import/selection checks. Synthetic values are software evidence only."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from tests.integration.test_source_preference_control import saved_scene
from tools.analysis import native_cv_state_control as p


def physical_cache(nodes=2):
    scene = dict(node_ids=[f'n{i}' for i in range(nodes)], sources=['a']*nodes,
                 grid=np.linspace(0, 10, 64))
    events = [dict(times=np.array([0., 10.]), values=np.array([[0., 0., 7., 45.], [0., 0., 7., 45.]]),
                   valid=np.ones((2, 4), bool), event_ids=[(f'n{i}', 0), (f'n{i}', 1)]) for i in range(nodes)]
    return p.pack_native_events(scene, events)


class NativeCVStateTests(unittest.TestCase):
    def test_rts_agrees_with_independent_joint_gaussian_normal_equations(self):
        times = np.array([0., 2., 5.])
        xy = np.array([[30., 120.], [30.0002, 120.0001], [30.0005, 120.0003]])
        query = np.array([-3., 0., 1., 2., 4., 5., 8.])
        rho = .01
        output, audit = p.cv_native_positions(times, xy, query, rho)
        timeline = np.array([0., 1., 2., 4., 5.])
        scale = p.RADIUS*np.pi/180
        east = scale*np.cos(np.deg2rad(xy[:, 0].mean()))
        y = (xy-xy[0])*[scale, east]
        # Batch information solution independently assembles all factors;
        # no Kalman gain or backward RTS recursion is reused.
        width = 4*len(timeline)
        precision, rhs = np.zeros((width, width)), np.zeros(width)
        precision[:4, :4] += np.diag([1., 1., 3600., 3600.])
        for index in range(1, len(timeline)):
            dt = timeline[index]-timeline[index-1]
            f = np.array([[1., 0., dt, 0.], [0., 1., 0., dt], [0., 0., 1., 0.], [0., 0., 0., 1.]])
            q = rho*np.array([[dt**3/3, 0., dt**2/2, 0.], [0., dt**3/3, 0., dt**2/2],
                             [dt**2/2, 0., dt, 0.], [0., dt**2/2, 0., dt]])
            factor = np.zeros((4, width))
            factor[:, 4*(index-1):4*index] = -f
            factor[:, 4*index:4*(index+1)] = np.eye(4)
            precision += factor.T@np.linalg.solve(q, factor)
        for time_index, value in zip(times[1:], y[1:]):
            j = int(np.flatnonzero(timeline == time_index)[0])*4
            precision[j:j+2, j:j+2] += np.eye(2)
            rhs[j:j+2] += value
        latent = np.linalg.solve(precision, rhs).reshape(-1, 4)
        expected = np.empty((len(query), 2))
        for i, q in enumerate(query):
            if q < 0:
                value = latent[0, :2]+q*latent[0, 2:]
            elif q > 5:
                value = latent[-1, :2]+(q-5)*latent[-1, 2:]
            else:
                value = latent[np.flatnonzero(timeline == q)[0], :2]
            expected[i] = xy[0]+value/[scale, east]
        np.testing.assert_allclose(output['rts'], expected, atol=1e-10, rtol=0)
        np.testing.assert_array_equal(output['forward'][0], xy[0])
        self.assertEqual(audit['kalman_observation_updates'], 2)
        self.assertEqual(audit['transition_steps'], 4)
        self.assertEqual(audit['outside_query_count'], 2)

    def test_date_line_and_no_negative_time_or_hidden_identity_inputs(self):
        out, _ = p.cv_native_positions(np.array([0., 3.]), np.array([[0., 179.9], [0., -179.9]]),
                                       np.array([-1., 0., 1., 3., 4.]), .01)
        self.assertTrue(np.all(np.abs(out['rts'][:, 1]) > 179.8))
        half, _ = p.cv_native_positions(np.array([0., 3.]), np.array([[0., 0.], [0., 180.]]),
                                        np.array([0., 1., 3.]), .01)
        self.assertGreater(half['rts'][1, 1], 0.)
        with self.assertRaises(ValueError):
            p.cv_transition(-1., .01)
        cache = physical_cache()
        records = list(p.native_records(cache))
        self.assertEqual(len(records), 2)
        np.testing.assert_array_equal(records[1]['values'][:, 2:], [[7., 45.], [7., 45.]])
        with self.assertRaises(ValueError):
            list(p.native_records(dict(cache, identities=np.array(['x', 'y']))))
        bad = deepcopy(cache)
        bad['event_canonical_indices'][1] = 0
        with self.assertRaises(ValueError):
            list(p.native_records(bad))

    def test_t_scale_is_rts_fragment_equal_and_missing_reference_population_retained(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root/'private/native-events/T_core').mkdir(parents=True)
            (root/'private/native-evaluation/T_core').mkdir(parents=True)
            for sid in ('a', 'b'):
                cache = physical_cache()
                valid = np.zeros((2, 64, 4), bool)
                if sid == 'a':
                    valid[0, :, :2] = True
                    valid[1, 0, :2] = True
                ev = dict(target_ids=np.array(['id-0', 'id-1']), identities=np.array(['id-0', 'id-1']),
                          target_states=np.zeros((2, 64, 4)), target_valid=valid)
                np.savez(root/'private/native-events/T_core'/f'scene-{sid}.npz', **cache)
                np.savez(root/'private/native-evaluation/T_core'/f'scene-{sid}.npz', **ev)
            def prediction(cache, rho, check=None):
                index = p.RHOS.index(rho)
                # A: fragment-equal (1+9)/2=5m, point-weighted=1.12m.
                # B: fragment-equal (6+3)/2=4.5m, point-weighted=5.95m.
                lengths = [1., 9.] if index == 0 else [6., 3.] if index == 1 else [100., 100.]
                states = np.zeros((2, 64, 2))
                states[..., 0] = np.asarray(lengths)[:, None]/(p.RADIUS*np.pi/180)
                counts = dict(native_event_count=4, kalman_observation_updates=2, transition_steps=2,
                              rts_backward_steps=2, inside_query_count=128, outside_query_count=0)
                return dict(rts=states, forward=np.zeros_like(states)), counts
            with patch.object(p, 'predict_scene', side_effect=prediction) as mock:
                result = p.fit_scale(root, ['a', 'b'], lambda **kw: None)
            self.assertEqual(result['selected_rho'], p.RHOS[1])
            self.assertEqual((result['eligible_scene_count'], result['missing_reference_scene_count']), (1, 1))
            self.assertEqual(mock.call_count, 18)
            self.assertIsNone(result['per_scene'][1]['objective_by_rho'])
            self.assertEqual(result['per_scene'][1]['missing_reference_fragments'], 2)

    def test_old_two_controls_exact_import_and_full_population_new_methods(self):
        fragment, membership, retained118, calibration, public118 = saved_scene()
        public121, _, retained, _ = p.prior.analyze_scene(fragment, membership, retained118, calibration, public118)
        selected = {name: retained[name] for name in p.METHODS[:2]}
        groups = p.old.groups_from_membership(membership, len(fragment['native_states']))
        predictions = {name: selected[name]['group_states'].copy() for name in selected}
        for name in p.METHODS[2:]:
            predictions[name] = selected['WS_MEAN']['group_states'].copy()
            predictions[name][..., 0] += .00001
        row, _, artifacts = p.analyze_scene(fragment, groups, selected, predictions,
                                            np.ones(fragment['support'].shape)*60)
        self.assertEqual((len(row['methods']), len(row['pairwise'])), (6, 15))
        for name in p.METHODS:
            np.testing.assert_array_equal(artifacts[name]['group_states'][..., 2:],
                                          selected['WS_MEAN']['group_states'][..., 2:])
            self.assertEqual(row['methods'][name]['population'], public121['methods']['WS_MEAN']['population'])
        selected['WS_LOW_SOURCE_SUPPORTED']['aligned_prediction'][0, 0, 0] += .01
        with self.assertRaisesRegex(ValueError, '121 exact'):
            p.analyze_scene(fragment, groups, selected, predictions, np.ones(fragment['support'].shape)*60)


if __name__ == '__main__':
    unittest.main()
