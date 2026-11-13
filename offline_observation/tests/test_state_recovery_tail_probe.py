"""Software checks for 141 private-state reconstruction, never MTAD evidence."""
import tempfile
from pathlib import Path
import unittest

import numpy as np

from tools.analysis import state_recovery_tail_probe as probe
from tools.analysis.native_cv_state_control import cv_native_positions


class TailTraceTests(unittest.TestCase):
    def test_original_kernel_equivalence_and_endpoint_information(self):
        t = np.array([0., 5., 11., 21., 35.])
        ll = np.column_stack((25.+t*1e-5, (179.9999+t*2e-5+180.) % 360.-180.))
        q = np.array([-20., -1., 0., 2., 13., 35., 51., 77.])
        for rho in (1e-8, 1e-4):
            actual = probe.trace_cv(t, ll, q, rho)
            original, _ = cv_native_positions(t, ll, q, rho)
            for direction in ('FORWARD', 'RTS'):
                np.testing.assert_allclose(actual[direction]['positions'], original[direction.lower()], atol=1e-10, rtol=1e-12)
                self.assertEqual(actual[direction]['endpoints'].shape, (2, 4))
            np.testing.assert_array_equal(actual['FORWARD']['endpoints'][0, 2:], 0.)
            self.assertGreater(np.linalg.norm(actual['RTS']['endpoints'][0, 2:]), 0.)

    def fixture(self, root):
        state, native = root/'state', root/'native'
        (state/'private').mkdir(parents=True)
        for folder in ('native-motion', 'native-events', 'native-evaluation'):
            (native/'private'/folder/'T_core').mkdir(parents=True)
        times = [np.array(v, float) for v in ([10, 20, 30], [-10, 0, 10], [0, 10, 20])]
        query, ids = np.arange(64.), np.array(['n0', 'n1', 'n2'])
        positions = [np.column_stack((np.zeros(3), t*2e-5)) for t in times]
        values = [np.column_stack((p, np.ones(3), np.ones(3)*90)) for p in positions]
        offsets = np.arange(4)*3
        support = np.asarray([(query >= t[0]) & (query <= t[-1]) for t in times])
        nearest = np.asarray([abs(query[:, None]-t).min(1) for t in times])
        states = np.zeros((3, 64, 4))
        states[:, :, 1] = query*2e-5
        states[:, :, 2:] = 1.
        sources = np.array(['sensorA', 'sensorA', 'sensorB'])
        motion = dict(node_ids=ids, query_seconds=query, states=states, support=support, nearest_seconds=nearest,
                      sources=sources, ols_velocity_mps=np.tile([0., 6371000*np.pi/180*2e-5], (3, 1)))
        events = dict(node_ids=ids, query_seconds=query, sources=sources, event_offsets=offsets,
                      event_times_seconds=np.concatenate(times), event_values=np.concatenate(values),
                      event_valid=np.ones((9, 4), bool), event_node_ids=np.repeat(ids, 3),
                      event_canonical_indices=np.tile(np.arange(3), 3))
        truth = states[[0, 2]].copy()
        ev = dict(node_ids=ids, query_seconds=query, identities=np.array(['a', 'a', 'b']),
                  target_ids=np.array(['a', 'b']), target_states=truth, target_valid=np.ones_like(truth, bool))
        for folder, data in [('native-motion', motion), ('native-events', events), ('native-evaluation', ev)]:
            np.savez(native/'private'/folder/'T_core/scene-fixture.npz', **data)
        saved = dict(node_ids=ids, query_seconds=query, membership=np.arange(3), unique_rhos=np.array([1e-8, 1e-4]),
                     reference_rho_index=np.asarray(0), obs_rho_index=np.asarray(1))
        group_outputs = {probe.BASE: states.copy()}
        for i, label in enumerate(('REFERENCE', 'OBS')):
            outputs = [cv_native_positions(t, p, query, saved['unique_rhos'][i])[0] for t, p in zip(times, positions)]
            for direction in ('FORWARD', 'RTS'):
                all_positions = np.asarray([o[direction.lower()] for o in outputs])
                saved[f'rho_{i}__{direction.lower()}_positions'] = all_positions
                gs = states.copy(); gs[..., :2] = all_positions
                group_outputs[f'WS_{label}_SCALE_{direction}_EQUAL'] = gs
        np.savez(state/'private/native-state-T_core-fixture.npz', **saved)
        aligned = np.array([0, 2, 1])
        aligned_truth = np.concatenate((truth, np.zeros((1, 64, 4))))
        truth_valid = np.ones_like(aligned_truth, bool); truth_valid[2] = False
        payload, expected = {}, dict(methods={})
        for name in probe.NAMES:
            gs = group_outputs[name]
            artifact = dict(target_ids=ev['target_ids'], aligned_target_ids=np.array(['a', 'b', '']),
                aligned_truth=aligned_truth, aligned_truth_valid=truth_valid, aligned_prediction=gs[aligned],
                aligned_prediction_valid=np.ones_like(gs, bool), group_states=gs, group_valid=np.ones_like(gs, bool),
                group_offsets=np.arange(4), group_node_indices=np.arange(3), aligned_group_indices=aligned,
                split=np.asarray('T_core'), correspondence_policy=np.asarray('anchor_identity_first'))
            payload.update({name+'_'+k: v for k, v in artifact.items()})
            errors = probe.distance(gs[aligned[:2], :, :2], truth[..., :2])
            identity = np.asarray([support[0] | support[1], support[2]])
            supported = support[[0, 2]]
            masks = dict(inside_support=identity, outside_support=~identity, source_support_zero=~supported,
                         source_support_one=supported, source_support_multi=np.zeros_like(supported))
            expected['methods'][name] = dict(position={'all': probe.error_stats(errors.ravel())},
                fixed_target_strata={key: dict(point_count=int(mask.sum()), **{
                    k: v for k, v in probe.error_stats(errors[mask]).items() if k in ('mae', 'mse', 'rmse', 'p95', 'p99')})
                    for key, mask in masks.items()})
        np.savez(state/'private/states-T_core-fixture.npz', **payload)
        return state, native, expected

    def test_original_mapping_duplicate_and_independent_identity_spans(self):
        with tempfile.TemporaryDirectory() as temp:
            state, native, expected = self.fixture(Path(temp))
            cases, _, _, strata, work, trace = probe.inspect_scene(state, native, 'T_core', 'fixture', expected)
            self.assertEqual(len(cases), 86)
            self.assertEqual(work['replay_calls'], 4)
            self.assertTrue(strata['inside_support'][0, :11].all())
            before = [c for c in cases if c['side'] == 'before']
            self.assertEqual(len(before), 10)
            self.assertTrue(all(c['identity_span_inside'] for c in before))
            self.assertTrue(any('endpoint_states' in k for k in trace))
            self.assertTrue(all(c['WS_OBS_SCALE_RTS_EQUAL__cached_replay_difference_m'] < 1e-6 for c in cases))

    def test_cached_node_and_native_support_tampering_rejected(self):
        for tamper in ('prediction', 'support', 'aligned'):
            with self.subTest(tamper=tamper), tempfile.TemporaryDirectory() as temp:
                state, native, expected = self.fixture(Path(temp))
                path = state/'private/native-state-T_core-fixture.npz'
                if tamper == 'support':
                    path = native/'private/native-motion/T_core/scene-fixture.npz'
                elif tamper == 'aligned':
                    path = state/'private/states-T_core-fixture.npz'
                data = probe.load(path)
                if tamper == 'prediction':
                    data['rho_1__rts_positions'][0, 0, 0] += .01
                elif tamper == 'support':
                    data['support'][1, 0] = False
                else:
                    data[probe.BASE+'_aligned_prediction'][0, 0, 0] += .01
                np.savez(path, **data)
                with self.assertRaises(ValueError):
                    probe.inspect_scene(state, native, 'T_core', 'fixture', expected)

    def test_real_nested_metric_contract_required(self):
        with tempfile.TemporaryDirectory() as temp:
            state, native, expected = self.fixture(Path(temp))
            expected['methods'][probe.BASE]['position'] = expected['methods'][probe.BASE]['position']['all']
            with self.assertRaises(KeyError):
                probe.inspect_scene(state, native, 'T_core', 'fixture', expected)


if __name__ == '__main__':
    unittest.main()
