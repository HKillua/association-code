import copy
import math
import unittest

import numpy as np
import torch

from association.data.motion_evidence import build_motion_evidence
from association.models.deepaf_peer import (
    DeepAFPeer, EARTH_RADIUS_M, MODEL_CONFIG, deepaf_batch_inputs, deepaf_inputs,
    deepaf_objectives, deepaf_supervision, local_to_positions, mahalanobis_scores,
    positions_to_local,
)
from tests.test_sequence_position import motion_fixture


def deepaf_motion_fixture():
    """Interface-only fixture: explicitly declared synthetic angular-rate evidence.

    The shared fixture has no original observations; these rates are not fitted
    from its interpolated query states or asserted to be native-data estimates.
    """
    motion = motion_fixture()
    motion['ols_slope_degrees_per_second'] = np.array(
        [[2e-5, 3e-5], [2e-5, 3e-5], [3e-5, 2e-5]])
    return motion


class DeepAFPeerModelTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(20261023)

    def inputs(self, members=(0, 1, 2)):
        return deepaf_inputs(deepaf_motion_fixture(), list(members))

    def test_source_core_architecture_and_exact_distance_scoring(self):
        model = DeepAFPeer()
        self.assertEqual(model.configuration, MODEL_CONFIG)
        self.assertEqual(sum(p.numel() for p in model.parameters()), 265732)
        for network in (model.position_network, model.velocity_network):
            self.assertEqual(len(network.encoder), 6)
            self.assertEqual(len(network.decoder), 6)
            self.assertEqual(network.output.in_features, 32)
            self.assertGreater(torch.count_nonzero(network.output.weight), 0)
            self.assertTrue(all(layer.cross_attention.mahalanobis for layer in network.decoder))
            self.assertTrue(all(not layer.attention.mahalanobis for layer in network.encoder))
        self.assertFalse(torch.equal(model.position_network.output.weight, model.velocity_network.output.weight))
        q = torch.tensor([[[1., 2.], [3., -2.]]])
        k = torch.tensor([[[1., 2.], [2., 4.], [-3., 2.]]])
        expected = -(q[:, :, None] - k[:, None]).square().sum(-1)
        torch.testing.assert_close(mahalanobis_scores(q, k), expected, rtol=0., atol=0.)
        self.assertEqual(float(mahalanobis_scores(q, k)[0, 0, 0]), 0.)
        with self.assertRaises(ValueError):
            DeepAFPeer(num_layers=2)

    def test_spherical_chart_round_trip_dateline_poles_and_antipodal_cut(self):
        origin = torch.tensor([[20., 179.8], [89.95, 35.], [-87., -130.]], dtype=torch.float64)
        points = torch.tensor([[[20.1, -179.9], [19.7, 179.2]],
                               [[89.8, 155.], [89.99, -37.]],
                               [[-85.5, 170.], [-87.3, -120.]]], dtype=torch.float64)
        local = positions_to_local(points, origin[:, None])
        recovered = local_to_positions(local, origin[:, None])
        torch.testing.assert_close(recovered, points, rtol=0., atol=2e-10)
        zero = torch.zeros((3, 2), dtype=torch.float64, requires_grad=True)
        initial = local_to_positions(zero, origin)
        self.assertTrue(torch.equal(initial, origin))
        initial.sum().backward()
        self.assertTrue(torch.isfinite(zero.grad).all())
        self.assertTrue(torch.equal(positions_to_local(origin, origin), torch.zeros_like(origin)))
        with self.assertRaisesRegex(ValueError, 'antipodal'):
            positions_to_local(torch.tensor([[0., 180.]], dtype=torch.float64),
                               torch.tensor([[0., 0.]], dtype=torch.float64))

    def test_visible_input_only_anchor_availability_and_helper_order(self):
        class VisibleOnly(dict):
            def __getitem__(self, name):
                if name in ('target', 'reference', 'identities', 'fusion_targets', 'source'):
                    raise AssertionError('truth or identity entered the model')
                return super().__getitem__(name)
        motion = VisibleOnly(deepaf_motion_fixture(), identities=object(), target=object())
        first = deepaf_inputs(motion, [0, 1, 2])
        reverse = deepaf_inputs(motion, [0, 2, 1])
        for key in first:
            self.assertTrue(torch.equal(first[key], reverse[key]), key)
        self.assertTrue(first['candidate_valid'][0].all())
        self.assertTrue(torch.equal(first['candidate_valid'][1:], torch.as_tensor(motion['support'][1:])))
        self.assertTrue(torch.equal(first['helper_support_count'], torch.as_tensor(motion['support'][1:].sum(0))))
        for bad in ([], [0., 1.], [0, 0], [0, 9]):
            with self.assertRaises(ValueError):
                deepaf_inputs(motion, bad)

    def test_bootstrap_is_tangent_derivative_at_actual_initial_chart_origin(self):
        # Real visible observations span latitudes 70--75 degrees. Their old
        # mean-latitude plane makes the initial east velocity about 12% too low.
        times = np.linspace(0., 54000., 128)
        anchor = np.column_stack((70. + 5. * times / times[-1],
                                  25. * times / times[-1],
                                  np.full_like(times, 20.), np.zeros_like(times)))
        helper = anchor.copy()
        helper[:, :2] += [-4., 2.]
        rate = torch.tensor([5., 25.], dtype=torch.float64) / times[-1]
        for members in ([0], [0, 1]):
            # A supported helper and an initially unsupported anchor activate
            # fixed095 for the mixed-origin case, using real native support.
            start = -1000. if len(members) > 1 else 0.
            motion = build_motion_evidence(np.stack((anchor, helper)),
                np.stack((times, times - 1000.)), np.ones((2, len(times)), bool),
                np.linspace(start, times[-1], 64), ['a', 'b'])
            inputs = deepaf_inputs(motion, members)
            origin = inputs['origin_latlon']
            # This curve is translated to base0. The test deliberately also
            # covers mixed base0 != the original anchor's first position.
            if len(members) > 1:
                anchor_initial = torch.as_tensor(motion['states'][0, 0, :2])
                self.assertFalse(torch.equal(origin, anchor_initial))
                self.assertNotEqual(float(origin[0]), float(anchor_initial[0]))
            epsilon = .1
            finite_difference = (positions_to_local(origin + epsilon * rate, origin)
                                 - positions_to_local(origin - epsilon * rate, origin)) / (2. * epsilon)
            torch.testing.assert_close(inputs['initial_velocity_mps'], finite_difference,
                                       rtol=1e-8, atol=1e-7)
            if len(members) == 1:
                old = torch.as_tensor(motion['ols_velocity_mps'][0])
                self.assertLess(float(old[1] / inputs['initial_velocity_mps'][1]), .89)
                dt = inputs['query_seconds'][1] - inputs['query_seconds'][0]
                self.assertGreater(float(torch.linalg.vector_norm(
                    (old - inputs['initial_velocity_mps']) * dt)), 1800.)

    def test_native_ols_rate_survives_when_query_interpolation_has_zero_slope(self):
        # The middle observation is not a query. Refitting just the two query
        # values would erase its contribution to the actual native OLS rate.
        times = np.array([0., 1., 100.])
        values = np.array([[20., 0., 2., 0.],
                           [20.001, .002, 2., 0.],
                           [20., 0., 2., 0.]])
        motion = build_motion_evidence(values[None], times[None],
            np.ones((1, 3), bool), np.array([0., 100.]), ['a'])
        centred = times - times.mean()
        native_rate = (centred[:, None] * (values[:, :2] - values[:, :2].mean(0))).sum(0) / (centred @ centred)
        np.testing.assert_allclose(motion['ols_slope_degrees_per_second'][0], native_rate,
                                   rtol=1e-12, atol=1e-15)
        np.testing.assert_array_equal(motion['states'][0, 0, :2], motion['states'][0, 1, :2])
        inputs = deepaf_inputs(motion, [0])
        self.assertGreater(float(torch.linalg.vector_norm(inputs['initial_velocity_mps'])), 0.)

    def test_bootstrap_rejects_missing_or_invalid_native_angular_rate(self):
        with self.assertRaisesRegex(ValueError, 'ols_slope_degrees_per_second'):
            deepaf_inputs(motion_fixture(), [0])
        for invalid in (np.zeros((2, 2)), np.zeros((3, 1)),
                        np.full((3, 2), np.nan), np.full((3, 2), np.inf)):
            with self.subTest(shape=invalid.shape, first=invalid[0, 0]):
                motion = deepaf_motion_fixture()
                motion['ols_slope_degrees_per_second'] = invalid
                with self.assertRaisesRegex(ValueError, 'OLS angular rates'):
                    deepaf_inputs(motion, [0])

    def test_closed_loop_equations_use_previous_v_and_cache_false_is_identical(self):
        inputs = self.inputs()
        batch = deepaf_batch_inputs([inputs, self.inputs((0,))])
        model = DeepAFPeer()
        result = model.forward_batch(batch)
        without_cache = model.forward_batch(batch, cache=False)
        self.assertNotIn('cache', without_cache)
        for key, value in without_cache.items():
            self.assertTrue(torch.equal(value, result[key]), key)
        self.assertTrue(torch.equal(result['states'][..., 2:], batch['mean_states'][..., 2:]))
        torch.testing.assert_close(result['positions_m'], result['previous_positions_m']
                                   + result['delta_seconds'][..., None] * result['p_velocity_mps'], rtol=0., atol=0.)
        torch.testing.assert_close(result['previous_positions_m'][:, 1:], result['positions_m'][:, :-1], rtol=0., atol=0.)
        torch.testing.assert_close(result['previous_positions_m'][:, 0],
                                   -batch['initial_velocity_mps'] * result['delta_seconds'][:, :1], rtol=0., atol=0.)
        cache = result['cache']
        self.assertTrue(torch.equal(cache['candidates'][:, 1:, 0],
                                    (result['v_velocity_mps'][:, :-1] / 10.).float()))
        self.assertTrue(torch.equal(cache['p_history'][:, 1:, -1],
                                    (result['p_velocity_mps'][:, :-1] / 10.).float()))
        self.assertTrue(torch.equal(cache['v_history'][:, 1:, -1],
                                    (result['v_velocity_mps'][:, :-1] / 10.).float()))
        self.assertEqual(int(cache['history_valid'][:, 0].sum()), 2)
        self.assertTrue(all(not value.requires_grad for value in cache.values()))
        reconstructed = local_to_positions(result['base_positions_m'] + result['residual_m'],
                                            result['origin_latlon'][:, None])
        torch.testing.assert_close(reconstructed, result['states'][..., :2], rtol=0., atol=1e-10)
        replay = model.replay_batch(result)
        torch.testing.assert_close(replay['p_normalized'].double() * 10., result['p_velocity_mps'].flatten(0, 1),
                                   rtol=2e-5, atol=2e-5)
        torch.testing.assert_close(replay['v_normalized'].double() * 10., result['v_velocity_mps'].flatten(0, 1),
                                   rtol=2e-5, atol=2e-5)

    def test_padding_and_masked_nan_cannot_affect_live_predictions(self):
        model = DeepAFPeer()
        batch = deepaf_batch_inputs([self.inputs(), self.inputs((0,))])
        changed = {key: value.clone() for key, value in batch.items()}
        changed['measurements_m'][~changed['candidate_valid']] = torch.nan
        original = model.forward_batch(batch, cache=False)
        masked = model.forward_batch(changed, cache=False)
        self.assertTrue(torch.equal(original['states'], masked['states']))
        singleton = model(self.inputs((0,)), cache=False)
        torch.testing.assert_close(singleton['states'], original['states'][1], rtol=0., atol=1e-8)
        changed['measurements_m'][0, 0, 0] = torch.inf
        with self.assertRaises(ValueError):
            model.forward_batch(changed)

    def test_future_measurements_cannot_change_prior_outputs(self):
        model = DeepAFPeer()
        batch = deepaf_batch_inputs([self.inputs()])
        changed = {key: value.clone() for key, value in batch.items()}
        changed['measurements_m'][:, :, -1] += 15000.
        original, shifted = model.forward_batch(batch), model.forward_batch(changed)
        self.assertTrue(torch.equal(original['states'][:, :-1], shifted['states'][:, :-1]))
        self.assertFalse(torch.equal(original['states'][:, -1], shifted['states'][:, -1]))
        # This property concerns query-wise model processing. The shared OLS
        # preprocessing itself intentionally uses canonical_all observations.

    def test_desired_increment_targets_masks_and_no_gap_velocity_labels(self):
        model = DeepAFPeer()
        batch = deepaf_batch_inputs([self.inputs(), self.inputs((0,))])
        rollout = model.forward_batch(batch)
        target = batch['base_states'].clone()
        target[0, :, 0] += .01
        valid = torch.ones_like(target, dtype=torch.bool)
        valid[1, 1, :2] = False
        target[1, 1, :2] = torch.nan
        supervision = deepaf_supervision(rollout, target, valid)
        self.assertEqual(supervision['valid_position_points'], [3, 2])
        self.assertEqual(supervision['valid_velocity_points'], [3, 0])
        p_target = supervision['p_target'].reshape(2, 3, 2)
        reconstructed = rollout['previous_positions_m'][0] + p_target[0] * 10. * rollout['delta_seconds'][0, :, None]
        expected = positions_to_local(target[0, :, :2], rollout['origin_latlon'][0])
        torch.testing.assert_close(reconstructed, expected, rtol=1e-12, atol=1e-8)
        velocity = supervision['v_target'].reshape(2, 3, 2)
        torch.testing.assert_close(velocity[0, 2] * 10., (expected[2] - expected[1]) / 40., rtol=1e-12, atol=1e-9)
        self.assertTrue(torch.equal(velocity[0, 0], velocity[0, 1]))
        self.assertTrue(torch.isfinite(supervision['p_target']).all())
        self.assertTrue(torch.isfinite(supervision['v_target']).all())
        self.assertAlmostEqual(float(supervision['p_weight'].sum()), 1.)
        self.assertAlmostEqual(float(supervision['v_weight'].sum()), .5)
        # Target values have no route back into saved inference inputs.
        before = copy.deepcopy(rollout['cache'])
        altered = target.clone()
        altered[0, :, 1] += .3
        deepaf_supervision(rollout, altered, valid)
        self.assertTrue(all(torch.equal(before[key], rollout['cache'][key]) for key in before))

    def test_chunk_losses_and_gradients_match_whole_replay_and_update_both_networks(self):
        batch = deepaf_batch_inputs([self.inputs(), self.inputs((0,))])
        model = DeepAFPeer()
        other = copy.deepcopy(model)
        rollout = model.forward_batch(batch)
        target = batch['base_states'].clone()
        target[..., 0] += .005
        valid = torch.ones_like(target, dtype=torch.bool)
        supervision = deepaf_supervision(rollout, target, valid)
        full, _ = deepaf_objectives(model.replay_batch(rollout), supervision)
        full['total'].backward()
        chunk_total = 0.
        for start, stop in ((0, 2), (2, 4), (4, 6)):
            partial, _ = deepaf_objectives(other.replay_batch(rollout, start, stop), supervision, start=start, stop=stop)
            partial['total'].backward()
            chunk_total += float(partial['total'].detach())
        self.assertAlmostEqual(chunk_total, float(full['total'].detach()), delta=abs(chunk_total) * 1e-6)
        for (name, parameter), (other_name, other_parameter) in zip(model.named_parameters(), other.named_parameters()):
            self.assertEqual(name, other_name)
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
            torch.testing.assert_close(parameter.grad, other_parameter.grad, rtol=2e-4, atol=2e-3)
        before = {name: p.detach().clone() for name, p in model.named_parameters()}
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        torch.optim.Adam(model.parameters(), lr=.001).step()
        for prefix in ('position_network.', 'velocity_network.'):
            self.assertTrue(any(not torch.equal(before[name], p) for name, p in model.named_parameters() if name.startswith(prefix)))

    def test_all_invalid_targets_keep_zero_gradients_for_every_parameter(self):
        model = DeepAFPeer()
        batch = deepaf_batch_inputs([self.inputs((0,))])
        rollout = model.forward_batch(batch)
        target = torch.full_like(batch['base_states'], torch.nan)
        valid = torch.zeros_like(target, dtype=torch.bool)
        supervision = deepaf_supervision(rollout, target, valid)
        losses, diag = deepaf_objectives(model.replay_batch(rollout), supervision)
        self.assertEqual(float(losses['total']), 0.)
        self.assertEqual(diag['valid_position_points'], [0])
        self.assertEqual(diag['valid_velocity_points'], [0])
        losses['total'].backward()
        for name, parameter in model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
            self.assertEqual(int(torch.count_nonzero(parameter.grad)), 0, name)

    def test_closed_loop_learns_two_simple_cv_groups_in_32_updates(self):
        """Non-production learnability check, never a Level1/peer result.

        Two fixed synthetic CV scenes, Q64, actual dt20s, two biased visible
        position streams, no observation noise, and visible OLS bootstrap.
        The same two groups are reused for 32 Adam updates (lr .001, clip1).
        This checks whether the implemented closed-loop P/V targets can be
        learned at all, not generalization, convergence, or relative strength.
        """
        q = 64
        times = torch.arange(q, dtype=torch.float64) * 20.
        rows, targets = [], []
        for origin_tuple, velocity_tuple in (((20., 115.), (2., 3.)),
                                             ((31., 179.5), (-1.5, 1.))):
            origin = torch.tensor(origin_tuple, dtype=torch.float64)
            velocity = torch.tensor(velocity_tuple, dtype=torch.float64)
            truth_m = times[:, None] * velocity
            truth = torch.cat((local_to_positions(truth_m, origin),
                               torch.full((q, 1), 5., dtype=torch.float64),
                               torch.zeros((q, 1), dtype=torch.float64)), -1)
            observations = []
            for bias in ((100., -80.), (-40., 50.)):
                position = local_to_positions(truth_m + torch.tensor(bias, dtype=torch.float64), origin)
                observations.append(torch.cat((position, truth[:, 2:]), -1).numpy())
            motion = build_motion_evidence(np.stack(observations),
                np.tile(times.numpy(), (2, 1)), np.ones((2, q), bool),
                times.numpy(), ['a', 'b'])
            rows.append(deepaf_inputs(motion, [0, 1]))
            targets.append(truth)
        batch, target = deepaf_batch_inputs(rows), torch.stack(targets)
        valid = torch.ones_like(target, dtype=torch.bool)
        model = DeepAFPeer()
        optimizer = torch.optim.Adam(model.parameters(), lr=.001)

        @torch.no_grad()
        def diagnostic():
            rollout = model.forward_batch(batch)
            supervision = deepaf_supervision(rollout, target, valid)
            losses, _ = deepaf_objectives(model.replay_batch(rollout), supervision)
            truth_m = positions_to_local(target[..., :2], rollout['origin_latlon'][:, None])
            mae = torch.linalg.vector_norm(rollout['positions_m'] - truth_m, dim=-1).mean()
            return {**{key: float(value) for key, value in losses.items()}, 'local_position_mae_m': float(mae)}

        initial = diagnostic()
        for _ in range(32):
            optimizer.zero_grad(set_to_none=True)
            rollout = model.forward_batch(batch)
            supervision = deepaf_supervision(rollout, target, valid)
            for start in range(0, 2 * q, 64):
                losses, _ = deepaf_objectives(model.replay_batch(rollout, start, start + 64),
                                             supervision, start=start, stop=start + 64)
                self.assertTrue(torch.isfinite(losses['total']))
                losses['total'].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
        final = diagnostic()
        for key, value in final.items():
            self.assertTrue(math.isfinite(value), key)
            self.assertLess(value, initial[key] * .1, key)


if __name__ == '__main__':
    unittest.main()
