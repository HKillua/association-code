"""Opt-in physical longitude corrections; synthetic data only, no MTAD scan."""
import copy
from contextlib import redirect_stderr, redirect_stdout
import io
import unittest
from unittest.mock import patch

import numpy as np
import torch

from association.data import DataError
from association.fusion import FusionLoss
from association.fusion_targets import (align_reference_states, build_fusion_targets,
                                        build_scene_fusion_targets)
from association.cooperative_training import training_configuration
from tools.experiments.cooperative_fit import synthetic_configuration


def reference(longitudes=(179., -179.), *, identity='ship-0'):
    return dict(scene_id='synthetic', split='T_core', state_fields=['lat', 'lon', 'vel', 'cou'],
                issues=[], rows=[dict(line=i+2, identity_id=identity, time=float(i*10),
                    state=[10.+i, lon, 2.+i, 350. if i == 0 else 10.])
                    for i, lon in enumerate(longitudes)])


def align(data, times, **kwargs):
    return align_reference_states(data, 'ship-0', times, time_offset_seconds=0.,
                                  max_gap_seconds=10., **kwargs)


class LongitudeBoundaryTests(unittest.TestCase):
    def test_reference_crossing_and_exact_points_preserve_legacy_opt_in(self):
        data = reference()
        before = copy.deepcopy(data)
        old = align(data, [0., 2.5, 5., 7.5, 10.])
        new = align(data, [0., 2.5, 5., 7.5, 10.], longitude_policy='shortest_arc')
        np.testing.assert_array_equal(old['states'][:, 1], [179., 89.5, 0., -89.5, -179.])
        np.testing.assert_array_equal(new['states'][:, 1], [179., 179.5, -180., -179.5, -179.])
        np.testing.assert_array_equal(old['states'][:, [0, 2, 3]], new['states'][:, [0, 2, 3]])
        np.testing.assert_array_equal(old['valid'], new['valid'])
        self.assertEqual(new['report']['longitude_policy'], 'shortest_arc')
        self.assertEqual(data, before)
        same = align(reference((180., -180.)), [0., 5., 10.], longitude_policy='shortest_arc')
        np.testing.assert_array_equal(same['states'][:, 1], [180., -180., -180.])

    def test_non_crossing_and_signed_half_turn_keep_exact_legacy_arithmetic(self):
        for endpoints in ((130.1234, 131.7182), (0., 180.), (0., -180.), (-90., 90.)):
            with self.subTest(endpoints=endpoints):
                old = align(reference(endpoints), [0., 1.3, 5., 7.1, 10.])
                new = align(reference(endpoints), [0., 1.3, 5., 7.1, 10.], longitude_policy='shortest_arc')
                np.testing.assert_array_equal(old['states'], new['states'])
                np.testing.assert_array_equal(old['valid'], new['valid'])

    def test_boundary_policy_keeps_masks_gap_limits_and_no_extrapolation(self):
        data = reference()
        data['rows'].append(dict(line=4, identity_id='ship-0', time=10., state=[11., -178., 3., 10.]))
        old = align(data, [-1., 0., 5., 10., 11.])
        new = align(data, [-1., 0., 5., 10., 11.], longitude_policy='shortest_arc')
        np.testing.assert_array_equal(old['states'], new['states'])
        np.testing.assert_array_equal(old['valid'], new['valid'])
        for policy in ('linear', 'shortest_arc'):
            result = align_reference_states(reference(), 'ship-0', [5.],
                time_offset_seconds=0., max_gap_seconds=9., longitude_policy=policy)
            self.assertFalse(result['valid'].any())
        with self.assertRaisesRegex(DataError, 'longitude_policy'):
            align(reference(), [5.], longitude_policy='guess')

    def test_pair_and_all_identity_builders_forward_longitude_policy(self):
        prepared = dict(scene_id='synthetic', split='T_core', grid=np.array([0., 5., 10.]),
            values=np.zeros((2, 3, 4)), observed=np.ones((2, 3), bool),
            identities=np.array(['ship-0', 'ship-0']), pairs=np.array([[0, 1]]), labels=np.array([1.]))
        refs = reference()
        refs['rows'] += reference((-179., 179.), identity='other-0')['rows']
        pairs = build_fusion_targets(prepared, refs, [0], split='T_core', nominal_grid=prepared['grid'],
            time_offset_seconds=0., max_gap_seconds=10., longitude_policy='shortest_arc')
        self.assertEqual(pairs['states'][0, 1, 1], -180.)
        all_targets = build_scene_fusion_targets(prepared, refs,
            target_identity_ids=['ship-0', 'other-0', 'absent-0'], time_offset_seconds=0.,
            max_gap_seconds=10., longitude_policy='shortest_arc')
        self.assertEqual(all_targets['other-0']['states'][1, 1], -180.)
        self.assertFalse(all_targets['absent-0']['valid'].any())
        self.assertTrue(all(item['report']['longitude_policy'] == 'shortest_arc'
                            for item in all_targets.values()))

    def loss(self, prediction, target, *, longitude_mode=None, position_mode='standard_smooth_l1', valid=None,
             normalization_counts=None, regularizer_weight=1.):
        kwargs = {} if longitude_mode is None else {'longitude_loss_mode': longitude_mode}
        objective = FusionLoss(position_loss_mode=position_mode, **kwargs).to(prediction)
        if valid is None:
            valid = torch.ones_like(target, dtype=torch.bool)
        return objective(prediction, target, valid, normalization_counts=normalization_counts,
                         regularizer_weight=regularizer_weight)

    def test_circular_loss_same_location_zero_and_correct_local_gradient(self):
        for pred_lon, target_lon, expected_loss, expected_gradient in (
                (-180., 180., 0., 0.), (179.9, -179.9, .1995, -1.),
                (-179.9, 179.9, .1995, 1.)):
            with self.subTest(pred=pred_lon, target=target_lon):
                pred = torch.tensor([[[0., pred_lon, 1., 0.]]], dtype=torch.float64, requires_grad=True)
                target = torch.tensor([[[0., target_lon, 1., 0.]]], dtype=torch.float64)
                result = self.loss(pred, target, longitude_mode='shortest_arc')
                result['losses']['lon'].backward()
                self.assertAlmostEqual(float(result['losses']['lon'].detach()), expected_loss, places=10)
                self.assertEqual(float(pred.grad[0, 0, 1]), expected_gradient)
                self.assertGreater(float(self.loss(pred.detach(), target)['losses']['lon']), 359.)

    def test_ordinary_loss_values_and_both_input_gradients_are_bitwise_unchanged(self):
        for dtype in (torch.float32, torch.float64):
            for position_mode in ('standard_smooth_l1', 'paper_printed'):
                baseline = None
                for longitude_mode in (None, 'linear', 'shortest_arc'):
                    pred = torch.tensor([[[1.001, 131.0001, 2., 359.], [2., -20., 3., 5.],
                                          [4., 0., 1., 10.]]], dtype=dtype, requires_grad=True)
                    target = torch.tensor([[[1., 131., 2.2, 1.], [2.2, 80., 2.8, 7.],
                                            [3., 180., 1., 10.]]], dtype=dtype, requires_grad=True)
                    result = self.loss(pred, target, longitude_mode=longitude_mode, position_mode=position_mode)
                    result['total'].backward()
                    actual = (torch.stack(list(result['losses'].values())).detach(), result['total'].detach(),
                              pred.grad, target.grad)
                    if baseline is None:
                        baseline = actual
                    else:
                        for value, expected in zip(actual, baseline):
                            torch.testing.assert_close(value, expected, rtol=0, atol=0)

    def test_masked_microbatch_circular_loss_matches_full_batch(self):
        pred = torch.tensor([[[1., 179.9, 2., 1.]], [[2., -180., 3., 359.]]],
                            dtype=torch.float64, requires_grad=True)
        target = torch.tensor([[[1.1, -179.9, 2.2, 0.]], [[float('nan'), 180., 3.1, 1.]]],
                              dtype=torch.float64)
        valid = torch.isfinite(target)
        full = self.loss(pred, target, longitude_mode='shortest_arc', valid=valid)['total']
        grad = torch.autograd.grad(full, pred)[0]
        streamed = sum(self.loss(pred[i:i+1], target[i:i+1], longitude_mode='shortest_arc',
                                valid=valid[i:i+1], normalization_counts=valid.sum((0, 1)),
                                regularizer_weight=float(i == 0))['total'] for i in range(2))
        torch.testing.assert_close(streamed, full, rtol=1e-14, atol=1e-14)
        torch.testing.assert_close(torch.autograd.grad(streamed, pred)[0], grad, rtol=0, atol=0)

    def test_training_default_and_v2_synthetic_are_explicit(self):
        old = training_configuration(synthetic_configuration())
        self.assertEqual(old['longitude_loss_mode'], 'linear')
        new = training_configuration(synthetic_configuration('support_residual_v2'))
        self.assertEqual(new['longitude_loss_mode'], 'shortest_arc')
        for value in ('guess', None, True):
            with self.subTest(value=value), self.assertRaises(ValueError):
                training_configuration(dict(old, longitude_loss_mode=value))
        with self.assertRaises(ValueError):
            FusionLoss(position_loss_mode='standard_smooth_l1', longitude_loss_mode='guess')

    def test_timed_loader_forwards_policy_before_reference_assembly(self):
        import test_timed_dataset as fixture_module
        fixture = fixture_module.TimedDatasetTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        from association.timed_dataset import build_scene_fusion_targets as build
        calls = []
        def record(*args, **kwargs):
            calls.append(kwargs['longitude_policy'])
            return build(*args, **kwargs)
        with patch('association.timed_dataset.build_scene_fusion_targets', side_effect=record):
            result = fixture.load(data_config=dict(fixture.data, reference_longitude_policy='shortest_arc'))
        self.assertEqual(calls, ['shortest_arc'] * 3)
        self.assertEqual(result['report']['reference_longitude_policy'], 'shortest_arc')
        for item in result['train'] + result['validation']:
            self.assertEqual(item['reference_alignment']['longitude_policy'], 'shortest_arc')
        self.assertFalse(result['report']['test_opened'])
        with patch('association.timed_dataset.MTADArchive', side_effect=AssertionError('no archive')):
            with self.assertRaisesRegex(DataError, 'reference_longitude_policy'):
                fixture.load(data_config=dict(fixture.data, reference_longitude_policy='guess'))

    def test_complete_reference_fit_accepts_opt_in_and_restores_loss_semantics(self):
        import test_timed_dataset as fixture_module
        from association import runtime
        from association.complete_training import load_complete_snapshot
        from tools.experiments.tmtaff_complete_fit import main
        fixture = fixture_module.TimedDatasetTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        threads = torch.get_num_threads()
        self.addCleanup(torch.set_num_threads, threads)
        fixture.config['data']['reference_longitude_policy'] = 'shortest_arc'
        fixture.config['training']['longitude_loss_mode'] = 'shortest_arc'
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            self.assertEqual(main([*fixture.cli_args('longitude-fit'), '--device', 'cpu']), 0)
        output = fixture.root/'longitude-fit'
        prepared = runtime.read_json(output/'preparation.json')
        self.assertEqual(prepared['dataset']['reference_longitude_policy'], 'shortest_arc')
        restored = load_complete_snapshot(output/'checkpoint.pt')
        self.assertEqual(restored['objective'].longitude_loss_mode, 'shortest_arc')


if __name__ == '__main__':
    unittest.main()
