import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from association.data.motion_evidence import scene_motion_evidence
from association.models.evidence_fusion import physical_errors
from association.models.sequence_position import (FEATURE_WIDTH, ROUTING_SEMANTICS, SCOPE,
    RELATIVE_HARM_SCOPE, RELATIVE_HARM_LOSS,
    SequencePosition, sequence_inputs, sequence_objectives, tangent_position)
from association.models.support_gate import SupportGate, gate_inputs
from association.training.sequence_position import configuration, load_checkpoint, train_sequence_position
from tools.experiments.evidence_fusion_fit import synthetic_scenes


def motion_fixture():
    values = np.array([
        [[20., 179., 2., 350.], [20., 179., 2., 350.], [20., 179., 2., 350.]],
        [[20., -179., 8., 10.], [20., -179., 8., 10.], [20., -179., 8., 10.]],
        [[20., -177., 6., 20.], [20., -177., 6., 20.], [20., -177., 6., 20.]]])
    return dict(states=values, query_seconds=np.array([100., 120., 160.]),
        support=np.array([[True, False, False], [True, True, False], [False, True, False]]),
        nearest_seconds=np.array([[0., 20., 40.], [0., 0., 20.], [20., 0., 20.]]),
        exact_observation=np.zeros((3, 3), bool), native_count=np.array([2, 4, 3]),
        span_seconds=np.array([10., 40., 20.]), fit_rmse_m=np.array([0., 10., 20.]),
        ols_velocity_mps=np.array([[2., 3.], [2., 3.], [3., 2.]]),
        endpoint_velocity_mps=np.array([[2., 3.], [3., 2.], [2., 3.]]))


def tiny_configuration(backbone='lstm'):
    return dict(scope=SCOPE, routing_semantics=ROUTING_SEMANTICS, position_loss='linear',
        data=dict(length=64, max_observations=128, time_unit_seconds=20., reference_longitude_policy='shortest_arc'),
        train_splits=['T_core'], motion_observation_policy='canonical_all',
        model=dict(backbone=backbone, hidden_size=64, num_layers=2, initial_gate=.95),
        training=dict(seed=20261023, updates=3, batch_size=2, learning_rate=.001,
                      gradient_clip=1., advisory_seconds=3600.))


def tiny_training():
    train, _ = synthetic_scenes(length=64, max_observations=4)
    for scene in train:
        scene['tracks'] = [dict(node_id=node_id, source=scene['sources'][node],
            canonical_values=scene['native_values'][node, scene['native_observed'][node]],
            canonical_times=scene['native_times'][node, scene['native_observed'][node]])
            for node, node_id in enumerate(scene['node_ids'])]
        scene['motion'] = scene_motion_evidence(scene, motion_observation_policy='canonical_all')
        scene['motion_observation_policy'] = 'canonical_all'
        del scene['tracks']
        for target in scene['fusion_targets'].values():
            target['states'][:, 0] += .02
    return train


class SequencePositionTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_visible_features_helper_permutation_real_time_and_fixed_initial(self):
        class VisibleOnly(dict):
            def __getitem__(self, key):
                if key in ('target', 'reference', 'identities', 'fusion_targets'):
                    raise AssertionError('reference/identity must not enter the encoder')
                return super().__getitem__(key)
        motion = VisibleOnly(motion_fixture(), reference=object(), identities=object())
        inputs = sequence_inputs(motion, [0, 1, 2])
        reverse = sequence_inputs(motion, [0, 2, 1])
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(inputs, reverse)))
        self.assertEqual(inputs[1].shape, (3, FEATURE_WIDTH))
        ordinary = SupportGate()(*gate_inputs(motion, [0, 1, 2]))['states']
        self.assertTrue(torch.equal(inputs[0], ordinary))
        shifted = copy.deepcopy(dict(motion))
        shifted['query_seconds'] += 100000.
        self.assertTrue(torch.equal(inputs[1], sequence_inputs(shifted, [0, 1, 2])[1]))
        shifted['query_seconds'] = np.array([100., 140., 220.])
        self.assertFalse(torch.equal(inputs[1], sequence_inputs(shifted, [0, 1, 2])[1]))
        shifted['query_seconds'] = np.array([100., 100., 220.])
        with self.assertRaises(ValueError):
            sequence_inputs(shifted, [0, 1, 2])
        for backbone in ('lstm', 'transformer'):
            result = SequencePosition(backbone=backbone)(*inputs)
            self.assertTrue(torch.equal(result['states'], ordinary))
            self.assertEqual(torch.count_nonzero(result['residual_m']).item(), 0)

    def test_singletons_all_queries_and_independent_directions(self):
        motion = motion_fixture()
        for backbone in ('lstm', 'transformer'):
            model = SequencePosition(backbone=backbone)
            singleton = sequence_inputs(motion, [0])
            self.assertTrue(torch.equal(model(*singleton)['states'], torch.tensor(motion['states'][0])))
            inputs = sequence_inputs(motion, [0, 1, 2])
            with torch.no_grad():
                model.output.bias[:] = torch.tensor([.5, 0.])
            for arguments in (singleton, inputs):
                result = model(*arguments)
                # North correction is independent of the east-only Mean delta,
                # and applies inside Anchor support and without any group support.
                self.assertTrue((result['states'][:, 0] > arguments[0][:, 0]).all())
                self.assertTrue(torch.equal(result['states'][:, 2:], arguments[2][:, 2:]))
                self.assertTrue(torch.equal(result['states'][:, 1], arguments[0][:, 1]))

    def test_poles_dateline_zero_residual_gradient_and_large_residual(self):
        base = torch.tensor([[90., 180., 1., 5.], [-90., -180., 2., 20.],
                             [20., 179.999, 3., 90.], [20., -179.999, 4., 180.]], dtype=torch.float64)
        residual = torch.zeros((4, 2), dtype=torch.float64, requires_grad=True)
        states = tangent_position(base, residual)
        self.assertTrue(torch.equal(states, base))
        target = base.clone()
        target[:, 0] = torch.tensor([89.99, -89.99, 20.02, 19.98])
        target[:, 1] = torch.tensor([90., -90., -179.99, 179.99])
        physical_errors(states, target)[:, 0].sum().backward()
        self.assertTrue(torch.isfinite(residual.grad).all())
        self.assertGreater(float(residual.grad.abs().sum()), 0.)
        residual = torch.tensor([[1e8, -1e8], [-1e8, 1e8], [0., 1000.], [0., -1000.]], dtype=torch.float64)
        states = tangent_position(base, residual)
        self.assertTrue(torch.isfinite(states).all())
        self.assertTrue((states[:, 0].abs() <= 90).all())
        self.assertTrue((states[:, 1] >= -180).all() and (states[:, 1] < 180).all())
        self.assertLess(float(states[2, 1]), 0.)
        self.assertGreater(float(states[3, 1]), 0.)

    def test_initial_head_learns_all_query_position_and_ignores_other_truth(self):
        arguments = sequence_inputs(motion_fixture(), [0])
        for backbone in ('lstm', 'transformer'):
            model = SequencePosition(backbone=backbone)
            result = model(*arguments)
            target = result['states'].detach().clone()
            target[:, 0] += .01
            target[:, 2:] = torch.nan
            target[1, :2] = torch.nan
            valid = torch.ones_like(target, dtype=torch.bool)
            valid[1, :2] = False
            losses, diagnostic = sequence_objectives(result, target, valid)
            self.assertEqual(diagnostic['valid_position_points'], 2)
            self.assertTrue(torch.isfinite(losses['linear']))
            losses['linear'].backward()
            self.assertGreater(float(model.output.weight.grad.abs().sum()), 0.)
            self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_real_tiny_training_endpoints_sampling_audit_and_checkpoint_rejection(self):
        train = tiny_training()
        with tempfile.TemporaryDirectory() as temp:
            logs = []
            for backbone in ('lstm', 'transformer'):
                config = tiny_configuration(backbone)
                path = Path(temp) / backbone
                result = train_sequence_position(train, config, path)
                self.assertEqual(result['status'], 'COMPLETED')
                self.assertEqual(result['optimizer_updates'], 3)
                self.assertEqual(result['exposure']['sampled_anchors'], 6)
                self.assertGreater(result['parameter_count'], 0)
                self.assertEqual(result['peak_cuda_memory_allocated_bytes'], 0)
                self.assertFalse(result['test_opened'])
                model, final = load_checkpoint(path / 'final.pt')
                initial, _ = load_checkpoint(path / 'step-0.pt')
                self.assertFalse(torch.equal(initial.output.bias, model.output.bias))
                self.assertEqual(final['step'], 3)
                rows = [json.loads(row) for row in (path / 'training.jsonl').read_text().splitlines()]
                self.assertEqual(len(rows), 3)
                logs.append([(r['mode'], r['batch']) for r in rows])
                windows = json.loads((path / 'training-windows.json').read_text())
                self.assertEqual(windows['windows'][0]['successful_updates'], 3)
                self.assertTrue(all(r['objectives']['valid_position_points'] > 0 for r in rows))
                for field, value in (('actual_updates', 2), ('loss_semantics', 'wrong')):
                    invalid = copy.deepcopy(final)
                    invalid[field] = value
                    torch.save(invalid, path / 'invalid.pt')
                    with self.assertRaises(ValueError):
                        load_checkpoint(path / 'invalid.pt')
                invalid = copy.deepcopy(final)
                next(iter(invalid['optimizer']['state'].values()))['step'] = torch.tensor(2.)
                torch.save(invalid, path / 'invalid.pt')
                with self.assertRaises(ValueError):
                    load_checkpoint(path / 'invalid.pt')
            self.assertEqual(logs[0], logs[1])

    def test_optimizer_failure_commits_no_exposure_and_rejects_validation_training(self):
        train = tiny_training()
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / 'failed'
            with patch.object(torch.optim.Adam, 'step', side_effect=RuntimeError('injected optimizer failure')):
                with self.assertRaisesRegex(RuntimeError, 'injected'):
                    train_sequence_position(train, tiny_configuration(), output)
            run = json.loads((output / 'run.json').read_text())
            self.assertEqual(run['status'], 'FAILED')
            self.assertEqual(run['optimizer_updates'], 0)
            self.assertEqual(run['exposure']['sampled_anchors'], 0)
            self.assertFalse((output / 'final.pt').exists())
            _, interrupted = load_checkpoint(output / 'interrupted.pt')
            self.assertEqual(interrupted['step'], 0)
            train[0]['split'] = 'V_select'
            with self.assertRaises(ValueError):
                train_sequence_position(train, tiny_configuration(), Path(temp) / 'invalid')
            self.assertFalse((Path(temp) / 'invalid').exists())

    def test_configuration_rejects_new_capacity_or_hidden_loss_changes(self):
        for key, value in (('position_loss', 'log1p'), ('train_splits', ['V_select'])):
            config = tiny_configuration()
            config[key] = value
            with self.assertRaises(ValueError):
                configuration(config)
        config = tiny_configuration()
        config['model']['hidden_size'] = 128
        with self.assertRaises(ValueError):
            configuration(config)

    def test_relative_harm_exact_masked_formula_detached_base_and_gradient(self):
        # Improve, worsen, tie, invalid; speed/course never enter this objective.
        prediction = torch.tensor([[.5, 0., 2., 3.], [2., 0., 2., 3.],
                                   [1., 0., 2., 3.], [40., 50., 2., 3.]],
                                  dtype=torch.float64, requires_grad=True)
        base = torch.tensor([[1., 0., 4., 5.]]*4, dtype=torch.float64, requires_grad=True)
        target = torch.zeros_like(prediction)
        target[3, :2] = torch.nan
        target[:, 2:] = torch.nan
        valid = torch.ones_like(prediction, dtype=torch.bool)
        valid[3, :2] = False
        result = dict(states=prediction, base_states=base,
                      residual_m=torch.zeros((4, 2), dtype=torch.float64, requires_grad=True))
        losses, audit = sequence_objectives(result, target, valid, position_loss=RELATIVE_HARM_LOSS)
        safe = torch.zeros_like(target)
        error = physical_errors(prediction, safe)[:3, 0]/1000.
        b = physical_errors(base, safe)[:3, 0].detach()/1000.
        torch.testing.assert_close(losses[RELATIVE_HARM_LOSS], (error+torch.relu(error-b)).mean(), rtol=0., atol=0.)
        ordinary = torch.autograd.grad(losses['linear'], prediction, retain_graph=True)[0]
        improved = torch.autograd.grad(losses[RELATIVE_HARM_LOSS], prediction, retain_graph=True)[0]
        torch.testing.assert_close(improved[:3, 0], ordinary[:3, 0]*torch.tensor([1., 2., 1.]))
        self.assertTrue(torch.equal(improved[:, 2:], torch.zeros_like(improved[:, 2:])))
        self.assertTrue(torch.equal(improved[3], torch.zeros_like(improved[3])))
        self.assertIsNone(torch.autograd.grad(losses[RELATIVE_HARM_LOSS], base, allow_unused=True)[0])
        self.assertEqual(audit['valid_position_points'], 3)
        self.assertGreater(audit['relative_harm_sum'], 0.)
        self.assertGreater(audit['relative_gain_sum'], 0.)

    def test_relative_harm_zero_empty_dateline_and_large_geographic_errors(self):
        base = torch.tensor([[0., 179.99, 0., 0.], [70., -179.99, 0., 0.],
                             [-70., 170., 0., 0.]], dtype=torch.float64)
        residual = torch.zeros((3, 2), dtype=torch.float64, requires_grad=True)
        result = dict(states=tangent_position(base, residual), base_states=base, residual_m=residual)
        mask = torch.ones_like(base, dtype=torch.bool)
        for valid in (mask, torch.zeros_like(mask)):
            losses, audit = sequence_objectives(result, base, valid, position_loss=RELATIVE_HARM_LOSS)
            self.assertEqual(float(losses[RELATIVE_HARM_LOSS].detach()), 0.)
            self.assertEqual(audit['relative_harm_sum'], 0.)
            gradient = torch.autograd.grad(losses[RELATIVE_HARM_LOSS], residual, retain_graph=True)[0]
            self.assertTrue(torch.equal(gradient, torch.zeros_like(gradient)))
        target = base.clone()
        target[:, :2] = torch.tensor([[0., -179.99], [-70., .01], [70., -10.]], dtype=torch.float64)
        residual = torch.tensor([[0., -1000.], [100000., 100000.], [-100000., 100000.]],
                                dtype=torch.float64, requires_grad=True)
        result = dict(states=tangent_position(base, residual), base_states=base, residual_m=residual)
        losses, audit = sequence_objectives(result, target, mask, position_loss=RELATIVE_HARM_LOSS)
        self.assertTrue(torch.isfinite(losses[RELATIVE_HARM_LOSS]))
        losses[RELATIVE_HARM_LOSS].backward()
        self.assertTrue(torch.isfinite(residual.grad).all())
        # Large absolute errors can have no relative penalty when better than b.
        self.assertGreater(audit['linear_sum'], 10000.)
        torch.testing.assert_close(losses[RELATIVE_HARM_LOSS],
            losses['linear']+audit['relative_harm_sum']/3, rtol=1e-12, atol=1e-12)

    def test_relative_harm_actual_training_pairing_logs_semantics_and_tamper_rejection(self):
        train = tiny_training()
        config = tiny_configuration('transformer')
        harm = copy.deepcopy(config)
        harm.update(scope=RELATIVE_HARM_SCOPE, position_loss=RELATIVE_HARM_LOSS)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            control = root/'control'
            train_sequence_position(train, config, control)
            result = train_sequence_position(train, harm, root/'harm', paired_reference_fit=control)
            self.assertEqual(result['paired_control']['status'], 'COMPLETED')
            self.assertEqual(result['paired_control']['actual_successful_batches_matched'], 3)
            self.assertTrue(result['paired_control']['sampling_exposure_exact'])
            old = [json.loads(line) for line in (control/'training.jsonl').read_text().splitlines()]
            rows = [json.loads(line) for line in (root/'harm/training.jsonl').read_text().splitlines()]
            self.assertEqual([r['batch'] for r in rows], [r['batch'] for r in old])
            self.assertEqual(rows[0]['loss'], old[0]['loss'])
            self.assertEqual(rows[0]['gradient_norm'], old[0]['gradient_norm'])
            for row in rows:
                terms = row['objectives']
                self.assertAlmostEqual(row['loss'], terms['anchor_mean_linear']+terms['anchor_mean_relative_harm'])
                self.assertGreaterEqual(terms['point_mean_relative_harm'], 0.)
                self.assertTrue(np.isfinite(row['gradient_norm']))
            _, checkpoint = load_checkpoint(root/'harm/final.pt')
            self.assertEqual(checkpoint['scope'], RELATIVE_HARM_SCOPE)
            checkpoint['loss_semantics'] = 'mean_valid_position_linear_haversine_m_div_1000_all_queries_no_auxiliary'
            torch.save(checkpoint, root/'bad.pt')
            with self.assertRaisesRegex(ValueError, 'contract'):
                load_checkpoint(root/'bad.pt')
            old[0]['batch'][0]['node_id'] = 'tampered'
            (control/'training.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in old))
            with self.assertRaisesRegex(ValueError, 'before update 1'):
                train_sequence_position(train, harm, root/'bad', paired_reference_fit=control)
            failed = json.loads((root/'bad/run.json').read_text())
            self.assertEqual(failed['status'], 'FAILED')
            self.assertEqual(failed['optimizer_updates'], 0)
            self.assertEqual(failed['exposure']['sampled_anchors'], 0)
            self.assertEqual(failed['paired_control']['actual_successful_batches_matched'], 0)
            self.assertFalse((root/'bad/final.pt').exists())
        with self.assertRaisesRegex(ValueError, 'requires'):
            train_sequence_position(train, harm, 'unused-without-control')
        harm['scope'] = SCOPE
        with self.assertRaises(ValueError):
            configuration(harm)


if __name__ == '__main__':
    unittest.main()
