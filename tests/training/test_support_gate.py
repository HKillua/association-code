import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from association.data.motion_evidence import scene_motion_evidence
from association.evaluation.fusion_controls import ordinary_fusion
from association.models.support_gate import (ROUTING_SEMANTICS, SCOPE, SupportGate,
    gate_inputs, mix_gate_states, support_gate_loss, support_gate_objectives)
from association.training.support_gate import configuration, load_checkpoint, train_support_gate
from association.training.support_gate import LOSS_SEMANTICS, LINEAR_LOSS_SEMANTICS
from tools.experiments.evidence_fusion_fit import synthetic_scenes


def fixture_motion():
    states = np.array([
        [[20., 179., 2., 350.], [20., 179., 2., 350.], [20., 179., 2., 350.]],
        [[20.01, -179., 8., 10.], [20.01, -179., 8., 10.], [20.01, -179., 8., 10.]],
        [[20.02, -177., 6., 20.], [20.02, -177., 6., 20.], [20.02, -177., 6., 20.]]])
    return dict(states=states, support=np.array([[True, False, False], [True, True, False], [False, True, False]]),
        nearest_seconds=np.array([[0., 20., 40.], [0., 0., 20.], [20., 0., 20.]]),
        exact_observation=np.zeros((3, 3), bool), native_count=np.array([2, 4, 3]),
        span_seconds=np.array([10., 40., 20.]), fit_rmse_m=np.array([0., 10., 20.]),
        ols_velocity_mps=np.array([[2., 3.], [2., 3.], [3., 2.]]),
        endpoint_velocity_mps=np.array([[2., 3.], [3., 2.], [2., 3.]]))


def tiny_config():
    return dict(scope=SCOPE, routing_semantics=ROUTING_SEMANTICS,
        data=dict(length=64, max_observations=128, time_unit_seconds=20., reference_longitude_policy='shortest_arc'),
        train_splits=['T_core'], model=dict(initial_gate=.95), motion_observation_policy='canonical_all',
        training=dict(seed=20261021, updates=3, batch_size=2, learning_rate=.001,
                      gradient_clip=1., advisory_seconds=3600.))


def tiny_train():
    train, _ = synthetic_scenes(length=64, max_observations=4)
    for scene in train:
        scene['tracks'] = [dict(node_id=node_id, source=scene['sources'][node],
            canonical_values=scene['native_values'][node, scene['native_observed'][node]],
            canonical_times=scene['native_times'][node, scene['native_observed'][node]])
            for node, node_id in enumerate(scene['node_ids'])]
        scene['motion'] = scene_motion_evidence(scene, motion_observation_policy='canonical_all')
        scene['motion_observation_policy'] = 'canonical_all'
        del scene['tracks']
        # A definite, finite position learning signal independent of other attrs.
        for target in scene['fusion_targets'].values():
            target['states'][:, 0] += .02
    return train


class SupportGateTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_visible_features_mean_endpoint_and_helper_order(self):
        motion = fixture_motion()
        class VisibleOnly(dict):
            def __getitem__(self, key):
                if key in ('target', 'reference', 'identities', 'fusion_targets'):
                    raise AssertionError('truth must never enter forward input preparation')
                return super().__getitem__(key)
        visible = VisibleOnly(motion, identities=object(), reference=object())
        first = gate_inputs(visible, [0, 1, 2])
        second = gate_inputs(visible, [0, 2, 1])
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(first, second)))
        expected = ordinary_fusion(motion['states'], motion['nearest_seconds'], motion['support'], [[0, 1, 2]])
        self.assertTrue(np.array_equal(first[1].numpy(), expected['predictions']['mean_supported'][0]))
        self.assertEqual(first[2].shape, (3, 56))
        self.assertEqual(first[3].tolist(), [False, True, False])
        self.assertTrue(torch.equal(first[2][2], torch.zeros(56)))
        result = SupportGate()(*first)
        self.assertTrue(torch.allclose(result['gate'], torch.full((3,), .95), atol=1e-7, rtol=0))
        self.assertTrue(torch.equal(result['states'][[0, 2], :2], first[0][[0, 2], :2]))
        self.assertTrue(torch.equal(result['states'][:, 2:], first[1][:, 2:]))
        # An alternative anchor remains first; its helpers are sorted separately.
        third = gate_inputs(motion, [2, 1, 0])
        self.assertTrue(torch.equal(third[0], torch.tensor(motion['states'][2])))
        with self.assertRaises(ValueError):
            gate_inputs(motion, [-1, 1])
        with self.assertRaises(ValueError):
            gate_inputs(motion, [0., 1.])

    def test_exact_endpoints_dateline_and_singleton(self):
        anchor = torch.tensor([[10., 179., 1., 20.], [10., -179., 1., 20.]], dtype=torch.float64)
        mean = torch.tensor([[12., -179., 7., 30.], [12., 179., 7., 30.]], dtype=torch.float64)
        eligible = torch.ones(2, dtype=torch.bool)
        for value, expected in ((0., anchor[:, :2]), (1., mean[:, :2])):
            result = mix_gate_states(anchor, mean, torch.full((2,), value), eligible)
            self.assertTrue(torch.equal(result[:, :2], expected))
            self.assertTrue(torch.equal(result[:, 2:], mean[:, 2:]))
        midpoint = mix_gate_states(anchor, mean, torch.full((2,), .5), eligible)
        self.assertTrue(torch.equal(midpoint[:, 1], torch.full((2,), -180., dtype=torch.float64)))
        motion = fixture_motion()
        arguments = gate_inputs(motion, [0])
        result = SupportGate()(*arguments)
        self.assertFalse(result['eligible'].any())
        self.assertTrue(torch.equal(result['states'], arguments[0]))
        self.assertTrue(torch.equal(arguments[2], torch.zeros((3, 56))))

    def test_loss_keeps_all_valid_queries_and_only_position_has_gradient(self):
        arguments = gate_inputs(fixture_motion(), [0, 1, 2])
        model = SupportGate()
        result = model(*arguments)
        target = arguments[0].clone()
        target[:, 2:] = float('nan')
        valid = torch.ones_like(target, dtype=torch.bool)
        loss = support_gate_loss(result, target, valid)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertGreater(float(model.head[-1].bias.grad.abs().sum()), 0.)
        from association.models.evidence_fusion import physical_errors
        safe = torch.nan_to_num(target)
        expected = torch.log1p(physical_errors(result['states'], safe)[:, 0] / 1000.).mean()
        self.assertTrue(torch.equal(loss, expected))
        no_truth = torch.zeros_like(valid)
        model.zero_grad(set_to_none=True)
        empty = support_gate_loss(model(*arguments), target, no_truth)
        empty.backward()
        self.assertEqual(float(empty.detach()), 0.)
        self.assertTrue(all(torch.count_nonzero(p.grad) == 0 for p in model.parameters()))

    def test_training_audit_fixed_endpoint_and_strict_checkpoint(self):
        config = tiny_config()
        train = tiny_train()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)/'fit'
            report = train_support_gate(train, config, root)
            self.assertEqual(report['status'], 'COMPLETED')
            self.assertEqual(report['optimizer_updates'], 3)
            audit = report['exposure']
            self.assertEqual(audit['successful_updates'], 3)
            self.assertEqual(audit['sampled_anchors'], 6)
            self.assertEqual(sum(audit['scenes'].values()), 6)
            self.assertEqual(sum(audit['anchors'].values()), 6)
            self.assertTrue(all(row['updates'] == 1 for row in audit['modes'].values()))
            logs = [json.loads(line) for line in (root/'training.jsonl').read_text().splitlines()]
            self.assertEqual(len(logs), 3)
            self.assertEqual(sum(row['eligible_query_points'] for log in logs for row in log['batch']),
                             audit['eligible_query_points'])
            model, state = load_checkpoint(root/'final.pt')
            initial, _ = load_checkpoint(root/'step-0.pt')
            self.assertEqual(state['step'], 3)
            self.assertTrue(any(not torch.equal(a, b) for a, b in zip(model.parameters(), initial.parameters())))
            repeat = Path(directory)/'repeat'
            other_report = train_support_gate(train, config, repeat)
            other, _ = load_checkpoint(repeat/'final.pt')
            self.assertEqual(audit, other_report['exposure'])
            self.assertTrue(all(torch.equal(a, b) for a, b in zip(model.parameters(), other.parameters())))
            for key, value in (('routing_semantics', 'legacy_relu'), ('scope', 'native-evidence-fusion-v1'),
                               ('loss_semantics', 'helper-quality'), ('step', 4), ('actual_updates', 2)):
                bad = copy.deepcopy(state)
                bad[key] = value
                torch.save(bad, root/'bad.pt')
                with self.subTest(key=key), self.assertRaises(ValueError):
                    load_checkpoint(root/'bad.pt')
            bad = copy.deepcopy(state)
            bad['configuration']['model']['initial_gate'] = .99
            torch.save(bad, root/'bad.pt')
            with self.assertRaises(ValueError):
                load_checkpoint(root/'bad.pt')
            with self.assertRaises(FileExistsError):
                train_support_gate(train, config, root)

    def test_linear_changes_only_error_transform_with_identical_valid_denominator(self):
        from association.models.evidence_fusion import physical_errors
        args = gate_inputs(fixture_motion(), [0, 1, 2])
        model = SupportGate()
        target = args[0].clone()
        target[0, :2] += .1  # R-exterior error still belongs in the denominator.
        target[2, :2] = float('nan')
        target[:, 2:] = float('nan')
        valid = torch.ones_like(target, dtype=torch.bool)
        valid[2, :2] = False
        result = model(*args)
        error = physical_errors(result['states'], torch.nan_to_num(target))[:2, 0]
        for transform in ('log1p', 'linear'):
            with self.subTest(transform=transform):
                expected = (torch.log1p(error / 1000.) if transform == 'log1p' else error / 1000.).mean()
                loss = support_gate_loss(result, target, valid, position_loss=transform)
                self.assertTrue(torch.equal(loss, expected))
                got = torch.autograd.grad(loss, model.head[-1].bias, retain_graph=True)[0]
                want = torch.autograd.grad(expected, model.head[-1].bias, retain_graph=True)[0]
                self.assertTrue(torch.equal(got, want))
                self.assertGreater(float(got.abs().sum()), 0.)
        with self.assertRaisesRegex(ValueError, 'position_loss'):
            support_gate_loss(result, target, valid, position_loss='automatic')

    def test_paired_linear_preserves_initialization_sampling_and_checkpoint_meaning(self):
        config, train = tiny_config(), tiny_train()
        linear = dict(config, position_loss='linear')
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline = train_support_gate(train, config, root/'baseline')
            trained = train_support_gate(train, linear, root/'linear', paired_reference_fit=root/'baseline')
            control = train_support_gate(train, config, root/'control', paired_reference_fit=root/'baseline')
            base_log = [json.loads(row) for row in (root/'baseline/training.jsonl').read_text().splitlines()]
            new_log = [json.loads(row) for row in (root/'linear/training.jsonl').read_text().splitlines()]
            control_log = [json.loads(row) for row in (root/'control/training.jsonl').read_text().splitlines()]
            _, base_initial = load_checkpoint(root/'baseline/step-0.pt')
            _, new_initial = load_checkpoint(root/'linear/step-0.pt')
            _, base_final = load_checkpoint(root/'baseline/final.pt')
            _, new_final = load_checkpoint(root/'linear/final.pt')
            _, control_final = load_checkpoint(root/'control/final.pt')
            self.assertTrue(all(torch.equal(base_initial['model'][key], new_initial['model'][key])
                                for key in base_initial['model']))
            self.assertTrue(torch.equal(base_initial['torch_rng'], new_initial['torch_rng']))
            self.assertEqual(base_initial['numpy_rng'], new_initial['numpy_rng'])
            # Reading reference checkpoints/adding diagnostics cannot alter a log control.
            self.assertTrue(all(torch.equal(base_final['model'][key], control_final['model'][key])
                                for key in base_final['model']))
            self.assertEqual(baseline['exposure'], control['exposure'])
            self.assertTrue(torch.equal(base_final['torch_rng'], control_final['torch_rng']))
            for old, new, replay in zip(base_log, new_log, control_log):
                self.assertEqual(set(old), {'step', 'mode', 'loss', 'gradient_norm', 'batch'})
                self.assertEqual(old['batch'], new['batch'])
                self.assertEqual(old['mode'], new['mode'])
                self.assertEqual(old, {key: replay[key] for key in old})
                self.assertAlmostEqual(new['loss'], new['objectives']['anchor_mean_linear'], places=14)
                self.assertGreaterEqual(new['objectives']['anchor_mean_linear'],
                                        new['objectives']['anchor_mean_log1p'])
                self.assertTrue(0 <= new['gradient_clip_scale'] <= 1)
            self.assertEqual(trained['pairing']['matched_updates'], 3)
            self.assertEqual(trained['pairing']['matched_anchors'], 6)
            self.assertEqual(trained['gradient_clipping']['clipped_updates'],
                             sum(row['gradient_clipped'] for row in new_log))
            self.assertEqual(new_final['loss_semantics'], LINEAR_LOSS_SEMANTICS)
            self.assertEqual(base_final['loss_semantics'], LOSS_SEMANTICS)
            self.assertTrue(any(not torch.equal(base_final['model'][key], new_final['model'][key])
                                for key in base_final['model']))
            for mutate in (lambda state: state.update(loss_semantics=LOSS_SEMANTICS),
                           lambda state: state['configuration'].pop('position_loss')):
                bad = copy.deepcopy(new_final)
                mutate(bad)
                torch.save(bad, root/'bad.pt')
                with self.assertRaisesRegex(ValueError, 'loss semantics'):
                    load_checkpoint(root/'bad.pt')

    def test_diagnostics_saturation_counts_exclude_invalid_or_ineligible_queries(self):
        args = gate_inputs(fixture_motion(), [0, 1, 2])
        target = args[0].clone()
        target[2, :2] = float('nan')
        valid = torch.ones_like(target, dtype=torch.bool)
        valid[2, :2] = False
        # The only valid opportunity is query 1. Query 0 has support, while
        # query 2 has no truth, so their saturated weights must not be counted.
        for value in (0., 1.):
            gates = torch.tensor([value, value, value])
            eligible = torch.tensor([False, True, True])
            result = dict(states=mix_gate_states(args[0], args[1], gates, eligible),
                          gate=gates, eligible=eligible, logits=torch.zeros(3, requires_grad=True))
            _, metrics = support_gate_objectives(result, target, valid)
            self.assertEqual(metrics['valid_position_points'], 2)
            self.assertEqual(metrics['eligible_valid_position_points'], 1)
            self.assertEqual(metrics['eligible_valid_gate_zero_count'], int(value == 0))
            self.assertEqual(metrics['eligible_valid_gate_one_count'], int(value == 1))
            self.assertEqual(metrics['eligible_valid_gate_min'], value)
            self.assertEqual(metrics['eligible_valid_gate_max'], value)

    def test_pairing_failures_stop_before_any_optimizer_update(self):
        config, train = tiny_config(), tiny_train()
        linear = dict(config, position_loss='linear')
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train_support_gate(train, config, root/'baseline')
            source = root/'baseline/run.json'
            original_run = json.loads(source.read_text())
            for index, mutate in enumerate((lambda row: row.update(status='FAILED'),
                    lambda row: row.update(optimizer_updates=2),
                    lambda row: row.update(test_opened=True),
                    lambda row: row.update(training_scene_ids=['different']),
                    lambda row: row['configuration']['training'].update(learning_rate=.002))):
                bad = copy.deepcopy(original_run)
                mutate(bad)
                source.write_text(json.dumps(bad))
                with patch('association.training.support_gate.prepare_evidence_scenes') as prepare, \
                        patch('torch.optim.Adam.step') as optimize:
                    with self.assertRaisesRegex(ValueError, 'paired reference'):
                        train_support_gate(train, linear, root/f'bad-source-{index}', paired_reference_fit=source.parent)
                prepare.assert_not_called()
                optimize.assert_not_called()
            source.write_text(json.dumps(original_run))
            initial = torch.load(source.parent/'step-0.pt', weights_only=False)
            altered = copy.deepcopy(initial)
            altered['model']['head.0.weight'][0, 0] += 1.
            torch.save(altered, source.parent/'step-0.pt')
            with patch('association.training.support_gate.prepare_evidence_scenes') as prepare:
                with self.assertRaisesRegex(ValueError, 'initial model parameters'):
                    train_support_gate(train, linear, root/'bad-initial', paired_reference_fit=source.parent)
            prepare.assert_not_called()
            torch.save(initial, source.parent/'step-0.pt')
            rows = [json.loads(line) for line in (source.parent/'training.jsonl').read_text().splitlines()]
            # Slot zero may accumulate a gradient; slot one must abort the whole update.
            rows[0]['batch'][1]['eligible_valid_position_points'] += 1
            (source.parent/'training.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
            with patch('torch.optim.Adam.step') as optimize:
                with self.assertRaisesRegex(ValueError, 'sampling mismatch before optimizer update 1, slot 1'):
                    train_support_gate(train, linear, root/'bad-batch', paired_reference_fit=source.parent)
            optimize.assert_not_called()
            failure = json.loads((root/'bad-batch/run.json').read_text())
            self.assertEqual(failure['status'], 'FAILED')
            self.assertEqual(failure['optimizer_updates'], 0)
            self.assertEqual(failure['exposure']['sampled_anchors'], 0)
            self.assertEqual(failure['pairing']['matched_updates'], 0)
            self.assertFalse((root/'bad-batch/training.jsonl').exists())
            self.assertFalse((root/'bad-batch/final.pt').exists())

    def test_zero_eligible_updates_are_not_silently_skipped(self):
        train = tiny_train()
        train[0]['motion']['support'][:] = True
        config = tiny_config()
        config['training']['updates'] = 1
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)/'fit'
            report = train_support_gate(train, config, root)
            audit = report['exposure']
            self.assertEqual(report['optimizer_updates'], 1)
            self.assertEqual(audit['zero_eligible_updates'], 1)
            self.assertEqual(audit['nonzero_gradient_updates'], 0)
            self.assertEqual(audit['sampled_anchors'], config['training']['batch_size'])
            model, state = load_checkpoint(root/'final.pt')
            initial, _ = load_checkpoint(root/'step-0.pt')
            self.assertTrue(all(torch.equal(a, b) for a, b in zip(model.parameters(), initial.parameters())))
            self.assertTrue(all(int(value['step']) == 1 for value in state['optimizer']['state'].values()))

    def test_failed_optimizer_does_not_count_exposure_and_no_validation_input(self):
        train, config = tiny_train(), tiny_config()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)/'failed'
            with patch('torch.optim.Adam.step', side_effect=RuntimeError('synthetic optimizer failure')):
                with self.assertRaisesRegex(RuntimeError, 'synthetic optimizer failure'):
                    train_support_gate(train, config, root)
            report = json.loads((root/'run.json').read_text())
            self.assertEqual(report['status'], 'FAILED')
            self.assertEqual(report['optimizer_updates'], 0)
            self.assertEqual(report['exposure']['sampled_anchors'], 0)
            self.assertFalse((root/'final.pt').exists())
            self.assertTrue((root/'interrupted.pt').exists())
            validation = copy.deepcopy(train)
            validation[0]['split'] = 'V_select'
            with self.assertRaisesRegex(ValueError, 'T_core'):
                train_support_gate(validation, config, Path(directory)/'wrong-split')
            self.assertEqual(json.loads((Path(directory)/'wrong-split/run.json').read_text())['status'], 'FAILED')
            bad = copy.deepcopy(config)
            bad['routing_semantics'] = 'unspecified'
            with self.assertRaises(ValueError):
                configuration(bad)


if __name__ == '__main__':
    unittest.main()
