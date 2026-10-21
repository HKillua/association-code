import copy
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
import numpy as np

from association.data.motion_evidence import scene_motion_evidence
from association.models.cooperative import CooperativeConfig
from association.models.evidence_fusion import evidence_fusion_loss, fusion_inputs
from association.models.evidence_association import features_from_motion
from association.evaluation.grouping import group_predicted_tracks
from association.training.evidence_fusion import (load_checkpoint, prepare_evidence_scenes,
    train_evidence_fusion, training_objective, validate_training_population, evaluate_evidence_fusion,
    bind_ols_working_point)
from tools.experiments.cooperative_fit import synthetic_scenes
from tools.experiments.evidence_fusion_fit import configuration, main


class EvidenceFusionTrainingTests(unittest.TestCase):
    def test_three_auxiliary_objectives_share_initial_state_and_record_training_only_diagnostics(self):
        torch.set_num_threads(1)
        train, validation = synthetic_scenes(CooperativeConfig(length=64, max_observations=4))
        base = json.loads(Path('configs/evidence-fusion.json').read_text())
        base['model']['utility_output_bias'] = .01
        base['training'].update(updates=3, batch_size=1, evaluate_every=3,
                                mixed_action_alpha=.5, utility_sign_weight=0.)
        initials, samples = [], []
        with tempfile.TemporaryDirectory() as directory:
            for objective in ('helper_quality', 'direct_only', 'mixed_action'):
                config = copy.deepcopy(base)
                config['training']['utility_objective'] = objective
                configuration(config)
                output = Path(directory)/objective
                report = train_evidence_fusion(train, validation, config, output)
                model, state = load_checkpoint(output/'final.pt')
                initial, _ = load_checkpoint(output/'step-0.pt')
                initials.append(initial.state_dict())
                samples.append(report['training_samples'])
                self.assertTrue(any(not torch.equal(a, b) for a, b in zip(initial.parameters(), model.parameters())))
                self.assertEqual(state['loss_semantics']['objective'], objective)
                self.assertEqual(model.utility_output_bias, .01)
                stats = report['training_diagnostics']
                self.assertEqual(stats['split'], 'T_core')
                self.assertEqual(len(stats['by_mode']), 3)
                self.assertTrue(all(row['gain_count'] > 0 for mode in stats['by_mode'].values()
                                    for row in mode.values()))
                self.assertTrue(all(row['active_helper_query_count'] <= row['helper_query_count']
                                    for mode in stats['by_mode'].values() for row in mode.values()))
                if objective == 'direct_only':
                    logs = [json.loads(row) for row in (output/'training.jsonl').read_text().splitlines()]
                    self.assertTrue(all(row['utility'] == row['utility_sign'] == 0. for row in logs))
                    # V truth is used only in monitoring; it cannot affect updates or T-only diagnostics.
                    altered = copy.deepcopy(validation)
                    for scene in altered:
                        for target in scene['fusion_targets'].values():
                            target['states'][:, 2] += 37.
                    other = Path(directory)/'direct_altered_V'
                    repeat = train_evidence_fusion(train, altered, config, other)
                    clone, _ = load_checkpoint(other/'final.pt')
                    self.assertEqual(stats, repeat['training_diagnostics'])
                    self.assertTrue(all(torch.equal(a, b) for a, b in zip(model.parameters(), clone.parameters())))
                for mutation in ('bias', 'objective', 'missing_semantics'):
                    bad = copy.deepcopy(state)
                    if mutation == 'bias':
                        bad['configuration']['model']['utility_output_bias'] = .02
                    elif mutation == 'objective':
                        bad['configuration']['training']['utility_objective'] = (
                            'mixed_action' if objective != 'mixed_action' else 'helper_quality')
                    else:
                        del bad['loss_semantics']
                    torch.save(bad, output/'bad.pt')
                    with self.assertRaises(ValueError):
                        load_checkpoint(output/'bad.pt')
            self.assertTrue(all(samples[0] == counts for counts in samples[1:]))
            self.assertTrue(all(all(torch.equal(state[key], initials[0][key]) for key in state)
                                for state in initials[1:]))

    def test_auxiliary_configuration_rejects_invalid_or_dead_direct_controls(self):
        base = json.loads(Path('configs/evidence-fusion.json').read_text())
        for key, value in (('utility_objective', 'unknown'), ('mixed_action_alpha', 0),
                           ('mixed_action_alpha', 1), ('mixed_action_alpha', True),
                           ('mixed_action_alpha', float('nan'))):
            config = copy.deepcopy(base)
            config['training'][key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                configuration(config)
        direct = copy.deepcopy(base)
        direct['training']['utility_objective'] = 'direct_only'
        with self.assertRaisesRegex(ValueError, 'positive utility_output_bias'):
            configuration(direct)
        direct['model']['utility_output_bias'] = .01
        direct['training']['utility_sign_weight'] = .1
        with self.assertRaisesRegex(ValueError, 'direct_only'):
            configuration(direct)

    def test_current_ols_working_point_preserves_edges_and_rejects_other_protocols(self):
        train, validation = synthetic_scenes(CooperativeConfig(length=64, max_observations=4))
        config = configuration(json.loads(Path('configs/evidence-fusion.json').read_text()))
        manifest = dict(dataset_level='Level1', splits=dict(
            T_core=[s['scene_id'] for s in train], V_select=[s['scene_id'] for s in validation]))
        scene = validation[0]
        motion = scene_motion_evidence(scene)
        pairs = np.column_stack(np.triu_indices(len(scene['node_ids']), 1))
        _, scores = features_from_motion(motion, pairs)
        record = dict(scope='ols-working-point-v1', synthetic=False, dataset_level='Level1',
            motion_observation_policy='selected_cap128',
            data={key: config['data'][key] for key in ('length', 'max_observations', 'time_unit_seconds')},
            training_scene_ids=manifest['splits']['T_core'], validation_scene_ids=manifest['splits']['V_select'],
            score_semantics='-log1p(rms_haversine_m/1000)', selection_split='V_select', threshold=float(scores[0]))
        # Include an exact-score tie and the explicit all-reject operating point.
        for threshold in (float(scores[0]), None):
            point = dict(record, threshold=threshold)
            bound = configuration(bind_ols_working_point(config, manifest, point))
            prepared = prepare_evidence_scenes(validation, grouping=bound['grouping'])
            expected = group_predicted_tracks(len(scene['node_ids']), pairs, scores,
                threshold=threshold, grouping_policy='anchor_direct', node_order='input')
            self.assertEqual(prepared[0]['groups']['predicted_anchor_direct'], expected)
        for key, value in (('dataset_level', 'Level2'), ('validation_scene_ids', ['another']),
                           ('motion_observation_policy', 'canonical_all'), ('synthetic', True)):
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'current dataset'):
                bind_ols_working_point(config, manifest, dict(record, **{key: value}))
        config = bind_ols_working_point(config, manifest, record)
        config['training'].update(updates=1, batch_size=1, evaluate_every=1)
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)/'fit'
            train_evidence_fusion(train, validation, config, output)
            model, _ = load_checkpoint(output/'final.pt')
            prepared = prepare_evidence_scenes(validation, grouping=config['grouping'])
            report = evaluate_evidence_fusion(model, prepared, device='cpu')
            self.assertEqual(report['grouping'], config['grouping'])
            prepared[0]['grouping'] = None
            with self.assertRaisesRegex(ValueError, 'OLS grouping'):
                evaluate_evidence_fusion(model, prepared, device='cpu')

    def test_level1_cli_requires_new_working_point_before_opening_archive(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'manifest.json').write_text(json.dumps(dict(dataset_level='Level1')))
            with patch('tools.experiments.evidence_fusion_fit.load_timed_dataset') as loader:
                code = main(['--mode', 'fit', '--archive', 'part1.zip', 'part2.zip',
                    '--manifest', str(root/'manifest.json'), '--alignment', 'unused.json',
                    '--config', 'configs/evidence-fusion.json', '--output', str(root/'fit')])
                self.assertEqual(code, 1)
                loader.assert_not_called()
                report = json.loads((root/'fit/run.json').read_text())
                self.assertIn('--ols-working-point', report['error'])
                self.assertEqual(report['optimizer_updates'], 0)

    def test_canonical_motion_fit_reload_and_wrong_policy_rejection(self):
        torch.set_num_threads(1)
        train, validation = synthetic_scenes(CooperativeConfig(length=64, max_observations=4))
        for scene in train + validation:
            scene['tracks'] = [dict(node_id=node_id, source=scene['sources'][node],
                canonical_values=scene['native_values'][node, scene['native_observed'][node]],
                canonical_times=scene['native_times'][node, scene['native_observed'][node]])
                for node, node_id in enumerate(scene['node_ids'])]
            scene['motion'] = scene_motion_evidence(scene, motion_observation_policy='canonical_all')
            scene['motion_observation_policy'] = 'canonical_all'
            del scene['tracks']
        config = json.loads(Path('configs/evidence-fusion.json').read_text())
        config['motion_observation_policy'] = 'canonical_all'
        config['training'].update(updates=1, batch_size=1, evaluate_every=1)
        configuration(config)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)/'canonical'
            report = train_evidence_fusion(train, validation, config, output)
            self.assertEqual(report['motion_observation_policy'], 'canonical_all')
            model, saved = load_checkpoint(output/'final.pt')
            prepared = prepare_evidence_scenes(validation, motion_observation_policy='canonical_all')
            evaluation = evaluate_evidence_fusion(model, prepared, device='cpu')
            self.assertEqual(evaluation['fragment_count'], 4)
            prepared[0]['motion_observation_policy'] = 'selected_cap128'
            with self.assertRaisesRegex(ValueError, 'motion_observation_policy'):
                evaluate_evidence_fusion(model, prepared, device='cpu')
            saved['configuration'].pop('motion_observation_policy')
            torch.save(saved, output/'wrong-policy.pt')
            with self.assertRaisesRegex(ValueError, 'motion_observation_policy'):
                load_checkpoint(output/'wrong-policy.pt')

    def test_three_modes_update_checkpoint_and_full_population(self):
        torch.set_num_threads(1)
        train, validation = synthetic_scenes(CooperativeConfig(length=64, max_observations=4))
        config = configuration(json.loads(Path('configs/evidence-fusion.json').read_text()))
        config['training'].update(updates=3, batch_size=2, evaluate_every=3)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)/'fit'
            report = train_evidence_fusion(train, validation, config, output)
            self.assertEqual(report['status'], 'COMPLETED')
            self.assertEqual(report['optimizer_updates'], 3)
            for counts in report['training_samples'].values():
                self.assertEqual(counts['anchors'], 2)
            model, state = load_checkpoint(output/'final.pt')
            initial, _ = load_checkpoint(output/'step-0.pt')
            self.assertEqual(state['step'], 3)
            self.assertEqual(state['loss_semantics']['utility_sign_weight'], 0.)
            self.assertEqual(report['split_unit'], 'shared_a_component')
            legacy = copy.deepcopy(state)
            del legacy['loss_semantics']
            torch.save(legacy, output/'legacy.pt')
            old, _ = load_checkpoint(output/'legacy.pt')
            self.assertTrue(all(torch.equal(a, b) for a, b in zip(model.parameters(), old.parameters())))
            self.assertTrue(any(not torch.equal(a, b)
                                for a, b in zip(model.parameters(), initial.parameters())))
            scene = prepare_evidence_scenes(validation)[0]
            arguments = fusion_inputs(scene['motion'], [0, 1])
            prediction = model(*arguments)['states']
            clone, _ = load_checkpoint(output/'final.pt')
            self.assertTrue(torch.equal(prediction, clone(*arguments)['states']))
            evaluation = json.loads((output/'evaluation-step-3.json').read_text())
            self.assertEqual(evaluation['fragment_count'], 4)
            self.assertEqual(evaluation['query_count'], 256)
            self.assertEqual(len(evaluation['per_anchor']), 4)
            self.assertFalse(evaluation['test_opened'])
            self.assertTrue((output/'predictions-final/states-synthetic-validation.npz').exists())


    def test_sign_auxiliary_uses_original_attribute_mask_and_equal_attribute_reduction(self):
        base = torch.zeros((3, 2, 4), dtype=torch.float64)
        base[0, :, 2], base[0, :, 3] = 10., 30.
        # Equal speed error gives exactly y=0 and must receive the nonpositive label.
        base[2, :, 2], base[2, :, 3] = 10., 60.
        target = torch.full((2, 4), float('nan'), dtype=torch.float64)
        valid = torch.zeros((2, 4), dtype=torch.bool)
        valid[:, 2], valid[1, 3] = True, True
        target[valid] = 0.
        logits = torch.zeros((2, 2, 3), requires_grad=True)
        result = dict(states=base[0], utility=logits)
        original = evidence_fusion_loss(result, base, target, valid)
        # Weight zero performs exactly the old computation, without even evaluating BCE.
        with patch('torch.nn.functional.binary_cross_entropy_with_logits', side_effect=AssertionError('disabled')):
            default = training_objective(result, base, target, valid)
            explicit_zero = training_objective(result, base, target, valid, sign_weight=0.)
        self.assertTrue(torch.equal(default['total'], original['total']))
        self.assertTrue(torch.equal(explicit_zero['total'], original['total']))
        expected_gradient = torch.autograd.grad(original['total'], logits, retain_graph=True)[0]
        actual_gradient = torch.autograd.grad(default['total'], logits, retain_graph=True)[0]
        self.assertTrue(torch.equal(expected_gradient, actual_gradient))
        active = training_objective(result, base, target, valid, sign_weight=.1)
        self.assertAlmostEqual(float(active['utility_sign'].detach()), math.log(2.), places=6)
        self.assertAlmostEqual(float(active['utility_sign_weighted'].detach()), .1*math.log(2.), places=6)
        self.assertTrue(torch.equal(active['total'], active['original_total']+active['utility_sign_weighted']))
        gradient = torch.autograd.grad(active['utility_sign'], logits)[0]
        expected = torch.zeros_like(logits)
        expected[0, :, 1], expected[1, :, 1] = -.0625, .0625
        expected[0, 1, 2], expected[1, 1, 2] = -.125, .125
        torch.testing.assert_close(gradient, expected, rtol=0, atol=0)
        # Empty helpers and all-masked targets do not manufacture binary labels.
        for count, reference_mask in ((1, valid), (3, torch.zeros_like(valid))):
            u = torch.zeros((count-1, 2, 3), requires_grad=True)
            loss = training_objective(dict(states=base[0], utility=u), base[:count],
                                      target, reference_mask, sign_weight=.1)
            self.assertEqual(float(loss['utility_sign'].detach()), 0.)
            self.assertTrue(torch.equal(torch.autograd.grad(loss['utility_sign'], u)[0], torch.zeros_like(u)))

    def test_sign_candidate_updates_logs_and_checkpoint_loss_semantics(self):
        torch.set_num_threads(1)
        train, validation = synthetic_scenes(CooperativeConfig(length=64, max_observations=4))
        config = configuration(json.loads(Path('configs/evidence-fusion-sign.json').read_text()))
        reference = configuration(json.loads(Path('configs/evidence-fusion.json').read_text()))
        only_change = copy.deepcopy(config)
        del only_change['training']['utility_sign_weight']
        self.assertEqual(only_change, reference)
        self.assertEqual(config['training']['updates'], 1000)
        config['training'].update(updates=1, batch_size=1, evaluate_every=1)
        # Explicit scene isolation permits shared A components without scene overlap.
        for scene in [*train, *validation]:
            scene['split_unit'] = 'scene'
            scene['component_id'] = 'shared-across-different-scenes'
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)/'fit'
            report = train_evidence_fusion(train, validation, config, output)
            self.assertEqual(report['split_unit'], 'scene')
            model, state = load_checkpoint(output/'final.pt')
            self.assertEqual(state['loss_semantics']['utility_sign_weight'], .1)
            self.assertTrue(state['loss_semantics']['utility_sign_enabled'])
            log = json.loads((output/'training.jsonl').read_text())
            self.assertGreater(log['utility_sign'], 0.)
            self.assertAlmostEqual(log['utility_sign_weighted'], .1*log['utility_sign'], places=7)
            self.assertAlmostEqual(log['total'], log['original_total']+log['utility_sign_weighted'], places=7)
            initial, _ = load_checkpoint(output/'step-0.pt')
            self.assertTrue(any(not torch.equal(a, b) for a, b in zip(model.parameters(), initial.parameters())))
            for kind in ('missing', 'conflicting'):
                bad = copy.deepcopy(state)
                if kind == 'missing':
                    del bad['loss_semantics']
                else:
                    bad['configuration']['training']['utility_sign_weight'] = 0.
                torch.save(bad, output/(kind+'.pt'))
                with self.assertRaisesRegex(ValueError, 'loss_semantics'):
                    load_checkpoint(output/(kind+'.pt'))
        for invalid in (-.1, True, float('nan'), float('inf')):
            bad = copy.deepcopy(reference)
            bad['training']['utility_sign_weight'] = invalid
            with self.assertRaisesRegex(ValueError, 'utility_sign_weight'):
                configuration(bad)

    def test_scene_and_component_isolation_are_explicit_and_cannot_be_mixed(self):
        train, validation = synthetic_scenes(CooperativeConfig(length=64, max_observations=4))
        self.assertEqual(validate_training_population(train, validation), 'shared_a_component')
        validation[0]['component_id'] = train[0]['component_id']
        with self.assertRaisesRegex(ValueError, 'component-isolated'):
            validate_training_population(train, validation)
        for scene in [*train, *validation]:
            scene['split_unit'] = 'scene'
        self.assertEqual(validate_training_population(train, validation), 'scene')
        self.assertEqual(prepare_evidence_scenes(train)[0]['split_unit'], 'scene')
        del validation[0]['split_unit']
        with self.assertRaisesRegex(ValueError, 'unmixed'):
            validate_training_population(train, validation)
        validation[0]['split_unit'] = 'scene'
        validation[0]['scene_id'] = train[0]['scene_id']
        with self.assertRaisesRegex(ValueError, 'scene IDs'):
            validate_training_population(train, validation)


if __name__ == '__main__':
    unittest.main()
