"""071 complete frozen-population evaluation and real tiny endpoint chain."""
from argparse import Namespace
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import torch

from association.evaluation.sequence_position import METHODS, evaluate_sequence_position, compare_endpoints, _weighted
from association.models.sequence_position import SequencePosition
from association.utils.runtime import read_json, write_json
from tests.fixtures.cache import support_gate_cache_fixture


class SequencePositionEvaluationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def evaluate(self, root, model, *, sparse=False, initial=False):
        source, metadata, validation = support_gate_cache_fixture(root, sparse=sparse)
        with patch('association.evaluation.sequence_position.frozen_motion', side_effect=lambda raw, *_: raw):
            result = evaluate_sequence_position(model, validation, source, metadata, root/'result', initial=initial)
        return result, source, metadata, validation

    def test_zero_initial_all_groups_truth_after_forward_and_full_population(self):
        for backbone in ('lstm', 'transformer'):
            with self.subTest(backbone=backbone), TemporaryDirectory() as directory:
                root = Path(directory)
                model = SequencePosition(backbone=backbone)
                calls = []
                hook = model.register_forward_hook(lambda module, inputs, output: calls.append(output['states'].clone()))
                summary, source, metadata, validation = self.evaluate(root, model, initial=True)
                hook.remove()
                self.assertEqual(len(calls), 4)  # Includes duplicate and no-reference outputs.
                self.assertEqual(summary['model_forwards'], 4)
                self.assertEqual(summary['method_names'], list(METHODS))
                self.assertTrue(summary['ordinary_source_replay_exact'])
                self.assertEqual(summary['methods']['learned_sequence']['population']['duplicate_prediction_group_count'], 1)
                self.assertEqual(summary['methods']['learned_sequence']['population']['target_identity_count'], 5)
                paper = summary['methods']['learned_sequence']['paper_fusion_metrics']
                self.assertEqual(set(paper['attributes']), {'longitude', 'latitude', 'speed', 'heading'})
                self.assertEqual(paper['attributes']['longitude']['declared_targets'], 5)
                self.assertEqual(paper['attributes']['speed']['unit'], 'native_speed')
                self.assertEqual(summary['methods']['learned_sequence']['full_target_metrics']['all']['position']['intersection_count'], 127)
                self.assertEqual(summary['pairwise']['learned_sequence__minus__fixed_prior095']['all']['signed_error_change_sum'], 0.)
                self.assertEqual(summary['component_count'], 1)
                with np.load(root/'result/predictions/states-3035.npz', allow_pickle=False) as data:
                    np.testing.assert_array_equal(data['group_tangent_residual_m'], np.zeros((4, 64, 2)))
                    np.testing.assert_array_equal(data['group_anchor_node_indices'], [0, 2, 4, 5])
                    self.assertNotIn('group_gate', data)
                with self.assertRaises(FileExistsError):
                    evaluate_sequence_position(model, validation, source, metadata, root/'result')

    def test_residual_changes_supported_and_no_support_queries_without_changing_masks(self):
        with TemporaryDirectory() as directory:
            model = SequencePosition()
            with torch.no_grad():
                model.output.bias.copy_(torch.tensor([.5, -.25]))
            result, _, _, _ = self.evaluate(Path(directory), model)
            with np.load(Path(directory)/'result/system-predictions/states-3035.npz', allow_pickle=False) as data:
                before, after = data['fixed_prior095_group_states'], data['learned_sequence_group_states']
                self.assertTrue(np.all(np.any(before[..., :2] != after[..., :2], axis=-1)))
                np.testing.assert_array_equal(after[..., 2:], data['mean_supported_group_states'][..., 2:])
                np.testing.assert_array_equal(data['learned_sequence_aligned_prediction_valid'],
                                              data['anchor_aligned_prediction_valid'])
            self.assertEqual(result['methods']['learned_sequence']['population'], result['methods']['anchor']['population'])

    def test_sparse_truth_does_not_filter_groups_and_empty_strata_are_null(self):
        with TemporaryDirectory() as directory:
            result, _, _, _ = self.evaluate(Path(directory), SequencePosition(), sparse=True, initial=True)
            self.assertEqual(result['model_forwards'], 4)
            self.assertEqual(result['position_weightings']['all']['learned_sequence']['point_equal']['unit_count'], 1)
            self.assertIsNone(result['position_weightings']['no_predicted_group']['learned_sequence']['point_equal']['mae'])

    def test_corrupt_source_rejected_before_forward_and_bad_model_outputs_rejected(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source, metadata, validation = support_gate_cache_fixture(root)
            path = source/'predictions/states-3035.npz'
            with np.load(path, allow_pickle=False) as saved:
                data = dict(saved)
            data['learned_assoc_anchor'][0, 0, 0] += .01
            np.savez_compressed(path, **data)
            model = SequencePosition()
            with patch.object(model, 'forward', wraps=model.forward) as forward, self.assertRaises(ValueError):
                evaluate_sequence_position(model, validation, source, metadata, root/'out')
            forward.assert_not_called()
            self.assertEqual(read_json(root/'out/run.json')['status'], 'FAILED')
        for field in ('speed', 'residual', 'state', 'base'):
            with self.subTest(field=field), TemporaryDirectory() as directory:
                model = SequencePosition()
                original = model.forward
                def bad(*args):
                    result = {key: value.clone() for key, value in original(*args).items()}
                    if field == 'speed':
                        result['states'][0, 2] += 1.
                    elif field == 'residual':
                        result['residual_m'][0, 0] = torch.nan
                    elif field == 'state':
                        result['states'][0, 0] += .01
                    else:
                        result['base_states'][0, 0] += .01
                    return result
                with patch.object(model, 'forward', side_effect=bad), self.assertRaises(ValueError):
                    self.evaluate(Path(directory), model)
                self.assertEqual(read_json(Path(directory)/'result/run.json')['status'], 'FAILED')

    def test_weighting_units_are_distinct_and_rmse_uses_weighted_squares(self):
        arrays = []
        for values in (np.array([[1., 3.], [5., np.nan]]), np.array([[10., 20.]])):
            data = {'x_position_error': values}
            for region in ('all', 'anchor_inside_support', 'anchor_outside_helper_supported',
                           'predicted_group_no_support', 'no_predicted_group'):
                data['region_'+region] = np.ones_like(values, bool) if region == 'all' else np.zeros_like(values, bool)
            arrays.append(data)
        result = _weighted([{}, {}], arrays, ('x',))['all']['x']
        self.assertEqual(result['point_equal']['mae'], 7.8)
        self.assertAlmostEqual(result['scene_target_equal']['mae'], (2+5+15)/3)
        self.assertEqual(result['scene_equal_point_mean']['mae'], 9.)
        self.assertEqual(result['scene_equal_target_mean']['mae'], 9.25)
        self.assertAlmostEqual(result['scene_equal_point_mean']['rmse'], np.sqrt((35/3+250)/2))
        self.assertEqual(result['point_equal']['p99'], np.quantile([1, 3, 5, 10, 20], .99))

    def test_real_training_endpoints_and_cache_pairing(self):
        from association.training.sequence_position import train_sequence_position
        from tests.fixtures.learning import tiny_train
        from tools.experiments import sequence_position_fit as entry
        for backbone in ('lstm', 'transformer'):
            with self.subTest(backbone=backbone), TemporaryDirectory() as directory:
                root = Path(directory)
                config = read_json(entry.ROOT/f'configs/sequence-position-{backbone}-level1.json')
                config['training'].update(updates=3, batch_size=2)
                training = train_sequence_position(tiny_train(), config, root/'fit', device='cpu')
                source, metadata, validation = support_gate_cache_fixture(root)
                with patch('association.evaluation.sequence_position.frozen_motion', side_effect=lambda raw, *_: raw):
                    for label, filename, step in (('initial', 'step-0.pt', 0), ('final', 'final.pt', 3)):
                        model = entry.endpoint_model(root/'fit'/filename, step, config, training)
                        evaluate_sequence_position(model, validation, source, metadata, root/label,
                                                   initial=(label == 'initial'))
                compared = compare_endpoints(root/'initial', root/'final')
                self.assertEqual(compared['status'], 'COMPLETED')
                self.assertTrue(compared['population_and_masks_exact'])
                self.assertEqual(compared['paired']['all']['final_minus_initial']['common_point_count'], 127)
                with self.assertRaisesRegex(ValueError, 'mismatch'):
                    entry.endpoint_model(root/'fit/final.pt', 0, config, training)
                with self.assertRaisesRegex(ValueError, 'completed'):
                    entry.endpoint_model(root/'fit/final.pt', 3, config, dict(training, status='FAILED'))
                path = root/'final/system-predictions/states-3035.npz'
                with np.load(path, allow_pickle=False) as saved:
                    altered = dict(saved)
                altered['learned_sequence_aligned_prediction_valid'][0, 0, 0] = False
                np.savez_compressed(path, **altered)
                with self.assertRaisesRegex(ValueError, 'fixed endpoint system'):
                    compare_endpoints(root/'initial', root/'final')

    def test_cli_frozen_configuration_and_source_failure_prevents_training(self):
        from tools.experiments import sequence_position_fit as entry
        manifest = read_json(entry.ROOT/'configs/data/mtad-level1-scene3000.json')
        alignment = read_json(entry.ROOT/'configs/data/mtad-level1-alignment.json')
        for backbone in ('lstm', 'transformer'):
            config = read_json(entry.ROOT/f'configs/sequence-position-{backbone}-level1.json')
            entry.production_contract(config, manifest, alignment)
            bad = deepcopy(config)
            bad['training']['updates'] += 1
            with self.assertRaisesRegex(ValueError, '1000-update'):
                entry.production_contract(bad, manifest, alignment)
        with TemporaryDirectory() as directory:
            root = Path(directory)
            args = Namespace(source_system=str(root/'source056/system'), output=str(root/'out'),
                archive=['not-opened.zip'], manifest=str(entry.ROOT/'configs/data/mtad-level1-scene3000.json'),
                alignment=str(entry.ROOT/'configs/data/mtad-level1-alignment.json'),
                config=str(entry.ROOT/'configs/sequence-position-lstm-level1.json'), device='cpu',
                cpu_threads=4, time_policy='advisory')
            with patch.object(entry, '_bind_source', side_effect=ValueError('source mismatch')), \
                    patch.object(entry, 'load_timed_dataset') as load, \
                    patch.object(entry, 'train_sequence_position') as train:
                with self.assertRaisesRegex(ValueError, 'source mismatch'):
                    entry.run(args)
            load.assert_not_called(); train.assert_not_called()
            self.assertEqual(read_json(root/'out/run.json')['status'], 'FAILED')
            self.assertFalse((root/'out/summary.json').exists())
            with self.assertRaises(FileExistsError):
                entry.run(args)

    def test_extended_profile_changes_only_the_fresh_update_budget(self):
        from tools.experiments import sequence_position_fit as entry
        manifest = read_json(entry.ROOT/'configs/data/mtad-level1-scene3000.json')
        alignment = read_json(entry.ROOT/'configs/data/mtad-level1-alignment.json')
        for backbone in ('lstm', 'transformer'):
            with self.subTest(backbone=backbone):
                pilot = read_json(entry.ROOT/f'configs/sequence-position-{backbone}-level1.json')
                extended = read_json(entry.ROOT/f'configs/sequence-position-{backbone}-extended-level1.json')
                expected = deepcopy(pilot)
                expected['training']['updates'] = 4000
                self.assertEqual(extended, expected)
                entry.production_contract(extended, manifest, alignment, profile='extended4000')
                with self.assertRaisesRegex(ValueError, '1000-update'):
                    entry.production_contract(extended, manifest, alignment)
                with self.assertRaisesRegex(ValueError, '4000-update'):
                    entry.production_contract(pilot, manifest, alignment, profile='extended4000')
        replica = read_json(entry.ROOT/'configs/sequence-position-transformer-extended-replica-level1.json')
        expected = read_json(entry.ROOT/'configs/sequence-position-transformer-extended-level1.json')
        expected['training']['seed'] = 20261021
        self.assertEqual(replica, expected)
        entry.production_contract(replica, manifest, alignment, profile='replica4000')
        with self.assertRaisesRegex(ValueError, '4000-update'):
            entry.production_contract(replica, manifest, alignment, profile='extended4000')
        lstm = read_json(entry.ROOT/'configs/sequence-position-lstm-extended-level1.json')
        with self.assertRaisesRegex(ValueError, 'Transformer'):
            entry.production_contract(lstm, manifest, alignment, profile='replica4000')
        for profile, suffix, original in (
                ('harm4000', 'relative-harm-level1', 'extended-level1'),
                ('harm-replica4000', 'relative-harm-replica-level1', 'extended-replica-level1')):
            config = read_json(entry.ROOT/f'configs/sequence-position-transformer-{suffix}.json')
            expected = read_json(entry.ROOT/f'configs/sequence-position-transformer-{original}.json')
            expected.update(scope='sequence-position-relative-harm-v1', position_loss='linear_relative_harm')
            self.assertEqual(config, expected)
            entry.production_contract(config, manifest, alignment, profile=profile)

    def test_real_small_cli_run_preserves_fixed_endpoints_without_validation_selection(self):
        from tests.fixtures.learning import tiny_train
        from tools.experiments import sequence_position_fit as entry
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source, metadata, validation = support_gate_cache_fixture(root)
            training = tiny_train()
            config = read_json(entry.ROOT/'configs/sequence-position-transformer-extended-replica-level1.json')
            config['training'].update(updates=2, batch_size=2)
            manifest = dict(splits={'T_core': [str(scene['scene_id']) for scene in training],
                                    'V_select': ['3035']}, components={'3035': 'shared'})
            for name, value in (('config', config), ('manifest', manifest), ('alignment', {})):
                write_json(root/(name+'.json'), value)
            args = Namespace(source_system=str(source), output=str(root/'out'), archive=['not-opened.zip'],
                manifest=str(root/'manifest.json'), alignment=str(root/'alignment.json'),
                config=str(root/'config.json'), device='cpu', cpu_threads=4, time_policy='advisory',
                profile='replica4000')
            # Only data I/O and large production-manifest binding are substituted.
            # Actual training, checkpoints, all-group inference, source-cache
            # replay, full-target evaluation and endpoint comparison all execute.
            with patch.object(entry, 'production_contract'), patch.object(entry, '_bind_source'), \
                    patch.object(entry, '_load_metadata', return_value=metadata), \
                    patch.object(entry, 'load_timed_dataset', return_value=dict(train=training,
                        validation=validation, report={'fixture': True})), \
                    patch('association.evaluation.sequence_position.frozen_motion', side_effect=lambda raw, *_: raw):
                result = entry.run(args)
            self.assertEqual(result['status'], 'COMPLETED')
            self.assertEqual(result['optimizer_updates'], 2)
            self.assertEqual(result['scope'], '073-generic-sequence-position-replica4000')
            self.assertTrue(result['limitations'][0].startswith('2 updates'))
            self.assertEqual(read_json(root/'out/run.json')['completed_endpoints'], ['initial', 'final'])
            self.assertFalse(read_json(root/'out/run.json')['validation_selection'])
            self.assertTrue(read_json(root/'out/endpoint-comparison.json')['population_and_masks_exact'])

    def test_relative_harm_full_chain_binds_same_seed_actual_control_and_all_cpu_outputs(self):
        from tests.fixtures.learning import tiny_train
        from tools.experiments import sequence_position_fit as entry
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source, metadata, validation = support_gate_cache_fixture(root)
            training = tiny_train()
            config = read_json(entry.ROOT/'configs/sequence-position-transformer-extended-level1.json')
            config['training'].update(updates=2, batch_size=2)
            manifest = dict(splits={'T_core': [str(scene['scene_id']) for scene in training],
                                    'V_select': ['3035']}, components={'3035': 'shared'})
            harm = deepcopy(config)
            harm.update(scope='sequence-position-relative-harm-v1', position_loss='linear_relative_harm')
            for name, value in (('control', config), ('harm', harm), ('manifest', manifest), ('alignment', {})):
                write_json(root/(name+'.json'), value)
            args = Namespace(source_system=str(source), output=str(root/'control'), archive=['not-opened.zip'],
                manifest=str(root/'manifest.json'), alignment=str(root/'alignment.json'),
                config=str(root/'control.json'), device='cpu', cpu_threads=4, time_policy='advisory',
                profile='extended4000', paired_control=None)
            with patch.object(entry, 'production_contract'), patch.object(entry, '_bind_source'), \
                    patch.object(entry, '_load_metadata', return_value=metadata), \
                    patch.object(entry, 'load_timed_dataset', side_effect=lambda *a, **k: dict(train=training,
                        validation=validation, report={'fixture': True})), \
                    patch('association.evaluation.sequence_position.frozen_motion', side_effect=lambda raw, *_: raw):
                entry.run(args)
                args = Namespace(**vars(args))
                args.config, args.output = str(root/'harm.json'), str(root/'harm')
                args.profile, args.paired_control = 'harm4000', str(root/'control')
                result = entry.run(args)
            self.assertEqual(result['status'], 'COMPLETED')
            self.assertEqual(result['training_pairing']['actual_successful_batches_matched'], 2)
            self.assertTrue(result['training_pairing']['initial_parameters_exact'])
            self.assertTrue(result['training_pairing']['sampling_exposure_exact'])
            self.assertEqual(read_json(root/'harm/run.json')['completed_endpoints'], ['initial', 'final'])
            self.assertEqual(read_json(root/'harm/final/summary.json')['checkpoint']['position_loss'], 'linear_relative_harm')
            self.assertTrue(read_json(root/'harm/endpoint-comparison.json')['population_and_masks_exact'])
            for label in ('initial', 'final'):
                with np.load(root/'harm'/label/'system-predictions/states-3035.npz', allow_pickle=False) as data:
                    np.testing.assert_array_equal(data['learned_sequence_group_states'][..., 2:],
                                                  data['mean_supported_group_states'][..., 2:])
                    if label == 'initial':
                        np.testing.assert_array_equal(data['learned_sequence_group_states'],
                                                      data['fixed_prior095_group_states'])
            record = read_json(root/'control/final/run.json')
            record['checkpoint']['step'] = 1
            write_json(root/'control/final/run.json', record)
            with self.assertRaisesRegex(ValueError, 'frozen endpoint'):
                entry.bind_linear_control(root/'control', harm, source, manifest)


if __name__ == '__main__':
    unittest.main()
