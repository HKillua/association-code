"""070 entry barriers: one paired fit and one final-only system evaluation."""
from argparse import Namespace
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from association.evaluation.support_gate import evaluate_support_gate
from association.evaluation.support_gate_linear import bind_linear_source, compare_linear_run
from association.training.support_gate import train_support_gate, loss_semantics
from association.utils.runtime import read_json, write_json
from tests.evaluation.test_support_gate import support_gate_cache_fixture
from tests.training.test_support_gate import tiny_config, tiny_train
from tools.experiments import support_gate_linear_fit as entry


class SupportGateLinearEntryTests(unittest.TestCase):
    def _args(self, root):
        return Namespace(source_system=str(root/'source056/system'),
            source_068=str(root/'source068'), output=str(root/'out'),
            archive=['must-not-be-opened.zip'],
            manifest=str(entry.ROOT/'configs/data/mtad-level1-scene3000.json'),
            alignment=str(entry.ROOT/'configs/data/mtad-level1-alignment.json'),
            config=str(entry.ROOT/'configs/support-gate-linear-level1.json'),
            device='cpu', cpu_threads=4)

    def test_only_loss_differs_from_frozen_068_and_other_changes_rejected(self):
        config = read_json(entry.ROOT/'configs/support-gate-linear-level1.json')
        original = read_json(entry.ROOT/'configs/support-gate-level1.json')
        manifest = read_json(entry.ROOT/'configs/data/mtad-level1-scene3000.json')
        alignment = read_json(entry.ROOT/'configs/data/mtad-level1-alignment.json')
        entry.production_contract(config, manifest, alignment)
        changed = deepcopy(config)
        self.assertEqual(changed.pop('position_loss'), 'linear')
        self.assertEqual(changed, original)
        for key, value in (('updates', 1001), ('seed', 20261022),
                           ('batch_size', 16), ('learning_rate', .002), ('gradient_clip', 2.)):
            changed = deepcopy(config)
            changed['training'][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                entry.production_contract(changed, manifest, alignment)
        for kind in ('log1p', 'helper_quality', None):
            changed = deepcopy(config)
            if kind is None:
                changed.pop('position_loss')
            else:
                changed['position_loss'] = kind
            with self.subTest(loss=kind), self.assertRaises(ValueError):
                entry.production_contract(changed, manifest, alignment)
        changed = deepcopy(alignment)
        changed[manifest['splits']['V_select'][0]]['time_offset_seconds'] += 1
        with self.assertRaises(ValueError):
            entry.production_contract(config, manifest, changed)
        changed = deepcopy(manifest)
        changed['splits']['V_select'] = list(reversed(changed['splits']['V_select']))
        with self.assertRaises(ValueError):
            entry.production_contract(config, changed, alignment)

    def test_source_binding_failure_prevents_data_training_and_final_summary(self):
        with TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            args = self._args(root)
            manifest = read_json(args.manifest)
            metadata = dict(scene_ids=manifest['splits']['V_select'])
            with patch.object(entry, '_bind_source'), \
                    patch.object(entry, '_load_metadata', return_value=metadata), \
                    patch.object(entry, 'bind_linear_source', side_effect=ValueError('source068 not completed')), \
                    patch.object(entry, 'load_timed_dataset') as prepare, \
                    patch.object(entry, 'train_support_gate') as train, \
                    patch.object(entry, 'evaluate_support_gate') as evaluate:
                with self.assertRaisesRegex(ValueError, 'source068 not completed'):
                    entry.run(args)
            prepare.assert_not_called()
            train.assert_not_called()
            evaluate.assert_not_called()
            state = read_json(root/'out/run.json')
            self.assertEqual(state['status'], 'FAILED')
            self.assertEqual(state['optimizer_updates'], 0)
            self.assertFalse((root/'out/summary.json').exists())
            with self.assertRaises(FileExistsError):
                entry.run(args)

    def test_completed_chain_fits_once_and_forwards_only_final_endpoint(self):
        with TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            args = self._args(root)
            manifest, config = read_json(args.manifest), read_json(args.config)
            train_rows = [dict(scene_id=sid, split='T_core') for sid in manifest['splits']['T_core']]
            validation_rows = [dict(scene_id=sid, split='V_select') for sid in manifest['splits']['V_select']]
            dataset = dict(train=train_rows, validation=validation_rows, report={'test_opened': False})
            metadata = dict(scene_ids=manifest['splits']['V_select'])
            training = dict(status='COMPLETED', optimizer_updates=1000, configuration=config)
            marker = object()

            def evaluate(model, validation, source, bound_metadata, output, **kwargs):
                self.assertIs(model, marker)
                self.assertEqual(validation, validation_rows)
                self.assertEqual(source, root/'source056/system')
                self.assertEqual(output, root/'out/final')
                output.mkdir(parents=True)
                write_json(output/'run.json', dict(status='COMPLETED', model_forwards=7))
                return dict(status='COMPLETED', model_forwards=7)

            with patch.object(entry, '_bind_source'), \
                    patch.object(entry, '_load_metadata', return_value=metadata), \
                    patch.object(entry, 'bind_linear_source', return_value={'verified': True}) as bind, \
                    patch.object(entry, 'load_timed_dataset', return_value=dataset), \
                    patch.object(entry, 'train_support_gate', return_value=training) as train, \
                    patch.object(entry, 'endpoint_model', return_value=marker) as endpoint, \
                    patch.object(entry, 'evaluate_support_gate', side_effect=evaluate) as evaluator, \
                    patch.object(entry, 'compare_linear_run', return_value={'status': 'COMPLETED'}) as compare:
                result = entry.run(args)
            bind.assert_called_once()
            train.assert_called_once()
            self.assertEqual(train.call_args.args[0], train_rows)
            self.assertEqual(train.call_args.kwargs['paired_reference_fit'], root/'source068/fit')
            endpoint.assert_called_once_with(root/'out/fit/final.pt', 1000, config, training)
            evaluator.assert_called_once()
            compare.assert_called_once()
            self.assertEqual(compare.call_args.args[:2], (root/'source068', root/'out/final'))
            self.assertEqual(result['status'], 'COMPLETED')
            self.assertTrue(result['initial_evaluation']['reused'])
            self.assertEqual(result['initial_evaluation']['model_forwards'], 0)
            state = read_json(root/'out/run.json')
            self.assertEqual(state['completed_endpoints'], ['final'])
            self.assertTrue(state['initial_evaluation_reused'])
            self.assertFalse(state['raw_preflight_repeated'])
            self.assertFalse(state['test_opened'])
            self.assertFalse((root/'out/initial').exists())
            self.assertFalse((root/'out/preflight').exists())
            for name in ('run.json', 'summary.json'):
                endpoint_record = read_json(root/'out/final'/name)
                self.assertEqual(endpoint_record['configuration'], config)
                self.assertEqual(endpoint_record['position_loss'], 'linear')

    def test_incomplete_prepared_scene_order_stops_before_training(self):
        with TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            args = self._args(root)
            manifest = read_json(args.manifest)
            with patch.object(entry, '_bind_source'), \
                    patch.object(entry, '_load_metadata', return_value={'scene_ids': manifest['splits']['V_select']}), \
                    patch.object(entry, 'bind_linear_source', return_value={'verified': True}), \
                    patch.object(entry, 'load_timed_dataset', return_value={'train': [], 'validation': [], 'report': {}}), \
                    patch.object(entry, 'train_support_gate') as train, \
                    patch.object(entry, 'evaluate_support_gate') as evaluate:
                with self.assertRaisesRegex(ValueError, 'population/order'):
                    entry.run(args)
            train.assert_not_called()
            evaluate.assert_not_called()
            self.assertEqual(read_json(root/'out/run.json')['status'], 'FAILED')
            self.assertFalse((root/'out/summary.json').exists())

    def test_real_paired_training_checkpoint_and_only_final_system_comparison(self):
        """Real three-update fits and CPU evaluation; production data adapters only are replaced."""
        with TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            original, config, train_rows = root/'old068', tiny_config(), tiny_train()
            original.mkdir()
            baseline_fit = train_support_gate(train_rows, config, original/'fit', device='cpu')
            source, metadata, validation = support_gate_cache_fixture(root)
            old_endpoints = {}
            with patch('association.evaluation.support_gate.frozen_motion', side_effect=lambda scene, *_: scene):
                for label, filename, step in (('initial', 'step-0.pt', 0), ('final', 'final.pt', 3)):
                    model = entry.endpoint_model(original/'fit'/filename, step, config, baseline_fit)
                    old_endpoints[label] = evaluate_support_gate(
                        model, validation, source, metadata, original/label)
                linear = dict(config, position_loss='linear')
                manifest = dict(splits=dict(T_core=[str(row['scene_id']) for row in train_rows],
                    V_select=metadata['scene_ids']), components=metadata['component_ids'])
                registry = dict(scene_ids=metadata['scene_ids'], components=metadata['component_ids'],
                    endpoints=old_endpoints, source_run=dict(configuration=config, source_system=str(source)))
                # Tiny scene manifests bypass only the fixed 300-scene production registry.
                with patch('association.evaluation.support_gate_linear.bind_source', return_value=registry):
                    binding = bind_linear_source(original, source, linear, manifest)
                result_dir = root/'new070'
                result_dir.mkdir()
                fit = train_support_gate(train_rows, linear, result_dir/'fit', device='cpu',
                    paired_reference_fit=original/'fit')
                model = entry.endpoint_model(result_dir/'fit/final.pt', 3, linear, fit)
                evaluated = evaluate_support_gate(model, validation, source, metadata, result_dir/'final')
            checkpoint = dict(path=str(result_dir/'fit/final.pt'), step=3, actual_updates=3,
                configured_final_updates=3, routing_semantics=linear['routing_semantics'],
                loss_semantics=loss_semantics(linear))
            for name in ('run.json', 'summary.json'):
                path = result_dir/'final'/name
                record = read_json(path)
                record.update(configuration=linear, position_loss='linear', checkpoint=checkpoint)
                write_json(path, record)
            comparison = compare_linear_run(original, result_dir/'final', binding, result_dir/'comparison')
            self.assertEqual(fit['pairing']['matched_updates'], 3)
            self.assertEqual(fit['pairing']['matched_anchors'], 6)
            self.assertTrue(fit['pairing']['initial_parameters_exact'])
            self.assertEqual(evaluated['model_forwards'], old_endpoints['initial']['model_forwards'])
            self.assertTrue(binding['initial_cache_reused'])
            self.assertFalse((result_dir/'initial').exists())
            self.assertEqual(set(comparison['comparisons']), {'vs_initial', 'vs_log_final'})
            for item in comparison['comparisons'].values():
                self.assertEqual(item['objectives']['regions']['all']['point_count'], 127)
                self.assertEqual(item['objectives']['population']['duplicate_prediction_group_count'], 1)
                self.assertTrue(item['physical']['population_and_masks_exact'])
                self.assertEqual(item['physical']['paired']['all']['speed']['signed_error_change_sum'], 0.)
                self.assertEqual(item['physical']['paired']['all']['heading']['signed_error_change_sum'], 0.)

    def test_output_cannot_replace_or_nest_either_frozen_source(self):
        with TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            args = self._args(root)
            for output in (root, root/'source056', root/'source056/system/new-output',
                           root/'source068', root/'source068/new-output'):
                args.output = str(output)
                with self.subTest(output=output), patch.object(entry, 'bind_linear_source') as bind, \
                        patch.object(entry, 'load_timed_dataset') as prepare:
                    with self.assertRaises((ValueError, FileExistsError)):
                        entry.run(args)
                    bind.assert_not_called()
                    prepare.assert_not_called()


if __name__ == '__main__':
    unittest.main()
