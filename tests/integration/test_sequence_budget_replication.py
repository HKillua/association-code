"""097 fixed second-seed source, real paired fit and complete cached evaluation."""
from argparse import Namespace
from contextlib import ExitStack
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import torch

from association.training.frozen_position_adapter import load_checkpoint
from association.utils.runtime import read_json, write_json
from tests.fixtures.cache import support_gate_cache_fixture
from tests.fixtures.learning import tiny_train
from tools.experiments import sequence_budget_replication as entry
from tools.experiments import sequence_position_fit as original
from tools.experiments import frozen_position_adapter_fit as old_entry
from tools.analysis import compare_frozen_position_adapter as compare
from tools.analysis import compare_sequence_budget_seeds as seeds_compare


class SequenceBudgetReplicationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_explicit_new_profile_preserves_old090_contract(self):
        manifest = read_json(entry.ROOT/'configs/data/mtad-level1-scene3000.json')
        alignment = read_json(entry.ROOT/'configs/data/mtad-level1-alignment.json')
        config = read_json(entry.ROOT/'configs'/entry.CONFIG_FILE)
        entry.production_contract(config, manifest, alignment)
        with self.assertRaises(ValueError):
            old_entry.production_contract(config, manifest, alignment)
        for key, value in [('seed', 20261023), ('updates', 8000), ('batch_size', 4)]:
            changed = deepcopy(config); changed['training'][key] = value
            with self.assertRaises(ValueError): entry.production_contract(changed, manifest, alignment)
        changed = deepcopy(config); changed['model']['mode'] = 'all_queries'
        with self.assertRaises(ValueError): entry.production_contract(changed, manifest, alignment)

    def test_source_only_waits_without_checkpoint_or_output_and_rejects_wrong_identity(self):
        with TemporaryDirectory() as directory:
            root = Path(directory); source = root/'source'; source.mkdir()
            args = Namespace(paired_control=str(source), source_system=str(root/'056/system'),
                config='unopened', manifest='unopened', alignment='unopened',
                cpu_threads=4, time_policy='advisory')
            with patch.object(torch, 'load', side_effect=AssertionError('active checkpoint read')):
                with self.assertRaises(entry.WaitingDependencies): entry.check_source(args)
                write_json(source/'run.json', dict(status='RUNNING'))
                with self.assertRaises(entry.WaitingDependencies): entry.check_source(args)
                write_json(source/'run.json', dict(status='FAILED'))
                with self.assertRaisesRegex(ValueError, 'failed'): entry.check_source(args)
                write_json(source/'run.json', dict(status='COMPLETED', profile='extended4000'))
                with self.assertRaisesRegex(ValueError, '073C'): entry.check_source(args)
            self.assertEqual(set(root.iterdir()), {source})

    def test_real_seed21_source_additional_fit_complete_endpoints_and_original_sample_pair(self):
        with TemporaryDirectory() as directory:
            root = Path(directory); source, metadata, validation = support_gate_cache_fixture(root)
            write_json(source/'run.json', dict(status='COMPLETED'))
            train = tiny_train()
            manifest = dict(splits={'T_core': [str(s['scene_id']) for s in train], 'V_select': ['3035']},
                components={'3035': 'shared'})
            base = read_json(entry.ROOT/'configs/sequence-position-transformer-extended-replica-level1.json')
            base['training'].update(updates=4, batch_size=2)
            config = read_json(entry.ROOT/'configs'/entry.CONFIG_FILE)
            config['training'].update(updates=4, batch_size=2)
            for name, value in [('base', base), ('config', config), ('manifest', manifest), ('alignment', {})]:
                write_json(root/(name+'.json'), value)
            args = Namespace(source_system=str(source), output=str(root/'073C'), paired_control=None,
                archive=['unread.zip'], manifest=str(root/'manifest.json'), alignment=str(root/'alignment.json'),
                config=str(root/'base.json'), device='cpu', cpu_threads=4, time_policy='advisory', profile='replica4000')
            dataset = lambda *a, **k: dict(train=train, validation=validation, report={'fixture': True})
            def meta(path, arm=None):
                self.assertEqual(arm, '073C_transformer4000_seed21')
                path = Path(path)
                return dict(path=path, method='learned_sequence', source_system=str(source.resolve()),
                    scene_ids=['3035'], components={'3035': 'shared'}, summary=read_json(path/'final/summary.json'))
            with patch('association.evaluation.sequence_position.frozen_motion', side_effect=lambda raw, *a: raw):
                with patch.object(original, 'production_contract'), patch.object(original, '_bind_source'), \
                        patch.object(original, '_load_metadata', return_value=metadata), \
                        patch.object(original, 'load_timed_dataset', side_effect=dataset):
                    original.run(args)
                source_bytes = (root/'073C/fit/final.pt').read_bytes()
                args.config = str(root/'config.json'); args.paired_control = str(root/'073C'); args.output = str(root/'097A')
                with ExitStack() as patches:
                    patches.enter_context(patch.object(entry, 'production_contract'))
                    patches.enter_context(patch.object(entry, 'frozen_configuration', side_effect=lambda value: value))
                    patches.enter_context(patch.object(entry, '_bind_source'))
                    patches.enter_context(patch.object(entry, '_load_metadata', return_value=metadata))
                    patches.enter_context(patch.object(entry, 'load_timed_dataset', side_effect=dataset))
                    patches.enter_context(patch.object(compare, '_metadata', side_effect=meta))
                    patches.enter_context(patch.object(compare, 'declared_manifest', return_value=manifest))
                    result = entry.run(args)
                    self.assertEqual(result['status'], 'COMPLETED')
                    self.assertEqual(result['scope'], entry.SCOPE)
                    self.assertEqual(result['budget_characterization']['seed'], 20261021)
                    self.assertEqual(result['total_stage_updates'], 8)
                    self.assertFalse(result['default_promoted'])
                    self.assertNotIn('scientific_screen', result)
                    with self.assertRaises(FileExistsError): entry.run(args)
                self.assertEqual((root/'073C/fit/final.pt').read_bytes(), source_bytes)
                initial, before = load_checkpoint(root/'097A/fit/step-0.pt')
                final, after = load_checkpoint(root/'097A/fit/final.pt')
                self.assertEqual(before['optimizer']['state'], {})
                self.assertEqual(after['step'], 4)
                self.assertTrue(all(torch.equal(before['model']['teacher.'+key], tensor)
                    for key, tensor in before['source_teacher']['model'].items()))
                self.assertTrue(any(not torch.equal(before['model'][key], tensor)
                    for key, tensor in after['model'].items()))
                self.assertEqual(result['trainable_parameter_count'], 105154)
                self.assertEqual(result['training_pairing']['actual_successful_batches_matched'], 4)
                self.assertTrue(result['training_pairing']['sampling_exposure_exact'])
                self.assertFalse(result['training_pairing']['teacher_unchanged'])
                import gzip, json
                table = json.load(gzip.open(root/'097A/reference-comparison/summary.json.gz', 'rt'))
                self.assertEqual(table['source_reference_arm'], '073C_transformer4000_seed21')
                self.assertEqual(table['position_population']['covered_target_points'], 127)
                self.assertTrue(table['initial_output_source_final_exact'])
                self.assertTrue(table['pairing']['arms']['finetune']['actual_successful_batches_exact'])
                self.assertNotIn('scientific_screen', table)
                # The second dimension is an independent seed, not equal source parameters or draws.
                base23, config23 = deepcopy(base), deepcopy(config)
                base23['training']['seed'] = config23['training']['seed'] = 20261023
                write_json(root/'base23.json', base23); write_json(root/'config23.json', config23)
                args23 = deepcopy(args)
                args23.profile = 'extended4000'; args23.paired_control = None
                args23.config = str(root/'base23.json'); args23.output = str(root/'072C')
                with patch.object(original, 'production_contract'), patch.object(original, '_bind_source'), \
                        patch.object(original, '_load_metadata', return_value=metadata), \
                        patch.object(original, 'load_timed_dataset', side_effect=dataset):
                    original.run(args23)
                def meta23(path, arm=None):
                    self.assertEqual(arm, '072C_transformer4000')
                    path = Path(path)
                    return dict(path=path, method='learned_sequence', source_system=str(source.resolve()),
                        scene_ids=['3035'], components={'3035': 'shared'}, summary=read_json(path/'final/summary.json'))
                args23.config = str(root/'config23.json'); args23.paired_control = str(root/'072C')
                args23.output = str(root/'090C')
                with patch.object(old_entry, 'production_contract'), \
                        patch.object(old_entry, 'frozen_configuration', side_effect=lambda value: value), \
                        patch.object(old_entry, '_bind_source'), patch.object(old_entry, '_load_metadata', return_value=metadata), \
                        patch.object(old_entry, 'load_timed_dataset', side_effect=dataset), \
                        patch.object(compare, '_metadata', side_effect=meta23), \
                        patch.object(compare, 'declared_manifest', return_value=manifest):
                    old_entry.run(args23)
                table_args = Namespace(seed23_run=str(root/'090C'), seed21_run=str(root/'097A'),
                    output=str(root/'budget-table'), cpu_threads=4)
                contract = lambda seed: (config23 if seed == 20261023 else config, manifest)
                with patch.object(seeds_compare, '_contract', side_effect=contract):
                    result_table = seeds_compare.run(table_args)
                    self.assertEqual([arm['seed'] for arm in result_table['arms']], [20261023, 20261021])
                    self.assertFalse(result_table['cross_seed_parameter_or_sample_equality_required'])
                    self.assertEqual(result_table['model_forwards'], 0)
                    self.assertEqual(result_table['arms'][0]['scene_directions']['total'], 1)
                    # Swapped inputs cannot become a valid two-seed result.
                    swapped = deepcopy(table_args); swapped.output = str(root/'wrong-table')
                    swapped.seed21_run, swapped.seed23_run = swapped.seed23_run, swapped.seed21_run
                    with self.assertRaisesRegex(ValueError, 'declared seed'):
                        seeds_compare.run(swapped)
                    self.assertFalse(Path(swapped.output).exists())
                # A source with altered seed cannot pass the actual pre-fit control binding.
                changed = read_json(root/'073C/run.json'); changed['configuration']['training']['seed'] = 20261023
                write_json(root/'073C/run.json', changed)
                with patch.object(entry, 'production_contract'):
                    with self.assertRaisesRegex(ValueError, 'same-seed'):
                        entry.check_source(args)


if __name__ == '__main__':
    unittest.main()
