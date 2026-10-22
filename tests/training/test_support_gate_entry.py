"""068 entry boundaries plus a real small fit-to-system endpoint check."""
from argparse import Namespace
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from association.evaluation.support_gate import evaluate_support_gate, compare_endpoints
from association.training.support_gate import train_support_gate
from association.utils.runtime import read_json
from tests.evaluation.test_support_gate import support_gate_cache_fixture
from tests.training.test_support_gate import tiny_config, tiny_train
from tools.experiments import support_gate_fit as entry


class SupportGateEntryTests(unittest.TestCase):
    def test_frozen_contract_rejects_budget_or_population_drift(self):
        config = read_json(entry.ROOT/'configs/support-gate-level1.json')
        manifest = read_json(entry.ROOT/'configs/data/mtad-level1-scene3000.json')
        alignment = read_json(entry.ROOT/'configs/data/mtad-level1-alignment.json')
        entry.production_contract(config, manifest, alignment)
        changed = deepcopy(config)
        changed['training']['updates'] += 1
        with self.assertRaisesRegex(ValueError, '1000-update'):
            entry.production_contract(changed, manifest, alignment)
        changed = deepcopy(alignment)
        changed[manifest['splits']['V_select'][0]]['time_offset_seconds'] += 1
        with self.assertRaisesRegex(ValueError, 'alignment'):
            entry.production_contract(config, manifest, changed)

    def test_preflight_failure_stops_before_full_data_or_training(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            args = Namespace(source_system=str(root/'source056/system'), output=str(root/'out'),
                archive=['not-opened.zip'], manifest=str(entry.ROOT/'configs/data/mtad-level1-scene3000.json'),
                alignment=str(entry.ROOT/'configs/data/mtad-level1-alignment.json'),
                config=str(entry.ROOT/'configs/support-gate-level1.json'), device='cpu', cpu_threads=4)
            with patch.object(entry, '_bind_source'), patch.object(entry, '_load_metadata', return_value={
                    'scene_ids': ['3035', '2535']}), \
                    patch.object(entry, 'run_preflight', side_effect=ValueError('raw cache mismatch')), \
                    patch.object(entry, 'load_timed_dataset') as load, \
                    patch.object(entry, 'train_support_gate') as train:
                with self.assertRaisesRegex(ValueError, 'raw cache mismatch'):
                    entry.run(args)
            load.assert_not_called()
            train.assert_not_called()
            state = read_json(root/'out/run.json')
            self.assertEqual(state['status'], 'FAILED')
            self.assertEqual(state['optimizer_updates'], 0)
            self.assertEqual(state['phase'], 'preflight3035_2535')
            self.assertFalse((root/'out/summary.json').exists())
            with self.assertRaises(FileExistsError):
                entry.run(args)

    def test_real_short_training_checkpoint_system_and_endpoint_pairing(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            config = tiny_config()
            report = train_support_gate(tiny_train(), config, root/'fit', device='cpu')
            source, metadata, validation = support_gate_cache_fixture(root)
            # Cache fixture has no raw observations; all model, checkpoint,
            # ordinary-cache replay, grouping, target alignment and metrics run.
            with patch('association.evaluation.support_gate.frozen_motion', side_effect=lambda raw, *_: raw):
                for label, filename, step in (('initial', 'step-0.pt', 0), ('final', 'final.pt', 3)):
                    model = entry.endpoint_model(root/'fit'/filename, step, config, report)
                    result = evaluate_support_gate(model, validation, source, metadata, root/label)
                    self.assertTrue(result['ordinary_source_replay_exact'])
            comparison = compare_endpoints(root/'initial', root/'final')
            self.assertTrue(comparison['population_and_masks_exact'])
            self.assertEqual(comparison['paired']['all']['speed']['signed_error_change_sum'], 0.)
            self.assertEqual(comparison['paired']['all']['heading']['signed_error_change_sum'], 0.)
            with self.assertRaisesRegex(ValueError, 'step/configuration mismatch'):
                entry.endpoint_model(root/'fit/final.pt', 0, config, report)
            failed = dict(report, status='FAILED')
            with self.assertRaisesRegex(ValueError, 'completed'):
                entry.endpoint_model(root/'fit/final.pt', 3, config, failed)


if __name__ == '__main__':
    unittest.main()
