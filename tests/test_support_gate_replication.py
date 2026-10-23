"""Independent training remains parallel; actual pairing is checked afterwards."""
from argparse import Namespace
from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import torch

from association.evaluation.support_gate import evaluate_support_gate
from association.evaluation.support_gate_replication import (
    SCOPE, verify_pairing, compare_cached_endpoints, compare_replica_runs)
from association.training.support_gate import train_support_gate
from association.utils.runtime import read_json, write_json
from tests.evaluation.test_support_gate import support_gate_cache_fixture
from tests.training.test_support_gate import tiny_config, tiny_train
from tools.experiments import support_gate_replication as entry


class SupportGateReplicationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def configs(self):
        a = tiny_config()
        a['training']['seed'] = 20261023
        a['position_loss'] = 'log1p'
        b = deepcopy(a); b['position_loss'] = 'linear'
        return a, b

    def test_production_only_changes_seed_and_loss(self):
        manifest = read_json(entry.ROOT/'configs/data/mtad-level1-scene3000.json')
        alignment = read_json(entry.ROOT/'configs/data/mtad-level1-alignment.json')
        original = read_json(entry.ROOT/'configs/support-gate-level1.json')
        for kind in ('log', 'linear'):
            cfg = read_json(entry.ROOT/f'configs/support-gate-replica-{kind}-level1.json')
            entry.production_contract(cfg, manifest, alignment)
            stripped = deepcopy(cfg); stripped.pop('position_loss'); stripped['training']['seed'] = 20261021
            self.assertEqual(stripped, original)
            for key, value in (('seed', 20261024), ('updates', 2000), ('learning_rate', .01)):
                changed = deepcopy(cfg); changed['training'][key] = value
                with self.assertRaises(ValueError):
                    entry.production_contract(changed, manifest, alignment)

    def test_independent_real_training_exact_pair_and_tamper_refusal(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            a, b = self.configs()
            # Linear runs first: it cannot wait for or read an existing log run.
            train_support_gate(tiny_train(), b, root/'linear', device='cpu')
            self.assertFalse((root/'log').exists())
            train_support_gate(tiny_train(), a, root/'log', device='cpu')
            result = verify_pairing(root/'log', root/'linear')
            self.assertEqual(result['status'], 'PAIR_VERIFIED')
            self.assertEqual(result['matched_updates'], 3)
            self.assertEqual(result['matched_anchors'], 6)
            rows = (root/'linear/training.jsonl').read_text()
            changed = [json.loads(line) for line in rows.splitlines()]
            changed[0]['batch'][0]['members'].append(999)
            (root/'linear/training.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in changed))
            with self.assertRaisesRegex(ValueError, 'batch pairing'):
                verify_pairing(root/'log', root/'linear')
            (root/'linear/training.jsonl').write_text(rows)
            record = read_json(root/'linear/run.json')
            record['exposure']['sampled_anchors'] += 1
            write_json(root/'linear/run.json', record)
            with self.assertRaisesRegex(ValueError, 'exposure'):
                verify_pairing(root/'log', root/'linear')
            record['exposure']['sampled_anchors'] -= 1
            record['status'] = 'FAILED'
            write_json(root/'linear/run.json', record)
            with self.assertRaisesRegex(ValueError, 'completed'):
                verify_pairing(root/'log', root/'linear')

    def test_initial_tensor_mismatch_is_rejected_even_if_initial_output_constant(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp); a, b = self.configs()
            for name, cfg in (('log', a), ('linear', b)):
                train_support_gate(tiny_train(), cfg, root/name, device='cpu')
            path = root/'linear/step-0.pt'
            checkpoint = torch.load(path, weights_only=False)
            checkpoint['model']['head.0.bias'][0] += .1
            torch.save(checkpoint, path)
            with self.assertRaisesRegex(ValueError, 'equal model parameters'):
                verify_pairing(root/'log', root/'linear')

    def test_detached_log_objective_audit_does_not_change_training(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp); cfg, _ = self.configs()
            for name, enabled in (('old', False), ('audited', True)):
                train_support_gate(tiny_train(), cfg, root/name, record_objectives=enabled)
            old, audited = [torch.load(root/name/'final.pt', weights_only=False)
                            for name in ('old', 'audited')]
            for key in old['model']:
                self.assertTrue(torch.equal(old['model'][key], audited['model'][key]))
            self.assertEqual(old['numpy_rng'], audited['numpy_rng'])
            self.assertTrue(torch.equal(old['torch_rng'], audited['torch_rng']))
            a, b = [[json.loads(line) for line in (root/name/'training.jsonl').read_text().splitlines()]
                    for name in ('old', 'audited')]
            for left, right in zip(a, b):
                self.assertEqual(left, {key: right[key] for key in left})
                self.assertIn('anchor_mean_linear', right['objectives'])
                self.assertIn('anchor_mean_log1p', right['objectives'])

    def test_real_training_cpu_full_population_and_posthoc_pair_chain(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp).resolve(); a, b = self.configs()
            source, metadata, validation = support_gate_cache_fixture(root)
            for name, cfg in (('linear', b), ('log', a)):
                out = root/name; out.mkdir()
                fit = train_support_gate(tiny_train(), cfg, out/'fit', device='cpu')
                with patch('association.evaluation.support_gate.frozen_motion', side_effect=lambda raw, *_: raw):
                    for label, file, step in (('initial', 'step-0.pt', 0), ('final', 'final.pt', 3)):
                        model = entry.endpoint_model(out/'fit'/file, step, cfg, fit)
                        evaluate_support_gate(model, validation, source, metadata, out/label)
                        entry.annotate_endpoint(out/label, cfg, step)
                compared = compare_cached_endpoints(out/'initial', out/'final', metadata['component_ids'], out/'comparison')
                self.assertEqual(compared['objectives']['regions']['all']['point_count'], 127)
                self.assertEqual(compared['objectives']['population']['duplicate_prediction_group_count'], 1)
                record = dict(scope=SCOPE, status='COMPLETED', configuration=cfg,
                    source_system=str(source), optimizer_updates=3, test_opened=False,
                    completed_endpoints=['initial', 'final'], validation_selection=False)
                write_json(out/'run.json', record); write_json(out/'summary.json', record)
            pair = compare_replica_runs(root/'log', root/'linear', root/'paired')
            self.assertEqual(pair['status'], 'COMPLETED')
            self.assertTrue(pair['initial_cpu_outputs_exact'])
            self.assertEqual(pair['pairing']['matched_updates'], 3)
            self.assertEqual(pair['final_population']['duplicate_prediction_group_count'], 1)
            self.assertTrue(read_json(root/'paired/run.json')['paired'])
            with self.assertRaises(FileExistsError):
                compare_replica_runs(root/'log', root/'linear', root/'paired')
            failed = read_json(root/'log/run.json'); failed['status'] = 'FAILED'
            write_json(root/'log/run.json', failed)
            with self.assertRaisesRegex(ValueError, 'completed'):
                compare_replica_runs(root/'log', root/'linear', root/'failed-pair')
            state = read_json(root/'failed-pair/run.json')
            self.assertEqual(state['status'], 'FAILED')
            self.assertFalse(state['paired'])
            self.assertFalse((root/'failed-pair/summary.json').exists())

    def test_entry_source_failure_never_prepares_or_trains(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = Namespace(source_system=str(root/'source056/system'), output=str(root/'out'),
                archive=['sealed-must-not-open.zip'], device='cpu', cpu_threads=4, time_policy='advisory',
                config=str(entry.ROOT/'configs/support-gate-replica-log-level1.json'),
                manifest=str(entry.ROOT/'configs/data/mtad-level1-scene3000.json'),
                alignment=str(entry.ROOT/'configs/data/mtad-level1-alignment.json'))
            with patch.object(entry, '_bind_source', side_effect=ValueError('source binding failed')), \
                    patch.object(entry, 'load_timed_dataset') as prepare, patch.object(entry, 'train_support_gate') as train:
                with self.assertRaisesRegex(ValueError, 'binding failed'):
                    entry.run(args)
            prepare.assert_not_called(); train.assert_not_called()
            self.assertEqual(read_json(root/'out/run.json')['status'], 'FAILED')
            self.assertFalse((root/'out/summary.json').exists())


if __name__ == '__main__':
    unittest.main()
