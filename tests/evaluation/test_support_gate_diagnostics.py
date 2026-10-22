"""069 execution binding and read-only failure boundaries, not a server replay."""
from argparse import Namespace
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from tools.analysis import support_gate_diagnostics as cli


class SupportGateDiagnosticEntryTests(unittest.TestCase):
    def bind_fixture(self, root):
        """Original returned records; empty files test inventory only, not arrays."""
        source = root/cli.TASK
        returned = cli.ROOT/'HandOver/results/LEVEL1-SUPPORT-GATE-20261023-068'
        source.mkdir()
        for name in ('run.json', 'summary.json', 'fit/run.json', 'preflight/report.json'):
            destination = source/name
            destination.parent.mkdir(parents=True, exist_ok=True)
            cli.write(destination, cli.read(returned/name))
        for endpoint, filename in (('initial', 'step-0.pt'), ('final', 'final.pt')):
            directory = source/endpoint
            directory.mkdir()
            for output, original in (('summary.json', 'summary.json'), ('run.json', 'endpoint-run.json')):
                record = cli.read(returned/endpoint/original)
                record['checkpoint']['path'] = str(source/'fit'/filename)
                cli.write(directory/output, record)
            for part in ('predictions', 'system-predictions'):
                (directory/part).mkdir()
                for sid in record['scene_ids']:
                    (directory/part/f'states-{sid}.npz').touch()
        return source

    def test_real_returned_metadata_bind_and_endpoint_budget_population_rejections(self):
        with TemporaryDirectory() as temporary:
            source = self.bind_fixture(Path(temporary))
            result = cli.bind_source(source)
            self.assertEqual(len(result['scene_ids']), 300)
            self.assertEqual(len(set(result['components'].values())), 38)
            path = source/'final/run.json'
            original = cli.read(path)
            for mutate in (lambda r: r['checkpoint'].update(step=999),
                           lambda r: r.update(completed_scenes=299),
                           lambda r: r.update(groups_recomputed=True)):
                changed = deepcopy(original)
                mutate(changed)
                cli.write(path, changed)
                with self.assertRaises(ValueError):
                    cli.bind_source(source)
                cli.write(path, original)
            (source/'initial/predictions/states-1.npz').unlink()
            with self.assertRaisesRegex(ValueError, 'missing/extra'):
                cli.bind_source(source)

    def test_no_source_write_no_retry_and_failure_never_emits_success_summary(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root/cli.TASK
            source.mkdir()
            marker = source/'marker'
            marker.write_text('immutable')
            for destination in (source, source/'nested', root):
                with self.assertRaisesRegex(ValueError, 'separate'):
                    cli.run(Namespace(source_068=source, output=destination, cpu_threads=4))
            output = root/'069'
            args = Namespace(source_068=source, output=output, cpu_threads=4)
            with patch.object(cli, 'bind_source', side_effect=ValueError('tampered source')):
                with self.assertRaisesRegex(ValueError, 'tampered'):
                    cli.run(args)
            state = cli.read(output/'run.json')
            self.assertEqual(state['status'], 'FAILED')
            self.assertFalse((output/'summary.json').exists())
            self.assertEqual(state['model_forwards'], 0)
            self.assertEqual(marker.read_text(), 'immutable')
            with self.assertRaises(FileExistsError):
                cli.run(args)

    def test_actual_small_training_log_and_component_report_are_consumable(self):
        # Producer executes only a tiny CPU software fixture; diagnostics cannot forward.
        from association.training.support_gate import train_support_gate
        from tests.training.test_support_gate import tiny_config, tiny_train
        from tests.evaluation.test_gate_reference_audit import GateReferenceAuditTests
        from association.evaluation.support_gate_objectives import REGIONS
        import torch

        torch.set_num_threads(1)
        reference_case = GateReferenceAuditTests()
        reference_case.setUp()
        self.addCleanup(reference_case.doCleanups)
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = tiny_config()
            config['training']['batch_size'] = 8
            train_support_gate(tiny_train(), config, root/'fit', device='cpu')
            with patch('torch.load', side_effect=AssertionError('no checkpoint')), \
                    patch('torch.nn.Module.__call__', side_effect=AssertionError('no forward')):
                training = cli.audit_training(root/'fit', expected_updates=3)
                reference = cli.audit_reference(reference_case.root, reference_case.rows)
                for row in reference_case.rows:
                    shape = row['errors']['initial'].shape
                    row['regions'] = {name: np.zeros(shape, bool) for name in REGIONS}
                    row['regions']['all'][:] = True
                    row['regions']['anchor_inside_support'][:] = True
                    row.update(population=dict(target_identity_count=len(row['target_ids'])),
                               model_state_comparison={})
                objectives = cli.summarize(reference_case.rows,
                    {row['scene_id']: 'shared' for row in reference_case.rows})
                report = cli.render_report(objectives, training, reference)
                self.assertIn('066 REJECTED不变', report)
                self.assertIn('成功更新 3', report)
                self.assertIn('不是清洗后成绩', report)
                cli.write(root/'outputs.json.gz', dict(objectives=objectives,
                    training=training, reference=reference))
                self.assertEqual(cli.read(root/'outputs.json.gz')['reference']['full_population_position_points'], 5)


if __name__ == '__main__':
    unittest.main()
