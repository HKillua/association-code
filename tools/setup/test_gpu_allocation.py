"""GPU allocation contract tests; nvidia-smi is mocked and no GPUs are touched."""
from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock

from tools.setup import gpu_allocation as allocation


DEVICES = '0, GPU-aa, NVIDIA A100, 40960, 2048, 85\n7, GPU-bb, NVIDIA RTX 6000, 49152, 8192, 99\n'


class GPUAllocationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.output = Path(self.temporary.name) / 'allocation.json'

    def run_cli(self, arguments=(), devices=DEVICES, process_output='', device_rc=0,
                process_rc=0, process_error=''):
        replies = [subprocess.CompletedProcess([], device_rc, devices, 'device query failed' if device_rc else ''),
                   subprocess.CompletedProcess([], process_rc, process_output, process_error)]
        stdout, stderr = io.StringIO(), io.StringIO()
        with mock.patch.object(allocation.subprocess, 'run', side_effect=replies) as run, \
                redirect_stdout(stdout), redirect_stderr(stderr):
            rc = allocation.main(['--output', str(self.output), *arguments])
        report = json.loads(self.output.read_text()) if self.output.exists() else None
        return rc, stdout.getvalue(), stderr.getvalue(), report, run

    def test_existing_process_and_high_utilization_do_not_prevent_auto_selection(self):
        rc, stdout, _, report, run = self.run_cli(process_output='GPU-bb, 1234, other-user-python, 30000\n')
        self.assertEqual(rc, 0)
        self.assertEqual(stdout, 'GPU-bb\n')
        self.assertEqual(report['selected']['index'], 7)
        self.assertEqual(report['process_query']['processes'][0]['pid'], 1234)
        self.assertTrue(report['shared_gpu_allowed'])
        self.assertEqual(len(report['candidates']), 2)
        self.assertTrue(all(call.kwargs['timeout'] == 15 for call in run.call_args_list))

    def test_arbitrary_physical_index_and_uuid_are_supported(self):
        for device in ('7', 'GPU-bb'):
            with self.subTest(device=device):
                if self.output.exists():
                    self.output.unlink()
                rc, stdout, _, report, _ = self.run_cli(['--device', device])
                self.assertEqual(rc, 0)
                self.assertEqual(stdout, 'GPU-bb\n')
                self.assertEqual(report['requested_device'], device)

    def test_all_cards_busy_with_free_memory_still_selects(self):
        devices = '4, GPU-aa, GPU A, 10000, 512, 100\n9, GPU-bb, GPU B, 10000, 1024, 100\n'
        rc, _, _, report, _ = self.run_cli(devices=devices, process_output='GPU-bb, 55, busy-python, 8000\n')
        self.assertEqual(rc, 0)
        self.assertEqual(report['selected']['index'], 9)

    def test_ties_use_utilization_then_stable_physical_index(self):
        devices = ('9, GPU-cc, GPU C, 10000, 4096, 10\n'
                   '2, GPU-aa, GPU A, 10000, 4096, 20\n'
                   '7, GPU-bb, GPU B, 10000, 4096, 10\n')
        rc, _, _, report, _ = self.run_cli(devices=devices)
        self.assertEqual(rc, 0)
        self.assertEqual(report['selected']['index'], 7)

    def test_device_query_failure_is_nonzero_and_preserves_evidence(self):
        rc, stdout, stderr, report, run = self.run_cli(device_rc=9)
        self.assertEqual(rc, 1)
        self.assertEqual(stdout, '')
        self.assertIn('Device query failed', stderr)
        self.assertEqual(report['device_query']['exit_code'], 9)
        self.assertEqual(report['status'], 'QUERY_FAILED')
        self.assertEqual(run.call_count, 1)

    def test_process_query_failure_is_recorded_but_selection_succeeds(self):
        rc, _, _, report, _ = self.run_cli(process_rc=13, process_error='permission denied')
        self.assertEqual(rc, 0)
        self.assertEqual(report['process_query']['exit_code'], 13)
        self.assertEqual(report['process_query']['stderr'], 'permission denied')
        self.assertIsNone(report['process_query']['processes'])

    def test_unparseable_device_data_is_query_error(self):
        rc, stdout, _, report, run = self.run_cli(devices='invalid,data\n')
        self.assertEqual(rc, 1)
        self.assertEqual(stdout, '')
        self.assertEqual(report['status'], 'INVALID_DEVICE_DATA')
        self.assertEqual(report['device_query']['exit_code'], 0)
        self.assertTrue(report['rejected_device_rows'])
        self.assertIsNone(report['selected'])
        self.assertEqual(run.call_count, 1)

    def test_no_valid_or_sufficient_memory_device_returns_three(self):
        scenarios = [([], ''),
                     ([], '7, GPU-bb, GPU, 10000, 0, 0\n'),
                     (['--min-free-mib', '9000'], DEVICES),
                     (['--device', '8'], DEVICES)]
        for arguments, devices in scenarios:
            with self.subTest(arguments=arguments, devices=devices):
                if self.output.exists():
                    self.output.unlink()
                rc, stdout, _, report, run = self.run_cli(arguments, devices=devices)
                self.assertEqual(rc, 3)
                self.assertEqual(stdout, '')
                self.assertEqual(report['status'], 'NO_SUITABLE_GPU')
                self.assertIsNone(report['selected'])
                self.assertEqual(run.call_count, 1)

    def test_invalid_inputs_do_not_query_devices(self):
        for arguments in (['--device', '-1'], ['--device', 'cuda:0'], ['--device', 'GPU-!'],
                          ['--min-free-mib', '-1'], ['--min-free-mib', 'nan'], ['--min-free-mib', 'inf']):
            with self.subTest(arguments=arguments), mock.patch.object(allocation.subprocess, 'run') as run, \
                    redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as stopped:
                    allocation.main(['--output', str(self.output), *arguments])
                self.assertEqual(stopped.exception.code, 2)
                run.assert_not_called()
                self.assertFalse(self.output.exists())

    def test_existing_output_is_never_overwritten_or_requeried(self):
        self.output.write_text('previous evidence\n')
        with mock.patch.object(allocation.subprocess, 'run') as run, redirect_stderr(io.StringIO()):
            rc = allocation.main(['--output', str(self.output)])
        self.assertEqual(rc, 1)
        self.assertEqual(self.output.read_text(), 'previous evidence\n')
        run.assert_not_called()

    def test_each_query_timeout_has_correct_failure_policy(self):
        timeout = subprocess.TimeoutExpired(['nvidia-smi'], 15, output=b'partial', stderr=b'timed out')
        with mock.patch.object(allocation.subprocess, 'run', side_effect=timeout):
            with self.assertRaises(allocation.AllocationError) as stopped:
                allocation.allocate()
        self.assertTrue(stopped.exception.report['device_query']['timed_out'])
        replies = [subprocess.CompletedProcess([], 0, DEVICES, ''), timeout]
        with mock.patch.object(allocation.subprocess, 'run', side_effect=replies):
            report = allocation.allocate()
        self.assertEqual(report['status'], 'SELECTED')
        self.assertTrue(report['process_query']['timed_out'])


if __name__ == '__main__':
    unittest.main()
