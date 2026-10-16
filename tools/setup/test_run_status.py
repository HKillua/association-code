"""Local shell-status regression tests; no server, GPU, package install or network."""
from pathlib import Path
import os
import signal
import subprocess
import tempfile
import time
import unittest


HELPER = Path(__file__).with_name('run_status.sh').resolve()


class RunStatusTests(unittest.TestCase):
    def run_task(self, body, interruption=None):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            command = ('set -euo pipefail\nsource "$1"\ntask_status_init "$2" setup\n' + body)
            process = subprocess.Popen(['bash', '-c', command, 'test', str(HELPER), str(output)],
                                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                                       start_new_session=True)
            try:
                if interruption is not None:
                    deadline = time.monotonic() + 5
                    while not (output / 'ready').exists():
                        if process.poll() is not None or time.monotonic() >= deadline:
                            self.fail('Task did not reach its interruptible stage')
                        time.sleep(.01)
                    os.killpg(process.pid, interruption)
                stdout, stderr = process.communicate(timeout=5)
                files = {path.name: path.read_text() for path in output.iterdir() if path.is_file()}
                return process.returncode, stdout, stderr, files
            finally:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.communicate(timeout=5)

    def test_explicit_completion_is_success(self):
        rc, _, stderr, files = self.run_task('task_begin cpu\ntask_done\ntask_complete\n')
        self.assertEqual(rc, 0, stderr)
        self.assertEqual(files['setup_exit_code.txt'], '0\n')
        self.assertEqual(files['setup_completed.txt'], 'completed\n')
        self.assertIn('\tcpu\tcomplete\n', files['stages.tsv'])

    def test_failed_command_does_not_complete_stage(self):
        rc, _, _, files = self.run_task('task_run_logged cpu "$2/cpu.log" bash -c "exit 7"\ntask_complete\n')
        self.assertEqual(rc, 7)
        self.assertEqual(files['cpu_exit_code.txt'], '7\n')
        self.assertEqual(files['setup_exit_code.txt'], '7\n')
        self.assertNotIn('setup_completed.txt', files)
        self.assertNotIn('\tcomplete\n', files['stages.tsv'])

    def test_clean_exit_without_acceptance_is_failure(self):
        rc, _, _, files = self.run_task('task_begin cpu\nexit 0\n')
        self.assertEqual(rc, 1)
        self.assertEqual(files['setup_exit_code.txt'], '1\n')
        self.assertNotIn('setup_completed.txt', files)

    def test_signals_preserve_nonzero_status_and_incomplete_stage(self):
        for signum, expected in ((signal.SIGTERM, 143), (signal.SIGINT, 130), (signal.SIGHUP, 129)):
            with self.subTest(signal=signum):
                rc, _, _, files = self.run_task('task_begin cpu\ntouch "$2/ready"\nsleep 30\ntask_done\ntask_complete\n', signum)
                self.assertEqual(rc, expected)
                self.assertEqual(files['setup_exit_code.txt'], str(expected) + '\n')
                self.assertEqual(files['setup_signal.txt'], signal.Signals(signum).name[3:] + '\n')
                self.assertNotIn('setup_completed.txt', files)
                self.assertNotIn('\tcomplete\n', files['stages.tsv'])


if __name__ == '__main__':
    unittest.main()
