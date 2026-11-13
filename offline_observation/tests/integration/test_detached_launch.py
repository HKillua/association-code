"""Real short POSIX jobs; systemd command construction is faked, not live-tested."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT/'tools/experiments/launch_detached.py'


@unittest.skipUnless(os.name == 'posix', 'requires POSIX sessions and process groups')
class DetachedLaunchTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def args(self, name, code, *, backend='session', budget=8., grace=.4, time_policy='hard'):
        return ['--backend', backend, '--cwd', str(self.root), '--record-dir', str(self.root/name),
                '--max-seconds', str(budget), '--grace-seconds', str(grace),
                '--time-policy', time_policy,
                '--env', 'DETACHED_TEST_LITERAL=$not_expanded;$(no_shell)',
                '--', sys.executable, '-c', code]

    def invoke(self, args, **kwargs):
        return subprocess.run([sys.executable, str(SCRIPT), *args], cwd=ROOT,
                              capture_output=True, text=True, timeout=10, **kwargs)

    def wait_file(self, path, *, terminal=False):
        until = time.monotonic()+10
        while time.monotonic() < until:
            if path.exists():
                if not terminal:
                    return path.read_text()
                value = json.loads(path.read_text())
                if value['status'] != 'RUNNING':
                    return value
            time.sleep(.025)
        self.fail(f'no expected artifact: {path}')

    def test_parent_termination_keeps_payload_logs_and_final_returncode(self):
        code = "import os,sys,time; from pathlib import Path; assert sys.stdin.read()==''; time.sleep(.5); Path('survived').write_text(os.environ['DETACHED_TEST_LITERAL']); print('out'); print('err',file=sys.stderr)"
        parent_code = "import sys,time; from pathlib import Path; from tools.experiments.launch_detached import main; assert main(sys.argv[1:])==0; Path('"+str(self.root/'parent-ready')+"').write_text('ready'); time.sleep(30)"
        parent = subprocess.Popen([sys.executable, '-c', parent_code, *self.args('parent', code)], cwd=ROOT,
                                  stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            self.wait_file(self.root/'parent-ready')
            parent.terminate()
            parent.communicate(timeout=3)
            result = self.wait_file(self.root/'parent'/'status.json', terminal=True)
            self.assertEqual(result['status'], 'COMPLETED')
            self.assertEqual(result['returncode'], 0)
            self.assertEqual((self.root/'survived').read_text(), '$not_expanded;$(no_shell)')
            self.assertEqual((self.root/'parent'/'stdout.log').read_text(), 'out\n')
            self.assertEqual((self.root/'parent'/'stderr.log').read_text(), 'err\n')
        finally:
            if parent.poll() is None:
                parent.kill()
                parent.communicate()

    def test_real_nonzero_exit_and_existing_directory_are_preserved(self):
        args = self.args('failure', 'raise SystemExit(7)')
        self.assertEqual(self.invoke(args).returncode, 0)  # Submission, not payload success.
        result = self.wait_file(self.root/'failure'/'status.json', terminal=True)
        self.assertEqual((result['status'], result['returncode']), ('FAILED', 7))
        self.assertEqual((self.root/'failure'/'exit_code.txt').read_text(), '7\n')
        original = {p.name: p.read_bytes() for p in (self.root/'failure').iterdir()}
        self.assertEqual(self.invoke(args).returncode, 1)
        self.assertEqual(original, {p.name: p.read_bytes() for p in (self.root/'failure').iterdir()})

    def test_timeout_kills_term_ignoring_grandchild_even_after_leader_exits(self):
        child = "import os,signal,time; from pathlib import Path; signal.signal(signal.SIGTERM,signal.SIG_IGN); Path('grandchild.pid').write_text(str(os.getpid())); time.sleep(4); Path('escaped').write_text('bad')"
        cooperative = "import signal,time; from pathlib import Path\ndef stop(signum, frame):\n Path('cleanup-started').write_text('yes'); time.sleep(.2); Path('cleanup-completed').write_text('yes'); raise SystemExit(0)\nsignal.signal(signal.SIGTERM, stop); Path('cooperative-ready').write_text('yes'); time.sleep(30)"
        code = "import subprocess,sys,time; subprocess.Popen([sys.executable,'-c',"+repr(child)+"]); subprocess.Popen([sys.executable,'-c',"+repr(cooperative)+"]); time.sleep(30)"
        self.assertEqual(self.invoke(self.args('timeout', code, budget=3., grace=.4)).returncode, 0)
        pid = int(self.wait_file(self.root/'grandchild.pid'))
        self.wait_file(self.root/'cooperative-ready')
        result = self.wait_file(self.root/'timeout'/'status.json', terminal=True)
        self.assertTrue(result['timed_out'])
        self.assertEqual(result['status'], 'TIMED_OUT')
        self.assertEqual(result['returncode'], -signal.SIGTERM)
        self.assertLess(result['elapsed_seconds'], 3.)
        # A killed orphan may briefly be a zombie awaiting init; it must not run.
        state = subprocess.run(['ps', '-p', str(pid), '-o', 'stat='], capture_output=True, text=True).stdout.strip()
        self.assertTrue(not state or state.startswith('Z'), state)
        self.assertFalse((self.root/'escaped').exists())
        self.assertTrue((self.root/'cleanup-completed').exists())

    def test_advisory_crossing_continues_the_same_payload_to_completion(self):
        code = "import time; time.sleep(.6); print('completed after recommendation')"
        self.assertEqual(self.invoke(self.args('advisory', code, budget=.2, time_policy='advisory')).returncode, 0)
        record = self.root/'advisory'
        advisory = json.loads(self.wait_file(record/'time_advisory.json'))
        result = self.wait_file(record/'status.json', terminal=True)
        self.assertEqual((result['status'], result['returncode']), ('COMPLETED', 0))
        self.assertTrue(result['time_advisory_issued'])
        self.assertFalse(result['timed_out'])
        self.assertFalse(result['deadline_exceeded'])
        self.assertTrue(advisory['needs_user_review'])
        self.assertEqual(advisory['action'], 'continued_current_job')
        self.assertEqual(advisory['recommended_seconds'], .2)
        self.assertGreaterEqual(advisory['elapsed_seconds'], .2)
        self.assertEqual(json.loads((record/'time_advisory.json').read_text()), advisory)
        self.assertEqual((record/'stdout.log').read_text(), 'completed after recommendation\n')
        self.assertEqual((record/'worker.stderr.log').read_text().count('TIME ADVISORY:'), 1)

    def test_explicit_stop_after_advisory_still_cleans_payload_group(self):
        child = "import os,signal,time; from pathlib import Path; signal.signal(signal.SIGTERM,signal.SIG_IGN); Path('advisory-child.pid').write_text(str(os.getpid())); time.sleep(30)"
        code = "import subprocess,sys,time; subprocess.Popen([sys.executable,'-c',"+repr(child)+"]); time.sleep(30)"
        self.assertEqual(self.invoke(self.args('stop-advisory', code, budget=.3, grace=.2, time_policy='advisory')).returncode, 0)
        record = self.root/'stop-advisory'
        pid = int(self.wait_file(self.root/'advisory-child.pid'))
        self.wait_file(record/'time_advisory.json')
        os.kill(int(self.wait_file(record/'worker.pid')), signal.SIGTERM)
        result = self.wait_file(record/'status.json', terminal=True)
        self.assertEqual(result['status'], 'INTERRUPTED')
        self.assertTrue(result['time_advisory_issued'])
        self.assertFalse(result['timed_out'])
        state = subprocess.run(['ps', '-p', str(pid), '-o', 'stat='], capture_output=True, text=True).stdout.strip()
        self.assertTrue(not state or state.startswith('Z'), state)

    def test_systemd_uses_detached_service_budget_and_linger_without_shell(self):
        binary = self.root/'bin'
        binary.mkdir()
        capture = self.root/'systemd-argv.json'
        for name, body in (
                ('loginctl', "print('yes')"),
                ('systemd-run', "import json,sys; from pathlib import Path; Path("+repr(str(capture))+").write_text(json.dumps(sys.argv[1:]))")):
            path = binary/name
            path.write_text('#!'+sys.executable+'\n'+body+'\n')
            path.chmod(0o755)
        env = {**os.environ, 'PATH': str(binary)+os.pathsep+os.environ.get('PATH', '')}
        args = self.args('systemd', "print('$literal;$(no_shell)')", backend='systemd', budget=20., grace=2.)
        self.assertEqual(self.invoke(args, env=env).returncode, 0)
        command = json.loads(capture.read_text())
        for required in ('--user', '--property=Type=exec', '--property=Restart=no',
                         '--property=KillMode=control-group', '--property=StandardInput=null'):
            self.assertIn(required, command)
        self.assertFalse(any(flag in command for flag in ('--scope', '--pipe', '--pty', '--wait', '--collect')))
        maximum = float(next(item.split('=', 2)[2][:-1] for item in command if item.startswith('--property=RuntimeMaxSec=')))
        self.assertLessEqual(maximum+2., 20.)
        self.assertIn('--property=TimeoutStopSec=2.000000s', command)
        request = json.loads((self.root/'systemd'/'request.json').read_text())
        self.assertEqual(request['argv'][-1], "print('$literal;$(no_shell)')")
        self.assertFalse((self.root/'systemd'/'status.json').exists())  # No real user manager used.
        advisory_args = self.args('systemd-advisory', 'pass', backend='systemd', budget=.1, time_policy='advisory')
        self.assertEqual(self.invoke(advisory_args, env=env).returncode, 0)
        self.assertIn('--property=RuntimeMaxSec=infinity', json.loads(capture.read_text()))
        advisory_request = json.loads((self.root/'systemd-advisory'/'request.json').read_text())
        self.assertIsNone(advisory_request['deadline_monotonic'])
        (binary/'loginctl').write_text('#!'+sys.executable+"\nprint('no')\n")
        self.assertEqual(self.invoke(self.args('no-linger', 'pass', backend='systemd'), env=env).returncode, 1)
        self.assertEqual(json.loads((self.root/'no-linger'/'launch.json').read_text())['status'], 'FAILED')


if __name__ == '__main__':
    unittest.main()
