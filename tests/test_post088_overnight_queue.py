"""Finite scheduling semantics; systemd/GPU discovery itself is server-verified."""
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from tools.experiments import post088_overnight_queue as queue


class OvernightQueueTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def record(self, name, *, state='COMPLETED', code=0, gpu='GPU-one', verified=True):
        output = self.root / name
        output.mkdir(exist_ok=True)
        launch = Path(str(output) + '-launch')
        launch.mkdir(exist_ok=True)
        queue.write(launch / 'launch.json', dict(status='SUBMITTED', backend='systemd', unit='mtad-test.service'))
        queue.write(launch / 'status.json', dict(status=state, returncode=code))
        queue.write(launch / 'request.json', dict(environment={'CUDA_VISIBLE_DEVICES': gpu}))
        queue.write(output / 'run.json', dict(status=state))
        if verified:
            queue.write(output / 'completion-value-verification.json', dict(status='COMPLETED'))
        return output

    def test_sources_require_exit_success_verification_and_released_service(self):
        output = self.record('source')
        self.assertEqual(queue.recorded_state(output, require_verification=True, inactive=lambda _: False), 'RUNNING')
        self.assertEqual(queue.recorded_state(output, require_verification=True, inactive=lambda _: True), 'COMPLETED')
        (output / 'completion-value-verification.json').unlink()
        self.assertEqual(queue.recorded_state(output, require_verification=True, inactive=lambda _: True), 'FAILED')
        queue.write(output / 'run.json', dict(status='RUNNING'))
        self.assertEqual(queue.recorded_state(output, inactive=lambda _: True), 'FAILED')
        self.assertEqual(queue.recorded_state(self.root / 'unknown'), 'NOT_STARTED')
        (self.root / 'untracked').mkdir()
        self.assertEqual(queue.recorded_state(self.root / 'untracked'), 'UNTRACKED')

    def test_failed_or_lost_worker_never_counts_as_success(self):
        failed = self.record('failed', state='FAILED', code=7)
        self.assertEqual(queue.recorded_state(failed, inactive=lambda _: True), 'FAILED')
        lost = self.record('lost', state='RUNNING', code=None)
        self.assertEqual(queue.recorded_state(lost, inactive=lambda _: True), 'FAILED')
        Path(str(lost) + '-launch/status.json').unlink()
        self.assertEqual(queue.recorded_state(lost, inactive=lambda _: True), 'FAILED')

    def test_unknown_submission_observes_live_or_completed_unit_without_retry(self):
        output = self.record('ambiguous', state='RUNNING', code=None)
        launch = Path(str(output) + '-launch')
        queue.write(launch / 'launch.json', dict(status='SUBMISSION_UNKNOWN', backend='systemd', unit='known.service'))
        self.assertEqual(queue.recorded_state(output, inactive=lambda _: False), 'RUNNING')
        queue.write(launch / 'status.json', dict(status='COMPLETED', returncode=0))
        queue.write(output / 'run.json', dict(status='COMPLETED'))
        self.assertEqual(queue.recorded_state(output, inactive=lambda _: True), 'COMPLETED')
        queue.write(launch / 'launch.json', dict(status='FAILED', backend='systemd', unit='known.service'))
        self.assertEqual(queue.recorded_state(output, inactive=lambda _: False), 'RUNNING')
        self.assertEqual(len(queue.active_gpu_claims(self.root, inactive=lambda _: False)), 1)

    def test_dead_submitted_unit_without_worker_status_releases_its_slot(self):
        output = self.record('dead', state='RUNNING', code=None)
        launch = Path(str(output) + '-launch')
        (launch / 'status.json').unlink()
        for state in ('SUBMITTED', 'SUBMISSION_UNKNOWN'):
            queue.write(launch / 'launch.json', dict(status=state, backend='systemd', unit='known.service'))
            self.assertEqual(queue.recorded_state(output, inactive=lambda _: True), 'FAILED')
            self.assertEqual(queue.active_gpu_claims(self.root, inactive=lambda _: True), [])
        queue.write(launch / 'launch.json', dict(status='SUBMISSION_UNKNOWN', backend='systemd'))
        self.assertEqual(queue.active_gpu_claims(self.root, inactive=lambda _: False), [])

    def test_global_gpu_count_keeps_foreign_pending_and_cleanup_claims(self):
        self.record('done')
        self.record('active', state='RUNNING', code=None)
        self.record('cpu', state='RUNNING', code=None, gpu='')
        (self.root / 'submitting-launch').mkdir()
        claims = queue.active_gpu_claims(self.root, inactive=lambda launch: launch['unit'] != 'active.service')
        # Distinct units let a live CPU phase retain its GPU claim.
        queue.write(self.root / 'active-launch/launch.json',
                    dict(status='SUBMITTED', backend='systemd', unit='active.service'))
        claims = queue.active_gpu_claims(self.root, inactive=lambda launch: launch['unit'] != 'active.service')
        self.assertEqual({Path(path).name for path in claims}, {'active-launch', 'submitting-launch'})
        self.assertEqual(len(queue.active_gpu_claims(self.root, inactive=lambda _: False)), 3)

    def test_gpu_in_cpu_phase_remains_reserved_even_if_nvidia_reports_idle(self):
        self.record('cpu-phase', state='RUNNING', code=None, gpu='GPU-one')
        claims = queue.active_gpu_claims(self.root, inactive=lambda _: False)
        self.assertEqual(queue.reserved_gpus(claims, ['GPU-one', 'GPU-two']), {'GPU-one'})
        queue.write(self.root / 'cpu-phase-launch/request.json',
                    dict(environment={'CUDA_VISIBLE_DEVICES': '0'}))
        self.assertEqual(queue.reserved_gpus(claims, ['GPU-one', 'GPU-two']), {'GPU-one', 'GPU-two'})
        (self.root / 'unknown-launch').mkdir()
        self.assertEqual(queue.reserved_gpus([self.root / 'unknown-launch'], ['GPU-one']), {'GPU-one'})

    def test_missing_request_does_not_hide_live_ownership(self):
        self.record('lost-request', state='RUNNING', code=None)
        (self.root / 'lost-request-launch/request.json').unlink()
        claims = queue.active_gpu_claims(self.root, inactive=lambda _: False)
        self.assertEqual(len(claims), 1)
        self.assertEqual(queue.reserved_gpus(claims, ['GPU-one']), {'GPU-one'})
        self.assertEqual(queue.active_gpu_claims(self.root, inactive=lambda _: True), [])

    def test_idle_gpu_selection_excludes_other_processes_and_unknown_devices(self):
        def run(argv, **kwargs):
            if '--query-gpu=uuid,memory.used,utilization.gpu' in argv:
                return SimpleNamespace(stdout='GPU-a, 12, 0\nGPU-b, 12, 0\nGPU-c, 300, 0\nGPU-d, 12, 1\n')
            return SimpleNamespace(stdout='GPU-b, 400\n')
        self.assertEqual(queue.available_gpus(['GPU-a', 'GPU-b', 'GPU-c', 'GPU-d'], run=run), ['GPU-a'])
        with self.assertRaisesRegex(ValueError, 'missing'):
            queue.available_gpus(['GPU-missing'], run=run)

    def test_plan_is_bounded_correct_dependencies_and_advisory(self):
        plan = queue.build_plan(self.root, [self.root / 'part1', self.root / 'part2', self.root / 'part3'])
        jobs = {row['id']: row for row in plan['jobs']}
        self.assertEqual(set(jobs), {'096A', '096B', '097A', '093-summary', '094-summary', '096-summary', '097-summary'})
        self.assertEqual(sum(row['gpu'] for row in jobs.values()), 3)
        self.assertEqual(jobs['096A']['dependencies'], ['093A', '094A'])
        self.assertEqual(jobs['096B']['dependencies'], ['093B', '094B'])
        self.assertEqual(jobs['097A']['dependencies'], [])
        self.assertIn('advisory', jobs['097A']['argv'])
        self.assertTrue(any('073C' in value for value in jobs['097A']['argv']))
        self.assertEqual(queue.dependency_decision(['a'], {'a': 'FAILED'}), 'BLOCKED')
        self.assertEqual(queue.dependency_decision(['a'], {'a': 'RUNNING'}), 'WAITING')
        self.assertEqual(queue.dependency_decision([], {}), 'READY')

    def test_one_attempt_failure_propagates_and_independent_job_completes(self):
        plan = dict(code=str(queue.ROOT), run_root=str(self.root), sources={}, jobs=[])
        for name, dependencies in [('bad', []), ('independent', []), ('dependent', ['bad'])]:
            plan['jobs'].append(dict(id=name, output=str(self.root / name), argv=['/usr/bin/python3', 'unused'],
                                     dependencies=dependencies, gpu=False, preflight=None, max_seconds=3600))
        submitted = []

        def launch(argv, **kwargs):
            record = Path(argv[argv.index('--record-dir') + 1])
            name = record.name.removesuffix('-launch')
            submitted.append(name)
            self.record(name, state='FAILED' if name == 'bad' else 'COMPLETED', code=7 if name == 'bad' else 0, gpu='')
            return SimpleNamespace(returncode=0)

        original = queue.recorded_state
        with patch.object(queue, 'recorded_state', side_effect=lambda path, **kw: original(path, **kw, inactive=lambda _: True)), \
             patch.object(queue, 'active_gpu_claims', return_value=[]), \
             patch.object(queue.subprocess, 'run', side_effect=launch), patch.object(queue.time, 'sleep'):
            code = queue.run_queue(plan, self.root / 'queue', ['GPU-one'])
        self.assertEqual(code, 1)
        self.assertEqual(submitted, ['bad', 'independent'])
        result = queue.read(self.root / 'queue/run.json')
        self.assertEqual(result['jobs']['dependent']['status'], 'BLOCKED')
        self.assertEqual(result['jobs']['independent']['status'], 'COMPLETED')
        self.assertEqual(result['jobs']['bad']['attempts'], 1)
        with self.assertRaises(FileExistsError):
            queue.run_queue(plan, self.root / 'queue', ['GPU-one'])

    def test_existing_completed_job_is_observed_not_submitted_again(self):
        existing = self.record('existing', gpu='')
        plan = dict(code=str(queue.ROOT), run_root=str(self.root), sources={}, jobs=[
            dict(id='existing', output=str(existing), argv=['unused'], dependencies=[],
                 gpu=False, preflight=None, max_seconds=3600)])
        original = queue.recorded_state
        with patch.object(queue, 'recorded_state', side_effect=lambda path, **kw: original(path, **kw, inactive=lambda _: True)), \
             patch.object(queue, 'active_gpu_claims', return_value=[]), \
             patch.object(queue.subprocess, 'run', side_effect=AssertionError('duplicate submit')):
            self.assertEqual(queue.run_queue(plan, self.root / 'queue', ['GPU-one']), 0)
        self.assertTrue(queue.read(self.root / 'queue/run.json')['jobs']['existing']['existing_attempt_reused'])


if __name__ == '__main__':
    unittest.main()
