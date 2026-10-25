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

    def test_dead_acknowledged_unit_releases_but_unknown_submission_keeps_ownership(self):
        output = self.record('dead', state='RUNNING', code=None)
        launch = Path(str(output) + '-launch')
        (launch / 'status.json').unlink()
        for state in ('SUBMITTED', 'SUBMISSION_UNKNOWN'):
            queue.write(launch / 'launch.json', dict(status=state, backend='systemd', unit='known.service'))
            self.assertEqual(queue.recorded_state(output, inactive=lambda _: True), 'FAILED')
            self.assertEqual(bool(queue.active_gpu_claims(self.root, inactive=lambda _: True)),
                             state == 'SUBMISSION_UNKNOWN')
        queue.write(launch / 'launch.json', dict(status='SUBMISSION_UNKNOWN', backend='systemd'))
        self.assertEqual(queue.active_gpu_claims(self.root, inactive=lambda _: False), [])

    def test_interrupted_submission_ends_wait_but_keeps_ambiguous_gpu_ownership(self):
        output = self.record('interrupted-submit', state='RUNNING', code=None)
        launch = Path(str(output) + '-launch')
        (launch / 'status.json').unlink()
        queue.write(launch / 'launch.json', dict(status='SUBMITTING', backend='systemd',
                    unit='known.service', launcher_pid=123))
        for unit_inactive, process_alive, expected in [
                (True, True, 'WAITING'), (False, False, 'WAITING'), (True, False, 'FAILED')]:
            self.assertEqual(queue.recorded_state(output, inactive=lambda _: unit_inactive,
                              submitting_alive=lambda _: process_alive), expected)
            claims = queue.active_gpu_claims(self.root, inactive=lambda _: unit_inactive)
            self.assertTrue(claims)
        self.assertTrue(queue.submission_failure_details(output)['gpu_claim_preserved'])
        (launch / 'request.json').unlink()
        self.assertTrue(queue.active_gpu_claims(self.root, inactive=lambda _: True))
        queue.write(launch / 'status.json', dict(status='FAILED', returncode=1))
        self.assertEqual(queue.active_gpu_claims(self.root, inactive=lambda _: True), [])

    def test_collected_units_are_inactive_but_bus_failures_do_not_release_claims(self):
        launch = dict(backend='systemd', unit='old.service')
        for returncode in (0, 4):
            with patch.object(queue.subprocess, 'run', return_value=SimpleNamespace(
                    returncode=returncode, stdout='LoadState=not-found\nActiveState=inactive\n')):
                self.assertTrue(queue.service_inactive(launch))
        for stdout in ('', 'ActiveState=inactive\n'):
            with patch.object(queue.subprocess, 'run', return_value=SimpleNamespace(
                    returncode=1, stdout=stdout)):
                with self.assertRaisesRegex(RuntimeError, 'cannot establish'):
                    queue.service_inactive(launch)
        with patch.object(queue.subprocess, 'run', return_value=SimpleNamespace(
                returncode=0, stdout='LoadState=loaded\nActiveState=active\n')):
            self.assertFalse(queue.service_inactive(launch))

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

    def dispatch_barrier_fixture(self):
        output = self.root/'LEVEL1-POST088-NIGHT-20261025-096-097-attempt-1'
        output.mkdir()
        plan = queue.build_plan(self.root, [self.root/'part1'])
        queue.write(output/'plan.json', plan)
        state = dict(status='RUNNING', jobs={job['id']: dict(status='WAITING') for job in plan['jobs']})
        queue.write(output/'run.json', state)
        for job in plan['jobs']:
            if not job['gpu']:
                continue
            record = Path(job['output']+'-launch')
            record.mkdir()
            queue.write(record/'request.json', dict(argv=job['argv'], cwd=plan['code'],
                backend='systemd', time_policy='advisory', environment={
                    'CUDA_VISIBLE_DEVICES': 'GPU-c5a00c37-f181-90b3-f220-be6e7457f473'}))
            queue.write(record/'launch.json', dict(status='SUBMITTED', backend='systemd',
                unit='mtad-a15de726acdc.service'))
        return output, plan, state

    def test_external_dispatch_waits_for_pending_old_gpu_but_not_running_children_or_cpu(self):
        output, plan, state = self.dispatch_barrier_fixture()
        # Persisted WAITING can be stale while its acknowledged reservation is
        # already visible; the old queue cannot submit it twice.
        self.assertEqual(queue.assert_gpu_dispatch_drained(output, self.root)['status'], 'GPU_DISPATCH_DRAINED')
        first = next(job for job in plan['jobs'] if job['id'] == '096A')
        Path(first['output']+'-launch/launch.json').unlink()
        with self.assertRaisesRegex(RuntimeError, 'WAITING_RESOURCE'):
            queue.assert_gpu_dispatch_drained(output, self.root)
        # A frozen terminal dependency failure is never retried by the old
        # controller; no need to stop it or wait for remaining CPU tables.
        state['jobs']['096A'] = dict(status='BLOCKED', reason='required source failed')
        queue.write(output/'run.json', state)
        result = queue.assert_gpu_dispatch_drained(output, self.root)
        self.assertEqual(result['jobs']['096A'], 'BLOCKED')
        self.assertTrue(result['requires_fresh_resource_check'])

    def test_external_dispatch_refuses_ambiguous_or_misbound_old_reservation(self):
        output, plan, _ = self.dispatch_barrier_fixture()
        job = next(row for row in plan['jobs'] if row['id'] == '096A')
        request_path, launch_path = Path(job['output']+'-launch/request.json'), Path(job['output']+'-launch/launch.json')
        original_request, original_launch = queue.read(request_path), queue.read(launch_path)
        for status in ('SUBMITTING', 'SUBMISSION_UNKNOWN', 'FAILED'):
            queue.write(launch_path, dict(original_launch, status=status))
            with self.assertRaisesRegex(RuntimeError, 'WAITING_RESOURCE'):
                queue.assert_gpu_dispatch_drained(output, self.root)
        queue.write(launch_path, original_launch)
        for changed in (dict(original_request, argv=['wrong']),
                        dict(original_request, cwd=str(self.root/'other-checkout')),
                        dict(original_request, environment={'CUDA_VISIBLE_DEVICES': '0'})):
            queue.write(request_path, changed)
            with self.assertRaisesRegex(RuntimeError, 'WAITING_RESOURCE'):
                queue.assert_gpu_dispatch_drained(output, self.root)
        queue.write(request_path, original_request)
        self.assertEqual(queue.assert_gpu_dispatch_drained(output, self.root)['status'], 'GPU_DISPATCH_DRAINED')

    def test_external_dispatch_refuses_unknown_controller_or_gpu_plan(self):
        output, plan, state = self.dispatch_barrier_fixture()
        state['status'] = 'FAILED'
        queue.write(output/'run.json', state)
        with self.assertRaisesRegex(RuntimeError, 'WAITING_RESOURCE'):
            queue.assert_gpu_dispatch_drained(output, self.root)
        state['status'] = 'RUNNING'
        queue.write(output/'run.json', state)
        plan['jobs'].append(dict(id='extra', gpu=True, output=str(self.root/'extra')))
        queue.write(output/'plan.json', plan)
        with self.assertRaisesRegex(RuntimeError, 'WAITING_RESOURCE'):
            queue.assert_gpu_dispatch_drained(output, self.root)

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

    def test_partial_record_after_failed_launcher_is_terminal_without_retry(self):
        plan = dict(code=str(queue.ROOT), run_root=str(self.root), sources={}, jobs=[
            dict(id='partial', output=str(self.root / 'partial'), argv=['unused'], dependencies=[],
                 gpu=False, preflight=None, max_seconds=3600)])
        submitted = []

        def interrupted_launch(argv, **kwargs):
            submitted.append(argv)
            Path(argv[argv.index('--record-dir') + 1]).mkdir()
            return SimpleNamespace(returncode=1)

        with patch.object(queue, 'active_gpu_claims', return_value=[]), \
             patch.object(queue.subprocess, 'run', side_effect=interrupted_launch):
            self.assertEqual(queue.run_queue(plan, self.root / 'queue', ['GPU-one']), 1)
        self.assertEqual(len(submitted), 1)
        self.assertEqual(queue.read(self.root / 'queue/run.json')['jobs']['partial']['status'], 'FAILED')

    def test_proven_own_pre_unit_failure_does_not_starve_independent_gpu_job(self):
        plan = dict(code=str(queue.ROOT), run_root=str(self.root), sources={}, jobs=[
            dict(id=name, output=str(self.root / name), argv=['unused'], dependencies=[],
                 gpu=True, preflight=['preflight'] if name == 'independent' else None, max_seconds=3600)
            for name in ('partial', 'independent')])
        submitted = []
        original = queue.recorded_state

        def launch(argv, **kwargs):
            if '--record-dir' not in argv:
                return SimpleNamespace(returncode=0)
            record = Path(argv[argv.index('--record-dir') + 1])
            name = record.name.removesuffix('-launch')
            submitted.append(name)
            if name == 'partial':
                record.mkdir()
                return SimpleNamespace(returncode=1)
            self.record(name)
            return SimpleNamespace(returncode=0)

        original_claims = queue.active_gpu_claims
        with patch.object(queue, 'recorded_state', side_effect=lambda path, **kw:
                          original(path, **kw, inactive=lambda _: True)), \
             patch.object(queue, 'active_gpu_claims', side_effect=lambda path:
                          original_claims(path, inactive=lambda _: True)), \
             patch.object(queue, 'available_gpus', return_value=['GPU-one']), \
             patch.object(queue.subprocess, 'run', side_effect=launch), patch.object(queue.time, 'sleep'):
            self.assertEqual(queue.run_queue(plan, self.root / 'queue', ['GPU-one']), 1)
        self.assertEqual(submitted, ['partial', 'independent'])
        state = queue.read(self.root / 'queue/run.json')
        self.assertTrue(state['jobs']['partial']['pre_unit_failure_confirmed'])
        self.assertEqual(state['jobs']['independent']['status'], 'COMPLETED')
        # An unproven historical partial record is never released by this rule.
        self.assertTrue(original_claims(self.root, inactive=lambda _: True))
        (self.root / 'partial-launch/launch.json').write_text('{}')
        with patch.object(queue, 'active_gpu_claims', return_value=[str(self.root / 'partial-launch')]):
            self.assertEqual(queue.queue_gpu_claims(plan, state), [str(self.root / 'partial-launch')])

    def test_dispatch_never_exceeds_four_claims_and_keeps_cpu_phase_devices_reserved(self):
        devices = ['GPU-one', 'GPU-two', 'GPU-three', 'GPU-four']
        for index, gpu in enumerate(devices[:3]):
            self.record('old-' + str(index), state='RUNNING', code=None, gpu=gpu)
        plan = dict(code=str(queue.ROOT), run_root=str(self.root), sources={}, jobs=[
            dict(id='new-' + str(index), output=str(self.root / ('new-' + str(index))),
                 argv=['unused'], dependencies=[], gpu=True, preflight=None, max_seconds=3600)
            for index in range(5)])
        phase, submitted, maximum = [0], [], [0]
        original_state, original_claims = queue.recorded_state, queue.active_gpu_claims
        inactive = lambda _: phase[0] > 0

        def launch(argv, **kwargs):
            record = Path(argv[argv.index('--record-dir') + 1])
            gpu = next(value.split('=', 1)[1] for value in argv if value.startswith('CUDA_VISIBLE_DEVICES='))
            claims = original_claims(self.root, inactive=inactive)
            maximum[0] = max(maximum[0], len(claims) + 1)
            self.assertNotIn(gpu, queue.reserved_gpus(claims, devices))
            submitted.append((record.name, gpu, phase[0]))
            self.record(record.name.removesuffix('-launch'), gpu=gpu)
            return SimpleNamespace(returncode=0)

        with patch.object(queue, 'recorded_state', side_effect=lambda path, **kw:
                          original_state(path, **kw, inactive=inactive)), \
             patch.object(queue, 'active_gpu_claims', side_effect=lambda path:
                          original_claims(path, inactive=inactive)), \
             patch.object(queue, 'available_gpus', return_value=devices), \
             patch.object(queue.subprocess, 'run', side_effect=launch), \
             patch.object(queue.time, 'sleep', side_effect=lambda _: phase.__setitem__(0, phase[0] + 1)):
            self.assertEqual(queue.run_queue(plan, self.root / 'queue', devices), 0)
        self.assertEqual(len(submitted), 5)
        self.assertEqual(submitted[0][1:], ('GPU-four', 0))
        self.assertTrue(all(item[2] > 0 for item in submitted[1:]))
        self.assertLessEqual(maximum[0], 4)


if __name__ == '__main__':
    unittest.main()
