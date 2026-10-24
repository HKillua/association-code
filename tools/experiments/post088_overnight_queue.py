"""Finite 096/097 queue; launch once under the existing systemd wrapper.

Only the explicitly listed experiments are dispatched. No training retries,
SSH, repository writes, permanent supervisor or automatic new experiments.
The server must give this queue exclusive dispatch ownership of these tasks.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
TERMINAL = {'COMPLETED', 'FAILED', 'INTERRUPTED', 'TIMED_OUT'}
DONE = {'COMPLETED', 'FAILED', 'BLOCKED'}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, data):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def utc():
    return datetime.now(timezone.utc).isoformat()


def service_inactive(launch):
    unit = launch.get('unit')
    if launch.get('backend') != 'systemd' or not isinstance(unit, str) or not unit.endswith('.service'):
        raise ValueError('a recorded systemd user service is required; no session fallback')
    result = subprocess.run(['systemctl', '--user', 'show', unit,
                             '--property=LoadState', '--property=ActiveState'],
                            capture_output=True, text=True, timeout=10, check=False)
    values = dict(line.split('=', 1) for line in result.stdout.splitlines() if '=' in line)
    # Successful transient units may already have been collected. Require
    # affirmative unit state, so an unavailable user bus never frees a claim.
    if values.get('LoadState') == 'not-found' and values.get('ActiveState') == 'inactive':
        return True
    if result.returncode != 0:
        raise RuntimeError('cannot establish service state: ' + unit)
    if 'LoadState' not in values or 'ActiveState' not in values:
        raise RuntimeError('incomplete service state: ' + unit)
    return values['ActiveState'] in ('inactive', 'failed')


def launcher_alive(launch):
    """An unfinished submission may still be between recording and systemd-run."""
    pid = launch.get('launcher_pid')
    if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def submission_failure_details(output):
    record = Path(str(output) + '-launch')
    if (record / 'launch.json').exists() and not (record / 'status.json').exists():
        launch = read(record / 'launch.json')
        if launch.get('status') in ('SUBMITTING', 'SUBMISSION_UNKNOWN', 'FAILED') and launch.get('unit'):
            request = record / 'request.json'
            gpu = not request.exists() or bool(read(request).get('environment', {}).get('CUDA_VISIBLE_DEVICES'))
            return dict(reason='ambiguous interrupted submission; ownership retained for server inspection',
                        gpu_claim_preserved=gpu)
    return {}


def recorded_state(output, *, require_verification=False, inactive=service_inactive,
                   submitting_alive=launcher_alive):
    """Wait for the launcher cleanup as well as a successful scientific wrapper."""
    output = Path(output)
    record = Path(str(output) + '-launch')
    if not record.exists():
        return 'UNTRACKED' if output.exists() else 'NOT_STARTED'
    if not (record / 'launch.json').exists():
        return 'WAITING'
    launch = read(record / 'launch.json')
    if launch.get('status') in ('FAILED', 'SUBMISSION_UNKNOWN'):
        # A systemd-run timeout can happen after the unit actually started.
        # Observe that same attempt; never retry or abandon a live payload.
        if not launch.get('unit'):
            return 'FAILED'
        if not inactive(launch):
            return 'RUNNING'
        if not (record / 'status.json').exists():
            return 'FAILED'
    if not (record / 'status.json').exists():
        if launch.get('status') == 'SUBMITTED' and inactive(launch):
            return 'FAILED'
        if (launch.get('status') == 'SUBMITTING' and inactive(launch)
                and not submitting_alive(launch)):
            return 'FAILED'
        return 'WAITING'
    status = read(record / 'status.json')
    if status.get('status') not in TERMINAL:
        # A stale RUNNING file is not proof of a running service or a free slot.
        return 'FAILED' if inactive(launch) else 'RUNNING'
    if not inactive(launch):
        return 'RUNNING'
    if status.get('status') != 'COMPLETED' or status.get('returncode') != 0:
        return 'FAILED'
    run_path = output / 'run.json'
    if not run_path.exists() or read(run_path).get('status') != 'COMPLETED':
        return 'FAILED'
    if require_verification:
        verification = output / 'completion-value-verification.json'
        if not verification.exists() or read(verification).get('status') != 'COMPLETED':
            return 'FAILED'
    return 'COMPLETED'


def active_gpu_claims(run_root, inactive=service_inactive):
    """Count all this repository's recorded GPU launches, not just our children."""
    active = []
    for record in sorted(Path(run_root).glob('*-launch')):
        request_path = record / 'request.json'
        if not request_path.exists():
            # Missing ownership metadata cannot make a live/unknown job free.
            launch_path = record / 'launch.json'
            if launch_path.exists():
                launch = read(launch_path)
                if (launch.get('unit') and inactive(launch)
                        and (launch.get('status') == 'SUBMITTED' or (record / 'status.json').exists())):
                    continue
                if not launch.get('unit') and launch.get('status') in ('FAILED', 'SUBMISSION_UNKNOWN'):
                    continue
            active.append(str(record))
            continue
        request = read(request_path)
        if not request.get('environment', {}).get('CUDA_VISIBLE_DEVICES', ''):
            continue
        launch_path, status_path = record / 'launch.json', record / 'status.json'
        if not launch_path.exists():
            active.append(str(record))
            continue
        launch = read(launch_path)
        if launch.get('status') in ('FAILED', 'SUBMISSION_UNKNOWN'):
            if not launch.get('unit'):
                continue
            # An interrupted/unknown systemd-run might still activate later.
            # Without worker evidence, retain ownership for server inspection.
            if not status_path.exists():
                active.append(str(record))
                continue
        if launch.get('status') == 'SUBMITTED' and launch.get('unit') and inactive(launch):
            continue
        if status_path.exists() and inactive(launch):
            continue
        active.append(str(record))
    return active


def reserved_gpus(claims, allowed):
    """A CPU preparation/evaluation phase still owns its claimed GPU slot."""
    reserved = set()
    for record in claims:
        request = Path(record) / 'request.json'
        if not request.exists():
            return set(allowed)  # Partially submitted ownership is not free.
        visible = read(request).get('environment', {}).get('CUDA_VISIBLE_DEVICES', '')
        for item in filter(None, (part.strip() for part in visible.split(','))):
            if not item.startswith('GPU-'):
                return set(allowed)  # Legacy numeric/MIG mapping is ambiguous.
            reserved.add(item)
    return reserved


def available_gpus(allowed, *, run=subprocess.run):
    """Conservative idle-only check; never displace another lab process."""
    query = run(['nvidia-smi', '--query-gpu=uuid,memory.used,utilization.gpu', '--format=csv,noheader,nounits'],
                capture_output=True, text=True, timeout=10, check=True)
    compute = run(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader,nounits'],
                  capture_output=True, text=True, timeout=10, check=True)
    occupied = {line.split(',')[0].strip() for line in compute.stdout.splitlines() if line.strip()}
    values = {}
    for line in query.stdout.splitlines():
        parts = [part.strip() for part in line.split(',')]
        if len(parts) != 3:
            raise ValueError('unexpected nvidia-smi GPU row')
        values[parts[0]] = (int(parts[1]), int(parts[2]))
    if any(uuid not in values for uuid in allowed):
        raise ValueError('an allowed GPU UUID is missing from nvidia-smi')
    return [uuid for uuid in allowed if uuid not in occupied
            and values[uuid][0] <= 256 and values[uuid][1] == 0]


def build_plan(run_root, archive, *, python=sys.executable, code=ROOT):
    run_root, code = Path(run_root).resolve(), Path(code).resolve()
    archive = [str(Path(item).resolve()) for item in archive]
    base = ['--manifest', str(code / 'configs/data/mtad-level1-scene3000.json'),
            '--alignment', str(code / 'configs/data/mtad-level1-alignment.json')]
    sources = {}
    for phase, number, kind in (('tfm', '093', 'TFM'), ('tam', '094', 'TAM')):
        for letter in ('A', 'B'):
            sources[number + letter] = run_root / f'LEVEL1-COMPLETION-VALUE-{kind}-20261024-{number}{letter}-attempt-1'
    jobs = []

    def add(name, output, command, dependencies, *, gpu=False, preflight=None, hours=1):
        jobs.append(dict(id=name, output=str(output), argv=[str(python), *map(str, command)],
                         dependencies=list(dependencies), gpu=gpu,
                         preflight=None if preflight is None else [str(python), *map(str, preflight)],
                         max_seconds=hours * 3600))

    for letter, policy in (('A', 'learned'), ('B', 'state_hold')):
        output = run_root / f'LEVEL1-COMPLETION-SYSTEM-20261025-096{letter}-attempt-1'
        common = [code / 'tools/experiments/completion_system_evaluate.py',
                  '--tam-run', sources['094' + letter], '--tfm-run', sources['093' + letter],
                  '--completion-policy', policy, *base, '--cpu-threads', '4']
        add('096' + letter, output,
            [*common, '--archive', *archive, '--output', output, '--device', 'cuda:0',
             '--max-seconds', '28800', '--time-policy', 'advisory'],
            ['093' + letter, '094' + letter], gpu=True,
            preflight=[*common, '--check-source-only'], hours=8)
    budget_output = run_root / 'LEVEL1-SEQUENCE-BUDGET-20261025-097A-attempt-1'
    common = [code / 'tools/experiments/sequence_budget_replication.py',
              '--source-system', run_root / 'LEVEL1-ASSOCIATION-20261022-056-attempt-1/system',
              '--paired-control', run_root / 'LEVEL1-SEQUENCE-TRANSFORMER-20261023-073C-attempt-1',
              '--config', code / 'configs/sequence-budget-replication-level1.json',
              *base, '--cpu-threads', '4', '--time-policy', 'advisory']
    add('097A', budget_output, [*common, '--archive', *archive, '--output', budget_output,
                               '--device', 'cuda:0'], [], gpu=True,
        preflight=[*common, '--check-source-only'], hours=4)
    # These are the original fixed CPU tables, not repetitions or new forwards.
    for phase, number, kind in (('tfm', '093', 'TFM'), ('tam', '094', 'TAM')):
        output = run_root / f'LEVEL1-COMPLETION-VALUE-{kind}-20261024-{number}-summary-attempt-1'
        add(number + '-summary', output,
            [code / 'tools/analysis/compare_completion_policies.py', '--phase', phase,
             '--learned-run', sources[number + 'A'], '--hold-run', sources[number + 'B'],
             '--output', output, '--cpu-threads', '4'], [number + 'A', number + 'B'])
    output = run_root / 'LEVEL1-COMPLETION-SYSTEM-20261025-096-summary-attempt-1'
    add('096-summary', output, [code / 'tools/analysis/compare_completion_systems.py',
        '--learned-run', jobs[0]['output'], '--hold-run', jobs[1]['output'],
        '--output', output, '--cpu-threads', '4'], ['096A', '096B'])
    output = run_root / 'LEVEL1-SEQUENCE-BUDGET-20261025-097-summary-attempt-1'
    add('097-summary', output, [code / 'tools/analysis/compare_sequence_budget_seeds.py',
        '--seed23-run', run_root / 'LEVEL1-FROZEN-ADAPTER-20261024-090C-attempt-1',
        '--seed21-run', budget_output, '--output', output, '--cpu-threads', '4'], ['097A'])
    return dict(format='post088_finite_queue_v1', code=str(code), run_root=str(run_root),
                max_gpu_claims=4, max_cpu_comparisons=1, sources={key: str(value) for key, value in sources.items()},
                jobs=jobs, retry_count=0, test_opened=False)


def dependency_decision(dependencies, states):
    if any(states.get(key) in ('FAILED', 'BLOCKED', 'UNTRACKED') for key in dependencies):
        return 'BLOCKED'
    return 'READY' if all(states.get(key) == 'COMPLETED' for key in dependencies) else 'WAITING'


def queue_gpu_claims(plan, state):
    # Only this controller's completed launcher calls can prove a pre-unit
    # failure. Unknown historical partial records stay conservatively reserved.
    proven = {job['output'] + '-launch' for job in plan['jobs']
              if state['jobs'].get(job['id'], {}).get('pre_unit_failure_confirmed')
              and not (Path(job['output'] + '-launch') / 'launch.json').exists()}
    return [record for record in active_gpu_claims(plan['run_root']) if record not in proven]


def run_queue(plan, output, allowed, *, poll_seconds=60):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    write(output / 'plan.json', plan)
    started = time.monotonic()
    state = dict(status='RUNNING', started_at=utc(), jobs={}, sources={},
                 max_gpu_claims=4, optimizer_updates_by_queue=0, retries=0, test_opened=False)
    lock = (Path(plan['run_root']) / '.post088-night-queue.lock').open('a+')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while True:
            for key, source in plan['sources'].items():
                state['sources'][key] = recorded_state(source, require_verification=True)
            for job in plan['jobs']:
                existing = recorded_state(job['output'])
                previous = state['jobs'].get(job['id'], {})
                if previous.get('status') in DONE:
                    continue
                if existing == 'UNTRACKED':
                    state['jobs'][job['id']] = dict(status='BLOCKED', reason='output exists without a launch record; never overwrite')
                elif existing != 'NOT_STARTED':
                    state['jobs'][job['id']] = dict(previous, status=existing, existing_attempt_reused=True)
                    if existing == 'FAILED':
                        state['jobs'][job['id']].update(submission_failure_details(job['output']))
                else:
                    state['jobs'].setdefault(job['id'], dict(status='WAITING', attempts=0))
            statuses = {**state['sources'], **{key: value['status'] for key, value in state['jobs'].items()}}
            gpu_claims = queue_gpu_claims(plan, state)
            state['active_gpu_claims'] = gpu_claims
            cpu_busy = any(not job['gpu'] and state['jobs'][job['id']]['status'] not in DONE
                           and Path(job['output'] + '-launch').exists() for job in plan['jobs'])
            used_gpus = set()
            for job in plan['jobs']:
                current = state['jobs'][job['id']]
                if current['status'] not in ('WAITING', 'NOT_STARTED'):
                    continue
                # Existing submitted launch folders are always observed, never submitted twice.
                if Path(job['output'] + '-launch').exists():
                    continue
                readiness = dependency_decision(job['dependencies'], statuses)
                if readiness == 'BLOCKED':
                    current.update(status='BLOCKED', reason='a required source failed or is untracked')
                    continue
                if readiness != 'READY' or (not job['gpu'] and cpu_busy) or (job['gpu'] and len(gpu_claims) >= 4):
                    continue
                gpu = ''
                if job['gpu']:
                    reserved = reserved_gpus(gpu_claims, allowed)
                    free = [uuid for uuid in available_gpus(allowed)
                            if uuid not in used_gpus and uuid not in reserved]
                    if not free:
                        continue
                    gpu = free[0]
                if job['preflight'] is not None:
                    with (output / (job['id'] + '-preflight.log')).open('a') as log:
                        result = subprocess.run(job['preflight'], cwd=plan['code'], stdin=subprocess.DEVNULL,
                            stdout=log, stderr=subprocess.STDOUT,
                            env={**os.environ, 'CUDA_VISIBLE_DEVICES': '', 'OMP_NUM_THREADS': '4',
                                 'MKL_NUM_THREADS': '4', 'OPENBLAS_NUM_THREADS': '4'}, check=False)
                    if result.returncode == 75:
                        current.update(status='WAITING', reason='preflight dependency pending; no attempt consumed')
                        continue
                    if result.returncode != 0:
                        current.update(status='FAILED', reason='source preflight failed', exit_code=result.returncode)
                        continue
                    # Preflight can take minutes; recheck actual resources immediately before submit.
                    gpu_claims = queue_gpu_claims(plan, state)
                    if (len(gpu_claims) >= 4 or gpu not in available_gpus(allowed)
                            or gpu in reserved_gpus(gpu_claims, allowed)):
                        continue
                record = job['output'] + '-launch'
                launch_argv = [sys.executable, str(Path(plan['code']) / 'tools/experiments/launch_detached.py'),
                    '--cwd', plan['code'], '--record-dir', record, '--time-policy', 'advisory',
                    '--max-seconds', str(job['max_seconds']), '--grace-seconds', '30',
                    '--env', 'CUDA_VISIBLE_DEVICES=' + gpu, '--env', 'OMP_NUM_THREADS=4',
                    '--env', 'MKL_NUM_THREADS=4', '--env', 'OPENBLAS_NUM_THREADS=4', '--', *job['argv']]
                current.update(status='SUBMITTING', attempts=1, gpu_uuid=gpu or None, submitted_at=utc(),
                               argv=job['argv'], launch_record=record)
                write(output / 'run.json', state)
                with (output / (job['id'] + '-submission.log')).open('x') as log:
                    try:
                        result = subprocess.run(launch_argv, cwd=plan['code'], stdin=subprocess.DEVNULL,
                                                stdout=log, stderr=subprocess.STDOUT, timeout=60, check=False)
                        submission_code = result.returncode
                    except subprocess.TimeoutExpired:
                        submission_code = None
                observed = 'SUBMITTED' if submission_code == 0 else recorded_state(job['output'])
                if (submission_code != 0 and not (Path(record) / 'launch.json').exists()):
                    # launch_detached writes its unit before invoking systemd-run.
                    # The launcher has now returned (or was killed by our timeout),
                    # so no unit-less partial record can acquire ownership later.
                    observed = 'FAILED'
                    current.update(pre_unit_failure_confirmed=True,
                                   reason='own launcher ended before recording a unit; no retry')
                if observed in ('NOT_STARTED', 'UNTRACKED'):
                    observed = 'FAILED'
                current.update(status=observed, submission_exit_code=submission_code)
                if observed == 'FAILED':
                    current.update(submission_failure_details(job['output']))
                if observed not in DONE:
                    if job['gpu']:
                        gpu_claims.append(record)
                        used_gpus.add(gpu)
                    else:
                        cpu_busy = True
                state['updated_at'] = utc()
                write(output / 'run.json', state)
            if all(row['status'] in DONE for row in state['jobs'].values()):
                state['status'] = ('COMPLETED' if all(row['status'] == 'COMPLETED' for row in state['jobs'].values())
                                   else 'COMPLETED_WITH_FAILURES')
                state.update(ended_at=utc(), elapsed_seconds=time.monotonic() - started)
                write(output / 'run.json', state)
                return 0 if state['status'] == 'COMPLETED' else 1
            state.update(updated_at=utc(), elapsed_seconds=time.monotonic() - started)
            write(output / 'run.json', state)
            time.sleep(poll_seconds)
    except BaseException as error:
        state.update(status='FAILED', ended_at=utc(), failure=f'{type(error).__name__}: {error}',
                     child_services_preserved=True, elapsed_seconds=time.monotonic() - started)
        write(output / 'run.json', state)
        raise
    finally:
        lock.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--archive', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--gpu-uuids', nargs='+', default=[])
    parser.add_argument('--print-plan', action='store_true')
    args = parser.parse_args(argv)
    plan = build_plan(args.run_root, args.archive)
    if args.print_plan:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return 0
    if args.output is None or not args.gpu_uuids or len(set(args.gpu_uuids)) != len(args.gpu_uuids):
        parser.error('execution requires a new --output and distinct, server-approved --gpu-uuids')
    if not args.run_root.is_dir() or any(not path.is_file() for path in args.archive):
        parser.error('the existing run root and all original archive parts must exist')
    return run_queue(plan, args.output, args.gpu_uuids)


if __name__ == '__main__':
    raise SystemExit(main())
