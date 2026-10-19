"""Launch one bounded job, normally as a detached systemd user service.

Example (use the separately approved budget, absolute Python and a NEW record):
  python tools/experiments/launch_detached.py --cwd /path/to/checkout \
    --record-dir /path/to/new-launch --max-seconds 3600 --grace-seconds 10 \
    --env CUDA_VISIBLE_DEVICES=GPU-... -- /path/to/venv/bin/python train.py ...

systemd requires an available user manager and Linger=yes; no silent fallback.
--backend session only survives parent exit/SIGHUP, NOT runtime tree/cgroup
reaping. A server-side CPU disconnect check must establish actual persistence.
The worker's absolute deadline includes launch/preparation/cleanup/result I/O;
an expired worker never starts its payload. systemd's RuntimeMaxSec is a
relative activation-time backup, so submission delays shift that backup only.
OS stalls or SIGKILL can prevent final records; this is not a real-time guarantee.
No shell interpolation, retries, resume, or permanently running supervisor.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import uuid

FINALIZE_SECONDS = 1.0


def _utc():
    return datetime.now(timezone.utc).isoformat()


def _write(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    temporary.replace(path)


def _signal_group(pid, sig):
    try:
        os.killpg(pid, sig)
        return True
    except ProcessLookupError:
        return False


def _worker(record):
    request = json.loads((record/'request.json').read_text(encoding='utf-8'))
    # Exclusive ownership also prevents accidentally invoking the internal worker twice.
    with (record/'worker.pid').open('x') as stream:
        stream.write(str(os.getpid())+'\n')
    if os.getsid(0) != os.getpid():
        os.setsid()
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    def stopped(signum, frame):
        raise InterruptedError(f'supervisor received signal {signum}')
    signal.signal(signal.SIGTERM, stopped)
    deadline = request['deadline_monotonic']
    terminate_at = deadline - request['grace_seconds'] - FINALIZE_SECONDS
    process, timed_out, interrupted, failure = None, False, False, None
    result = dict(status='RUNNING', worker_pid=os.getpid(), payload_pid=None,
                  payload_pgid=None, returncode=None, timed_out=False,
                  worker_started_at=_utc(), backend=request['backend'])
    _write(record/'status.json', result)
    try:
        if time.monotonic() >= terminate_at:
            raise subprocess.TimeoutExpired(request['argv'], 0)
        with (record/'stdout.log').open('xb') as stdout, (record/'stderr.log').open('xb') as stderr:
            process = subprocess.Popen(request['argv'], cwd=request['cwd'],
                env={**os.environ, **request['environment']}, stdin=subprocess.DEVNULL,
                stdout=stdout, stderr=stderr, start_new_session=True, close_fds=True)
            result.update(payload_pid=process.pid, payload_pgid=process.pid, payload_started_at=_utc())
            _write(record/'status.json', result)
            process.wait(timeout=max(0., terminate_at-time.monotonic()))
    except subprocess.TimeoutExpired:
        timed_out = True
    except (Exception, KeyboardInterrupt) as error:
        interrupted = isinstance(error, (InterruptedError, KeyboardInterrupt))
        failure = f'{type(error).__name__}: {error}'
    finally:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        # Kill the GROUP even when its leader already exited: a grandchild may
        # still be running. systemd additionally cleans escaped process groups
        # in this service's cgroup; session mode cannot offer that guarantee.
        if process is not None:
            group_remained = _signal_group(process.pid, signal.SIGTERM)
            result['group_cleanup_requested'] = group_remained
            kill_at = min(time.monotonic()+request['grace_seconds'], deadline-FINALIZE_SECONDS)
            # The leader can exit before a cooperative child finishes cleanup.
            # This bounded wait concerns only our own process group.
            while time.monotonic() < kill_at:
                process.poll()  # Reap the leader without shortening child grace.
                if not _signal_group(process.pid, 0):
                    break
                time.sleep(min(.025, max(0., kill_at-time.monotonic())))
            _signal_group(process.pid, signal.SIGKILL)
            try:
                process.wait(timeout=max(.001, deadline-time.monotonic()))
            except subprocess.TimeoutExpired:
                failure = 'payload was sent SIGKILL but no exit status was available before the deadline'
            result['returncode'] = process.returncode
        status = ('TIMED_OUT' if timed_out else 'INTERRUPTED' if interrupted else
                  'COMPLETED' if failure is None and result['returncode'] == 0 else 'FAILED')
        result.update(status=status, timed_out=timed_out, failure=failure, ended_at=_utc(),
                      elapsed_seconds=time.monotonic()-request['started_monotonic'],
                      deadline_exceeded=time.monotonic() > deadline)
        (record/'exit_code.txt').write_text(
            'UNAVAILABLE\n' if result['returncode'] is None else str(result['returncode'])+'\n')
        _write(record/'status.json', result)
    if timed_out:
        return 124
    if interrupted:
        return 130
    code = result['returncode']
    return (code if code >= 0 else 128-code) if code is not None and failure is None else 1


def main(argv=None):
    args = sys.argv[1:] if argv is None else argv
    if len(args) == 2 and args[0] == '--_worker':
        return _worker(Path(args[1]))
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--record-dir', required=True)
    parser.add_argument('--cwd', required=True)
    parser.add_argument('--max-seconds', type=float, required=True)
    parser.add_argument('--grace-seconds', type=float, default=5.)
    parser.add_argument('--backend', choices=('systemd', 'session'), default='systemd')
    parser.add_argument('--env', action='append', default=[], metavar='NAME=VALUE',
                        help='explicit payload environment; systemd does not inherit the launching shell')
    parser.add_argument('command', nargs=argparse.REMAINDER)
    options = parser.parse_args(args)
    started = time.monotonic()
    record, owned = Path(options.record_dir).resolve(), False
    launch = dict(status='SUBMITTING', backend=options.backend, launcher_pid=os.getpid())
    try:
        command = options.command[1:] if options.command[:1] == ['--'] else options.command
        if not command or not Path(command[0]).is_absolute():
            raise ValueError('provide -- followed by an argv with an absolute executable path')
        if (not math.isfinite(options.max_seconds) or not math.isfinite(options.grace_seconds)
                or options.grace_seconds <= 0 or options.max_seconds <= options.grace_seconds+FINALIZE_SECONDS):
            raise ValueError('budget must exceed positive grace plus one second reserved for final records')
        cwd = Path(options.cwd).resolve(strict=True)
        if not cwd.is_dir():
            raise ValueError('--cwd must be a directory')
        environment = {}
        for assignment in options.env:
            key, separator, value = assignment.partition('=')
            if not separator or not key or '\x00' in assignment:
                raise ValueError('--env requires NAME=VALUE')
            environment[key] = value
        record.mkdir(parents=True, exist_ok=False, mode=0o700)
        owned = True
        request = dict(argv=command, cwd=str(cwd), environment=environment,
            backend=options.backend, started_at=_utc(), started_monotonic=started,
            max_seconds=options.max_seconds, grace_seconds=options.grace_seconds,
            finalize_seconds=FINALIZE_SECONDS, deadline_monotonic=started+options.max_seconds)
        _write(record/'request.json', request)
        worker = [os.path.abspath(sys.executable), str(Path(__file__).resolve()), '--_worker', str(record)]
        if options.backend == 'systemd':
            # Do not enable persistent host configuration or retry implicitly.
            linger = subprocess.run(['loginctl', 'show-user', str(os.getuid()), '--property=Linger', '--value'],
                stdin=subprocess.DEVNULL, capture_output=True, text=True, check=True,
                timeout=min(5., options.max_seconds-options.grace_seconds-FINALIZE_SECONDS))
            launch['linger'] = linger.stdout.strip()
            if launch['linger'] != 'yes':
                raise RuntimeError('Linger is not yes: user services may stop at logout; configure the user manager before launching')
            remaining = request['deadline_monotonic']-time.monotonic()
            if remaining <= options.grace_seconds+FINALIZE_SECONDS:
                raise TimeoutError('launch exhausted the total budget')
            unit = 'mtad-'+uuid.uuid4().hex[:12]+'.service'
            launch['unit'] = unit
            properties = ['Type=exec', 'Restart=no', 'KillMode=control-group', 'SendSIGKILL=yes',
                f'RuntimeMaxSec={remaining-options.grace_seconds:.6f}s',
                f'TimeoutStopSec={options.grace_seconds:.6f}s', 'StandardInput=null',
                'WorkingDirectory='+str(cwd).replace('%', '%%'),
                'StandardOutput=append:'+str(record/'worker.stdout.log').replace('%', '%%'),
                'StandardError=append:'+str(record/'worker.stderr.log').replace('%', '%%')]
            # Payload argv is transported as JSON, never through systemd's
            # environment expansion. Escape $ only in the small worker argv.
            invocation = ['systemd-run', '--user', '--no-ask-password', '--unit='+unit,
                *['--property='+value for value in properties], '--',
                *[value.replace('$', '$$') for value in worker]]
            launch['systemd_argv'] = invocation
            _write(record/'launch.json', launch)
            with (record/'launch.stdout.log').open('xb') as stdout, (record/'launch.stderr.log').open('xb') as stderr:
                submitted = subprocess.run(invocation, stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr,
                    timeout=remaining-options.grace_seconds-FINALIZE_SECONDS, check=True)
            launch['submission_returncode'] = submitted.returncode
        else:
            launch['limitation'] = 'session detachment does not escape AI runtime process-tree or cgroup reclamation'
            print('WARNING: '+launch['limitation'], file=sys.stderr, flush=True)
            with (record/'worker.stdout.log').open('xb') as stdout, (record/'worker.stderr.log').open('xb') as stderr:
                worker_process = subprocess.Popen(worker, stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr,
                                                  start_new_session=True, close_fds=True, cwd=cwd)
            launch['worker_pid'] = worker_process.pid
        launch.update(status='SUBMITTED', submitted_at=_utc(), elapsed_seconds=time.monotonic()-started)
        _write(record/'launch.json', launch)
        print(f'SUBMITTED {record}; status.json is the worker result, not this submission exit code', flush=True)
        return 0
    except (Exception, KeyboardInterrupt) as error:
        if owned:
            launch.update(status='SUBMISSION_UNKNOWN' if isinstance(error, subprocess.TimeoutExpired) else 'FAILED',
                          failure=f'{type(error).__name__}: {error}', ended_at=_utc(),
                          elapsed_seconds=time.monotonic()-started)
            _write(record/'launch.json', launch)
        print(f'FAILED {type(error).__name__}: {error}', file=sys.stderr, flush=True)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
