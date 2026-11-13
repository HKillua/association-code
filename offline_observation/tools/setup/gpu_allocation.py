#!/usr/bin/env python3
"""Select a GPU from a live nvidia-smi snapshot; shared GPUs are permitted.

No reservations, GPU computation, process termination, waiting or retries are
performed. A free-memory snapshot cannot guarantee that a workload will fit.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import io
import json
import math
from pathlib import Path
import re
import subprocess
import sys


UUID_PATTERN = re.compile(r'GPU-[a-fA-F0-9]+(?:-[a-fA-F0-9]+)*')
GPU_FIELDS = 'index,uuid,name,memory.total,memory.free,utilization.gpu'
PROCESS_FIELDS = 'gpu_uuid,pid,process_name,used_gpu_memory'


class AllocationError(RuntimeError):
    def __init__(self, message, report):
        super().__init__(message)
        self.report = report


class NoSuitableGPU(AllocationError):
    """No current device meets the request and caller's memory estimate."""


def device_argument(value):
    if value == 'auto' or re.fullmatch(r'[0-9]+', value) or UUID_PATTERN.fullmatch(value):
        return value
    raise argparse.ArgumentTypeError('device must be auto, a nonnegative physical index, or a GPU UUID')


def memory_argument(value):
    try:
        number = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError('min-free-mib must be a finite nonnegative number') from exc
    if not math.isfinite(number) or number < 0:
        raise argparse.ArgumentTypeError('min-free-mib must be a finite nonnegative number')
    return number


def _text(value):
    return value.decode('utf-8', errors='replace') if isinstance(value, bytes) else (value or '')


def _query(command):
    evidence = {'command': command, 'timeout_seconds': 15}
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=15)
        evidence.update(exit_code=result.returncode, stdout=result.stdout, stderr=result.stderr,
                        timed_out=False)
    except subprocess.TimeoutExpired as exc:
        evidence.update(exit_code=None, stdout=_text(exc.stdout), stderr=_text(exc.stderr),
                        timed_out=True, error='nvidia-smi query exceeded 15 seconds')
    except OSError as exc:
        evidence.update(exit_code=None, stdout='', stderr=str(exc), timed_out=False,
                        error='Unable to execute nvidia-smi')
    return evidence


def _number(value):
    number = float(value)
    if not math.isfinite(number):
        raise ValueError('nonfinite numeric field')
    return number


def _candidates(output):
    candidates, rejected = [], []
    seen_indices, seen_uuids = set(), set()
    for row in csv.reader(io.StringIO(output)):
        if not row or not any(field.strip() for field in row):
            continue
        values = [field.strip() for field in row]
        try:
            if len(values) != 6:
                raise ValueError('expected six device fields')
            index, uuid, name, total, free, utilization = values
            if not re.fullmatch(r'[0-9]+', index) or not UUID_PATTERN.fullmatch(uuid) or not name:
                raise ValueError('invalid device index, UUID or name')
            index, total, free = int(index), _number(total), _number(free)
            if total <= 0 or free < 0 or free > total:
                raise ValueError('invalid device memory values')
            if utilization in ('N/A', '[N/A]', 'Not Supported', '[Not Supported]'):
                utilization = None
            else:
                utilization = _number(utilization)
                if not 0 <= utilization <= 100:
                    raise ValueError('invalid utilization value')
            if index in seen_indices or uuid.lower() in seen_uuids:
                raise ValueError('duplicate device index or UUID')
            seen_indices.add(index)
            seen_uuids.add(uuid.lower())
            candidates.append({'index': index, 'uuid': uuid, 'name': name,
                               'memory_total_mib': total, 'memory_free_mib': free,
                               'utilization_gpu': utilization})
        except ValueError as exc:
            rejected.append({'row': values, 'reason': str(exc)})
    return candidates, rejected


def _processes(output):
    processes, unparsed = [], []
    for row in csv.reader(io.StringIO(output)):
        if not row or not any(field.strip() for field in row):
            continue
        values = [field.strip() for field in row]
        if len(values) != 4 or not UUID_PATTERN.fullmatch(values[0]) or not values[1].isdigit():
            unparsed.append(values)
            continue
        uuid, pid, name, memory = values
        # Retain unsupported memory values as text; process evidence never vetoes a GPU.
        try:
            memory_value = _number(memory)
        except ValueError:
            memory_value = memory
        processes.append({'gpu_uuid': uuid, 'pid': int(pid), 'process_name': name,
                          'used_gpu_memory_mib': memory_value})
    return processes, unparsed


def allocate(device='auto', min_free_mib=0) -> dict:
    device = device_argument(str(device))
    min_free_mib = memory_argument(str(min_free_mib))
    report = {'checked_at_utc': datetime.now(timezone.utc).isoformat(),
              'requested_device': device, 'min_free_mib': min_free_mib,
              'memory_requirement_note': 'Caller estimate only; the snapshot is not a reservation or a workload fit measurement.',
              'shared_gpu_allowed': True, 'candidates': [], 'selected': None}
    query = _query(['nvidia-smi', '--query-gpu=' + GPU_FIELDS, '--format=csv,noheader,nounits'])
    report['device_query'] = query
    if query['exit_code'] != 0:
        report.update(status='QUERY_FAILED', selection_reason='Device query failed; no GPU selected.')
        raise AllocationError(report['selection_reason'], report)
    candidates, rejected = _candidates(query['stdout'])
    report['candidates'], report['rejected_device_rows'] = candidates, rejected
    if not candidates and rejected:
        report.update(status='INVALID_DEVICE_DATA', selection_reason=
                      'Device query returned nonempty data but no valid device rows.')
        raise AllocationError(report['selection_reason'], report)
    for candidate in candidates:
        candidate['meets_memory_estimate'] = (candidate['memory_free_mib'] > 0 and
                                              candidate['memory_free_mib'] >= min_free_mib)
    eligible = [candidate for candidate in candidates if candidate['meets_memory_estimate']]
    if device != 'auto':
        eligible = [candidate for candidate in eligible if
                    (candidate['index'] == int(device) if device.isdigit() else
                     candidate['uuid'].lower() == device.lower())]
    if not eligible:
        report.update(status='NO_SUITABLE_GPU', selection_reason=
                      'No valid requested GPU has positive free memory meeting min_free_mib.')
        raise NoSuitableGPU(report['selection_reason'], report)
    selected = min(eligible, key=lambda gpu: (-gpu['memory_free_mib'],
                    gpu['utilization_gpu'] if gpu['utilization_gpu'] is not None else math.inf,
                    gpu['index']))
    report['selected'] = dict(selected)
    report['selection_reason'] = (
        'Auto: highest free memory, then lowest known utilization, then lowest physical index; sharing is allowed.'
        if device == 'auto' else 'Explicit device matched the live device query and memory estimate; sharing is allowed.')
    process_query = _query(['nvidia-smi', '-i', selected['uuid'],
                            '--query-compute-apps=' + PROCESS_FIELDS, '--format=csv,noheader,nounits'])
    if process_query['exit_code'] == 0:
        process_query['processes'], process_query['unparsed_rows'] = _processes(process_query['stdout'])
    else:
        process_query.update(processes=None, unparsed_rows=[])
    report.update(status='SELECTED', process_query=process_query)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', type=device_argument, default='auto')
    parser.add_argument('--min-free-mib', type=memory_argument, default=0,
                        help='Caller-estimated minimum free MiB; no workload measurement is implied')
    parser.add_argument('--output', type=Path, required=True, help='New JSON evidence file; never overwritten')
    args = parser.parse_args(argv)
    try:
        # Claim the output before querying to reject accidental reuse immediately.
        with args.output.open('x', encoding='utf-8') as stream:
            exit_code = 0
            try:
                report = allocate(args.device, args.min_free_mib)
            except NoSuitableGPU as exc:
                report, exit_code = exc.report, 3
            except AllocationError as exc:
                report, exit_code = exc.report, 1
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.write('\n')
        if exit_code == 0:
            print(report['selected']['uuid'])
        else:
            print(report['selection_reason'], file=sys.stderr)
        return exit_code
    except (OSError, ValueError) as exc:
        print(f'GPU allocation failed: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
