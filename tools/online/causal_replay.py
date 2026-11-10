"""Replay ideal measurement arrivals on a fixed 1 Hz clock."""
from __future__ import annotations

import argparse
import csv
from itertools import groupby
import json
import math
from pathlib import Path
from time import perf_counter_ns
from zipfile import BadZipFile

from association.data.archive import MTADArchive
from association.data.legacy_dataset import validate_archive
from association.online.replay import csv_rows
from association.online.stream import StreamState


_LEVEL1_MANIFEST = Path(__file__).resolve().parents[2] / 'configs/data/mtad-level1-scene3000.json'
_DEVELOPMENT_SPLITS = ('T_core', 'T_head', 'V_select')


def _read_archive_rows(parts, manifest_path, scene):
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    official = json.loads(_LEVEL1_MANIFEST.read_text(encoding='utf-8'))
    for roster in (manifest, official):
        if (roster.get('dataset_level') != 'Level1'
                or not any(scene in roster['splits'][split] for split in _DEVELOPMENT_SPLITS)):
            raise ValueError('scene must be selected for development in the frozen Level1 roster')
    with MTADArchive(parts) as archive:
        validate_archive(archive, manifest)
        return archive._rows(scene, 'measurements')


def _schedule(rows, start, stop):
    selected = []
    for index, row in enumerate(rows, start=1):
        try:
            stamp = float(row['time'])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f'measurement row {index} has no schedulable time') from exc
        if not math.isfinite(stamp):
            raise ValueError(f'measurement row {index} has no schedulable time')
        if start <= stamp <= stop:
            selected.append((stamp, row))
    selected.sort(key=lambda entry: entry[0])
    return selected


def _json(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(',', ':'))


def _write_ticks(out, state, schedule, start, stop, *, clock_ns=perf_counter_ns):
    pending = iter(schedule)
    next_row = next(pending, None)
    previous_ready = 0.
    for tick in range(start, stop + 1):
        started = clock_ns()
        received = []
        while next_row is not None and next_row[0] <= tick:
            received.append(next_row)
            next_row = next(pending, None)
        for arrival_time, entries in groupby(received, key=lambda entry: entry[0]):
            state.ingest([row for _, row in entries], arrival_time=arrival_time)
        result = _json(state.predict(tick))
        service = (clock_ns() - started) / 1e9

        release = tick - start
        wait = max(0., previous_ready - release)
        latency = wait + service
        previous_ready = release + latency
        timing = _json(dict(service_seconds=service, modeled_queue_wait_seconds=wait,
                            modeled_result_latency_seconds=latency,
                            modeled_ready_elapsed_seconds=previous_ready))
        out.write(f'{{"record":"tick","received_rows":{len(received)},'
                  f'"result":{result},"timing":{timing}}}\n')


def main(argv=None, *, prog=None):
    parser = argparse.ArgumentParser(prog=prog, description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--measurements', type=Path,
                        help='standalone MTAD-format measurement CSV supplied for development')
    source.add_argument('--archive', nargs='+', type=Path,
                        help='Level1 ZIP or its ordered continuous byte parts')
    parser.add_argument('--manifest', type=Path, help='required for an archive source')
    parser.add_argument('--scene', type=int, help='required Level1 development scene for an archive source')
    parser.add_argument('--start', type=int, required=True, help='first replay second, inclusive')
    parser.add_argument('--stop', type=int, required=True, help='last replay second, inclusive')
    parser.add_argument('--stale-after-seconds', type=float, required=True)
    parser.add_argument('--output', type=Path, required=True, help='new JSONL file for private results')
    args = parser.parse_args(argv)
    if not -(2**53) <= args.start <= args.stop <= 2**53:
        parser.error('replay bounds must be ordered integer seconds representable by float64')
    if not math.isfinite(args.stale_after_seconds) or args.stale_after_seconds <= 0:
        parser.error('--stale-after-seconds must be finite and positive')
    if args.archive and (args.manifest is None or args.scene is None):
        parser.error('--archive requires --manifest and --scene')
    if args.measurements and (args.manifest is not None or args.scene is not None):
        parser.error('--manifest and --scene apply only to --archive')

    try:
        preparation_started = perf_counter_ns()
        if args.archive:
            rows = _read_archive_rows(args.archive, args.manifest, str(args.scene))
        else:
            rows = csv_rows(args.measurements.read_bytes())
        schedule = _schedule(rows, args.start, args.stop)
        state = StreamState(stale_after_seconds=args.stale_after_seconds)
        preparation = (perf_counter_ns() - preparation_started) / 1e9
        header = dict(record='protocol', arrival_clock='ideal_measurement_time',
                      start=args.start, stop=args.stop, cadence_seconds=1, warmup=False,
                      stale_after_seconds=args.stale_after_seconds, ticks=args.stop - args.start + 1,
                      source='level1_archive' if args.archive else 'measurement_csv', scene=args.scene,
                      source_rows=len(rows), scheduled_rows=len(schedule),
                      preparation_seconds=preparation,
                      queue_model='virtual_serial_fixed_release',
                      measured_service='release_batch+ingest+predict+serialize_result')
        with args.output.open('x', encoding='utf-8') as output:
            output.write(_json(header) + '\n')
            _write_ticks(output, state, schedule, args.start, args.stop)
    except (OSError, ValueError, KeyError, csv.Error, BadZipFile) as exc:
        parser.error(str(exc))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
