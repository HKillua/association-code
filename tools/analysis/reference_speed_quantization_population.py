"""C120: audit native reference-speed quantization on the fixed C117 development roster."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
import os
from pathlib import Path
import resource
import time

import numpy as np

from association.data.physical_units import KNOT_TO_MPS
from association.data.reference_states import iter_reference_states
from tools.analysis.held_fusion_population import PARENTS, SPLIT
from tools.analysis.raw_reference_speed_origin import raw_track
from tools.analysis.reference_interval_fingerprint import ORDINARY, TAIL, geometry, wrap_degrees
from tools.analysis.same_source_swap_pilot import MAX_AS, peak_rss_bytes, read, write

C117 = 'LEVEL1-REFERENCE-INTERVAL-FINGERPRINT-C117-20261112-attempt-1'
GRIDS = {
    'tenths_kn_exact_mps': (Decimal(463), Decimal(9000)),
    'tenths_kn_approx_mps': (Decimal('0.0514'), Decimal(1)),
    'tenths_mps': (Decimal('0.1'), Decimal(1)),
}
TOLERANCES = {'within_0_5e-6': Decimal('0.0000005'),
              'within_1e-6': Decimal('0.000001'),
              'within_2e-6': Decimal('0.000002')}
PRIMARY = 'tenths_kn_exact_mps'


def grid_error(value, grid):
    numerator, denominator = GRIDS[grid]
    scaled = value * denominator
    index = int((scaled / numerator).to_integral_value(rounding=ROUND_HALF_UP))
    return index, abs(scaled - numerator * index) / denominator


def speed_at_time(rows):
    values = set()
    for row in rows:
        try:
            value = Decimal(row['raw']['vel'])
        except InvalidOperation:
            return 'invalid', None
        if not value.is_finite() or value < 0:
            return 'invalid', None
        values.add(value)
    if len(values) != 1:
        return 'conflict', None
    value = next(iter(values))
    return ('positive' if value > 0 else 'zero'), value


def endpoint_class(value, state):
    if state == 'zero':
        return 'zero'
    if state != 'positive':
        return 'unresolved'
    return 'on' if grid_error(value, PRIMARY)[1] <= TOLERANCES['within_1e-6'] else 'off'


def grid_summary(histogram):
    summary = {}
    for grid in GRIDS:
        errors = [(grid_error(value, grid), count) for value, count in histogram.items()]
        main = TOLERANCES['within_1e-6']
        summary[grid] = {
            **{label: sum(count for (_, error), count in errors if error <= tol)
               for label, tol in TOLERANCES.items()},
            'distinct_within_1e-6': sum(error <= main for (_, error), _ in errors),
            'max_error_mps': float(max((error for (_, error), _ in errors), default=Decimal(0))),
            'numeric_code_1022': sum(count for (index, error), count in errors if index == 1022 and error <= main),
            'numeric_code_1023': sum(count for (index, error), count in errors if index == 1023 and error <= main),
        }
    return summary


def endpoint_signature(states, start, count):
    return '+'.join(sorted(states[start:start+count]))


def track_joint(times, values, endpoint_classes, previous):
    elapsed = np.diff(times)
    metres, start_angle, end_angle = geometry(values[:-1, :2], values[1:, :2])
    speed = metres / elapsed
    left, right = values[:-1, 2], values[1:, 2]
    largest, smallest = np.maximum(left, right), np.minimum(left, right)
    comparable = np.isfinite(speed) & np.isfinite(left) & np.isfinite(right)
    excess = comparable & (speed - largest > 10.)
    deficit = comparable & (smallest - speed > 10.)
    other = comparable & ~excess & ~deficit
    alt_unit = comparable & (speed - largest * KNOT_TO_MPS > 10.)
    interval_counts = Counter()
    for bucket, population in [('up_to_600_seconds', elapsed <= 600.),
                               ('over_600_seconds', elapsed > 600.)]:
        reference = previous[bucket]
        for name, mask in [('comparable', comparable), ('above_gt10', excess), ('below_gt10', deficit),
                           ('above_gt10_if_kn', alt_unit)]:
            assert int((mask & population).sum()) == reference[name], (bucket, name)
        assert int(population.sum()) == reference['intervals']
        for name, mask in [('positive_excess', excess), ('negative_excess', deficit), ('other', other)]:
            for i in np.flatnonzero(mask & population):
                interval_counts[f'{bucket}/{name}/{endpoint_signature(endpoint_classes, i, 2)}'] += 1
    short_excess = excess & (elapsed <= 600.)
    pair = short_excess[:-1] & short_excess[1:]
    turning = abs(wrap_degrees(start_angle[1:] - end_angle[:-1]))
    net_metres = geometry(values[:-2, :2], values[2:, :2])[0]
    path = metres[:-1] + metres[1:]
    reversal = pair & (turning >= 120.) & (net_metres <= .5 * path)
    same_direction = pair & (turning <= 60.)
    middle = pair & ~reversal & ~same_direction
    assert int(pair.sum()) == previous['adjacent']['pairs']
    pair_counts = Counter()
    for name, mask in [('reversal', reversal), ('same_direction', same_direction), ('other', middle)]:
        assert int(mask.sum()) == previous['adjacent'][name], name
        for i in np.flatnonzero(mask):
            pair_counts[f'{name}/{endpoint_signature(endpoint_classes, i, 3)}'] += 1
    return interval_counts, pair_counts


def scene_report(scene, old):
    by_identity = defaultdict(list)
    invalid_identity = 0
    for row in scene['rows']:
        fields = row['identity_id'].split('-')
        if len(fields) == 2 and all(fields):
            by_identity[row['identity_id']].append(row)
        else:
            invalid_identity += 1
    assert scene['scene_id'] == old['scene']
    assert len(scene['rows']) == old['rows'] and invalid_identity == old['invalid_identity_rows']
    assert len(by_identity) == old['tracks']
    positive_values, speed_counts = Counter(), Counter()
    interval_counts, pair_counts = Counter(), Counter()
    track_rows = []
    for (track, identity), previous in zip(enumerate(sorted(by_identity), 1), old['by_track'], strict=True):
        times, states, raw_counts, groups = raw_track(by_identity[identity])
        assert track == previous['track'] and dict(raw_counts) == previous['raw']
        track_speed, track_positive = Counter(), Counter()
        endpoint_classes = []
        for timestamp in times:
            status, value = speed_at_time(groups[timestamp])
            track_speed[status] += 1
            if status == 'positive':
                track_positive[value] += 1
            endpoint_classes.append(endpoint_class(value, status))
        assert sum(track_speed.values()) == len(times)
        track_speed['unlocatable_rows'] = raw_counts['unlocatable_rows']
        track_intervals, track_pairs = track_joint(times, states, endpoint_classes, previous)
        primary = grid_summary(track_positive)[PRIMARY]
        track_rows.append(dict(track=track, speed_counts=dict(track_speed), main_grid_positive=primary['within_1e-6'],
                               interval_counts=dict(track_intervals), pair_counts=dict(track_pairs)))
        positive_values.update(track_positive)
        speed_counts.update(track_speed)
        interval_counts.update(track_intervals)
        pair_counts.update(track_pairs)
    assert sum(speed_counts[s] for s in ('positive', 'zero', 'invalid', 'conflict')) == old['counts']['raw']['times']
    report = dict(scene=old['scene'], tracks=len(by_identity), rows=old['rows'],
                  invalid_identity_rows=invalid_identity, speed_counts=dict(speed_counts),
                  distinct_positive_values=len(positive_values), grid_summary=grid_summary(positive_values),
                  interval_counts=dict(interval_counts), pair_counts=dict(pair_counts), by_track=track_rows)
    return report, positive_values


def run(runs, output):
    start = time.monotonic()
    output.mkdir(exist_ok=False)
    record = dict(status='RUNNING', idea='IDEA-C120-REFERENCE-SPEED-QUANTIZATION-20261112', plan='1.0',
                  started_at=datetime.now(timezone.utc).isoformat(), affinity=sorted(os.sched_getaffinity(0)),
                  cuda=os.environ.get('CUDA_VISIBLE_DEVICES'), raw_reference_members_read=0,
                  raw_label_members_read=0, raw_measurement_members_read=0,
                  test_opened=False, updates=0, predictions=0)
    write(output/'run.json', record)
    try:
        assert record['affinity'] == [27] and record['cuda'] == ''
        previous = runs / C117
        original = read(previous/'run.json')
        assert original['status'] == 'COMPLETED' and original['raw_reference_members_read'] == 20
        scope = read(previous/'scope.json')
        cohorts = {'ordinary': ORDINARY, 'tail_context': TAIL}
        parent = read(runs/PARENTS['full_data']/'run.json')
        assert parent['status'] == 'COMPLETED' and parent['exit_code'] == 0
        assert parent['reference_speed_unit'] == 'mps' and parent['measurement_speed_unit'] == 'kn'
        manifest = read(Path(parent['manifest']))
        components = manifest['components']
        assert set(ORDINARY + TAIL).issubset(manifest['splits'][SPLIT])
        assert not set(ORDINARY + TAIL).intersection(manifest['splits']['T_test'])
        assert not {components[s] for s in ORDINARY}.intersection(components[s] for s in TAIL)
        write(output/'scope.json', scope)
        for cohort, scenes in cohorts.items():
            assert tuple(scope[cohort]) == scenes
            original_scenes = read(previous/f'{cohort}-by-scene.json.gz')
            assert tuple(s['scene'] for s in original_scenes) == scenes
            raw_scenes = iter_reference_states(parent['archive'], manifest, split=SPLIT, scene_ids=scenes)
            reports, total_values = [], Counter()
            totals = defaultdict(Counter)
            for raw, old in zip(raw_scenes, original_scenes, strict=True):
                record['raw_reference_members_read'] += 1
                report, values = scene_report(raw, old)
                reports.append(report)
                total_values.update(values)
                for name in ('speed_counts', 'interval_counts', 'pair_counts'):
                    totals[name].update(report[name])
            assert sum(total_values.values()) == totals['speed_counts']['positive']
            summary = dict(scenes=len(reports), components=len({components[s] for s in scenes}),
                           tracks=sum(s['tracks'] for s in reports), rows=sum(s['rows'] for s in reports),
                           invalid_identity_rows=sum(s['invalid_identity_rows'] for s in reports),
                           distinct_positive_values=len(total_values), grid_summary=grid_summary(total_values),
                           **{name: dict(count) for name, count in totals.items()})
            write(output/f'{cohort}-by-scene.json.gz', reports)
            write(output/f'{cohort}-positive-speed-frequencies.json.gz',
                  [[format(value, 'f'), count] for value, count in sorted(total_values.items())])
            write(output/f'{cohort}.json', summary)
        assert record['raw_reference_members_read'] == 20
        record.update(status='COMPLETED', exit_code=0)
    except BaseException as error:
        record.update(status='FAILED', exit_code=1, error_type=type(error).__name__)
        raise
    finally:
        record.update(elapsed_seconds=time.monotonic()-start, peak_rss_bytes=peak_rss_bytes(),
                      finished_at=datetime.now(timezone.utc).isoformat())
        write(output/'run.json', record)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    resource.setrlimit(resource.RLIMIT_AS, (MAX_AS, MAX_AS))
    run(args.runs, args.output)
