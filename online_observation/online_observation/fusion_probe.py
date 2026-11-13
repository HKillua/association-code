"""Position-only 1 Hz fusion diagnostic with fixed T_core identity correspondence."""
from __future__ import annotations

import argparse
from bisect import bisect_left
from collections import Counter, defaultdict, deque
import json
import math
from pathlib import Path
from time import perf_counter

from .evaluation_data import read_labels, read_references
from .measurements import read_archive_rows
from .positions import event_batches, position_step
from .stream import _advance_position


SCENES = ('0', '3', '7')
START, STOP = 0, 7199
STALE_SECONDS = 60
FIT_SECONDS = 60
METHODS = ('hold', 'secant', 'trailing_ols')
READOUTS = ('hold', 'newest_source', 'freshness_weighted')
FRESHNESS_HALF_LIFE_SECONDS = 20
AGE_BINS = ('exact', '0_to_10s', '10_to_20s', '20_to_40s', '40_to_60s')
SOURCE_SCOPES = ('single_source', 'multi_source')
STRATA = ('all', *SOURCE_SCOPES, *AGE_BINS,
          *(f'{scope}/{age}' for scope in SOURCE_SCOPES for age in AGE_BINS))


def _motion(history):
    """Estimate two velocities in the tangent plane at the latest observation."""
    if len(history) == 1:
        return (0., 0.), (0., 0.), False, False
    latest_time, origin = history[-1]
    previous_time, previous = history[-2]
    distance, reverse_bearing = position_step(origin, previous)
    secant = (distance / (latest_time - previous_time), (reverse_bearing + 180.) % 360.)

    window = [(time - latest_time, position) for time, position in history
              if latest_time - time <= FIT_SECONDS]
    if len(window) < 2:
        return secant, (0., 0.), True, False
    if len(window) == 2:
        return secant, secant, True, True
    center = math.fsum(time for time, _ in window) / len(window)
    north_products, east_products, time_squares = [], [], []
    for time, position in window:
        distance, bearing = position_step(origin, position)
        centered = time - center
        angle = math.radians(bearing)
        north_products.append(centered * distance * math.cos(angle))
        east_products.append(centered * distance * math.sin(angle))
        time_squares.append(centered * centered)
    denominator = math.fsum(time_squares)
    north = math.fsum(north_products) / denominator
    east = math.fsum(east_products) / denominator
    fitted = (math.hypot(north, east), math.degrees(math.atan2(east, north)) % 360.)
    return secant, fitted, True, True


def _forecasts(rows, *, hold_only=False):
    """Yield each tick's independent fragment forecasts without evaluation inputs."""
    events = iter((time, points) for time, points in event_batches(rows) if START <= time <= STOP)
    pending = next(events, None)
    histories = defaultdict(deque)
    active = {}
    for tick in range(START, STOP + 1):
        updates = 0
        while pending is not None and pending[0] <= tick:
            time, positions = pending
            updates += len(positions)
            for key, position in positions.items():
                if hold_only:
                    active[key] = (time, position, None)
                    continue
                history = histories[key]
                history.append((time, position))
                while len(history) > 2 and time - history[0][0] > FIT_SECONDS:
                    history.popleft()
                active[key] = (time, position, _motion(history))
            pending = next(events, None)

        forecasts = {}
        expired = []
        for key, (time, position, motion) in active.items():
            age = tick - time
            if age > STALE_SECONDS:
                expired.append(key)
                continue
            if motion is None:
                forecasts[key] = (age, (position,), False, False)
                continue
            secant, fit, has_secant, has_fit = motion
            predictions = (position, _advance_position(*position, *secant, age),
                           _advance_position(*position, *fit, age))
            forecasts[key] = (age, predictions, has_secant, has_fit)
        for key in expired:
            del active[key]
        yield tick, forecasts, updates


def _sphere_mean(positions, weights=None):
    """Weighted mean of directions on the sphere, equal by default."""
    if len(positions) == 1:
        return positions[0]
    weights = [1.] * len(positions) if weights is None else weights
    xyz = []
    for (latitude, longitude), weight in zip(positions, weights):
        lat, lon = math.radians(latitude), math.radians(longitude)
        xyz.append((weight * (math.cos(lat) * math.cos(lon)),
                    weight * (math.cos(lat) * math.sin(lon)), weight * math.sin(lat)))
    x, y, z = (math.fsum(axis) for axis in zip(*xyz))
    if math.hypot(x, y, z) <= 1e-12 * math.fsum(weights):
        return None
    return math.degrees(math.atan2(z, math.hypot(x, y))), math.degrees(math.atan2(y, x))


def _fuse(sources):
    result = []
    for method in range(len(METHODS)):
        per_source = [_sphere_mean([forecast[1][method] for forecast in fragments])
                      for fragments in sources.values()]
        result.append(None if None in per_source else _sphere_mean(per_source))
    return result


def _fuse_hold_readouts(sources):
    positions = [_sphere_mean([forecast[1][0] for forecast in fragments])
                 for fragments in sources.values()]
    if None in positions:
        return [None] * len(READOUTS)
    ages = [min(forecast[0] for forecast in fragments) for fragments in sources.values()]
    youngest = min(ages)
    newest_positions = [position for position, age in zip(positions, ages) if age == youngest]
    weights = [2. ** (-age / FRESHNESS_HALF_LIFE_SECONDS) for age in ages]
    return [_sphere_mean(positions), _sphere_mean(newest_positions), _sphere_mean(positions, weights)]


def _reference_tracks(reference):
    """Keep invalid timed positions on the reference axis to block interpolation."""
    tracks = {}
    audit = Counter()
    for identity, rows in reference.by_identity.items():
        by_time = defaultdict(list)
        for raw_time, lat, lon, _, _ in rows:
            audit['rows'] += 1
            if raw_time is None or not math.isfinite(raw_time):
                audit['unlocatable_time_rows'] += 1
                continue
            stamp = raw_time - reference.time_offset_seconds
            if not math.isfinite(stamp):
                audit['unlocatable_time_rows'] += 1
                continue
            valid = (lat is not None and lon is not None and math.isfinite(lat)
                     and math.isfinite(lon) and abs(lat) <= 90. and abs(lon) <= 180.)
            audit['invalid_position_rows'] += not valid
            by_time[stamp].append((lat, lon) if valid else None)

        times = sorted(by_time)
        positions = []
        for time in times:
            values = by_time[time]
            audit['duplicate_time_groups'] += len(values) > 1
            conflict = any(value != values[0] for value in values[1:])
            audit['invalid_duplicate_position_groups'] += conflict
            positions.append(None if conflict else values[0])
        tracks[identity] = times, positions
    return tracks, dict(audit)


def _truth_position(track, tick, max_gap):
    if track is None:
        return None, 'no_reference_identity'
    times, positions = track
    if not times:
        return None, 'no_locatable_reference_time'
    right = bisect_left(times, tick)
    if right < len(times) and tick == times[right]:
        position = positions[right]
        return position, None if position is not None else 'invalid_reference_position'
    if right == 0 or right == len(times):
        return None, 'outside_reference_support'
    left = right - 1
    delta = times[right] - times[left]
    if delta > max_gap:
        return None, 'reference_gap_exceeded'
    a, b = positions[left], positions[right]
    if a is None or b is None:
        return None, 'invalid_reference_position'
    fraction = (tick - times[left]) / delta
    longitude_delta = b[1] - a[1]
    if abs(longitude_delta) > 180.:
        longitude_delta = (longitude_delta + 180.) % 360. - 180.
    longitude = a[1] + fraction * longitude_delta
    if abs(longitude) > 180.:
        longitude = (longitude + 180.) % 360. - 180.
    return (a[0] + fraction * (b[0] - a[0]), longitude), None


def _count_reference_ticks(tracks, max_gap):
    """Count all valid reference identity-ticks, including targets never emitted."""
    count = 0
    for times, positions in tracks.values():
        for time, position in zip(times, positions):
            count += position is not None and START <= time <= STOP and time.is_integer()
        for index in range(1, len(times)):
            left, right = times[index - 1], times[index]
            if positions[index - 1] is None or positions[index] is None or right - left > max_gap:
                continue
            first = max(START, math.floor(left) + 1)
            last = min(STOP, math.ceil(right) - 1)
            count += max(0, last - first + 1)
    return count


def _age_bin(age):
    if age == 0.:
        return AGE_BINS[0]
    return next(name for limit, name in zip((10, 20, 40, 60), AGE_BINS[1:]) if age <= limit)


def _percentile(sorted_values, probability):
    index = probability * (len(sorted_values) - 1)
    low, high = math.floor(index), math.ceil(index)
    return sorted_values[low] + (index - low) * (sorted_values[high] - sorted_values[low])


def _metrics(errors):
    count = len(errors)
    if not count:
        return dict(count=0, mae_m=None, rmse_m=None, p95_m=None, p99_m=None)
    ordered = sorted(errors)
    return dict(count=count, mae_m=math.fsum(errors) / count,
                rmse_m=math.sqrt(math.fsum(error * error for error in errors) / count),
                p95_m=_percentile(ordered, .95), p99_m=_percentile(ordered, .99))


def _scores(methods=METHODS):
    return {stratum: {method: [] for method in methods} for stratum in STRATA}


def _summarize(errors):
    return {stratum: {method: _metrics(values) for method, values in series.items()}
            for stratum, series in errors.items()}


def _evaluate(rows, labels, reference, pooled, *, readout_controls=False):
    methods = READOUTS if readout_controls else METHODS
    fuse = _fuse_hold_readouts if readout_controls else _fuse
    tracks, reference_audit = _reference_tracks(reference)
    errors = _scores(methods)
    missing_truth = Counter()
    counts = Counter()
    target_sums = defaultdict(Counter)
    max_gap = reference.max_gap_seconds
    counts['valid_reference_target_ticks'] = _count_reference_ticks(tracks, max_gap)
    for tick, forecasts, updates in _forecasts(rows, hold_only=readout_controls):
        counts['ticks'] += 1
        counts['position_updates'] += updates
        counts['ticks_without_position_updates'] += updates == 0
        counts['active_fragment_ticks'] += len(forecasts)
        objects = defaultdict(lambda: defaultdict(list))
        for key, forecast in forecasts.items():
            if not readout_controls:
                _, _, has_secant, has_fit = forecast
                counts['secant_hold_fallback_fragment_ticks'] += not has_secant
                counts['trailing_ols_hold_fallback_fragment_ticks'] += not has_fit
            identity = labels.get(key)
            if identity is None:
                counts['unmatched_label_fragment_ticks'] += 1
            else:
                objects[identity][key[0]].append(forecast)

        for identity, sources in objects.items():
            counts['oracle_group_target_ticks'] += 1
            counts['multi_source_group_target_ticks'] += len(sources) > 1
            truth, reason = _truth_position(tracks.get(identity), tick, max_gap)
            if truth is None:
                missing_truth[reason] += 1
                continue
            predictions = fuse(sources)
            if any(position is None for position in predictions):
                counts['common_fusion_undefined_target_ticks'] += 1
                continue
            age = min(forecast[0] for fragments in sources.values() for forecast in fragments)
            source_scope = 'multi_source' if len(sources) > 1 else 'single_source'
            age_bin = _age_bin(age)
            target = target_sums[identity]
            target['ticks'] += 1
            for method, position in zip(methods, predictions):
                error, _ = position_step(truth, position)
                target[method] += error
                for stratum in ('all', source_scope, age_bin, f'{source_scope}/{age_bin}'):
                    errors[stratum][method].append(error)

    counts['common_scored_target_ticks'] = len(errors['all']['hold'])
    counts['valid_reference_target_ticks_without_common_output'] = (
        counts['valid_reference_target_ticks'] - counts['common_scored_target_ticks'])
    for stratum, series in errors.items():
        for method, values in series.items():
            pooled[stratum][method].extend(values)
    counts['common_scored_targets'] = len(target_sums)
    target_mae = {method: _metrics([target[method] / target['ticks'] for target in target_sums.values()])
                  for method in methods}
    return dict(counts=dict(counts), excluded_output_truth=dict(missing_truth),
                reference_audit=reference_audit, time_offset_seconds=reference.time_offset_seconds,
                max_reference_gap_seconds=max_gap, position=_summarize(errors),
                target_position_mae_distribution=target_mae)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', nargs='+', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--readout-controls', action='store_true',
                        help='hold each fragment and compare across-source weighting')
    args = parser.parse_args(argv)
    manifest = json.loads(args.manifest.read_text(encoding='utf-8'))
    if tuple(manifest['splits']['T_core'][:len(SCENES)]) != SCENES:
        parser.error('expected the frozen T_core fusion-probe scenes')

    started = perf_counter()
    pooled = _scores(READOUTS if args.readout_controls else METHODS)
    scenes = {}
    for scene in SCENES:
        rows = read_archive_rows(args.archive, args.manifest, scene)
        labels = read_labels(args.archive, args.manifest, scene)
        reference = read_references(args.archive, args.manifest, scene)
        scenes[scene] = _evaluate(rows, labels, reference, pooled, readout_controls=args.readout_controls)
    result = dict(
        role='T_core', level='Level1', scene_ids=SCENES, start=START, stop=STOP, cadence_seconds=1,
        arrival_clock='ideal_measurement_time', warmup=False, stale_after_seconds=STALE_SECONDS,
        fit_window_seconds=FIT_SECONDS, inputs=['source', 'batch', 'time', 'lat', 'lon'],
        motion=dict(hold='latest observed position',
                    secant='latest two distinct event times, with no extra bound on the interval',
                    trailing_ols='tangent-plane OLS slope over the latest 60 event seconds, anchored at the latest position',
                    insufficient_history='hold'),
        grouping='A-B fixed identity oracle applied to independent active fragment forecasts',
        fusion='spherical equal mean within each source, then equal mean across sources',
        age='age of youngest contributing fragment, nonzero bins have inclusive upper bounds',
        population='all reference identities with valid position support; score the common output intersection',
        pooled_position=_summarize(pooled), scenes=scenes, wall_seconds=perf_counter() - started,
    )
    if args.readout_controls:
        del result['fit_window_seconds']
        result['motion'] = dict(hold='latest observed position')
        result['fusion'] = 'spherical equal mean within each source before each across-source readout'
        result['readout_controls'] = dict(
            hold='equal across sources', newest_source='only the youngest sources, equal across exact ties',
            freshness_weighted='weights proportional to 2 ** (-source_age_seconds / half_life_seconds)',
            half_life_seconds=FRESHNESS_HALF_LIFE_SECONDS,
            source_age='age of the youngest contributing fragment from that source',
        )
    with args.output.open('x', encoding='utf-8') as output:
        json.dump(result, output, indent=2, allow_nan=False)
        output.write('\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
