"""Audit when fields in MTAD Level1 measurement rows can be used online."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
from statistics import median
from time import perf_counter

from .measurements import read_archive_rows
from .positions import angle_error, position_step
from .stream import EARTH_RADIUS_M, KNOT_TO_MPS


SCENES = ("0", "3", "7", "9", "10", "11", "14", "17")


def audit_scene(rows, observations):
    groups = defaultdict(list)
    invalid = Counter()
    raw_order_reversals = 0
    last = -math.inf
    for row in rows:
        stamp = float(row["time"])
        position = (float(row["lat"]), float(row["lon"]))
        speed, course = float(row["vel"]), float(row["cou"])
        source, batch = row["source"], row["batch"]
        if not source or not batch or not math.isfinite(stamp):
            raise ValueError("measurement row lacks schedulable time or fragment key")
        raw_order_reversals += stamp < last
        last = stamp
        if (not all(map(math.isfinite, position)) or abs(position[0]) > 90 or abs(position[1]) > 180):
            invalid["position"] += 1
        if not math.isfinite(speed) or speed < 0:
            invalid["speed"] += 1
        if not math.isfinite(course) or not 0 <= course <= 360:
            invalid["course"] += 1
        groups[source, batch].append((stamp, position, speed, course))

    counts = Counter()
    endpoints = []
    for (source, batch), samples in groups.items():
        samples.sort()
        endpoints.append((int(batch), samples[0][0]))
        times = Counter(sample[0] for sample in samples)
        for first, second in zip(samples, samples[1:]):
            a, position_a, speed_a, course_a = first
            b, position_b, speed_b, course_b = second
            counts["intervals"] += 1
            if times[a] != 1 or times[b] != 1:
                counts["duplicate_time_intervals"] += 1
                continue
            if any(not math.isfinite(value) or abs(value) > bound
                   for position in (position_a, position_b) for value, bound in zip(position, (90, 180))):
                counts["invalid_position_intervals"] += 1
                continue
            distance, forward = position_step(position_a, position_b)
            observed = observations[source]
            observed["gaps"].append(b - a)
            if distance <= 1.0:
                counts["displacement_at_most_1m"] += 1
                continue
            geometric_speed = distance / (b - a)
            if all(math.isfinite(speed) and speed >= 0 for speed in (speed_a, speed_b)):
                errors = [abs(speed * KNOT_TO_MPS - geometric_speed) for speed in (speed_a, speed_b)]
                observed["speed_first_error"].append(errors[0])
                observed["speed_second_error"].append(errors[1])
                observed["speed_first_within_1pct"].append(errors[0] < 0.01 * geometric_speed)
                observed["speed_second_within_1pct"].append(errors[1] < 0.01 * geometric_speed)
            if all(math.isfinite(course) and 0 <= course <= 360 for course in (course_a, course_b)):
                _, reverse = position_step(position_b, position_a)
                observed["course_first_error"].append(angle_error(course_a, forward))
                observed["course_second_error"].append(angle_error(course_b, reverse + 180))
    for (source, _), samples in groups.items():
        if len(samples) < 2:
            continue
        previous, last_sample = samples[-2:]
        dt = last_sample[0] - previous[0]
        if dt <= 0 or any(not math.isfinite(x) or abs(x) > bound
                          for sample in (previous, last_sample)
                          for x, bound in zip(sample[1], (90, 180))):
            continue
        distance, course = position_step(previous[1], last_sample[1])
        if distance <= 1.0:
            continue
        observed = observations[source]
        if math.isfinite(last_sample[2]) and last_sample[2] >= 0:
            observed["terminal_speed_error"].append(abs(last_sample[2] * KNOT_TO_MPS - distance / dt))
        if math.isfinite(last_sample[3]) and 0 <= last_sample[3] <= 360:
            observed["terminal_course_error"].append(angle_error(last_sample[3], course))

    endpoints.sort()
    first_tick = math.floor(min(samples[0][0] for samples in groups.values()))
    last_tick = math.ceil(max(samples[-1][0] for samples in groups.values()))
    arrivals = Counter(math.ceil(sample[0]) for samples in groups.values() for sample in samples)
    return dict(
        rows=len(rows), fragments=len(groups), invalid=dict(invalid),
        csv_time_reversals=raw_order_reversals,
        batch_values_repeated_across_sources=len(groups) - len({batch for _, batch in groups}),
        first_time_reversals_in_numeric_batch_order=sum(a[1] > b[1] for a, b in zip(endpoints, endpoints[1:])),
        tick_count=last_tick - first_tick + 1,
        ticks_without_arrivals=last_tick - first_tick + 1 - len(arrivals),
        largest_arrival_batch=max(arrivals.values()), intervals=dict(counts),
    )


def _summary(values):
    return dict(
        intervals=len(values["gaps"]), gap_median_s=median(values["gaps"]),
        gap_max_s=max(values["gaps"]), gap_over_30s=sum(gap > 30 for gap in values["gaps"]),
        speed_intervals=len(values["speed_first_error"]),
        speed_first_error_median_mps=median(values["speed_first_error"]),
        speed_second_error_median_mps=median(values["speed_second_error"]),
        speed_first_within_1pct=sum(values["speed_first_within_1pct"]),
        speed_second_within_1pct=sum(values["speed_second_within_1pct"]),
        course_intervals=len(values["course_first_error"]),
        course_first_error_median_deg=median(values["course_first_error"]),
        course_second_error_median_deg=median(values["course_second_error"]),
        course_first_within_1deg=sum(error <= 1 for error in values["course_first_error"]),
        course_second_within_1deg=sum(error <= 1 for error in values["course_second_error"]),
        terminal_speed_count=len(values["terminal_speed_error"]),
        terminal_speed_error_median_mps=median(values["terminal_speed_error"]),
        terminal_course_count=len(values["terminal_course_error"]),
        terminal_course_error_median_deg=median(values["terminal_course_error"]),
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", nargs="+", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if tuple(manifest["splits"]["T_core"][:len(SCENES)]) != SCENES:
        parser.error("expected the frozen T_core field-audit scenes")
    began = perf_counter()
    scenes, pooled = {}, defaultdict(lambda: defaultdict(list))
    for scene in SCENES:
        by_source = defaultdict(lambda: defaultdict(list))
        counts = audit_scene(read_archive_rows(args.archive, args.manifest, scene), by_source)
        counts["sources"] = {source: _summary(values) for source, values in by_source.items()}
        scenes[scene] = counts
        for source, values in by_source.items():
            for name, series in values.items():
                pooled[source][name].extend(series)
    result = dict(
        level="Level1", role="T_core", scene_ids=SCENES, row_access="measurements_only",
        displacement_floor_m=1, earth_radius_m=EARTH_RADIUS_M, scenes=scenes,
        pooled_by_source={source: _summary(values) for source, values in pooled.items()},
        wall_seconds=perf_counter() - began,
    )
    with args.output.open("x", encoding="utf-8") as output:
        json.dump(result, output, indent=2, allow_nan=False)
        output.write("\n")


if __name__ == "__main__":
    main()
