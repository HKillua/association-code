"""Measure causal history support at first and second Level1 fragment reports."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from itertools import groupby
import json
import math
from pathlib import Path
from time import perf_counter

from .continuous_factorial import ROSTER
from .continuous_tracker import FRESH_S, RADIUS_M, Plane
from .evaluation_data import read_labels
from .measurements import read_archive_rows
from .positions import event_batches
from .registration_probe import FIT_STOP_S, estimate_translation
from .rejection_probe import Fragment


HISTORY_S = 120
MIN_NEIGHBOR_POINTS = 5
GATE_SQUARED = RADIUS_M**2
COHORTS = ("first", "second", "first_with_second", "first_missing_second", "first_timed_out_second")
POOL_SIZES = ("0-3", "4-15", "16+")


@dataclass
class History:
    first_stamp: float
    latest_stamp: float
    point: tuple[float, float]
    recent: deque[float]
    report_count: int = 1

    def count_recent(self, stamp):
        while self.recent and stamp - self.recent[0] > HISTORY_S:
            self.recent.popleft()
        return len(self.recent)


@dataclass(frozen=True)
class Candidate:
    key: Fragment
    distance_rank: int
    latest_stamp: float
    points: int
    neighbors: int


@dataclass(frozen=True)
class Sample:
    key: Fragment
    stamp: float
    points: int
    neighbors: int
    candidates: tuple[Candidate, ...]


@dataclass(frozen=True)
class Prepared:
    first: dict[Fragment, Sample]
    second: dict[Fragment, Sample]
    timed_out: frozenset[Fragment]
    first_seen: dict[Fragment, float]
    stop: int
    native_stop: float | None
    fit: dict


def _squared(first, second):
    east = first[0] - second[0]
    north = first[1] - second[1]
    return east * east + north * north


def _snapshot(stamp, queries, histories, shifts):
    fresh = {key: history for key, history in histories.items() if stamp - history.latest_stamp <= FRESH_S}
    points = {key: history.count_recent(stamp) for key, history in fresh.items()}
    mature = defaultdict(list)
    for key, count in points.items():
        if count >= MIN_NEIGHBOR_POINTS:
            mature[key[0]].append((key, fresh[key].point))
    neighbors = {}

    def neighbor_count(key):
        if key not in neighbors:
            neighbors[key] = sum(other != key and _squared(fresh[key].point, point) <= GATE_SQUARED
                                 for other, point in mature[key[0]])
        return neighbors[key]

    corrected = {}
    for key, history in fresh.items():
        shift = shifts.get(key[0])
        corrected[key] = history.point if shift is None else (
            history.point[0] + shift[0], history.point[1] + shift[1])

    samples = []
    for key in queries:
        in_gate = []
        for other, point in corrected.items():
            if key[0] != other[0]:
                distance = _squared(corrected[key], point)
                if distance <= GATE_SQUARED:
                    in_gate.append((other, distance))
        in_gate.sort(key=lambda pair: pair[1])
        candidates = []
        rank = 1
        for _, tied in groupby(in_gate, key=lambda pair: pair[1]):
            cohort = tuple(other for other, _ in tied)
            candidates.extend(Candidate(other, rank, fresh[other].latest_stamp, points[other], neighbor_count(other))
                              for other in cohort)
            rank += len(cohort)
        samples.append(Sample(key, stamp, points[key], neighbor_count(key), tuple(candidates)))
    return samples


def prepare_support(events):
    """Freeze snapshots from strictly increasing native event batches and positions."""
    events = tuple(events)
    fit = estimate_translation(events)
    shifts = {}
    translation = fit["translation_east_north_m"]
    if translation is not None:
        left, right = fit["source_order"]
        half = tuple(value / 2 for value in translation)
        shifts = {left: half, right: tuple(-value for value in half)}
    if not events:
        return Prepared({}, {}, frozenset(), {}, 0, None, fit)

    plane = Plane.from_first_batch(events[0][1].values())
    histories, first, second = {}, {}, {}
    timed_out = set()
    previous_stamp = -math.inf
    for stamp, positions in events:
        if not math.isfinite(stamp) or stamp < 0 or stamp <= previous_stamp:
            raise ValueError("expected strictly increasing nonnegative native event batches")
        previous_stamp = stamp
        queries = {}
        for key, coordinates in positions.items():
            point = plane.project(coordinates)
            if key not in histories:
                histories[key] = History(stamp, stamp, point, deque((stamp,)))
                queries[key] = first
                continue
            history = histories[key]
            history.latest_stamp = stamp
            history.point = point
            history.recent.append(stamp)
            history.count_recent(stamp)
            history.report_count += 1
            if history.report_count == 2:
                if stamp - history.first_stamp <= HISTORY_S:
                    queries[key] = second
                else:
                    timed_out.add(key)
        if queries:
            for sample in _snapshot(stamp, queries, histories, shifts if stamp > FIT_STOP_S else {}):
                queries[sample.key][sample.key] = sample

    return Prepared(first, second, frozenset(timed_out),
                    {key: history.first_stamp for key, history in histories.items()},
                    math.ceil(events[-1][0]), events[-1][0], fit)


def _support(candidate):
    return candidate["recent_points"] >= MIN_NEIGHBOR_POINTS or candidate["qualified_neighbors"] > 0


def _serialize_sample(sample, labels, anonymous, source_index, first_per_identity_source):
    identity = labels[sample.key]
    history_visible = any(source != sample.key[0] and seen <= sample.stamp
                          for source, seen in first_per_identity_source[identity].items())
    candidates = [dict(
        fragment_index=anonymous[candidate.key], source_index=source_index[candidate.key[0]],
        distance_rank=candidate.distance_rank,
        latest_native_time=candidate.latest_stamp, recent_points=candidate.points,
        qualified_neighbors=candidate.neighbors, is_true=labels[candidate.key] == identity,
    ) for candidate in sample.candidates]
    candidates.sort(key=lambda candidate: (candidate["distance_rank"], candidate["fragment_index"]))
    nearest = [candidate for candidate in candidates if candidate["distance_rank"] == 1]
    selected = nearest[0] if len(nearest) == 1 else None
    false_candidates = [candidate for candidate in candidates if not candidate["is_true"]]
    false_rank = false_candidates[0]["distance_rank"] if false_candidates else None
    nearest_false = [candidate for candidate in false_candidates if candidate["distance_rank"] == false_rank]
    return dict(
        native_time=sample.stamp, query_recent_points=sample.points,
        query_qualified_neighbors=sample.neighbors, true_opposite_history_visible=history_visible,
        candidates=candidates,
        nearest_outcome=("no_candidate" if not nearest else "tie" if selected is None else
                         "true" if selected["is_true"] else "false"),
        nearest_candidate=selected["fragment_index"] if selected is not None else None,
        nearest_false_candidate=nearest_false[0]["fragment_index"] if len(nearest_false) == 1 else None,
        nearest_false_tied=len(nearest_false) > 1,
    )


def _event_counts(sample):
    candidates = sample["candidates"]
    true = [candidate for candidate in candidates if candidate["is_true"]]
    false = next((candidate for candidate in candidates
                  if candidate["fragment_index"] == sample["nearest_false_candidate"]), None)
    nearest_false = sample["nearest_outcome"] == "false"
    wrong_with_true = nearest_false and bool(true)
    true_five = any(candidate["recent_points"] >= MIN_NEIGHBOR_POINTS for candidate in true)
    false_five = false is not None and false["recent_points"] >= MIN_NEIGHBOR_POINTS
    true_neighbor = any(candidate["qualified_neighbors"] > 0 for candidate in true)
    true_five_with_neighbor = any(candidate["recent_points"] >= MIN_NEIGHBOR_POINTS
                                  and candidate["qualified_neighbors"] > 0 for candidate in true)
    query_points = sample["query_recent_points"]
    query_neighbor = sample["query_qualified_neighbors"] > 0
    counts = dict(
        events=1, candidate_edges=len(candidates), true_candidate_edges=len(true),
        has_true_opposite_history=sample["true_opposite_history_visible"],
        true_reachable=bool(true), nearest_wrong_with_true_reachable=wrong_with_true,
        wrong_with_true_reachable_any_true_peer_5=wrong_with_true and true_five,
        wrong_with_true_reachable_both_peers_5=wrong_with_true and true_five and false_five,
        wrong_with_true_reachable_any_true_peer_has_neighbor=wrong_with_true and true_neighbor,
        wrong_with_true_reachable_true_peer_5_and_neighbor=wrong_with_true and true_five_with_neighbor,
        wrong_with_true_reachable_both_peers_5_and_true_peer_neighbor=(
            wrong_with_true and true_five_with_neighbor and false_five),
        sole_false_without_true_history=(len(candidates) == 1 and nearest_false
                                         and not sample["true_opposite_history_visible"]),
        query_points_at_least_2=query_points >= 2, query_points_at_least_5=query_points >= 5,
        query_has_qualified_neighbor=query_neighbor,
        query_5_points_or_neighbor=query_points >= MIN_NEIGHBOR_POINTS or query_neighbor,
        any_true_peer_5_points_or_neighbor=any(map(_support, true)),
        nearest_false_exists=false is not None, nearest_false_tied=sample["nearest_false_tied"],
        nearest_false_5_points_or_neighbor=false is not None and _support(false),
        selected_false_5_points_or_neighbor=nearest_false and false is not None and _support(false),
    )
    for k in (2, 5):
        any_support = any(candidate["recent_points"] >= k for candidate in candidates)
        true_support = any(candidate["recent_points"] >= k for candidate in true)
        counts.update({
            f"any_peer_points_at_least_{k}": any_support,
            f"true_peer_points_at_least_{k}": true_support,
            f"both_sides_{k}_points_any_edge": query_points >= k and any_support,
            f"both_sides_{k}_points_true_edge": query_points >= k and true_support,
            f"nearest_false_points_at_least_{k}": false is not None and false["recent_points"] >= k,
        })
    counts.update(
        true_peer_has_qualified_neighbor=true_neighbor,
        nearest_false_has_qualified_neighbor=false is not None and false["qualified_neighbors"] > 0,
        both_sides_5_points_or_neighbor_true_edge=(counts["query_5_points_or_neighbor"]
                                                   and counts["any_true_peer_5_points_or_neighbor"]),
    )
    counts.update({f"nearest_{outcome}": sample["nearest_outcome"] == outcome
                   for outcome in ("true", "false", "tie", "no_candidate")})
    return {field: int(value) for field, value in counts.items()}


def _paired_counts(first, second, before, after, first_seen_by_index):
    prior_true = {candidate["fragment_index"] for candidate in first["candidates"] if candidate["is_true"]}
    later_true = {candidate["fragment_index"] for candidate in second["candidates"] if candidate["is_true"]}
    return dict(
        gained_true_reachability=int(not prior_true and bool(later_true)),
        lost_true_reachability=int(bool(prior_true) and not later_true),
        newly_reachable_true_candidate=int(bool(later_true - prior_true)),
        newly_seen_true_candidate=int(any(first_seen_by_index[index] > first["native_time"] for index in later_true)),
        true_candidate_with_new_opposite_report=int(any(
            candidate["is_true"] and candidate["latest_native_time"] > first["native_time"]
            for candidate in second["candidates"])),
        gained_true_both_sides_2_points=int(not before["both_sides_2_points_true_edge"]
                                           and after["both_sides_2_points_true_edge"]),
        gained_true_both_sides_5_points_or_neighbor=int(not before["both_sides_5_points_or_neighbor_true_edge"]
                                                       and after["both_sides_5_points_or_neighbor_true_edge"]),
    )


def _cohort_counters():
    return {cohort: Counter(events=0) for cohort in COHORTS}


def _dump_cohorts(cohorts):
    return {cohort: dict(counts) for cohort, counts in cohorts.items()}


def score_support(prepared, labels):
    """Attach labels only after the label-free history and candidates are frozen."""
    if not prepared.first_seen.keys() <= labels.keys():
        raise ValueError("unmapped measurement fragment in post-replay labels")
    anonymous = {key: index for index, key in enumerate(prepared.first_seen, start=1)}
    source_index = {source: index for index, source in
                    enumerate(sorted({key[0] for key in prepared.first_seen}), start=1)}
    first_seen_by_index = {anonymous[key]: stamp for key, stamp in prepared.first_seen.items()}
    first_per_identity_source = defaultdict(dict)
    for key, stamp in prepared.first_seen.items():
        by_source = first_per_identity_source[labels[key]]
        by_source[key[0]] = min(stamp, by_source.get(key[0], stamp))
    cohorts = _cohort_counters()
    by_source = defaultdict(_cohort_counters)
    by_pool = {size: _cohort_counters() for size in POOL_SIZES}

    def count_sample(cohort, source, counts):
        size = counts["candidate_edges"]
        pool = POOL_SIZES[0 if size <= 3 else 1 if size <= 15 else 2]
        cohorts[cohort].update(counts)
        by_source[source][cohort].update(counts)
        by_pool[pool][cohort].update(counts)

    paired = Counter(dict(first_births=len(prepared.first), second_observed=0, second_missing=0,
                          second_timed_out=0, second_missing_incomplete_followup=0))
    events = []
    for key, snapshot in prepared.first.items():
        source = source_index[key[0]]
        first = _serialize_sample(snapshot, labels, anonymous, source_index, first_per_identity_source)
        counts = _event_counts(first)
        count_sample("first", source, counts)
        second = None
        changes = None
        if key in prepared.second:
            second = _serialize_sample(prepared.second[key], labels, anonymous, source_index, first_per_identity_source)
            second_counts = _event_counts(second)
            count_sample("second", source, second_counts)
            count_sample("first_with_second", source, counts)
            paired["second_observed"] += 1
            changes = _paired_counts(first, second, counts, second_counts, first_seen_by_index)
            paired.update(changes)
            status = "observed"
        elif key in prepared.timed_out:
            count_sample("first_timed_out_second", source, counts)
            paired["second_timed_out"] += 1
            status = "timeout"
        else:
            count_sample("first_missing_second", source, counts)
            paired["second_missing"] += 1
            paired["second_missing_incomplete_followup"] += prepared.native_stop - snapshot.stamp < HISTORY_S
            status = "missing"
        events.append(dict(fragment_index=anonymous[key], source_index=source, first=first, second=second,
                           second_status=status, paired_change=changes))
    events.sort(key=lambda event: event["fragment_index"])
    return dict(cohorts=_dump_cohorts(cohorts),
                by_source={source: _dump_cohorts(group) for source, group in sorted(by_source.items())},
                by_pool_size={size: _dump_cohorts(group) for size, group in by_pool.items()},
                paired=dict(paired), events=events)


def _pool(scenes):
    cohorts = _cohort_counters()
    by_source = defaultdict(_cohort_counters)
    by_pool = {size: _cohort_counters() for size in POOL_SIZES}
    paired = Counter()
    for scene in scenes:
        for cohort in COHORTS:
            cohorts[cohort].update(scene["cohorts"][cohort])
            for source, group in scene["by_source"].items():
                by_source[source][cohort].update(group[cohort])
            for size, group in scene["by_pool_size"].items():
                by_pool[size][cohort].update(group[cohort])
        paired.update(scene["paired"])
    return dict(cohorts=_dump_cohorts(cohorts), paired=dict(paired),
                by_source={source: _dump_cohorts(group) for source, group in sorted(by_source.items())},
                by_pool_size={size: _dump_cohorts(group) for size, group in by_pool.items()})


def _fit_summary(fit):
    return {field: fit[field] for field in (
        "status", "reason", "eligible_vectors", "epochs_with_votes", "occupied_bins",
        "peak_count", "next_lower_count", "tied_peak_bins")}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", nargs=3, type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(args.output)
    began = perf_counter()
    scene_to_stratum = {scene: stratum for stratum, members in ROSTER.items() for scene in members}
    prepared = {}
    for scene in scene_to_stratum:
        rows = read_archive_rows(args.archive, args.manifest, scene, required_split="T_core")
        prepared[scene] = prepare_support(event_batches(rows))

    scenes = {}
    for scene, snapshot in prepared.items():
        labels = read_labels(args.archive, args.manifest, scene)
        scenes[scene] = dict(score_support(snapshot, labels), query_start=0, query_stop=snapshot.stop,
                             last_native_time=snapshot.native_stop,
                             stratum=scene_to_stratum[scene], fit=_fit_summary(snapshot.fit))
    summary = dict(
        scene_ids=list(scenes), scenes=scenes, aggregate=_pool(scenes.values()),
        by_roster_stratum={stratum: _pool(scenes[scene] for scene in members)
                           for stratum, members in ROSTER.items()},
        gate_metres=RADIUS_M, fresh_seconds=FRESH_S, history_seconds=HISTORY_S,
        second_report_max_wait_seconds=HISTORY_S, neighbor_min_distinct_reports=MIN_NEIGHBOR_POINTS,
        fit_stop_seconds=FIT_STOP_S, clock="ideal_native_measurement_time",
        geometry="four-arm local plane and half-source registration shifts, HOLD positions",
        source_index="per-scene lexical source order, one-based anonymous role",
        nearest_ties="abstain; distance ranks retain exact ties", wall_seconds=perf_counter() - began,
    )
    with args.output.open("x", encoding="utf-8") as output:
        json.dump(summary, output, indent=2, allow_nan=False)
        output.write("\n")


if __name__ == "__main__":
    main()
