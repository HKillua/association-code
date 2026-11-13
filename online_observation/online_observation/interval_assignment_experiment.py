"""Replay four interval assignment comparisons on the original causal graph."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import replace
import gzip
from itertools import count
import json
import math
from pathlib import Path
import pickle
from time import perf_counter, perf_counter_ns

from .continuous_metrics import Birth, evaluate_publications
from .continuous_tracker import FRESH_S, MEMORY_S, Plane
from .evaluation_data import read_labels
from .interval_assignment import ARMS, decide
from .nemot_episodes import _cross_pairs
from .nemot_experiment import _events, _write_json, _latencies
from .nemot_replay import (
    _birth_opportunities, _inherit_ids, _native_links, _native_summary,
    _registration_shifts, publications, replay,
)
from .registration_probe import FIT_STOP_S, estimate_translation


ROSTER = Path(__file__).with_name("interval_assignment_roster.json")
METHODS = ("kf", *ARMS)
COMPARISONS = (("kf", "point_swap"), ("kf", "interval_swap"), ("point_swap", "interval_swap"),
               ("kf", "point_null"), ("kf", "interval_null"), ("point_null", "interval_null"))


def _roster(manifest):
    roster = json.loads(ROSTER.read_text())
    meta = json.loads(manifest.read_text())
    train, held = (roster["splits"][role] for role in ("development", "held"))
    components = meta["components"]
    if (len(train) != 28 or len(held) != 48 or len(set(train + held)) != len(train + held)
            or not set(train + held) <= set(meta["splits"]["T_core"])
            or {components[s] for s in train} & {components[s] for s in held}
            or roster["components"] != {s: components[s] for s in train + held}):
        raise ValueError("frozen interval assignment roster disagrees with the development split")
    old = set()
    for name in roster["history_rosters"]:
        previous = json.loads(ROSTER.with_name(name).read_text())
        for ids in previous["splits"].values():
            old.update(ids)
        for key in ("historical_scene_ids", "prior_online_scene_ids"):
            old.update(previous.get("exclusions", {}).get(key, ()))
    if old.intersection(held):
        raise ValueError("a held scene was used by an earlier online study")
    return roster


class IntervalReplay:
    def __init__(self):
        self.histories = defaultdict(list)
        self.prefix = []
        self.fit = None
        self.shifts = {}
        self.plane = None
        self.first_seen = {}
        self.owners = {arm: {} for arm in ARMS}
        self.identifiers = {arm: count(1) for arm in ARMS}
        self.current = {arm: () for arm in ARMS}
        self.changes = {arm: [] for arm in ARMS}
        self.actions = {arm: [] for arm in ARMS}
        self.diagnostics = Counter()
        self.timing = []

    def ingest(self, events):
        for stamp, positions in events:
            if self.plane is None:
                self.plane = Plane.from_first_batch(positions.values())
            if stamp <= FIT_STOP_S:
                self.prefix.append((stamp, positions))
            elif self.fit is None:
                self.fit = estimate_translation(self.prefix)
                self.shifts = _registration_shifts(self.fit)
                self.prefix.clear()
            for key, point in positions.items():
                history = self.histories[key]
                if history and stamp - history[-1][0] > MEMORY_S:
                    for owners in self.owners.values():
                        owners.pop(key, None)
                history.append((stamp, self.plane.project(point)))
            yield stamp, positions

    def __call__(self, stamp, updated, born, fresh, costs, memberships):
        began = perf_counter_ns()
        self.first_seen.update(dict.fromkeys(born, stamp))
        if len({key[0] for key in fresh}) > 2:
            raise ValueError("this comparison requires two-source input")
        base = {edge for _, members in memberships["kf"] for edge in _cross_pairs(members)}
        decisions, diagnostics = decide(stamp, updated, costs["kf"], base, self.histories, self.shifts)
        self.diagnostics.update(diagnostics)
        for arm, actions in decisions.items():
            selected = set(base)
            for action in actions:
                selected.difference_update(action.removed)
                selected.update(action.added)
                self.actions[arm].append(action)
                if born.intersection(key for edge in (*action.removed, *action.added) for key in edge):
                    raise RuntimeError("an interval action changed a first report")
            matched = {key for edge in selected for key in edge}
            if len(matched) != 2 * len(selected) or not matched <= fresh:
                raise RuntimeError("an interval assignment duplicates or invents a fragment")
            groups = tuple(sorted(selected | {(key,) for key in fresh - matched}))
            prior = tuple((identifier, tuple(key for key in members if key in fresh and key in self.owners[arm]))
                          for identifier, members in self.current[arm])
            current = _inherit_ids(groups, prior, self.owners[arm], self.first_seen, self.identifiers[arm])
            if current != self.current[arm]:
                self.current[arm] = current
                self.changes[arm].append((stamp, current))
        self.timing.append(perf_counter_ns() - began)


def capture(events):
    observer = IntervalReplay()
    baseline = replay(observer.ingest(events), lambda *_: {}, observer=observer)
    if observer.fit is not None and observer.fit != baseline.fit:
        raise RuntimeError("the interval score did not use the baseline registration")
    result = replace(baseline, runs={"kf": baseline.runs["kf"], **observer.changes})
    return result, observer.actions, dict(observer.diagnostics), _latencies(observer.timing)


def _pair_spans(reports, changes):
    """Native published pair intervals clipped by original report freshness."""
    intervals = defaultdict(list)
    last, opened = {}, {}
    changes = iter(changes)
    next_change = next(changes, None)
    groups = ()
    for stamp, keys in reports:
        for pair, start in tuple(opened.items()):
            expiry = min(last[key] + FRESH_S for key in pair)
            if expiry < stamp:
                intervals[pair].append((start, expiry, False))
                del opened[pair]
        last.update(dict.fromkeys(keys, stamp))
        if next_change is not None and next_change[0] == stamp:
            groups = next_change[1]
            next_change = next(changes, None)
        pairs = {edge for _, members in groups for edge in _cross_pairs(members)
                 if all(stamp - last[key] <= FRESH_S for key in edge)}
        for pair in opened.keys() - pairs:
            intervals[pair].append((opened.pop(pair), stamp, False))
        for pair in pairs - opened.keys():
            opened[pair] = stamp
    for pair, start in opened.items():
        intervals[pair].append((start, stamp, True))
    return intervals


def _overlap(start, stop, candidates):
    return math.fsum(max(0.0, min(stop, other_stop) - max(start, other_start))
                     for other_start, other_stop, _ in candidates
                     if other_start < stop and other_stop > start)


def _retention(reference, other):
    total, kept, long_total, long_kept = [], [], [], []
    shortened = 0
    for edge, spans in reference.items():
        observed = other.get(edge, ())
        for start, stop, _ in spans:
            duration = stop - start
            overlap = _overlap(start, stop, observed)
            total.append(duration)
            kept.append(overlap)
            if duration > 60:
                long_total.append(duration)
                long_kept.append(overlap)
                shortened += overlap < duration - 1e-8
    return dict(pair_seconds=math.fsum(total), retained_seconds=math.fsum(kept),
                long_episodes=len(long_total), long_seconds=math.fsum(long_total),
                retained_long_seconds=math.fsum(long_kept), shortened_long_episodes=shortened)


def _spans_summary(spans):
    intervals = [record for records in spans.values() for record in records]
    lengths = sorted(stop - start for start, stop, _ in intervals)
    n = len(lengths)
    return dict(episodes=n, distinct_edges=len(spans), seconds=math.fsum(lengths),
                over60_episodes=sum(length > 60 for length in lengths),
                over60_seconds=math.fsum(length for length in lengths if length > 60),
                right_censored=sum(censored for _, _, censored in intervals),
                p95_seconds=lengths[math.ceil(.95 * n) - 1] if n else None,
                max_seconds=lengths[-1] if n else None)


def _action_counts(actions, is_true):
    def count_actions(members):
        total = Counter()
        unique = set()
        for action in members:
            old = sum(is_true(edge) for edge in action.removed)
            new = sum(is_true(edge) for edge in action.added)
            total.update(actions=1, removed_true=old, added_true=new, removed_false=2-old,
                         added_false=len(action.added)-new, improve_true=new > old, lose_true=new < old,
                         all_false_to_all_true=old == 0 and new == len(action.added))
            unique.add((action.removed, action.added))
        return dict(total) | {"distinct_actions": len(unique)}

    return dict(all=count_actions(actions),
                complete=count_actions(action for action in actions if action.complete),
                incomplete=count_actions(action for action in actions if not action.complete))


def score(result, actions, labels):
    def is_true(pair):
        return labels[pair[0]] == labels[pair[1]]

    arms, assignments, births_by_arm, first_by_arm = {}, {}, {}, {}
    stop = math.ceil(result.native_reports[-1][0])
    opportunities = _birth_opportunities(result, labels)
    for arm in METHODS:
        spans = _pair_spans(result.native_reports, result.runs[arm])
        good = {pair: spans[pair] for pair in spans if is_true(pair)}
        bad = {pair: spans[pair] for pair in spans if not is_true(pair)}
        assignments[arm] = good, bad
        grid = evaluate_publications(result.native_reports, publications(result, arm, stop), labels)
        births = {key: Birth(first.stamp, opportunities[key]) for key, first in result.first_candidates.items()}
        first_by_arm[arm] = _native_links(result.runs[arm], births, labels)
        births_by_arm[arm] = births
        arms[arm] = dict(pairs=grid["pairs"]["cross_source"], identity=grid["counts"],
                         wrong=_spans_summary(bad), birth=_native_summary(list(births.values())),
                         sparse_birth=_native_summary([births[key] for key, first in result.first_candidates.items()
                                                       if first.peer_count <= 3]))
        if arm != "kf":
            arms[arm]["actions"] = _action_counts(actions[arm], is_true)
    if any(first_by_arm[arm] != first_by_arm["kf"] for arm in ARMS):
        raise RuntimeError("first native report decisions were not preserved")

    comparisons = {}
    for before, after in COMPARISONS:
        old_good, old_bad = assignments[before]
        new_good, new_bad = assignments[after]
        delay = Counter()
        for key, old in births_by_arm[before].items():
            earlier = old.first_correct_delay
            later = births_by_arm[after][key].first_correct_delay
            delay["all"] += 1
            for limit in (60, 240):
                delay[f"before_confirmed_by_{limit}"] += earlier is not None and earlier <= limit
                delay[f"after_confirmed_by_{limit}"] += later is not None and later <= limit
            if earlier is not None and later is not None:
                delay["both_confirmed"] += 1
                delay["earlier"] += later < earlier
                delay["later"] += later > earlier
        comparisons[f"{after}_vs_{before}"] = dict(
            baseline_true=_retention(old_good, new_good), baseline_false=_retention(old_bad, new_bad),
            candidate_false=_retention(new_bad, old_bad),
            original_kf_false=_retention(assignments["kf"][1], new_bad), confirmation=dict(delay),
        )
    return arms, comparisons


def _run(archive, manifest, phase, output):
    roster = _roster(manifest)
    output.mkdir(mode=0o700, exist_ok=False)
    frozen = output / "private_frozen"
    frozen.mkdir(mode=0o700)
    began = perf_counter()
    scenes = {}
    for scene in roster["splits"][phase]:
        events = _events(archive, manifest, scene)
        result, actions, diagnostics, timing = capture(events)
        if timing["n"] != len(events):
            raise RuntimeError("replay and scoring clocks differ")
        with gzip.open(frozen / f"{scene}.pkl.gz", "xb", compresslevel=1) as stream:
            pickle.dump((result, actions), stream, protocol=pickle.HIGHEST_PROTOCOL)
        births = result.diagnostics["first_total"]
        scenes[scene] = dict(component=roster["components"][scene],
                             sparse=2 * result.diagnostics["first_peers_le3"] >= births > 0,
                             fit_status=result.fit["status"], reports=result.diagnostics["reports"],
                             events=len(events), candidate_edge_events=result.diagnostics["fixed_candidate_edge_events"],
                             diagnostics=diagnostics, four_arm_extra_timing=timing)
        print(f"Frozen {len(scenes)}/{len(roster['splits'][phase])}, {len(events)} events", flush=True)
    print("All assignments frozen. Reading evaluation identities.", flush=True)
    for scene, entry in scenes.items():
        with gzip.open(frozen / f"{scene}.pkl.gz", "rb") as stream:
            result, actions = pickle.load(stream)
        entry["arms"], entry["comparisons"] = score(result, actions, read_labels(archive, manifest, scene))
    _write_json(output / "summary.json", dict(phase=phase, methods=METHODS, scenes=scenes,
        training_updates=0, clock="ideal_native_measurement_time", first_report_partners_equal=True,
        publication_delay_added_seconds=0, wall_seconds=perf_counter() - began))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, nargs=3, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--phase", choices=("development", "held"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    _run(args.archive, args.manifest, args.phase, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
