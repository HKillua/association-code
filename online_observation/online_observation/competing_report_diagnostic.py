"""Audit KF-selected episodes and frozen competing edges after native replay."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass, field
import gzip
import math
from pathlib import Path
import pickle
from time import perf_counter

from .continuous_metrics import evaluate_publications
from .continuous_tracker import FRESH_S, NULL_COST
from .evaluation_data import read_labels
from .nemot_episodes import _published, _truth
from .nemot_experiment import _events, _write_json, read_roster
from .nemot_replay import Edge, publications, replay


NULL = 2 * NULL_COST
WINDOWS = (60, 240)


def _freeze(events, path):
    with gzip.open(path, "xb", compresslevel=1) as stream:
        def record(stamp, updated, born, fresh, costs, memberships):
            selected = _published(memberships["kf"], fresh)
            frame = (stamp, tuple(sorted(updated)), tuple(sorted(born)), tuple(sorted(fresh)),
                     tuple(sorted(costs["kf"].items())), tuple(sorted(selected)))
            pickle.dump(frame, stream, protocol=pickle.HIGHEST_PROTOCOL)

        snapshot = replay(events, lambda *_: {}, observer=record)
        pickle.dump(snapshot, stream, protocol=pickle.HIGHEST_PROTOCOL)
    return snapshot.diagnostics, snapshot.fit["status"]


@dataclass
class Episode:
    start: float
    correct: bool
    first_report: bool
    sparse_first_report: bool
    onset_rank: str
    opportunities: tuple[str, ...]
    incumbent_cost: float | None
    alternatives: dict[Edge, float]
    relevant_keys: frozenset
    duration: float = 0.0
    end: str = ""
    relevant_delay: float | None = None
    comparable_delay: float | None = None
    first_entrant_delay: float | None = None
    first_entrant_lower_delay: float | None = None
    entrants: set = field(default_factory=set)
    reversal: dict | None = None
    events: int = 0
    relevant_events: int = 0
    relevant_reports: int = 0

    def close(self, stamp, reason):
        self.duration = stamp - self.start
        self.end = reason

    def outcome(self, horizon):
        if self.reversal is not None and self.reversal["delay_seconds"] <= horizon:
            return "reversal"
        if self.onset_rank != "eligible":
            return self.onset_rank
        return self.end if self.duration <= horizon else "window_elapsed"


def _incident(costs):
    result = defaultdict(dict)
    for edge, cost in costs.items():
        for key in edge:
            result[key][edge] = cost
    return result


def _alternative_costs(edge, incident, labels, correct):
    return {candidate: cost for key in edge for candidate, cost in incident[key].items()
            if (labels[candidate[0]] == labels[candidate[1]]) != correct}


def _opportunity(key, stamp, incident, latest_by_source, labels):
    true_cost = min((cost for (left, right), cost in incident[key].items()
                     if labels[left] == labels[right]), default=math.inf)
    if true_cost < math.inf:
        return "in_graph_below_null" if true_cost < NULL else "in_graph_at_or_above_null"
    other_reports = [time for source, time in latest_by_source[labels[key]].items() if source != key[0]]
    if any(stamp - time <= FRESH_S for time in other_reports):
        return "fresh_outside_graph"
    return "stale" if other_reports else "never_seen"


def _start(edge, stamp, born, fresh, costs, incident, latest_by_source, labels):
    correct = labels[edge[0]] == labels[edge[1]]
    original = {alt: cost for alt, cost in _alternative_costs(edge, incident, labels, correct).items()
                if cost < NULL}
    incumbent = costs.get(edge)
    if incumbent is None:
        rank = "incumbent_outside_graph_at_onset"
    elif incumbent >= NULL:
        rank = "incumbent_at_or_above_null_at_onset"
    elif not original:
        rank = "no_frozen_alternative"
    else:
        rank = "already_lower_at_onset" if min(original.values()) < incumbent else "eligible"

    first = born.intersection(edge)
    sparse_first = any(sum(peer[0] != key[0] for peer in fresh) <= 3 for key in first)
    opportunities = tuple(_opportunity(key, stamp, incident, latest_by_source, labels) for key in edge)
    relevant = frozenset(edge).union(key for alt in original for key in alt)
    return Episode(stamp, correct, bool(first), sparse_first, rank, opportunities, incumbent, original, relevant)


def _observe(episode, edge, stamp, updated, costs, selected, incident, labels):
    delay = stamp - episode.start
    if delay > WINDOWS[-1]:
        return
    incumbent = costs.get(edge, math.inf)
    old_reports = updated.intersection(edge)
    for alt, cost in _alternative_costs(edge, incident, labels, episode.correct).items():
        if alt in episode.alternatives or cost >= NULL or not (old_reports or updated.intersection(alt)):
            continue
        episode.entrants.add(alt)
        if episode.first_entrant_delay is None:
            episode.first_entrant_delay = delay
        if cost < incumbent < NULL and episode.first_entrant_lower_delay is None:
            episode.first_entrant_lower_delay = delay

    if episode.reversal is not None:
        return
    episode.events += 1
    reports = updated.intersection(episode.relevant_keys)
    if not reports:
        return
    episode.relevant_events += 1
    episode.relevant_reports += len(reports)
    if episode.relevant_delay is None:
        episode.relevant_delay = delay
    if episode.onset_rank != "eligible" or incumbent >= NULL:
        return

    comparable = {alt: costs[alt] for alt in episode.alternatives if costs.get(alt, math.inf) < NULL
                  and (old_reports or updated.intersection(alt))}
    if not comparable:
        return
    if episode.comparable_delay is None:
        episode.comparable_delay = delay
    lower = {alt: cost for alt, cost in comparable.items() if cost < incumbent}
    if lower:
        alternative_selected = bool(lower.keys() & selected)
        incumbent_selected = edge in selected
        actual = ("both" if incumbent_selected else "alternative_selected") if alternative_selected else (
            "incumbent_retained" if incumbent_selected else "other")
        episode.reversal = dict(
            delay_seconds=delay, native_events=episode.events, relevant_events=episode.relevant_events,
            relevant_reports=episode.relevant_reports, actual_selection=actual,
            incumbent_cost_m2=incumbent, minimum_alternative_cost_m2=min(lower.values()),
        )


def score_native(frames, labels):
    """Score one stream of already frozen post-decision frames."""
    counts = Counter()
    active, episodes, last_seen = {}, [], {}
    latest_by_source = defaultdict(dict)
    for stamp, report_keys, born_keys, fresh_keys, candidate_items, selected_pairs in frames:
        updated, born, fresh = set(report_keys), set(born_keys), set(fresh_keys)
        costs, selected = dict(candidate_items), set(selected_pairs)
        incident = _incident(costs)

        # A key refreshed after its deadline cannot bridge an expired episode.
        for edge, episode in list(active.items()):
            expiry = min(last_seen[key] + FRESH_S for key in edge)
            if expiry < stamp:
                episode.close(expiry, "expiry")
                episodes.append(active.pop(edge))
        for key in updated:
            last_seen[key] = stamp
            latest_by_source[labels[key]][key[0]] = stamp

        truth = _truth(fresh, labels)
        counts.update(native_events=1, original_reports=len(updated), born_reports=len(born),
                      candidate_edge_events=len(costs), truth_pair_events=len(truth),
                      correct_pair_events=len(truth & selected), missed_pair_events=len(truth - selected),
                      false_pair_events=len(selected - truth))
        for edge, episode in list(active.items()):
            _observe(episode, edge, stamp, updated, costs, selected, incident, labels)
            if edge not in selected:
                episode.close(stamp, "selection_change")
                episodes.append(active.pop(edge))
        for edge in sorted(selected - active.keys()):
            active[edge] = _start(edge, stamp, born, fresh, costs, incident, latest_by_source, labels)

    for episode in active.values():
        episode.close(stamp, "right_censored")
        episodes.append(episode)
    return counts, episodes


def _distribution(values):
    values = sorted(values)
    n = len(values)
    return dict(n=n, p50=values[math.ceil(.50 * n) - 1] if n else None,
                p95=values[math.ceil(.95 * n) - 1] if n else None, max=values[-1] if n else None)


def _by_window(episodes, window):
    reversed_episodes = [e.reversal for e in episodes if e.outcome(window) == "reversal"]
    eligible = [e for e in episodes if e.onset_rank == "eligible"]
    return dict(
        outcomes=Counter(e.outcome(window) for e in episodes),
        eligible_episodes=len(eligible),
        eligible_with_relevant_report=sum(e.relevant_delay is not None and e.relevant_delay <= window
                                          for e in eligible),
        eligible_with_comparable_costs=sum(e.comparable_delay is not None and e.comparable_delay <= window
                                           for e in eligible),
        episodes_with_entrant=sum(e.first_entrant_delay is not None and e.first_entrant_delay <= window
                                  for e in episodes),
        episodes_with_lower_entrant=sum(e.first_entrant_lower_delay is not None and e.first_entrant_lower_delay <= window
                                        for e in episodes),
        actual_selection=Counter(r["actual_selection"] for r in reversed_episodes),
        reversal={metric: _distribution(r[metric] for r in reversed_episodes)
                  for metric in ("delay_seconds", "native_events", "relevant_events", "relevant_reports")},
    )


def _episode_summary(episodes):
    long = [e for e in episodes if e.duration > FRESH_S]
    return dict(
        episodes=len(episodes), duration_seconds=_distribution(e.duration for e in episodes),
        pair_seconds=math.fsum(e.duration for e in episodes),
        over_60_seconds=dict(episodes=len(long), pair_seconds=math.fsum(e.duration for e in long)),
        ends=Counter(e.end for e in episodes), onset_ranks=Counter(e.onset_rank for e in episodes),
        onset_endpoint_opportunities=Counter(status for e in episodes for status in e.opportunities),
        onset_alternative_edges=sum(len(e.alternatives) for e in episodes),
        entrant_edges=sum(len(e.entrants) for e in episodes),
        windows={str(window): _by_window(episodes, window) for window in WINDOWS},
    )


def _strata(episodes):
    return {name: _episode_summary(members) for name, members in {
        "all": episodes, "first_report": [e for e in episodes if e.first_report],
        "sparse_first_report": [e for e in episodes if e.sparse_first_report],
        "continuation": [e for e in episodes if not e.first_report],
    }.items()}


def _episode_row(e):
    return dict(
        duration_seconds=e.duration, end=e.end, first_report=e.first_report,
        sparse_first_report=e.sparse_first_report, onset_rank=e.onset_rank,
        onset_opportunities=sorted(e.opportunities), onset_incumbent_cost_m2=e.incumbent_cost,
        onset_alternative_count=len(e.alternatives),
        onset_minimum_alternative_cost_m2=min(e.alternatives.values(), default=None),
        first_relevant_report_delay_seconds=e.relevant_delay, first_comparable_delay_seconds=e.comparable_delay,
        first_entrant_delay_seconds=e.first_entrant_delay, first_lower_entrant_delay_seconds=e.first_entrant_lower_delay,
        entrants=len(e.entrants), reversal=e.reversal,
        outcomes={str(window): e.outcome(window) for window in WINDOWS},
    )


def _run(archive, manifest, destination):
    members = read_roster(manifest)["splits"]["held"]
    if len(members) != 28:
        raise ValueError("this diagnostic requires the original 28-scene held roster")
    destination.mkdir(mode=0o700, exist_ok=False)
    spool = destination / "private_frozen"
    spool.mkdir(mode=0o700)
    began = perf_counter()
    records, scenes = [], {}
    for index, scene in enumerate(members, start=1):
        key = f"scene_{index:02d}"
        events = _events(archive, manifest, scene)
        path = spool / f"{key}.pkl.gz"
        geometry, fit_status = _freeze(events, path)
        births = geometry["first_total"]
        records.append((scene, key, path, len(events), geometry))
        scenes[key] = dict(sparse=2 * geometry["first_peers_le3"] >= births > 0,
                           fit_status=fit_status, native_events=len(events))
        print(f"Frozen {index}/{len(members)}, {len(events)} native events", flush=True)

    print("All native traces frozen. Reading A-B labels for diagnostic scoring.", flush=True)
    by_scene = {}
    for scene, key, path, event_count, geometry in records:
        labels = read_labels(archive, manifest, scene)
        with gzip.open(path, "rb") as stream:
            counts, episodes = score_native((pickle.load(stream) for _ in range(event_count)), labels)
            snapshot = pickle.load(stream)
        if (counts["native_events"], counts["original_reports"], counts["born_reports"], counts["candidate_edge_events"]) != (
                geometry["events"], geometry["reports"], geometry["first_total"], geometry["fixed_candidate_edge_events"]):
            raise RuntimeError(f"native diagnostic does not reconcile with the replay for {key}")
        stop = math.ceil(snapshot.native_reports[-1][0])
        grid = evaluate_publications(snapshot.native_reports, publications(snapshot, "kf", stop), labels)
        wrong = [e for e in episodes if not e.correct]
        correct = [e for e in episodes if e.correct]
        scenes[key].update(native_pairs=counts, grid_pair_seconds=grid["pairs"]["cross_source"],
                           wrong_episodes=_episode_summary(wrong), true_episodes=_episode_summary(correct),
                           wrong_episode_rows=[_episode_row(e) for e in wrong])
        by_scene[key] = wrong, correct

    strata = {
        "all": list(scenes),
        "sparse": [key for key, detail in scenes.items() if detail["sparse"]],
        "other": [key for key, detail in scenes.items() if not detail["sparse"]],
        "sparse_fitted": [key for key, detail in scenes.items() if detail["sparse"] and detail["fit_status"] == "fitted"],
        "sparse_abstained": [key for key, detail in scenes.items() if detail["sparse"] and detail["fit_status"] == "abstained"],
    }
    aggregate = {}
    for name, keys in strata.items():
        native = Counter()
        grid = Counter()
        for key in keys:
            native.update(scenes[key]["native_pairs"])
            grid.update(scenes[key]["grid_pair_seconds"])
        aggregate[name] = dict(
            scenes=len(keys), native_pairs=native, grid_pair_seconds=grid,
            wrong_episodes=_strata([e for key in keys for e in by_scene[key][0]]),
            true_episodes=_strata([e for key in keys for e in by_scene[key][1]]),
        )
    _write_json(destination / "summary.json", dict(
        scenes=scenes, strata=aggregate, training_updates=0, clock="ideal_native_measurement_time",
        freshness_seconds=FRESH_S, comparison_windows_seconds=WINDOWS, two_endpoint_null_cost_m2=NULL,
        notes=[
            "True selected episodes use onset false competitors as a symmetric descriptive comparison.",
            "Reversals compare one incident competitor at a time and require a new report on the incumbent or that competitor.",
            "A late entrant is counted separately if newly below null on a report affecting the incumbent or entrant.",
            "An ending native decision is included when the incumbent endpoints remained continuously fresh.",
            "The observation clock ends at the last native event; one-second pair counts use the original scoring clock.",
        ],
        wall_seconds=perf_counter() - began,
    ))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", nargs=3, type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    _run(args.archive, args.manifest, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
