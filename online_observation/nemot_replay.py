"""Compare motion predictions on one event-time association graph."""
from __future__ import annotations

from array import array
from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import combinations, count
import math
from time import perf_counter_ns

from .continuous_assignment import solve_global
from .continuous_metrics import Birth, OPPORTUNITIES, _birth_summary
from .continuous_tracker import FRESH_S, MEMORY_S, NULL_COST, RADIUS_M, Plane
from .nemot_inputs import MotionInputs
from .registration_probe import FIT_STOP_S, estimate_translation


ARMS = ("kf", "history", "neighbors")
GATE_SQUARED = RADIUS_M**2
Fragment = tuple[str, str]
Point = tuple[float, float]
Edge = tuple[Fragment, Fragment]
Memberships = tuple[tuple[str, tuple[Fragment, ...]], ...]


@dataclass(frozen=True)
class FirstCandidate:
    stamp: float
    peer_count: int
    peers: tuple[Fragment, ...]
    nearest: tuple[Fragment | None, ...]


@dataclass(frozen=True)
class ReplayResult:
    runs: dict[str, list[tuple[float, Memberships]]]
    native_reports: list[tuple[float, tuple[Fragment, ...]]]
    first_candidates: dict[Fragment, FirstCandidate]
    diagnostics: dict
    fit: dict
    timing: dict


def _distance_sq(left: Point, right: Point) -> float:
    return (left[0] - right[0]) ** 2 + (left[1] - right[1]) ** 2


def _add(left: Point, right: Point) -> Point:
    return left[0] + right[0], left[1] + right[1]


def _registration_shifts(fit):
    shift = fit["translation_east_north_m"]
    if shift is None:
        return {}
    left, right = fit["source_order"]
    east, north = (value / 2 for value in shift)
    return {left: (east, north), right: (-east, -north)}


def _other(edge: Edge, key: Fragment) -> Fragment:
    return edge[1] if edge[0] == key else edge[0]


def _fixed_changes(fresh, updated, tracks, shifts, old_edges, reset):
    """Update the HOLD graph; no prediction is consulted here."""
    removed = {edge for edge in old_edges if edge[0] not in fresh or edge[1] not in fresh}
    changed = set()
    frontier = fresh if reset else updated
    positions = {key: _add(tracks[key].observation, shifts.get(key[0], (0.0, 0.0))) for key in fresh}
    for key in frontier:
        for peer in fresh:
            if peer[0] == key[0]:
                continue
            edge = (key, peer) if key < peer else (peer, key)
            if _distance_sq(positions[key], positions[peer]) <= GATE_SQUARED:
                changed.add(edge)
            elif edge in old_edges:
                removed.add(edge)
    return removed, changed


def _solve_groups(costs, fresh):
    """Solve each source pair and discard connected groups with a repeated source."""
    source_pairs = defaultdict(dict)
    for edge, cost in costs.items():
        source_pairs[edge[0][0], edge[1][0]][edge] = cost
    connections = defaultdict(set)
    for pair_costs in source_pairs.values():
        rows = sorted({left for left, _ in pair_costs})
        cols = sorted({right for _, right in pair_costs})
        matrix = [[pair_costs.get((left, right)) for right in cols] for left in rows]
        solution = solve_global(matrix, (NULL_COST,) * len(rows), (NULL_COST,) * len(cols))
        for row, col in solution.pairs:
            left, right = rows[row], cols[col]
            connections[left].add(right)
            connections[right].add(left)
    groups = []
    pending = set(fresh)
    for key in sorted(fresh):
        if key not in pending:
            continue
        members = [key]
        pending.remove(key)
        for member in members:
            for peer in connections[member] & pending:
                pending.remove(peer)
                members.append(peer)
        if len({member[0] for member in members}) == len(members):
            groups.append(tuple(sorted(members)))
        else:
            groups.extend((member,) for member in sorted(members))
    return tuple(sorted(groups))


def _inherit_ids(groups, old_groups, owners, first_seen, identifiers):
    """Preserve unchanged groups, then inherit IDs without duplication."""
    fixed = {members: identifier for identifier, members in old_groups}
    assigned = {group: fixed[group] for group in groups if group in fixed}
    remaining = set(assigned.values())
    overlaps = {group: Counter(owners[key] for key in group if key in owners)
                for group in groups if group not in assigned}

    def priority(group):
        return min(first_seen[key] for key in group), group

    # When membership changes, prefer preserving multiple shared members.
    multi = sorted((-size, int(identifier), priority(group), group)
                   for group, candidates in overlaps.items() for identifier, size in candidates.items()
                   if size > 1)
    for _, identifier_number, _, group in multi:
        identifier = str(identifier_number)
        if group not in assigned and identifier not in remaining:
            assigned[group] = identifier
            remaining.add(identifier)

    # Maximize the remaining one-member reuses. Consider earlier IDs first
    # and use an augmenting path only when no free ID is available directly.
    eligible = {group: tuple(sorted(set(candidates) - remaining, key=int))
                for group, candidates in overlaps.items() if group not in assigned}
    inheritors = {}

    def inherit(group, seen):
        for identifier in eligible[group]:
            if identifier not in seen and identifier not in inheritors:
                inheritors[identifier] = group
                return True
        for identifier in eligible[group]:
            if identifier in seen:
                continue
            seen.add(identifier)
            if inherit(inheritors[identifier], seen):
                inheritors[identifier] = group
                return True
        return False

    for group in sorted(eligible, key=priority):
        inherit(group, set())
    assigned.update((group, identifier) for identifier, group in inheritors.items())
    for group in sorted(groups, key=priority):
        if group not in assigned:
            assigned[group] = str(next(identifiers))
    for group, identifier in assigned.items():
        owners.update(dict.fromkeys(group, identifier))
    return tuple(sorted(((identifier, group) for group, identifier in assigned.items()),
                        key=lambda item: int(item[0])))


def _nearest(key, edges, costs):
    if not edges:
        return None
    best = min(costs[edge] for edge in edges)
    winners = [edge for edge in edges if costs[edge] == best]
    return _other(winners[0], key) if len(winners) == 1 else None


def replay(events, predict_learned, *, fit=None):
    """Apply three scorers to one geometry graph and record native-time changes.

    `predict_learned(context, stamp, keys)` returns residuals in metres for
    each learned arm. A missing residual leaves the shared KF center intact.
    """
    context = MotionInputs()
    plane = None
    prefix = []
    shifts = {}
    activated = False
    previous = -math.inf
    fresh = set()
    old_edges = set()
    costs = {arm: {} for arm in ARMS}
    owners = {arm: {} for arm in ARMS}
    identifiers = {arm: count(1) for arm in ARMS}
    memberships = {arm: () for arm in ARMS}
    runs = {arm: [] for arm in ARMS}
    native_reports = []
    first_seen = {}
    first_candidates = {}
    counters = Counter()
    event_time, prediction_time = array("Q"), array("Q")
    solve_time = {arm: array("Q") for arm in ARMS}
    for stamp, positions in events:
        began = perf_counter_ns()
        if not math.isfinite(stamp) or stamp < 0 or stamp <= previous or not positions:
            raise ValueError("native batches must be nonempty and strictly increasing from time zero")
        previous = stamp
        if plane is None:
            plane = Plane.from_first_batch(positions.values())
        if stamp <= FIT_STOP_S and fit is None:
            prefix.append((stamp, positions))
        reset = stamp > FIT_STOP_S and not activated
        if reset:
            fit = estimate_translation(prefix) if fit is None else fit
            shifts = _registration_shifts(fit)
            activated = True
            prefix.clear()

        updated = positions.keys()
        restarted = {key for key in updated & context.tracks.keys()
                     if stamp - context.tracks[key].stamp > MEMORY_S}
        for key in restarted:
            for arm in ARMS:
                owners[arm].pop(key, None)
        context.observe(stamp, {key: plane.project(positions[key]) for key in sorted(positions)})
        tracks = context.tracks
        old_fresh = fresh
        fresh = {key for key in old_fresh if stamp - tracks[key].stamp <= FRESH_S} | updated
        removed, changed = _fixed_changes(fresh, updated, tracks, shifts, old_edges, reset)
        old_edges.difference_update(removed)
        old_edges.update(changed)
        endpoints = {key for edge in changed for key in edge}
        needed = tuple(sorted(key for key in endpoints if tracks[key].stamp < stamp))
        centers = {key: _add(tracks[key].position(stamp, True), shifts.get(key[0], (0.0, 0.0)))
                   for key in endpoints}
        prediction_began = perf_counter_ns()
        residuals = predict_learned(context, stamp, needed) if needed else {}
        prediction_time.append(perf_counter_ns() - prediction_began if needed else 0)
        counters.update(events=1, reports=len(updated), fixed_candidate_edge_events=len(old_edges),
                        rescored_edge_events=len(changed), requested_predictions=len(needed),
                        prediction_events=bool(needed), nonempty_candidate_events=bool(old_edges))
        counters["peak_candidate_edges"] = max(counters["peak_candidate_edges"], len(old_edges))

        born = sorted(updated - first_seen.keys())
        first_seen.update(dict.fromkeys(born, stamp))
        changed_groups = old_fresh != fresh or removed or changed or restarted
        for arm in ARMS:
            arm_began = perf_counter_ns()
            scores = costs[arm]
            for edge in removed:
                del scores[edge]
            correction = residuals.get(arm, {}) if arm != "kf" else {}
            predictions = dict(centers)
            for key in needed:
                if key in correction:
                    predictions[key] = _add(centers[key], correction[key])
            scores.update((edge, _distance_sq(predictions[edge[0]], predictions[edge[1]])) for edge in changed)
            if changed_groups:
                groups = _solve_groups(scores, fresh)
                prior = tuple((identifier, tuple(key for key in members if key in fresh and key in owners[arm]))
                              for identifier, members in memberships[arm])
                current = _inherit_ids(groups, prior, owners[arm], first_seen, identifiers[arm])
                if current != memberships[arm]:
                    memberships[arm] = current
                    runs[arm].append((stamp, current))
            solve_time[arm].append(perf_counter_ns() - arm_began)

        for key in born:
            edges = sorted(edge for edge in old_edges if key in edge)
            peer_count = sum(peer[0] != key[0] for peer in fresh)
            first_candidates[key] = FirstCandidate(
                stamp, peer_count, tuple(_other(edge, key) for edge in edges),
                tuple(_nearest(key, edges, costs[arm]) for arm in ARMS),
            )
            counters.update(first_total=1, first_peers_le3=peer_count <= 3,
                            first_gate_le3=len(edges) <= 3, first_empty_gate=not edges,
                            first_peer_total=peer_count, first_gate_edges=len(edges))
        native_reports.append((stamp, tuple(sorted(updated))))
        event_time.append(perf_counter_ns() - began)

    if fit is None:
        fit = estimate_translation(prefix)
    diagnostics = dict(counters)
    timing = dict(event_service_ns=event_time, two_model_prediction_ns=prediction_time,
                  arm_cost_and_assignment_ns=solve_time)
    return ReplayResult(runs, native_reports, first_candidates, diagnostics, fit, timing)


def publications(result, arm, stop):
    """Sample already produced native states on the fixed integer scoring grid."""
    reports = iter(result.native_reports)
    changes = iter(result.runs[arm])
    current_report = next(reports, None)
    current_change = next(changes, None)
    latest = {}
    groups = ()
    for tick in range(stop + 1):
        while current_report is not None and current_report[0] <= tick:
            stamp, keys = current_report
            latest.update(dict.fromkeys(keys, stamp))
            current_report = next(reports, None)
        while current_change is not None and current_change[0] <= tick:
            _, groups = current_change
            current_change = next(changes, None)
        tracks = []
        for identifier, members in groups:
            fresh_members = tuple(key for key in members if tick - latest[key] <= FRESH_S)
            if fresh_members:
                tracks.append(dict(track_id=identifier, members=fresh_members))
        yield dict(query_time=tick, tracks=tracks)


def _birth_opportunities(result, labels):
    latest = defaultdict(dict)
    opportunities = {}
    for stamp, updated in result.native_reports:
        for key in updated:
            latest[labels[key]][key[0]] = stamp
        for key in updated:
            if result.first_candidates[key].stamp != stamp:
                continue
            ages = [stamp - seen for source, seen in latest[labels[key]].items() if source != key[0]]
            if any(age <= FRESH_S for age in ages):
                opportunities[key] = "fresh"
            else:
                opportunities[key] = "stale" if ages else "never_observed"
    return opportunities


def _native_links(runs, births, labels):
    first = {}
    for stamp, memberships in runs:
        for _, members in memberships:
            for key in members:
                peers = tuple(other for other in members if other[0] != key[0])
                record = births[key]
                true = any(labels[peer] == labels[key] for peer in peers)
                false = any(labels[peer] != labels[key] for peer in peers)
                if stamp == record.stamp:
                    record.first_decision = "incorrect" if false else "correct" if true else "rejected"
                    first[key] = peers
                if true and not false and record.first_correct_delay is None:
                    record.first_correct_delay = stamp - record.stamp
    if any(record.first_decision is None for record in births.values()):
        raise ValueError("a first native fragment was missing from its event output")
    return first


def _candidate_scores(result, births, picked, labels, arm_index):
    by_pool = {"0-3": Counter(), "4+": Counter()}
    by_gate = {"0-3": Counter(), "4+": Counter()}
    all_counts = Counter()
    for key, candidate in result.first_candidates.items():
        peers = candidate.peers
        nearest = candidate.nearest[arm_index]
        record = births[key]
        decision = record.first_decision
        true_in_gate = any(labels[peer] == labels[key] for peer in peers)
        tallies = dict(
            births=1, candidate_edges=len(peers), true_reachable=true_in_gate,
            no_candidate=not peers, nearest_correct=nearest is not None and labels[nearest] == labels[key],
            nearest_wrong=nearest is not None and labels[nearest] != labels[key],
            nearest_tied=nearest is None and bool(peers),
            first_correct=decision == "correct", first_wrong=decision == "incorrect",
            first_rejected=decision == "rejected", first_wrong_when_true_reachable=decision == "incorrect" and true_in_gate,
            first_rejected_when_true_reachable=decision == "rejected" and true_in_gate,
            first_wrong_without_opposite_history=decision == "incorrect" and record.opportunity == "never_observed",
            first_sole_false_without_opposite_history=(decision == "incorrect" and len(peers) == 1
                                                       and record.opportunity == "never_observed"),
            selected_candidate_edges=sum(peer in peers for peer in picked[key]),
        )
        all_counts.update(tallies)
        by_pool["0-3" if candidate.peer_count <= 3 else "4+"].update(tallies)
        by_gate["0-3" if len(peers) <= 3 else "4+"].update(tallies)
    return dict(all_counts) | {"by_first_fresh_peer_count": {k: dict(v) for k, v in by_pool.items()},
                               "by_first_gate_peer_count": {k: dict(v) for k, v in by_gate.items()}}


def score_native_births(result, labels):
    """Score exact-time confirmation and first choices after replay is frozen."""
    if result.first_candidates.keys() - labels.keys():
        raise ValueError("unmapped measurement fragment in post-replay labels")
    opportunities = _birth_opportunities(result, labels)
    scores = {}
    decisions = {}
    for index, arm in enumerate(ARMS):
        births = {key: Birth(candidate.stamp, opportunities[key]) for key, candidate in result.first_candidates.items()}
        picked = _native_links(result.runs[arm], births, labels)
        decisions[arm] = (births, picked)
        summary = _native_summary(list(births.values()))
        summary["by_opportunity"] = {name: _native_summary([birth for birth in births.values()
                                                           if birth.opportunity == name])
                                     for name in OPPORTUNITIES}
        scores[arm] = dict(births=summary, candidates=_candidate_scores(result, births, picked, labels, index))
    for before, after in combinations(ARMS, 2):
        old, old_picked = decisions[before]
        new, new_picked = decisions[after]
        transitions = Counter(f"{old[key].first_decision}_to_{new[key].first_decision}" for key in old)
        transitions["changed_selected_partner"] = sum(old_picked[key] != new_picked[key] for key in old)
        scores[after].setdefault("first_choice_comparisons", {})[before] = dict(transitions)
    return scores


def _native_summary(births):
    summary = _birth_summary(births)
    histogram = Counter(round(1000 * birth.first_correct_delay) for birth in births
                        if birth.first_correct_delay is not None)
    summary["delay_milliseconds_histogram"] = dict(sorted(histogram.items()))
    return summary
