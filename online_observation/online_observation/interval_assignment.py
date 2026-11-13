"""Score bounded partial matching changes with shared causal interval evidence."""
from __future__ import annotations

from bisect import bisect_left, bisect_right
from collections import Counter
from dataclasses import dataclass
from itertools import groupby
from math import fsum

from .continuous_tracker import FRESH_S, NULL_COST
from .nemot_replay import Edge, Fragment, Point


ARMS = ("point_swap", "interval_swap", "point_null", "interval_null")
HISTORY_SECONDS = 120.0
MIN_SPAN_SECONDS = 20.0
PAIR_NULL_COST = 2 * NULL_COST


@dataclass(frozen=True)
class Action:
    removed: tuple[Edge, Edge]
    added: tuple[Edge, ...]
    gain_m2: float
    complete: bool
    span_seconds: float


def _edge(left: Fragment, right: Fragment) -> Edge:
    return (left, right) if left < right else (right, left)


def _stamp(report):
    return report[0]


def _bracketed(history, start, stop):
    first = bisect_right(history, start, key=_stamp) - 1
    last = bisect_left(history, stop, key=_stamp)
    support = history[first:last + 1]
    if any(after[0] - before[0] > FRESH_S for before, after in zip(support, support[1:])):
        return None
    return support


def _coordinates(support, times, shift):
    """Interpolate on an interval entirely bracketed by arrived reports."""
    index = 0
    for time in times:
        while time > support[index + 1][0]:
            index += 1
        (t0, p0), (t1, p1) = support[index:index + 2]
        if time == t0:
            point = p0
        elif time == t1:
            point = p1
        else:
            fraction = (time - t0) / (t1 - t0)
            point = (p0[0] + fraction * (p1[0] - p0[0]), p0[1] + fraction * (p1[1] - p0[1]))
        yield point[0] + shift[0], point[1] + shift[1]


def _support(stamp, keys, histories, shifts):
    tracks = {key: histories.get(key, ()) for key in keys}
    if any(len(track) < 2 for track in tracks.values()):
        return None
    if any(track[-1][0] > stamp for track in tracks.values()):
        raise ValueError("interval assignment received a report after its decision time")

    start = max(stamp - HISTORY_SECONDS, *(track[0][0] for track in tracks.values()))
    stop = min(track[-1][0] for track in tracks.values())
    span = stop - start
    if span < MIN_SPAN_SECONDS:
        return None

    supports = {key: _bracketed(track, start, stop) for key, track in tracks.items()}
    if any(track is None for track in supports.values()):
        return None
    times = sorted({start, stop, *(t for track in supports.values() for t, _ in track if start < t < stop)})
    coordinates = {key: tuple(_coordinates(track, times, shifts.get(key[0], (0.0, 0.0))))
                   for key, track in supports.items()}
    return times, coordinates, span


def _dot(left: Point, right: Point) -> float:
    return left[0] * right[0] + left[1] * right[1]


def _costs(edge, times, coordinates, span):
    left, right = edge
    residuals = tuple((p[0] - q[0], p[1] - q[1]) for p, q in zip(coordinates[left], coordinates[right]))
    endpoint = _dot(residuals[-1], residuals[-1])
    # For a linear residual r(u) on [0,1], the mean of ||r||² is
    # (||r(0)||² + r(0)·r(1) + ||r(1)||²) / 3.
    integral = fsum((t1 - t0) * (_dot(r0, r0) + _dot(r0, r1) + _dot(r1, r1))
                    for t0, t1, r0, r1 in zip(times, times[1:], residuals, residuals[1:]))
    return endpoint, integral / (3.0 * span)


def _group_options(costs, selected):
    owner = {key: edge for edge in selected for key in edge}
    groups = set()
    for edge, cost in costs.items():
        if cost >= PAIR_NULL_COST or edge in selected:
            continue
        left, right = (owner.get(key) for key in edge)
        if left is None or right is None or left == right:
            continue
        groups.add(tuple(sorted((left, right))))

    for removed in sorted(groups):
        (a, b), (c, d) = removed
        crossed = tuple(sorted((_edge(a, d), _edge(c, b))))
        alternatives = tuple(edge for edge in crossed if edge in costs and costs[edge] < PAIR_NULL_COST)
        if alternatives:
            yield removed, alternatives


def _accept(actions):
    """Choose disjoint positive changes and abstain from conflicting ties."""
    used = set()
    accepted = []
    for _, tied in groupby(sorted(actions, key=lambda action: -action.gain_m2), key=lambda action: action.gain_m2):
        tied = tuple(tied)
        counts = Counter(edge for action in tied for edge in action.removed)
        for action in tied:
            if any(counts[edge] > 1 or edge in used for edge in action.removed):
                continue
            accepted.append(action)
            used.update(action.removed)
    return tuple(accepted)


def decide(stamp, updated, costs, selected, histories, shifts):
    """Return one bounded change set per arm on the same two-source KF graph.

    `updated` is the common event's set of newly reported fragments. The same
    eligible action can persist on other events while its support stays valid.
    Histories contain native arrivals only. They use one projected position
    system and receive the same source translations as the KF graph.
    """
    actions = {arm: [] for arm in ARMS}
    diagnostics = Counter()
    for removed, crossed in _group_options(costs, selected):
        complete = len(crossed) == 2
        diagnostics.update(groups_considered=1, complete_groups=int(complete),
                           incomplete_groups=int(not complete), structural_swap_actions=int(complete),
                           structural_null_actions=len(crossed))
        keys = tuple(key for edge in removed for key in edge)
        diagnostics["groups_with_new_report"] += bool(updated.intersection(keys))
        support = _support(stamp, keys, histories, shifts)
        if support is None:
            diagnostics["groups_insufficient"] += 1
            continue
        times, coordinates, span = support
        diagnostics.update(groups_eligible=1, eligible_swap_actions=int(complete),
                           eligible_null_actions=len(crossed))
        values = {edge: _costs(edge, times, coordinates, span) for edge in (*removed, *crossed)}

        options = [("null", (edge,)) for edge in crossed]
        if complete:
            options.append(("swap", crossed))
        for family, added in options:
            for index, model in enumerate(("point", "interval")):
                gain = fsum([*(values[edge][index] for edge in removed),
                             *(-values[edge][index] for edge in added),
                             -PAIR_NULL_COST * (len(removed) - len(added))])
                if gain > 0:
                    arm = f"{model}_{family}"
                    actions[arm].append(Action(removed, added, gain, complete, span))
    decisions = {arm: _accept(options) for arm, options in actions.items()}
    for arm in ARMS:
        diagnostics[f"{arm}_positive"] = len(actions[arm])
        diagnostics[f"{arm}_accepted"] = len(decisions[arm])
    return decisions, dict(diagnostics)
