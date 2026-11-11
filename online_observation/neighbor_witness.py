"""Construct leave-target-out position witnesses from the arrived prefix."""
from __future__ import annotations

from bisect import bisect_right
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from itertools import product
from math import dist

from .causal_anchor_support import Fragment, Pair, Point
from .causal_common_motion import MAX_GAP_SECONDS, History, _TIME, _bracket, _point_at


WITNESS_TOLERANCE_M = 400
NEIGHBOR_RADIUS_M = 8000
MAX_NEIGHBORS = 4
MAX_REPORT_AGE_SECONDS = 60
MAX_BRIDGE_SECONDS = 20


@dataclass(frozen=True, slots=True)
class NeighborVote:
    pair: Pair
    offset: Point


@dataclass(frozen=True, slots=True)
class NeighborFit:
    offset: Point
    witnesses: tuple[Pair, ...]
    assignment_ambiguous: bool

    @property
    def support(self) -> int:
        return len(self.witnesses)


@dataclass(frozen=True, slots=True)
class NeighborConsensus:
    best: NeighborFit | None
    competing: NeighborFit | None
    candidate: NeighborFit | None
    residual_m: float | None


@dataclass(frozen=True, slots=True)
class NeighborEvidence:
    epoch: float
    latest_epoch: float
    strict_offset: Point | None
    bridge_offset: Point | None
    nearby_counts: tuple[int, int]
    eligible_counts: tuple[int, int]
    cotimed_counts: tuple[int, int]
    votes: tuple[NeighborVote, ...]
    consensus: NeighborConsensus

    @property
    def candidate_offset(self) -> Point | None:
        return self.strict_offset if self.strict_offset is not None else self.bridge_offset

    @property
    def clock(self) -> str:
        if self.strict_offset is not None:
            return "strict"
        return "bridge" if self.bridge_offset is not None else "unavailable"

    @property
    def truncated(self) -> tuple[bool, bool]:
        left, right = self.eligible_counts
        return left > MAX_NEIGHBORS, right > MAX_NEIGHBORS


def _difference(right: Point, left: Point) -> Point:
    return right[0] - left[0], right[1] - left[1]


def _squared_distance(first: Point, second: Point) -> float:
    x, y = _difference(first, second)
    return x * x + y * y


def _support_at(votes: tuple[NeighborVote, ...], centre: Point) -> tuple[NeighborFit, float]:
    limit_sq = WITNESS_TOLERANCE_M ** 2
    eligible = []
    for vote in votes:
        error_sq = _squared_distance(vote.offset, centre)
        if error_sq <= limit_sq:
            eligible.append((vote.pair, error_sq))

    left_keys = {pair[0] for pair, _ in eligible}
    right_keys = {pair[1] for pair, _ in eligible}
    if len(left_keys) == len(right_keys) == len(eligible):
        return NeighborFit(centre, tuple(pair for pair, _ in eligible), False), sum(
            error_sq for _, error_sq in eligible)

    right_bits = {key: 1 << index for index, key in enumerate(sorted(right_keys))}
    by_left = {key: [] for key in sorted(left_keys)}
    for pair, cost in eligible:
        by_left[pair[0]].append((right_bits[pair[1]], cost, pair))

    # Retain the minimum-cost assignment and up to two feasible assignments
    # per used-column mask. The latter detects ambiguity at equal cardinality.
    best = {0: (0.0, ())}
    ways = {0: 1}
    for edges in by_left.values():
        next_best, next_ways = best.copy(), ways.copy()
        for used, (total, chosen) in best.items():
            for bit, cost, pair in edges:
                if used & bit:
                    continue
                mask = used | bit
                proposal = total + cost, (*chosen, pair)
                if mask not in next_best or proposal < next_best[mask]:
                    next_best[mask] = proposal
                next_ways[mask] = min(2, next_ways.get(mask, 0) + ways[used])
        best, ways = next_best, next_ways

    count = max(mask.bit_count() for mask in best)
    total, witnesses = min(state for mask, state in best.items() if mask.bit_count() == count)
    ambiguous = sum(ways[mask] for mask in ways if mask.bit_count() == count) > 1
    return NeighborFit(centre, witnesses, ambiguous), total


def summarize_votes(votes: Iterable[NeighborVote], candidate_offset: Point | None) -> NeighborConsensus:
    """Find independently generated offset hypotheses and a candidate check."""
    ordered = tuple(sorted(votes, key=lambda vote: vote.pair))
    centres = {vote.offset for vote in ordered}
    fits = {centre: _support_at(ordered, centre) for centre in centres}
    ranked = sorted(fits.values(), key=lambda row: (-row[0].support, row[1], row[0].offset, row[0].witnesses))
    best = ranked[0][0] if ranked else None
    competing = None if best is None else next(
        (fit for fit, _ in ranked[1:] if dist(fit.offset, best.offset) > WITNESS_TOLERANCE_M), None)
    candidate = None
    if candidate_offset is not None:
        candidate = (fits[candidate_offset] if candidate_offset in fits else _support_at(ordered, candidate_offset))[0]
    residual = None if best is None or candidate_offset is None else dist(best.offset, candidate_offset)
    return NeighborConsensus(best, competing, candidate, residual)


class NeighborWitnessIndex:
    """Per-decision positions with cached strictly co-timed neighbor snapshots."""

    def __init__(self, stamp: float, fresh: Iterable[Fragment], histories: Mapping[Fragment, History]):
        self.histories = histories
        self.arrived: dict[Fragment, int] = {}
        self.latest: dict[Fragment, tuple[float, Point]] = {}
        self._snapshots: dict[float, dict[Fragment, Point]] = {}
        for key in sorted(set(fresh)):
            history = histories[key]
            count = bisect_right(history, stamp, key=_TIME)
            if count and stamp - history[count - 1][0] <= MAX_REPORT_AGE_SECONDS:
                self.arrived[key] = count
                self.latest[key] = history[count - 1]

    def _snapshot(self, epoch: float) -> dict[Fragment, Point]:
        if epoch not in self._snapshots:
            positions = {}
            for key, count in self.arrived.items():
                history = self.histories[key]
                bracket = _bracket(history, count, epoch)
                if bracket is not None and history[bracket[1]][0] - history[bracket[0]][0] <= MAX_GAP_SECONDS:
                    positions[key] = _point_at(history, epoch, bracket)
            self._snapshots[epoch] = positions
        return self._snapshots[epoch]

    def _extrapolate(self, key: Fragment, epoch: float) -> Point | None:
        stamp, point = self.latest[key]
        age = epoch - stamp
        if age == 0:
            return point
        count = self.arrived[key]
        if age > MAX_BRIDGE_SECONDS or count < 2:
            return None
        previous_stamp, previous_point = self.histories[key][count - 2]
        interval = stamp - previous_stamp
        if not 0 < interval <= MAX_GAP_SECONDS:
            return None
        delta_x, delta_y = _difference(point, previous_point)
        return point[0] + delta_x * age / interval, point[1] + delta_y * age / interval

    def _neighbors(self, key: Fragment, snapshot: Mapping[Fragment, Point]) -> tuple[list[Fragment], int, int]:
        local = []
        nearby_count = 0
        origin = self.latest[key][1]
        limit_sq = NEIGHBOR_RADIUS_M ** 2
        for peer, (_, position) in self.latest.items():
            if peer == key or peer[0] != key[0]:
                continue
            nearby_count += _squared_distance(origin, position) <= limit_sq
            if peer in snapshot:
                distance_sq = _squared_distance(origin, snapshot[peer])
                if distance_sq <= limit_sq:
                    local.append((distance_sq, peer))
        chosen = [peer for _, peer in sorted(local)[:MAX_NEIGHBORS]]
        return chosen, nearby_count, len(local)

    def lookup(self, pair: Pair) -> NeighborEvidence:
        left, right = pair
        if left[0] >= right[0]:
            raise ValueError("candidate endpoints must use increasing source order")
        first, second = self.latest[left][0], self.latest[right][0]
        epoch, latest = min(first, second), max(first, second)
        snapshot = self._snapshot(epoch)
        strict = _difference(snapshot[right], snapshot[left]) if left in snapshot and right in snapshot else None
        bridged_left, bridged_right = self._extrapolate(left, latest), self._extrapolate(right, latest)
        bridge = None if bridged_left is None or bridged_right is None else _difference(bridged_right, bridged_left)
        left_neighbors, left_nearby, left_eligible = self._neighbors(left, snapshot)
        right_neighbors, right_nearby, right_eligible = self._neighbors(right, snapshot)
        votes = tuple(NeighborVote((a, b), _difference(snapshot[b], snapshot[a]))
                      for a, b in product(left_neighbors, right_neighbors))
        candidate_offset = strict if strict is not None else bridge
        return NeighborEvidence(epoch, latest, strict, bridge, (left_nearby, right_nearby),
                                (left_eligible, right_eligible), (len(left_neighbors), len(right_neighbors)),
                                votes, summarize_votes(votes, candidate_offset))
