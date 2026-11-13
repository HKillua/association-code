"""Offset evidence from arrived positions and earlier geometry decisions."""
from __future__ import annotations

from bisect import bisect_right
from collections import deque
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from itertools import combinations
from math import dist
from statistics import median

from .causal_anchor_support import CONSISTENCY_M, Fragment, Pair
from .causal_common_motion import MAX_GAP_SECONDS, CommonMotion, History, Point, _TIME, _bracket, _point_at


WITNESS_AGE_SECONDS = 240


@dataclass(frozen=True)
class OffsetEvidence:
    past_drift_m: float | None
    held_residual_m: float | None


@dataclass(frozen=True)
class WitnessEvidence:
    count: int
    residual_m: float | None
    witnesses: tuple[Pair, ...]
    offsets: tuple[tuple[Pair, Point], ...] = field(default=(), compare=False)


@dataclass(frozen=True)
class _OffsetSample:
    observed_at: float
    common_stamp: float
    report_times: tuple[float, float]
    offset: Point


def _right_minus_left(left: Point, right: Point) -> Point:
    return right[0] - left[0], right[1] - left[1]


def _centre(points: Iterable[Point]) -> Point:
    east, north = zip(*points)
    return median(east), median(north)


def common_offset_evidence(common: CommonMotion) -> OffsetEvidence:
    """Use the first two offsets to score a distinct third co-timed point."""
    if not common.supported or common.left_points is None or common.right_points is None:
        return OffsetEvidence(None, None)
    first, second, held = (_right_minus_left(left, right)
                           for left, right in zip(common.left_points, common.right_points, strict=True))
    return OffsetEvidence(dist(first, second), dist(_centre((first, second)), held))


def _latest_common_sample(left: History, right: History, decision_time: float) -> _OffsetSample | None:
    left_count = bisect_right(left, decision_time, key=_TIME)
    right_count = bisect_right(right, decision_time, key=_TIME)
    if not left_count or not right_count:
        return None
    common_stamp = min(left[left_count - 1][0], right[right_count - 1][0])
    left_bracket = _bracket(left, left_count, common_stamp)
    right_bracket = _bracket(right, right_count, common_stamp)
    if left_bracket is None or right_bracket is None:
        return None
    report_times = left[left_bracket[1]][0], right[right_bracket[1]][0]
    if (report_times[0] - left[left_bracket[0]][0] > MAX_GAP_SECONDS
            or report_times[1] - right[right_bracket[0]][0] > MAX_GAP_SECONDS):
        return None
    left_point = _point_at(left, common_stamp, left_bracket)
    right_point = _point_at(right, common_stamp, right_bracket)
    return _OffsetSample(decision_time, common_stamp, report_times, _right_minus_left(left_point, right_point))


def current_common_offset(left_history: History, right_history: History, t: float) -> Point | None:
    """Return right minus left at the latest point with arrived brackets."""
    sample = _latest_common_sample(left_history, right_history, t)
    return sample.offset if sample is not None else None


class HistoricalGeometryWitnesses:
    """Keep recent co-timed offsets for G pairs in native replay order."""

    def __init__(self):
        self._samples: dict[Pair, deque[_OffsetSample]] = {}

    def record_sample(self, pair: Pair, sample: _OffsetSample) -> bool:
        """Record a co-timed sample when both original report bounds advanced."""
        # One extra slot allows a same-epoch observation to be excluded.
        history = self._samples.setdefault(pair, deque(maxlen=3))
        if history and not all(new > previous for new, previous in
                               zip(sample.report_times, history[-1].report_times, strict=True)):
            return False
        history.append(sample)
        return True

    def observe(self, t: float, geometry_links: Iterable[Pair],
                histories: Mapping[Fragment, History]) -> tuple[tuple[Pair, _OffsetSample], ...]:
        recorded = []
        for left, right in geometry_links:
            sample = _latest_common_sample(histories[left], histories[right], t)
            if sample is not None and self.record_sample((left, right), sample):
                recorded.append(((left, right), sample))
        return tuple(recorded)

    def lookup(self, pair: Pair, t: float, offset: Point | None) -> WitnessEvidence:
        """Compare with independently stable G pairs selected strictly earlier."""
        centres = {}
        sources = pair[0][0], pair[1][0]
        for witness, samples in self._samples.items():
            if (witness[0][0], witness[1][0]) != sources or any(key in pair for key in witness):
                continue
            recent = [sample.offset for sample in samples
                      if sample.observed_at < t and t - sample.common_stamp <= WITNESS_AGE_SECONDS]
            if len(recent) >= 2 and dist(recent[-2], recent[-1]) <= CONSISTENCY_M:
                centres[witness] = _centre(recent[-2:])

        agrees = centres and all(dist(first, second) <= CONSISTENCY_M
                                 for first, second in combinations(centres.values(), 2))
        residual = dist(offset, _centre(centres.values())) if offset is not None and agrees else None
        return WitnessEvidence(len(centres), residual, tuple(sorted(centres)), tuple(sorted(centres.items())))
