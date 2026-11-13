"""Compare two past displacement intervals on the same observed clock."""
from __future__ import annotations

from bisect import bisect_left, bisect_right
from collections.abc import Sequence
from dataclasses import dataclass
from math import hypot, sqrt
from operator import itemgetter


Point = tuple[float, float]
History = Sequence[tuple[float, Point]]
WindowPoints = tuple[Point, Point, Point]

SEGMENT_SECONDS = 60
MAX_GAP_SECONDS = 45
MIN_PATH_METRES = 240
MIN_CHANGE_METRES = 120
MAX_SEGMENT_ERROR_METRES = 200
MAX_CHANGE_ERROR_METRES = 120
_TIME = itemgetter(0)


@dataclass(frozen=True)
class CommonMotion:
    """Observed evidence at one decision time, including its report availability."""

    status: str
    interval_end: float | None = None
    available_at: float | None = None
    left_points: WindowPoints | None = None
    right_points: WindowPoints | None = None
    score_m: float | None = None

    @property
    def supported(self) -> bool:
        return self.score_m is not None

    @property
    def qualified(self) -> bool:
        return self.status == "qualified"


def _bracket(history: History, arrived: int, stamp: float) -> tuple[int, int] | None:
    upper = bisect_left(history, stamp, hi=arrived, key=_TIME)
    if upper == arrived:
        return None
    if history[upper][0] == stamp:
        return upper, upper
    return (upper - 1, upper) if upper else None


def _point_at(history: History, stamp: float, bracket: tuple[int, int]) -> Point:
    lower, upper = bracket
    earlier, (east, north) = history[lower]
    if lower == upper:
        return east, north
    later, (other_east, other_north) = history[upper]
    fraction = (stamp - earlier) / (later - earlier)
    return east + fraction * (other_east - east), north + fraction * (other_north - north)


def _window(history: History, arrived: int, stamps: tuple[float, float, float]):
    start = _bracket(history, arrived, stamps[0])
    middle = _bracket(history, arrived, stamps[1])
    end = _bracket(history, arrived, stamps[2])
    if start is None or middle is None or end is None:
        return "missing_window", None, None
    if any(history[index + 1][0] - history[index][0] > MAX_GAP_SECONDS
           for index in range(start[0], end[1])):
        return "gap_too_wide", None, None
    points = (_point_at(history, stamps[0], start), _point_at(history, stamps[1], middle),
              _point_at(history, stamps[2], end))
    return "supported", points, history[end[1]][0]


def _difference(left: Point, right: Point) -> Point:
    return left[0] - right[0], left[1] - right[1]


def compare_common_motion(left_history: History, right_history: History, decision_time: float) -> CommonMotion:
    """Use strictly time-ordered local histories in a shared planar metre frame.

    The two 60-second intervals end at the earlier latest arrival. All
    interpolating coordinates and adjacent-gap checks use reports that have
    arrived by the decision. A time-constant source translation cancels from
    each displacement, while uninformative or inconsistent motion abstains.
    """
    left_arrived = bisect_right(left_history, decision_time, key=_TIME)
    right_arrived = bisect_right(right_history, decision_time, key=_TIME)
    if not left_arrived or not right_arrived:
        return CommonMotion("missing_window")
    end = min(left_history[left_arrived - 1][0], right_history[right_arrived - 1][0])
    stamps = end - 2 * SEGMENT_SECONDS, end - SEGMENT_SECONDS, end
    left_status, left, left_available = _window(left_history, left_arrived, stamps)
    right_status, right, right_available = _window(right_history, right_arrived, stamps)
    if left is None or right is None:
        statuses = left_status, right_status
        status = "missing_window" if "missing_window" in statuses else "gap_too_wide"
        return CommonMotion(status, interval_end=end)

    left_first, left_second = _difference(left[1], left[0]), _difference(left[2], left[1])
    right_first, right_second = _difference(right[1], right[0]), _difference(right[2], right[1])
    left_change = _difference(left_second, left_first)
    right_change = _difference(right_second, right_first)
    first_error = hypot(*_difference(left_first, right_first))
    second_error = hypot(*_difference(left_second, right_second))
    change_error = hypot(*_difference(left_change, right_change))
    if min(hypot(*left_first) + hypot(*left_second), hypot(*right_first) + hypot(*right_second)) < MIN_PATH_METRES:
        status = "low_motion"
    elif min(hypot(*left_change), hypot(*right_change)) < MIN_CHANGE_METRES:
        status = "degenerate"
    elif max(first_error, second_error) > MAX_SEGMENT_ERROR_METRES or change_error > MAX_CHANGE_ERROR_METRES:
        status = "mismatch"
    else:
        status = "qualified"
    return CommonMotion(status, end, max(left_available, right_available), left, right,
                        sqrt((first_error**2 + second_error**2) / 2))
