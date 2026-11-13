"""Measure continuity of published alternatives just before a native report."""
from __future__ import annotations

from bisect import bisect_left
from operator import itemgetter

from .interval_assignment_experiment import _pair_spans


_START = itemgetter(0)
_RUNS = {"kf": "kf", "i22": "interval_swap"}


class RejoinPersistence:
    """Index original online pair spans once for a frozen scene."""

    def __init__(self, result):
        self.spans = {arm: _pair_spans(result.native_reports, result.runs[run])
                      for arm, run in _RUNS.items()}

    def duration(self, arm, edges, stamp):
        """Return seconds the complete matching was continuously published up to t−.

        The underlying spans use the original freshness rule. An edge opened by
        the batch at t cannot count, and an edge closed by that batch still
        counts up to t. Only times within the original native replay are queried.
        """
        starts = []
        for edge in edges:
            spans = self.spans[arm].get(edge, ())
            index = bisect_left(spans, stamp, key=_START)
            if index == 0 or spans[index - 1][1] < stamp:
                return 0.0
            starts.append(spans[index - 1][0])
        return stamp - max(starts) if starts else 0.0

    def at(self, switch, stamp):
        """Read the two complete candidate matchings from each original run."""
        return {f"{arm}_{side}_s": self.duration(arm, getattr(switch, side), stamp)
                for arm in _RUNS for side in ("release", "retain")}
