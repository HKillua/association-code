"""Trace which raw reports and cache removals form a fixed-window gap."""
from __future__ import annotations

from bisect import bisect_left
from itertools import pairwise
from operator import itemgetter

from .causal_common_motion import MAX_GAP_SECONDS
from .continuous_tracker import FRESH_S
from .interval_assignment import MIN_SPAN_SECONDS


_TIME = itemgetter(0)
_OPPONENTS = ("release", "retain")


def _bracket(times, query):
    upper = bisect_left(times, query)
    if upper == len(times):
        return None
    if times[upper] == query:
        return query, query
    return (times[upper - 1], times[upper]) if upper else None


def _window(times, left, right):
    if right is None:
        return (None, None), ()
    brackets = (_bracket(times, left), _bracket(times, right))
    if None in brackets:
        return brackets, ()
    start, stop = brackets[0][0], brackets[1][1]
    return brackets, times[bisect_left(times, start):bisect_left(times, stop) + 1]


def _max_gap(times):
    return max((stop - start for start, stop in pairwise(times)), default=None)


def _violation(raw, removed, start, stop):
    chain = raw[bisect_left(raw, start):bisect_left(raw, stop) + 1]
    steps = tuple(b - a for a, b in pairwise(chain))
    interior = chain[1:-1]
    removed_times = tuple(time for time in interior if time in removed)
    native_max = max(steps)
    if not removed_times:
        kind = "native"
    elif native_max > MAX_GAP_SECONDS:
        kind = "mixed"
    else:
        kind = "cache_induced"
    return dict(start=start, end=stop, span_s=stop - start, raw_interior_times=interior,
                removed_times=removed_times, native_steps_s=steps, native_max_s=native_max, kind=kind)


def _role(raw, retained, removed, left, right, onset, stop):
    raw_brackets, raw_span = _window(raw, left, right)
    raw_age = stop - raw[-1] if raw else None
    result = dict(raw_brackets=raw_brackets, raw_max_gap_s=_max_gap(raw_span), raw_age_s=raw_age,
                  raw_count=len(raw), raw_post_onset=tuple(t for t in raw if t > onset),
                  retained_brackets=None, retained_max_gap_s=None, retained_age_s=None, retained_count=None,
                  native_max_inside_retained_span_s=None, old_cache_count=None, novel_post_onset=None,
                  violations=())
    if retained is None:
        return result

    retained_brackets, retained_span = _window(retained, left, right)
    native_span = ()
    if retained_span:
        native_span = raw[bisect_left(raw, retained_span[0]):bisect_left(raw, retained_span[-1]) + 1]
    violations = tuple(_violation(raw, removed, a, b) for a, b in pairwise(retained_span)
                       if b - a > MAX_GAP_SECONDS)
    return result | dict(retained_brackets=retained_brackets, retained_max_gap_s=_max_gap(retained_span),
                         retained_age_s=stop - retained[-1] if retained else None, retained_count=len(retained),
                         native_max_inside_retained_span_s=_max_gap(native_span),
                         old_cache_count=len(raw) - len(retained),
                         novel_post_onset=tuple(t for t in retained if t > onset), violations=violations)


def _status(roles, view):
    ages = [role[f"{view}_age_s"] for role in roles.values()]
    if None in ages:
        return "missing_report"
    if any(age > FRESH_S for age in ages):
        return "stale"
    if any(None in role[f"{view}_brackets"] for role in roles.values()):
        return "missing_window"
    if any((role[f"{view}_max_gap_s"] or 0) > MAX_GAP_SECONDS for role in roles.values()):
        return "gap_too_wide"
    return "supported"


def trace_gap_lineage(sample, histories, cache_provenance, onset):
    """Explain fixed-interval gap eligibility from raw timestamps alone.

    Use the C50 right query (the trigger reporter's previous native report),
    its 20-second interval and the original span and freshness rules. A native
    gap refers to the observed stream, whose missing-report causes are unknown.
    Old-cache deletion is keyed by track and timestamp and only affects opponents.
    """
    stop, reporter = sample["stamp"], sample["reporter"]
    partners = {side: sample["partners"][side] for side in _OPPONENTS}
    keys = dict(reporter=reporter, **partners)
    raw = {}
    for role, key in keys.items():
        history = histories.get(key, ())
        raw[role] = tuple(row[0] for row in history[:bisect_left(history, stop, key=_TIME)])
    right = raw["reporter"][-1] if raw["reporter"] else None
    left = right - MIN_SPAN_SECONDS if right is not None else None
    verified = cache_provenance is not None and all(key in cache_provenance for key in partners.values())

    roles = {}
    for role, key in keys.items():
        removed = set(cache_provenance[key]) if verified and role in _OPPONENTS else set()
        if any(time >= stop for time in removed):
            raise ValueError("the old cache includes a current or future opposite report")
        retained = tuple(t for t in raw[role] if t not in removed) if verified else None
        roles[role] = _role(raw[role], retained, removed, left, right, onset, stop)

    return dict(left=left, right=right, cache_verified=verified, raw_status=_status(roles, "raw"),
                retained_status=_status(roles, "retained") if verified else "missing_cache_provenance",
                novel_roles=tuple(side for side in _OPPONENTS if roles[side]["novel_post_onset"]), roles=roles)
