"""Compare changes in signed cross-source residuals on two arrived reports."""
from __future__ import annotations

from math import isclose

from .continuous_tracker import FRESH_S
from .nemot_replay import _distance_sq


SIDES = ("release", "retain")
_STAMP_TOLERANCE_S = 1e-6


def _source_sign(older, newer):
    return 1 if older["reporter"][0] == newer["reporter"][0] else -1


def _orient(vector, sign):
    return tuple(sign * coordinate for coordinate in vector)


def _vertices(sample, side):
    return {sample["reporter"], sample["partners"][side]}


def pair_kind(older, newer):
    """Classify a strictly earlier same-registration report in a fixed 2×2 window."""
    if not 0 < newer["stamp"] - older["stamp"] <= FRESH_S:
        return None
    sign = _source_sign(older, newer)
    if _orient(older["shift_delta"], sign) != tuple(newer["shift_delta"]):
        return None
    if older["reporter"] == newer["reporter"]:
        return "repeat"

    overlap = []
    for side in SIDES:
        before, after = _vertices(older, side), _vertices(newer, side)
        if before.isdisjoint(after):
            continue
        if before != after:
            raise ValueError("reports disagree on the fixed matching")
        overlap.append(side)
    if not overlap:
        return "disjoint"
    if len(overlap) == 1:
        return f"reciprocal_{overlap[0]}"
    raise ValueError("distinct reports are reciprocal under both competing matchings")


def select_pairs(history, current):
    """Select the latest eligible earlier report per kind using only times and keys."""
    chosen = {}
    for older in history:
        kind = pair_kind(older, current)
        if kind is not None and (kind not in chosen or older["stamp"] > chosen[kind]["stamp"]):
            chosen[kind] = older
    return chosen


def score_pair(older, newer):
    """Use the change in residual to compare release and retain on the same pair.

    Masking the newer report preserves the older actual observation and both
    frozen forecasts. Signed orientation cancels a common time-constant source
    translation. The score is a geometric contrast in squared metres.
    """
    kind = pair_kind(older, newer)
    if kind is None:
        raise ValueError("report pair is not in the same recent causal window and registration phase")
    sign = _source_sign(older, newer)
    sides = {}
    for side in SIDES:
        previous = _orient(older["cross"][side], sign)
        previous_masked = _orient(older["pred_delta"][side], sign)
        current = newer["cross"][side]
        current_masked = newer["pred_delta"][side]
        sides[side] = dict(actual=_distance_sq(current, previous),
                           masked=_distance_sq(current_masked, previous),
                           earlier_masked=_distance_sq(current, previous_masked),
                           both_masked=_distance_sq(current_masked, previous_masked))

    margins = {name: sides["retain"][name] - sides["release"][name]
               for name in ("actual", "masked", "earlier_masked", "both_masked")}
    reused = False
    if kind.startswith("reciprocal_"):
        side = kind.removeprefix("reciprocal_")
        last_report = newer["stamp"] - newer["age"][side]
        reused = isclose(last_report, older["stamp"], rel_tol=0.0, abs_tol=_STAMP_TOLERANCE_S)
    return dict(kind=kind, lag_s=newer["stamp"] - older["stamp"], older_stamp=older["stamp"],
                newer_stamp=newer["stamp"], sides=sides,
                reused_older_report=reused, delta=margins["actual"] - margins["masked"], **margins)
