"""Check a frozen rejoin against native coordinates before the current batch."""
from __future__ import annotations

from bisect import bisect_left
from operator import itemgetter

from .causal_common_motion import MAX_GAP_SECONDS, _bracket, _point_at
from .interval_assignment import MIN_SPAN_SECONDS
from .nemot_replay import _add, _distance_sq
from .relative_motion_gap import _sub


_TIME = itemgetter(0)
_SIDES = ("release", "retain")


def _brackets(tracks, arrived, stamp):
    brackets = {key: _bracket(track, arrived[key], stamp) for key, track in tracks.items()}
    if any(pair is None for pair in brackets.values()):
        return "missing_window", None
    if any(tracks[key][upper][0] - tracks[key][lower][0] > MAX_GAP_SECONDS
           for key, (lower, upper) in brackets.items()):
        return "gap_too_wide", None
    return "supported", brackets


def _snapshot(tracks, brackets, stamp, reporter, partners, shift):
    points = {key: _point_at(tracks[key], stamp, pair) for key, pair in brackets.items()}
    reporter_position = _add(points[reporter], shift)
    residuals = {side: _sub(reporter_position, points[partner]) for side, partner in partners.items()}
    squared = {side: _distance_sq(reporter_position, points[partner]) for side, partner in partners.items()}
    provenance = {key: tuple(tracks[key][index][0] for index in sorted(set(pair)))
                  for key, pair in brackets.items()}
    return dict(margin_m2=squared["retain"] - squared["release"], squared_m2=squared,
                residuals_m=residuals, provenance=provenance,
                available_at=max(times[-1] for times in provenance.values()))


def score_reverse_witness(sample, histories):
    """Read the latest prior reporter coordinate against both opposite histories.

    Histories are native projected coordinates ordered by strictly increasing
    timestamps. A shared additive registration of the reporter relative to
    its opponents is provided by the frozen batch. Current or later reports
    on any participant are excluded. Positive margins favor release.
    """
    stamp, reporter = sample["stamp"], sample["reporter"]
    partners = {side: sample["partners"][side] for side in _SIDES}
    release, retain = partners.values()
    if release == retain or release[0] != retain[0] or reporter[0] == release[0]:
        raise ValueError("the witness requires two distinct candidates from the opposite source")
    roles = dict(reporter=reporter, **partners)
    tracks = {key: histories.get(key, ()) for key in roles.values()}
    arrived = {key: bisect_left(track, stamp, key=_TIME) for key, track in tracks.items()}
    latest = {key: tracks[key][count - 1][0] if count else None for key, count in arrived.items()}
    query = latest[reporter]
    ages = {role: stamp - latest[key] if latest[key] is not None else None for role, key in roles.items()}
    result = dict(query=query, ages_s=ages, endpoint_status="missing_report", interval_status="missing_report",
                  endpoint=None, interval=None)
    if not all(arrived.values()):
        return result

    status, endpoint_brackets = _brackets(tracks, arrived, query)
    result.update(endpoint_status=status, interval_status=status)
    if endpoint_brackets is None:
        return result
    shift = sample["shift_delta"]
    endpoint = _snapshot(tracks, endpoint_brackets, query, reporter, partners, shift)
    late = {side: endpoint["provenance"][key][-1] > query for side, key in partners.items()}
    endpoint.update(partner_after_query=late, both_partners_after_query=all(late.values()),
                    late_partner_count=sum(late.values()))
    result["endpoint"] = endpoint

    start = query - MIN_SPAN_SECONDS
    status, earlier_brackets = _brackets(tracks, arrived, start)
    result["interval_status"] = status
    if earlier_brackets is None:
        return result
    if any(track[index + 1][0] - track[index][0] > MAX_GAP_SECONDS for key, track in tracks.items()
           for index in range(earlier_brackets[key][0], endpoint_brackets[key][1])):
        return result | {"interval_status": "gap_too_wide"}

    earlier = _snapshot(tracks, earlier_brackets, start, reporter, partners, shift)
    change = {side: endpoint["squared_m2"][side] - earlier["squared_m2"][side] for side in _SIDES}
    displacement_squared = {side: _distance_sq(endpoint["residuals_m"][side], earlier["residuals_m"][side])
                            for side in _SIDES}
    provenance = {key: tuple(sorted(set(endpoint["provenance"][key] + earlier["provenance"][key])))
                  for key in tracks}
    result["interval"] = dict(start=start, earlier=earlier, residual_change_m2=change,
                              margin_increment_m2=change["retain"] - change["release"],
                              displacement_margin_m2=displacement_squared["retain"] - displacement_squared["release"],
                              displacement_squared_m2=displacement_squared, provenance=provenance,
                              available_at=max(endpoint["available_at"], earlier["available_at"]))
    return result
