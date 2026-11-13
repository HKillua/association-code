"""Compare causal motion profiles for two frozen four-endpoint matchings."""
from __future__ import annotations

from bisect import bisect_right
from operator import itemgetter

from .causal_common_motion import MAX_GAP_SECONDS, SEGMENT_SECONDS, _bracket, _difference, _point_at
from .hypothesis_gap_diagnostic import _margin
from .interval_assignment import MIN_SPAN_SECONDS
from .nemot_replay import _distance_sq


_TIME = itemgetter(0)


def _track_support(history, arrived, stamps):
    brackets = tuple(_bracket(history, arrived, stamp) for stamp in stamps)
    if None in brackets:
        return "missing_window", None
    first, last = brackets[0][0], brackets[-1][1]
    if any(history[index + 1][0] - history[index][0] > MAX_GAP_SECONDS
           for index in range(first, last)):
        return "gap_too_wide", None
    return "supported", brackets


def _features(points):
    start, middle, recent, end = points
    first = _difference(middle, start)
    second = _difference(end, middle)
    return dict(net=_difference(end, start), change=_difference(second, first),
                recent=_difference(end, recent))


def _scores(switch, coordinates):
    features = {key: _features(points) for key, points in coordinates.items()}
    margins = {name: _margin(switch, lambda edge: _distance_sq(*(features[key][name] for key in edge)))
               for name in ("net", "change", "recent")}
    margins["two_block"] = (margins["net"] + margins["change"]) / 2
    return margins


def score_past_motion(sample, switch, histories):
    """Read four arrived histories on a common clock, with a same-support mask.

    Histories are ordered native reports in planar metres and can extend
    beyond the decision. Positive squared-distance margins favor release.
    The mask replaces just the latest reporter coordinate with its saved KF
    forecast while retaining the same interpolation brackets and clocks.
    """
    stamp, reporter = sample["stamp"], sample["reporter"]
    if set(sample["keys"]) != switch.keys or reporter not in switch.keys:
        raise ValueError("the report does not belong to the frozen switch")
    tracks = {key: histories.get(key, ()) for key in sorted(switch.keys)}
    arrived = {key: bisect_right(track, stamp, key=_TIME) for key, track in tracks.items()}
    if not arrived[reporter] or tracks[reporter][arrived[reporter] - 1][0] != stamp:
        raise ValueError("the reported coordinate is missing at the frozen decision")
    missing = dict(status="missing_window", end=None, right_age=None, available_at=None,
                   current_used=False, provenance=None, actual=None, masked=None)
    if not all(arrived.values()):
        return missing

    end = min(tracks[key][count - 1][0] for key, count in arrived.items())
    stamps = end - 2 * SEGMENT_SECONDS, end - SEGMENT_SECONDS, end - MIN_SPAN_SECONDS, end
    supports = {key: _track_support(track, arrived[key], stamps) for key, track in tracks.items()}
    statuses = {status for status, _ in supports.values()}
    result = missing | {"end": end, "right_age": stamp - end}
    if "missing_window" in statuses:
        return result
    if "gap_too_wide" in statuses:
        return result | {"status": "gap_too_wide"}

    provenance = {key: tuple(sorted({tracks[key][index][0] for pair in support for index in pair}))
                  for key, (_, support) in supports.items()}
    coordinates = {key: tuple(_point_at(tracks[key], time, pair) for time, pair in zip(stamps, support, strict=True))
                   for key, (_, support) in supports.items()}
    available_at = max(times[-1] for times in provenance.values())
    actual = _scores(switch, coordinates)
    current_used = stamp in provenance[reporter]
    masked = actual.copy()
    if current_used:
        reporter_track = tracks[reporter]
        last = arrived[reporter] - 1
        masked_reporter = []
        for time, point, (lower, upper) in zip(stamps, coordinates[reporter], supports[reporter][1], strict=True):
            if upper != last:
                masked_reporter.append(point)
                continue
            weight = 1 if lower == upper else (time - reporter_track[lower][0]) / (stamp - reporter_track[lower][0])
            masked_reporter.append(tuple(value - weight * innovation
                                         for value, innovation in zip(point, sample["self"], strict=True)))
        masked = _scores(switch, coordinates | {reporter: masked_reporter})
    return dict(status="supported", end=end, right_age=stamp - end, available_at=available_at,
                current_used=current_used, provenance=provenance, actual=actual, masked=masked)
