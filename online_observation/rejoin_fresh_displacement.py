"""Read cache-disjoint opposite-source displacement on a fixed past interval."""
from __future__ import annotations

from bisect import bisect_left
from math import fsum
from operator import itemgetter

from .continuous_tracker import FRESH_S
from .interval_assignment import MIN_SPAN_SECONDS
from .nemot_replay import _distance_sq
from .past_motion_shape import _track_support


_TIME = itemgetter(0)
_OPPONENTS = ("release", "retain")


def _weights(stamps, query):
    lower, upper = stamps
    if lower == upper:
        return {lower: 1.0}
    gap = upper - lower
    return {lower: (upper - query) / gap, upper: (query - lower) / gap}


def _displacement(track, indices, queries):
    brackets = tuple(tuple(track[index][0] for index in pair) for pair in indices)
    left, right = (_weights(pair, query) for pair, query in zip(brackets, queries, strict=True))
    coefficients = {}
    for time in sorted(left.keys() | right.keys()):
        weight = right.get(time, 0.0) - left.get(time, 0.0)
        if weight:
            coefficients[time] = weight

    points = {track[index][0]: track[index][1] for pair in indices for index in pair}
    # Use relative coordinates so rounding in the weight sum cannot leak a translation.
    origin = points[brackets[0][0]]
    displacement = tuple(fsum(weight * (points[time][axis] - origin[axis])
                              for time, weight in coefficients.items()) for axis in range(2))
    return brackets, coefficients, displacement


def _readout(tracks, queries, onset, stamp):
    ages = {role: stamp - track[-1][0] if track else None for role, track in tracks.items()} if tracks else None
    result = dict(status="missing_report", ages_s=ages, margin_m2=None, mismatch_m2=None, displacements_m=None,
                  brackets=None, coefficients=None, available_at=None, latest_contributor_at=None,
                  post_onset_roles=(), post_onset_count=0, qualified=False)
    if tracks is None:
        return result | {"status": "missing_cache_provenance"}
    if not all(tracks.values()):
        return result
    if any(age > FRESH_S for age in ages.values()):
        return result | {"status": "stale"}

    supports = {role: _track_support(track, len(track), queries) for role, track in tracks.items()}
    statuses = {status for status, _ in supports.values()}
    for status in ("missing_window", "gap_too_wide"):
        if status in statuses:
            return result | {"status": status}

    brackets, coefficients, motions = {}, {}, {}
    for role, track in tracks.items():
        brackets[role], coefficients[role], motions[role] = _displacement(track, supports[role][1], queries)
    mismatches = {side: _distance_sq(motions["reporter"], motions[side]) for side in _OPPONENTS}
    post_onset = {side: sum(onset < time for time in coefficients[side]) for side in _OPPONENTS}
    return result | dict(
        status="supported", margin_m2=mismatches["retain"] - mismatches["release"], mismatch_m2=mismatches,
        displacements_m=motions, brackets=brackets, coefficients=coefficients,
        available_at=max(pairs[-1][-1] for pairs in brackets.values()),
        latest_contributor_at=max(time for weights in coefficients.values() for time in weights),
        post_onset_roles=tuple(side for side, count in post_onset.items() if count),
        post_onset_count=sum(post_onset.values()), qualified=any(post_onset.values()),
    )


def score_fresh_displacement(sample, histories, cache_provenance, onset):
    """Compare the two opponents' displacement with the same prior reporter interval.

    Histories contain strictly ordered native reports projected into a metre plane.
    Cache provenance maps original track keys to the native timestamps used by the
    previous shape. Filtering is limited to the opposite tracks. A positive margin
    favors release and a post-onset count includes only nonzero displacement weights.
    """
    stamp, reporter = sample["stamp"], sample["reporter"]
    partners = {side: sample["partners"][side] for side in _OPPONENTS}
    release, retain = partners.values()
    if release == retain or release[0] != retain[0] or reporter[0] == release[0]:
        raise ValueError("displacement requires two distinct candidates from the opposite source")
    tracks = {}
    for role, key in dict(reporter=reporter, **partners).items():
        history = histories.get(key, ())
        tracks[role] = history[:bisect_left(history, stamp, key=_TIME)]
    right = tracks["reporter"][-1][0] if tracks["reporter"] else None
    left = right - MIN_SPAN_SECONDS if right is not None else None

    verified = cache_provenance is not None and all(key in cache_provenance for key in partners.values())
    decached = {"reporter": tracks["reporter"]} if verified else None
    post_onset_masked = {"reporter": tracks["reporter"]} if verified else None
    opportunities = {}
    for side, key in partners.items():
        history = tracks[side]
        outside_cache = None
        if verified:
            cached = set(cache_provenance[key])
            if any(time >= stamp for time in cached):
                raise ValueError("the old cache includes a current or future opposite report")
            outside_cache = [row for row in history if row[0] not in cached]
            decached[side] = outside_cache
            post_onset_masked[side] = [row for row in outside_cache if row[0] <= onset]
        opportunities[side] = dict(
            before_batch=len(history), removed_cache=len(history) - len(outside_cache) if verified else None,
            post_onset=sum(onset < time for time, _ in history),
            post_onset_outside_cache=sum(onset < time for time, _ in outside_cache) if verified else None,
        )

    queries = (left, right)
    return dict(left=left, right=right, cache_verified=verified, opportunities=opportunities,
                unfiltered=_readout(tracks, queries, onset, stamp),
                decached=_readout(decached, queries, onset, stamp),
                post_onset_masked=_readout(post_onset_masked, queries, onset, stamp))
