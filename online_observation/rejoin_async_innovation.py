"""Score cache-excluded native opposite reports on a frozen switch."""
from __future__ import annotations

from bisect import bisect_left
from operator import itemgetter
from statistics import fmean

from .async_report_scores import _innovation_variance
from .continuous_tracker import FRESH_S, MEMORY_S, LocalTrack
from .nemot_replay import _add, _distance_sq, _other, _registration_shifts
from .registration_probe import FIT_STOP_S


_TIME = itemgetter(0)
_SIDES = ("release", "retain")
_PRIORS = ("rolling", "onset_origin")
_SCORES = (
    "actual_m2", "masked_m2", "increment_m2",
    "release_actual_m2", "retain_actual_m2", "release_masked_m2", "retain_masked_m2",
    "release_change_m2", "retain_change_m2", "common_actual", "common_masked", "common_increment",
    "unregistered_actual_m2", "unregistered_masked_m2", "unregistered_increment_m2",
)


def _prior_states(history, stamp, onset):
    current = origin = None
    for time, point in history:
        if time >= stamp:
            break
        current = (current.updated(time, point) if current is not None and time - current.stamp <= MEMORY_S
                   else LocalTrack(time, point, point))
        if time <= onset:
            origin = current
    return current, origin


def _margins(actual, masked, release, retain):
    distances = dict(release_actual_m2=_distance_sq(actual, release), retain_actual_m2=_distance_sq(actual, retain),
                     release_masked_m2=_distance_sq(masked, release), retain_masked_m2=_distance_sq(masked, retain))
    changes = {f"{side}_change_m2": distances[f"{side}_actual_m2"] - distances[f"{side}_masked_m2"]
               for side in _SIDES}
    actual_m2 = distances["retain_actual_m2"] - distances["release_actual_m2"]
    masked_m2 = distances["retain_masked_m2"] - distances["release_masked_m2"]
    return distances | changes | dict(actual_m2=actual_m2, masked_m2=masked_m2, increment_m2=actual_m2 - masked_m2)


def _readout(stamp, point, priors, shift):
    stamps = {role: track.stamp if track is not None else None for role, track in priors.items()}
    ages = {role: stamp - time if time is not None else None for role, time in stamps.items()}
    result = dict.fromkeys(_SCORES) | dict(prior_stamps=stamps, ages_s=ages, variance_m2=None,
                                          common_variance_m2=None, status="missing_prior")
    if any(time is None for time in stamps.values()):
        return result
    if any(not 0 < age <= FRESH_S for age in ages.values()):
        return result | {"status": "stale_prior"}

    positions = {role: track.position(stamp, True) for role, track in priors.items()}
    variances = {role: _innovation_variance(track, ages[role]) for role, track in priors.items()}
    common_variance = fmean(variances[side] for side in _SIDES)
    release, retain = (positions[side] for side in _SIDES)
    unregistered = _margins(point, positions["own"], release, retain)
    registered = _margins(_add(point, shift), _add(positions["own"], shift), release, retain)
    return result | registered | dict(
        status="supported", variance_m2=variances, common_variance_m2=common_variance,
        common_actual=registered["actual_m2"] / common_variance,
        common_masked=registered["masked_m2"] / common_variance,
        common_increment=registered["increment_m2"] / common_variance,
        unregistered_actual_m2=unregistered["actual_m2"], unregistered_masked_m2=unregistered["masked_m2"],
        unregistered_increment_m2=unregistered["increment_m2"],
    )


def _report(switch, histories, selection, onset, stop, shifts):
    if selection is None:
        return None
    key, stamp = selection
    if key not in switch.keys or not onset < stamp < stop:
        raise ValueError("selected opposite report must belong to the switch and strictly precede rejoin")
    history = histories.get(key, ())
    index = bisect_left(history, stamp, key=_TIME)
    if index == len(history) or history[index][0] != stamp:
        raise ValueError("selected opposite report was not found on its native clock")

    partners = {side: _other(next(edge for edge in getattr(switch, side) if key in edge), key) for side in _SIDES}
    if partners["release"] == partners["retain"] or any(partner[0] == key[0] for partner in partners.values()):
        raise ValueError("selected report needs two distinct cross-source alternatives")
    states = {role: _prior_states(histories.get(track, ()), stamp, onset)
              for role, track in dict(own=key, **partners).items()}
    source_shift = shifts.get(key[0], (0.0, 0.0)) if stamp > FIT_STOP_S else (0.0, 0.0)
    partner_shift = shifts.get(partners["release"][0], (0.0, 0.0)) if stamp > FIT_STOP_S else (0.0, 0.0)
    if partners["release"][0] != partners["retain"][0]:
        raise ValueError("the alternatives must share one source")
    shift = tuple(first - second for first, second in zip(source_shift, partner_shift, strict=True))
    scores = {arm: _readout(stamp, history[index][1], {role: pair[i] for role, pair in states.items()}, shift)
              for i, arm in enumerate(_PRIORS)}
    return dict(key=key, stamp=stamp, age_to_stop_s=stop-stamp, partners=partners, shift_m=shift,
                refreshed_roles=tuple(role for role, (rolling, _) in states.items()
                                      if rolling is not None and rolling.stamp > onset), **scores)


def _aggregate(reports, arm):
    supported = [report[arm] for report in reports.values() if report is not None and report[arm]["status"] == "supported"]
    scores = ({name: fmean(report[name] for report in supported) for name in _SCORES}
              if len(supported) == 2 else dict.fromkeys(_SCORES))
    return dict(support_count=len(supported), **scores)


def score_report_innovations(switch, histories, novel, onset, stop, fit):
    """Compare two latest opposite reports at their individual native timestamps.

    The caller supplies selected `(track_key, stamp)` pairs from outside the old
    shape's cache, using `None` for missing selections. Per-key histories contain
    ordered projected raw reports. Forecasts use strictly earlier reports, and
    the onset-origin control freezes each local prior at or before the onset.
    A positive margin favors release. The two-report mean is descriptive and
    does not interpret its terms as independent observations.
    """
    selected_keys = [selection[0] for selection in novel.values() if selection is not None]
    if len(selected_keys) == 2 and (selected_keys[0] == selected_keys[1] or selected_keys[0][0] != selected_keys[1][0]):
        raise ValueError("two selected reports must come from distinct tracks of one opposite source")
    shifts = _registration_shifts(fit)
    reports = {side: _report(switch, histories, novel[side], onset, stop, shifts) for side in _SIDES}
    return dict(reports=reports, **{arm: _aggregate(reports, arm) for arm in _PRIORS})
