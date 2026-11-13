"""Audit strict same-track innovation histories for native-time rejoin reports."""
from __future__ import annotations

from bisect import bisect_left
from operator import itemgetter
from statistics import fmean

from .continuous_tracker import FRESH_S, MEMORY_S, LocalTrack
from .interval_assignment import _dot
from .nemot_replay import _add, _distance_sq, _other, _registration_shifts
from .registration_probe import FIT_STOP_S
from .rejoin_async_innovation import _margins, _readout
from .relative_motion_gap import _sub


_TIME = itemgetter(0)
_SIDES = ("release", "retain")
_ROLES = ("own", *_SIDES)
_ARMS = ("rolling", "own_corrected", "opponents_corrected", "both_corrected")
_SCORES = (
    "actual_m2", "masked_m2", "increment_m2", "release_actual_m2", "retain_actual_m2",
    "release_masked_m2", "retain_masked_m2", "release_change_m2", "retain_change_m2",
)
_ZERO = (0.0, 0.0)


def _states(history):
    track, streak = None, 0
    records = []
    for time, point in history:
        age = time - track.stamp if track is not None else None
        if age is not None and 0 < age <= FRESH_S:
            streak += 1
        else:
            streak = 1
        track = (track.updated(time, point) if age is not None and 0 < age <= MEMORY_S
                 else LocalTrack(time, point, point))
        records.append((track, streak))
    return records


def _strict_context(roles, histories, states, stamp):
    indexes = {role: bisect_left(histories.get(key, ()), stamp, key=_TIME) for role, key in roles.items()}
    context = {role: states[key][indexes[role]-1] if indexes[role] else (None, 0) for role, key in roles.items()}
    return indexes, context


def _relative_shift(roles, stamp, shifts):
    if stamp <= FIT_STOP_S:
        return _ZERO
    return _sub(shifts.get(roles["own"][0], _ZERO), shifts.get(roles["release"][0], _ZERO))


def _history(history, states, index, stamp):
    first = max(index - 2, 0)
    result = dict(status="missing_history", innovation_stamps=tuple(history[i][0] for i in range(first, index)),
                  prior_stamps=tuple(history[i-1][0] if i else None for i in range(first, index)),
                  innovations_m=None, holdout_loss_m2=None, holdout_baseline_m2=None, bias_m=None)
    if index < 3:
        return result
    clocks = (history[first-1][0], *result["innovation_stamps"], stamp)
    if any(not 0 < later - earlier <= FRESH_S for earlier, later in zip(clocks, clocks[1:])):
        return result | {"status": "stale_history"}
    if any(states[i-1][1] < 2 for i in range(first, index)):
        return result | {"status": "immature_history"}

    innovations = tuple(_sub(history[i][1], states[i-1][0].position(history[i][0], True))
                        for i in range(first, index))
    previous, last = innovations
    heldout = _distance_sq(last, previous)
    baseline = _dot(last, last)
    result.update(innovations_m=innovations, holdout_loss_m2=heldout, holdout_baseline_m2=baseline)
    if heldout >= baseline:
        return result | {"status": "no_holdout_gain"}
    return result | dict(status="qualified", bias_m=tuple(fmean(pair) for pair in zip(previous, last, strict=True)))


def _preceding(roles, histories, states, index, onset, shifts):
    own_history = histories[roles["own"]]
    results = []
    for stamp, point in own_history[max(index-2, 0):index]:
        _, context = _strict_context(roles, histories, states, stamp)
        priors = {role: state for role, (state, _) in context.items()}
        shift = _relative_shift(roles, stamp, shifts)
        readout = _readout(stamp, point, priors, shift)
        if readout["status"] == "supported" and any(streak < 2 for _, streak in context.values()):
            readout = dict.fromkeys(readout) | dict(status="immature_prior", prior_stamps=readout["prior_stamps"],
                                                 ages_s=readout["ages_s"])
        results.append(dict(stamp=stamp, after_onset=stamp > onset, **readout))
    return tuple(results)


def _corrected(actual, positions, histories, own, opponents):
    corrected_roles = ("own",) * own + _SIDES * opponents
    if any(histories[role]["bias_m"] is None for role in corrected_roles):
        return dict.fromkeys(_SCORES) | {"status": "unqualified_history"}
    forecast = {role: _add(point, histories[role]["bias_m"]) if role in corrected_roles else point
                for role, point in positions.items()}
    return _margins(actual, *(forecast[role] for role in _ROLES)) | {"status": "supported"}


def _report(switch, histories, states, selection, onset, shifts):
    if selection is None:
        return None
    key, stamp = selection
    if key not in switch.keys or stamp <= onset:
        raise ValueError("a selected report must belong to the switch and follow onset")
    partners = {side: _other(next(edge for edge in getattr(switch, side) if key in edge), key) for side in _SIDES}
    if (partners["release"] == partners["retain"] or partners["release"][0] != partners["retain"][0]
            or partners["release"][0] == key[0]):
        raise ValueError("the report must have two competing cross-source forecasts")
    roles = dict(own=key, **partners)
    indexes, context = _strict_context(roles, histories, states, stamp)
    own_history = histories.get(key, ())
    own_index = indexes["own"]
    if own_index == len(own_history) or own_history[own_index][0] != stamp:
        raise ValueError("the selected report is absent on the native clock")
    point = own_history[own_index][1]
    priors = {role: state for role, (state, _) in context.items()}
    shift = _relative_shift(roles, stamp, shifts)
    rolling = _readout(stamp, point, priors, shift)
    preceding = _preceding(roles, histories, states, own_index, onset, shifts)
    result = dict(stamp=stamp, status=rolling["status"], rolling=rolling, preceding=preceding, history=None,
                  own_loss_m2=None, own_corrected_loss_m2=None, decomposition=None, innovation_m=None)
    result.update({arm: dict.fromkeys(_SCORES) | {"status": "unsupported_rolling"} for arm in _ARMS[1:]})
    if rolling["status"] != "supported":
        return result

    history = {role: _history(histories[track], states[track], indexes[role], stamp) for role, track in roles.items()}
    positions = {role: state.position(stamp, True) for role, state in priors.items()}
    innovation = _sub(point, positions["own"])
    result.update(history=history, innovation_m=innovation, own_loss_m2=_dot(innovation, innovation))
    actual = _add(point, shift)
    positions["own"] = _add(positions["own"], shift)
    for arm, own, opponents in (("own_corrected", True, False), ("opponents_corrected", False, True),
                                ("both_corrected", True, True)):
        result[arm] = _corrected(actual, positions, history, own, opponents)
    persistent = history["own"]["bias_m"]
    if persistent is not None:
        remainder = _sub(innovation, persistent)
        axis = _sub(positions["release"], positions["retain"])
        result.update(own_corrected_loss_m2=_dot(remainder, remainder), decomposition=dict(
            persistent_m=persistent, remainder_m=remainder, axis_m=axis,
            persistent_increment_m2=2 * _dot(persistent, axis), remainder_increment_m2=2 * _dot(remainder, axis),
        ))
    return result


def _aggregate(reports, arm):
    supported = [report[arm] for report in reports.values() if report is not None and report[arm]["status"] == "supported"]
    scores = ({name: fmean(report[name] for report in supported) for name in _SCORES}
              if len(supported) == 2 else dict.fromkeys(_SCORES))
    return dict(support_count=len(supported), **scores)


def score_track_history(switch, histories, novel, onset, fit):
    """Return C51 scores, same-track loss and fixed-prior counterfactuals.

    The caller supplies the original cache-excluded selections by role. Each
    projected history is ordered by distinct native timestamps. The two
    immediately preceding own reports never change to recover missing support.
    All forecasts and prior innovations use each track's strictly earlier
    observations. Innovation means qualify using only two earlier reports.
    """
    selections = [entry for entry in novel.values() if entry is not None]
    if len(selections) > 2 or (len(selections) == 2 and
            (selections[0][0] == selections[1][0] or selections[0][0][0] != selections[1][0][0])):
        raise ValueError("selected reports must come from distinct tracks of one source")
    if not selections:
        reports = dict.fromkeys(novel)
    else:
        latest = max(stamp for _, stamp in selections)
        states = {key: _states(history[:bisect_left(history, latest, key=_TIME)])
                  for key, history in histories.items() if key in switch.keys}
        shifts = _registration_shifts(fit)
        reports = {side: _report(switch, histories, states, selection, onset, shifts)
                   for side, selection in novel.items()}
    return dict(reports=reports, **{arm: _aggregate(reports, arm) for arm in _ARMS})
