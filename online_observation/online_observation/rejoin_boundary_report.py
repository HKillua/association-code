"""Test the original MTAD trigger against four strictly batch-prior forecasts."""
from __future__ import annotations

from bisect import bisect_left
from math import log, tau
from operator import itemgetter

from .hypothesis_gap_diagnostic import _position_readouts
from .nemot_replay import _distance_sq, _other, _registration_shifts
from .registration_probe import FIT_STOP_S
from .rejoin_async_innovation import _prior_states, _readout
from .rejoin_next_report import _focal_paths
from .relative_motion_gap import _decompose, _sub


_TIME = itemgetter(0)
_SIDES = ("release", "retain")
_OMISSIONS = {"assimilate": (), "skip_release": ("release",),
              "skip_retain": ("retain",), "skip_both": _SIDES}


def _selected_path(history, stamp, stop):
    index = bisect_left(history, stamp, key=_TIME)
    end = bisect_left(history, stop, key=_TIME)
    if index >= end or history[index][0] != stamp:
        raise ValueError("an earlier selected report is absent from the strict prefix")
    status, paths = _focal_paths(history, index, end)
    if paths is None:
        paths = dict(assimilate=_prior_states(history, stop, stop)[0], skip=None)
    metadata = dict(status=status, stamp=stamp,
                    intervening_stamps=tuple(t for t, _ in history[index + 1:end]),
                    prior_stamps={arm: state.stamp if state is not None else None for arm, state in paths.items()})
    return metadata, paths


def _score(switch, roles, stamp, point, states, shifts):
    track_priors = {roles[role]: state for role, state in states.items() if state is not None}
    trigger_key = roles["trigger"]
    view = _position_readouts(switch.keys, stamp, {trigger_key: point}, track_priors, shifts)
    local = _readout(stamp, point, {"own": states["trigger"], **{side: states[side] for side in _SIDES}},
                     _sub(shifts.get(trigger_key[0], (0., 0.)), shifts.get(roles["release"][0], (0., 0.))))
    times = {role: state.stamp if state is not None else None for role, state in states.items()}
    ages = {role: stamp - time if time is not None else None for role, time in times.items()}
    score = dict(status="missing_prior" if any(time is None for time in times.values()) else "stale_prior",
                 actual_m2=None, masked_m2=None, increment_m2=None, local=local, edges=None,
                 prior_stamps=times, ages_s=ages, nominal_nll=None)
    if view is None:
        return score

    masked, observed, updated = view
    group = _decompose(switch, masked, observed, updated)
    edges = {}
    for side in _SIDES:
        companion_edge = next(edge for edge in getattr(switch, side) if trigger_key not in edge)
        companion_m2 = _distance_sq(*(masked[key] for key in companion_edge))
        edges[side] = dict(trigger_m2=local[f"{side}_actual_m2"], companion_m2=companion_m2,
                           trigger_masked_m2=local[f"{side}_masked_m2"], companion_masked_m2=companion_m2)
    nll = {}
    for side in _SIDES:
        variance = local["variance_m2"][side]
        nll[side] = log(tau * variance) + edges[side]["trigger_m2"] / (2 * variance)
    return score | dict(status="supported", actual_m2=group["actual"], masked_m2=group["masked"],
                        increment_m2=group["increment"], edges=edges, nominal_nll=nll)


def score_boundary(switch, histories, selections, trigger, stop, fit):
    """Score the one observed group trigger without updating any forecast at its time.

    ``selections`` identifies the earlier report on each of the trigger's two
    opposite candidates. Each counterfactual removes only the named earlier
    update, replaying all other events strictly before ``stop``. Every score
    uses the same two fixed matchings and the same trigger or its prior mask.
    The caller audits original graph/shape use separately from these scores.
    """
    release, retain = (selections[side][0] for side in _SIDES)
    if (len(switch.keys) != 4 or not {trigger, release, retain} <= switch.keys or release == retain
            or release[0] == trigger[0] or retain[0] != release[0]
            or any(tuple(sorted((trigger, selections[side][0]))) not in getattr(switch, side) for side in _SIDES)):
        raise ValueError("the selected tracks must be the trigger's original two opposing candidates")
    companion, = switch.keys - {trigger, release, retain}
    roles = dict(trigger=trigger, companion=companion, release=release, retain=retain)
    indexes = {role: bisect_left(histories.get(key, ()), stop, key=_TIME) for role, key in roles.items()}
    present = {role for role, key in roles.items()
               if indexes[role] < len(histories.get(key, ())) and histories[key][indexes[role]][0] == stop}
    if present != {"trigger"}:
        raise ValueError("the original decision needs exactly its one fixed-group trigger report")

    same_priors = {role: _prior_states(histories.get(roles[role], ()), stop, stop)[0]
                   for role in ("trigger", "companion")}
    focal, states = {}, {}
    for side in _SIDES:
        key, stamp = selections[side]
        focal[side], states[side] = _selected_path(histories.get(key, ()), stamp, stop)
    point = histories[trigger][indexes["trigger"]][1]
    shifts = _registration_shifts(fit) if stop > FIT_STOP_S else {}
    scores = {arm: _score(switch, roles, stop, point, same_priors | {
        side: states[side]["skip" if side in omitted else "assimilate"] for side in _SIDES}, shifts)
        for arm, omitted in _OMISSIONS.items()}
    trigger_prior = same_priors["trigger"]
    innovation = _sub(point, trigger_prior.position(stop, True)) if trigger_prior is not None else None
    available = sum(score["status"] == "supported" for score in scores.values())
    status = "supported" if available == len(scores) else "partial_support" if available else "unavailable"
    return dict(status=status, stamp=stop, trigger_innovation_m=innovation, focals=focal, scores=scores)
