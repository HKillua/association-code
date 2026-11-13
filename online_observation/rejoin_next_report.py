"""Compare retaining and omitting one native update on a later raw report."""
from __future__ import annotations

from bisect import bisect_left, bisect_right
from math import log, tau
from operator import itemgetter

from .async_report_scores import _innovation_variance
from .continuous_tracker import FRESH_S, MEMORY_S, LocalTrack
from .nemot_replay import _distance_sq, _other, _registration_shifts
from .registration_probe import FIT_STOP_S
from .rejoin_async_innovation import _prior_states, _readout
from .relative_motion_gap import _sub


_TIME = itemgetter(0)
_SIDES = ("release", "retain")
_ARMS = ("assimilate", "skip")


def _opportunity(history, focal_stamp, floor, stop, cache_times):
    first = bisect_right(history, focal_stamp, key=_TIME)
    start = bisect_right(history, floor, key=_TIME)
    end = bisect_left(history, stop, key=_TIME)
    raw = tuple(stamp for stamp, _ in history[start:end])
    cached = tuple(stamp for stamp in raw if stamp in cache_times)
    novel = tuple(stamp for stamp in raw if stamp not in cache_times)
    at_stop = end < len(history) and history[end][0] == stop
    return dict(raw_after_focal_before_stop=tuple(stamp for stamp, _ in history[first:end]),
                raw_after_floor_before_stop=raw, cached_after_floor_before_stop=cached,
                novel_after_floor_before_stop=novel, at_stop=at_stop, at_stop_cached=at_stop and stop in cache_times)


def _focal_paths(history, index, end):
    stamp, point = history[index]
    before, _ = _prior_states(history[:index], stamp, stamp)
    if before is None:
        return "missing_prior", None
    if stamp - before.stamp > MEMORY_S:
        return "reset", None
    states = {"assimilate": before.updated(stamp, point), "skip": before}
    for time, observed in history[index + 1:end]:
        states = {arm: state.updated(time, observed) if time - state.stamp <= MEMORY_S
                  else LocalTrack(time, observed, observed) for arm, state in states.items()}
    return "supported", states


def _report_score(stamp, point, priors, shift, focal_role, focal_state):
    score = _readout(stamp, point, priors, shift)
    own = priors["own"]
    own_loss = own_nll = None
    if own is not None and 0 < stamp - own.stamp <= FRESH_S:
        own_age = stamp - own.stamp
        own_loss = _distance_sq(point, own.position(stamp, True))
        variance = _innovation_variance(own, own_age)
        own_nll = log(tau * variance) + own_loss / (2 * variance)
    age = stamp - focal_state.stamp
    affected = focal_role if focal_role != "own" else None
    return score | dict(focal_prior_stamp=focal_state.stamp, focal_prior_age_s=age,
                        focal_innovation_variance_m2=_innovation_variance(focal_state, age),
                        own_loss_m2=own_loss, own_nominal_nll=own_nll, affected_edge=affected,
                        affected_edge_loss_m2=score[f"{affected}_actual_m2"] if affected else None)


def score_next_report(switch, histories, focal, witness_key, cache_times, stop, fit, *, after=None):
    """Hold out the first cache-excluded witness strictly before the old rejoin.

    Each history contains ordered native report stamps and projected positions.
    Only the focal track update at ``focal`` differs between the two forecasts;
    intervening updates, witness coordinates, and the two matchings are shared.
    The witness and all other reports at its time are strictly excluded from
    the forecasts. ``after`` may impose a later start for an opposite witness.
    """
    focal_key, focal_stamp = focal
    if focal_key not in switch.keys or witness_key not in switch.keys or not focal_stamp < stop:
        raise ValueError("focal and witness tracks must belong to the fixed window")
    focal_history = histories.get(focal_key, ())
    focal_index = bisect_left(focal_history, focal_stamp, key=_TIME)
    if focal_index == len(focal_history) or focal_history[focal_index][0] != focal_stamp:
        raise ValueError("the focal update does not exist at its frozen time")
    floor = max(focal_stamp, after) if after is not None else focal_stamp
    if not floor < stop:
        raise ValueError("the witness lower bound must precede the fixed rejoin")
    history = histories.get(witness_key, ())
    support = _opportunity(history, focal_stamp, floor, stop, set(cache_times))
    own = witness_key == focal_key
    result = dict(**support, focal_stamp=focal_stamp, floor_stamp=floor, witness_is_focal_track=own,
                  witness_stamp=None, witness_delay_s=None, lead_to_stop_s=None, scores=None,
                  intervening_focal_stamps=(), focal_prior_stamps=None, focal_forecast_delta_m=None)
    if not support["novel_after_floor_before_stop"]:
        status = "cached_only" if support["raw_after_floor_before_stop"] else "no_later_raw"
        return result | {"status": status}

    stamp = support["novel_after_floor_before_stop"][0]
    end = bisect_left(focal_history, stamp, key=_TIME)
    result.update(witness_stamp=stamp, witness_delay_s=stamp - focal_stamp, lead_to_stop_s=stop - stamp,
                  intervening_focal_stamps=tuple(time for time, _ in focal_history[focal_index + 1:end]))
    status, paths = _focal_paths(focal_history, focal_index, end)
    if paths is None:
        return result | {"status": status}
    result["focal_prior_stamps"] = {arm: state.stamp for arm, state in paths.items()}
    if any(not 0 < stamp - state.stamp <= FRESH_S for state in paths.values()):
        return result | {"status": "stale_prior"}

    roles = dict(own=witness_key, **{side: _other(next(edge for edge in getattr(switch, side) if witness_key in edge),
                                                witness_key) for side in _SIDES})
    if focal_key not in roles.values():
        raise ValueError("an opposite witness must directly compete for the focal track")
    focal_role = next(role for role, key in roles.items() if key == focal_key)
    common_priors = {role: _prior_states(histories.get(key, ()), stamp, stamp)[0]
                     for role, key in roles.items() if role != focal_role}
    if any(prior is None for prior in common_priors.values()):
        status = "missing_prior"
    elif any(not 0 < stamp - prior.stamp <= FRESH_S for prior in common_priors.values()):
        status = "stale_prior"
    if status != "supported" and not own:
        return result | {"status": status}

    shifts = _registration_shifts(fit) if stamp > FIT_STOP_S else {}
    shift = _sub(shifts.get(witness_key[0], (0.0, 0.0)), shifts.get(roles["release"][0], (0.0, 0.0)))
    point = history[bisect_left(history, stamp, key=_TIME)][1]
    scores = {arm: _report_score(stamp, point, common_priors | {focal_role: state}, shift, focal_role, state)
              for arm, state in paths.items()}
    delta = _sub(*(paths[arm].position(stamp, True) for arm in _ARMS))
    return result | dict(status=status if status == "supported" else "own_only", scores=scores,
                         focal_forecast_delta_m=delta)
