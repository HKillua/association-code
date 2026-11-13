"""Remove one reused report from a reciprocal branch's local forecast."""
from __future__ import annotations

from math import isclose

from .async_report_scores import _innovation_variance
from .continuous_tracker import FRESH_S, MEMORY_S
from .disjoint_report_closure import _STAMP_TOLERANCE_S, pair_kind, score_pair


def score_removal(older, newer, baseline_pair, prior, posterior):
    """Score the same report pair without its old report's local KF update.

    The older report remains explicit in the pair score. Only its later use as
    a reciprocal prediction is removed. The counterfactual reads the track
    state saved before the older report and preserves the other forecasts.
    """
    kind = pair_kind(older, newer)
    if (kind is None or not kind.startswith("reciprocal_") or kind != baseline_pair["kind"]
            or baseline_pair["older_stamp"] != older["stamp"]
            or baseline_pair["newer_stamp"] != newer["stamp"]
            or not baseline_pair["reused_older_report"]):
        raise ValueError("the frozen pair must have a directly reused reciprocal report")
    side = kind.removeprefix("reciprocal_")
    old_stamp, stamp = older["stamp"], newer["stamp"]
    original_age = stamp - old_stamp
    if (posterior is None or posterior.stamp != old_stamp
            or not isclose(original_age, newer["age"][side], rel_tol=0.0, abs_tol=_STAMP_TOLERANCE_S)):
        raise ValueError("the reciprocal forecast must use the older report as its latest update")

    original = posterior.position(stamp, True)
    omitted_age = stamp - prior.stamp if prior is not None else None
    result = dict(side=side, original_age=original_age, omitted_age=omitted_age,
                  original_prediction=original, omitted_prediction=None, prediction_delta=None,
                  original_variance=_innovation_variance(posterior, original_age), omitted_variance=None,
                  omitted=None)
    if prior is None:
        return result | {"status": "no_prior"}
    if prior.stamp >= old_stamp:
        raise ValueError("the saved state must precede the removed update")
    if old_stamp - prior.stamp > MEMORY_S:
        return result | {"status": "reset"}
    if omitted_age > FRESH_S:
        return result | {"status": "stale"}

    omitted = tuple(position + omitted_age * speed
                    for position, speed in zip(prior.location, prior.velocity, strict=True))
    delta = tuple(old - new for old, new in zip(original, omitted, strict=True))
    shifted = {field: newer[field] | {
        side: tuple(value + correction for value, correction in zip(newer[field][side], delta, strict=True))
    } for field in ("cross", "pred_delta")}
    omitted_score = score_pair(older, newer | shifted) | {"reused_older_report": False}
    return result | dict(status="supported", omitted_prediction=omitted, prediction_delta=delta,
                         omitted_variance=_innovation_variance(prior, omitted_age),
                         omitted=omitted_score)
