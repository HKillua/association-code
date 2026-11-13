"""Compare single-anchor corrections on a frozen pair of assignment alternatives."""
from __future__ import annotations

from math import isclose

from .async_report_scores import reported_direction
from .continuous_tracker import FRESH_S
from .cross_source_calibration import MAX_AGE_DIFFERENCE_S, ZERO, predictive_scores
from .interval_assignment import MIN_SPAN_SECONDS
from .nemot_replay import _distance_sq


SIDES = ("release", "retain")


def _positive(*margins):
    return all(reported_direction(margin) > 0 for margin in margins)


def _recent_anchor(snapshot):
    keys = set(snapshot["keys"])
    sources = {key[0] for key in keys}
    reporter = snapshot["reporter"][0]
    stamp = snapshot["stamp"]
    provenance = snapshot["anchors"]["graph"]["provenance"]
    candidates = []
    for anchor, (since, qualified, _) in provenance:
        if not (qualified - since >= MIN_SPAN_SECONDS and qualified < anchor.stamp < stamp
                and keys.isdisjoint(anchor.edge) and {key[0] for key in anchor.edge} == sources
                and anchor.reporter in anchor.edge and anchor.reporter[0] == reporter
                and all(abs(anchor.age - snapshot["age"][side]) <= MAX_AGE_DIFFERENCE_S for side in SIDES)):
            raise ValueError("single-anchor qualification differs from the frozen provenance")
        if stamp - anchor.stamp <= FRESH_S:
            candidates.append(anchor)
    return min(candidates, key=lambda anchor: (-anchor.stamp, anchor.edge), default=None)


def _terms(actual, masked, mean, variance):
    quadratic = _distance_sq(mean, ZERO) / variance
    linear = -2 * sum(r * b for r, b in zip(actual, mean, strict=True)) / variance
    masked_linear = -2 * sum(r * b for r, b in zip(masked, mean, strict=True)) / variance
    return dict(raw_squared=_distance_sq(actual, ZERO), adjusted_squared=_distance_sq(actual, mean),
                masked_squared=_distance_sq(masked, ZERO), adjusted_masked_squared=_distance_sq(masked, mean),
                variance=variance, linear=linear, masked_linear=masked_linear, quadratic=quadratic,
                delta_cost=linear + quadratic, masked_delta_cost=masked_linear + quadratic)


def score_snapshot(snapshot):
    """Return raw controls, mean correction and a veto conditional on agreement.

    All contrasts are retain cost minus release cost. The same earlier anchor
    shifts the actual and masked residuals of both alternatives. The reporting
    direction and forecast horizon qualifications come from frozen C-ON41 reads.
    """
    actual, masked, variances = [tuple(snapshot[name][side] for side in SIDES)
                                for name in ("cross", "pred_delta", "variance")]
    raw, masked_score = (snapshot["scores"][key] for key in ("raw", "masked"))
    common_variance = sum(variances) / 2

    def common_margin(residuals, mean):
        release, retain = residuals
        return (_distance_sq(retain, mean) - _distance_sq(release, mean)) / common_variance

    equal = dict(raw=common_margin(actual, ZERO), masked=common_margin(masked, ZERO),
                 always=None, always_masked=None)
    scores = dict(raw=raw, masked=masked_score, raw_agreement=0.0, always=0.0,
                  always_masked=0.0, corroborated=0.0, hold=0.0)
    if _positive(raw, masked_score, equal["raw"], equal["masked"]):
        scores["raw_agreement"] = raw
    anchor = _recent_anchor(snapshot)
    if anchor is None:
        return dict(scores=scores, reason="no_recent", anchor=None, decomposition=None,
                    equal_variance=equal, margin_delta=None)

    mean = anchor.residual
    fit = predictive_scores(*actual, *masked, *variances, {"mean": mean, "scale": None})
    scores.update(always=fit["mean"], always_masked=fit["mean_masked"])
    equal.update(always=common_margin(actual, mean), always_masked=common_margin(masked, mean))

    decomposition = {side: _terms(a, m, mean, v)
                     for side, a, m, v in zip(SIDES, actual, masked, variances, strict=True)}
    release, retain = (decomposition[side] for side in SIDES)
    delta = {field: retain[field] - release[field]
             for field in ("linear", "masked_linear", "quadratic")}
    delta.update(actual=delta["linear"] + delta["quadratic"],
                 masked=delta["masked_linear"] + delta["quadratic"])
    if not (isclose(raw + delta["actual"], scores["always"], rel_tol=1e-10, abs_tol=1e-8)
            and isclose(masked_score + delta["masked"], scores["always_masked"], rel_tol=1e-10, abs_tol=1e-8)):
        raise ValueError("the signed single-anchor margin decomposition does not balance")

    release_actual = _positive(scores["always"])
    release_masked = _positive(scores["always_masked"])
    if release_actual != release_masked:
        reason = "corrected_mask_disagrees"
    elif not release_actual:
        reason = "neither_release"
    elif not _positive(equal["always"], equal["always_masked"]):
        reason = "equal_variance_disagrees"
    else:
        reason = "eligible"
        scores["corroborated"] = scores["always"]
    return dict(scores=scores, reason=reason, anchor=anchor, decomposition=decomposition,
                equal_variance=equal, margin_delta=delta)
