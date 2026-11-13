"""Decompose a frozen single-report geometry increment and its nominal scale."""
from __future__ import annotations

from math import hypot, isfinite, sqrt

from .continuous_tracker import POSITION_VARIANCE
from .cross_source_calibration import predictive_scores
from .interval_assignment import _dot
from .relative_motion_gap import _same, _sub


def score_increment(sample, geometry):
    """Use only C-ON41 batch-prior forecasts and its one arrived report.

    A positive margin favors release. In the full-group Euclidean increment,
    all unchanged edges cancel, leaving twice the dot product of the reporter
    innovation and the difference between its release and retain forecasts.
    Joint scaling additionally treats the two partner predictions as mutually
    independent and independent of the reporter innovation. It is a nominal
    scale under a zero-mean isotropic innovation, not a calibrated likelihood.
    """
    actual, masked = sample["cross"], sample["pred_delta"]
    release_variance, retain_variance = (sample["variance"][side] for side in ("release", "retain"))
    reporter_variance = sample["self_variance"]
    variances = release_variance, retain_variance, reporter_variance
    if any(not isfinite(value) or value < POSITION_VARIANCE for value in variances):
        raise ValueError("the frozen innovation variances must include the observation variance")

    innovation = sample["self"]
    axis = _sub(masked["retain"], masked["release"])
    separation = hypot(*axis)
    increment = geometry["actual"] - geometry["masked"]
    residuals = {}
    for side in ("release", "retain"):
        if not all(_same(observed, expected) for observed, expected in
                   zip(_sub(actual[side], masked[side]), innovation, strict=True)):
            raise ValueError("a partner residual does not include the same report innovation")
        observed, prediction = _dot(actual[side], actual[side]), _dot(masked[side], masked[side])
        residuals.update({f"{side}_actual_m2": observed, f"{side}_masked_m2": prediction,
                          f"{side}_change_m2": observed - prediction})
    residual_increment = residuals["retain_change_m2"] - residuals["release_change_m2"]
    if not (_same(increment, 2 * _dot(innovation, axis)) and _same(increment, residual_increment)):
        raise ValueError("the single-report increment does not reproduce frozen group geometry")

    # Each C-ON41 partner innovation variance includes the same measurement
    # variance; only the forecast part contributes to axis uncertainty.
    partner_forecast_variance = release_variance + retain_variance - 2 * POSITION_VARIANCE
    joint_sd = 2 * sqrt(reporter_variance * (separation**2 + 2 * partner_forecast_variance))
    projected = _dot(innovation, axis) / separation if separation else None
    scores = predictive_scores(actual["release"], actual["retain"], masked["release"], masked["retain"],
                               release_variance, retain_variance, None)
    return dict(actual_m2=geometry["actual"], masked_m2=geometry["masked"], increment_m2=increment,
                separation_m=separation, projected_m=projected,
                relative_increment=projected / separation if projected is not None else None,
                reporter_z=projected / sqrt(reporter_variance) if projected is not None else None,
                joint_z=increment / joint_sd if joint_sd else None,
                innovation_norm_m=hypot(*innovation), reporter_variance_m2=reporter_variance,
                partner_forecast_variance_m2=partner_forecast_variance,
                gaussian_actual=scores["raw"], gaussian_masked=scores["masked"],
                gaussian_increment=scores["raw"]-scores["masked"], **residuals)
