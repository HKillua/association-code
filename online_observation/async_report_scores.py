"""Compare frozen associations using each newly arrived position report."""
from __future__ import annotations

import math
from statistics import fmean

from .continuous_tracker import ACCELERATION_SPECTRAL_DENSITY, POSITION_VARIANCE
from .hypothesis_gap_diagnostic import _margin, _position_readouts
from .nemot_replay import _distance_sq


def reported_direction(value: float) -> int:
    """Map a dimensionless margin to release (+1), retain (-1), or a tie."""
    if value > 1e-9:
        return 1
    if value < -1e-9:
        return -1
    return 0


def _innovation_variance(track, age):
    return (POSITION_VARIANCE + track.pp + 2 * age * track.pv + age**2 * track.vv
            + ACCELERATION_SPECTRAL_DENSITY * age**3 / 3)


def _report_margins(reports, *, masked):
    suffix = "_masked_squared" if masked else "_squared"
    release = [row[f"release{suffix}"] for row in reports]
    retain = [row[f"retain{suffix}"] for row in reports]
    differences = [hold - swap for hold, swap in zip(retain, release, strict=True)]
    return {
        "euclidean": fmean(differences),
        "common": fmean(2 * delta / (row["retain_variance"] + row["release_variance"])
                        for row, delta in zip(reports, differences, strict=True)),
        "mahalanobis": fmean(hold / row["retain_variance"] - swap / row["release_variance"]
                             for row, hold, swap in zip(reports, retain, release, strict=True)),
    }


def score_batch(switch, stamp, points, priors, shifts):
    """Score actual and masked positions against identical batch-prior forecasts.

    Each likelihood term compares the current report with the two proposed
    partners at its native timestamp. Candidate forecasts use only earlier
    batches, including when a proposed partner also reports in this batch.
    Averaging gives a score per report without counting simultaneous terms as
    independent. Positive margins favor release for every readout.
    """
    if not switch.keys.intersection(points):
        return None
    readouts = _position_readouts(switch.keys, stamp, points, priors, shifts)
    if readouts is None:
        return None
    masked, actual, new = readouts
    ages = {key: stamp - priors[key].stamp for key in masked}
    variance = {key: _innovation_variance(priors[key], age) for key, age in ages.items()}
    partners = [{key: other for left, right in edges for key, other in ((left, right), (right, left))}
                for edges in (switch.release, switch.retain)]
    reports = []
    for key in sorted(new):
        release_key, retain_key = (partner[key] for partner in partners)
        reports.append(dict(
            key=key, age=ages[key], release_key=release_key, retain_key=retain_key,
            release_age=ages[release_key], retain_age=ages[retain_key],
            release_variance=variance[release_key], retain_variance=variance[retain_key],
            release_squared=_distance_sq(actual[key], masked[release_key]),
            retain_squared=_distance_sq(actual[key], masked[retain_key]),
            release_masked_squared=_distance_sq(masked[key], masked[release_key]),
            retain_masked_squared=_distance_sq(masked[key], masked[retain_key]),
            self_squared=_distance_sq(actual[key], masked[key]), self_variance=variance[key],
        ))
    log_volume = fmean(2 * math.log(row["retain_variance"] / row["release_variance"]) for row in reports)
    result = dict(reports=tuple(reports), log_volume=log_volume)
    for name, coordinates in (("actual", actual), ("masked", masked)):
        result[f"position_{name}"] = _margin(switch, lambda edge: _distance_sq(*(coordinates[key] for key in edge)))
        margins = _report_margins(reports, masked=name == "masked")
        result.update({f"{metric}_{name}": value for metric, value in margins.items()})
        result[f"likelihood_{name}"] = margins["mahalanobis"] + log_volume
    return result
