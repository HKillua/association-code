"""Separate common and differential innovations in two opposite reports."""
from __future__ import annotations

from math import hypot
from statistics import fmean

from .interval_assignment import _dot
from .rejoin_async_innovation import _margins
from .relative_motion_gap import _sub


_SIDES = ("release", "retain")
_ARMS = ("raw", "demeaned", "external", "equal_shrink", "shared_external")
_SCORES = (
    "actual_m2", "masked_m2", "increment_m2", "release_actual_m2", "retain_actual_m2",
    "release_masked_m2", "retain_masked_m2", "release_change_m2", "retain_change_m2",
)


def _scale(point, factor):
    return point[0] * factor, point[1] * factor


def _mean(left, right):
    return fmean((left[0], right[0])), fmean((left[1], right[1]))


def _capped_corrections(donor, innovation):
    donor_norm, own_norm = hypot(*donor), hypot(*innovation)
    applied_norm = min(donor_norm, own_norm)
    return (_scale(donor, applied_norm / donor_norm) if donor_norm else (0.0, 0.0),
            _scale(innovation, applied_norm / own_norm) if own_norm else (0.0, 0.0))


def _arm(geometry, corrections):
    reports = {}
    for side in _SIDES:
        row, correction = geometry[side], corrections[side]
        if row is None or correction is None:
            reports[side] = None
            continue
        reports[side] = _margins(_sub(row["actual"], correction), row["masked"], row["release"], row["retain"])
    usable = [report for report in reports.values() if report is not None]
    scores = ({name: fmean(report[name] for report in usable) for name in _SCORES}
              if len(usable) == 2 else dict.fromkeys(_SCORES))
    return dict(support_count=len(usable), reports=reports, **scores)


def _squared(vector):
    return _dot(vector, vector)


def compare_common_innovations(geometry, donors):
    """Compare frozen native-time reports using the same masks and partner forecasts.

    The two dictionaries are keyed by report role, with None for unavailable
    geometry or external estimates. Geometry contains each report's actual and
    masked coordinates in the partner frame and the release/retain forecast
    coordinates. Donor means are supplied by a source-local history excluding
    the four involved tracks. The same-pair mean is an algebraic diagnostic;
    it does not independently estimate a source error.
    """
    present = [side for side in _SIDES if geometry[side] is not None]
    innovations = {side: _sub(geometry[side]["actual"], geometry[side]["masked"]) for side in present}
    axes = {side: _sub(geometry[side]["release"], geometry[side]["retain"]) for side in present}
    corrections = {arm: dict.fromkeys(_SIDES) for arm in _ARMS}
    corrections["raw"] = dict.fromkeys(_SIDES, (0.0, 0.0))
    decomposition = None
    if len(present) == 2:
        e_first, e_second = (innovations[side] for side in _SIDES)
        axis_first, axis_second = (axes[side] for side in _SIDES)
        mean_innovation, differential_innovation = _mean(e_first, e_second), _scale(_sub(e_first, e_second), 0.5)
        mean_axis, differential_axis = _mean(axis_first, axis_second), _scale(_sub(axis_first, axis_second), 0.5)
        decomposition = dict(common_increment_m2=2 * _dot(mean_innovation, mean_axis),
                             differential_increment_m2=2 * _dot(differential_innovation, differential_axis),
                             mean_innovation_m=mean_innovation, differential_innovation_m=differential_innovation,
                             mean_axis_m=mean_axis, differential_axis_m=differential_axis)
        corrections["demeaned"] = dict.fromkeys(_SIDES, mean_innovation)

    first_donor, second_donor = (donors[side] for side in _SIDES)
    donor_dot = None
    if len(present) == 2 and first_donor is not None and second_donor is not None:
        corrections["shared_external"] = dict.fromkeys(_SIDES, _mean(first_donor, second_donor))
        donor_dot = _dot(first_donor, second_donor)

    reports = dict.fromkeys(_SIDES)
    for side in present:
        innovation, donor = innovations[side], donors[side]
        applied, shrink = _capped_corrections(donor, innovation) if donor is not None else (None, None)
        corrections["external"][side], corrections["equal_shrink"][side] = applied, shrink
        raw_residual = _squared(innovation)
        external_raw = _squared(_sub(innovation, donor)) if donor is not None else None
        external_applied = _squared(_sub(innovation, applied)) if applied is not None else None
        reports[side] = dict(
            innovation_m=innovation, axis_m=axes[side], donor_m=donor, donor_applied_m=applied,
            innovation_squared_m2=raw_residual, external_raw_residual_m2=external_raw,
            external_applied_residual_m2=external_applied,
            external_raw_reduction_m2=raw_residual - external_raw if external_raw is not None else None,
            external_applied_reduction_m2=raw_residual - external_applied if external_applied is not None else None,
            shrink_residual_m2=_squared(_sub(innovation, shrink)) if shrink is not None else None,
            demeaned_residual_m2=_squared(_sub(innovation, corrections["demeaned"][side]))
            if decomposition is not None else None,
        )

    return dict(reports=reports, decomposition=decomposition, donor_dot_m2=donor_dot,
                **{arm: _arm(geometry, corrections[arm]) for arm in _ARMS})
