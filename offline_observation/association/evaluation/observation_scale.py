"""Reference-blind heldout-observation selection of a working CV scale.

The target is a noisy native observation, not a latent-state reference. Full
retained-window charts and RTS smoothing make this an offline diagnostic.
"""
from __future__ import annotations

import numpy as np

from association.evaluation.source_risk import _position_distance

RHOS = tuple(10.**k for k in range(-10, -1))
WORK_KEYS = ('native_event_count', 'kalman_observation_updates', 'transition_steps',
             'rts_backward_steps', 'inside_query_count', 'outside_query_count')


def _rhos(rhos):
    result = np.asarray(rhos, np.float64)
    if result.shape != (len(RHOS),) or not np.array_equal(result, RHOS):
        raise ValueError('the original nine fixed increasing rhos are required')
    return result


def split_native_observations(times, latlon):
    times, latlon = np.asarray(times, np.float64), np.asarray(latlon, np.float64)
    if (times.ndim != 1 or len(times) < 2 or latlon.shape != (len(times), 2)
            or not np.isfinite(times).all() or (np.diff(times) <= 0).any()
            or not np.isfinite(latlon).all() or (np.abs(latlon[:, 0]) > 90).any()
            or (np.abs(latlon[:, 1]) > 180).any()):
        raise ValueError('finite ordered native times and legal latlon are required')
    index = np.arange(len(times), dtype=np.int64)
    held = (index > 0) & (index < len(times)-1) & (index % 3 == 1)
    return dict(retained_indices=index[~held], heldout_indices=index[held],
        retained_times=times[~held].copy(), retained_latlon=latlon[~held].copy(),
        heldout_times=times[held].copy(), heldout_latlon=latlon[held].copy(),
        eligible=bool(held.any()))


def heldout_rho_errors(*, times, latlon, rhos=RHOS, check=None):
    # The established 127 kernel is deliberately reused, not reimplemented with
    # different priors, observation updates, chart geometry, or extrapolation.
    from tools.analysis.native_cv_state_control import cv_native_positions

    candidates = _rhos(rhos)
    split = split_native_observations(times, latlon)
    h = len(split['heldout_indices'])
    forward = np.empty((len(candidates), h, 2), np.float64)
    smoothed = np.empty_like(forward)
    errors = np.zeros((len(candidates), h), np.float64)
    audit = {key: 0 for key in WORK_KEYS}
    if split['eligible']:
        for i, rho in enumerate(candidates):
            # Heldout positions are absent from every estimator argument; their
            # known timestamps only define requested predictions.
            predictions, work = cv_native_positions(split['retained_times'],
                split['retained_latlon'], split['heldout_times'], float(rho))
            forward[i], smoothed[i] = predictions['forward'], predictions['rts']
            errors[i] = _position_distance(smoothed[i], split['heldout_latlon'])**2
            for key in WORK_KEYS:
                audit[key] += int(work[key])
            if check is not None:
                check()
    mse = errors.mean(1) if h else np.zeros(len(candidates), np.float64)
    if not np.isfinite(mse).all():
        raise FloatingPointError('nonfinite native heldout squared distance')
    return dict(**split, mse_m2=mse, rts_positions=smoothed,
        forward_positions=forward, squared_errors_m2=errors, analytic_work=audit,
        analytic_filter_smoother_fits=len(candidates) if h else 0,
        native_event_count=len(times), retained_event_count=len(split['retained_indices']),
        heldout_event_count=h)


def select_observation_rho(scene_fragment_errors, *, rhos=RHOS):
    candidates = _rhos(rhos)
    rows, criteria, seen = [], [], set()
    for item in scene_fragment_errors:
        sid = str(item['scene_id'])
        if sid in seen:
            raise ValueError('duplicate scene in observation selection')
        seen.add(sid)
        mse, eligible = np.asarray(item['mse_m2'], np.float64), np.asarray(item['eligible'])
        if (mse.ndim != 2 or mse.shape[1] != len(candidates)
                or eligible.shape != (len(mse),) or eligible.dtype != bool
                or not np.isfinite(mse).all() or (mse < 0).any()):
            raise ValueError('finite nonnegative fragment-by-rho errors and eligibility required')
        value = mse[eligible].mean(0) if eligible.any() else None
        if value is not None:
            criteria.append(value)
        rows.append(dict(scene_id=sid, fragment_count=len(mse),
            eligible_fragment_count=int(eligible.sum()), missing_fragment_count=int((~eligible).sum()),
            objective_by_rho=None if value is None else value.tolist()))
    if not criteria:
        raise ValueError('no eligible native heldout observations in T; cannot select rho')
    objective = np.mean(criteria, axis=0)
    selected = int(np.argmin(objective))
    return dict(rhos=candidates.tolist(), objective_by_rho=objective.tolist(),
        selected_index=selected, selected_rho=float(candidates[selected]),
        tie_count=int((objective == objective[selected]).sum()),
        scene_count=len(rows), eligible_scene_count=len(criteria),
        missing_scene_count=len(rows)-len(criteria), per_scene=rows,
        selection='RTS scene-equal then eligible-fragment-equal then heldout-point mean radial squared metres; exact ties lower rho',
        forward_uses_same_rho=True, reference_used=False, criterion_unit='m2',
        time_permission='offline full retained window; endpoints retained; no online claim')
