"""Small physical R/S operators, not trained fusion methods.

All geometry and optimization use float64. The caller chooses one chart origin
from visible input, never a target. Risk units must be the square of the point
units (metres/metres² or kilometres/kilometres²); a risk estimate is not a
calibration guarantee. No source/event deduplication or target lookup occurs here.
"""
from __future__ import annotations

import numpy as np

EARTH_RADIUS_M = 6371000.
_ANTIPODE_MARGIN = 1e-7


def _chart(origin):
    origin = np.asarray(origin, np.float64)
    if origin.shape != (2,) or not np.isfinite(origin).all() or abs(origin[0]) > 90:
        raise ValueError('chart origin must be finite geographic [latitude, longitude]')
    lat, lon = np.deg2rad(origin)
    slat, clat, slon, clon = np.sin(lat), np.cos(lat), np.sin(lon), np.cos(lon)
    return (np.array([clat * clon, clat * slon, slat]),
            np.array([-slat * clon, -slat * slon, clat]),
            np.array([-slon, clon, 0.]))


def geographic_log(positions, origin):
    """Sphere logarithm: lat/lon degrees -> north/east metres at one origin.

This is paired with geographic_exp, unlike an equirectangular approximation
paired with a normalized tangent map. The near-antipodal chart singularity is
rejected, not clipped. A chart preserves radial distance, not all pair distances.
"""
    points = np.asarray(positions, np.float64)
    if points.ndim < 1 or points.shape[-1] != 2 or not np.isfinite(points).all() or (np.abs(points[..., 0]) > 90).any():
        raise ValueError('positions must be finite geographic [...,2]')
    center, north, east = _chart(origin)
    lat, lon = np.deg2rad(points[..., 0]), np.deg2rad(points[..., 1])
    vector = np.stack((np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)), -1)
    tangent = np.stack((vector @ north, vector @ east), -1)
    sine = np.linalg.norm(tangent, axis=-1)
    angle = np.arctan2(sine, np.clip(vector @ center, -1., 1.))
    if (angle >= np.pi - _ANTIPODE_MARGIN).any():
        raise ValueError('near-antipodal positions do not have a unique stable local chart')
    ratio = np.ones_like(angle)
    np.divide(angle, sine, out=ratio, where=sine > 1e-15)
    return tangent * (EARTH_RADIUS_M * ratio)[..., None]


def geographic_exp(coordinates, origin):
    """Inverse of geographic_log on the open sphere chart; returns lat/lon."""
    xy = np.asarray(coordinates, np.float64)
    if xy.ndim < 1 or xy.shape[-1] != 2 or not np.isfinite(xy).all():
        raise ValueError('chart coordinates must be finite [...,2] metres')
    center, north, east = _chart(origin)
    angle = np.linalg.norm(xy, axis=-1) / EARTH_RADIUS_M
    if (angle >= np.pi - _ANTIPODE_MARGIN).any():
        raise ValueError('coordinates exceed the unique non-antipodal chart')
    vector = (np.cos(angle)[..., None] * center
              + np.sinc(angle / np.pi)[..., None]
              * (xy[..., :1] * north + xy[..., 1:] * east) / EARTH_RADIUS_M)
    result = np.rad2deg(np.stack((np.arctan2(vector[..., 2], np.linalg.norm(vector[..., :2], axis=-1)),
                                 np.arctan2(vector[..., 1], vector[..., 0])), -1))
    result[..., 1] = (result[..., 1] + 180.) % 360. - 180.
    canonical_origin = np.asarray(origin, np.float64).copy()
    canonical_origin[1] = (canonical_origin[1] + 180.) % 360. - 180.
    return np.where((angle == 0)[..., None], canonical_origin, result)


def risk_simplex_weights(points, risks, valid=None, *, tolerance=1e-10, max_iterations=20000):
    """Minimize ||P.T w||² + (risk-||P_s||²).T w on the valid simplex.

Pairwise exact line minimization solves the convex simplex problem without
projecting an estimated error matrix to PSD. KKT residual is max gradient on
positive weights minus min gradient on all valid candidates. ``tolerance`` is
relative to a translation-invariant squared-distance/risk scale; diagnostics
also give the residual and objective in original squared point units.

Returns (weights[S], diagnostics). Invalid/nonfinite candidates get zero weight;
no valid candidates returns all zero and converged=False so the caller must
apply its declared fallback. A nonconverged iterate is explicitly marked and
must not be advertised as optimal. Repeated coordinates are NOT deduplicated.
Equal-cost identical candidates have nonunique weights, not causal importance.
"""
    p, r = np.asarray(points, np.float64), np.asarray(risks, np.float64)
    if p.ndim != 2 or p.shape[1] != 2 or r.shape != (len(p),):
        raise ValueError('points[S,2] and risks[S] are required')
    mask = np.ones(len(p), bool) if valid is None else np.asarray(valid)
    if mask.shape != (len(p),) or mask.dtype != np.bool_:
        raise ValueError('valid must be a boolean candidate mask')
    if not np.isfinite(tolerance) or tolerance <= 0 or type(max_iterations) is not int or max_iterations < 0:
        raise ValueError('positive tolerance and nonnegative iteration limit required')
    mask = mask & np.isfinite(p).all(1) & np.isfinite(r)
    ids = np.flatnonzero(mask)
    result = np.zeros(len(p), np.float64)
    if not len(ids):
        return result, dict(converged=False, iterations=0, valid_candidates=0,
                            kkt_residual=None, relative_kkt_residual=None,
                            objective=None, reason='no_valid_candidates')
    # Canonical order makes nondegenerate results independent of source order.
    ids = ids[np.lexsort((r[ids], p[ids, 1], p[ids, 0]))]
    centered = p[ids] - p[ids].mean(0)
    # A common risk level contributes only a constant on the simplex. It must
    # not loosen stationarity accuracy or erase genuine differences in weights.
    risk_offset = float(np.min(r[ids]))
    shifted_risk = r[ids] - risk_offset
    scale = max(1., float(np.max(np.sum(centered * centered, axis=1))), float(np.max(shifted_risk)))
    q = centered / np.sqrt(scale)
    a = shifted_risk / scale - np.sum(q * q, axis=1)
    w = np.full(len(ids), 1. / len(ids))
    iterations = 0
    while True:
        position = q.T @ w
        gradient = 2. * (q @ position) + a
        entering = int(np.argmin(gradient))
        active = np.flatnonzero(w > 0.)
        leaving = int(active[np.argmax(gradient[active])])
        gap = max(0., float(gradient[leaving] - gradient[entering]))
        if gap <= tolerance or iterations >= max_iterations:
            break
        direction = q[entering] - q[leaving]
        curvature = 2. * float(direction @ direction)
        change = w[leaving] if curvature == 0. else min(w[leaving], gap / curvature)
        if change <= 0. or not np.isfinite(change):
            break
        w[entering] += change
        w[leaving] = max(0., w[leaving] - change)
        iterations += 1
    result[ids] = w
    objective = float(position @ position + a @ w) * scale + risk_offset
    return result, dict(converged=gap <= tolerance, iterations=iterations,
                        valid_candidates=len(ids), kkt_residual=gap * scale,
                        relative_kkt_residual=gap, objective=objective,
                        squared_unit_scale=scale,
                        reason='converged' if gap <= tolerance else 'iteration_limit_or_stall')


def smooth_bounded_positions(base, times, strength, radius, *, tau=20.,
                             tolerance=1e-10, max_iterations=20000):
    """Solve ||X-B||² + sum(lambda * ||tau² D_time² X||²), |X-B|<=r.

``base[Q,2]`` uses one fixed local metre chart for the complete trajectory.
``times`` are true increasing seconds, ``tau`` is a fixed time unit in seconds,
and strength is a nonnegative scalar or Q-2 array. Radius is metres PER
COORDINATE: Euclidean displacement is bounded by sqrt(2)*radius, not radius.
The two coordinates are separate strictly convex box QPs. Clipping is used
inside convergent coordinate descent, never as a claimed one-shot optimum.

Returns (positions, diagnostics). Nonconvergence returns exact B and records
fallback; no output rows are removed. Zero strength/radius returns B exactly.
This CPU operator is not an autograd training layer.
"""
    b, t = np.asarray(base, np.float64), np.asarray(times, np.float64)
    if (b.ndim != 2 or b.shape[1] != 2 or len(b) == 0 or t.shape != (len(b),)
            or not np.isfinite(b).all() or not np.isfinite(t).all() or (np.diff(t) <= 0).any()):
        raise ValueError('finite base[Q,2] and strictly increasing times[Q] required')
    if (not np.isfinite(radius) or radius < 0 or not np.isfinite(tau) or tau <= 0
            or not np.isfinite(tolerance) or tolerance <= 0
            or type(max_iterations) is not int or max_iterations < 0):
        raise ValueError('finite nonnegative radius, positive tau/tolerance and valid iteration limit required')
    n = max(0, len(b) - 2)
    lam = np.asarray(strength, np.float64)
    if lam.ndim == 0:
        lam = np.full(n, float(lam)) if np.isfinite(lam) and lam >= 0 else np.array([np.nan])
    if lam.shape != (n,) or not np.isfinite(lam).all() or (lam < 0).any():
        raise ValueError('strength must be a nonnegative scalar or Q-2 vector')
    if not n or radius == 0 or not np.any(lam):
        return b.copy(), dict(converged=True, fallback=False, iterations=0,
                             kkt_residual=0., max_coordinate_change=0., reason='identity')
    h = np.diff(t)
    d = np.zeros((n, len(b)), np.float64)
    row = np.arange(n)
    factor = 2. * tau ** 2 / (h[:-1] + h[1:])
    d[row, row] = factor / h[:-1]
    d[row, row + 1] = -factor * (1. / h[:-1] + 1. / h[1:])
    d[row, row + 2] = factor / h[1:]
    penalty = d.T @ (lam[:, None] * d)
    a = np.eye(len(b)) + penalty
    centered = b - b.mean(0)
    c = d.T @ (lam[:, None] * (d @ centered))
    if not np.isfinite(a).all() or not np.isfinite(c).all():
        return b.copy(), dict(converged=False, fallback=True, iterations=0,
                             kkt_residual=None, max_coordinate_change=0., reason='nonfinite_system')
    try:
        u = np.clip(np.linalg.solve(a, -c), -radius, radius)
    except np.linalg.LinAlgError:
        return b.copy(), dict(converged=False, fallback=True, iterations=0,
                             kkt_residual=None, max_coordinate_change=0., reason='linear_solve_failure')
    threshold = tolerance * max(1., float(np.max(np.abs(centered))))
    iterations = 0
    while True:
        g = a @ u + c  # Half-gradient of the stated squared-error objective.
        violation = np.where(u == -radius, np.minimum(g, 0.),
                             np.where(u == radius, np.maximum(g, 0.), g))
        residual = float(np.max(np.abs(violation)))
        if residual <= threshold or iterations >= max_iterations:
            break
        for i in range(len(b)):
            updated = np.clip(u[i] - g[i] / a[i, i], -radius, radius)
            change = updated - u[i]
            u[i] = updated
            g += a[:, i, None] * change[None, :]
        iterations += 1
    converged = bool(np.isfinite(residual) and residual <= threshold)
    return (b + u if converged else b.copy()), dict(
        converged=converged, fallback=not converged, iterations=iterations,
        kkt_residual=residual, kkt_tolerance=threshold,
        max_coordinate_change=float(np.max(np.abs(u))) if converged else 0.,
        reason='converged' if converged else 'iteration_limit_or_stall')
