"""Select disjoint historical anchors and separate residual location from spread."""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from math import isfinite, log
from statistics import fmean

from .continuous_tracker import MEMORY_S
from .nemot_replay import Edge, Fragment, Point, _distance_sq


MAX_AGE_DIFFERENCE_S = 10.0
ZERO = (0.0, 0.0)


@dataclass(frozen=True)
class AnchorSample:
    """An earlier signed report-minus-opposite-forecast residual."""

    edge: Edge
    reporter: Fragment
    stamp: float
    age: float
    residual: Point
    variance: float
    motion: bool
    registered: bool


def eligible_anchors(records: Iterable[AnchorSample], keys, reporter, stamp, ages, registered, *, motion=False):
    """Pick the latest disjoint anchors on the same direction, clock and horizon."""
    excluded = set(keys)
    sources = {key[0] for key in excluded}
    release_age, retain_age = ages
    candidates = (
        record for record in records
        if (0 < stamp - record.stamp <= MEMORY_S
            and record.registered == registered
            and record.reporter[0] == reporter[0]
            and {key[0] for key in record.edge} == sources
            and abs(record.age - release_age) <= MAX_AGE_DIFFERENCE_S
            and abs(record.age - retain_age) <= MAX_AGE_DIFFERENCE_S
            and (record.motion or not motion))
    )
    anchors = []
    for record in sorted(candidates, key=lambda sample: (-sample.stamp, sample.edge)):
        if excluded.isdisjoint(record.edge):
            anchors.append(record)
            excluded.update(record.edge)
    return tuple(anchors)


def decompose(vectors: Iterable[Point]):
    """Decompose the average squared residual into squared mean and centered trace."""
    vectors = tuple(vectors)
    if not vectors:
        return dict(n=0, mean=None, mean_squared=None, mean_norm_squared=None, centered_spread=None)
    centre = tuple(map(fmean, zip(*vectors, strict=True)))
    return dict(n=len(vectors), mean=centre, mean_squared=fmean(_distance_sq(point, ZERO) for point in vectors),
                mean_norm_squared=_distance_sq(centre, ZERO),
                centered_spread=fmean(_distance_sq(point, centre) for point in vectors))


def fit_moments(samples: Iterable[AnchorSample]):
    """Fit a relative mean and, when supported, a centered variance multiplier."""
    samples = tuple(samples)
    mean = decompose(sample.residual for sample in samples)["mean"]
    scale = None
    if len(samples) >= 3:
        centered = sum(_distance_sq(sample.residual, mean) / sample.variance for sample in samples)
        estimate = centered / (2 * (len(samples) - 1))
        if isfinite(estimate) and estimate > 0:
            scale = estimate
    return dict(mean=mean, scale=scale)


def predictive_scores(release, retain, release_masked, retain_masked, release_variance, retain_variance, fit):
    """Compare doubled Gaussian negative log likelihoods; positive favors release."""
    volume = 2 * log(retain_variance / release_variance)

    def contrast(first, second, mean):
        return (_distance_sq(second, mean) / retain_variance
                - _distance_sq(first, mean) / release_variance)

    raw = contrast(release, retain, ZERO)
    masked = contrast(release_masked, retain_masked, ZERO)
    mean, scale = (fit["mean"], fit["scale"]) if fit is not None else (None, None)
    result = dict(raw=raw + volume, masked=masked + volume,
                  mean=None, mean_masked=None, scale=None, both=None, both_masked=None)
    if scale is not None:
        # A common variance multiplier cancels from the log-volume term.
        result["scale"] = raw / scale + volume
    if mean is not None:
        centered = contrast(release, retain, mean)
        centered_masked = contrast(release_masked, retain_masked, mean)
        result.update(mean=centered + volume, mean_masked=centered_masked + volume)
        if scale is not None:
            result.update(both=centered / scale + volume, both_masked=centered_masked / scale + volume)
    return result
