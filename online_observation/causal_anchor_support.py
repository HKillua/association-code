"""Transfer a translation from independent, already observed witness pairs."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from itertools import combinations
from math import dist
from statistics import median
from typing import Iterable


Fragment = tuple[str, str]
Point = tuple[float, float]
Pair = tuple[Fragment, Fragment]
CONSISTENCY_M = 500


@dataclass(frozen=True)
class AnchorVote:
    """A motion-qualified pair and its three co-timed right-minus-left offsets."""

    left: Fragment
    right: Fragment
    offsets: tuple[Point, Point, Point]


@dataclass(frozen=True)
class AnchorFit:
    status: str
    translation: Point | None
    single_translation: Point | None
    witnesses: tuple[Pair, ...]
    candidate_pairs: int
    unique_pairs: int
    stable_pairs: int


def _centre(points: Iterable[Point]) -> Point:
    east, north = zip(*points)
    return median(east), median(north)


def fit_causal_anchor(votes: Iterable[AnchorVote], left: Fragment, right: Fragment) -> AnchorFit:
    """Fit a common shift from a single decision's arrived witness candidates.

    The caller supplies only votes available at the decision time. Source order
    fixes the orientation of the shift. A stable single witness is available
    for diagnosis and does not produce an applied translation.
    """
    if left[0] >= right[0]:
        raise ValueError("target endpoints must use increasing source order")

    candidates = {
        vote for vote in votes
        if (vote.left[0], vote.right[0]) == (left[0], right[0])
        and vote.left != left and vote.right != right
    }
    degree = Counter(key for vote in candidates for key in (vote.left, vote.right))
    unique = [vote for vote in candidates if degree[vote.left] == degree[vote.right] == 1]
    stable = {}
    for vote in unique:
        centre = _centre(vote.offsets)
        if all(dist(centre, offset) <= CONSISTENCY_M for offset in vote.offsets):
            stable[(vote.left, vote.right)] = centre

    translation = single_translation = None
    if not candidates:
        status = "none"
    elif not unique:
        status = "ambiguous"
    elif not stable:
        status = "unstable"
    elif len(stable) == 1:
        status = "single"
        single_translation = next(iter(stable.values()))
    elif any(dist(first, second) > CONSISTENCY_M for first, second in combinations(stable.values(), 2)):
        status = "disagreement"
    else:
        status = "fitted"
        translation = _centre(stable.values())

    return AnchorFit(status, translation, single_translation, tuple(sorted(stable)),
                     len(candidates), len(unique), len(stable))
