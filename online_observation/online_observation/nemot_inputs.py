"""Build arrival-only motion features and same-fragment forecast supervision."""
from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, replace
from hashlib import blake2b
from heapq import nsmallest
import json
import math

import numpy as np
import torch

from .continuous_tracker import FRESH_S, MEMORY_S, RADIUS_M, LocalTrack, Plane
from .nemot_motion import HISTORY_STEPS, MAX_NEIGHBORS, MotionBatch


HISTORY_SECONDS = 120
NEIGHBOR_POINTS = 5
SAMPLES_PER_SCENE = 384
MAX_TARGET_REPORTS = 3


@dataclass(frozen=True)
class Observation:
    stamp: float
    previous_stamp: float
    own: tuple[float, float, float, float, float]
    neighbors: tuple[tuple[float, ...], ...] = ()
    weights: tuple[float, ...] = ()


@dataclass(frozen=True)
class MotionExample:
    history: tuple[Observation, ...]
    horizon: float
    center: tuple[float, float]
    target: tuple[float, float] | None


class MotionInputs:
    """Share per-key KF and observed history across replay and learning."""

    def __init__(self):
        self.tracks = {}
        self.history = defaultdict(lambda: deque(maxlen=HISTORY_STEPS))
        self.source_keys = defaultdict(set)
        self.stamp = -math.inf

    def observe(self, stamp, points):
        if stamp <= self.stamp:
            raise ValueError("motion inputs require strictly increasing native times")
        self.stamp = stamp
        for key, point in points.items():
            prior = self.tracks.get(key)
            if prior is not None and stamp - prior.stamp <= MEMORY_S:
                state = prior.updated(stamp, point)
            else:
                state = LocalTrack(stamp, point, point)
            if prior is not None and stamp - prior.stamp <= HISTORY_SECONDS:
                dx = point[0] - prior.observation[0]
                dy = point[1] - prior.observation[1]
                dt = stamp - prior.stamp
                previous_stamp = prior.stamp
            else:
                dx = dy = dt = 0.0
                previous_stamp = stamp
            self.tracks[key] = state
            self.history[key].append(Observation(stamp, previous_stamp, (dx, dy, dt, *state.velocity)))
            self.source_keys[key[0]].add(key)

        for key in points:
            feature, weights = self._neighbors(key, stamp)
            self.history[key][-1] = replace(self.history[key][-1], neighbors=feature, weights=weights)

    def _neighbors(self, key, stamp):
        east, north = self.tracks[key].observation
        candidates = []
        for other in self.source_keys[key[0]]:
            if other == key:
                continue
            state = self.tracks[other]
            age = stamp - state.stamp
            if age > FRESH_S:
                continue
            recent = self.history[other]
            if len(recent) < NEIGHBOR_POINTS or stamp - recent[-NEIGHBOR_POINTS].stamp > HISTORY_SECONDS:
                continue
            dx, dy = state.observation[0] - east, state.observation[1] - north
            distance = math.hypot(dx, dy)
            if distance <= RADIUS_M:
                # The physical feature tuple breaks boundary ties independent of key names.
                candidates.append((distance, (dx, dy, *state.velocity, age)))
        nearest = nsmallest(MAX_NEIGHBORS, candidates)
        inverse = [1.0 / max(distance, 1.0) for distance, _ in nearest]
        total = math.fsum(inverse)
        return tuple(values for _, values in nearest), tuple(weight / total for weight in inverse)

    def example(self, key, query, target=None):
        """Use only the history available at the most recent observe call."""
        track = self.tracks[key]
        horizon = query - track.stamp
        if not 0 <= horizon <= FRESH_S:
            raise ValueError("forecast query must be no more than 60 seconds after the last report")
        cutoff = track.stamp - HISTORY_SECONDS
        history = tuple(record for record in self.history[key] if record.stamp >= cutoff)
        if history[0].previous_stamp < cutoff:
            oldest = history[0]
            history = (replace(oldest, own=(0.0, 0.0, 0.0, *oldest.own[3:])), *history[1:])
        return MotionExample(history, horizon, track.position(query, True), target)


def motion_batch(examples):
    count = len(examples)
    own = np.zeros((count, HISTORY_STEPS, 5), dtype=np.float32)
    neighbors = np.zeros((count, HISTORY_STEPS, MAX_NEIGHBORS, 5), dtype=np.float32)
    neighbor_weights = np.zeros((count, HISTORY_STEPS, MAX_NEIGHBORS), dtype=np.float32)
    lengths = np.empty(count, dtype=np.int64)
    horizon = np.empty(count, dtype=np.float32)
    center = np.empty((count, 2), dtype=np.float64)
    supervised = bool(count and examples[0].target is not None)
    target = np.empty_like(center) if supervised else None
    for index, example in enumerate(examples):
        if (example.target is not None) != supervised:
            raise ValueError("cannot mix scored targets and inference queries")
        lengths[index] = len(example.history)
        horizon[index] = example.horizon
        center[index] = example.center
        if supervised:
            target[index] = example.target
        for step, record in enumerate(example.history):
            own[index, step] = record.own
            if record.neighbors:
                number = len(record.neighbors)
                neighbors[index, step, :number] = record.neighbors
                neighbor_weights[index, step, :number] = record.weights
    return MotionBatch(torch.from_numpy(own), torch.from_numpy(neighbors), torch.from_numpy(neighbor_weights),
                       torch.from_numpy(lengths), torch.from_numpy(horizon), torch.from_numpy(center),
                       torch.from_numpy(target) if target is not None else None)


def _ticket(kind, scene, key, stamp):
    record = json.dumps(["MTAD-L1-NEMOT-MOTION-27", kind, scene, key, float(stamp).hex()], separators=(",", ":"))
    return int.from_bytes(blake2b(record.encode(), digest_size=8).digest(), "big")


def _anchors(scene, series):
    for key, reports in series.items():
        for index in range(1, len(reports) - 1):
            stamp, _ = reports[index]
            if stamp - reports[index - 1][0] > HISTORY_SECONDS:
                continue
            choices = [future for future in reports[index + 1:index + 1 + MAX_TARGET_REPORTS]
                       if future[0] - stamp <= FRESH_S]
            if choices:
                future_stamp, target = choices[_ticket("horizon", scene, key, stamp) % len(choices)]
                yield _ticket("anchor", scene, key, stamp), stamp, key, future_stamp, target


def motion_examples(scene, events):
    """Draw capped forecast targets, then build each input at its own anchor."""
    if not events:
        return []
    series = defaultdict(list)
    for stamp, points in events:
        for key, point in points.items():
            series[key].append((stamp, point))
    selected = nsmallest(SAMPLES_PER_SCENE, _anchors(scene, series))
    by_anchor = {(stamp, key): (future, point) for _, stamp, key, future, point in selected}
    last_anchor = max((stamp for _, stamp, *_ in selected), default=-math.inf)
    plane = Plane.from_first_batch(events[0][1].values())
    inputs = MotionInputs()
    examples = []
    for stamp, positions in events:
        if stamp > last_anchor:
            break
        inputs.observe(stamp, {key: plane.project(point) for key, point in positions.items()})
        for key in sorted(positions):
            if (stamp, key) in by_anchor:
                future, target = by_anchor[stamp, key]
                examples.append(inputs.example(key, future, plane.project(target)))
    return examples
