"""Position-only continuous tracking with a persistent, source-aware identity."""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import math
from statistics import fmean

from .continuous_assignment import solve_global, solve_greedy
from .registration_probe import FIT_STOP_S
from .stream import EARTH_RADIUS_M


FRESH_S = 60
MEMORY_S = 600
RADIUS_M = 4000
NULL_COST = RADIUS_M**2 / 2
POSITION_VARIANCE = 150.0**2
VELOCITY_VARIANCE = 12.0**2
ACCELERATION_SPECTRAL_DENSITY = 0.1
ARMS = ("hold_local", "hold_global", "kf_local", "kf_global")


@dataclass(frozen=True)
class Plane:
    latitude: float
    longitude: float

    @classmethod
    def from_first_batch(cls, positions):
        coordinates = sorted(positions)
        angles = [math.radians(longitude) for _, longitude in coordinates]
        longitude = math.atan2(math.fsum(map(math.sin, angles)), math.fsum(map(math.cos, angles)))
        return cls(math.radians(fmean(latitude for latitude, _ in coordinates)), longitude)

    def project(self, point):
        latitude, longitude = map(math.radians, point)
        east = math.remainder(longitude - self.longitude, math.tau) * math.cos(self.latitude)
        return EARTH_RADIUS_M * east, EARTH_RADIUS_M * (latitude - self.latitude)


@dataclass(frozen=True)
class LocalTrack:
    stamp: float
    observation: tuple[float, float]
    location: tuple[float, float]
    velocity: tuple[float, float] = (0.0, 0.0)
    pp: float = POSITION_VARIANCE
    pv: float = 0.0
    vv: float = VELOCITY_VARIANCE

    def updated(self, stamp, observation):
        dt = stamp - self.stamp
        q = ACCELERATION_SPECTRAL_DENSITY
        pp = self.pp + 2 * dt * self.pv + dt**2 * self.vv + q * dt**3 / 3
        pv = self.pv + dt * self.vv + q * dt**2 / 2
        vv = self.vv + q * dt
        innovation_variance = pp + POSITION_VARIANCE
        innovations = tuple(z - (x + dt * v) for z, x, v in zip(observation, self.location, self.velocity))
        location = tuple(x + dt * v + pp / innovation_variance * residual
                         for x, v, residual in zip(self.location, self.velocity, innovations))
        velocity = tuple(v + pv / innovation_variance * residual
                         for v, residual in zip(self.velocity, innovations))
        return LocalTrack(stamp, observation, location, velocity,
                          pp * POSITION_VARIANCE / innovation_variance,
                          pv * POSITION_VARIANCE / innovation_variance,
                          vv - pv**2 / innovation_variance)

    def position(self, stamp, use_motion):
        if not use_motion or stamp == self.stamp:
            return self.observation
        dt = min(FRESH_S, stamp - self.stamp)
        return tuple(x + dt * v for x, v in zip(self.location, self.velocity))


@dataclass
class SystemTrack:
    members: dict[str, tuple[str, str]]
    created: float
    established: bool = False


def _centroid(positions):
    east, north = zip(*positions)
    return fmean(east), fmean(north)


class ContinuousTracker:
    """Ingest native events, then read any integer query using its arrived prefix."""

    def __init__(self, arm):
        if arm not in ARMS:
            raise ValueError("unknown continuous tracker arm")
        self.motion = arm.startswith("kf_")
        self.assign = solve_global if arm.endswith("_global") else solve_greedy
        self.plane = None
        self._shift = {}
        self._clock = -math.inf
        self._last_query = -math.inf
        self._next_id = 0
        self._locals = {}
        self._owner = {}
        self._systems = {}
        self.diagnostics = Counter()

    def install_registration(self, fit):
        translation = fit["translation_east_north_m"]
        if self._clock > FIT_STOP_S:
            raise ValueError("registration must be installed before post-prefix reports")
        if translation is not None:
            left, right = fit["source_order"]
            half = tuple(value / 2 for value in translation)
            self._shift = {left: half, right: tuple(-value for value in half)}

    def _last_stamp(self, system):
        return max(self._locals[key].stamp for key in system.members.values())

    def _fresh(self, system, stamp):
        return tuple(key for key in system.members.values() if stamp - self._locals[key].stamp <= FRESH_S)

    def _position(self, key, stamp, motion):
        point = self._locals[key].position(stamp, motion)
        shift = self._shift.get(key[0]) if stamp > FIT_STOP_S else None
        if shift is None:
            return point
        return point[0] + shift[0], point[1] + shift[1]

    def _cost(self, first, second):
        east = first[0] - second[0]
        north = first[1] - second[1]
        cost = east * east + north * north
        return cost if cost <= RADIUS_M**2 else None

    def _pair(self, costs, n_columns, stage):
        count = sum(cost is not None for row in costs for cost in row)
        if not count:
            return ()
        self.diagnostics[f"{stage}_candidate_edges"] += count
        self.diagnostics[f"{stage}_nonempty_problems"] += 1
        self.diagnostics[f"{stage}_competing_rows"] += sum(
            sum(cost is not None for cost in row) > 1 for row in costs)
        self.diagnostics[f"{stage}_competing_columns"] += sum(
            sum(row[column] is not None for row in costs) > 1 for column in range(n_columns))
        result = self.assign(costs, (NULL_COST,) * len(costs), (NULL_COST,) * n_columns)
        self.diagnostics[f"{stage}_ambiguous_vertices"] += (
            len(result.ambiguous_rows) + len(result.ambiguous_columns))
        return result.pairs

    def _drop_expired(self, stamp):
        expired = [track_id for track_id, system in self._systems.items()
                   if stamp - self._last_stamp(system) > MEMORY_S]
        for track_id in expired:
            for key in self._systems.pop(track_id).members.values():
                del self._owner[key]
        self.diagnostics["expired_systems"] += len(expired)

    def _start(self, key, stamp):
        self._next_id += 1
        track_id = self._next_id
        self._systems[track_id] = SystemTrack({key[0]: key}, stamp)
        self._owner[key] = track_id
        self.diagnostics["new_systems"] += 1

    def _attach(self, donor, receiver, stamp):
        current = self._systems[receiver]
        incoming = self._systems.pop(donor)
        for source, key in incoming.members.items():
            previous = current.members.get(source)
            if previous is not None:
                if stamp - self._locals[previous].stamp <= FRESH_S:
                    raise RuntimeError("attempted to replace a fresh source member")
                del self._owner[previous]
            current.members[source] = key
            self._owner[key] = receiver
        current.established = True

    def _join_existing(self, stamp, updated):
        contenders = defaultdict(list)
        targets = []
        for track_id, system in self._systems.items():
            fresh = self._fresh(system, stamp)
            if system.established or not fresh:
                members = fresh
                if not members:
                    latest = self._last_stamp(system)
                    members = tuple(key for key in system.members.values()
                                    if self._locals[key].stamp == latest)
                position = _centroid([self._position(key, stamp, self.motion) for key in members])
                targets.append((track_id, {key[0] for key in fresh}, position, not fresh))
            elif fresh[0] in updated:
                contenders[fresh[0][0]].append((track_id, self._position(fresh[0], stamp, self.motion)))
        if not targets:
            return

        proposals = []
        for source, rows in contenders.items():
            columns = [(track_id, position, dormant) for track_id, occupied, position, dormant in targets
                       if source not in occupied]
            if not columns:
                continue
            costs = [[self._cost(first, second) for _, second, _ in columns] for _, first in rows]
            for row, column in self._pair(costs, len(columns), "existing"):
                donor = rows[row][0]
                receiver, _, dormant = columns[column]
                proposals.append((donor, receiver))
                self.diagnostics["dormant_links" if dormant else "active_links"] += 1
        for donor, receiver in proposals:
            self._attach(donor, receiver, stamp)

    def _join_singletons(self, stamp, updated):
        groups = defaultdict(list)
        for track_id, system in self._systems.items():
            if system.established:
                continue
            fresh = self._fresh(system, stamp)
            if fresh:
                key = fresh[0]
                groups[key[0]].append((track_id, key in updated, self._position(key, stamp, self.motion)))
        sources = sorted(groups)
        edges = defaultdict(set)
        for index, source in enumerate(sources):
            left = groups[source]
            for other in sources[index + 1:]:
                right = groups[other]
                if not any(updated_now for _, updated_now, _ in (*left, *right)):
                    continue
                costs = [[self._cost(first, second) if first_updated or second_updated else None
                          for _, second_updated, second in right]
                         for _, first_updated, first in left]
                for row, column in self._pair(costs, len(right), "singleton"):
                    first, second = left[row][0], right[column][0]
                    edges[first].add(second)
                    edges[second].add(first)
        visited = set()
        for first in edges:
            if first in visited:
                continue
            component = {first}
            queue = [first]
            for node in queue:
                unseen = edges[node] - component
                component.update(unseen)
                queue.extend(unseen)
            visited.update(component)
            sources_in_component = {source for track_id in component for source in self._systems[track_id].members}
            if len(sources_in_component) != len(component):
                self.diagnostics["multi_source_conflicted_components"] += 1
                continue
            receiver = min(component, key=lambda track_id: (self._systems[track_id].created, track_id))
            for donor in sorted(component - {receiver}):
                self._attach(donor, receiver, stamp)
                self.diagnostics["singleton_links"] += 1

    def ingest(self, stamp, positions):
        if stamp <= max(self._clock, self._last_query):
            raise ValueError("ingest each native event once in strictly increasing time")
        self._clock = stamp
        if not positions:
            return
        if self.plane is None:
            self.plane = Plane.from_first_batch(positions.values())
        self._drop_expired(stamp)
        for key, coordinates in sorted(positions.items()):
            point = self.plane.project(coordinates)
            previous = self._locals.get(key)
            if self.motion and previous is not None and stamp - previous.stamp <= MEMORY_S:
                self._locals[key] = previous.updated(stamp, point)
            else:
                self._locals[key] = LocalTrack(stamp, point, point)
            if key not in self._owner:
                self._start(key, stamp)
        self.diagnostics["events"] += 1
        self.diagnostics["reports"] += len(positions)
        self._join_existing(stamp, positions)
        self._join_singletons(stamp, positions)

    def publish(self, stamp):
        if stamp < max(self._clock, self._last_query):
            raise ValueError("cannot publish a snapshot in the past")
        self._last_query = stamp
        tracks = []
        for track_id, system in sorted(self._systems.items()):
            members = self._fresh(system, stamp)
            if not members:
                continue
            position = _centroid([self._position(key, stamp, False) for key in members])
            tracks.append(dict(track_id=str(track_id), members=tuple(sorted(members)),
                               position_m=position, measurement_time=max(self._locals[key].stamp for key in members)))
        return dict(query_time=stamp, tracks=tracks)
