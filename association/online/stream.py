"""Causal current-state control for individual reported fragments."""
from __future__ import annotations

from collections import Counter
import math

import numpy as np

from association.data.archive import DataError, _measurement_values
from association.data.physical_units import KNOT_TO_MPS
from association.evaluation.native_compatibility import RADIUS, advance


def _advance_position(latitude, longitude, speed, course, age):
    if abs(latitude) == 90.0:
        # The usual longitude formula is singular at the starting pole.
        angular = (age / RADIUS) * speed
        sine = math.sin(angular)
        latitude_out = (latitude / 90.0) * math.degrees(math.atan2(math.cos(angular), abs(sine)))
        longitude_offset = 180.0 - course if latitude > 0.0 else course
        if sine < 0.0:
            longitude_offset += 180.0
        return latitude_out, (longitude + longitude_offset + 180.0) % 360.0 - 180.0
    bearing = math.radians(course)
    velocity = (speed * math.cos(bearing), speed * math.sin(bearing))
    return tuple(map(float, advance((latitude, longitude), velocity, age)))


class StreamState:
    """Receive MTAD observations in arrival order and publish at increasing ticks.

    The visible (source, batch) key identifies one reported fragment. The last
    observation by event time drives its state. Raw samples are retained for
    exact duplicate/conflict detection, and only active fragments are predicted.
    A conflict or invalid state quarantines its key from that arrival onwards.
    """

    def __init__(self, *, stale_after_seconds):
        if not math.isfinite(stale_after_seconds) or stale_after_seconds <= 0:
            raise ValueError("stale_after_seconds must be finite and positive")
        self.stale_after_seconds = stale_after_seconds
        self._last_arrival = -math.inf
        self._last_query = -math.inf
        self._observed = {}
        self._latest = {}
        self._active = set()
        self._published = set()
        self._quarantined = set()
        self._quarantine_reasons = Counter()
        self._received_rows = 0
        self._duplicates = 0

    def ingest(self, rows, *, arrival_time):
        """Ingest one arrival batch. Clock or key errors reject the whole batch."""
        if (not math.isfinite(arrival_time) or arrival_time < self._last_arrival
                or arrival_time <= self._last_query):
            raise ValueError("arrival must be ordered and later than the last query")
        observations = []
        for row in rows:
            key = (row["source"], row["batch"])
            if not all(key):
                raise DataError("measurement lacks source/batch")
            try:
                stamp = float(row["time"])
            except ValueError as exc:
                raise DataError("measurement time must be numeric") from exc
            if not math.isfinite(stamp) or stamp > arrival_time:
                raise DataError("measurement time must be finite and at or before arrival")
            values, issues = _measurement_values(row)
            observations.append((key, stamp, values, issues))

        self._last_arrival = arrival_time
        self._received_rows += len(observations)
        for key, stamp, values, issues in observations:
            if key in self._quarantined:
                continue
            seen = self._observed.setdefault(key, {})
            if values is not None and stamp in seen and values != seen[stamp]:
                issues.add("conflicting_same_time")
            if issues:
                self._quarantined.add(key)
                self._quarantine_reasons.update(issues)
                self._active.discard(key)
                self._latest.pop(key, None)
                del self._observed[key]
                continue
            if stamp in seen:
                self._duplicates += 1
                continue
            seen[stamp] = values
            if key not in self._latest or stamp > self._latest[key][0][0]:
                self._latest[key] = (values, arrival_time)
                self._active.add(key)

    def predict(self, query_time):
        """Emit the state at the query time, including on ticks without arrivals."""
        if (not math.isfinite(query_time) or query_time <= self._last_query
                or query_time < self._last_arrival):
            raise ValueError("query must increase and include all ingested arrivals")
        active = {
            key for key in self._active
            if query_time - self._latest[key][0][0] <= self.stale_after_seconds
        }
        tracks = []
        for key in sorted(active):
            (stamp, latitude, longitude, speed_kn, course), arrival = self._latest[key]
            age = query_time - stamp
            speed = speed_kn * KNOT_TO_MPS
            course %= 360.0
            if age > 0 and speed > 0:
                latitude, longitude = _advance_position(latitude, longitude, speed, course, age)
                if not np.isfinite([latitude, longitude]).all():
                    raise FloatingPointError("nonfinite propagated position")
            tracks.append(dict(
                source=key[0], batch=key[1], latitude=latitude, longitude=longitude,
                speed_mps=speed, course_degrees=course,
                observation_time=stamp, arrival_time=arrival, observation_age_seconds=age,
                latest_observation_updated=arrival > self._last_query, predicted=age > 0,
                appeared=key not in self._published,
            ))
        disappeared = [
            dict(source=source, batch=batch,
                 reason="quarantined" if (source, batch) in self._quarantined else "stale")
            for source, batch in sorted(self._published - active)
        ]
        self._active = active
        self._published = active.copy()
        self._last_query = query_time
        return dict(
            query_time=query_time, tracks=tracks, disappeared=disappeared,
            audit=dict(received_rows=self._received_rows, active_fragments=len(tracks),
                       active_sources=len({track["source"] for track in tracks}),
                       exact_duplicate_rows_removed=self._duplicates,
                       quarantined_fragments=len(self._quarantined),
                       quarantine_reasons=dict(self._quarantine_reasons)),
        )
