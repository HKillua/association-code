"""Causal current-state control for individual reported fragments."""
from __future__ import annotations

from collections import Counter
import math

KNOT_TO_MPS = 1852.0 / 3600.0
EARTH_RADIUS_M = 6_371_000.0


def _advance_position(latitude, longitude, speed, course, age):
    """Propagate the last report on a sphere using its tangent bearing."""
    angular = (age / EARTH_RADIUS_M) * speed
    if not math.isfinite(angular):
        raise FloatingPointError("nonfinite travel angle")
    if angular == 0.0:
        return latitude, longitude
    lat, bearing = math.radians(latitude), math.radians(course)
    sin_lat = math.sin(lat)
    cos_lat = 0.0 if abs(latitude) == 90.0 else math.cos(lat)
    radial = math.cos(angular)
    north = math.sin(angular) * math.cos(bearing)
    east = math.sin(angular) * math.sin(bearing)
    # Cartesian coordinates relative to the starting longitude.
    x = radial * cos_lat - north * sin_lat
    z = radial * sin_lat + north * cos_lat
    latitude_out = math.degrees(math.atan2(z, math.hypot(x, east)))
    longitude_out = longitude + math.degrees(math.atan2(east, x))
    return latitude_out, (longitude_out + 180.0) % 360.0 - 180.0


def _measurement_values(row, stamp):
    try:
        lat, lon, speed, course = (float(row[field]) for field in ("lat", "lon", "vel", "cou"))
    except (TypeError, ValueError):
        return None, {"nonnumeric_measurement"}
    if not all(math.isfinite(value) for value in (lat, lon, speed, course)):
        return None, {"nonfinite_measurement"}
    issues = set()
    if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
        issues.add("coordinate_out_of_range")
    if speed < 0.0:
        issues.add("negative_velocity")
    if course == -1.0:
        issues.add("missing_course")
    elif not 0.0 <= course <= 360.0:
        issues.add("course_out_of_range")
    return (stamp, lat, lon, speed, course), issues


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
                raise ValueError("measurement lacks source/batch")
            try:
                stamp = float(row["time"])
            except (TypeError, ValueError) as exc:
                raise ValueError("measurement time must be numeric") from exc
            if not math.isfinite(stamp) or stamp > arrival_time:
                raise ValueError("measurement time must be finite and at or before arrival")
            values, issues = _measurement_values(row, stamp)
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
