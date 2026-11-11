"""Position and time inputs for the strict Level1 replay."""
from __future__ import annotations

from itertools import groupby
import math

from .stream import EARTH_RADIUS_M


def event_batches(rows):
    """Release only time, position, and equality of sensor fragment keys."""
    events = []
    for row in rows:
        stamp, lat, lon = (float(row[field]) for field in ("time", "lat", "lon"))
        key = (row["source"], row["batch"])
        if (not all(key) or not math.isfinite(stamp) or not -90 <= lat <= 90
                or not -180 <= lon <= 180):
            raise ValueError("invalid required position observation")
        events.append((stamp, key, (lat, lon)))
    events.sort(key=lambda item: item[0])
    for stamp, epoch in groupby(events, key=lambda item: item[0]):
        positions = {}
        for _, key, position in epoch:
            if key in positions and positions[key] != position:
                raise ValueError("conflicting same-time positions in one fragment")
            positions[key] = position
        yield stamp, positions


def position_step(first, second):
    """Distance in metres and initial bearing in degrees on a sphere."""
    lat1, lon1 = map(math.radians, first)
    lat2, lon2 = map(math.radians, second)
    lon_delta = lon2 - lon1
    hav = math.sin((lat2 - lat1) / 2) ** 2 + (
        math.cos(lat1) * math.cos(lat2) * math.sin(lon_delta / 2) ** 2
    )
    distance = 2 * EARTH_RADIUS_M * math.asin(math.sqrt(min(1.0, hav)))
    east = math.sin(lon_delta) * math.cos(lat2)
    north = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(lon_delta)
    return distance, math.degrees(math.atan2(east, north)) % 360


def angle_error(a, b):
    return abs((a - b + 180) % 360 - 180)
