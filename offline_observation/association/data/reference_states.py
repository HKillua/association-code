"""Read explicitly selected raw MTAD reference states, never model inputs.

This is a supervision-data reader, not a fusion model or an alignment policy.
No timestamps, states, duplicates, or row ordering are repaired here.
"""
from __future__ import annotations

from collections import defaultdict
import csv
import io
import math
import re

from association.data.archive import (DataError, _name, _scene_id, open_mtad_zip,
                                     validate_dataset_level, validate_manifest, zip_dataset_level)


REFERENCE_HEADER = ["MMSI", "time", "lat", "lon", "vel", "cou"]
STATE_FIELDS = ("lat", "lon", "vel", "cou")
ALLOWED_SPLITS = ("T_core", "T_head", "V_select")


def _read_rows(payload, member):
    # Same encoding order as MTADArchive._rows; reference access stays isolated
    # from that class, whose observation-only contract must remain unchanged.
    try:
        content = payload.decode("utf-8-sig")
    except UnicodeDecodeError:
        content = payload.decode("gb18030")
    reader = csv.DictReader(io.StringIO(content, newline=""), strict=True)
    if reader.fieldnames != REFERENCE_HEADER:
        raise DataError(f"{member} has unexpected reference CSV header {reader.fieldnames!r}")
    rows, issues, groups = [], [], defaultdict(list)
    for raw in reader:
        line = reader.line_num
        if None in raw or any(value is None for value in raw.values()):
            raise DataError(f"{member} line {line} has wrong column count")
        identity = raw["MMSI"].strip()
        parts = identity.split("-")
        valid_identity = len(parts) == 2 and all(parts)
        if not valid_identity:
            issues.append({"code": "invalid_identity", "lines": [line], "field": "MMSI",
                           "value": raw["MMSI"], "expected": "A-B"})
        numbers = {}
        for field in ("time", *STATE_FIELDS):
            try:
                numbers[field] = float(raw[field])
            except ValueError:
                numbers[field] = None
                issues.append({"code": "nonnumeric", "lines": [line], "field": field,
                               "value": raw[field]})
                continue
            if not math.isfinite(numbers[field]):
                issues.append({"code": "nonfinite", "lines": [line], "field": field,
                               "value": raw[field]})
        row = {"line": line, "identity_id": identity, "time": numbers["time"],
               "state": [numbers[field] for field in STATE_FIELDS], "raw": dict(raw)}
        rows.append(row)
        if valid_identity and row["time"] is not None and math.isfinite(row["time"]):
            groups[(identity, row["time"])].append(row)
    for (identity, timestamp), duplicates in groups.items():
        if len(duplicates) < 2:
            continue
        comparable = all(value is not None and math.isfinite(value)
                         for row in duplicates for value in row["state"])
        if not comparable:
            code = "duplicate_unresolved"
        else:
            code = ("duplicate_identical" if all(row["state"] == duplicates[0]["state"]
                                                for row in duplicates[1:]) else "duplicate_conflict")
        issues.append({"code": code, "identity_id": identity, "time": timestamp,
                       "lines": [row["line"] for row in duplicates]})
    return rows, issues


def iter_reference_states(archive_path, manifest, *, split, scene_ids):
    """Stream raw reference scenes, keeping only one scene's rows in memory.

    Each output scene contains ``rows`` in original CSV order, numeric ``time``
    and ``state=[lat,lon,vel,cou]``, original strings in ``raw``, and ``issues``.
    Nonfinite numbers remain nonfinite; unparseable numbers are None plus their
    original strings. No row is dropped. The caller must explicitly resolve
    issues before deriving supervision; this reader creates no model inputs.
    """
    validate_manifest(manifest)
    if split not in ALLOWED_SPLITS:
        raise DataError("Reference reader permits only explicit training/validation splits; T_test is sealed")
    if isinstance(scene_ids, (str, bytes)):
        raise DataError("scene_ids must be an explicit sequence, not a string")
    requested = [_scene_id(scene_id) for scene_id in scene_ids]
    if not requested or len(requested) != len(set(requested)):
        raise DataError("scene_ids must be nonempty and contain no duplicate scenes")
    if not set(requested).issubset(set(manifest["splits"][split])):
        raise DataError("Requested reference scenes are outside the declared split")
    with open_mtad_zip(archive_path) as archive:
        validate_dataset_level(zip_dataset_level(archive), manifest)
        members, seen = {}, set()
        for info in archive.infolist():
            name = _name(info)
            if name in seen:
                raise DataError("Duplicate normalized ZIP member: " + name)
            seen.add(name)
            match = re.search(r"(?:^|/)真实场景/场景-(\d+)\.csv$", name)
            if info.is_dir() or not match:
                continue
            sid = _scene_id(match.group(1))
            if sid not in requested:
                continue
            if sid in members:
                raise DataError("Ambiguous reference member for scene " + sid)
            members[sid] = (info, name)
        if set(members) != set(requested):
            raise DataError("Missing requested reference scenes: " + ", ".join(sorted(set(requested) - set(members))))
        for sid in requested:
            info, name = members[sid]
            rows, issues = _read_rows(archive.read(info), name)
            yield {"scene_id": sid, "split": split, "member": name,
                   "state_fields": list(STATE_FIELDS), "rows": rows, "issues": issues}


def read_reference_states(archive_path, manifest, *, split, scene_ids):
    """Compatibility list reader; use iter_reference_states for large splits."""
    return list(iter_reference_states(archive_path, manifest, split=split, scene_ids=scene_ids))
