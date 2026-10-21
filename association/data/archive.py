"""MTAD identity-pair data without reference-state access.

The registration policy is intentionally explicit; see docs/data_contract.md.
"""
from __future__ import annotations

from bisect import bisect_right
from collections import Counter, defaultdict
from collections.abc import Sequence
from contextlib import contextmanager, ExitStack
import copy
from dataclasses import dataclass
import csv
import hashlib
import io
import json
import math
import operator
import os
from pathlib import Path, PurePosixPath
import re
import zipfile

import numpy as np


SPLIT_NAMES = ("T_core", "T_head", "V_select", "T_test")
# Level1 scene 0 was previewed during intake. All Level2 scenes sharing any
# of its A identities must join the development side; see docs/level1_intake.md.
LEVEL1_PREVIEW_RELATED_LEVEL2_SCENES = (
    "133", "878", "938", "1174", "1214", "1372", "1443", "1853", "1929",
    "2052", "2244", "2697", "3048", "3670", "3839", "4576", "4682", "4731",
)
DEFAULT_DEVELOPMENT_SCENES = tuple(sorted(
    {str(x) for x in [0, 1, 2, *range(100, 5000, 100), 4999]}
    | set(LEVEL1_PREVIEW_RELATED_LEVEL2_SCENES), key=int))
REGISTRATION_POLICY = "scene64-quantized-endpoints-in-support-v1"
UNIT_CONTRACT = {"time": "native elapsed seconds", "lat": "degrees",
                 "lon": "degrees", "vel": "native MTAD velocity (no conversion)",
                 "cou": "degrees clockwise; 360 mapped to 0"}
HEADERS = {"labels": ["t_s", "t_e", "mmsi", "source", "batch"],
           "measurements": ["batch", "source", "time", "lat", "lon", "vel", "cou"]}


class DataError(ValueError):
    """An input cannot be interpreted without an implicit data decision."""


@dataclass
class Track:
    node_id: str
    source: str
    batch: str
    identity_id: str
    ancestor_id: str
    times: np.ndarray
    values: np.ndarray


@dataclass
class Scene:
    scene_id: str
    tracks: list[Track]
    audit: dict


def _name(info):
    name = info.filename
    if not info.flag_bits & 0x800:
        try:
            name = name.encode("cp437").decode("gbk")
        except UnicodeError:
            pass
    parts = PurePosixPath(name).parts
    if (not parts or name.startswith("/") or ".." in parts or
            "\\" in name or ":" in name or "\x00" in name):
        raise DataError("Unsafe ZIP member name")
    return name


def _scene_id(value):
    value = str(value)
    if not value.isdigit():
        raise DataError("Scene IDs must be nonnegative integers")
    return str(int(value))


def zip_dataset_level(archive):
    """Read only member directories; unknown/mixed levels return None.

    External ZIP filenames and scene counts are deliberately not evidence of
    Level1 versus Level2. Every recognized data member must declare one common
    level in its parent directories for the result to be usable.
    """
    levels = set()
    for info in archive.infolist():
        name = _name(info)
        if info.is_dir() or not re.search(
                r'(?:^|/)(?:关联表/关联结果|量测场景/场景|真实场景/场景)-\d+\.csv$', name):
            continue
        parents = '/'.join(PurePosixPath(name).parts[:-2])
        markers = set(re.findall(r'(?<![a-z0-9])level([12])(?![a-z0-9])', parents, re.IGNORECASE))
        if len(markers) != 1:
            return None
        levels.update(markers)
    return 'Level' + next(iter(levels)) if len(levels) == 1 else None


def validate_dataset_level(actual_level, manifest):
    """Bind an explicit level claim to the archive; old unlabeled protocols replay."""
    expected = manifest.get('dataset_level')
    if expected is not None and (expected not in ('Level1', 'Level2') or expected != actual_level):
        raise DataError(f'Archive dataset level {actual_level!r} differs from manifest {expected!r}')


def _digest(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _numeric_sort(values):
    return sorted(values, key=lambda value: int(value))


def _archive_paths(path_or_parts):
    """Keep the caller's explicit part order; never discover or sort siblings."""
    if isinstance(path_or_parts, (str, os.PathLike)):
        paths = (Path(path_or_parts),)
    elif isinstance(path_or_parts, Sequence) and not isinstance(path_or_parts, bytes):
        paths = tuple(Path(path) for path in path_or_parts)
    else:
        raise DataError("Archive must be one ZIP path or an ordered sequence of ZIP parts")
    if not paths:
        raise DataError("Archive parts must be nonempty")
    if len({path.resolve() for path in paths}) != len(paths):
        raise DataError("Archive parts contain a repeated path")
    return paths


class _ConcatenatedReader(io.RawIOBase):
    """Seekable read-only view of a byte-split ZIP, without copying its parts.

    These are consecutive bytes of one ZIP, not independent ZIP archives or
    the multi-disk ZIP format. ZipFile validates the resulting central directory.
    """

    def __init__(self, paths):
        super().__init__()
        self._files, self._offsets, self._position = [], [0], 0
        try:
            for path in paths:
                handle = Path(path).open("rb")
                self._files.append(handle)
                size = os.fstat(handle.fileno()).st_size
                if not size:
                    raise DataError("Archive part is empty: " + str(path))
                self._offsets.append(self._offsets[-1] + size)
        except BaseException:
            self.close()
            raise

    def readable(self):
        return not self.closed

    def seekable(self):
        return not self.closed

    def tell(self):
        self._checkClosed()
        return self._position

    def seek(self, offset, whence=os.SEEK_SET):
        self._checkClosed()
        offset = operator.index(offset)
        if whence == os.SEEK_SET:
            position = offset
        elif whence == os.SEEK_CUR:
            position = self._position + offset
        elif whence == os.SEEK_END:
            position = self._offsets[-1] + offset
        else:
            raise ValueError("Invalid seek mode")
        if position < 0:
            raise ValueError("Negative seek position")
        self._position = position
        return position

    def read(self, size=-1):
        self._checkClosed()
        size = -1 if size is None else operator.index(size)
        available = max(0, self._offsets[-1] - self._position)
        remaining = available if size < 0 else min(size, available)
        chunks = []
        while remaining:
            part = bisect_right(self._offsets, self._position) - 1
            offset = self._position - self._offsets[part]
            count = min(remaining, self._offsets[part + 1] - self._position)
            handle = self._files[part]
            handle.seek(offset)
            chunk = handle.read(count)
            if len(chunk) != count:
                raise OSError("Archive part was truncated while reading")
            chunks.append(chunk)
            self._position += count
            remaining -= count
        return b"".join(chunks)

    def readinto(self, buffer):
        payload = self.read(len(buffer))
        buffer[:len(payload)] = payload
        return len(payload)

    def close(self):
        try:
            for handle in self._files:
                handle.close()
        finally:
            super().close()


@contextmanager
def open_mtad_zip(path_or_parts):
    """Open one ZIP or explicitly ordered continuous parts, closing all handles."""
    paths = _archive_paths(path_or_parts)
    with ExitStack() as resources:
        source = paths[0] if len(paths) == 1 else resources.enter_context(_ConcatenatedReader(paths))
        archive = resources.enter_context(zipfile.ZipFile(source))
        # ZipFile can parse a surviving final part's directory while silently
        # giving earlier members negative offsets. Reject incomplete part sets
        # before they appear to be a complete scene inventory.
        if any(info.header_offset < 0 or info.header_offset + 30 + info.compress_size > archive.start_dir
               for info in archive.infolist()):
            raise DataError("ZIP member offsets exceed the supplied bytes; check archive parts and order")
        yield archive


class MTADArchive:
    """Read label/measurement CSVs in a ZIP, never reference members.

    Duplicate normalized names and duplicate logical scene members are rejected
    instead of relying on ZipFile's last-entry lookup behavior.
    """

    def __init__(self, path):
        self.paths = _archive_paths(path)
        self.path = self.paths[0]
        self._resources = ExitStack()
        self._members = {"labels": {}, "measurements": {}}
        self.label_members = {}
        self.members_read = []
        self._label_cache = {}
        try:
            self._zip = self._resources.enter_context(open_mtad_zip(self.paths))
            seen = set()
            for info in self._zip.infolist():
                name = _name(info)
                if name in seen:
                    raise DataError("Duplicate normalized ZIP member: " + name)
                seen.add(name)
                if info.is_dir():
                    continue
                for kind, pattern in (
                    ("labels", r"(?:^|/)关联表/关联结果-(\d+)\.csv$"),
                    ("measurements", r"(?:^|/)量测场景/场景-(\d+)\.csv$"),
                ):
                    match = re.search(pattern, name)
                    if match:
                        sid = _scene_id(match.group(1))
                        if sid in self._members[kind]:
                            raise DataError("Ambiguous %s member for scene %s" % (kind, sid))
                        self._members[kind][sid] = (info, name)
            labels, measurements = map(set, (self._members["labels"], self._members["measurements"]))
            if not labels or labels != measurements:
                raise DataError("Label and measurement scene sets must be nonempty and identical")
            self.dataset_level = zip_dataset_level(self._zip)
        except BaseException:
            self.close()
            raise

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def close(self):
        self._resources.close()

    @property
    def metadata(self):
        parts = [{"name": path.name, "bytes": path.stat().st_size} for path in self.paths]
        if len(parts) == 1:
            return parts[0]
        return {"parts": parts, "bytes": sum(part["bytes"] for part in parts)}

    def scene_ids(self):
        return _numeric_sort(self._members["labels"])

    def _rows(self, scene_id, kind):
        sid = _scene_id(scene_id)
        try:
            info, name = self._members[kind][sid]
        except KeyError as exc:
            raise DataError("Unknown scene %s" % sid) from exc
        payload = self._zip.read(info)
        self.members_read.append({"kind": kind, "scene_id": sid, "member": name})
        if kind == "labels":
            self.label_members[sid] = {"member": name, "bytes": len(payload)}
        try:
            content = payload.decode("utf-8-sig")
        except UnicodeDecodeError:
            content = payload.decode("gb18030")
        reader = csv.DictReader(io.StringIO(content, newline=""), strict=True)
        if reader.fieldnames != HEADERS[kind]:
            raise DataError("%s has unexpected CSV header %r" % (name, reader.fieldnames))
        rows = []
        for row in reader:
            if None in row or any(value is None for value in row.values()):
                raise DataError("%s line %d has wrong column count" % (name, reader.line_num))
            rows.append({key: value.strip() for key, value in row.items()})
        return rows

    def read_labels(self, scene_id):
        sid = _scene_id(scene_id)
        if sid in self._label_cache:
            return [dict(row) for row in self._label_cache[sid]]
        rows = self._rows(sid, "labels")
        for row in rows:
            if any(not row[key] for key in HEADERS["labels"]):
                raise DataError("Scene %s: empty label field" % sid)
            parts = row["mmsi"].split("-")
            if len(parts) != 3 or not all(parts):
                raise DataError("Scene %s: label identity must be A-B-C" % sid)
            try:
                row["t_s"], row["t_e"] = float(row["t_s"]), float(row["t_e"])
            except ValueError as exc:
                raise DataError("Scene %s: nonnumeric label interval" % sid) from exc
            if not (math.isfinite(row["t_s"]) and math.isfinite(row["t_e"]) and row["t_s"] <= row["t_e"]):
                raise DataError("Scene %s: invalid label interval" % sid)
        self._label_cache[sid] = rows
        return [dict(row) for row in rows]

    def load_scene(self, scene_id):
        sid = _scene_id(scene_id)
        labels = self.read_labels(sid)
        measurements = self._rows(sid, "measurements")
        labels_by_key, points_by_key = defaultdict(list), defaultdict(list)
        for row in labels:
            labels_by_key[(row["source"], row["batch"])].append(row)
        for row in measurements:
            if not row["source"] or not row["batch"]:
                raise DataError("Scene %s: measurement lacks source/batch" % sid)
            points_by_key[(row["source"], row["batch"])].append(row)
        reasons = Counter()
        quarantined, tracks = [], []
        duplicates = 0
        outside_label_intervals = 0
        for source, batch in sorted(set(labels_by_key) | set(points_by_key)):
            key = (source, batch)
            node_id = "%s:%s:%s" % (sid, source, batch)
            entries, raw = labels_by_key[key], points_by_key[key]
            node_reasons = set()
            identities = {row["mmsi"] for row in entries}
            if not entries:
                node_reasons.add("missing_label")
            elif len(identities) != 1:
                node_reasons.add("ambiguous_label_identity")
            if not raw:
                node_reasons.add("missing_measurements")
            numeric = []
            for row in raw:
                try:
                    values = tuple(float(row[k]) for k in ("time", "lat", "lon", "vel", "cou"))
                except ValueError:
                    node_reasons.add("nonnumeric_measurement")
                    continue
                if not all(math.isfinite(x) for x in values):
                    node_reasons.add("nonfinite_measurement")
                    continue
                time, lat, lon, vel, cou = values
                if vel < 0:
                    node_reasons.add("negative_velocity")
                if cou == -1:
                    node_reasons.add("missing_course")
                elif not 0 <= cou <= 360:
                    node_reasons.add("course_out_of_range")
                if not -90 <= lat <= 90 or not -180 <= lon <= 180:
                    node_reasons.add("coordinate_out_of_range")
                # Keep raw representation for exact-duplicate/conflict checks.
                numeric.append(values)
                if entries and not any(e["t_s"] <= time <= e["t_e"] for e in entries):
                    outside_label_intervals += 1
            unique = sorted(set(numeric))
            duplicates += len(numeric) - len(unique)
            if len({value[0] for value in unique}) != len(unique):
                node_reasons.add("conflicting_same_time")
            if len({value[0] for value in unique}) < 2:
                node_reasons.add("fewer_than_two_distinct_times")
            if node_reasons:
                reasons.update(node_reasons)
                quarantined.append({"node_id": node_id, "source": source, "batch": batch,
                                    "raw_mmsi": sorted(identities), "rows": len(raw),
                                    "reasons": sorted(node_reasons)})
                continue
            identity = next(iter(identities))
            parts = identity.split("-")
            array = np.asarray(unique, dtype=np.float64)
            array[:, 4] %= 360
            tracks.append(Track(node_id, source, batch, "-".join(parts[:2]), parts[0],
                                array[:, 0].copy(), array[:, 1:].copy()))
        known = [next(iter({row["mmsi"] for row in entries})).rsplit("-", 1)[0]
                 for entries in labels_by_key.values() if len({row["mmsi"] for row in entries}) == 1]
        all_counts, kept_counts = Counter(known), Counter(track.identity_id for track in tracks)
        positive_before = sum(n * (n - 1) // 2 for n in all_counts.values())
        positive_after = sum(n * (n - 1) // 2 for n in kept_counts.values())
        audit = {"scene_id": sid, "raw_measurement_rows": len(measurements), "label_rows": len(labels),
                 "candidate_nodes": len(set(labels_by_key) | set(points_by_key)), "valid_nodes": len(tracks),
                 "quarantined_nodes": len(quarantined), "quarantine_reasons": dict(sorted(reasons.items())),
                 "quarantined": quarantined, "exact_duplicate_rows_removed": duplicates,
                 "measurement_rows_outside_label_interval_diagnostic_only": outside_label_intervals,
                 "unambiguous_label_positive_edges_before_quarantine": positive_before,
                 "positive_edges_after_quarantine": positive_after,
                 "positive_edges_lost_to_quarantine": positive_before - positive_after,
                 "unit_contract": UNIT_CONTRACT.copy(), "reference_members_read": 0,
                 "valid_raw_mmsi": {track.node_id: labels_by_key[(track.source, track.batch)][0]["mmsi"]
                                    for track in tracks}}
        return Scene(sid, tracks, audit)


def build_manifest(archive, seed, caps, development_scenes):
    """Freeze all scene components before deterministic within-split caps.

    Component membership is built from A identifiers in *all* labels, including
    scenes that will later be capped out. No measurement or reference is read.
    """
    scene_ids = archive.scene_ids()
    if set(caps) != set(SPLIT_NAMES):
        raise DataError("caps must specify exactly " + ", ".join(SPLIT_NAMES))
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in caps.values()):
        raise DataError("caps must be nonnegative integers")
    development = {_scene_id(sid) for sid in development_scenes}
    if not development.issubset(scene_ids):
        raise DataError("Development scene IDs absent from archive: " + ",".join(_numeric_sort(development - set(scene_ids))))
    parent = {sid: sid for sid in scene_ids}

    def find(sid):
        while parent[sid] != sid:
            parent[sid] = parent[parent[sid]]
            sid = parent[sid]
        return sid

    def union(left, right):
        a, b = find(left), find(right)
        if a != b:
            parent[max(a, b, key=int)] = min(a, b, key=int)

    ancestor_first = {}
    for sid in scene_ids:
        for row in archive.read_labels(sid):
            ancestor = row["mmsi"].split("-", 1)[0]
            if ancestor in ancestor_first:
                union(sid, ancestor_first[ancestor])
            else:
                ancestor_first[ancestor] = sid
    groups = defaultdict(list)
    for sid in scene_ids:
        groups[find(sid)].append(sid)
    component_members = {}
    components = {}
    component_keys = {}
    forced = set()
    for members in groups.values():
        members = _numeric_sort(members)
        serialized = ",".join(members)
        cid = "component-" + _digest(serialized)[:24]
        component_members[cid] = members
        component_keys[cid] = _digest(str(seed) + ":" + serialized)
        components.update({sid: cid for sid in members})
        if development.intersection(members):
            forced.add(cid)
    component_splits = {cid: "T_core" for cid in forced}
    for index, cid in enumerate(sorted(set(component_members) - forced, key=lambda c: (component_keys[c], c))):
        component_splits[cid] = SPLIT_NAMES[index % len(SPLIT_NAMES)]
    assigned = {split: [] for split in SPLIT_NAMES}
    for sid in scene_ids:
        assigned[component_splits[components[sid]]].append(sid)
    splits = {split: sorted(assigned[split], key=lambda sid: (_digest(str(seed) + ":" + sid), int(sid)))[:caps[split]]
              for split in SPLIT_NAMES}
    counts = {split: {"assigned_scenes": len(assigned[split]), "selected_scenes": len(splits[split]),
                      "assigned_components": sum(value == split for value in component_splits.values()),
                      "selected_components": len({components[sid] for sid in splits[split]}),
                      "largest_selected_component_scenes": max(Counter(components[sid] for sid in splits[split]).values(), default=0)}
              for split in SPLIT_NAMES}
    manifest = {"manifest_version": 1, "protocol": "MTAD-Level2 A-B identity-pair pilot v1",
                "seed": int(seed), "caps": dict(caps), "splits": splits, "components": components,
                "component_members": component_members, "component_splits": component_splits,
                "split_counts": counts, "development_scenes": _numeric_sort(development),
                "development_forced_components": sorted(forced), "all_scenes": scene_ids,
                "archive": archive.metadata,
                "label_members": dict(archive.label_members),
                "split_policy": "whole shared-A components; dev components T_core; remaining hash-sorted cyclic assignment; within-side scene hash caps",
                "selection_note": "Unselected component members retain their split; no scene moves. Development is never held out, but hash caps may leave a development scene unselected.",
                "content_coverage": "all labels; selected measurements audited at preprocessing; no reference reads",
                "unit_contract": UNIT_CONTRACT.copy(), "registration_policy": REGISTRATION_POLICY}
    validate_manifest(manifest)
    return manifest


def validate_manifest(manifest):
    """Check actual split/component structure."""
    if set(manifest["splits"]) != set(SPLIT_NAMES):
        raise DataError("Invalid manifest split names")
    all_scenes = manifest["all_scenes"]
    if len(all_scenes) != len(set(all_scenes)) or set(manifest["components"]) != set(all_scenes):
        raise DataError("Inconsistent manifest scene inventory")
    members = [sid for group in manifest["component_members"].values() for sid in group]
    if len(members) != len(set(members)) or set(members) != set(all_scenes):
        raise DataError("Component members must partition the scene inventory")
    split_unit = manifest.get("split_unit", "shared_a_component")
    if split_unit not in ("shared_a_component", "scene"):
        raise DataError("Unknown manifest split unit")
    if split_unit == "shared_a_component":
        if set(manifest["component_splits"]) != set(manifest["component_members"]):
            raise DataError("Missing component split assignment")
        if any(split not in SPLIT_NAMES for split in manifest["component_splits"].values()):
            raise DataError("Invalid component split assignment")
    if set(manifest["label_members"]) != set(all_scenes):
        raise DataError("Incomplete label provenance")
    seen = set()
    for split in SPLIT_NAMES:
        if len(manifest["splits"][split]) > manifest["caps"][split]:
            raise DataError("Selected scenes exceed cap")
        for sid in manifest["splits"][split]:
            if sid in seen:
                raise DataError("Scene appears in more than one split")
            if sid not in manifest["components"]:
                raise DataError("Selected scene absent from component inventory")
            seen.add(sid)
            cid = manifest["components"][sid]
            if split_unit == "shared_a_component" and manifest["component_splits"][cid] != split:
                raise DataError("Component crosses split boundary")
    if split_unit == "shared_a_component":
        for sid in manifest["development_scenes"]:
            if manifest["component_splits"][manifest["components"][sid]] != "T_core":
                raise DataError("Development component escaped T_core")
    else:
        _validate_scene_manifest(manifest)
    for cid, members in manifest["component_members"].items():
        if any(manifest["components"].get(sid) != cid for sid in members):
            raise DataError("Inconsistent component membership")
    return True


def _historical_scene_roles(manifest):
    """Direct exposure restrictions; shared-A neighbours are not newly exposed."""
    history = manifest["historical_usage"]
    selected = history["selected_splits"]
    train = set(selected["T_core"]) | set(selected["T_head"]) | set(manifest["development_scenes"])
    validation = set(selected["V_select"])
    test = {sid for sid, cid in manifest["components"].items()
            if history["component_splits"][cid] == "T_test"}
    return {"T_core": train, "T_head": set(), "V_select": validation, "T_test": test}


def _validate_scene_manifest(manifest):
    """Validate the explicitly selected scene protocol, keeping real components."""
    if (set(manifest["caps"]) != set(SPLIT_NAMES)
            or any(isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in manifest["caps"].values())
            or manifest["caps"]["T_head"] != 0):
        raise DataError("Scene protocol requires nonnegative caps and an empty T_head")
    if any(len(manifest["splits"][split]) != manifest["caps"][split] for split in SPLIT_NAMES):
        raise DataError("Scene protocol must fill its declared counts exactly")
    expected_roles = dict.fromkeys(manifest["all_scenes"], "unused")
    expected_roles.update({sid: split for split, scenes in manifest["splits"].items() for sid in scenes})
    if manifest["scene_splits"] != expected_roles:
        raise DataError("Scene role map differs from selected and unused scenes")
    if manifest.get("split_origin") == "fresh_archive":
        from association.data.scene_manifest import validate_fresh_scene_usage
        validate_fresh_scene_usage(manifest)
        return
    history = manifest["historical_usage"]
    if (set(history["component_splits"]) != set(manifest["component_members"])
            or any(role not in SPLIT_NAMES for role in history["component_splits"].values())
            or set(history["selected_splits"]) != set(SPLIT_NAMES)):
        raise DataError("Incomplete historical split provenance")
    historical_seen = set()
    for split, scenes in history["selected_splits"].items():
        for sid in scenes:
            if (sid not in manifest["components"] or sid in historical_seen
                    or history["component_splits"][manifest["components"][sid]] != split):
                raise DataError("Inconsistent historical scene role")
            historical_seen.add(sid)
    for sid in manifest["development_scenes"]:
        if (sid not in manifest["components"]
                or history["component_splits"][manifest["components"][sid]] != "T_core"):
            raise DataError("Inconsistent historical development scene")
    protected = _historical_scene_roles(manifest)
    for split, scenes in protected.items():
        if not scenes.issubset(manifest["splits"][split]):
            raise DataError("Historical scenes must remain in " + split)
    if manifest["historical_protection_counts"] != {split: len(scenes) for split, scenes in protected.items()}:
        raise DataError("Historical protection counts differ from actual scenes")


def build_scene_manifest(source_manifest, seed=20, caps=None, *, source_description="prior shared-A manifest"):
    """Select scenes from existing metadata only; never open an MTAD archive.

    Preserve the historical selected training/head/preview, validation and full
    test-role populations. Fill their deficits from the remaining scene IDs in
    deterministic seed order. Real shared-A components remain available for
    dependence reporting, but no longer constrain the new scene split.
    """
    validate_manifest(source_manifest)
    if source_manifest.get("split_unit", "shared_a_component") != "shared_a_component":
        raise DataError("Scene protocol must start from the historical shared-A manifest")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise DataError("Scene selection seed must be an integer")
    caps = dict(caps if caps is not None else {"T_core": 2100, "T_head": 0, "V_select": 300, "T_test": 600})
    if (set(caps) != set(SPLIT_NAMES) or caps["T_head"] != 0
            or any(isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in caps.values())):
        raise DataError("Scene protocol requires nonnegative caps and an empty T_head")
    manifest = {key: copy.deepcopy(source_manifest[key]) for key in (
        "all_scenes", "components", "component_members", "development_scenes", "label_members")}
    for key in ("archive", "unit_contract"):
        if key in source_manifest:
            manifest[key] = copy.deepcopy(source_manifest[key])
    manifest.update(manifest_version=2, protocol="MTAD-Level2 scene-level historically constrained v1",
        split_unit="scene", seed=seed, caps=caps,
        historical_usage=dict(source=source_description, source_seed=source_manifest.get("seed"),
            source_protocol=source_manifest.get("protocol"),
            source_registration_policy=source_manifest.get("registration_policy"),
            selected_splits=copy.deepcopy(source_manifest["splits"]),
            component_splits=copy.deepcopy(source_manifest["component_splits"])))
    protected = _historical_scene_roles(manifest)
    used = set().union(*protected.values())
    if len(used) != sum(map(len, protected.values())):
        raise DataError("Historical exposure restrictions conflict")
    deficits = {split: caps[split] - len(protected[split]) for split in SPLIT_NAMES}
    free = sorted(set(manifest["all_scenes"]) - used,
                  key=lambda sid: (_digest(str(seed) + ":" + sid), int(sid)))
    if min(deficits.values()) < 0 or sum(deficits.values()) > len(free):
        raise DataError("Requested counts cannot retain historical populations")
    splits, offset = {}, 0
    for split in SPLIT_NAMES:
        stop = offset + deficits[split]
        selected = protected[split] | set(free[offset:stop])
        splits[split] = [sid for sid in manifest["all_scenes"] if sid in selected]
        offset = stop
    manifest.update(splits=splits, scene_splits=dict.fromkeys(manifest["all_scenes"], "unused"),
        historical_protection_counts={split: len(scenes) for split, scenes in protected.items()},
        split_counts={split: {"selected_scenes": len(scenes),
            "selected_components": len({manifest["components"][sid] for sid in scenes}),
            "largest_selected_component_scenes": max(Counter(manifest["components"][sid] for sid in scenes).values(), default=0)}
            for split, scenes in splits.items()},
        split_policy="scene-level; retain historical train/head/preview, validation and full test-role scenes; "
            "sort remaining IDs by SHA256(str(seed)+':'+scene_id), fill T_core then V_select then T_test; remainder unused",
        selection_note="Historically constrained project selection, not the authors' random scene list. "
            f"Seed {seed} is our split choice; the paper specifies 20 for weight initialization, not a split seed. "
            "Shared-A identities may cross splits; components are retained for dependence reporting, not isolation. "
            "The digest only orders scene IDs and is not an admission gate.",
        content_coverage="existing manifest metadata only; no new label, measurement or reference reads; "
            "new train/validation preparation and frozen test evaluation have not run",
        preparation_status="manifest only; native loader compatible; training component-isolation checks unchanged")
    manifest["scene_splits"].update({sid: split for split, scenes in splits.items() for sid in scenes})
    validate_manifest(manifest)
    return manifest


def resample_scene(scene, length=64):
    """Assign in-support samples to quantized common-clock token slots.

    ``observed`` means a real/interpolated in-support value exists, NOT that its
    actual time equals the nominal grid timestamp. ``observation_times`` carries
    that actual time. No coordinate extrapolation or short-track exclusion occurs.
    """
    if isinstance(length, bool) or not isinstance(length, int) or length < 2:
        raise DataError("Registration length must be an integer >= 2")
    n = len(scene.tracks)
    values = np.zeros((n, length, 4), dtype=np.float32)
    observed = np.zeros((n, length), dtype=bool)
    observation_times = np.full((n, length), np.nan, dtype=np.float64)
    for track in scene.tracks:
        times = np.asarray(track.times, dtype=np.float64)
        raw = np.asarray(track.values, dtype=np.float64)
        if (times.ndim != 1 or len(times) < 2 or raw.shape != (len(times), 4) or
                not np.isfinite(times).all() or not np.isfinite(raw).all() or not np.all(np.diff(times) > 0)):
            raise DataError("Invalid Track passed to resample_scene: " + track.node_id)
    if not n:
        grid = np.zeros(length, dtype=np.float64)
    else:
        start = min(track.times[0] for track in scene.tracks)
        stop = max(track.times[-1] for track in scene.tracks)
        if not math.isfinite(start) or not math.isfinite(stop) or stop <= start:
            raise DataError("Scene has invalid temporal range")
        grid = np.linspace(start, stop, length)
    max_displacement = 0.0
    short_tracks = 0
    quantized = []
    for index, track in enumerate(scene.tracks):
        times = np.asarray(track.times, dtype=np.float64)
        raw = np.asarray(track.values, dtype=np.float64)
        nominal = (length - 1) * (times[[0, -1]] - grid[0]) / (grid[-1] - grid[0])
        first, last = int(np.floor(nominal[0])), int(np.ceil(nominal[1]))
        first, last = max(0, first), min(length - 1, last)
        if last <= first:
            if first < length - 1:
                last = first + 1
            else:
                first = last - 1
        inside_count = int(np.count_nonzero((grid >= times[0]) & (grid <= times[-1])))
        short_tracks += inside_count < 2
        actual_times = np.linspace(times[0], times[-1], last - first + 1)
        out = np.column_stack([np.interp(actual_times, times, raw[:, column]) for column in range(3)])
        # Unwrap along the shortest angular arc, interpolate, then restore [0,360).
        angle = np.unwrap(np.deg2rad(raw[:, 3] % 360))
        course = np.rad2deg(np.interp(actual_times, times, angle)) % 360
        values[index, first:last + 1] = np.column_stack([out, course])
        observed[index, first:last + 1] = True
        observation_times[index, first:last + 1] = actual_times
        displacement = float(np.max(np.abs(actual_times - grid[first:last + 1])))
        max_displacement = max(max_displacement, displacement)
        quantized.append({"node_id": track.node_id, "first_slot": first, "last_slot": last,
                          "support_start": float(times[0]), "support_stop": float(times[-1]),
                          "nominal_in_support_slots": inside_count,
                          "max_absolute_time_displacement": displacement})
    audit = dict(scene.audit)
    audit["registration"] = {"policy": REGISTRATION_POLICY, "length": length,
                             "short_support_tracks": short_tracks,
                             "max_absolute_time_displacement": max_displacement,
                             "observed_definition": "actual/interpolated in-support value, assigned to a quantized nominal slot",
                             "per_track": quantized, "coordinate_extrapolation": False,
                             "excluded_for_short_support": 0}
    return {"values": values, "observed": observed, "grid": grid,
            "observation_times": observation_times, "node_ids": [track.node_id for track in scene.tracks],
            "identities": [track.identity_id for track in scene.tracks],
            "sources": [track.source for track in scene.tracks], "audit": audit}
