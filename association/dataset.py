"""Prepare portable scene records and split provenance for the CDF1 pilot."""
from __future__ import annotations

from collections import Counter
import csv
import hashlib
import io
import json
from pathlib import Path
import zipfile

import numpy as np

from association.data import (DEFAULT_DEVELOPMENT_SCENES, DataError, MTADArchive,
                              REGISTRATION_POLICY, SPLIT_NAMES, UNIT_CONTRACT,
                              build_manifest, resample_scene, validate_manifest)


def _write_json(path, payload):
    """Write strict JSON without importing the model/runtime dependencies."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                         encoding="utf-8")
    temporary.replace(path)


def prepare_scene(archive, scene_id, length, component_id):
    """Load one scene, retaining all valid nodes and all unordered pairs."""
    scene = archive.load_scene(scene_id)
    record = resample_scene(scene, length)
    n = len(scene.tracks)
    pairs = np.column_stack(np.triu_indices(n, 1)).astype(np.int64)
    identities = np.asarray(record["identities"], dtype=str)
    labels = (identities[pairs[:, 0]] == identities[pairs[:, 1]]).astype(np.float32)
    record.update({"scene_id": str(scene.scene_id), "component_id": str(component_id),
                   "pairs": pairs, "labels": labels,
                   "intervals": np.asarray([[track.times[0], track.times[-1]] for track in scene.tracks],
                                           dtype=np.float64).reshape(n, 2)})
    return record


def validate_archive(archive, manifest):
    """Verify a persisted manifest and its exact archive content binding."""
    validate_manifest(manifest)
    if archive.path.stat().st_size != manifest["archive"]["bytes"]:
        raise DataError("Archive size differs from frozen manifest")
    if archive.archive_sha256() != manifest["archive"]["sha256"]:
        raise DataError("Archive SHA-256 differs from frozen manifest")
    if archive.scene_ids() != manifest["all_scenes"]:
        raise DataError("Archive scenes differ from frozen manifest")


def load_split_from_archive(archive, manifest, split, length):
    """Use an already open archive; its SHA-256 is computed only once."""
    validate_archive(archive, manifest)
    if split not in SPLIT_NAMES:
        raise DataError("Unknown split " + str(split))
    return [prepare_scene(archive, sid, length, manifest["components"][sid])
            for sid in manifest["splits"][split]]


def load_split(archive_path, manifest, split, length):
    """Read only measurement/label members selected for the requested split."""
    with MTADArchive(archive_path) as archive:
        return load_split_from_archive(archive, manifest, split, length)


def _config(config):
    if isinstance(config, (str, Path)):
        with Path(config).open(encoding="utf-8") as handle:
            config = json.load(handle)
    if not isinstance(config, dict):
        raise DataError("Configuration must be a dictionary or JSON path")
    if not isinstance(config.get("model"), dict):
        raise DataError("Configuration requires model.length")
    length = config["model"].get("length")
    if isinstance(length, bool) or not isinstance(length, int) or length < 2:
        raise DataError("Configuration model.length must be >= 2")
    seed = config.get("split_seed")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise DataError("Configuration split_seed must be an integer")
    caps = config.get("caps", {})
    if set(caps) != set(SPLIT_NAMES) or any(isinstance(x, bool) or not isinstance(x, int) or x < 1 for x in caps.values()):
        raise DataError("Preparation requires a positive cap for each of the four splits")
    return config


def _split_audit(records, component_ids):
    quarantine = Counter()
    result = {"status": "inspected", "selected_scenes": len(records),
              "selected_components": len(set(component_ids)), "raw_nodes": 0,
              "valid_nodes": 0, "quarantined_nodes": 0, "candidate_edges": 0,
              "scenes_with_at_least_two_valid_nodes": 0, "exact_duplicate_rows_removed": 0,
              "registration_short_support_tracks": 0,
              "registration_max_absolute_time_displacement": 0.0,
              "quarantine_reasons": {}, "scenes": []}
    for record in records:
        audit = record["audit"]
        registration = audit["registration"]
        result["raw_nodes"] += audit["candidate_nodes"]
        result["valid_nodes"] += audit["valid_nodes"]
        result["quarantined_nodes"] += audit["quarantined_nodes"]
        result["candidate_edges"] += len(record["pairs"])
        result["scenes_with_at_least_two_valid_nodes"] += audit["valid_nodes"] >= 2
        result["exact_duplicate_rows_removed"] += audit["exact_duplicate_rows_removed"]
        result["registration_short_support_tracks"] += registration["short_support_tracks"]
        result["registration_max_absolute_time_displacement"] = max(
            result["registration_max_absolute_time_displacement"], registration["max_absolute_time_displacement"])
        quarantine.update(audit["quarantine_reasons"])
        # Coverage only: no per-identity, positive-label balance, model scores,
        # or reference content is included in the preparation report.
        result["scenes"].append({"scene_id": record["scene_id"], "component_id": record["component_id"],
                                 "raw_nodes": audit["candidate_nodes"], "valid_nodes": audit["valid_nodes"],
                                 "quarantined_nodes": audit["quarantined_nodes"],
                                 "quarantine_reasons": audit["quarantine_reasons"],
                                 "candidate_edges": len(record["pairs"]),
                                 "registration_short_support_tracks": registration["short_support_tracks"],
                                 "registration_max_absolute_time_displacement": registration["max_absolute_time_displacement"]})
    result["quarantine_reasons"] = dict(sorted(quarantine.items()))
    return result


def prepare(archive_path, config, output, synthetic=False):
    """Write a fresh manifest and train/validation coverage audit.

    Test measurements are deliberately not opened. They remain untouched until
    the frozen evaluation command calls ``load_split(..., 'T_test', ...)``.
    This function writes no model, normalization moments, or score statistics.
    """
    config = _config(config)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    config_sha = hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":"),
                                          ensure_ascii=False, allow_nan=False).encode("utf-8")).hexdigest()
    audit = {"schema": 1, "status": "RUNNING", "synthetic": bool(synthetic),
             "scientific_result": False, "config_sha256": config_sha,
             "registration_policy": REGISTRATION_POLICY, "length": config["model"]["length"],
             "units": UNIT_CONTRACT.copy(), "reference_members_read": 0,
             "test_measurements_read": 0, "splits": {},
             "scope": "input coverage only; all labels for isolated split; test measurements deferred until frozen evaluation"}
    _write_json(output / "audit.json", audit)
    try:
        with MTADArchive(archive_path) as archive:
            manifest = build_manifest(archive, config["split_seed"], config["caps"],
                                      [] if synthetic else DEFAULT_DEVELOPMENT_SCENES)
            empty = [split for split in SPLIT_NAMES if not manifest["splits"][split]]
            if empty:
                raise DataError("Preparation produced empty splits: " + ", ".join(empty))
            _write_json(output / "manifest.json", manifest)
            audit["manifest_sha256"] = manifest["manifest_sha256"]
            audit["archive"] = manifest["archive"]
            for split in ("T_core", "T_head", "V_select"):
                # Only one scene is resident at once during the audit.
                records = (prepare_scene(archive, sid, config["model"]["length"], manifest["components"][sid])
                           for sid in manifest["splits"][split])
                audit["splits"][split] = _split_audit_stream(records, manifest, split)
                _write_json(output / "audit.json", audit)
            audit["splits"]["T_test"] = {
                "status": "deferred_until_frozen_evaluation",
                "selected_scenes": len(manifest["splits"]["T_test"]),
                "selected_components": manifest["split_counts"]["T_test"]["selected_components"],
                "valid_nodes": None, "candidate_edges": None, "quarantined_nodes": None,
                "reason": "No held-out measurements opened during preparation"}
            audit["measurement_members_read"] = sum(row["kind"] == "measurements" for row in archive.members_read)
            audit["label_members_read"] = sum(row["kind"] == "labels" for row in archive.members_read)
        audit["status"] = "COMPLETED"
        _write_json(output / "audit.json", audit)
        return manifest
    except BaseException as exc:
        audit.update(status="FAILED", error_type=type(exc).__name__, error=str(exc))
        _write_json(output / "audit.json", audit)
        raise


def _split_audit_stream(records, manifest, split):
    """Aggregate coverage while avoiding a full split of input arrays in RAM."""
    combined = _split_audit([], [])
    combined["selected_components"] = manifest["split_counts"][split]["selected_components"]
    counts = Counter()
    sum_fields = ("selected_scenes", "raw_nodes", "valid_nodes", "quarantined_nodes", "candidate_edges",
                  "scenes_with_at_least_two_valid_nodes", "exact_duplicate_rows_removed",
                  "registration_short_support_tracks")
    for record in records:
        item = _split_audit([record], [record["component_id"]])
        for key in sum_fields:
            combined[key] += item[key]
        combined["registration_max_absolute_time_displacement"] = max(
            combined["registration_max_absolute_time_displacement"], item["registration_max_absolute_time_displacement"])
        combined["scenes"].extend(item["scenes"])
        counts.update(item["quarantine_reasons"])
    combined["quarantine_reasons"] = dict(sorted(counts.items()))
    return combined


def make_synthetic_archive(path, n_scenes=32):
    """Create a deterministic tiny native-format fixture; no scientific meaning.

    A identifiers are independent per scene. Within a scene each of two or three
    identities has two or three fragments, with multiple sensors, irregular
    support, and mildly perturbed measurements. No reference table is created.
    """
    if isinstance(n_scenes, bool) or not isinstance(n_scenes, int) or n_scenes < 4:
        raise DataError("Synthetic archive requires at least four scenes")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def payload(header, rows):
        stream = io.StringIO(newline="")
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerows(rows)
        return stream.getvalue().encode("utf-8-sig")

    # x mode prevents a smoke run from replacing a previously supplied archive.
    with zipfile.ZipFile(path, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for sid in range(n_scenes):
            labels, measurements = [], []
            batch = 0
            for identity in range(2 + sid % 2):
                for fragment in range(2 + (identity + sid) % 2):
                    source = str(9001 + fragment % 2)
                    start = float(3 * fragment + (sid % 3) * 0.2 + identity)
                    times = start + np.asarray([0.0, 1.7, 4.3, 8.0 + identity * 0.3])
                    ancestor = str(100000 + sid * 10 + identity)
                    mmsi = "%s-0-%d" % (ancestor, fragment)
                    labels.append([times[0], times[-1], mmsi, source, str(batch)])
                    for time in times:
                        noise = 0.0001 * fragment
                        measurements.append([str(batch), source, float(time),
                                             20 + identity * 0.03 + float(time) * 0.0002 + noise,
                                             130 + identity * 0.04 + float(time) * 0.0003 - noise,
                                             5 + identity + fragment * 0.01,
                                             (350 + identity * 20 + float(time) * 0.4) % 360])
                    batch += 1
            measurements.sort(key=lambda row: (row[2], row[1], int(row[0])))
            for member, header, rows in (
                ("关联表/关联结果-%d.csv" % sid, ["t_s", "t_e", "mmsi", "source", "batch"], labels),
                ("量测场景/场景-%d.csv" % sid, ["batch", "source", "time", "lat", "lon", "vel", "cou"], measurements),
            ):
                # Fixed timestamp makes byte-level fixture hashes reproducible.
                info = zipfile.ZipInfo("数据集level2/" + member, date_time=(2026, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                archive.writestr(info, payload(header, rows))
    return path
