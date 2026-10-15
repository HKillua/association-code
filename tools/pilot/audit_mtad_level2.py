#!/usr/bin/env python3
"""Read-only structural and sampled-content acceptance for the official MTAD Level2 ZIP.

No archive members are extracted. Only a new output directory is written.
This is neither an official association scorer nor a physical truth validation.
"""
import argparse
from collections import Counter, defaultdict
import csv
import datetime
import hashlib
import io
import json
import math
from pathlib import Path, PurePosixPath
import shlex
import sys
import zipfile

OFFICIAL_MD5 = "8ddcea161a49b47ecdac72ae45804438"
DEFAULT_SCENES = (0, 1, 2, 100, 4999)
HEADERS = {
    "labels": ["t_s", "t_e", "mmsi", "source", "batch"],
    "reference": ["MMSI", "time", "lat", "lon", "vel", "cou"],
    "measurements": ["batch", "source", "time", "lat", "lon", "vel", "cou"],
}
NUMERIC = {
    "labels": ("t_s", "t_e"),
    "reference": ("time", "lat", "lon", "vel", "cou"),
    "measurements": ("time", "lat", "lon", "vel", "cou"),
}


class AuditError(ValueError):
    pass


def hashes(path):
    md5, sha256 = hashlib.md5(), hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            md5.update(chunk)
            sha256.update(chunk)
    return {"md5": md5.hexdigest(), "sha256": sha256.hexdigest()}


def decoded_name(info):
    name = info.filename
    if not info.flag_bits & 0x800:
        try:
            name = name.encode("cp437").decode("gbk")
        except UnicodeError:
            pass
    return name


def safe_name(name):
    parts = PurePosixPath(name).parts
    return bool(parts) and not name.startswith("/") and ".." not in parts and "\\" not in name and ":" not in name and "\x00" not in name


def scene_paths(scene_id):
    root = "数据集level2/"
    return {
        "labels": root + "关联表/关联结果-%d.csv" % scene_id,
        "reference": root + "真实场景/场景-%d.csv" % scene_id,
        "measurements": root + "量测场景/场景-%d.csv" % scene_id,
    }


def read_csv(zf, info, kind, coverage=None):
    # Read only this sampled member in memory. CRC is checked before content
    # validation, so NaN or a bad header does not prevent that CRC check.
    data = zf.read(info)
    if coverage is not None:
        coverage["sample_csv_members_read"] += 1
    with io.StringIO(data.decode("utf-8-sig"), newline="") as stream:
        reader = csv.DictReader(stream, strict=True)
        if reader.fieldnames != HEADERS[kind]:
            raise AuditError("%s: expected header %r; got %r" % (decoded_name(info), HEADERS[kind], reader.fieldnames))
        rows, errors = [], []
        for row in reader:
            if None in row or any(value is None or not value.strip() for value in row.values()):
                errors.append("line %d: wrong column count or empty value" % reader.line_num)
                continue
            invalid = False
            for key in NUMERIC[kind]:
                try:
                    value = float(row[key])
                except ValueError:
                    errors.append("line %d: non-numeric %s" % (reader.line_num, key))
                    invalid = True
                    continue
                if not math.isfinite(value):
                    errors.append("line %d: non-finite %s" % (reader.line_num, key))
                    invalid = True
                row[key] = value
            if invalid:
                continue
            if kind == "labels" and row["t_e"] < row["t_s"]:
                errors.append("line %d: reversed label interval" % reader.line_num)
                continue
            rows.append(row)
        if errors:
            raise AuditError("%s: %d invalid field/row findings: %s" % (decoded_name(info), len(errors), "; ".join(errors)))
        return rows


def track_stats(rows, key_fields):
    groups = defaultdict(list)
    out_of_bounds = 0
    for row in rows:
        groups[tuple(row[k] for k in key_fields)].append(row["time"])
        out_of_bounds += not (-90 <= row["lat"] <= 90 and -180 <= row["lon"] <= 180)
    return {
        "points": len(rows), "tracks": len(groups),
        "nonmonotonic_tracks": sum(any(b < a for a, b in zip(times, times[1:])) for times in groups.values()),
        "decreasing_time_transitions": sum(sum(b < a for a, b in zip(times, times[1:])) for times in groups.values()),
        "duplicate_time_points_beyond_first": sum(len(times) - len(set(times)) for times in groups.values()),
        "coordinate_out_of_bounds_points": out_of_bounds,
    }


def prefix_conflicts(ids):
    groups = defaultdict(set)
    for identity in ids:
        groups[identity.split("-", 1)[0]].add(identity)
    return [{"A": key, "A_B_values": sorted(values)} for key, values in sorted(groups.items()) if len(values) > 1]


def ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def audit_scene(scene_id, labels, reference, measurements):
    label_groups, measurement_groups = defaultdict(list), defaultdict(list)
    ref_groups, ref_times = defaultdict(list), defaultdict(list)
    for row in labels:
        label_groups[(row["source"], row["batch"])].append(row)
    for row in measurements:
        measurement_groups[(row["source"], row["batch"])].append(row)
    for row in reference:
        ref_groups[row["MMSI"]].append(row)
        ref_times[(row["MMSI"], row["time"])].append(row)
    ambiguous = []
    for key, entries in sorted(label_groups.items()):
        identities = sorted({r["mmsi"] for r in entries})
        if len(identities) > 1:
            ambiguous.append({"source": key[0], "batch": key[1], "identities": identities})
    source_counts = defaultdict(Counter)
    interval_summary = Counter()
    label_bounds = []
    for key, entries in sorted(label_groups.items()):
        points = measurement_groups.get(key, [])
        for label in entries:
            inside = [r["time"] for r in points if label["t_s"] <= r["time"] <= label["t_e"]]
            interval_summary["label_rows"] += 1
            if not inside:
                interval_summary["label_rows_without_measurements_in_interval"] += 1
                continue
            minimum, maximum = min(inside), max(inside)
            start_delta, end_delta = minimum - label["t_s"], maximum - label["t_e"]
            interval_summary["label_rows_with_measurements_in_interval"] += 1
            if start_delta != 0 or end_delta != 0:
                interval_summary["label_rows_with_nonexact_observed_boundaries"] += 1
            label_bounds.append({"source": key[0], "batch": key[1], "mmsi": label["mmsi"],
                                 "t_s": label["t_s"], "t_e": label["t_e"],
                                 "measurement_min_inside": minimum, "measurement_max_inside": maximum,
                                 "start_delta": start_delta, "end_delta": end_delta})
    for row in measurements:
        counts = source_counts[row["source"]]
        counts["measurement_points"] += 1
        entries = label_groups.get((row["source"], row["batch"]), [])
        if not entries:
            counts["missing_label_key_points"] += 1
            continue
        matches = {r["mmsi"] for r in entries if r["t_s"] <= row["time"] <= r["t_e"]}
        if not matches:
            counts["outside_all_label_intervals_points"] += 1
            continue
        if len(matches) != 1:
            counts["ambiguous_label_at_time_points"] += 1
            continue
        identity = next(iter(matches))
        counts["unique_label_in_interval_points"] += 1
        if "-" not in identity:
            counts["label_without_reference_suffix_points"] += 1
            continue
        ref_id = identity.rsplit("-", 1)[0]
        if ref_id not in ref_groups:
            counts["missing_reference_identity_points"] += 1
            continue
        counts["candidate_reference_identity_found_points"] += 1
        candidates = ref_times.get((ref_id, row["time"]), [])
        if not candidates:
            counts["missing_exact_reference_time_points"] += 1
            continue
        counts["exact_reference_time_found_points"] += 1
        if len(candidates) > 1:
            counts["duplicate_reference_time_points"] += 1
        if any(row["lat"] == ref["lat"] and row["lon"] == ref["lon"] for ref in candidates):
            counts["exact_time_position_match_points"] += 1
        if any(all(row[k] == ref[k] for k in ("lat", "lon", "vel", "cou")) for ref in candidates):
            counts["exact_time_four_features_match_points"] += 1
    sources = {}
    count_names = ["measurement_points", "missing_label_key_points", "outside_all_label_intervals_points",
                   "ambiguous_label_at_time_points", "unique_label_in_interval_points", "label_without_reference_suffix_points",
                   "missing_reference_identity_points", "candidate_reference_identity_found_points",
                   "missing_exact_reference_time_points", "exact_reference_time_found_points", "duplicate_reference_time_points",
                   "exact_time_position_match_points", "exact_time_four_features_match_points"]
    for source, counts in sorted(source_counts.items()):
        sources[source] = {key: counts[key] for key in count_names}
        sources[source]["position_match_fraction_of_all_measurements"] = ratio(counts["exact_time_position_match_points"], counts["measurement_points"])
        sources[source]["position_match_fraction_given_exact_reference_time"] = ratio(counts["exact_time_position_match_points"], counts["exact_reference_time_found_points"])
    label_reference_ids = {r["mmsi"].rsplit("-", 1)[0] for r in labels if "-" in r["mmsi"]}
    warning_counts = Counter()
    warning_counts["ambiguous_label_keys"] = len(ambiguous)
    warning_counts["reference_identities_without_any_label"] = len(set(ref_groups) - label_reference_ids)
    warning_counts["label_reference_identities_absent_from_reference"] = len(label_reference_ids - set(ref_groups))
    warning_counts["measurement_tracks_without_label_key"] = len(set(measurement_groups) - set(label_groups))
    warning_counts["label_keys_without_measurements"] = len(set(label_groups) - set(measurement_groups))
    warning_counts["reference_A_with_multiple_A_B"] = len(prefix_conflicts(ref_groups))
    for counts in source_counts.values():
        for key in ("missing_label_key_points", "outside_all_label_intervals_points", "ambiguous_label_at_time_points",
                    "label_without_reference_suffix_points", "missing_reference_identity_points",
                    "missing_exact_reference_time_points", "duplicate_reference_time_points"):
            warning_counts[key] += counts[key]
    stats = {"measurements": track_stats(measurements, ("source", "batch")), "reference": track_stats(reference, ("MMSI",))}
    for kind, values in stats.items():
        for field in ("nonmonotonic_tracks", "duplicate_time_points_beyond_first", "coordinate_out_of_bounds_points"):
            warning_counts[kind + "_" + field] = values[field]
    return {
        "scene_id": scene_id, "label_rows": len(labels), "label_keys": len(label_groups),
        "label_key_identity_is_unique": not ambiguous, "ambiguous_label_keys": ambiguous,
        "label_interval_summary": dict(interval_summary), "label_observed_interval_boundaries": label_bounds,
        "reference_A_to_multiple_A_B": prefix_conflicts(ref_groups),
        "label_reference_A_to_multiple_A_B": prefix_conflicts(label_reference_ids),
        "reference_identities_without_any_label": sorted(set(ref_groups) - label_reference_ids),
        "label_reference_identities_absent_from_reference": sorted(label_reference_ids - set(ref_groups)),
        "stats": stats, "source_comparisons": sources,
        "warning_counts": {k: v for k, v in warning_counts.items() if v},
    }


def markdown_report(report):
    coverage = report["coverage"]
    included = [scene["scene_id"] for scene in report["scenes"]]
    excluded = [scene for scene in report["scene_ids"] if scene not in included]
    lines = ["# MTAD Level2 可复跑只读验收", "", "状态：**%s**。这是结构/抽样内容验收，不是模型实验或官方评分。" % report["status"], "",
             "- ZIP MD5：`%s`；官方预期：`%s`。" % (report.get("archive", {}).get("md5", "unavailable"), report["expected_md5"]),
             "- 全量中央目录检查：%s 个 CSV；预期 %s 个。" % (coverage.get("indexed_csv_members", 0), coverage["expected_csv_members"]),
             "- 选定场景：%s；成功完整读入 %s 个 CSV（包含这些成员的 CRC 检查）。" % (", ".join(map(str, report["scene_ids"])), coverage["sample_csv_members_read"]),
             "- 字段内容检查通过：%s / %s 个已读 CSV。" % (coverage["sample_csv_members_content_valid"], coverage["sample_csv_members_read"]),
             "- 未选场景的 CSV 内容和 CRC 未检查；ZIP 全文件哈希已单独核验。",
             "- 状态 failed 表示数据验收未通过；errors 中记录内容异常或输入问题，不意味着脚本崩溃。", "",
             "| 场景 | 量测轨迹/点数 | reference 轨迹/点数 | 标签键唯一 | 同 A 多 A-B |", "|---|---:|---:|---|---:|"]
    for scene in report["scenes"]:
        m, r = scene["stats"]["measurements"], scene["stats"]["reference"]
        lines.append("| %d | %d / %d | %d / %d | %s | %d |" % (scene["scene_id"], m["tracks"], m["points"], r["tracks"], r["points"], scene["label_key_identity_is_unique"], len(scene["reference_A_to_multiple_A_B"])))
    lines += ["", "以下聚合仅包含内容验收完整的场景：%s；未进入聚合的选定场景：%s（原因见失败记录）。" % (", ".join(map(str, included)) or "无", ", ".join(map(str, excluded)) or "无"), "",
              "同时间位置完全相同：先用 `(source,batch)` 与包含端点的标签时间区间匹配，再以标签去末段作为 reference 候选身份；不插值。", "",
              "| 源 | 全部量测点 | 找到同时间 reference | 同时间位置完全相同 | 相同/全部 | 相同/同时间 |", "|---|---:|---:|---:|---:|---:|"]
    for source, counts in report["aggregate_source_comparisons"].items():
        total, found, same = counts.get("measurement_points", 0), counts.get("exact_reference_time_found_points", 0), counts.get("exact_time_position_match_points", 0)
        fractions = ["%.4f%%" % (100 * same / d) if d else "未定义" for d in (total, found)]
        lines.append("| %s | %d | %d | %d | %s | %s |" % (source, total, found, same, *fractions))
    lines += ["", "数据警告（只说明抽样数据特点，不自动使验收失败）：", "", "```json", json.dumps(report["aggregate_warning_counts"], ensure_ascii=False, indent=2), "```", "", "限制："]
    lines += ["- " + value for value in report["limitations"]]
    if report["errors"]:
        lines += ["", "失败："] + ["- " + message for message in report["errors"]]
    lines += ["", "复跑命令：", "", "```sh", report["command"], "```", "", "机器可读细节见 `audit.json`；完整成员元数据见 `archive_index.json`。", ""]
    return "\n".join(lines)


def run_audit(archive, output, scene_ids=DEFAULT_SCENES, expected_md5=OFFICIAL_MD5,
              expected_scene_ids=range(5000), command=None):
    archive, output = Path(archive).expanduser().resolve(), Path(output).expanduser()
    # Atomic directory creation prevents overwriting any earlier run, including a failed run.
    output.mkdir(parents=True, exist_ok=False)
    expected_scene_ids, scene_ids = set(expected_scene_ids), list(dict.fromkeys(scene_ids))
    report = {
        "audit_version": "1.2", "created_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "status": "failed", "archive_path": str(archive), "expected_md5": expected_md5,
        "script_sha256": hashes(Path(__file__).resolve())["sha256"],
        "command": command or shlex.join([sys.executable, str(Path(__file__).resolve()), "--archive", str(archive), "--output", str(output), "--scene-ids", ",".join(map(str, scene_ids))]),
        "scene_ids": scene_ids, "selection": "Explicit deterministic scene IDs; convenience sample, not a random prevalence estimate.",
        "coverage": {"expected_scene_count": len(expected_scene_ids), "expected_csv_members": 3 * len(expected_scene_ids), "sample_csv_members_read": 0, "sample_csv_members_content_valid": 0},
        "scenes": [], "aggregate_source_comparisons": {}, "aggregate_warning_counts": {}, "errors": [],
        "limitations": [
            "全量内容/CRC 未检查；只对选定场景完整读取 CSV，所有场景仅检查中央目录与 ZIP 全文件哈希。",
            "样本成员先完整读取并校验 CRC，再校验字段。含无效字段的场景不生成匹配统计，避免默默删掉异常行改变分母；其他场景继续验收。",
            "去掉标签 mmsi 最后一段只构成待验证的 reference 候选映射；不是官方身份定义或官方评分实现。",
            "数值相等率不等于物理精度或融合误差；目录名真实场景不足以证明无偏物理真值。",
            "同时间指解析后的浮点数完全相等，无容差、插值或重采样；重复 reference 时刻存在时按任一位置匹配，另记重复计数。",
              "本脚本不估计或校正两类 CSV 的时间偏移；raw exact 匹配缺失可能来自时钟偏移，不可解释成测量误差。",
              "时间、速度、航向等单位尚待文档核验；经纬度范围检查仅按字段名称作常规坐标合理性筛查。",
            "抽样警告不外推至完整数据集；统计中的轨迹身份与数量均限定在各自场景内。",
        ],
    }
    index = []
    try:
        if not scene_ids or set(scene_ids) - expected_scene_ids:
            raise AuditError("Selected scene IDs must be a nonempty subset of expected scene IDs")
        report["archive"] = {"size_bytes": archive.stat().st_size, **hashes(archive)}
        if report["archive"]["md5"] != expected_md5:
            report["errors"].append("Archive MD5 mismatch: expected %s, got %s" % (expected_md5, report["archive"]["md5"]))
        with zipfile.ZipFile(archive) as zf:
            members = {}
            for info in zf.infolist():
                name = decoded_name(info)
                index.append({"path": name, "size_bytes": info.file_size, "compressed_bytes": info.compress_size,
                              "crc32": "%08x" % info.CRC, "is_directory": info.is_dir()})
                if not safe_name(name):
                    report["errors"].append("Unsafe archive member path (never extracted): %r" % name)
                    continue
                if name in members:
                    report["errors"].append("Duplicate decoded archive member: %s" % name)
                members[name] = info
            expected_paths = {path for scene_id in expected_scene_ids for path in scene_paths(scene_id).values()}
            missing = sorted(path for path in expected_paths if path not in members or members[path].is_dir())
            csv_paths = {name for name, info in members.items() if not info.is_dir() and name.lower().endswith(".csv")}
            unexpected = sorted(csv_paths - expected_paths)
            report["coverage"].update({"indexed_members": len(index), "indexed_csv_members": len(csv_paths), "missing_expected_csv_members": missing, "unexpected_csv_members": unexpected})
            if missing:
                report["errors"].append("Missing expected CSV members (%d): %s" % (len(missing), ", ".join(missing[:10])))
            if unexpected:
                report["errors"].append("Unexpected CSV members (%d): %s" % (len(unexpected), ", ".join(unexpected[:10])))
            if not report["errors"]:
                for scene_id in scene_ids:
                    tables = {}
                    for kind, path in scene_paths(scene_id).items():
                        try:
                            tables[kind] = read_csv(zf, members[path], kind, report["coverage"])
                            report["coverage"]["sample_csv_members_content_valid"] += 1
                        except (AuditError, UnicodeError, csv.Error, zipfile.BadZipFile, RuntimeError, OSError) as exc:
                            report["errors"].append("Scene %d: %s" % (scene_id, exc))
                    if len(tables) == 3:
                        report["scenes"].append(audit_scene(scene_id, **tables))
    except (AuditError, OSError, zipfile.BadZipFile, RuntimeError) as exc:
        report["errors"].append(str(exc))
    aggregates, warnings = defaultdict(Counter), Counter()
    for scene in report["scenes"]:
        warnings.update(scene["warning_counts"])
        for source, values in scene["source_comparisons"].items():
            aggregates[source].update({key: value for key, value in values.items() if isinstance(value, int)})
    report["aggregate_source_comparisons"] = {key: dict(value) for key, value in sorted(aggregates.items())}
    report["aggregate_warning_counts"] = dict(warnings)
    report["status"] = "passed_with_data_warnings" if not report["errors"] and warnings else "passed" if not report["errors"] else "failed"
    for name, value in (("audit.json", report), ("archive_index.json", index)):
        with (output / name).open("x", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
    with (output / "audit.md").open("x", encoding="utf-8") as stream:
        stream.write(markdown_report(report))
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, help="Official Level2 ZIP, read only")
    parser.add_argument("--output", required=True, help="New report directory; never overwrite an existing directory")
    parser.add_argument("--scene-ids", default=",".join(map(str, DEFAULT_SCENES)), help="Comma-separated IDs, each in 0..4999")
    args = parser.parse_args(argv)
    try:
        scene_ids = [int(value) for value in args.scene_ids.split(",")]
        if not scene_ids or any(value < 0 or value > 4999 for value in scene_ids):
            raise ValueError("out of range")
    except ValueError:
        parser.error("--scene-ids must be comma-separated integers in 0..4999")
    try:
        report = run_audit(args.archive, args.output, scene_ids)
    except FileExistsError:
        parser.error("Output already exists; choose a new --output directory. Nothing was overwritten.")
    except OSError as exc:
        print("Cannot write audit output: %s" % exc, file=sys.stderr)
        return 2
    print("%s: %s" % (report["status"], Path(args.output).resolve()))
    for error in report["errors"]:
        print(error, file=sys.stderr)
    return 1 if report["errors"] else 0


if __name__ == "__main__":
    sys.exit(main())
