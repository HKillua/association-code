"""Replay the local KF ancestry of reciprocal reports on the frozen MTAD graph."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gzip
import json
from pathlib import Path
import pickle
from time import perf_counter

from .async_report_scores import _innovation_variance
from .continuous_tracker import MEMORY_S, LocalTrack, Plane
from .cross_source_calibration import predictive_scores
from .disjoint_report_closure import SIDES, score_pair
from .hypothesis_gap_diagnostic import _load_frozen, _update_priors
from .interval_assignment_experiment import _roster
from .nemot_experiment import _events, _write_json, read_roster
from .nemot_replay import _add, _registration_shifts
from .reciprocal_report_downdate import score_removal
from .registration_probe import FIT_STOP_S
from .relative_motion_gap import _same, _sub


KINDS = ("reciprocal_release", "reciprocal_retain")


def _read_capture(path):
    with gzip.open(path, "rb") as stream:
        return pickle.load(stream)


def _requests(samples, reports):
    selected = defaultdict(set)
    for window, frozen in zip(samples, reports, strict=True):
        for (sample, single), report in zip(window, frozen, strict=True):
            stamp = sample["stamp"]
            if stamp != report["stamp"] or single["scores"] != report["single"]:
                raise ValueError("the frozen C-ON42 and C-ON43 readouts disagree")
            for kind in KINDS:
                if kind not in report["pairs"]:
                    continue
                pair = report["pairs"][kind]
                selected[pair["earlier_stamp"]].add(sample["partners"][kind.removeprefix("reciprocal_")])
                selected[stamp].update((sample["reporter"], *sample["partners"].values()))
    return selected


def _replay(events, native_reports, selected):
    plane = Plane.from_first_batch(events[0][1].values())
    needed_keys = set().union(*selected.values())
    priors, history = {}, {}
    for (stamp, observations), native in zip(events, native_reports, strict=True):
        if (stamp, tuple(sorted(observations))) != native:
            raise ValueError("the native clock differs from the frozen candidate replay")
        points = {key: plane.project(point) for key, point in observations.items() if key in needed_keys}
        for key in selected.get(stamp, ()):
            history[stamp, key] = (priors.get(key), points.get(key))
        _update_priors(priors, stamp, points)
    if len(history) != sum(map(len, selected.values())):
        raise ValueError("some reciprocal history requests are outside the native clock")
    return history


def _reconcile(sample, history, shifts):
    stamp, reporter = sample["stamp"], sample["reporter"]
    before, observation = history[stamp, reporter]
    origin = (0., 0.)
    reported_shift = shifts.get(reporter[0], origin)
    forecast = _add(before.position(stamp, True), reported_shift)
    actual = _add(observation, reported_shift)
    for side, partner in sample["partners"].items():
        state, _ = history[stamp, partner]
        prediction = _add(state.position(stamp, True), shifts.get(partner[0], origin))
        age = stamp - state.stamp
        checks = zip((*sample["cross"][side], *sample["pred_delta"][side],
                      sample["age"][side], sample["variance"][side]),
                     (*_sub(actual, prediction), *_sub(forecast, prediction),
                      age, _innovation_variance(state, age)), strict=True)
        if not all(_same(saved, replayed) for saved, replayed in checks):
            raise ValueError("the original reciprocal input did not reproduce")


def _later_only(newer, removal):
    side, delta = removal["side"], removal["prediction_delta"]
    cross = newer["cross"] | {side: _add(newer["cross"][side], delta)}
    masked = newer["pred_delta"] | {side: _add(newer["pred_delta"][side], delta)}
    variance = newer["variance"] | {side: removal["omitted_variance"]}
    score = predictive_scores(*(mapping[name] for mapping in (cross, masked, variance) for name in SIDES), None)
    return dict(single_omitted=score["raw"], single_omitted_masked=score["masked"])


def _freeze(samples, reports, events, original):
    history = _replay(events, original.native_reports, _requests(samples, reports))
    shifts = _registration_shifts(original.fit)
    captured = []
    for window, frozen in zip(samples, reports, strict=True):
        indexed = {sample["stamp"]: sample for sample, _ in window}
        readouts = []
        for item in frozen:
            stamp = item["stamp"]
            newer = indexed[stamp]
            pairs = {}
            for kind in KINDS:
                if kind not in item["pairs"]:
                    continue
                baseline = item["pairs"][kind]
                older = indexed[baseline["earlier_stamp"]]
                if any(baseline[name] != value for name, value in score_pair(older, newer).items()):
                    raise ValueError("the original reciprocal pair scores did not reproduce")
                _reconcile(newer, history, shifts if stamp > FIT_STOP_S else {})
                key = older["reporter"]
                prior, point = history[older["stamp"], key]
                posterior, _ = history[stamp, key]
                if point is None or posterior is None or posterior.observation != point:
                    raise ValueError("the reused state does not include the recorded earlier report")
                stamp_old = older["stamp"]
                expected = (prior.updated(stamp_old, point) if prior is not None and stamp_old-prior.stamp <= MEMORY_S
                            else LocalTrack(stamp_old, point, point))
                if expected != posterior:
                    raise ValueError("the earlier update did not reproduce its frozen posterior")
                removed = score_removal(older, newer, baseline, prior, posterior)
                if removed["status"] == "supported":
                    removed.update(_later_only(newer, removed))
                pairs[kind] = dict(original=baseline, removal=removed)
            readouts.append(item | {"pairs": pairs})
        captured.append(readouts)
    return captured


def _join(captured, previous):
    output = []
    counts = Counter()
    for reports, old in zip(captured, previous, strict=True):
        for report, before in zip(reports, old["reports"], strict=True):
            if (report["stamp"], report["single"]) != (before["stamp"], before["single"]):
                raise ValueError("the evaluated report clock or controls changed")
            expected = before["pairs"].keys() & set(KINDS)
            if report["pairs"].keys() != expected:
                raise ValueError("the original reciprocal-pair census changed")
            moment = "early" if report["stamp"] < old["stop"] else "boundary"
            for kind, pair in report["pairs"].items():
                if (pair["original"]["older_stamp"], pair["original"]["newer_stamp"]) != (
                        before["pairs"][kind]["older_stamp"], before["pairs"][kind]["newer_stamp"]):
                    raise ValueError("the evaluated reciprocal pair changed")
                counts[moment] += 1
        output.append(old | {"reports": reports})
    return output, counts


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, nargs=3, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--gap-run", type=Path, required=True)
    parser.add_argument("--anchor-run", type=Path, required=True)
    parser.add_argument("--closure-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    scenes = _roster(args.manifest)["splits"]["development"]
    if len(scenes) != 28 or scenes != read_roster(args.manifest)["splits"]["held"]:
        raise ValueError("the diagnostic requires the original 28 scenes")
    args.output_dir.mkdir(mode=0o700, exist_ok=False)
    frozen = args.output_dir / "frozen"
    frozen.mkdir(mode=0o700)
    began = perf_counter()
    windows = reports = 0
    for index, scene in enumerate(scenes, 1):
        name = f"scene_{index:02d}.pkl.gz"
        original, _ = _load_frozen(args.gap_run / "private_frozen" / name)
        captured = _freeze(_read_capture(args.anchor_run / "private_frozen" / name),
                           _read_capture(args.closure_run / "frozen" / name),
                           _events(args.archive, args.manifest, scene), original)
        with gzip.open(frozen / name, "xb", compresslevel=1) as stream:
            pickle.dump(captured, stream, protocol=pickle.HIGHEST_PROTOCOL)
        windows += len(captured)
        reports += sum(map(len, captured))
        print(f"Frozen {index}/{len(scenes)}, {len(captured)} groups", flush=True)
    print("All report-removal contrasts frozen. Joining previous anonymous evaluation.", flush=True)
    previous = json.loads((args.closure_run / "summary.json").read_text(encoding="utf-8"))
    if (windows, reports) != (previous["frozen_windows"], previous["frozen_batches"]):
        raise ValueError("the original window or report census changed")
    output, census = {}, Counter()
    for index in range(1, len(scenes)+1):
        alias = f"scene_{index:02d}"
        old = previous["scenes"][alias]
        rows, counts = _join(_read_capture(frozen / f"{alias}.pkl.gz"), old["rows"])
        census.update(counts)
        output[alias] = {key: old[key] for key in ("component", "sparse")} | {"rows": rows}
    if census != {"early": 618, "boundary": 429}:
        raise ValueError("the original 618 early and 429 boundary reciprocal reads did not reproduce")
    _write_json(args.output_dir / "summary.json", dict(scenes=output, clock=previous["clock"],
                candidate_graph_unchanged=True, publications_unchanged=True, training_updates=0,
                frozen_windows=windows, frozen_batches=reports, reciprocal_reads=dict(census),
                wall_seconds=perf_counter()-began))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
