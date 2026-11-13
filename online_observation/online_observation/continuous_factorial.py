"""Run the frozen four-arm Level1 continuous identity comparison."""
from __future__ import annotations

import argparse
from array import array
from collections import Counter
import gzip
import json
import math
from pathlib import Path
import pickle
from tempfile import TemporaryDirectory
from time import perf_counter, perf_counter_ns

from .continuous_metrics import evaluate_publications
from .continuous_tracker import (
    ACCELERATION_SPECTRAL_DENSITY, ARMS, FRESH_S, MEMORY_S, NULL_COST, POSITION_VARIANCE,
    RADIUS_M, VELOCITY_VARIANCE, ContinuousTracker,
)
from .evaluation_data import read_labels
from .measurements import read_archive_rows
from .positions import event_batches
from .registration_probe import FIT_STOP_S, estimate_translation


ROSTER = {
    "largest_component": ("1975", "2537", "3368", "3844", "4276", "4383"),
    "second_component": ("1059", "2636", "2970", "3122", "3300", "4728"),
    "other_touched_components": ("2408", "2449", "2980", "3169", "3885", "3928"),
    "untouched_components": ("2184", "2291", "3760", "4175", "4615", "4646"),
}
BIRTH_DEADLINES_S = (1, 5, 20, 60, 120)
QUANTILES = (.50, .95, .99)


def replay_arm(events, stop, arm):
    tracker = ContinuousTracker(arm)
    pending = iter(events)
    event = next(pending, None)
    prefix = []
    fit = None
    last_memberships = None
    runs = []
    service_ns, ready_ns = array("Q"), array("Q")
    age_ms = Counter()
    peak_published_tracks = 0
    prior_ready = 0
    for tick in range(stop + 1):
        began = perf_counter_ns()
        if tick == FIT_STOP_S + 1:
            fit = estimate_translation(prefix)
            tracker.install_registration(fit)
        while event is not None and event[0] <= tick:
            stamp, positions = event
            tracker.ingest(stamp, positions)
            if stamp <= FIT_STOP_S:
                prefix.append(event)
            event = next(pending, None)
        publication = tracker.publish(tick)
        service = perf_counter_ns() - began
        release = tick * 1_000_000_000
        prior_ready = max(release, prior_ready) + service
        service_ns.append(service)
        ready_ns.append(prior_ready - release)
        tracks = publication["tracks"]
        peak_published_tracks = max(peak_published_tracks, len(tracks))
        memberships = tuple((track["track_id"], track["members"]) for track in tracks)
        if memberships != last_memberships:
            runs.append((tick, memberships))
            last_memberships = memberships
        age_ms.update(math.ceil((tick - track["measurement_time"]) * 1000) for track in tracks)
    if event is not None:
        raise ValueError("replay stopped before the last native event")
    return runs, dict(service_ns=service_ns, ready_ns=ready_ns, age_ms=age_ms,
                      peak_published_tracks=peak_published_tracks), dict(tracker.diagnostics), fit


def publications(runs, stop):
    """Reconstruct every original membership snapshot from its lossless run encoding."""
    changes = iter(runs)
    change = next(changes, None)
    members = None
    for tick in range(stop + 1):
        if change is not None and change[0] == tick:
            members = change[1]
            change = next(changes, None)
        if members is None:
            raise ValueError("membership recording lacks its first snapshot")
        yield dict(query_time=tick, tracks=[dict(track_id=track_id, members=keys) for track_id, keys in members])
    if change is not None:
        raise ValueError("membership recording exceeds the frozen final tick")


def _quantiles(values, scale):
    ordered = sorted(values)
    return {f"p{int(q * 100)}": ordered[math.ceil(q * len(ordered)) - 1] / scale if ordered else None
            for q in QUANTILES} | {"max": ordered[-1] / scale if ordered else None}


def _histogram_quantiles(histogram, probabilities, scale=1):
    total = sum(histogram.values())
    ordered = sorted(histogram.items())
    values = {}
    for q in probabilities:
        index = math.ceil(q * total)
        remaining = index
        quantile = None
        for value, count in ordered:
            remaining -= count
            if remaining <= 0:
                quantile = value / scale
                break
        values[f"p{int(q * 100)}"] = quantile
    return values


def _timing_summary(timing):
    service, ready = timing["service_ns"], timing["ready_ns"]
    wait = (result - update for result, update in zip(ready, service, strict=True))
    return dict(
        service_ms=_quantiles(service, 1e6), modeled_ready_ms=_quantiles(ready, 1e6),
        modeled_queue_wait_ms=_quantiles(wait, 1e6), ticks=len(service),
        ready_after_next_release=sum(value > 1_000_000_000 for value in ready),
        published_state_age_seconds=_histogram_quantiles(timing["age_ms"], QUANTILES, 1000),
        peak_published_tracks=timing["peak_published_tracks"],
    )


def _accumulate_timing(pooled, timing):
    for key in ("service_ns", "ready_ns"):
        pooled[key].extend(timing[key])
    pooled["age_ms"].update(timing["age_ms"])
    pooled["peak_published_tracks"] = max(pooled["peak_published_tracks"], timing["peak_published_tracks"])


def _sum_dicts(parts):
    result = Counter()
    for part in parts:
        result.update(part)
    return dict(result)


def _delay_counts(summary):
    histogram = summary["first_correct_delay_seconds"]["ceil_seconds_histogram"]
    return {str(deadline): sum(count for seconds, count in histogram.items() if int(seconds) <= deadline)
            for deadline in BIRTH_DEADLINES_S}


def _pool_births(parts):
    histogram = _sum_dicts(p["first_correct_delay_seconds"]["ceil_seconds_histogram"] for p in parts)
    result = dict(births=sum(p["births"] for p in parts),
                  ever_correct=sum(p["ever_correct"] for p in parts),
                  unconfirmed_by_end=sum(p["unconfirmed_by_end"] for p in parts),
                  first_publication=_sum_dicts(p["first_publication"] for p in parts),
                  first_correct_delay_seconds=dict(
                      ceil_seconds_histogram=histogram,
                      rounded_up_quantiles=_histogram_quantiles(histogram, (.50, .95))))
    result["first_correct_by_seconds"] = _delay_counts(result)
    return result


def _pair_rates(counts):
    correct = counts["correct_pairs"]
    wrong = counts["false_pairs"]
    missed = counts["missed_pairs"]
    return dict(precision=correct / (correct + wrong) if correct + wrong else None,
                recall=correct / (correct + missed) if correct + missed else None,
                f1=2 * correct / (2 * correct + wrong + missed) if 2 * correct + wrong + missed else None,
                errors=wrong + missed)


def _pool(parts):
    pair_counts = {scope: _sum_dicts(p["pairs"][scope] for p in parts) for scope in ("cross_source", "same_source")}
    births = [p["births"] for p in parts]
    opportunity = {key: _pool_births([p["by_opportunity"][key] for p in births])
                   for key in births[0]["by_opportunity"]}
    sources = sorted({source for p in parts for source in p["by_source"]})
    return dict(counts=_sum_dicts(p["counts"] for p in parts),
                pairs={scope: dict(counts, rates=_pair_rates(counts)) for scope, counts in pair_counts.items()},
                target_seconds_by_active_source_count=_sum_dicts(
                    p["target_seconds_by_active_source_count"] for p in parts),
                by_source={source: _sum_dicts(p["by_source"][source] for p in parts if source in p["by_source"])
                           for source in sources},
                births=dict(_pool_births(births), by_opportunity=opportunity, by_source={
                    source: _pool_births([p["by_source"][source] for p in births if source in p["by_source"]])
                    for source in sources}),
                diagnostics=_sum_dicts(p["diagnostics"] for p in parts))


def _run(archive, manifest, spool):
    scene_to_stratum = {scene: group for group, scenes in ROSTER.items() for scene in scenes}
    metadata = json.loads(manifest.read_text(encoding="utf-8"))
    timing_pool = {arm: dict(service_ns=array("Q"), ready_ns=array("Q"), age_ms=Counter(), peak_published_tracks=0)
                   for arm in ARMS}
    scenes = {}
    for scene, stratum in scene_to_stratum.items():
        began = perf_counter()
        events = tuple(event_batches(read_archive_rows(archive, manifest, scene, required_split="T_core")))
        if events and events[0][0] < 0:
            raise ValueError(f"scene {scene} has a native report before the frozen clock start")
        stop = math.ceil(events[-1][0]) if events else 0
        prepared = dict(stratum=stratum, component_id=metadata["components"][scene], stop=stop,
                        native_events=len(events), preparation_seconds=perf_counter() - began, arms={})
        print(f"Predicting scene {scene} with {stop + 1} query ticks", flush=True)
        saved = {"events": [(stamp, tuple(positions)) for stamp, positions in events], "arms": {}}
        for arm in ARMS:
            runs, timing, diagnostics, fit = replay_arm(events, stop, arm)
            if arm == ARMS[0]:
                prepared["fit"] = fit
            elif prepared["fit"] != fit:
                raise RuntimeError("paired runs did not share the same prefix registration")
            saved["arms"][arm] = runs
            prepared["arms"][arm] = dict(timing=_timing_summary(timing), diagnostics=diagnostics)
            _accumulate_timing(timing_pool[arm], timing)
        with gzip.open(spool / f"{scene}.pkl.gz", "wb") as output:
            pickle.dump(saved, output, protocol=pickle.HIGHEST_PROTOCOL)
        scenes[scene] = prepared

    print(f"All {len(scenes)} scenes frozen. Scoring publication membership.", flush=True)
    for scene, prepared in scenes.items():
        with gzip.open(spool / f"{scene}.pkl.gz", "rb") as saved_file:
            saved = pickle.load(saved_file)
        labels = read_labels(archive, manifest, scene)
        for arm in ARMS:
            scores = evaluate_publications(saved["events"], publications(saved["arms"][arm], prepared["stop"]), labels)
            scores["births"]["first_correct_by_seconds"] = _delay_counts(scores["births"])
            prepared["arms"][arm].update(scores)

    return dict(
        scene_ids=list(scenes), scenes=scenes, arm_names=ARMS,
        aggregate={arm: dict(_pool([scene["arms"][arm] for scene in scenes.values()]),
                             timing=_timing_summary(timing_pool[arm]))
                   for arm in ARMS},
        clock="ideal_measurement_time", query_period_seconds=1, query_start=0,
        query_stop="ceil(last_native_report)", fresh_seconds=FRESH_S, memory_seconds=MEMORY_S,
        gate_metres=RADIUS_M, null_cost_per_endpoint_metres_squared=NULL_COST,
        motion=dict(position_variance=POSITION_VARIANCE, velocity_variance=VELOCITY_VARIANCE,
                    acceleration_spectral_density=ACCELERATION_SPECTRAL_DENSITY),
        registration_activation="strictly_after_600_seconds", queue_model="virtual_serial_fixed_release",
        service_measures="native_event_ingest+membership_and_position_snapshot; excludes scoring and serialization",
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", nargs=3, type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(args.output)
    began = perf_counter()
    with TemporaryDirectory(prefix="online-continuous-") as temporary:
        result = _run(args.archive, args.manifest, Path(temporary))
    result["wall_seconds"] = perf_counter() - began
    with args.output.open("x", encoding="utf-8") as output:
        json.dump(result, output, indent=2, allow_nan=False)
        output.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
