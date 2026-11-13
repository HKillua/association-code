"""Audit signed cross-source errors and previously qualified anchors on the frozen graph."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict, deque
import gzip
import json
from pathlib import Path
import pickle
from time import perf_counter

from .async_report_scores import _innovation_variance
from .causal_common_motion import compare_common_motion
from .continuous_tracker import FRESH_S, MEMORY_S, Plane
from .cross_source_calibration import AnchorSample, eligible_anchors, fit_moments, predictive_scores
from .evaluation_data import read_labels
from .hypothesis_gap_diagnostic import Switch, _legal, _load_frozen, _position_readouts, _update_priors
from .interval_assignment import MIN_SPAN_SECONDS
from .interval_assignment_experiment import _roster
from .nemot_experiment import _events, _write_json, read_roster
from .nemot_replay import _add, _other, _registration_shifts
from .registration_probe import FIT_STOP_S
from .relative_motion_gap import _same, _sub


SIDES = ("release", "retain")
STAGES = ("onset", "later_first", "later_last", "latest", "boundary")
TIERS = ("graph", "motion")


class AnchorHistory:
    def __init__(self):
        self.stamp = None
        self.registered = False
        self.stable = {}
        self.reports = defaultdict(list)
        self.samples = defaultdict(deque)
        self.counts = Counter()

    def read(self, keys, reporter, stamp, ages, shifts):
        records = self.samples[reporter[0]]
        while records and stamp - records[0][0].stamp > MEMORY_S:
            records.popleft()
        provenance = dict(records)
        output = {}
        for tier in TIERS:
            kwargs = {"motion": tier == "motion"}
            args = (keys, reporter, stamp, ages, bool(shifts))
            pool = eligible_anchors(provenance, *args, **kwargs)
            recent = eligible_anchors((item for item in provenance if stamp-item.stamp <= FRESH_S), *args, **kwargs)
            output[tier] = dict(n=len(pool), recent_n=len(recent), provenance=[(item, provenance[item]) for item in pool],
                                fit=fit_moments(pool) if len(pool) >= 2 else {"mean": None, "scale": None})
        return output

    def advance(self, stamp, points, priors, costs, selected, shifts):
        registered = bool(shifts)
        if registered != self.registered:
            self.stable.clear()
        for edge, since in self.stable.items():
            new = [key for key in edge if key in points]
            if len(new) != 1 or self.stamp-since < MIN_SPAN_SECONDS:
                continue
            if any(priors[key].stamp < since or not 0 < stamp-priors[key].stamp <= FRESH_S for key in edge):
                continue
            reporter, = new
            partner = _other(edge, reporter)
            age = stamp-priors[partner].stamp
            forecast = priors[partner].position(stamp, True)
            motion = compare_common_motion(self.reports[edge[0]], self.reports[edge[1]], self.stamp)
            shift_delta = _sub(shifts.get(reporter[0], (0., 0.)), shifts.get(partner[0], (0., 0.)))
            residual = _add(_sub(points[reporter], forecast), shift_delta)
            sample = AnchorSample(edge, reporter, stamp, age, residual, _innovation_variance(priors[partner], age),
                                  motion.qualified, registered)
            self.samples[reporter[0]].append((sample, (since, self.stamp, motion.available_at)))
            self.counts.update(writes=1, **{f"motion_{motion.status}": 1})

        degree = Counter(key for edge in costs for key in edge)
        self.stable = {edge: self.stable.get(edge, stamp) for edge in selected
                       if all(degree[key] == 1 for key in edge) and _legal((edge,), costs)}
        self.stamp, self.registered = stamp, registered
        for key, point in points.items():
            self.reports[key].append((stamp, point))


def _readout(switch, stamp, points, priors, shifts, bank):
    if len(switch.keys.intersection(points)) != 1:
        return None
    readout = _position_readouts(switch.keys, stamp, points, priors, shifts)
    if readout is None:
        return None
    forecasts, positions, new = readout
    reporter, = new
    partners = {name: _other(next(edge for edge in getattr(switch, name) if reporter in edge), reporter)
                for name in SIDES}
    ages = {name: stamp-priors[partner].stamp for name, partner in partners.items()}
    variance = {name: _innovation_variance(priors[partner], ages[name]) for name, partner in partners.items()}
    innovation = _sub(positions[reporter], forecasts[reporter])
    delta = {name: _sub(forecasts[reporter], forecasts[partner]) for name, partner in partners.items()}
    residual = {name: _sub(positions[reporter], forecasts[partner]) for name, partner in partners.items()}
    if not all(all(_same(actual, rebuilt) for actual, rebuilt in zip(residual[name], _add(innovation, delta[name])))
               for name in SIDES):
        raise ValueError("the signed cross-source innovation does not decompose")
    arguments = (*residual.values(), *delta.values(), *variance.values())
    anchors = bank.read(switch.keys, reporter, stamp, tuple(ages.values()), shifts)
    for tier in anchors.values():
        tier["scores"] = predictive_scores(*arguments, tier["fit"])
    return dict(stamp=stamp, reporter=reporter, partners=partners, keys=tuple(sorted(switch.keys)), cross=residual,
                pred_delta=delta, self=innovation, self_variance=_innovation_variance(priors[reporter], stamp-priors[reporter].stamp),
                age=ages, variance=variance, scores=predictive_scores(*arguments, None), anchors=anchors,
                shift_delta=_sub(shifts.get(reporter[0], (0., 0.)), shifts.get(partners["release"][0], (0., 0.))))


def _capture(events, result, graph, gaps):
    starts = defaultdict(list)
    switches = [Switch(**gap["switch"]) for gap in gaps]
    for index, gap in enumerate(gaps):
        starts[gap["start"]].append(index)
    output = [[] for _ in gaps]
    active = []
    plane = Plane.from_first_batch(events[0][1].values())
    shifts = _registration_shifts(result.fit)
    bank, priors = AnchorHistory(), {}
    candidate_count = 0
    for (stamp, observations), native in zip(events, result.native_reports, strict=True):
        graph_stamp, keys, _, _, candidates, selected = pickle.load(graph)
        if (stamp, tuple(sorted(observations))) != native or (graph_stamp, keys) != native:
            raise ValueError("native graph and event clocks differ")
        candidate_count += len(candidates)
        active = [index for index in active if stamp <= gaps[index]["stop"]]
        active.extend(starts.pop(stamp, ()))
        points = {key: plane.project(point) for key, point in observations.items()}
        phase = shifts if stamp > FIT_STOP_S else {}
        costs = dict(candidates)
        for index in active:
            gap = gaps[index]
            if stamp == gap["stop"] and gap["end"] != "kf_rejoin":
                continue
            if not _legal(switches[index].edges, costs):
                raise ValueError("comparison moved outside original candidate graph")
            row = _readout(switches[index], stamp, points, priors, phase, bank)
            if row is not None:
                output[index].append(row)
        bank.advance(stamp, points, priors, costs, selected, phase)
        _update_priors(priors, stamp, points)
    original = pickle.load(graph)
    if (starts or candidate_count != result.diagnostics["fixed_candidate_edge_events"]
            or original.native_reports != result.native_reports or original.runs["kf"] != result.runs["kf"]
            or original.fit != result.fit or graph.read(1)):
        raise ValueError("original frozen graph differs from the calibration replay")
    return output, dict(bank.counts)


def _public_sample(row, labels, pairs):
    target = labels[row["reporter"]]
    true_sides = [name for name, partner in row["partners"].items() if labels[partner] == target]
    side = true_sides[0] if len(true_sides) == 1 else None
    pair = tuple(sorted((row["reporter"], row["partners"][side]))) if side is not None else None
    group_labels = {labels[key] for key in row["keys"]}
    anchors = {}
    for tier, record in row["anchors"].items():
        samples = [sample for sample, _ in record["provenance"]]
        identities = [(labels[sample.edge[0]], labels[sample.edge[1]]) for sample in samples]
        correct = [a for a, b in identities if a == b]
        anchors[tier] = {name: record[name] for name in ("n", "recent_n", "scores")} | record["fit"] | dict(
            wrong=len(identities)-len(correct), overlap=sum(bool(set(ab) & group_labels) for ab in identities),
            duplicate=len(correct)-len(set(correct)))
    public = {name: row[name] for name in ("stamp", "cross", "pred_delta", "self", "self_variance",
                                          "age", "variance", "scores", "shift_delta")}
    source_pair = sorted({key[0] for key in row["keys"]})
    return public | dict(source=str(source_pair.index(row["reporter"][0])), true_side=side,
                         true_pair=pairs.setdefault(pair, len(pairs)) if pair is not None else None, anchors=anchors)


def _evaluate(payload, old_scene, labels):
    pairs = {}
    rows = []
    for captured, old in zip(payload, old_scene["rows"], strict=True):
        indexed = {}
        for sample, baseline in zip(captured, old["batches"], strict=True):
            if (sample["stamp"] != baseline["stamp"] or len(baseline["reports"]) != 1
                    or not _same(sample["scores"]["raw"], baseline["likelihood_actual"])
                    or not _same(sample["scores"]["masked"], baseline["likelihood_masked"])):
                raise ValueError("the original Gaussian and masked readouts no longer agree")
            indexed[sample["stamp"]] = sample
        early = indexed[old["latest"]["stamp"]] if old["latest"] is not None else None
        late = indexed[old["boundary"]["stamp"]] if old["boundary"] is not None else None
        base = {k: old[k] for k in ("start", "stop", "end", "late_support", "full_rejoin", "boundary_reports",
                                    "focal", "focal_true", "true_release", "true_retain", "target_edges")}
        base.update(switch_source=early["reporter"][0] != late["reporter"][0] if early and late else None,
                    same_identity=labels[early["reporter"]] == labels[late["reporter"]] if early and late else None)
        for stage in STAGES:
            base[stage] = _public_sample(indexed[old[stage]["stamp"]], labels, pairs) if old[stage] else None
        rows.append(base)
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, nargs=3, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--graph-run", type=Path, required=True)
    parser.add_argument("--gap-run", type=Path, required=True)
    parser.add_argument("--async-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    roster = _roster(args.manifest)
    scenes = roster["splits"]["development"]
    if scenes != read_roster(args.manifest)["splits"]["held"]:
        raise ValueError("the diagnostic must use the original 28 scenes")
    args.output_dir.mkdir(mode=0o700, exist_ok=False)
    frozen = args.output_dir / "private_frozen"
    frozen.mkdir(mode=0o700)
    began = perf_counter()
    entries = {}
    for index, scene in enumerate(scenes, 1):
        alias = f"scene_{index:02d}"
        name = f"{alias}.pkl.gz"
        result, gaps = _load_frozen(args.gap_run / "private_frozen" / name)
        with gzip.open(args.graph_run / "private_frozen" / name, "rb") as graph:
            payload, counts = _capture(_events(args.archive, args.manifest, scene), result, graph, gaps)
        with gzip.open(frozen / name, "xb", compresslevel=1) as out:
            pickle.dump(payload, out, protocol=pickle.HIGHEST_PROTOCOL)
        entries[alias] = dict(component=roster["components"][scene], counts=counts,
                              registration=bool(_registration_shifts(result.fit)),
                              sparse=2*result.diagnostics["first_peers_le3"] >= result.diagnostics["first_total"] > 0,
                              events=result.diagnostics["events"], reports=result.diagnostics["reports"],
                              candidate_edge_events=result.diagnostics["fixed_candidate_edge_events"])
        print(f"Frozen {index}/{len(scenes)}, {len(gaps)} groups", flush=True)

    print("All signed residuals, anchor identities and scores frozen. Reading evaluation labels.", flush=True)
    old = json.loads((args.async_run / "summary.json").read_text())["scenes"]
    for index, scene in enumerate(scenes, 1):
        alias = f"scene_{index:02d}"
        with gzip.open(frozen / f"{alias}.pkl.gz", "rb") as stream:
            payload = pickle.load(stream)
        entries[alias]["rows"] = _evaluate(payload, old[alias], read_labels(args.archive, args.manifest, scene))
    _write_json(args.output_dir / "summary.json", dict(scenes=entries, clock="ideal_native_measurement_time",
                publications_unchanged=True, candidate_graph_unchanged=True, training_updates=0,
                wall_seconds=perf_counter()-began))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
