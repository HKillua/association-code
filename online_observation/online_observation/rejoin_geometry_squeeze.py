"""Summarize frozen rejoin geometry and truth-referenced residual changes."""
from __future__ import annotations

from collections import Counter

from .async_report_squeeze import _age_bucket, _primary_control
from .disjoint_report_closure_squeeze import _metrics
from .hypothesis_gap_diagnostic import _direction
from .relative_motion_gap import _same


ARMS = (
    "original", "current", "cache", "union", "raw_positive", "nominal_gaussian",
    "nominal_correction", "geometry", "geometry_masked",
)
CONTROLS = ("current", "cache", "union", "raw_positive", "nominal_gaussian", "geometry_masked")
RANK_FIELDS = ("increment_m2", "projected_m", "relative_increment", "reporter_z", "joint_z", "gaussian_increment")


def _metrics_for(target, correct, arm):
    def positive(rows):
        return sum(entry["decisions"][arm] is True for entry in rows)

    return _metrics(positive(target), positive(correct), len(target), len(correct))


def _available(rows, *arms):
    return [entry for entry in rows if all(entry["decisions"][arm] is not None for arm in arms)]


def _changes(rows, control):
    outcomes = Counter((entry["decisions"][control], entry["decisions"]["nominal_correction"]) for entry in rows)
    return dict(added=outcomes[False, True], removed=outcomes[True, False],
                inherited=outcomes[True, True], stayed_clear=outcomes[False, False])


def _compare(target, correct):
    full = {arm: _metrics_for(target, correct, arm) | dict(
        target_supported=len(_available(target, arm)), correct_supported=len(_available(correct, arm))) for arm in ARMS}
    paired = {}
    for arm in CONTROLS:
        same_target = _available(target, arm, "nominal_correction")
        same_correct = _available(correct, arm, "nominal_correction")
        candidate = _metrics_for(same_target, same_correct, "nominal_correction")
        control = _metrics_for(same_target, same_correct, arm)
        paired[arm] = dict(candidate=candidate, control=control, net_delta=candidate["net"] - control["net"],
                           target_changes=_changes(same_target, arm), correct_changes=_changes(same_correct, arm))
    return dict(target_windows=len(target), correct_windows=len(correct), full=full, paired=paired)


def _truth_side(entry):
    row = entry["row"]
    truth = row["true_release"], row["true_retain"]
    return {(2, 0): "release", (0, 2): "retain"}.get(truth)


def _truth_changes(rows):
    output = {}
    for side in ("release", "retain"):
        competitor = "retain" if side == "release" else "release"
        population = [entry for entry in rows if _truth_side(entry) == side]
        changes, reasons, flips = Counter(), Counter(), Counter()
        for entry in population:
            features = entry["features"]
            if features is None:
                continue
            true_gain = features[f"{side}_masked_m2"] - features[f"{side}_actual_m2"]
            false_gain = features[f"{competitor}_masked_m2"] - features[f"{competitor}_actual_m2"]
            margin_gain = true_gain - false_gain
            truth_direction = 1 if side == "release" else -1
            if not _same(margin_gain, truth_direction * features["increment_m2"]):
                raise ValueError("signed single-report residual changes disagree with geometry increment")
            geometry_before = truth_direction * _direction(features["geometry_masked_m2"])
            geometry_after = truth_direction * _direction(features["geometry_m2"])
            flips[f"{geometry_before}:{geometry_after}"] += 1
            true_direction, false_direction = _direction(true_gain), _direction(false_gain)
            gain_direction = truth_direction * _direction(features["increment_m2"])
            reasons[f"{true_direction}:{false_direction}"] += 1
            changes.update(
                scored=1, true_edge_improved=true_direction > 0, true_edge_worsened=true_direction < 0,
                competitor_improved=false_direction > 0, competitor_worsened=false_direction < 0,
                both_improved=true_direction > 0 and false_direction > 0,
                both_worsened=true_direction < 0 and false_direction < 0,
                truthward_increment=gain_direction > 0,
                true_correction=true_direction > 0 and gain_direction > 0,
                truthward_while_true_worsened=true_direction < 0 and gain_direction > 0,
                truthward_without_true_improvement=true_direction <= 0 and gain_direction > 0,
                falseward_while_true_improved=true_direction > 0 and gain_direction < 0,
                flips_toward_truth=geometry_before < 0 < geometry_after,
                flips_away_from_truth=geometry_after < 0 < geometry_before,
            )
        output[side] = dict(windows=len(population), **changes,
                            residual_directions=dict(reasons), geometry_directions=dict(flips))
    return output


def _timing(features):
    reporter = _age_bucket(features["reporter_age_s"])
    release = _age_bucket(features["release_age_s"])
    retain = _age_bucket(features["retain_age_s"])
    partner_max = _age_bucket(max(features["release_age_s"], features["retain_age_s"]))
    return dict(reporter=reporter, partner_max=partner_max, all_three=(reporter, release, retain))


def _ranks(pairs):
    output = {}
    for field in RANK_FIELDS:
        counter = Counter()
        for target, correct in pairs:
            hit, veto = target.get(field), correct.get(field)
            if hit is None or veto is None:
                counter["missing"] += 1
                continue
            direction = _direction(hit-veto) if field == "increment_m2" else (hit > veto) - (hit < veto)
            counter[{1: "better", 0: "tie", -1: "reversed"}[direction]] += 1
        output[field] = dict(pairs=len(pairs), **{name: counter[name] for name in ("better", "tie", "reversed", "missing")})
    return output


def _case(entry):
    row = entry["row"]
    return dict(scene=entry["scene"], component=entry["component"], start=row["start"], stop=row["stop"],
                switch_source=row["switch_source"], features=entry["features"], decisions=entry["decisions"])


def _cache_only(target, correct, *, cases=False):
    def selected(rows):
        return [entry for entry in rows if entry["decisions"]["cache"] is True and entry["decisions"]["current"] is False]

    targets, controls = selected(target), selected(correct)
    pairs = [(a["features"], b["features"]) for a in targets for b in controls]
    timing = {field: [] for field in ("reporter", "partner_max", "all_three")}
    for left, right in pairs:
        left_time, right_time = _timing(left), _timing(right)
        for field in timing:
            if left_time[field] == right_time[field]:
                timing[field].append((left, right))
    ranks = dict(
        target_windows=len(targets), correct_windows=len(controls), ranks=_ranks(pairs),
        matched_age={field: dict(pairs=len(matched), ranks=_ranks(matched)) for field, matched in timing.items()},
        target_components=dict(Counter(entry["component"] for entry in targets)),
        correct_components=dict(Counter(entry["component"] for entry in controls)),
        truth_changes=_truth_changes(targets + controls),
    )
    if cases:
        ranks["cases"] = dict(target=list(map(_case, targets)), correct=list(map(_case, controls)))
    return ranks


def _in_components(rows, components):
    return [entry for entry in rows if entry["component"] in components]


def _switched(rows):
    return [entry for entry in rows if entry["row"]["switch_source"] is True]


def _broad(rows):
    truth = {"release": [], "retain": [], None: []}
    for entry in rows:
        if entry["decisions"]["original"] is not None:
            truth[_truth_side(entry)].append(entry)
    release, retain, mixed = truth["release"], truth["retain"], truth[None]
    return dict(
        eligible_windows=sum(map(len, truth.values())),
        pure=_compare(release, retain), truth_changes=_truth_changes(release + retain),
        cache_only=_cache_only(release, retain, cases=True),
        mixed=dict(windows=len(mixed), arms={arm: dict(
            vetoes=sum(entry["decisions"][arm] is True for entry in mixed),
            supported=len(_available(mixed, arm))) for arm in ARMS}),
    )


def summarize(rows):
    """Join fixed-arm diagnostics to the original anonymous evaluation labels."""
    sparse = [entry for entry in rows if entry["sparse"]]
    target = [entry for entry in sparse if entry["row"]["target_edges"] > 0]
    correct = [entry for entry in sparse if _primary_control(entry["row"])]
    components = Counter(entry["component"] for entry in target)
    largest = min(components, key=lambda key: (-components[key], key)) if components else None
    same = _in_components(correct, components)
    remainder = set(components) - {largest}
    return dict(
        primary=_compare(target, correct), primary_truth_changes=_truth_changes(target + correct),
        cache_only=_cache_only(target, correct, cases=True), same_components=_compare(target, same),
        source_switch=_compare(_switched(target), _switched(correct)),
        same_components_switch=_compare(_switched(target), _switched(same)),
        by_component={component: _compare(_in_components(target, {component}), _in_components(correct, {component}))
                      for component in sorted(components)},
        without_largest_target_component=dict(component=largest, comparison=_compare(
            _in_components(target, remainder), _in_components(correct, remainder))),
        broad=_broad(rows), broad_sparse=_broad(sparse),
    )
