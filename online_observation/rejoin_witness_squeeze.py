"""Summarize causal opponent-history witnesses on fixed MTAD rejoin windows."""
from __future__ import annotations

from collections import Counter

from .async_report_squeeze import _primary_control
from .disjoint_report_closure_squeeze import _metrics
from .hypothesis_gap_diagnostic import _direction


ARMS = (
    "original", "current", "cache", "union", "geometry", "geometry_masked",
    "raw_positive", "nominal_gaussian", "witness_position", "witness_motion", "witness_exclusion",
)
WITNESS_ARMS = ("witness_position", "witness_motion", "witness_exclusion")
CONTROLS = ("current", "cache", "union", "geometry_masked", "raw_positive", "nominal_gaussian",
            "witness_position", "witness_motion")


def _truth_side(entry):
    row = entry["row"]
    return {(2, 0): "release", (0, 2): "retain"}.get((row["true_release"], row["true_retain"]))


def _truthward(rows):
    farther, closer = [], []
    for entry in rows:
        side = _truth_side(entry)
        features = entry["features"]
        if side is None or features is None:
            continue
        direction = 1 if side == "release" else -1
        if direction * _direction(features["increment_m2"]) <= 0:
            continue
        change = _direction(features[f"{side}_change_m2"])
        if change:
            (farther if change > 0 else closer).append(entry)
    return farther, closer


def _direction_label(value):
    return {1: "release", 0: "tie", -1: "retain"}[_direction(value)]


def _witness_flags(witness):
    endpoint, interval = witness["endpoint"], witness["interval"]
    both = endpoint is not None and endpoint["both_partners_after_query"]
    qualified_endpoint = both and witness["fresh"]
    return endpoint, interval, both, qualified_endpoint


def _witness_counts(rows):
    counts, endpoint_status, interval_status = Counter(), Counter(), Counter()
    directions = {name: Counter() for name in ("position", "motion", "exclusion")}
    truth_components, false_components = Counter(), Counter()
    for entry in rows:
        features = entry["features"]
        if features is None:
            counts["unavailable_original"] += 1
            continue
        witness = features["witness"]
        endpoint, interval, both, qualified_endpoint = _witness_flags(witness)
        endpoint_status[witness["endpoint_status"]] += 1
        interval_status[witness["interval_status"]] += 1
        counts.update(
            scored=1, endpoint_available=endpoint is not None, interval_available=interval is not None,
            endpoint_with_one_late_partner=endpoint is not None and endpoint["late_partner_count"] == 1,
            endpoint_with_both_late_partners=both, qualified_endpoint=qualified_endpoint,
            qualified_interval=qualified_endpoint and interval is not None,
            post_onset_opposite_report=witness["post_onset_opposite_reports"] > 0,
            opposite_cache_overlap=witness["opposite_overlap_cache"] is not None
            and witness["opposite_overlap_cache"] > 0,
            opposite_cache_overlap_unavailable=witness["opposite_overlap_cache"] is None,
        )
        side = _truth_side(entry)
        if side is not None:
            truth_sign = 1 if side == "release" else -1
            counts["geometry_mask_truthward"] += truth_sign * _direction(features["geometry_masked_m2"]) > 0
            counts["geometry_actual_truthward"] += truth_sign * _direction(features["geometry_m2"]) > 0
        if qualified_endpoint:
            directions["position"][_direction_label(endpoint["margin_m2"])] += 1
            if side is not None:
                counts["mask_favors_truth_on_endpoint_support"] += (
                    truth_sign * _direction(features["geometry_masked_m2"]) > 0)
        if qualified_endpoint and interval is not None:
            directions["motion"][_direction_label(interval["displacement_margin_m2"])] += 1
            if side is not None:
                counts["mask_favors_truth_on_interval_support"] += (
                    truth_sign * _direction(features["geometry_masked_m2"]) > 0)
        for branch in ("release", "retain"):
            if witness[f"{branch}_exclusion"]:
                if not qualified_endpoint or interval is None:
                    raise ValueError("exclusion requires two fresh opposite brackets and an interval")
                directions["exclusion"][branch] += 1
                counts["geometry_mask_agrees_exclusion"] += _direction_label(features["geometry_masked_m2"]) == branch
        if side is not None:
            competitor = "retain" if side == "release" else "release"
            counts["position_favors_truth"] += qualified_endpoint and _direction_label(endpoint["margin_m2"]) == side
            counts["motion_favors_truth"] += (qualified_endpoint and interval is not None
                                             and _direction_label(interval["displacement_margin_m2"]) == side)
            counts["exclusion_truthward"] += witness[f"{side}_exclusion"]
            counts["excludes_true"] += witness[f"{competitor}_exclusion"]
            if witness[f"{side}_exclusion"]:
                truth_components[entry["component"]] += 1
            if witness[f"{competitor}_exclusion"]:
                false_components[entry["component"]] += 1
    return dict(
        windows=len(rows), components=dict(Counter(entry["component"] for entry in rows)), counts=dict(counts),
        endpoint_status=dict(endpoint_status), interval_status=dict(interval_status),
        directions={name: dict(values) for name, values in directions.items()},
        exclusion_truthward_components=dict(truth_components), excludes_true_components=dict(false_components),
    )


def _subsets(rows):
    return dict(combined=_witness_counts(rows), **{
        side: _witness_counts([entry for entry in rows if _truth_side(entry) == side])
        for side in ("release", "retain")
    })


def _available(rows, *arms):
    return [entry for entry in rows if all(entry["decisions"][arm] is not None for arm in arms)]


def _metrics_for(target, correct, arm):
    hits = sum(entry["decisions"][arm] is True for entry in target)
    false = sum(entry["decisions"][arm] is True for entry in correct)
    return _metrics(hits, false, len(target), len(correct))


def _changes(rows, arm, reference):
    changes = Counter((entry["decisions"][reference], entry["decisions"][arm]) for entry in rows)
    return dict(added=changes[False, True], removed=changes[True, False],
                inherited=changes[True, True], stayed_clear=changes[False, False])


def _on_support(entry, arm):
    features = entry["features"]
    if features is None:
        return False
    _, interval, _, qualified_endpoint = _witness_flags(features["witness"])
    return qualified_endpoint and (arm == "witness_position" or interval is not None)


def _paired(target, correct, arm, reference):
    targets = _available(target, arm, reference)
    controls = _available(correct, arm, reference)
    candidate = _metrics_for(targets, controls, arm)
    comparator = _metrics_for(targets, controls, reference)
    return dict(candidate=candidate, control=comparator, net_delta=candidate["net"] - comparator["net"],
                target_changes=_changes(targets, arm, reference), correct_changes=_changes(controls, arm, reference))


def _compare(target, correct):
    full = {arm: _metrics_for(target, correct, arm) | dict(
        target_supported=len(_available(target, arm)), correct_supported=len(_available(correct, arm))) for arm in ARMS}
    paired, on_witness_support = {}, {}
    for arm in WITNESS_ARMS:
        supported_target = [entry for entry in target if _on_support(entry, arm)]
        supported_correct = [entry for entry in correct if _on_support(entry, arm)]
        controls = [reference for reference in CONTROLS if reference != arm]
        paired[arm] = {reference: _paired(target, correct, arm, reference) for reference in controls}
        on_witness_support[arm] = {reference: _paired(supported_target, supported_correct, arm, reference)
                                   for reference in controls}
    return dict(target_windows=len(target), correct_windows=len(correct), full=full,
                paired=paired, paired_on_witness_support=on_witness_support)


def _case(entry):
    row = entry["row"]
    return dict(scene=entry["scene"], component=entry["component"], start=row["start"], stop=row["stop"],
                truth_side=_truth_side(entry), switch_source=row["switch_source"], features=entry["features"],
                decisions=entry["decisions"])


def _cache_only(rows):
    return [entry for entry in rows if entry["features"] is not None and entry["features"]["cache_only"]]


def _cache_summary(target, correct):
    targets, controls = _cache_only(target), _cache_only(correct)
    return dict(target=_witness_counts(targets), correct=_witness_counts(controls),
                cases=dict(target=list(map(_case, targets)), correct=list(map(_case, controls))))


def _broad(rows):
    release, retain, mixed = [], [], []
    for entry in rows:
        if entry["features"] is None:
            continue
        {"release": release, "retain": retain, None: mixed}[_truth_side(entry)].append(entry)
    farther, closer = _truthward(release + retain)
    return dict(
        eligible_windows=len(release) + len(retain) + len(mixed),
        pure=_compare(release, retain), cache_only=_cache_summary(release, retain),
        truthward_farther=_subsets(farther), truthward_closer=_subsets(closer),
        correct_rejoin=_witness_counts(retain),
        correct_rejoin_false_vetoes={
            arm: _witness_counts([entry for entry in retain if entry["decisions"][arm] is True])
            for arm in ("current", "cache", "union", "nominal_gaussian", "geometry", "geometry_masked", *WITNESS_ARMS)
        },
        mixed=dict(
            windows=len(mixed), arms={arm: dict(
                vetoes=sum(entry["decisions"][arm] is True for entry in mixed),
                supported=len(_available(mixed, arm)),
            ) for arm in ARMS},
        ),
    )


def summarize(rows):
    """Join frozen causal witness features to the original anonymous evaluation."""
    sparse = [entry for entry in rows if entry["sparse"]]
    target = [entry for entry in sparse if entry["row"]["target_edges"] > 0]
    correct = [entry for entry in sparse if _primary_control(entry["row"])]
    components = Counter(entry["component"] for entry in target)
    largest = min(components, key=lambda key: (-components[key], key)) if components else None

    def only(rows, selected):
        return [entry for entry in rows if entry["component"] in selected]

    def switched(rows):
        return [entry for entry in rows if entry["row"]["switch_source"] is True]

    same = only(correct, components)
    remaining = set(components) - {largest}
    farther, closer = _truthward(target + correct)
    return dict(
        primary=_compare(target, correct),
        same_components=_compare(target, same),
        source_switch=_compare(switched(target), switched(correct)),
        same_components_switch=_compare(switched(target), switched(same)),
        by_component={component: _compare(only(target, {component}), only(correct, {component}))
                      for component in sorted(components)},
        without_largest_target_component=dict(
            component=largest, comparison=_compare(only(target, remaining), only(correct, remaining))),
        cache_only=_cache_summary(target, correct),
        primary_truthward_farther=_subsets(farther), primary_truthward_closer=_subsets(closer),
        broad=_broad(rows), broad_sparse=_broad(sparse),
    )
