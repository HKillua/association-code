"""Summarize frozen cache additions and paired MTAD rejoin comparisons."""

from __future__ import annotations

from collections import Counter

from .async_report_squeeze import _primary_control
from .disjoint_report_closure_squeeze import _metrics


ARMS = (
    "original", "geometry", "geometry_masked", "current", "cache", "union",
    "age", "persistence", "geometry_agreement", "qualified",
)
CONTROLS = ("current", "cache", "union")
RANK_DIRECTIONS = {
    "cache_age_s": -1,
    "support_age_s": -1,
    "cache_end_age_s": -1,
    "geometry_m2": 1,
    "geometry_masked_m2": 1,
    "geometry_increment_m2": 1,
    "absolute_geometry_increment_m2": -1,
    "kf_release_s": 1,
    "i22_retain_s": 1,
    "release_duration_advantage_s": 1,
    "release_at_cache_s": 1,
}
FLAGS = ("cache_end_within_release", "recent", "geometry_stable", "continuous_from_cache_end")


def _metrics_for(target, correct, arm):
    def count(population):
        return sum(entry["decisions"][arm] is True for entry in population)

    return _metrics(count(target), count(correct), len(target), len(correct))


def _changes(rows, reference):
    counts = Counter((entry["decisions"][reference], entry["decisions"]["qualified"]) for entry in rows)
    return dict(added=counts[False, True], removed=counts[True, False],
                inherited=counts[True, True], stayed_clear=counts[False, False])


def _compare(target, correct):
    full = {}
    for arm in ARMS:
        full[arm] = _metrics_for(target, correct, arm) | {
            "target_supported": sum(entry["decisions"][arm] is not None for entry in target),
            "correct_supported": sum(entry["decisions"][arm] is not None for entry in correct),
        }

    def paired_rows(population, control):
        return [entry for entry in population
                if all(entry["decisions"][arm] is not None for arm in ("qualified", control))]

    paired = {}
    for control in CONTROLS:
        targets = paired_rows(target, control)
        controls = paired_rows(correct, control)
        candidate = _metrics_for(targets, controls, "qualified")
        reference = _metrics_for(targets, controls, control)
        paired[control] = dict(
            candidate=candidate, control=reference, net_delta=candidate["net"] - reference["net"],
            target_changes=_changes(targets, control), correct_changes=_changes(controls, control),
        )
    return dict(target_windows=len(target), correct_windows=len(correct), full=full, paired=paired)


def _observed_value(features, metric):
    if metric == "absolute_geometry_increment_m2":
        increment = features.get("geometry_increment_m2")
        return abs(increment) if increment is not None else None
    if metric == "release_duration_advantage_s":
        release, retain = features.get("kf_release_s"), features.get("i22_retain_s")
        return release - retain if release is not None and retain is not None else None
    return features.get(metric)


def _ranks(target, correct):
    ranks = {}
    for metric, direction in RANK_DIRECTIONS.items():
        counts = Counter()
        for positive in target:
            gain = _observed_value(positive["features"], metric)
            for negative in correct:
                veto = _observed_value(negative["features"], metric)
                if gain is None or veto is None:
                    counts["missing"] += 1
                else:
                    signed = direction * (gain - veto)
                    counts["favors_hit" if signed > 0 else "favors_veto" if signed < 0 else "tied"] += 1
        ranks[metric] = dict(direction="higher" if direction > 0 else "lower", pairs=len(target) * len(correct),
                             **{name: counts[name] for name in ("favors_hit", "favors_veto", "tied", "missing")})
    return ranks


def _cache_only(target, correct):
    def select(population):
        return [entry for entry in population if entry["decisions"]["cache"] is True
                and entry["decisions"]["current"] is False]

    def cases_summary(cases):
        by_component = Counter(entry["component"] for entry in cases)
        known = [(entry["scene"], str(identifier)) for entry in cases
                 if (identifier := entry["features"].get("cache_id")) is not None]
        reuse = Counter(
            str(entry["features"].get("cache_support_reuse", entry["features"].get("cache_shape_reused")))
            for entry in cases
        )
        return dict(
            windows=len(cases), qualified=sum(entry["decisions"]["qualified"] is True for entry in cases),
            components=dict(sorted(by_component.items())), known_cache_ids=len(known),
            distinct_cache_ids=len(set(known)), support_reuse=dict(reuse),
            flags={flag: dict(Counter(str(entry["features"].get(flag)) for entry in cases)) for flag in FLAGS},
        )

    targets, controls = select(target), select(correct)
    def missing_current(rows):
        return sum(entry["decisions"]["cache"] is True and entry["decisions"]["current"] is None for entry in rows)
    return dict(
        target=cases_summary(targets), correct=cases_summary(controls),
        cache_veto_with_current_missing=dict(target=missing_current(target), correct=missing_current(correct)),
        ranks=_ranks(targets, controls),
    ), targets, controls


def _case(entry):
    row = entry["row"]
    return dict(scene=entry["scene"], component=entry["component"], start=row["start"], stop=row["stop"],
                switch_source=row["switch_source"], features=entry["features"], decisions=entry["decisions"])


def _in_components(rows, components):
    return [entry for entry in rows if entry["component"] in components]


def _switched(rows):
    return [entry for entry in rows if entry["row"]["switch_source"] is True]


def _broad(rows):
    release, retain, mixed = [], [], []
    for entry in rows:
        if entry["decisions"]["original"] is None:
            continue
        row = entry["row"]
        truth = row["true_release"], row["true_retain"]
        destination = {(2, 0): release, (0, 2): retain}.get(truth, mixed)
        destination.append(entry)
    cache_only, _, _ = _cache_only(release, retain)
    return dict(
        eligible_windows=len(release) + len(retain) + len(mixed),
        pure=_compare(release, retain), cache_only=cache_only,
        mixed=dict(
            windows=len(mixed), cache_only=sum(entry["decisions"]["cache"] is True
                                               and entry["decisions"]["current"] is False for entry in mixed),
            arms={arm: dict(vetoes=sum(entry["decisions"][arm] is True for entry in mixed),
                            supported=sum(entry["decisions"][arm] is not None for entry in mixed)) for arm in ARMS},
        ),
    )


def summarize(rows):
    """Aggregate precomputed decisions after joining the frozen truth cohorts."""
    sparse = [entry for entry in rows if entry["sparse"]]
    target = [entry for entry in sparse if entry["row"]["target_edges"] > 0]
    correct = [entry for entry in sparse if _primary_control(entry["row"])]
    components = Counter(entry["component"] for entry in target)
    largest = min(components, key=lambda key: (-components[key], key)) if components else None
    same = _in_components(correct, components)
    remaining = set(components) - {largest}
    cache_only, cases_target, cases_correct = _cache_only(target, correct)
    cache_only["cases"] = dict(
        target=[_case(entry) for entry in cases_target], correct=[_case(entry) for entry in cases_correct],
    )
    return dict(
        primary=_compare(target, correct), cache_only=cache_only,
        same_components=_compare(target, same),
        source_switch=_compare(_switched(target), _switched(correct)),
        same_components_switch=_compare(_switched(target), _switched(same)),
        by_component={
            key: _compare(_in_components(target, {key}), _in_components(correct, {key}))
            for key in sorted(components)
        },
        without_largest_target_component=dict(
            component=largest,
            comparison=_compare(_in_components(target, remaining), _in_components(correct, remaining)),
        ),
        broad=_broad(rows), broad_sparse=_broad(sparse),
    )
