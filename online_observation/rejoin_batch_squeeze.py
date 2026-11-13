"""Summarize paired same-batch vetoes on the frozen MTAD rejoin cohorts."""
from __future__ import annotations

from collections import Counter

from .async_report_squeeze import _primary_control
from .disjoint_report_closure_squeeze import _metrics


MAIN = "two_block_geometry"
ARMS = (
    "original", "geometry", "geometry_masked", "net_geometry", MAIN,
    "masked_shape_geometry", "cached_shape_geometry",
)


def _metrics_for(target, correct, arm):
    hit = sum(decisions.get(arm) is True for _, _, decisions in target)
    false = sum(decisions.get(arm) is True for _, _, decisions in correct)
    return _metrics(hit, false, len(target), len(correct))


def _paired_changes(rows, arm):
    changes = Counter((decisions[arm], decisions[MAIN]) for _, _, decisions in rows)
    return dict(added=changes[False, True], removed=changes[True, False],
                inherited=changes[True, True], stayed_clear=changes[False, False])


def _compare(target, correct, arms):
    def support(rows, *required):
        return [row for row in rows if all(row[2].get(arm) is not None for arm in required)]

    full, paired = {}, {}
    for arm in arms:
        full[arm] = _metrics_for(target, correct, arm) | dict(
            target_supported=len(support(target, arm)), correct_supported=len(support(correct, arm)))
        if arm == MAIN:
            continue
        target_common, correct_common = (support(population, MAIN, arm) for population in (target, correct))
        candidate = _metrics_for(target_common, correct_common, MAIN)
        control = _metrics_for(target_common, correct_common, arm)
        paired[arm] = dict(candidate=candidate, control=control,
                           net_delta=candidate["net"]-control["net"],
                           target_changes=_paired_changes(target_common, arm),
                           correct_changes=_paired_changes(correct_common, arm))
    return dict(target_windows=len(target), correct_windows=len(correct), full=full, paired=paired)


def _in_components(rows, components):
    return [entry for entry in rows if entry[0]["component"] in components]


def _switched(rows):
    return [entry for entry in rows if entry[1]["switch_source"] is True]


def _truth_direction(row):
    direction = row["true_release"], row["true_retain"]
    return {(2, 0): "release", (0, 2): "retain"}.get(direction, "mixed")


def _broad(rows, arms):
    truth = {name: [] for name in ("release", "retain", "mixed")}
    for entry in rows:
        if entry[2]["original"] is not None:
            truth[_truth_direction(entry[1])].append(entry)
    return dict(eligible_windows=sum(map(len, truth.values())),
                pure=_compare(truth["release"], truth["retain"], arms),
                mixed=dict(windows=len(truth["mixed"]), arms={
                    arm: dict(vetoes=sum(decisions.get(arm) is True for _, _, decisions in truth["mixed"]),
                              supported=sum(decisions.get(arm) is not None for _, _, decisions in truth["mixed"]))
                    for arm in arms
                }))


def summarize(rows):
    """Aggregate (scene, row, decisions) triples after causal decisions freeze."""
    rows = list(rows)
    extras = sorted({arm for _, _, decisions in rows for arm in decisions} - set(ARMS))
    arms = (*ARMS, *extras)
    sparse = [entry for entry in rows if entry[0]["sparse"]]
    target = [entry for entry in sparse if entry[1]["target_edges"] > 0]
    correct = [entry for entry in sparse if _primary_control(entry[1])]
    target_counts = Counter(scene["component"] for scene, _, _ in target)
    components = set(target_counts)
    largest = min(components, key=lambda key: (-target_counts[key], key)) if components else None
    same = _in_components(correct, components)
    remainder = components - {largest}
    return dict(
        primary=_compare(target, correct, arms),
        same_components=_compare(target, same, arms),
        source_switch=_compare(_switched(target), _switched(correct), arms),
        same_components_switch=_compare(_switched(target), _switched(same), arms),
        by_component={key: _compare(_in_components(target, {key}), _in_components(correct, {key}), arms)
                      for key in sorted(components)},
        without_largest_target_component=dict(component=largest,
            comparison=_compare(_in_components(target, remainder), _in_components(correct, remainder), arms)),
        broad=_broad(rows, arms),
        broad_sparse=_broad(sparse, arms),
    )
