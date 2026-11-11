"""Post-replay scoring for causal cross-source geometry recovery."""
from __future__ import annotations

from collections import Counter, defaultdict
from itertools import combinations
from math import ceil
from statistics import median


ARMS = ("geometry", "wide", "anchor", "motion")
DEADLINES = (0, 20, 60, 120, 240)
HORIZON = DEADLINES[-1]
COHORTS = ("first", "outside")
QUERY_GROUPS = ("all", "first", "continuation", "fresh_0_3", "first_fresh_0_3")
QUERY_COUNTS = (
    "count", "fresh_true", "geometry_reachable", "outside", "outside_true_edge_queries", "beyond_ring_only",
    "ring", "open", "ring_true_edge_queries", "open_true_edge_queries", "motion_supported", "motion_qualified",
    "motion_unique", "anchor_fitted", "anchor_in_gate", "single_in_gate", "single_in_gate_with_true_witness",
    "anchor_in_gate_with_all_true_witnesses", "geometry_correct", "geometry_wrong", "geometry_unmatched",
    "motion_supported_in_ring", "motion_qualified_in_ring", "anchor_fitted_in_ring",
    "anchor_in_gate_in_ring", "single_in_gate_in_ring",
)
ARM_COUNTS = ("true_edges", "false_edges", "added_true", "added_false", "events_with_added_true",
              "events_with_added_false", "added_with_false_witness", "distinct_added_true", "distinct_added_false",
              "distinct_true", "distinct_false")


def _cross_source_pairs(keys):
    for left, right in combinations(sorted(keys), 2):
        if left[0] != right[0]:
            yield left, right


def _true_pairs(fresh, labels):
    by_identity = defaultdict(list)
    for key in fresh:
        by_identity[labels[key]].append(key)
    return {pair for keys in by_identity.values() for pair in _cross_source_pairs(keys)}


def _incident(pairs):
    edges = defaultdict(set)
    for left, right in pairs:
        pair = (left, right)
        edges[left].add(pair)
        edges[right].add(pair)
    return edges


def _endpoints(pairs):
    return {key for pair in pairs for key in pair}


def _evidence_counts(evidence, truth):
    counts = Counter(ring=bool(evidence), ring_true_edge_queries=len(evidence), beyond_ring_only=not evidence)
    counts.update(motion_supported_in_ring=any(edge.motion_supported for edge in evidence),
                  motion_qualified_in_ring=any(edge.motion_status == "qualified" for edge in evidence),
                  anchor_fitted_in_ring=any(edge.anchor_status == "fitted" for edge in evidence),
                  anchor_in_gate_in_ring=any(edge.anchor_in_gate for edge in evidence),
                  single_in_gate_in_ring=any(edge.single_in_gate for edge in evidence))
    opened = [edge for edge in evidence if edge.open]
    counts.update(open=bool(opened), open_true_edge_queries=len(opened),
                  motion_supported=any(edge.motion_supported for edge in opened),
                  motion_qualified=any(edge.motion_status == "qualified" for edge in opened),
                  motion_unique=any(edge.motion_unique for edge in opened),
                  anchor_fitted=any(edge.anchor_status == "fitted" for edge in opened),
                  anchor_in_gate=any(edge.anchor_in_gate for edge in opened),
                  single_in_gate=any(edge.single_in_gate for edge in opened),
                  single_in_gate_with_true_witness=any(edge.single_in_gate and len(edge.witnesses) == 1
                                                       and edge.witnesses[0] in truth for edge in opened),
                  anchor_in_gate_with_all_true_witnesses=any(edge.anchor_in_gate and edge.witnesses
                                                             and all(witness in truth for witness in edge.witnesses)
                                                             for edge in opened))
    # With multiple genuine opposite-source fragments, a query can enter multiple status bins.
    counts.update({f"motion_status_{status}": 1 for status in {edge.motion_status for edge in opened}})
    counts.update({f"anchor_status_{status}": 1 for status in {edge.anchor_status for edge in opened}})
    return counts


def _quantiles(samples):
    ordered = sorted(samples)
    if not ordered:
        return dict(p50=None, p95=None, max=None)
    return dict(p50=median(ordered), p95=ordered[ceil(.95 * len(ordered)) - 1], max=ordered[-1])


def _delay_summary(population, truncated, delays, paired):
    arms = {}
    for arm, part in delays.items():
        samples = sorted(part["samples_seconds"])
        arms[arm] = dict(confirmed=len(samples), unconfirmed=population - len(samples),
                         unconfirmed_with_truncated_followup=part["unconfirmed_with_truncated_followup"],
                         by_seconds={str(t): sum(value <= t for value in samples) for t in DEADLINES},
                         delay_seconds=_quantiles(samples), samples_seconds=samples)
    output = dict(population=population, followup_truncated=truncated, arms=arms)
    for reference, groups in paired.items():
        comparisons = {}
        for arm, part in groups.items():
            differences = sorted(part["difference_samples_seconds"])
            only_arm = sorted(part["only_arm_samples_seconds"])
            reference_only = part["only_reference_confirmed"]
            comparisons[arm] = dict(
                both_confirmed=len(differences), only_arm_confirmed=len(only_arm),
                neither_confirmed=population - len(differences) - len(only_arm) - reference_only,
                earlier=sum(value < 0 for value in differences), same=sum(value == 0 for value in differences),
                later=sum(value > 0 for value in differences), difference_seconds=_quantiles(differences),
                difference_samples_seconds=differences, only_arm_delay_seconds=_quantiles(only_arm),
                only_arm_samples_seconds=only_arm,
            )
            comparisons[arm][f"only_{reference}_confirmed"] = reference_only
        output[f"versus_{reference}"] = comparisons
    return output


def _grade_delay(starts, confirmed, scene_end, references):
    truncated = {key for key, stamp in starts.items() if scene_end - stamp < HORIZON}
    delays = {arm: dict(samples_seconds=list(values.values()),
                        unconfirmed_with_truncated_followup=len(truncated - values.keys()))
              for arm, values in confirmed.items()}
    paired = {}
    for reference in references:
        baseline = confirmed[reference]
        paired[reference] = {
            arm: dict(difference_samples_seconds=[values[key] - baseline[key] for key in baseline.keys() & values],
                      only_arm_samples_seconds=[values[key] for key in values.keys() - baseline],
                      only_reference_confirmed=len(baseline.keys() - values.keys()))
            for arm, values in confirmed.items() if arm != reference
        }
    return _delay_summary(len(starts), len(truncated), delays, paired)


def score_recovery(frames, labels, *, arms=ARMS, references=("geometry",), cohorts=COHORTS):
    """Grade frozen decisions and return scene counts without fragment identities."""
    augmented = tuple(arm for arm in arms if arm != "geometry")
    query_counts = (*QUERY_COUNTS, *(f"{arm}_{metric}" for arm in augmented
                                    for metric in ("recovered", "added_true", "added_false")))
    queries = {group: Counter(dict.fromkeys(query_counts, 0)) for group in QUERY_GROUPS}
    arm_counts = {arm: Counter(dict.fromkeys(ARM_COUNTS, 0)) for arm in arms}
    distinct = {arm: {name: set() for name in ("true", "false", "added_true", "added_false")} for arm in arms}
    starts = {name: {} for name in cohorts}
    confirmed = {name: {arm: {} for arm in arms} for name in cohorts}
    outside_distinct = set()
    event_count = 0
    scene_end = 0.
    for frame in frames:
        stamp = frame.stamp
        scene_end = stamp
        event_count += 1
        truth = _true_pairs(frame.fresh, labels)
        truth_by_key = _incident(truth)
        candidates = set(frame.geometry_edges)
        base = set(frame.geometry_links)
        base_true, base_false = base & truth, base - truth
        base_true_keys, base_false_keys = _endpoints(base_true), _endpoints(base_false)
        evidence = {item.pair: item for item in frame.evidence}
        selected_true_keys = {}
        added_true_keys, added_false_keys = {}, {}
        for arm in arms:
            added = set(frame.additions.get(arm, ()))
            true, false = added & truth, added - truth
            selected_true_keys[arm] = base_true_keys | _endpoints(true)
            added_true_keys[arm] = _endpoints(true)
            added_false_keys[arm] = _endpoints(false)
            counts = arm_counts[arm]
            counts.update(true_edges=len(base_true) + len(true), false_edges=len(base_false) + len(false),
                          added_true=len(true), added_false=len(false),
                          events_with_added_true=bool(true), events_with_added_false=bool(false))
            if arm == "anchor":
                counts["added_with_false_witness"] += sum(any(witness not in truth for witness in evidence[pair].witnesses)
                                                          for pair in added)
            for name, pairs in (("true", base_true | true), ("false", base_false | false),
                                ("added_true", true), ("added_false", false)):
                distinct[arm][name].update(pairs)

        born = set(frame.born)
        by_source = Counter(key[0] for key in frame.fresh)
        for key in born:
            if "first" in starts:
                starts["first"].setdefault(key, stamp)
            if "first_sparse" in starts and len(frame.fresh) - by_source[key[0]] <= 3:
                starts["first_sparse"].setdefault(key, stamp)
        for key in frame.updated:
            true_peers = truth_by_key[key]
            reachable = not candidates.isdisjoint(true_peers)
            outside = bool(true_peers) and not reachable
            sparse = len(frame.fresh) - by_source[key[0]] <= 3
            counts = Counter(count=1, fresh_true=bool(true_peers), geometry_reachable=reachable,
                             geometry_correct=key in base_true_keys, geometry_wrong=key in base_false_keys,
                             geometry_unmatched=key not in base_true_keys and key not in base_false_keys)
            if outside:
                if "outside" in starts:
                    starts["outside"].setdefault(key, stamp)
                if "outside_sparse" in starts and sparse:
                    starts["outside_sparse"].setdefault(key, stamp)
                outside_distinct.update(true_peers)
                counts.update(outside=1, outside_true_edge_queries=len(true_peers))
                counts.update(_evidence_counts([evidence[pair] for pair in true_peers if pair in evidence], truth))
            for arm in augmented:
                counts.update({f"{arm}_recovered": outside and key in added_true_keys[arm],
                               f"{arm}_added_true": key in added_true_keys[arm],
                               f"{arm}_added_false": key in added_false_keys[arm]})
            groups = ["all", "first" if key in born else "continuation"]
            if sparse:
                groups.append("fresh_0_3")
                if key in born:
                    groups.append("first_fresh_0_3")
            for group in groups:
                queries[group].update(counts)

        for cohort in cohorts:
            initial = starts[cohort]
            for arm, keys in selected_true_keys.items():
                outcome = confirmed[cohort][arm]
                for key in keys & initial.keys() - outcome.keys():
                    elapsed = stamp - initial[key]
                    if elapsed <= HORIZON:
                        outcome[key] = elapsed

    for arm, kinds in distinct.items():
        arm_counts[arm].update({f"distinct_{kind}": len(pairs) for kind, pairs in kinds.items()})
    return dict(events=event_count, distinct_outside_true=len(outside_distinct),
                queries={name: dict(value) for name, value in queries.items()},
                arms={arm: dict(value) for arm, value in arm_counts.items()},
                delays={name: _grade_delay(starts[name], confirmed[name], scene_end, references) for name in cohorts})


def _sum_counts(parts):
    counts = Counter()
    for part in parts:
        counts.update(part)
    return dict(counts)


def pool_recovery(results, *, arms=ARMS, references=("geometry",), cohorts=COHORTS):
    """Pool additive counts and recompute all empirical delay summaries."""
    scenes = list(results)
    delays = {}
    for cohort in cohorts:
        parts = [scene["delays"][cohort] for scene in scenes]
        arm_delays = {
            arm: dict(samples_seconds=[sample for part in parts for sample in part["arms"][arm]["samples_seconds"]],
                      unconfirmed_with_truncated_followup=sum(part["arms"][arm]["unconfirmed_with_truncated_followup"]
                                                              for part in parts)) for arm in arms
        }
        paired = {}
        for reference in references:
            groups = {}
            for arm in arms:
                if arm == reference:
                    continue
                comparisons = [part[f"versus_{reference}"][arm] for part in parts]
                groups[arm] = dict(difference_samples_seconds=[value for row in comparisons
                                                               for value in row["difference_samples_seconds"]],
                                   only_arm_samples_seconds=[value for row in comparisons
                                                             for value in row["only_arm_samples_seconds"]],
                                   only_reference_confirmed=sum(row[f"only_{reference}_confirmed"] for row in comparisons))
            paired[reference] = groups
        delays[cohort] = _delay_summary(sum(part["population"] for part in parts),
                                        sum(part["followup_truncated"] for part in parts), arm_delays, paired)
    return dict(events=sum(scene["events"] for scene in scenes),
                distinct_outside_true=sum(scene["distinct_outside_true"] for scene in scenes),
                queries={group: _sum_counts(scene["queries"][group] for scene in scenes) for group in QUERY_GROUPS},
                arms={arm: _sum_counts(scene["arms"][arm] for scene in scenes) for arm in arms}, delays=delays)
