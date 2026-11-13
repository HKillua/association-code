"""Summarize fixed MTAD rejoins with cache-disjoint displacement evidence."""
from __future__ import annotations

from collections import Counter

from .async_report_squeeze import _primary_control
from .disjoint_report_closure_squeeze import _metrics
from .hypothesis_gap_diagnostic import _direction
from .rejoin_witness_squeeze import ARMS as WITNESS_ARMS
from .rejoin_witness_squeeze import _direction_label, _truth_side, _truthward, _witness_flags


FRESH_ARMS = ("fresh_rank", "fresh_attributed", "fresh_guard")
ARMS = (*WITNESS_ARMS, "unfiltered_rank", *FRESH_ARMS)
READOUTS = ("unfiltered", "decached", "post_onset_masked")
SIDES = ("release", "retain")


def _outcomes(target, correct, arm):
    return _metrics(sum(entry["decisions"].get(arm) is True for entry in target),
                    sum(entry["decisions"].get(arm) is True for entry in correct),
                    len(target), len(correct)) | dict(
        target_supported=sum(entry["decisions"].get(arm) is not None for entry in target),
        correct_supported=sum(entry["decisions"].get(arm) is not None for entry in correct),
    )


def _action_changes(rows, arm):
    counts = Counter((entry["decisions"].get("current"), entry["decisions"].get(arm)) for entry in rows)
    return dict(added=counts[False, True], removed=counts[True, False],
                inherited=counts[True, True], stayed_clear=counts[False, False],
                new_support=sum(counts[None, decision] for decision in (True, False)),
                lost_support=sum(counts[decision, None] for decision in (True, False)),
                added_vetoes=counts[False, True] + counts[None, True],
                removed_vetoes=counts[True, False] + counts[True, None],
                neither_supported=counts[None, None])


def _compare_actions(target, correct):
    outcomes = {arm: _outcomes(target, correct, arm) for arm in ARMS}
    return dict(target_windows=len(target), correct_windows=len(correct), arms=outcomes,
                changes_vs_current={arm: dict(target=_action_changes(target, arm),
                                              correct=_action_changes(correct, arm),
                                              net_delta=outcomes[arm]["net"]-outcomes["current"]["net"])
                                    for arm in FRESH_ARMS})


def _common_support(rows, require_nested_mask=False):
    return [entry for entry in rows if entry.get("fresh") is not None
            and entry["fresh"]["decached"]["qualified"]
            and entry["decisions"].get("current") is not None
            and entry["decisions"].get("geometry_masked") is not None
            and (not require_nested_mask or entry["fresh"]["post_onset_masked"]["margin_m2"] is not None)]


def _compare(target, correct):
    return dict(full=_compare_actions(target, correct),
                same_support=_compare_actions(_common_support(target), _common_support(correct)),
                same_attribution_support=_compare_actions(_common_support(target, require_nested_mask=True),
                                                          _common_support(correct, require_nested_mask=True)))


def _sign(value):
    return "unavailable" if value is None else _direction_label(value)


def _old_margin(witness):
    _, interval, _, qualified = _witness_flags(witness)
    return interval["displacement_margin_m2"] if qualified and interval is not None else None


def _old_post_onset(rows):
    return [entry for entry in rows if entry["features"]["witness"]["post_onset_opposite_reports"] > 0]


def _readout_counts(rows, arm):
    statuses, roles, directions, qualified_directions = Counter(), Counter(), Counter(), Counter()
    correct = wrong = ties = qualified_correct = qualified_wrong = qualified_ties = 0
    post_onset_reports = 0
    for entry in rows:
        result = entry["fresh"][arm]
        side = _truth_side(entry)
        direction = _sign(result["margin_m2"])
        statuses[result["status"]] += 1
        roles["+".join(result["post_onset_roles"]) or "none"] += 1
        post_onset_reports += result["post_onset_count"]
        directions[direction] += 1
        correct += side is not None and direction == side
        wrong += side is not None and direction in SIDES and direction != side
        ties += side is not None and direction == "tie"
        if result["qualified"]:
            qualified_directions[direction] += 1
            qualified_correct += side is not None and direction == side
            qualified_wrong += side is not None and direction in SIDES and direction != side
            qualified_ties += side is not None and direction == "tie"
    return dict(statuses=dict(statuses), post_onset_roles=dict(roles), post_onset_reports=post_onset_reports,
                supported=len(rows)-directions["unavailable"], directions=dict(directions), correct=correct,
                wrong=wrong, ties=ties, qualified=sum(qualified_directions.values()),
                qualified_directions=dict(qualified_directions), qualified_correct=qualified_correct,
                qualified_wrong=qualified_wrong, qualified_ties=qualified_ties)


def _candidate_opportunities(rows):
    counts = {side: Counter() for side in SIDES}
    for entry in rows:
        for side, values in entry["fresh"]["opportunities"].items():
            counts[side].update({key: value for key, value in values.items() if value is not None})
            counts[side]["windows_with_post_onset_outside_cache"] += (values["post_onset_outside_cache"] or 0) > 0
    return {side: dict(totals) for side, totals in counts.items()}


def _direction_transitions(rows, first, second):
    counts = Counter(f"{_sign(first(entry))}->{_sign(second(entry))}" for entry in rows)
    return dict(counts)


def _cohort(rows):
    decached = [entry for entry in rows if entry["fresh"]["decached"]["qualified"]]
    paired = [entry for entry in decached if entry["fresh"]["post_onset_masked"]["margin_m2"] is not None]
    attribution_status, attribution_side, counts = Counter(), Counter(), Counter()
    for entry in rows:
        truth = _truth_side(entry)
        old = entry["features"]["witness"]
        fresh = entry["fresh"]
        attribution = entry["attribution"]
        attribution_status[attribution["result"]] += 1
        attributed = [side for side in SIDES if attribution[side]]
        if len(attributed) > 1:
            raise ValueError("one displacement cannot attribute exclusion in both directions")
        exclusion = attributed[0] if attributed else None
        attribution_side[exclusion or "none"] += 1
        mismatch_change = attribution.get("mismatch_change_m2")
        if mismatch_change is not None:
            direction = _sign(fresh["decached"]["margin_m2"])
            previous = _sign(fresh["post_onset_masked"]["margin_m2"])
            counts.update(
                paired_direction_changed=direction != previous,
                paired_both_mismatches_grew=all(_direction(mismatch_change[side]) > 0 for side in SIDES),
                paired_reordered_without_exclusion=direction in SIDES and direction != previous and exclusion is None,
            )
        counts.update(
            old_post_onset=old["post_onset_opposite_reports"] > 0,
            old_supported=_old_margin(old) is not None,
            old_motion_correct=truth is not None and _sign(_old_margin(old)) == truth,
            historical_interval_right_after_onset=fresh["right"] is not None and fresh["right"] > entry["row"]["start"],
            attributed_correct=truth is not None and exclusion == truth,
            attributed_wrong=truth is not None and exclusion is not None and exclusion != truth,
            geometry_mask_correct=truth is not None and _sign(entry["features"]["geometry_masked_m2"]) == truth,
        )

    def readout(arm):
        return lambda entry: entry["fresh"][arm]["margin_m2"]

    def support_controls(entries):
        return dict(windows=len(entries), geometry_mask_correct=sum(
            _truth_side(entry) == _sign(entry["features"]["geometry_masked_m2"]) for entry in entries),
            current_vetoes=sum(entry["decisions"]["current"] is True for entry in entries),
            current_scored=sum(entry["decisions"]["current"] is not None for entry in entries))

    return dict(
        windows=len(rows), truth_sides=dict(Counter(_truth_side(entry) or "mixed" for entry in rows)),
        scenes=dict(Counter(entry["scene"] for entry in rows)),
        components=dict(Counter(entry["component"] for entry in rows)), counts=dict(counts),
        cache_status=dict(Counter(entry["cache_status"] for entry in rows)),
        opportunities=_candidate_opportunities(rows), readouts={arm: _readout_counts(rows, arm) for arm in READOUTS},
        on_decached_support=support_controls(decached), on_paired_support=support_controls(paired),
        directions=dict(
            c49_to_unfiltered=_direction_transitions(rows, lambda entry: _old_margin(entry["features"]["witness"]),
                                                     readout("unfiltered")),
            unfiltered_to_decached=_direction_transitions(rows, readout("unfiltered"), readout("decached")),
            post_onset_masked_to_decached=_direction_transitions(paired, readout("post_onset_masked"),
                                                                 readout("decached")),
        ),
        attribution=dict(statuses=dict(attribution_status), sides=dict(attribution_side)),
        actions={arm: dict(vetoes=sum(entry["decisions"].get(arm) is True for entry in rows),
                           available=sum(entry["decisions"].get(arm) is not None for entry in rows)) for arm in ARMS},
    )


def _case(entry):
    row, features = entry["row"], entry["features"]
    witness = features["witness"]
    return dict(scene=entry["scene"], component=entry["component"], sparse=entry["sparse"],
                start=row["start"], stop=row["stop"], truth_side=_truth_side(entry),
                cache_status=entry["cache_status"], fresh=entry["fresh"], attribution=entry["attribution"],
                old_displacement_margin_m2=_old_margin(witness),
                old_post_onset_opposite_reports=witness["post_onset_opposite_reports"],
                old_opposite_cache_overlap=witness["opposite_overlap_cache"],
                geometry_m2=features["geometry_m2"], geometry_masked_m2=features["geometry_masked_m2"],
                increment_m2=features["increment_m2"], decisions=entry["decisions"])


def summarize(rows):
    """Evaluate frozen actions on the original sparse and pure-truth rejoin populations."""
    eligible = [entry for entry in rows if entry.get("fresh") is not None]
    sparse = [entry for entry in rows if entry["sparse"]]
    target = [entry for entry in sparse if entry["row"]["target_edges"] > 0]
    correct = [entry for entry in sparse if _primary_control(entry["row"])]
    release = [entry for entry in eligible if _truth_side(entry) == "release"]
    retain = [entry for entry in eligible if _truth_side(entry) == "retain"]
    mixed = [entry for entry in eligible if _truth_side(entry) is None]
    farther, closer = _truthward(release + retain)
    far_post, near_post, correct_post = map(_old_post_onset, (farther, closer, retain))
    cache_hits = [entry for entry in release if entry["features"]["cache_only"]]
    cache_errors = [entry for entry in retain if entry["features"]["cache_only"]]
    components = sorted({entry["component"] for entry in target + correct + release + retain})

    def component_subset(entries, component):
        return [entry for entry in entries if entry["component"] == component]

    return dict(
        frozen_windows=len(rows), eligible_windows=len(eligible), primary=_compare(target, correct),
        broad=_compare(release, retain),
        broad_sparse=_compare([entry for entry in release if entry["sparse"]],
                              [entry for entry in retain if entry["sparse"]]),
        by_component={component: dict(
            primary=_compare(component_subset(target, component), component_subset(correct, component)),
            broad=_compare(component_subset(release, component), component_subset(retain, component)),
        ) for component in components},
        mechanism=dict(
            all=_cohort(eligible), mixed=_cohort(mixed), truthward_farther=_cohort(farther),
            farther_post_onset=_cohort(far_post) | {"cases": list(map(_case, far_post))},
            truthward_closer=_cohort(closer), closer_post_onset=_cohort(near_post),
            correct_rejoin=_cohort(retain), correct_post_onset=_cohort(correct_post),
            current_false_vetoes=_cohort([entry for entry in retain if entry["decisions"]["current"] is True]),
            cache_only=dict(target=_cohort(cache_hits) | {"cases": list(map(_case, cache_hits))},
                            correct=_cohort(cache_errors) | {"cases": list(map(_case, cache_errors))}),
        ),
    )
