"""Post-replay membership and identity metrics on fixed report-supported seconds."""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import combinations, groupby
import math
from statistics import median


SUPPORT_AGE_SECONDS = 60
OPPORTUNITIES = ("fresh", "stale", "never_observed")
FIRST_DECISIONS = ("correct", "incorrect", "rejected", "unpublished")
PAIR_COUNTS = ("truth_pairs", "correct_pairs", "missed_pairs", "false_pairs")
TICK_COUNTS = (
    "ticks", "report_supported_fragment_seconds", "published_fragment_seconds",
    "report_supported_target_seconds", "represented_target_seconds",
    "pure_tracked_target_seconds", "uniquely_tracked_target_seconds",
    "targets_without_pure_output_seconds", "duplicate_target_seconds",
    "duplicate_pure_outputs", "output_track_seconds", "pure_output_seconds",
    "mixed_output_seconds", "unsupported_output_seconds",
    "stale_membership_entries", "duplicate_membership_entries", "repeated_member_seconds",
    "id_switches_consecutive", "id_switches_after_gap", "system_id_identity_transfers",
    "reacquired_same_id", "reacquired_changed_id", "reacquired_after_report_support_gap",
    "reacquired_while_report_supported",
)
SOURCE_COUNTS = ("report_supported_fragment_seconds", "published_fragment_seconds",
                 "report_supported_target_seconds", "pure_tracked_target_seconds")


@dataclass
class Birth:
    stamp: float
    opportunity: str
    first_decision: str | None = None
    first_correct_delay: float | None = None


def _key(value):
    if (not isinstance(value, (tuple, list)) or len(value) != 2
            or any(not isinstance(item, str) or not item for item in value)):
        raise ValueError("expected a nonempty (source, batch) key")
    return tuple(value)


def _event_batches(events):
    previous = -math.inf
    for stamp, group in groupby(events, key=lambda event: event[0]):
        if not math.isfinite(stamp) or stamp < previous:
            raise ValueError("events must have finite nondecreasing timestamps")
        previous = stamp
        yield stamp, {_key(key) for _, members in group for key in members}


def _publication_groups(publication, active, seen, counts):
    groups = {}
    for track in publication["tracks"]:
        track_id = track["track_id"]
        if not isinstance(track_id, str) or not track_id or track_id in groups:
            raise ValueError("publication track IDs must be nonempty and unique at each tick")
        members = [_key(key) for key in track["members"]]
        if any(key not in seen for key in members):
            raise ValueError("publication references a fragment before its first report")
        distinct = set(members)
        supported = distinct & active.keys()
        counts["duplicate_membership_entries"] += len(members) - len(distinct)
        counts["stale_membership_entries"] += len(distinct) - len(supported)
        groups[track_id] = supported
    return groups


def _pair_metrics(active, groups, labels, totals):
    by_target = defaultdict(Counter)
    for key in active:
        by_target[labels[key]][key[0]] += 1
    same_truth = sum(n * (n - 1) // 2 for sources in by_target.values() for n in sources.values())
    target_sizes = (sum(sources.values()) for sources in by_target.values())
    all_truth = sum(n * (n - 1) // 2 for n in target_sizes)
    totals["same_source"]["truth_pairs"] += same_truth
    totals["cross_source"]["truth_pairs"] += all_truth - same_truth

    correct_cross, wrong_cross = set(), set()
    correct = Counter()
    seen_pairs = set()
    for members in groups.values():
        for left, right in combinations(members, 2):
            pair = frozenset((left, right))
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            same_source = left[0] == right[0]
            scope = "same_source" if same_source else "cross_source"
            is_true = labels[left] == labels[right]
            correct[scope] += is_true
            totals[scope]["false_pairs"] += not is_true
            if not same_source:
                (correct_cross if is_true else wrong_cross).update((left, right))

    for scope, truth in (("same_source", same_truth), ("cross_source", all_truth - same_truth)):
        totals[scope]["correct_pairs"] += correct[scope]
        totals[scope]["missed_pairs"] += truth - correct[scope]
    return by_target, correct_cross, wrong_cross


def _grade_outputs(groups, labels, counts):
    memberships = Counter()
    pure_ids = defaultdict(list)
    for track_id, members in groups.items():
        memberships.update(members)
        identities = {labels[key] for key in members}
        if not identities:
            counts["unsupported_output_seconds"] += 1
        elif len(identities) == 1:
            pure_ids[next(iter(identities))].append(track_id)
            counts["pure_output_seconds"] += 1
        else:
            counts["mixed_output_seconds"] += 1
    counts["output_track_seconds"] += len(groups)
    counts["published_fragment_seconds"] += len(memberships)
    counts["repeated_member_seconds"] += sum(n > 1 for n in memberships.values())
    counts["represented_target_seconds"] += len({labels[key] for key in memberships})
    counts["pure_tracked_target_seconds"] += len(pure_ids)
    counts["uniquely_tracked_target_seconds"] += sum(len(ids) == 1 for ids in pure_ids.values())
    counts["duplicate_target_seconds"] += sum(len(ids) > 1 for ids in pure_ids.values())
    counts["duplicate_pure_outputs"] += sum(len(ids) - 1 for ids in pure_ids.values())
    return memberships.keys(), pure_ids


def _update_identity(tick, by_target, pure_ids, counts, last_match, last_support, support_gaps, last_owner):
    for identity in by_target:
        prior_support = last_support.get(identity)
        if prior_support is not None and prior_support < tick - 1:
            support_gaps.add(identity)
        last_support[identity] = tick
        ids = pure_ids.get(identity, ())
        if len(ids) != 1:
            continue
        track_id = ids[0]
        if identity in last_match:
            previous_id, previous_tick = last_match[identity]
            changed = previous_id != track_id
            if previous_tick == tick - 1:
                counts["id_switches_consecutive"] += changed
            else:
                counts["id_switches_after_gap"] += changed
                counts["reacquired_changed_id" if changed else "reacquired_same_id"] += 1
                counts["reacquired_after_report_support_gap" if identity in support_gaps
                       else "reacquired_while_report_supported"] += 1
        counts["system_id_identity_transfers"] += track_id in last_owner and last_owner[track_id] != identity
        last_match[identity] = track_id, tick
        last_owner[track_id] = identity
        support_gaps.discard(identity)


def _birth_decision(key, emitted, correct, wrong):
    if key in wrong:
        return "incorrect"
    if key in correct:
        return "correct"
    return "rejected" if key in emitted else "unpublished"


def _birth_summary(births):
    decisions = dict.fromkeys(FIRST_DECISIONS, 0)
    delays = []
    for birth in births:
        decisions[birth.first_decision] += 1
        if birth.first_correct_delay is not None:
            delays.append(birth.first_correct_delay)
    delays.sort()
    count = len(delays)
    return dict(
        births=len(births), first_publication=decisions, ever_correct=count,
        unconfirmed_by_end=len(births) - count,
        first_correct_delay_seconds=dict(
            n=count, sum=sum(delays), min=delays[0] if count else None,
            median=median(delays) if count else None,
            p95=delays[math.ceil(.95 * count) - 1] if count else None,
            max=delays[-1] if count else None,
            ceil_seconds_histogram=dict(sorted(Counter(map(math.ceil, delays)).items())),
        ),
    )


def evaluate_publications(events, publications, labels, *, stale_after_seconds=SUPPORT_AGE_SECONDS):
    """Grade consecutive 1 Hz membership snapshots on ideal measurement-time arrival.

    Events are ordered (timestamp, keys) batches, with mappings accepted for keys.
    Publications contain query_time and tracks with a track_id and members.
    The independent scoring population uses the input report ages, not tracker age.
    Label access belongs here, after all predictions have been frozen by the caller.
    """
    if not math.isfinite(stale_after_seconds) or stale_after_seconds <= 0:
        raise ValueError("report-support age must be finite and positive")
    counts = Counter(dict.fromkeys(TICK_COUNTS, 0))
    pair_totals = {kind: Counter(dict.fromkeys(PAIR_COUNTS, 0)) for kind in ("cross_source", "same_source")}
    by_source = defaultdict(lambda: Counter(dict.fromkeys(SOURCE_COUNTS, 0)))
    source_support = Counter()
    active, births = {}, {}
    seen = set()
    latest_by_identity_source = defaultdict(dict)
    last_match, last_support, last_owner = {}, {}, {}
    support_gaps = set()
    event_iterator = iter(_event_batches(events))
    pending = next(event_iterator, None)
    start = previous_tick = None

    for publication in publications:
        tick = publication["query_time"]
        if type(tick) is not int or (previous_tick is not None and tick != previous_tick + 1):
            raise ValueError("publications must include every consecutive integer query second")
        if start is None:
            start = tick
        previous_tick = tick
        new_births = set()
        while pending is not None and pending[0] <= tick:
            stamp, keys = pending
            if any(key not in labels for key in keys):
                raise ValueError("unmapped measurement fragment in post-replay labels")
            born = keys - seen
            seen.update(keys)
            active.update(dict.fromkeys(keys, stamp))
            for key in keys:
                latest_by_identity_source[labels[key]][key[0]] = stamp
            if stamp >= start:
                for key in born:
                    opposite = [stamp - latest for source, latest in latest_by_identity_source[labels[key]].items()
                                if source != key[0]]
                    opportunity = ("fresh" if any(age <= stale_after_seconds for age in opposite)
                                   else "stale" if opposite else "never_observed")
                    births[key] = Birth(stamp, opportunity)
                new_births.update(born)
            pending = next(event_iterator, None)
        active = {key: stamp for key, stamp in active.items() if tick - stamp <= stale_after_seconds}
        groups = _publication_groups(publication, active, seen, counts)
        by_target, true_cross, false_cross = _pair_metrics(active, groups, labels, pair_totals)
        emitted, pure_ids = _grade_outputs(groups, labels, counts)
        _update_identity(tick, by_target, pure_ids, counts, last_match, last_support, support_gaps, last_owner)

        counts["ticks"] += 1
        counts["report_supported_fragment_seconds"] += len(active)
        counts["report_supported_target_seconds"] += len(by_target)
        counts["targets_without_pure_output_seconds"] += len(by_target) - len(pure_ids)
        for identity, sources in by_target.items():
            source_support[len(sources)] += 1
            for source, fragments in sources.items():
                by_source[source]["report_supported_target_seconds"] += 1
                by_source[source]["report_supported_fragment_seconds"] += fragments
                by_source[source]["pure_tracked_target_seconds"] += identity in pure_ids
        for source, _ in emitted:
            by_source[source]["published_fragment_seconds"] += 1
        for key in new_births:
            births[key].first_decision = _birth_decision(key, emitted, true_cross, false_cross)
        for key in (true_cross - false_cross) & births.keys():
            birth = births[key]
            if birth.first_correct_delay is None:
                birth.first_correct_delay = tick - birth.stamp

    records = list(births.values())
    source_births = defaultdict(list)
    for (source, _), birth in births.items():
        source_births[source].append(birth)
    return dict(
        counts=dict(counts),
        pairs={scope: dict(totals) for scope, totals in pair_totals.items()},
        by_source={source: dict(totals) for source, totals in sorted(by_source.items())},
        target_seconds_by_active_source_count=dict(sorted(source_support.items())),
        births=dict(_birth_summary(records), by_opportunity={
            opportunity: _birth_summary([birth for birth in records if birth.opportunity == opportunity])
            for opportunity in OPPORTUNITIES
        }, by_source={source: _birth_summary(members) for source, members in sorted(source_births.items())}),
        protocol=dict(
            report_support_age_seconds=stale_after_seconds,
            arrival_clock="ideal_measurement_time", query_cadence_seconds=1,
            first_correct_link="published active true cross-source link without an active false cross-source link",
            identity_binding="unique pure published group for a report-supported target",
            delay_clock="query time minus native birth time, excludes service and queue",
        ),
    )
