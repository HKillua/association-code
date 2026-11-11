"""Post-replay lifetimes of wrong and missing cross-source links."""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import combinations
import math

from .continuous_metrics import SUPPORT_AGE_SECONDS
from .nemot_replay import ARMS


ERRORS = ("false", "missed")
COMPARISONS = (("kf", "history"), ("history", "neighbors"), ("kf", "neighbors"))
DIRECTIONS = ("added", "removed")
STARTS = ("first_grid_second", "joint_support_already_present", "first_observed_endpoint", "returning_support")
ENDS = ("support_lost", "coactive_state_change", "right_censored")
LENGTH_BINS = ("1-5", "6-60", "61+")


@dataclass(frozen=True)
class _Start:
    second: int
    reason: str


class _Episodes:
    def __init__(self):
        self.open = {}
        self.histogram = Counter()
        self.starts = {reason: Counter() for reason in STARTS}
        self.ends = {reason: Counter() for reason in ENDS}

    def _close(self, pair, next_second, reason):
        start = self.open.pop(pair)
        duration = next_second - start.second
        self.histogram[duration] += 1
        self.starts[start.reason].update(episodes=1, pair_seconds=duration)
        self.ends[reason].update(episodes=1, pair_seconds=duration)

    def update(self, second, current, active, previous_active, newly_seen):
        for left, right in self.open.keys() - current:
            end = "coactive_state_change" if left in active and right in active else "support_lost"
            self._close((left, right), second, end)
        for left, right in current - self.open.keys():
            if second == 0:
                start = "first_grid_second"
            elif left in previous_active and right in previous_active:
                start = "joint_support_already_present"
            elif left in newly_seen or right in newly_seen:
                start = "first_observed_endpoint"
            else:
                start = "returning_support"
            self.open[left, right] = _Start(second, start)

    def finish(self, end_second):
        for pair in tuple(self.open):
            self._close(pair, end_second, "right_censored")
        return _episode_summary(self.histogram, self.starts, self.ends)


def _tally(count=0, seconds=0):
    return dict(episodes=count, pair_seconds=seconds)


def _histogram_summary(histogram):
    lengths = sorted(histogram)
    total = sum(histogram.values())
    result = dict(_tally(total, sum(length * histogram[length] for length in lengths)),
                  observed_length_histogram=dict(sorted(histogram.items())),
                  observed_length_seconds=dict(p50=None, p95=None, max=max(lengths, default=None)))
    if total:
        cumulative = 0
        levels = (("p50", math.ceil(.50 * total)), ("p95", math.ceil(.95 * total)))
        for length in lengths:
            cumulative += histogram[length]
            for name, rank in levels:
                if cumulative >= rank and result["observed_length_seconds"][name] is None:
                    result["observed_length_seconds"][name] = length
    return result


def _episode_summary(histogram, starts, ends):
    bins = {name: Counter() for name in LENGTH_BINS}
    for length, count in histogram.items():
        name = LENGTH_BINS[0 if length <= 5 else 1 if length <= 60 else 2]
        bins[name].update(episodes=count, pair_seconds=length * count)
    result = _histogram_summary(histogram)
    result["by_length"] = {name: _tally(counts["episodes"], counts["pair_seconds"]) for name, counts in bins.items()}
    result["by_start"] = {name: _tally(counts["episodes"], counts["pair_seconds"]) for name, counts in starts.items()}
    result["by_end"] = {name: _tally(counts["episodes"], counts["pair_seconds"]) for name, counts in ends.items()}
    return result


def _cross_pairs(members):
    return ((left, right) for left, right in combinations(sorted(members), 2) if left[0] != right[0])


def _truth(active, labels):
    by_identity = defaultdict(list)
    for key in active:
        by_identity[labels[key]].append(key)
    return {pair for group in by_identity.values() for pair in _cross_pairs(group)}


def _published(groups, active):
    return {pair for _, members in groups
            for pair in _cross_pairs({key for key in members if key in active})}


def _comparison_name(before, after):
    return f"{after}_minus_{before}"


def _compare(counts, previous, new):
    added = new - previous
    removed = previous - new
    counts.update(added_pair_seconds=len(added), removed_pair_seconds=len(removed),
                  common_pair_seconds=len(previous) - len(removed), net_pair_seconds=len(added) - len(removed))
    return added, removed


def _arm_summary(truth_pairs, episodes):
    missed = episodes["missed"]["pair_seconds"]
    return dict(pairs=dict(truth_pairs=truth_pairs, correct_pairs=truth_pairs - missed,
                           missed_pairs=missed, false_pairs=episodes["false"]["pair_seconds"]),
                **episodes)


def score_episodes(result, labels, *, stop=None):
    """Count maximal contiguous runs of false/missed unordered fragment pairs.

    The grid is 0..ceil(last report) unless `stop` is supplied. A final open
    run contributes its observed length and is right-censored. Loss of
    joint report support is distinct from a change while both endpoints
    remain fresh. This scorer does not inspect future observations or
    infer assignment causes from the error lifetimes.
    """
    reports = iter(result.native_reports)
    pending = next(reports, None)
    if stop is None:
        stop = math.ceil(result.native_reports[-1][0]) if pending is not None else 0
    if type(stop) is not int or stop < 0:
        raise ValueError("episode scoring requires a nonnegative integer stop")

    changes = {arm: iter(result.runs[arm]) for arm in ARMS}
    next_change = {arm: next(changes[arm], None) for arm in ARMS}
    groups = {arm: () for arm in ARMS}
    trackers = {arm: {error: _Episodes() for error in ERRORS} for arm in ARMS}
    comparisons = {_comparison_name(before, after): {error: Counter() for error in ERRORS}
                   for before, after in COMPARISONS}
    differences = {name: {error: {direction: _Episodes() for direction in DIRECTIONS} for error in ERRORS}
                   for name in comparisons}
    latest, seen, previous_active = {}, set(), set()
    truth = set()
    truth_pairs = 0
    for second in range(stop + 1):
        newly_seen = set()
        while pending is not None and pending[0] <= second:
            stamp, keys = pending
            newly_seen.update(set(keys) - seen)
            seen.update(keys)
            latest.update(dict.fromkeys(keys, stamp))
            pending = next(reports, None)
        latest = {key: stamp for key, stamp in latest.items() if second - stamp <= SUPPORT_AGE_SECONDS}
        active = set(latest)
        if active != previous_active:
            truth = _truth(active, labels)
        truth_pairs += len(truth)

        errors_by_arm = {}
        for arm in ARMS:
            change = next_change[arm]
            while change is not None and change[0] <= second:
                groups[arm] = change[1]
                change = next(changes[arm], None)
            next_change[arm] = change
            selected = _published(groups[arm], active)
            errors = {"false": selected - truth, "missed": truth - selected}
            errors_by_arm[arm] = errors
            for error, pairs in errors.items():
                trackers[arm][error].update(second, pairs, active, previous_active, newly_seen)
        for before, after in COMPARISONS:
            name = _comparison_name(before, after)
            for error in ERRORS:
                changed = _compare(comparisons[name][error], errors_by_arm[before][error], errors_by_arm[after][error])
                for direction, pairs in zip(DIRECTIONS, changed, strict=True):
                    differences[name][error][direction].update(second, pairs, active, previous_active, newly_seen)
        previous_active = active

    arms = {arm: _arm_summary(truth_pairs, {error: trackers[arm][error].finish(stop + 1) for error in ERRORS})
            for arm in ARMS}
    paired = {}
    for name, errors in comparisons.items():
        paired[name] = {}
        for error, counts in errors.items():
            paired[name][error] = dict(counts) | {
                f"{direction}_episodes": differences[name][error][direction].finish(stop + 1)
                for direction in DIRECTIONS}
    return dict(ticks=stop + 1, truth_pairs=truth_pairs, arms=arms, comparisons=paired)


def _pool_lengths(summaries):
    histogram = Counter()
    starts = {reason: Counter() for reason in STARTS}
    ends = {reason: Counter() for reason in ENDS}
    for summary in summaries:
        histogram.update({int(length): count for length, count in summary["observed_length_histogram"].items()})
        for reason, counts in summary["by_start"].items():
            starts[reason].update(counts)
        for reason, counts in summary["by_end"].items():
            ends[reason].update(counts)
    return _episode_summary(histogram, starts, ends)


def pool_episodes(results):
    """Combine scene-level counts, deriving quantiles from pooled run lengths."""
    results = list(results)
    truth_pairs = sum(result["truth_pairs"] for result in results)
    arms = {}
    for arm in ARMS:
        summaries = {error: _pool_lengths(result["arms"][arm][error] for result in results) for error in ERRORS}
        arms[arm] = _arm_summary(truth_pairs, summaries)

    comparisons = {}
    for before, after in COMPARISONS:
        name = _comparison_name(before, after)
        comparisons[name] = {}
        for error in ERRORS:
            tally = Counter()
            fields = [result["comparisons"][name][error] for result in results]
            for field in fields:
                tally.update({key: count for key, count in field.items() if key.endswith("_pair_seconds")})
            comparisons[name][error] = dict(tally) | {
                f"{direction}_episodes": _pool_lengths(field[f"{direction}_episodes"] for field in fields)
                for direction in DIRECTIONS}
    return dict(scenes=len(results), ticks=sum(result["ticks"] for result in results),
                truth_pairs=truth_pairs, arms=arms, comparisons=comparisons)
