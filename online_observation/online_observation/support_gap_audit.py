"""Audit four-fragment support timing on the frozen online disagreement graph."""
from __future__ import annotations

from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict
from dataclasses import dataclass, field
import gzip
import pickle

from .continuous_tracker import FRESH_S
from .hypothesis_gap_diagnostic import Switch, WINDOW_SECONDS
from .interval_assignment import MIN_SPAN_SECONDS, PAIR_NULL_COST
from .nemot_episodes import _published


ZONES = ("within", "boundary", "beyond")
MILESTONES = ("new_report", "common_bracket", "comparable_bracket",
              "full_refresh", "refreshed_bracket", "current_legal_bracket",
              "original_assignment_bracket", "legal_refreshed_bracket",
              "original_refreshed_bracket", "after20_refreshed_bracket")


def _timing(keys, histories, stamp):
    times = [histories[key] for key in keys]
    if any(not track for track in times):
        raise ValueError("a frozen switch endpoint has never reported")
    latest = tuple(track[-1] for track in times)
    prior = tuple((track[-2] if len(track) > 1 else None) if last == stamp else last
                  for track, last in zip(times, latest, strict=True))
    right = min(latest)
    left = right - MIN_SPAN_SECONDS
    too_few = sum(len(track) < 2 for track in times)
    starts_late = sum(track[0] > left for track in times)
    long_gaps = 0
    new_support = 0
    for track in times:
        if len(track) < 2 or track[0] > left:
            continue
        first = bisect_right(track, left) - 1
        last = bisect_left(track, right)
        long_gaps += any(track[index + 1] - track[index] > FRESH_S
                         for index in range(first, last))
        new_support += track[last] == stamp
    bracket = not (too_few or starts_late or long_gaps)
    return dict(common_right=right, latest=latest, prior=prior, group_reports=latest.count(stamp),
                fewer_than_two_reports=too_few, starts_after_left=starts_late,
                excessive_bracket_gaps=long_gaps, bracket=bracket,
                new_support_reports=new_support if bracket else 0)


@dataclass
class _Audit:
    row: dict
    switch: Switch
    keys: tuple
    followup_stop: float
    first: dict = field(default_factory=lambda: {
        zone: dict.fromkeys(MILESTONES) for zone in ZONES
    })
    events: dict = field(default_factory=lambda: {zone: Counter() for zone in ZONES})
    onset: dict | None = None
    boundary: dict | None = None
    first_expiry: float | None = None
    inferred: tuple[float, str] | None = None

    def _infer_close(self, stamp, expiry, state):
        if self.inferred is not None or stamp == self.row["start"]:
            return
        deadline = self.row["start"] + WINDOW_SECONDS
        if min(expiry, deadline) < stamp:
            self.inferred = (expiry, "expiry") if expiry < deadline else (deadline, "window")
        elif not state["legal"]:
            self.inferred = stamp, "graph_loss"
        elif state["kf_rejoin"]:
            self.inferred = stamp, "kf_rejoin"
        elif state["i22_withdrawal"]:
            self.inferred = stamp, "interval_withdrawal"
        elif not state["original_assignment"]:
            self.inferred = stamp, "other_assignment"

    def observe(self, stamp, histories, fresh, costs, kf, i22):
        start, stop = self.row["start"], self.row["stop"]
        timing = _timing(self.keys, histories, stamp)
        latest, prior = timing.pop("latest"), timing.pop("prior")
        prior_fresh = all(value is not None and 0 < stamp - value <= FRESH_S for value in prior)
        expiry = min(value + FRESH_S for value in prior) if all(value is not None for value in prior) else None
        if stamp > start and expiry is None:
            raise ValueError("a frozen switch lost its pre-batch report history")
        if self.first_expiry is None and stamp > start and expiry < min(stamp, self.followup_stop):
            self.first_expiry = expiry

        currently_fresh = all(key in fresh for key in self.keys)
        if currently_fresh != all(stamp - value <= FRESH_S for value in latest):
            raise ValueError("frozen freshness differs from native endpoint reports")
        missing = sum(edge not in costs for edge in self.switch.edges)
        above_null = sum(edge in costs and costs[edge] >= PAIR_NULL_COST for edge in self.switch.edges)
        legal = not missing and all(costs[edge] < PAIR_NULL_COST for edge in self.switch.edges)
        kf_rejoin = any(edge in kf for edge in self.row["focal"])
        i22_withdrawal = any(edge not in i22 for edge in self.row["focal"])
        assignment = all(edge in kf for edge in self.switch.release) and all(
            edge in i22 for edge in self.switch.retain)
        state = dict(stamp=stamp, delay=stamp-start, prior_fresh=prior_fresh, fresh=currently_fresh,
                     continuous_fresh=self.first_expiry is None, missing_edges=missing,
                     at_or_above_null=above_null, legal=legal, kf_rejoin=kf_rejoin,
                     i22_withdrawal=i22_withdrawal, original_assignment=assignment, **timing)
        self._infer_close(stamp, expiry, state)
        if stamp == stop:
            self.boundary = state
        if stamp == start:
            if not (legal and assignment and currently_fresh and not kf_rejoin and not i22_withdrawal):
                raise ValueError("the original gap onset no longer contains the frozen disagreement")
            self.onset = state
            return
        if stamp > self.followup_stop:
            return

        zone = "within" if stamp < stop else "boundary" if stamp == stop else "beyond"
        common = bool(state["new_support_reports"])
        refreshed = state["common_right"] > start
        comparable = common and prior_fresh
        legal_bracket = comparable and currently_fresh and legal
        legal_refreshed = legal_bracket and refreshed
        milestones = dict(new_report=bool(state["group_reports"]), common_bracket=common,
                          comparable_bracket=comparable, full_refresh=refreshed,
                          refreshed_bracket=common and refreshed, current_legal_bracket=legal_bracket,
                          original_assignment_bracket=legal_bracket and assignment,
                          legal_refreshed_bracket=legal_refreshed,
                          original_refreshed_bracket=legal_refreshed and assignment and state["continuous_fresh"],
                          after20_refreshed_bracket=common and state["common_right"] >= start + MIN_SPAN_SECONDS)
        self.events[zone].update(events=1, fresh=int(currently_fresh), prior_fresh=int(prior_fresh),
                                 continuous_fresh=int(state["continuous_fresh"]), legal=int(legal),
                                 original_assignment=int(assignment), bracket=int(state["bracket"]),
                                 short_history=int(bool(state["fewer_than_two_reports"])),
                                 shallow_overlap=int(bool(state["starts_after_left"])),
                                 bracket_gap=int(bool(state["excessive_bracket_gaps"])),
                                 old_support_only=int(state["bracket"] and not common),
                                 **{name: int(passed) for name, passed in milestones.items()})
        for name, passed in milestones.items():
            if passed and self.first[zone][name] is None:
                self.first[zone][name] = state

    def finish(self, scene_stop):
        start, stop, end = (self.row[key] for key in ("start", "stop", "end"))
        inferred = self.inferred if self.inferred is not None else (scene_stop, "scene_end")
        if self.onset is None or inferred != (stop, end):
            raise ValueError("inferred termination differs from the frozen C-ON37 gap")
        return dict(start=start, stop=stop, end=end, followup_stop=self.followup_stop,
                    termination=dict(reason=end, time=stop, native_event=self.boundary,
                                     censored=end == "scene_end"),
                    first_expiry=self.first_expiry, onset=self.onset,
                    events={zone: dict(counts) for zone, counts in self.events.items()},
                    first=self.first)


def audit_support(result, gaps, graph_path):
    """Return timing and eligibility audits aligned with the frozen input gaps.

    The exact C-ON38 timing control is the first within-gap comparable bracket.
    Boundary and beyond-gap snapshots are retrospective timing opportunities.
    They do not extend the original four-edge disagreement.
    """
    if not result.native_reports:
        raise ValueError("the frozen scene has no native events")
    scene_stop = result.native_reports[-1][0]
    audits = []
    starts = defaultdict(list)
    for row in gaps:
        start, stop = row["start"], row["stop"]
        followup_stop = min(start + WINDOW_SECONDS, scene_stop)
        if not start <= stop <= followup_stop:
            raise ValueError("frozen gap falls outside its original time horizon")
        switch = Switch(**row["switch"])
        audit = _Audit(row, switch, tuple(sorted(switch.keys)), followup_stop)
        audits.append(audit)
        starts[start].append(audit)

    histories = {key: [] for audit in audits for key in audit.keys}
    interval_changes = iter(result.runs["interval_swap"])
    next_interval = next(interval_changes, None)
    interval = ()
    active = []
    candidate_events = 0
    report_count = 0
    with gzip.open(graph_path, "rb") as stream:
        for stamp, keys in result.native_reports:
            time, reports, _, fresh_keys, costs, selected = pickle.load(stream)
            if (time, reports) != (stamp, keys):
                raise ValueError("original candidate and report clocks differ")
            if next_interval is not None and next_interval[0] == stamp:
                interval = next_interval[1]
                next_interval = next(interval_changes, None)
            for key in keys:
                if key in histories:
                    histories[key].append(stamp)
            candidate_events += len(costs)
            report_count += len(keys)
            active.extend(starts.pop(stamp, ()))
            if active:
                fresh = set(fresh_keys)
                costs, kf = dict(costs), set(selected)
                i22 = _published(interval, fresh)
                for audit in active:
                    audit.observe(stamp, histories, fresh, costs, kf, i22)
                active = [audit for audit in active if stamp <= audit.followup_stop]
        original = pickle.load(stream)
        if (starts or next_interval is not None or original.native_reports != result.native_reports
                or original.runs["kf"] != result.runs["kf"] or original.fit != result.fit
                or candidate_events != result.diagnostics["fixed_candidate_edge_events"]
                or report_count != result.diagnostics["reports"] or stream.read(1)):
            raise ValueError("the original graph differs from the frozen interval replay")
    return [audit.finish(scene_stop) for audit in audits]
