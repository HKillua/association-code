"""Evaluate a prefix-fit-dependent birth gate on frozen Level1 held scenes."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from time import perf_counter

from .evaluation_data import read_labels
from .held_gate_probe import (
    OPPORTUNITIES, OUTCOMES, POOL_SIZES, Birth, _arm_summary, predict, score,
)
from .measurements import read_archive_rows
from .positions import event_batches
from .registration_probe import BIRTH_AGE_S, FIT_STOP_S, estimate_translation


ROSTER = {
    "largest_component": (
        "134", "468", "545", "555", "763", "1023", "1139", "1144", "1317", "1624", "1710",
        "1789", "1870", "2295", "2483", "2531", "2900", "2926", "2957", "2969", "3047", "3215",
        "3292", "3388", "3720", "3800", "4003", "4348", "4503", "4534", "4748", "4807",
    ),
    "second_component": (
        "103", "174", "206", "313", "442", "514", "656", "664", "672", "1116", "1201", "1368",
        "1679", "2122", "2320", "2456", "2459", "2480", "2598", "2824", "3159", "3205", "3349",
        "3557", "3599", "4381", "4389", "4612", "4649", "4709", "4939", "4954",
    ),
    "other_touched_components": (
        "149", "269", "393", "576", "799", "1000", "1244", "1377", "1479", "1805", "2041", "2053",
        "2157", "2189", "2204", "2698", "2830", "3045", "3148", "3200", "3208", "3246", "3279",
        "3631", "3967", "4130", "4239", "4296", "4742", "4871", "4884", "4980",
    ),
    "untouched_components": (
        "84", "253", "295", "326", "522", "610", "708", "724", "965", "1209", "1282", "1433",
        "1502", "1511", "1766", "1913", "1924", "2024", "2121", "2397", "2451", "2592", "2621",
        "2656", "2669", "2710", "3641", "3821", "3909", "4114", "4606", "4778",
    ),
}
FIXED_1 = "fixed_1km"
FIXED_4 = "fixed_4km"
CONDITIONAL = "prefix_conditional"
ARMS = (FIXED_1, FIXED_4, CONDITIONAL)


def freeze_decisions(events, fit):
    """Fix the same-scene policy gate and its candidate decisions without labels."""
    first_seen, births = predict(events, fit)
    selected_gate = 1000 if fit["status"] == "fitted" else 4000
    decisions = {}
    for key, birth in births.items():
        narrow = birth.choices["corrected|1000"]
        wide = birth.choices["corrected|4000"]
        conditional = narrow if selected_gate == 1000 else wide
        if ((narrow.winner is not None and narrow.winner != conditional.winner)
                or (conditional.winner is not None and conditional.winner != wide.winner)):
            raise RuntimeError("prefix rule changed a retained candidate key")
        decisions[key] = Birth(birth.fresh, dict(zip(ARMS, (narrow, wide, conditional), strict=True)),
                               birth.fitted_pairs)
    return first_seen, decisions, selected_gate


def _compare(rows, control):
    matrix = {outcome: dict.fromkeys(OUTCOMES, 0) for outcome in OUTCOMES}
    no_history_removed = no_history_restored = 0
    for row in rows:
        before = row.arms[control].decision
        after = row.arms[CONDITIONAL].decision
        if before != after and (after if control == FIXED_4 else before) != "reject":
            raise RuntimeError("prefix gate violated the nested acceptance populations")
        matrix[before][after] += 1
        if row.opportunity == "never_observed":
            no_history_removed += before == "wrong" and after == "reject"
            no_history_restored += before == "reject" and after == "wrong"
    return dict(
        matrix=matrix,
        delta={outcome: sum(matrix[other][outcome] - matrix[outcome][other] for other in OUTCOMES)
               for outcome in OUTCOMES} if rows else None,
        true_accepts_lost=matrix["correct"]["reject"],
        false_accepts_removed=matrix["wrong"]["reject"],
        true_accepts_restored=matrix["reject"]["correct"],
        false_accepts_restored=matrix["reject"]["wrong"],
        no_history_false_accepts_removed=no_history_removed,
        no_history_false_accepts_restored=no_history_restored,
    )


def _summarize(rows):
    counts = Counter(row.opportunity for row in rows)
    candidate_pairs = sum(row.candidate_pairs for row in rows)
    fitted_pairs = sum(row.fitted_pairs for row in rows)
    return dict(
        births=len(rows), opportunities={kind: counts[kind] for kind in OPPORTUNITIES},
        candidate_pairs=candidate_pairs, fitted_candidate_pairs=fitted_pairs,
        fallback_candidate_pairs=candidate_pairs - fitted_pairs,
        arms={arm: _arm_summary(rows, arm, counts["fresh"]) for arm in ARMS},
        comparisons={f"{CONDITIONAL}_minus_{control}": _compare(rows, control)
                     for control in (FIXED_1, FIXED_4)},
    )


def summarize(rows):
    by_source = defaultdict(list)
    by_pool = {name: [] for name in POOL_SIZES}
    for row in rows:
        by_source[row.source].append(row)
        index = 0 if row.candidate_pairs <= 3 else 1 if row.candidate_pairs <= 15 else 2
        by_pool[POOL_SIZES[index]].append(row)
    return dict(_summarize(rows),
                by_source={source: _summarize(members) for source, members in sorted(by_source.items())},
                by_pool_size={size: _summarize(members) for size, members in by_pool.items()})


def _summarize_groups(groups):
    return {group: dict(summarize([row for _, rows in members for row in rows]),
                        scene_ids=[scene for scene, _ in members],
                        scenes_with_births=sum(bool(rows) for _, rows in members))
            for group, members in groups.items()}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", nargs=3, type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(args.output)
    began = perf_counter()
    scene_strata = {scene: group for group, members in ROSTER.items() for scene in members}
    if not scene_strata or len(scene_strata) != sum(map(len, ROSTER.values())):
        raise ValueError("invalid frozen scene roster")
    components = json.loads(args.manifest.read_text(encoding="utf-8"))["components"]

    prepared = {}
    for scene in scene_strata:
        events = list(event_batches(read_archive_rows(args.archive, args.manifest, scene, required_split="T_core")))
        fit = estimate_translation(events)
        first_seen, births, selected_gate = freeze_decisions(events, fit)
        prepared[scene] = fit, first_seen, births, selected_gate

    scenes, all_rows = {}, []
    fit_groups = {"fitted": [], "raw_fallback": []}
    roster_groups = {group: [] for group in ROSTER}
    for scene, (fit, first_seen, births, selected_gate) in prepared.items():
        rows = score(first_seen, births, read_labels(args.archive, args.manifest, scene))
        mode = "fitted" if selected_gate == 1000 else "raw_fallback"
        group = scene_strata[scene]
        scenes[scene] = dict(summarize(rows), fit=fit, correction_mode=mode,
                             conditional_gate_metres=selected_gate, roster_stratum=group,
                             component_id=components[scene])
        fit_groups[mode].append((scene, rows))
        roster_groups[group].append((scene, rows))
        all_rows.extend(rows)

    summary = dict(
        scene_ids=list(scene_strata), scenes=scenes, aggregate=summarize(all_rows),
        by_fit_status=_summarize_groups(fit_groups), by_roster_stratum=_summarize_groups(roster_groups),
        clock="ideal_measurement_time", fit_stop_seconds=FIT_STOP_S, birth_max_age_seconds=BIRTH_AGE_S,
        gate_metres={FIXED_1: 1000, FIXED_4: 4000, CONDITIONAL: {"fitted": 1000, "raw_fallback": 4000}},
        tie_policy="abstain on exact nearest ties",
        history_definition="opposite-source first report time at or before the birth",
        wall_seconds=perf_counter() - began,
    )
    with args.output.open("x", encoding="utf-8") as output:
        json.dump(summary, output, indent=2, allow_nan=False)
        output.write("\n")


if __name__ == "__main__":
    main()
