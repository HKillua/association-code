"""Post-replay check for a shared relative translation on coincident reports."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
from statistics import median
from time import perf_counter

from .association_probe import SCENES
from .evaluation_data import read_labels
from .measurements import read_archive_rows
from .positions import event_batches, position_step


def coincident_displacements(rows, labels):
    signed = defaultdict(list)
    ambiguous = 0
    for _, positions in event_batches(rows):
        if len({key[0] for key in positions}) < 2:
            continue
        by_target = defaultdict(lambda: defaultdict(list))
        for key, position in positions.items():
            by_target[labels[key]][key[0]].append(position)
        for target, reports in by_target.items():
            if len(reports) < 2:
                continue
            if len(reports) != 2 or any(len(points) != 1 for points in reports.values()):
                ambiguous += 1
                continue
            left, right = (reports[source][0] for source in sorted(reports))
            distance, bearing = position_step(left, right)
            azimuth = math.radians(bearing)
            signed[target].append((distance * math.sin(azimuth), distance * math.cos(azimuth)))
    return signed, ambiguous


def summarize(by_target, ambiguous):
    target_ids = sorted(by_target)
    target_centres = {
        key: (median(x for x, _ in values), median(y for _, y in values))
        for key, values in by_target.items()
    }
    folds = []
    for offset in (0, 1):
        fit_ids, hold_ids = target_ids[offset::2], target_ids[1-offset::2]
        if not fit_ids or not hold_ids:
            continue
        east = median(target_centres[key][0] for key in fit_ids)
        north = median(target_centres[key][1] for key in fit_ids)
        rows = [value for key in hold_ids for value in by_target[key]]
        folds.append(dict(
            fitted_targets=len(fit_ids), held_targets=len(hold_ids), held_rows=len(rows),
            east_m=east, north_m=north,
            raw_distance_median_m=median(math.hypot(x, y) for x, y in rows),
            shifted_distance_median_m=median(math.hypot(x-east, y-north) for x, y in rows),
            raw_target_distance_median_m=median(math.hypot(*target_centres[key]) for key in hold_ids),
            shifted_target_distance_median_m=median(
                math.hypot(target_centres[key][0]-east, target_centres[key][1]-north)
                for key in hold_ids
            ),
        ))
    return dict(
        coincident_pairs=sum(map(len, by_target.values())), targets=len(by_target),
        ambiguous_target_times=ambiguous, folds=folds,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", nargs="+", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    began = perf_counter()
    results = {}
    for scene in SCENES:
        rows = read_archive_rows(args.archive, args.manifest, scene)
        labels = read_labels(args.archive, args.manifest, scene)
        results[scene] = summarize(*coincident_displacements(rows, labels))
    with args.output.open("x", encoding="utf-8") as output:
        json.dump(dict(scenes=results, wall_seconds=perf_counter()-began,
                       purpose="truth_conditioned_coincident_relative_translation"),
                  output, indent=2, allow_nan=False)
        output.write("\n")


if __name__ == "__main__":
    main()
