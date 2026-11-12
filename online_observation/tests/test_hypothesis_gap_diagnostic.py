from collections import Counter
import gzip
from io import BytesIO
import json
from pathlib import Path
import pickle
from tempfile import TemporaryDirectory
import unittest

from online_observation.continuous_tracker import LocalTrack
from online_observation.hypothesis_gap_diagnostic import Gap, Switch, _collect, _load_frozen, _measure, _score
from online_observation.hypothesis_gap_later import _capture_later
from online_observation.nemot_replay import ReplayResult


A, C = ("left", "1"), ("left", "2")
B, D = ("right", "1"), ("right", "2")
KEEP = ((A, B), (C, D))
DROP = ((A, D), (C, B))
POINTS = {A: (0., 0.), B: (0., 0.), C: (3., 0.), D: (3., 0.)}


def groups(edges):
    return tuple((str(i), edge) for i, edge in enumerate(edges, 1))


class GapDiagnosticTest(unittest.TestCase):
    def test_only_new_reports_can_change_the_matched_margin(self):
        gap = Gap(Switch(DROP, KEEP), 10., (KEEP[0],), True, True, False, False, -18.)
        priors = {key: LocalTrack(1., point, point) for key, point in POINTS.items()}
        costs = dict.fromkeys((*KEEP, *DROP), 1.)
        quiet = _measure(gap, 10., {}, priors, {}, costs)
        arrival = _measure(gap, 10., {B: (4., 0.)}, priors, {}, costs)
        self.assertEqual(quiet["increment_m2"], 0.)
        self.assertEqual(quiet["masked_m2"], arrival["masked_m2"])
        self.assertEqual((arrival["masked_m2"], arrival["actual_m2"]), (-18., 6.))
        self.assertIsNone(_measure(gap, 62., {B: (4., 0.)}, priors, {}, costs))

    def test_two_focal_edges_one_group_and_rejoin_closes(self):
        events = ((0., POINTS), (10., {A: (0., 0.)}), (12., {A: (0., 0.)}))
        reports = [(t, tuple(sorted(points))) for t, points in events]
        fit = {"translation_east_north_m": None}
        costs = tuple((edge, 1. if edge in DROP else 2.) for edge in (*KEEP, *DROP))
        result = ReplayResult({"kf": [(0., groups(KEEP)), (10., groups(DROP)), (12., groups(KEEP))],
                               "interval_swap": [(0., groups(KEEP))]}, reports, {},
                              {"fixed_candidate_edge_events": 12, "reports": 6}, fit, {})
        stream = BytesIO()
        for t, keys in reports:
            selected = DROP if t == 10. else KEEP
            pickle.dump((t, keys, keys if not t else (), tuple(POINTS), costs, selected), stream)
        pickle.dump(result, stream)
        stream.seek(0)
        gaps, counts = _collect(events, stream, result)
        self.assertEqual(counts["focal_departures"], 2)
        self.assertEqual(len(gaps), 1)
        self.assertEqual(gaps[0]["focal"], KEEP)
        self.assertEqual(gaps[0]["first_new"]["delay"], 0.)
        self.assertEqual((gaps[0]["end"], gaps[0]["stop"]), ("kf_rejoin", 12.))
        rows, bridges = _score(result, gaps, {A: 1, B: 1, C: 2, D: 2})
        self.assertEqual((rows[0]["true_release"], rows[0]["true_retain"]), (0, 2))
        self.assertEqual(json.loads(json.dumps(rows))[0]["counts"]["new_events"], 1)
        self.assertEqual(bridges, [])

    def test_resume_preserves_counters_in_the_frozen_capture(self):
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "frozen.pkl.gz"
            with gzip.open(path, "wb") as stream:
                pickle.dump(("snapshot", [{"counts": Counter([("new_events", 4), ("quiet_events", 1)])}]), stream)
            self.assertEqual(_load_frozen(path), ("snapshot", [{"counts": {"new_events": 4, "quiet_events": 1}}]))

    def test_later_samples_exclude_both_start_and_rejoin(self):
        events = ((0., POINTS), (10., {A: POINTS[A]}), (11., {A: POINTS[A]}),
                  (12., {B: POINTS[B]}), (13., {D: POINTS[D]}))
        reports = [(t, tuple(sorted(points))) for t, points in events]
        fit = {"translation_east_north_m": None}
        costs = tuple((edge, 1. if edge in DROP else 2.) for edge in (*KEEP, *DROP))
        result = ReplayResult({"kf": [(0., groups(KEEP)), (10., groups(DROP)), (13., groups(KEEP))],
                               "interval_swap": [(0., groups(KEEP))]}, reports, {},
                              {"fixed_candidate_edge_events": 20, "reports": 8}, fit, {})
        stream = BytesIO()
        for t, keys in reports:
            pickle.dump((t, keys, keys if not t else (), tuple(POINTS), costs,
                         DROP if 10. <= t < 13. else KEEP), stream)
        pickle.dump(result, stream)
        stream.seek(0)
        gaps, _ = _collect(events, stream, result)
        stream.seek(0)
        samples = _capture_later(events, stream, result, gaps)
        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0]["first_later"]["delay"], 1.)
        self.assertEqual(samples[0]["first_other_source"]["delay"], 2.)
        self.assertEqual(samples[0]["first_other_source"]["remaining_seconds"], 1.)


if __name__ == "__main__":
    unittest.main()
