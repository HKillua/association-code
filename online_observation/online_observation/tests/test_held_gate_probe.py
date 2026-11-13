import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from online_observation import held_gate_probe as probe
from online_observation.registration_probe import estimate_translation


class HeldGateProbeTests(unittest.TestCase):
    def test_time_boundary_same_batch_and_future_history(self):
        old, edge, competitor, future, same = [("B", name) for name in
                                               ("old", "edge", "competitor", "future", "same")]
        at_cutoff, stale, fresh, future_truth, simultaneous, empty = [
            ("A", name) for name in ("at_cutoff", "stale", "fresh", "future_truth", "same", "empty")]
        events = [
            (540.999, {old: (0, .1)}),
            (541, {edge: (0, .2), competitor: (0, .004)}),
            (600, {at_cutoff: (0, 50)}),
            (601, {stale: (0, .1), fresh: (0, .2), future_truth: (0, 0),
                   simultaneous: (0, .4), same: (0, .404)}),
            (602, {future: (0, .001)}),
            (800, {empty: (0, .1)}),
        ]
        fit = estimate_translation(events)
        self.assertEqual(fit["status"], "abstained")
        first_seen, births = probe.predict(events, fit)
        self.assertNotIn(at_cutoff, births)
        self.assertNotIn(old, births[stale].fresh)
        self.assertIn(edge, births[fresh].fresh)
        self.assertIn(same, births[simultaneous].fresh)
        self.assertIn(simultaneous, births[same].fresh)
        self.assertEqual(births[future_truth], probe.predict(events[:4], fit)[1][future_truth])
        labels = {key: key[1] for key in first_seen}
        labels.update({old: "stale", edge: "fresh", future: "future_truth", empty: "stale"})
        query_keys = (stale, fresh, future_truth, simultaneous, empty)
        rows = probe.score(first_seen, {key: births[key] for key in query_keys}, labels)
        self.assertEqual([row.opportunity for row in rows],
                         ["stale", "fresh", "never_observed", "fresh", "stale"])
        self.assertEqual(rows[-1].arms["raw|1000"].reject_reason, "no_candidate")
        for row in rows:
            self.assertEqual(row.arms["raw|1000"], row.arms["corrected|1000"])
            self.assertEqual(row.fitted_pairs, 0)
        with self.assertRaisesRegex(ValueError, "unmapped measurement fragment"):
            probe.score(first_seen, births, {key: label for key, label in labels.items() if key != old})

    def test_gate_boundaries_and_exact_tie(self):
        one, two, three = [("B", name) for name in ("one", "two", "three")]
        self.assertEqual(probe._choose({one: 1000, two: 1000.001}, 1000).winner, one)
        self.assertEqual(probe._choose({one: 4000}, 4000).winner, one)
        self.assertEqual(probe._choose({one: 4000.001}, 4000).reject_reason, "distance")
        self.assertEqual(probe._choose({one: 1000, two: 1000, three: 2000}, 1000).reject_reason, "tie")

    def test_report_reconciles_misses_transitions_and_no_history_false_accepts(self):
        queries = {("A", str(i)): (0, float(i)) for i in range(6)}
        peers = {("B", str(i)): (0, lon) for i, lon in enumerate((.004, 1.02, 2.005, 3.02, 5.001, 5.001, 2.007))}
        events = [(601, queries | peers)]
        first_seen, births = probe.predict(events, estimate_translation(events))
        labels = {key: f"{key[0]}-{key[1]}" for key in first_seen}
        labels.update({("B", "0"): "A-0", ("B", "1"): "A-1", ("B", "4"): "A-5"})
        rows = probe.score(first_seen, {key: births[key] for key in queries}, labels)
        report = probe.summarize(rows)
        self.assertEqual(report["births"], 6)
        self.assertEqual(report["opportunities"], dict(fresh=3, stale=0, never_observed=3))
        narrow, wide = (report["arms"][f"corrected|{gate}"] for gate in (1000, 4000))
        self.assertEqual((wide["correct"], wide["wrong"], wide["reject"]), (2, 2, 2))
        self.assertEqual((narrow["correct"], narrow["wrong"], narrow["reject"]), (1, 1, 4))
        self.assertEqual(narrow["fresh_true_outside_gate"], 1)
        self.assertEqual(narrow["reject_with_true_in_gate"], 1)
        self.assertEqual(narrow["missed_fresh_truth"], 2)
        self.assertEqual(narrow["reject_reasons"], dict(no_candidate=0, distance=3, tie=1))
        self.assertEqual(narrow["wrong_by_in_gate_multiplicity"], dict(single=0, multiple=1))
        self.assertEqual(wide["no_history_wrong_by_in_gate_multiplicity"], dict(single=1, multiple=1))
        changes = report["transitions_4000_to_1000"]["corrected"]
        self.assertEqual(changes["true_accepts_lost"], 1)
        self.assertEqual(changes["false_accepts_removed"], 1)
        self.assertEqual(changes["no_history_false_accepts_remaining_1000"], 1)
        self.assertEqual(changes["no_history_false_accepts_removed"], 1)
        self.assertEqual(report["by_pool_size"]["4-15"]["births"], 6)
        self.assertEqual(report["by_pool_size"]["0-3"]["births"], 0)
        self.assertIsNone(report["by_pool_size"]["16+"]["arms"]["raw|1000"]["rates"]["coverage"])

    def test_fitted_and_unrecognized_source_pairs_share_the_birth_pool(self):
        one, two, three = ("A", "one"), ("B", "two"), ("C", "three")
        fit = dict(source_order=["A", "B"], translation_east_north_m=[2223.898532891175, 0])
        first_seen, births = probe.predict([(601, {one: (0, 0), two: (0, .02), three: (0, .05)})], fit)
        self.assertEqual(births[one].fresh, {two, three})
        self.assertEqual(births[one].fitted_pairs, 1)
        self.assertEqual(births[three].fitted_pairs, 0)
        self.assertIsNone(births[one].choices["raw|1000"].winner)
        self.assertEqual(births[one].choices["corrected|1000"].winner, two)
        rows = probe.score(first_seen, births, {one: "shared-1", two: "shared-1", three: "other-1"})
        report = probe.summarize(rows)
        self.assertEqual((report["candidate_pairs"], report["fitted_candidate_pairs"],
                          report["fallback_candidate_pairs"]), (6, 2, 4))

    def test_cli_freezes_every_scene_before_labels_and_keeps_zero_birth_scene(self):
        calls = []
        real_predict = probe.predict

        def measurements(parts, manifest, scene, *, required_split):
            self.assertEqual(required_split, "T_core")
            calls.append(("measurement", scene))
            return [dict(source="A", batch="private-batch", time="600", lat="0", lon="0")]

        def predict(events, fit):
            calls.append(("prediction", None))
            return real_predict(events, fit)

        def labels(parts, manifest, scene):
            self.assertEqual(calls[:2 * len(probe.SCENES)], [
                call for sid in probe.SCENES for call in (("measurement", sid), ("prediction", None))])
            calls.append(("label", scene))
            return {("A", "private-batch"): f"private-target-{1 if scene in probe.SCENES else 2}"}

        with TemporaryDirectory() as folder:
            destination = Path(folder) / "summary.json"
            arguments = ["--archive", "p1", "p2", "p3", "--manifest", "synthetic", "--output", str(destination)]
            with (patch.object(probe, "read_archive_rows", side_effect=measurements),
                  patch.object(probe, "predict", side_effect=predict),
                  patch.object(probe, "read_labels", side_effect=labels)):
                probe.main(arguments)
                with self.assertRaises(FileExistsError):
                    probe.main(arguments)
            payload = destination.read_text(encoding="utf-8")
        self.assertNotIn("private-", payload)
        summary = json.loads(payload)
        self.assertEqual(summary["scene_ids"], list(probe.SCENES))
        self.assertEqual(summary["aggregate"]["births"], 0)
        self.assertEqual(len(summary["by_fit_status"]["raw_fallback"]["scene_ids"]), 32)
        self.assertEqual(summary["by_fit_status"]["raw_fallback"]["scenes_with_births"], 0)
        overlap = summary["identity_overlap_with_development"]
        self.assertEqual((overlap["A"]["shared"], overlap["A-B"]["shared"]), (1, 0))
        self.assertEqual(calls[2 * len(probe.SCENES):], [
            ("label", scene) for scene in (*probe.SCENES, *probe.DEVELOPMENT_SCENES)])


if __name__ == "__main__":
    unittest.main()
