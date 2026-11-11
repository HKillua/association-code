import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from online_observation import prefix_gate_probe as probe
from online_observation.registration_probe import estimate_translation


class PrefixGateProbeTests(unittest.TestCase):
    def test_fitted_and_fallback_choices_and_paired_tradeoffs(self):
        prefix_points = {(source, f"prefix-{i}"): (0, lon + shift)
                         for source, shift in (("A", 0), ("B", .02))
                         for i, lon in enumerate((0, .5, 1))}
        prefix = [(stamp, prefix_points) for stamp in (30, 60, 90)]
        queries = {("A", "near"): (0, 0), ("A", "far"): (0, 1), ("A", "wrong"): (0, 2)}
        late = [(601, queries | {("B", "near"): (0, .023), ("B", "far"): (0, 1.035),
                                 ("B", "wrong-peer"): (0, 2.035)})]
        scored = []
        for events, gate in ((prefix + late, 1000), (late, 4000)):
            first_seen, births, selected = probe.freeze_decisions(events, estimate_translation(events))
            self.assertEqual(selected, gate)
            arm = probe.FIXED_1 if gate == 1000 else probe.FIXED_4
            for key in queries:
                self.assertIs(births[key].choices[probe.CONDITIONAL], births[key].choices[arm])
            scored.extend(probe.score(first_seen, {key: births[key] for key in queries},
                                      {key: key[1] for key in first_seen}))

        report = probe.summarize(scored)
        self.assertEqual(report["births"], 6)
        self.assertEqual(report["opportunities"], dict(fresh=4, stale=0, never_observed=2))
        self.assertEqual(report["arms"][probe.CONDITIONAL]["correct"], 3)
        self.assertEqual(report["arms"][probe.CONDITIONAL]["wrong"], 1)
        self.assertEqual(report["arms"][probe.CONDITIONAL]["reject"], 2)
        narrow = report["comparisons"]["prefix_conditional_minus_fixed_1km"]
        self.assertEqual((narrow["true_accepts_restored"], narrow["false_accepts_restored"],
                          narrow["no_history_false_accepts_restored"]), (2, 1, 1))
        self.assertEqual(narrow["delta"], dict(correct=2, wrong=1, reject=-3))
        wide = report["comparisons"]["prefix_conditional_minus_fixed_4km"]
        self.assertEqual((wide["true_accepts_lost"], wide["false_accepts_removed"],
                          wide["no_history_false_accepts_removed"]), (1, 1, 1))
        self.assertEqual(wide["delta"], dict(correct=-1, wrong=-1, reject=2))
        self.assertEqual(report["by_pool_size"]["0-3"]["births"], 6)
        self.assertEqual(report["by_source"]["A"]["births"], 6)
        self.assertIsNone(report["by_pool_size"]["4-15"]["arms"][probe.FIXED_4]["rates"]["coverage"])

    def test_cutoffs_tie_and_late_new_source_fallback(self):
        a, b, c = ("A", "query"), ("B", "peer"), ("C", "late-source")
        fit = dict(status="fitted", source_order=["A", "B"], translation_east_north_m=[0., 0.])
        events = [(541, {b: (0, .004)}), (600, {("A", "cutoff"): (0, 50)}),
                  (601, {a: (0, 0), c: (0, .004)})]
        _, births, gate = probe.freeze_decisions(events, fit)
        self.assertEqual(gate, 1000)
        self.assertNotIn(("A", "cutoff"), births)
        self.assertEqual(births[a].fresh, {b, c})
        self.assertEqual(births[a].fitted_pairs, 1)
        for arm in probe.ARMS:
            self.assertEqual(births[a].choices[arm].reject_reason, "tie")

    def test_cli_freezes_every_scene_before_label_reads(self):
        roster = {"first": ("10", "20"), "second": ("30",)}
        calls = []
        freeze = probe.freeze_decisions

        def measurements(parts, manifest, scene, *, required_split):
            self.assertEqual(required_split, "T_core")
            calls.append(("measurement", scene))
            return [dict(time="600", source="A", batch="private-batch", lat="0", lon="0")]

        def freeze_decisions(events, fit):
            calls.append(("frozen", None))
            return freeze(events, fit)

        def labels(parts, manifest, scene):
            self.assertEqual(calls[:6], [
                ("measurement", "10"), ("frozen", None), ("measurement", "20"), ("frozen", None),
                ("measurement", "30"), ("frozen", None),
            ])
            calls.append(("label", scene))
            return {("A", "private-batch"): "private-identity"}

        with TemporaryDirectory() as folder:
            manifest = Path(folder) / "manifest.json"
            manifest.write_text(json.dumps({"components": {"10": "component-a", "20": "component-a",
                                                          "30": "component-b"}}), encoding="utf-8")
            output = Path(folder) / "summary.json"
            args = ["--archive", "one", "two", "three", "--manifest", str(manifest), "--output", str(output)]
            with (patch.object(probe, "ROSTER", roster),
                  patch.object(probe, "read_archive_rows", side_effect=measurements),
                  patch.object(probe, "freeze_decisions", side_effect=freeze_decisions),
                  patch.object(probe, "read_labels", side_effect=labels)):
                probe.main(args)
                with self.assertRaises(FileExistsError):
                    probe.main(args)
            payload = output.read_text(encoding="utf-8")
        self.assertNotIn("private-", payload)
        self.assertEqual(calls[6:], [("label", "10"), ("label", "20"), ("label", "30")])
        result = json.loads(payload)
        self.assertEqual(result["scene_ids"], ["10", "20", "30"])
        self.assertEqual(result["scenes"]["10"]["component_id"], "component-a")
        self.assertEqual(result["scenes"]["10"]["conditional_gate_metres"], 4000)
        self.assertEqual(result["aggregate"]["births"], 0)
        self.assertEqual(result["by_fit_status"]["raw_fallback"]["scene_ids"], ["10", "20", "30"])
        self.assertEqual(result["by_roster_stratum"]["first"]["scene_ids"], ["10", "20"])
        self.assertEqual(result["by_roster_stratum"]["first"]["scenes_with_births"], 0)


if __name__ == "__main__":
    unittest.main()
