import math
import unittest
from unittest.mock import patch

from online_observation.causal_recovery import capture_scene
from online_observation.causal_recovery_metrics import score_recovery
from online_observation.stream import EARTH_RADIUS_M


def _position(east, north=0):
    return math.degrees(north / EARTH_RADIUS_M), math.degrees(east / EARTH_RADIUS_M)


def _pair(key):
    return ("A", key), ("B", key)


def _two_witnesses():
    target, first, second, protected = map(_pair, ("target", "first", "second", "protected"))
    fake = ("B", "fake")
    batches = []
    for stamp in range(0, 121, 20):
        before = 5 * min(stamp, 60)
        after = 5 * max(stamp - 60, 0)
        batches.append((stamp, {
            target[0]: _position(0), target[1]: _position(5000),
            first[0]: _position(20000 + before, after), first[1]: _position(25000 + before, after),
            second[0]: _position(40000 + before, -after), second[1]: _position(45000 + before, -after),
            protected[0]: _position(60000), protected[1]: _position(60100), fake: _position(65000),
        }))
    return tuple(batches)


class CausalRecoveryTests(unittest.TestCase):
    def test_independent_anchor_requires_its_two_completed_witnesses(self):
        batches = _two_witnesses()
        frames, _ = capture_scene(batches)
        prefix, _ = capture_scene(batches[:-1])
        self.assertEqual(frames[:-1], prefix)
        target, first, second, protected = map(_pair, ("target", "first", "second", "protected"))
        protected_alternative = protected[0], ("B", "fake")

        for frame in frames:
            self.assertIn(protected, frame.geometry_links)
            self.assertIn(target, frame.additions["wide"])
            self.assertNotIn(target, frame.additions["motion"])
            self.assertTrue(all(protected_alternative not in pairs for pairs in frame.additions.values()))
            self.assertFalse(next(row for row in frame.evidence if row.pair == protected_alternative).open)
        self.assertTrue(all(target not in frame.additions["anchor"] for frame in prefix))
        self.assertIn(target, frames[-1].additions["anchor"])
        self.assertIn(first, frames[-1].additions["motion"])
        self.assertIn(second, frames[-1].additions["motion"])
        target_evidence = next(row for row in frames[-1].evidence if row.pair == target)
        self.assertEqual(target_evidence.motion_status, "low_motion")
        self.assertEqual(target_evidence.anchor_status, "fitted")
        self.assertEqual(set(target_evidence.witnesses), {first, second})

        one_witness = tuple((stamp, {key: point for key, point in points.items() if key != second[1]})
                            for stamp, points in batches)
        single, _ = capture_scene(one_witness)
        single_evidence = next(row for row in single[-1].evidence if row.pair == target)
        self.assertEqual(single_evidence.anchor_status, "single")
        self.assertTrue(single_evidence.single_in_gate)
        self.assertNotIn(target, single[-1].additions["anchor"])

        labels = {key: key[1] for key in batches[0][1]}
        outside = score_recovery(frames, labels)["delays"]["outside"]
        self.assertEqual(outside["population"], 6)
        self.assertEqual(outside["arms"]["geometry"]["samples_seconds"], [])
        self.assertEqual(outside["arms"]["wide"]["samples_seconds"], [0] * 6)
        self.assertEqual(outside["arms"]["anchor"]["samples_seconds"], [120] * 2)
        self.assertEqual(outside["arms"]["motion"]["samples_seconds"], [120] * 4)

    def test_original_fit_changes_ring_only_strictly_after_600(self):
        pair = _pair("target")
        positions = {pair[0]: _position(0), pair[1]: _position(5000)}
        fit = dict(status="fitted", reason=None, source_order=["A", "B"],
                   translation_east_north_m=[2000, 0])
        with patch("online_observation.causal_recovery.estimate_translation", return_value=fit):
            frames, _ = capture_scene(((600, positions), (601, positions)))

        self.assertNotIn(pair, frames[0].geometry_edges)
        self.assertIn(pair, frames[0].additions["wide"])
        self.assertIn(pair, frames[1].geometry_edges)
        self.assertIn(pair, frames[1].geometry_links)
        self.assertTrue(all(pair not in additions for additions in frames[1].additions.values()))
