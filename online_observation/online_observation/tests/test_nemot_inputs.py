import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from online_observation import nemot_experiment
from online_observation.continuous_tracker import Plane
from online_observation.nemot_inputs import MotionInputs, motion_batch, motion_examples


class MotionInputTests(unittest.TestCase):
    def test_complete_batch_and_neighbor_age_history_source(self):
        own = ("A", "query")
        fresh = ("A", "fresh")
        thin = ("A", "four-points")
        old = ("A", "old-history")
        opposite = ("B", "opposite")
        reports = {
            own: {70.0, 80.0, 140.0, 140.01},
            fresh: {40.0, 50.0, 60.0, 70.0, 80.0},
            thin: {40.0, 50.0, 60.0, 80.0},
            old: {0.0, 10.0, 20.0, 30.0, 120.0},
            opposite: {40.0, 50.0, 60.0, 70.0, 80.0, 140.0},
        }
        points = {own: (0.0, 0.0), fresh: (100.0, 0.0), thin: (5.0, 0.0),
                  old: (3.0, 0.0), opposite: (1.0, 0.0)}
        forward, reversed_batch = MotionInputs(), MotionInputs()
        for stamp in sorted(set().union(*reports.values())):
            batch = {key: points[key] for key in reports if stamp in reports[key]}
            forward.observe(stamp, batch)
            reversed_batch.observe(stamp, dict(reversed(tuple(batch.items()))))
            if own in batch:
                self.assertEqual(forward.example(own, stamp), reversed_batch.example(own, stamp))

        records = {record.stamp: record for record in forward.example(own, 140.01).history}
        self.assertEqual(records[70.0].neighbors, ())
        self.assertEqual(records[80.0].neighbors, ((100.0, 0.0, 0.0, 0.0, 0.0),))
        self.assertEqual(records[80.0].weights, (1.0,))
        self.assertEqual(records[140.0].neighbors, ((100.0, 0.0, 0.0, 0.0, 60.0),))
        self.assertEqual(records[140.01].neighbors, ())

    def test_training_features_equal_prefix_replay_after_future_positions_change(self):
        own, peer = ("A", "query"), ("A", "peer")
        events = [(stamp, {peer: (0.0, 0.001 + stamp / 100000.0)})
                  for stamp in (0.0, 10.0, 20.0, 30.0, 39.0)]
        events[0][1][own] = (0.0, 0.0)
        events[-1][1][own] = (0.0, 0.001)
        events.append((64.0, {own: (0.0, 0.002)}))
        plane = Plane.from_first_batch(events[0][1].values())
        prefix = MotionInputs()
        for stamp, positions in events[:-1]:
            prefix.observe(stamp, {key: plane.project(point) for key, point in positions.items()})

        training, = [example for example in motion_examples("synthetic", events)
                     if example.history[-1].stamp == 39.0]
        changed_events = [*events[:-1], (64.0, {own: (0.0, 0.02)})]
        changed, = [example for example in motion_examples("synthetic", changed_events)
                    if example.history[-1].stamp == 39.0]
        replay = prefix.example(own, 64.0)
        self.assertEqual(training.history, replay.history)
        self.assertEqual(training.center, replay.center)
        self.assertEqual(training.horizon, 25.0)
        self.assertEqual(changed.history, training.history)
        self.assertEqual(changed.center, training.center)
        self.assertNotEqual(changed.target, training.target)
        self.assertTrue(training.history[-1].neighbors)
        with self.assertRaisesRegex(ValueError, "no more than 60 seconds"):
            prefix.example(own, 99.01)

    def test_scene_cap_is_deterministic_and_targets_are_eligible_future_reports(self):
        own = ("A", "query")
        stamps = [0.0]
        for step in range(430):
            stamps.append(stamps[-1] + (6.0, 13.0, 27.0)[step % 3])
        events = [(stamp, {own: (0.0, stamp / 1_000_000.0)}) for stamp in stamps]
        plane = Plane.from_first_batch(events[0][1].values())
        examples = motion_examples("synthetic", events)
        self.assertEqual(examples, motion_examples("synthetic", events))
        self.assertEqual(len(examples), 384)
        anchor_indices = {stamp: index for index, stamp in enumerate(stamps)}
        for example in examples:
            anchor = example.history[-1].stamp
            index = anchor_indices[anchor]
            self.assertGreaterEqual(len(example.history), 2)
            target_time = anchor + example.horizon
            self.assertIn(target_time, stamps[index + 1:index + 4])
            self.assertGreater(example.horizon, 0)
            self.assertLessEqual(example.horizon, 60)
            self.assertEqual(example.target, plane.project((0.0, target_time / 1_000_000.0)))
        batch = motion_batch(examples)
        self.assertEqual(len(batch), 384)
        self.assertEqual(batch.target.shape, (384, 2))

    def test_frozen_roles_are_component_disjoint(self):
        root = Path(__file__).resolve().parents[2]
        manifest_path = root / "configs/data/mtad-level1-scene3000.json"
        roster = nemot_experiment.read_roster(manifest_path)
        manifest = json.loads(manifest_path.read_text())
        components = manifest["components"]
        self.assertEqual({role: len(scenes) for role, scenes in roster["splits"].items()},
                         {"train": 64, "select": 12, "held": 28})
        selected_components = {components[scene] for scene in roster["splits"]["select"]}
        assigned = set().union(*map(set, roster["splits"].values()))
        conflicting = next(scene for scene in manifest["splits"]["T_core"]
                           if scene not in assigned and components[scene] in selected_components)
        roster["splits"]["held"][0] = conflicting
        roster["components"][conflicting] = components[conflicting]
        with TemporaryDirectory() as folder:
            altered = Path(folder) / "roster.json"
            altered.write_text(json.dumps(roster))
            with (patch.object(nemot_experiment, "ROSTER_PATH", altered),
                  self.assertRaisesRegex(ValueError, "held scene or shared-A component isolation")):
                nemot_experiment.read_roster(manifest_path)


if __name__ == "__main__":
    unittest.main()
