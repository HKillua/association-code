import unittest

from online_observation.association_probe import birth_candidates, evaluate_births
from online_observation.fusion_probe import _forecasts
from online_observation.positions import event_batches
from online_observation.registration_probe import estimate_translation


def row(time, source, batch, lon):
    return dict(time=time, source=source, batch=batch, lat=0, lon=lon,
                vel="unavailable", cou="unavailable")


class StrictReplayTests(unittest.TestCase):
    def test_forecast_uses_only_prefix_and_ignores_speed_and_course_validity(self):
        prefix = [row(0, "A", "1", 0.), row(2, "A", "1", .0001)]
        modified = [dict(observation, vel=-9, cou="nan") for observation in prefix]
        modified.append(row(3, "A", "1", 80.))
        first = _forecasts(prefix)
        second = _forecasts(modified)
        self.assertEqual([next(first) for _ in range(3)], [next(second) for _ in range(3)])

    def test_same_epoch_is_atomic_and_raw_fragment_number_has_no_rank(self):
        original = [row(0, "A", "1", 0.), row(0, "B", "2", .001),
                    row(0, "B", "3", -.001)]
        labels = {("A", "1"): "truth", ("B", "2"): "truth", ("B", "3"): "other"}
        first = evaluate_births(birth_candidates(original), labels)
        changed = [dict(observation, batch="alias-" + observation["batch"])
                   for observation in reversed(original)]
        changed_labels = {(source, "alias-" + batch): identity
                          for (source, batch), identity in labels.items()}
        second = evaluate_births(birth_candidates(changed), changed_labels)
        self.assertEqual(first, second)
        self.assertEqual(first["cross_source", 60, "hold", 250]["nearest_mixed_tie"], 1)

    def test_conflicting_positions_fail_even_if_unused_fields_agree(self):
        with self.assertRaisesRegex(ValueError, "conflicting same-time positions"):
            list(event_batches([row(0, "A", "1", 0), row(0, "A", "1", .1)]))

    def test_frozen_registration_ignores_the_evaluation_period(self):
        points = {(source, str(i)): (0, lon + shift)
                  for source, shift in (("A", 0), ("B", .02))
                  for i, lon in enumerate((0, .5, 1))}
        prefix = [(time, points) for time in (30, 60, 90)]
        fit = estimate_translation(prefix)
        self.assertEqual(fit["status"], "fitted")
        self.assertAlmostEqual(fit["translation_east_north_m"][0], 2223.8985, places=3)
        self.assertEqual(fit, estimate_translation(
            prefix + [(601, {("C", "0"): (30, 30)})]
        ))


if __name__ == "__main__":
    unittest.main()
