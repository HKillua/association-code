import unittest
from dataclasses import replace
from math import log

from online_observation.continuous_tracker import LocalTrack, MEMORY_S
from online_observation.cross_source_calibration import (
    AnchorSample, decompose, eligible_anchors, fit_moments, predictive_scores,
)
from online_observation.cross_source_calibration_gap import AnchorHistory


KEYS = {("a", "0"), ("a", "1"), ("b", "0"), ("b", "1")}


def sample(left, right, stamp, age=15., residual=(2., 4.), variance=2., motion=True, registered=True):
    edge = (("a", left), ("b", right))
    return AnchorSample(edge, edge[0], stamp, age, residual, variance, motion, registered)


class CrossSourceCalibrationTest(unittest.TestCase):
    def test_anchor_write_uses_the_earlier_graph_and_reads_only_after_its_report(self):
        left, right = ("a", "2"), ("b", "2")
        edge = left, right
        bank = AnchorHistory()
        graph = {edge: 1.0}
        bank.advance(0., {}, {}, graph, (edge,), {})
        bank.advance(20., {}, {}, graph, (edge,), {})
        prior = LocalTrack(11., (0., 0.), (0., 0.))
        bank.advance(21., {left: (25., 0.)}, {left: prior, right: prior}, {}, (), {})
        self.assertEqual(bank.read(KEYS, ("a", "0"), 21., (10., 10.), {})["graph"]["n"], 0)
        readout = bank.read(KEYS, ("a", "0"), 22., (10., 10.), {})
        self.assertEqual(readout["graph"]["n"], 1)
        observed, certificate = readout["graph"]["provenance"][0]
        self.assertEqual(observed.residual, (25., 0.))
        self.assertEqual(certificate[:2], (0., 20.))
        self.assertEqual(readout["motion"]["n"], 0)
        bank.advance(22., {left: (25., 0.)}, {left: prior, right: prior}, graph, (edge,), {})
        self.assertEqual(bank.counts["writes"], 1)

    def test_anchor_eligibility_respects_strict_prefix_direction_and_both_horizons(self):
        records = [sample("2", "2", 100.), sample("3", "3", 101.),
                   sample("4", "4", 90., age=7.), sample("5", "5", 95., age=21.),
                   sample("0", "6", 90.), sample("7", "7", 93., registered=False),
                   sample("8", "8", 100.-MEMORY_S), sample("9", "9", 99.-MEMORY_S)]
        reversed_direction = sample("10", "10", 97.)
        records.append(replace(reversed_direction, reporter=reversed_direction.edge[1]))
        exact_age = sample("11", "11", 98., age=10.)
        records.append(exact_age)
        result = eligible_anchors(records, KEYS, ("a", "0"), 100., (10., 20.), True)
        self.assertEqual(result, (exact_age, records[6]))

    def test_anchor_eligibility_is_endpoint_disjoint_and_motion_is_stronger(self):
        first = sample("2", "3", 98., motion=False)
        tied = sample("2", "4", 98.)
        overlaps = sample("5", "3", 97.)
        independent = sample("6", "6", 96.)
        records = (independent, overlaps, tied, first)
        self.assertEqual(eligible_anchors(records, KEYS, ("a", "0"), 100., (10., 20.), True),
                         (first, independent))
        self.assertEqual(eligible_anchors(records, KEYS, ("a", "0"), 100., (10., 20.), True, motion=True),
                         (tied, overlaps, independent))

    def test_residual_energy_and_fitted_spread_are_distinct(self):
        vectors = ((2., 4.), (4., 4.), (6., 4.))
        parts = decompose(vectors)
        self.assertEqual(parts["n"], 3)
        self.assertEqual(parts["mean"], (4., 4.))
        self.assertEqual(parts["mean_norm_squared"], 32.)
        self.assertAlmostEqual(parts["centered_spread"], 8. / 3.)
        self.assertAlmostEqual(parts["mean_squared"], parts["mean_norm_squared"] + parts["centered_spread"])
        self.assertEqual(decompose(())["n"], 0)
        records = [sample(str(i), str(i), 50.+i, residual=point) for i, point in enumerate(vectors)]
        self.assertEqual(fit_moments(records), {"mean": (4., 4.), "scale": 1.})
        self.assertIsNone(fit_moments(records[:2])["scale"])
        self.assertEqual(fit_moments(()), {"mean": None, "scale": None})
        self.assertIsNone(fit_moments([records[0]] * 3)["scale"])

    def test_common_scale_changes_the_residual_term_only(self):
        actual = ((0., 0.), (3., 0.))
        masked = ((2., 0.), (1., 0.))
        fitted = {"mean": (1., 0.), "scale": 4.}
        score = predictive_scores(*actual, *masked, 4., 1., fitted)
        volume = 2 * log(1. / 4.)
        self.assertAlmostEqual(score["raw"], 9. + volume)
        self.assertAlmostEqual(score["masked"], volume)
        self.assertAlmostEqual(score["scale"], 9. / 4. + volume)
        self.assertAlmostEqual(score["mean"], 3.75 + volume)
        self.assertAlmostEqual(score["both"], 3.75 / 4. + volume)
        self.assertAlmostEqual(score["both_masked"], -0.25 / 4. + volume)
        equal = predictive_scores(*actual, *masked, 4., 4., fitted)
        self.assertEqual(equal["scale"], equal["raw"] / 4.)
        no_fit = predictive_scores(*actual, *masked, 4., 4., None)
        self.assertIsNone(no_fit["mean"])
        self.assertIsNone(no_fit["scale"])


if __name__ == "__main__":
    unittest.main()
