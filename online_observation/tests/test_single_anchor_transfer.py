import unittest

from online_observation.cross_source_calibration import AnchorSample, predictive_scores
from online_observation.single_anchor_transfer import score_snapshot


REPORTER = ("a", "0")
KEYS = (REPORTER, ("a", "1"), ("b", "0"), ("b", "1"))


def _anchor(stamp, value, index):
    edge = (("a", str(index)), ("b", str(index)))
    anchor = AnchorSample(edge, edge[0], stamp, 5.0, (value, 0.0), 1.0, False, False)
    return anchor, (0.0, 20.0, None)


def _snapshot(actual, masked, anchors, variances=(1.0, 1.0)):
    observed = tuple((value, 0.0) for value in actual)
    predicted = tuple((value, 0.0) for value in masked)
    sides = ("release", "retain")
    return dict(stamp=100.0, keys=KEYS, reporter=REPORTER,
                cross=dict(zip(sides, observed, strict=True)),
                pred_delta=dict(zip(sides, predicted, strict=True)),
                age=dict.fromkeys(sides, 5.0), variance=dict(zip(sides, variances, strict=True)),
                scores=predictive_scores(*observed, *predicted, *variances, None),
                anchors={"graph": {"provenance": anchors}})


class SingleAnchorTransferTest(unittest.TestCase):
    def test_both_edges_can_get_closer_while_the_wrong_edge_improves_more(self):
        snapshot = _snapshot((12.0, 10.0), (12.0, 10.0), [_anchor(96.0, 12.0, 2)])
        result = score_snapshot(snapshot)
        false, true = (result["decomposition"][side] for side in ("release", "retain"))
        self.assertEqual((true["raw_squared"], true["adjusted_squared"]), (100.0, 4.0))
        self.assertEqual((false["raw_squared"], false["adjusted_squared"]), (144.0, 0.0))
        self.assertLess(false["delta_cost"], true["delta_cost"])
        self.assertEqual(result["margin_delta"]["quadratic"], 0.0)
        self.assertEqual(result["margin_delta"]["actual"], 48.0)
        self.assertEqual(result["scores"]["raw"], -44.0)
        self.assertEqual(result["scores"]["corroborated"], 4.0)
        self.assertEqual(result["reason"], "eligible")

    def test_masked_disagreement_rejects_an_actual_warning(self):
        snapshot = _snapshot((12.0, 10.0), (14.0, 12.0), [_anchor(96.0, 12.0, 2)])
        result = score_snapshot(snapshot)
        self.assertGreater(result["scores"]["always"], 0.0)
        self.assertLess(result["scores"]["always_masked"], 0.0)
        self.assertEqual(result["scores"]["corroborated"], 0.0)
        self.assertEqual(result["reason"], "corrected_mask_disagrees")

    def test_equal_variance_control_rejects_a_variance_driven_warning(self):
        snapshot = _snapshot((14.0, 13.0), (14.0, 13.0), [_anchor(96.0, 10.0, 2)], (4.0, 1.0))
        result = score_snapshot(snapshot)
        self.assertGreater(result["scores"]["always"], 0.0)
        self.assertGreater(result["scores"]["always_masked"], 0.0)
        self.assertLess(result["equal_variance"]["always"], 0.0)
        self.assertEqual(result["scores"]["corroborated"], 0.0)
        self.assertEqual(result["reason"], "equal_variance_disagrees")

    def test_anchor_expiry_and_deterministic_newest_selection(self):
        for anchors in ([], [_anchor(39.9, 7.0, 2)]):
            with self.subTest(anchors=anchors):
                result = score_snapshot(_snapshot((3.0, 4.0), (3.0, 4.0), anchors))
                self.assertIsNone(result["anchor"])
                self.assertEqual(result["reason"], "no_recent")
                self.assertEqual(result["scores"]["always"], result["scores"]["hold"])
                self.assertGreater(result["scores"]["raw_agreement"], 0.0)
        boundary = _anchor(40.0, 7.0, 2)
        result = score_snapshot(_snapshot((3.0, 4.0), (3.0, 4.0), [boundary]))
        self.assertEqual(result["anchor"], boundary[0])
        newest = _anchor(50.0, 8.0, 3)
        result = score_snapshot(_snapshot((3.0, 4.0), (3.0, 4.0), [boundary, newest]))
        self.assertEqual(result["anchor"], newest[0])

    def test_rejects_noncausal_frozen_anchor(self):
        snapshot = _snapshot((3.0, 4.0), (3.0, 4.0), [_anchor(100.0, 7.0, 2)])
        with self.assertRaisesRegex(ValueError, "frozen provenance"):
            score_snapshot(snapshot)


if __name__ == "__main__":
    unittest.main()
