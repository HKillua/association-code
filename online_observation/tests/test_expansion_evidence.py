import unittest

from online_observation.causal_common_motion import compare_common_motion
from online_observation.expansion_evidence import (
    HistoricalGeometryWitnesses,
    OffsetEvidence,
    WitnessEvidence,
    common_offset_evidence,
    current_common_offset,
)


TARGET = (("a", "target"), ("b", "target"))


def _pair(name):
    return ("a", name), ("b", name)


def _histories(offsets):
    histories = {}
    for (left, right), offset in offsets.items():
        histories[left] = [(0, (0, 0)), (10, (10, 0))]
        histories[right] = [(0, (offset, 0)), (10, (offset + 10, 0))]
    return histories


class ExpansionEvidenceTests(unittest.TestCase):
    def test_straight_line_uses_a_distinct_held_offset(self):
        stamps = range(0, 121, 20)
        left = [(t, (3 * t, 0)) for t in stamps]
        right = [(t, (3 * t + (100 + t / 6 if t <= 60 else 110 + t - 60), 0)) for t in stamps]
        common = compare_common_motion(left, right, 120)
        self.assertTrue(common.supported)
        self.assertEqual(common.status, "degenerate")
        self.assertEqual(common_offset_evidence(common), OffsetEvidence(10, 65))
        self.assertEqual(common_offset_evidence(compare_common_motion(left, right, 60)),
                         OffsetEvidence(None, None))

    def test_latest_common_offset_requires_arrived_brackets_and_limits_gap(self):
        left = [(10, (10, 0)), (30, (30, 0))]
        right = [(20, (120, 0))]
        self.assertIsNone(current_common_offset(left, right, 20))
        self.assertEqual(current_common_offset(left, right, 30), (100, 0))
        self.assertEqual(current_common_offset([(20, (20, 0))], right, 20), (100, 0))
        self.assertIsNone(current_common_offset([(0, (0, 0)), (46, (46, 0))], right, 46))

    def test_witness_needs_both_source_updates_and_strictly_earlier_selection(self):
        witness = _pair("witness")
        left, right = witness
        histories = {left: [(0, (0, 0)), (10, (10, 0))],
                     right: [(0, (100, 0)), (20, (120, 0))]}
        recorder = HistoricalGeometryWitnesses()
        for stamp in (0, 10, 20):
            recorder.observe(stamp, (witness,), histories)
        self.assertEqual(recorder.lookup(TARGET, 20, (100, 0)), WitnessEvidence(0, None, ()))
        self.assertEqual(recorder.lookup(TARGET, 21, (105, 0)), WitnessEvidence(1, 5, (witness,)))
        self.assertEqual(recorder.lookup(TARGET, 240, None), WitnessEvidence(1, None, (witness,)))
        self.assertEqual(recorder.lookup(TARGET, 240.001, (100, 0)), WitnessEvidence(0, None, ()))

    def test_target_exclusion_and_cross_witness_agreement(self):
        first, second = _pair("one"), _pair("two")
        overlapping = TARGET[0], ("b", "other")
        offsets = {first: 100, second: 300, overlapping: 900}
        recorder = HistoricalGeometryWitnesses()
        for stamp in (0, 10):
            recorder.observe(stamp, offsets, _histories(offsets))
        self.assertEqual(recorder.lookup(TARGET, 11, (220, 0)),
                         WitnessEvidence(2, 20, (first, second)))
        conflicts = HistoricalGeometryWitnesses()
        offsets = {first: 100, second: 601}
        for stamp in (0, 10):
            conflicts.observe(stamp, offsets, _histories(offsets))
        self.assertEqual(conflicts.lookup(TARGET, 11, (100, 0)),
                         WitnessEvidence(2, None, (first, second)))


if __name__ == "__main__":
    unittest.main()
