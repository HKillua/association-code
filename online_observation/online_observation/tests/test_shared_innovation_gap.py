import unittest

from online_observation.continuous_tracker import LocalTrack
from online_observation.hypothesis_gap_diagnostic import Gap, Switch
from online_observation.shared_innovation_gap import Innovation, InnovationHistory, _sample


A, C = ("left", "1"), ("left", "2")
B, D = ("right", "1"), ("right", "2")
E, F = ("left", "3"), ("left", "4")
SWITCH = Switch(((A, B), (C, D)), ((A, D), (C, B)))


class SharedInnovationGapTest(unittest.TestCase):
    def test_neighbors_predict_held_out_report_and_change_partial_update(self):
        bank = InnovationHistory()
        positions = {A: (1., 0.), C: (0., 0.), B: (-1., 0.), D: (0., 0.),
                     E: (2., 0.), F: (4., 0.)}
        bank.priors.update({key: LocalTrack(8., position, position) for key, position in positions.items()})
        bank.latest["left"].update({E: Innovation(8., (-2., 0.), 2.), F: Innovation(8., (-2., 0.), 2.)})
        gap = Gap(SWITCH, 10., ((A, D),), True, True, False, False, 0.)
        result = _sample(gap, 10., {A: (-1., 0.)}, bank)
        self.assertEqual(result["status"], "supported")
        self.assertEqual(result["witnesses"], (E, F))
        self.assertEqual((result["raw"], result["masked"], result["neighbor"], result["shrink"]),
                         (2., -2., -2., -2.))
        self.assertEqual((result["raw_sq"], result["raw_neighbor_sq"]), (4., 0.))

    def test_prior_ledger_requires_two_continuous_reports_and_excludes_same_batch(self):
        bank = InnovationHistory()
        bank.ingest(0., {A: (0., 0.)})
        bank.ingest(10., {A: (0., 0.)})
        self.assertFalse(bank.candidates("left", 11., (0., 0.), set()))
        bank.ingest(20., {A: (1., 0.)})
        self.assertFalse(bank.candidates("left", 20., (0., 0.), set()))
        self.assertEqual(len(bank.candidates("left", 21., (0., 0.), set())), 1)
        self.assertFalse(bank.candidates("left", 81., (0., 0.), set()))
        bank.ingest(81., {A: (2., 0.)})
        self.assertFalse(bank.candidates("left", 82., (0., 0.), set()))


if __name__ == "__main__":
    unittest.main()
