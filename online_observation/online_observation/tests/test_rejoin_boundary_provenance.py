import unittest

from online_observation.continuous_tracker import LocalTrack
from online_observation.hypothesis_gap_diagnostic import Switch
from online_observation.rejoin_boundary_provenance import audit_boundary


class BoundaryProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.trigger, self.sibling = ("A", "1"), ("A", "2")
        self.retain, self.release = ("B", "1"), ("B", "2")
        self.batch_peer, self.tie_peer, self.expired = ("B", "3"), ("B", "4"), ("B", "5")
        self.switch = Switch(((self.trigger, self.release), (self.sibling, self.retain)),
                             ((self.trigger, self.retain), (self.sibling, self.release)))
        self.points = {self.trigger: (0.0, 0.0), self.batch_peer: (7000.0, 0.0)}
        self.priors = {
            self.trigger: LocalTrack(19, (5200.0, 0.0), (5200.0, 0.0)),
            self.retain: LocalTrack(19, (1000.0, 0.0), (1400.0, 0.0)),
            self.release: LocalTrack(19, (2000.0, 0.0), (900.0, 0.0)),
            self.batch_peer: LocalTrack(19, (100.0, 0.0), (100.0, 0.0)),
            self.tie_peer: LocalTrack(19, (4000.0, 0.0), (4000.0, 0.0)),
        }
        self.costs = {(self.trigger, self.retain): 1_960_000, (self.trigger, self.release): 810_000,
                      (self.sibling, self.retain): 3_000_000, (self.sibling, self.release): 100_000,
                      (self.trigger, self.tie_peer): 16_000_000}
        self.frame = (20, tuple(self.points), (), (*self.priors, self.sibling), tuple(self.costs.items()),
                      self.switch.retain)
        self.previous = (19, (), (), (), (*tuple(self.costs.items()), ((self.trigger, self.expired), 1000)),
                         self.switch.release)
        self.shape = dict(stamp=20, geometry=dict(actual=5, masked=-10),
                          shape=dict(status="supported", available_at=20, current_used=True,
                                     provenance={key: (19, 20) if key == self.trigger else (19,)
                                                 for key in self.switch.keys},
                                     actual=dict(two_block=8), masked=dict(two_block=2)))

    def audit(self):
        return audit_boundary(self.switch, self.trigger, 20, self.priors, self.points, {}, self.frame,
                              self.previous, (self.switch.retain[0],), self.shape)

    def test_gate_null_eligibility_and_fixed_margin_are_distinct(self):
        graph = self.audit()["graph"]
        self.assertEqual(graph["peers_same_batch"], 1)
        self.assertEqual(graph["fixed_incident"]["gate"], dict(actual=2, masked=1, loses=1, gains=0))
        self.assertEqual(graph["fixed_incident"]["legal"], dict(actual=2, masked=0, loses=2, gains=0))
        self.assertEqual(graph["incident"]["gate"], dict(actual=3, masked=3, loses=1, gains=1))
        self.assertEqual(graph["incident"]["legal"], dict(actual=2, masked=2, loses=2, gains=2))
        self.assertEqual(graph["incident_previous_to_actual"]["gate"]["loses"], 1)
        self.assertEqual(graph["factual_cached_margin_m2"], -1_750_000)
        self.assertEqual(graph["coordinate_mask_cached_margin_m2"], -6_950_000)
        self.assertTrue(graph["retain_selected"])
        self.assertFalse(graph["previous_retain_selected"])

    def test_shape_use_can_differ_from_action_dependence(self):
        shape = self.audit()["shape"]
        self.assertTrue(shape["trigger_in_shape_provenance"])
        self.assertTrue(shape["current_action"])
        self.assertTrue(shape["shape_only_mask_action"])
        self.assertFalse(shape["full_coordinate_mask_action"])

    def test_frozen_graph_disagreement_fails(self):
        self.costs[(self.trigger, self.release)] += 1
        self.frame = (*self.frame[:4], tuple(self.costs.items()), self.frame[-1])
        with self.assertRaisesRegex(ValueError, "differs from the original graph"):
            self.audit()
