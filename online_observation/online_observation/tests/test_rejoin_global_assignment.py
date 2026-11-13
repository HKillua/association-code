import unittest

from online_observation.continuous_tracker import LocalTrack
from online_observation.hypothesis_gap_diagnostic import Switch
from online_observation.rejoin_global_assignment import score_global


class GlobalRejoinAssignmentTests(unittest.TestCase):
    def setUp(self):
        self.trigger, self.sibling, self.extra = (("A", str(index)) for index in range(1, 4))
        self.retain, self.release, self.other = (("B", str(index)) for index in range(1, 4))
        self.switch = Switch(((self.trigger, self.release), (self.sibling, self.retain)),
                             ((self.trigger, self.retain), (self.sibling, self.release)))
        self.points = {self.trigger: (0., 0.)}
        self.priors = {key: LocalTrack(19, (value, 0.), (value, 0.)) for key, value in (
            (self.trigger, 10.), (self.retain, 0.), (self.release, 10.), (self.other, 9.))}
        self.costs = dict(zip(self.switch.edges, (100., 10., 0., 0.), strict=True))
        self.focal = (self.switch.retain[1],)

    def solve(self, selected):
        fresh = tuple(sorted({key for edge in self.costs for key in edge}))
        frame = (20, tuple(self.points), (), fresh, tuple(self.costs.items()), selected)
        return score_global(self.switch, self.focal, self.trigger, frame, self.priors, self.points, {})

    def test_closed_four_endpoint_group_reverses_and_loses_rejoin(self):
        result = self.solve(self.switch.retain)
        self.assertEqual(result["actual"]["pattern"], "retain")
        self.assertEqual(result["masked"]["pattern"], "release")
        self.assertFalse(result["masked"]["rejoined"])
        self.assertEqual(result["selected_masked"], tuple(sorted(self.switch.release)))
        self.assertEqual(result["graph"]["actual_margin_m2"], -110)
        self.assertEqual(result["graph"]["masked_margin_m2"], 90)
        self.assertEqual(result["graph"]["trigger_incident_edges"], 2)
        self.assertEqual(result["affected"]["changed_outside_endpoints"], 0)
        self.assertEqual(result["affected"]["removed_edges"], 2)

    def test_local_reversal_can_keep_one_original_focal_in_full_assignment(self):
        self.costs.update({(self.trigger, self.other): 81., (self.extra, self.retain): 0.,
                           (self.extra, self.other): 40.})
        result = self.solve((*self.switch.retain, (self.extra, self.other)))
        self.assertEqual(result["local_masked"]["pattern"], "release")
        self.assertEqual(result["masked"]["pattern"], "partial_retain")
        self.assertTrue(result["masked"]["rejoined"])
        self.assertEqual(result["fixed_mask"]["pattern"], "release")
        self.assertFalse(result["fixed_mask"]["rejoined"])
        self.assertEqual(result["masked"]["external_edges"], 2)
        self.assertEqual(result["affected"]["removed_fully_external_edges"], 1)
        self.assertEqual(result["affected"]["changed_outside_endpoints"], 2)
        self.assertEqual(result["component_keys"], tuple(sorted((*self.switch.keys, self.extra, self.other))))

    def test_original_null_and_same_batch_peer_coordinates_are_kept(self):
        removed = ("B", "4")
        self.points[self.other] = (4000., 0.)
        self.priors[removed] = LocalTrack(19, (-3999., 0.), (-3999., 0.))
        self.costs.update({(self.trigger, self.other): 16_000_000.,
                           (self.trigger, removed): 15_992_001.})
        result = self.solve(self.switch.retain)
        self.assertEqual(result["graph"]["incident_null_gains"], 1)
        self.assertEqual(result["graph"]["incident_null_losses"], 1)
        self.assertEqual(result["graph"]["candidate_edges"], 6)
        self.assertEqual(result["graph"]["trigger_incident_costs_changed"], 4)

    def test_rejects_a_nonreproducing_factual_assignment(self):
        with self.assertRaisesRegex(ValueError, "original rejoin assignment"):
            self.solve(self.switch.release)
