import unittest
from types import SimpleNamespace

from online_observation.hypothesis_gap_diagnostic import Switch
from online_observation.rejoin_persistence import RejoinPersistence


A = ("a", "1")
B = ("a", "2")
C = ("b", "1")
D = ("b", "2")
SWITCH = Switch(release=((A, D), (B, C)), retain=((A, C), (B, D)))


def _groups(edges):
    return tuple(enumerate(edges, 1))


class RejoinPersistenceTests(unittest.TestCase):
    def test_excludes_rejoin_batch_and_uses_both_edges(self):
        reports = [(stamp, (A, B, C, D)) for stamp in (0., 4., 8., 10.)]
        result = SimpleNamespace(native_reports=reports, runs={
            "kf": [(0., _groups(SWITCH.retain)), (4., _groups(SWITCH.release[:1])),
                   (8., _groups(SWITCH.release)), (10., _groups(SWITCH.retain))],
            "interval_swap": [(0., _groups(SWITCH.retain))],
        })
        history = RejoinPersistence(result)
        self.assertEqual(history.at(SWITCH, 10.), dict(kf_release_s=2., kf_retain_s=0.,
                                                     i22_release_s=0., i22_retain_s=10.))
        self.assertEqual(history.duration("kf", SWITCH.retain, 4.), 4.)
        self.assertEqual(history.duration("kf", SWITCH.release, 8.), 0.)

    def test_expiry_and_renewal_do_not_bridge_unpublished_time(self):
        reports = [(stamp, (A, B, C, D)) for stamp in (0., 80., 82.)]
        result = SimpleNamespace(native_reports=reports, runs={
            "kf": [(0., _groups(SWITCH.retain))],
            "interval_swap": [(0., _groups(SWITCH.retain))],
        })
        history = RejoinPersistence(result)
        self.assertEqual(history.duration("kf", SWITCH.retain, 60.), 60.)
        self.assertEqual(history.duration("kf", SWITCH.retain, 80.), 0.)
        self.assertEqual(history.duration("kf", SWITCH.retain, 82.), 2.)
