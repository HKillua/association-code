import gzip
from pathlib import Path
import pickle
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

from online_observation.interval_assignment import PAIR_NULL_COST
from online_observation.support_gap_audit import audit_support


A1, A2 = ("A", "1"), ("A", "2")
B1, B2 = ("B", "1"), ("B", "2")
ALL = A1, A2, B1, B2
P = ((A1, B1), (A2, B2))
Q = ((A1, B2), (A2, B1))
COSTS = dict.fromkeys((*P, *Q), 1.)


def _memberships(edges):
    return tuple((str(index), edge) for index, edge in enumerate(edges))


class SupportGapAuditTests(unittest.TestCase):
    def _audit(self, frames, stop, end):
        native = [(stamp, keys) for stamp, keys, _, _ in frames]
        result = SimpleNamespace(native_reports=native, runs={
            "kf": [], "interval_swap": [(0., _memberships(Q))],
        }, fit={}, diagnostics={
            "fixed_candidate_edge_events": sum(len(costs) for _, _, costs, _ in frames),
            "reports": sum(len(keys) for _, keys, _, _ in frames),
        })
        gap = dict(start=10., stop=stop, end=end, focal=(Q[0],),
                   switch=dict(release=P, retain=Q))
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "graph.pkl.gz"
            latest = {}
            with gzip.open(path, "wb") as stream:
                for stamp, keys, costs, kf in frames:
                    born = tuple(key for key in keys if key not in latest)
                    latest.update(dict.fromkeys(keys, stamp))
                    fresh = tuple(sorted(key for key, last in latest.items() if stamp-last <= 60.))
                    pickle.dump((stamp, keys, born, fresh, tuple(costs.items()), kf), stream)
                pickle.dump(result, stream)
            return audit_support(result, [gap], path)[0]

    def test_common_window_then_original_graph_exit(self):
        changed = dict(COSTS)
        del changed[Q[0]]
        changed[Q[1]] = PAIR_NULL_COST
        audit = self._audit([
            (0., ALL, COSTS, Q), (10., ALL, COSTS, P), (25., (A1, A2), COSTS, P),
            (30., (B1, B2), COSTS, P), (35., (A1, A2), changed, P),
            (40., (B1, B2), COSTS, P),
        ], 35., "graph_loss")
        within = audit["first"]["within"]
        self.assertEqual(within["comparable_bracket"]["stamp"], 30.)
        self.assertEqual(within["full_refresh"]["common_right"], 25.)
        self.assertEqual(within["original_refreshed_bracket"]["stamp"], 30.)
        self.assertIsNone(within["after20_refreshed_bracket"])
        boundary = audit["termination"]["native_event"]
        self.assertEqual((boundary["missing_edges"], boundary["at_or_above_null"]), (1, 1))
        self.assertTrue(boundary["fresh"])
        self.assertEqual(audit["first"]["beyond"]["legal_refreshed_bracket"]["stamp"], 40.)

    def test_expiry_between_batches_survives_post_expiry_refresh(self):
        audit = self._audit([
            (0., ALL, COSTS, Q), (10., (A1, A2), COSTS, P),
            (65., ALL, COSTS, P), (70., ALL, COSTS, P),
        ], 60., "expiry")
        self.assertEqual(audit["first_expiry"], 60.)
        self.assertIsNone(audit["termination"]["native_event"])
        after = audit["first"]["beyond"]["full_refresh"]
        self.assertTrue(after["fresh"])
        self.assertFalse(after["continuous_fresh"])
        self.assertFalse(after["prior_fresh"])
        self.assertIsNone(audit["first"]["beyond"]["original_refreshed_bracket"])

    def test_kf_rejoin_can_coincide_with_first_full_refresh(self):
        audit = self._audit([
            (0., ALL, COSTS, Q), (10., ALL, COSTS, P), (25., (A1, A2), COSTS, P),
            (30., (B1, B2), COSTS, Q),
        ], 30., "kf_rejoin")
        self.assertIsNone(audit["first"]["within"]["full_refresh"])
        self.assertEqual(audit["first"]["boundary"]["legal_refreshed_bracket"]["stamp"], 30.)
        self.assertIsNone(audit["first"]["boundary"]["original_refreshed_bracket"])
        self.assertTrue(audit["termination"]["native_event"]["kf_rejoin"])
