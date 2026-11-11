import json
from types import SimpleNamespace
import unittest

from online_observation.neighbor_audit import pool_neighbor_audit, score_neighbor_audit
from online_observation.neighbor_witness import NeighborEvidence, NeighborVote, summarize_votes


def _evidence(offset, votes=()):
    votes = tuple(NeighborVote(pair, xy) for pair, xy in votes)
    n = len(votes)
    return NeighborEvidence(10, 10, offset, None, (n, n), (n, n), (n, n), votes,
                            summarize_votes(votes, offset))


def _frame(stamp, updated, fresh, born=(), ring=(), wide=(), geometry=(), extra=()):
    return SimpleNamespace(stamp=stamp, updated=updated, fresh=fresh, born=born,
                           evidence=tuple(SimpleNamespace(pair=pair) for pair in ring),
                           additions={"wide": wide, "test": extra},
                           geometry_edges=geometry, geometry_links=geometry)


def _expansion(stamp, edges=()):
    return SimpleNamespace(stamp=stamp, edges=edges)


def _edge(pair, legacy_offsets=(), residual=None):
    return SimpleNamespace(pair=pair, witness_offsets=legacy_offsets,
                           witnesses=tuple(witness for witness, _ in legacy_offsets),
                           features={"witness_residual_m": residual})


def _probe(ring=None, writes=None, parallel=None):
    return SimpleNamespace(ring=ring or {}, writes=writes or {}, parallel=parallel or {})


class NeighborAuditTests(unittest.TestCase):
    def test_birth_uses_seen_prefix_and_separates_query_from_both_endpoints(self):
        left = ("A", "q")
        rival_left = ("A", "r")
        q, r = ("B", "q"), ("B", "r")
        true, false = (left, q), (left, r)
        labels = {key: key[1] for key in (left, rival_left, q, r)}
        frames = (
            _frame(0, (q, r), (q, r), born=(q, r)),
            _frame(1, (left,), (left, q, r), born=(left,), ring=(true, false), wide=(false,), extra=(true,)),
            _frame(2, (rival_left,), (rival_left, left, q, r), born=(rival_left,), wide=(false,)),
        )
        expansions = (_expansion(0), _expansion(1, (_edge(true), _edge(false))), _expansion(2))
        probes = (_probe(), _probe({true: _evidence((0, 0)), false: _evidence(None)},
                                  parallel={tuple(sorted((q, r))): True}), _probe())
        result = score_neighbor_audit(frames, expansions, probes, labels, arms=("geometry", "wide", "test"))
        sparse = result["birth"]["sparse"]
        self.assertEqual(sparse["queries"]["true_open_ring"], 1)
        self.assertEqual(sparse["queries"]["unseen_truth"], 2)
        self.assertEqual(sparse["comparison"]["parallel_comparisons"], 1)
        self.assertEqual(sparse["outcomes"]["wide"]["false_unseen_truth"], 1)
        self.assertEqual(sparse["outcomes"]["wide"]["own_unseen_wrong"], 0)
        self.assertEqual(sparse["paired"]["wide"]["test"]["wrong_to_correct"], 1)
        self.assertEqual(result["selections"]["wide"]["false_both_seen"], 1)
        self.assertEqual(result["selections"]["wide"]["distinct_false_unseen_truth"], 1)
        pooled = pool_neighbor_audit([json.loads(json.dumps(result)), result])
        self.assertEqual(pooled["birth"]["sparse"]["queries"]["n"], 8)

    def test_clean_counterfactual_uses_independent_objects_and_same_legacy_thresholds(self):
        target = (("A", "target"), ("B", "target"))
        known = tuple((("A", name), ("B", name)) for name in ("a", "b", "c"))
        false = tuple((known[i][0], known[(i + 1) % 3][1]) for i in range(3))
        labels = {key: key[1] for pair in (target, *known) for key in pair}
        duplicates = (("A", "fragment"), ("B", "fragment"))
        labels.update({key: "target" for key in duplicates})
        votes = (*((pair, (1500, 0)) for pair in false),
                 (known[0], (0, 0)), (known[1], (0, 0)), (duplicates, (2400, 0)))
        legacy = ((known[0], (0, 0)), (false[0], (500, 0)))
        frame = _frame(10, tuple(labels), tuple(labels), born=target[:1], ring=(target,), wide=(target,))
        expansion = _expansion(10, (_edge(target, legacy, residual=450),))
        result = score_neighbor_audit((frame,), (expansion,), (_probe({target: _evidence((-200, 0), votes)}),), labels)
        true = result["birth"]["all"]["open_edges"]["true"]
        self.assertEqual(true["best_witness_all_false"], 1)
        self.assertEqual(true["signal_conflict"], 1)
        self.assertEqual(true["clean_compatible"], 1)
        self.assertEqual(true["removable_pollution"], 1)
        self.assertEqual(true["legacy_removable_pollution"], 1)

        evidence = _evidence((1700, 0), votes)
        result = score_neighbor_audit((frame,), (expansion,), (_probe({target: evidence}),), labels)
        self.assertEqual(result["birth"]["all"]["open_edges"]["true"]["strict_clean_conflict"], 1)
