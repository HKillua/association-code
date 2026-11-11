import unittest

from online_observation.expansion_confirmation import ExpansionEdge, ExpansionFrame
from online_observation.expansion_evidence import HistoricalGeometryWitnesses
from online_observation.neighbor_witness import NeighborEvidence, NeighborVote, summarize_votes
from online_observation.neighbor_write_policy import NeighborWritePolicy


def _pair(name):
    return ("a", name), ("b", name)


def _evidence(strict=None, bridge=None, support=False):
    offset = strict if strict is not None else bridge
    votes = tuple(NeighborVote(_pair(name), (100, 0)) for name in ("one", "two")) if support else ()
    return NeighborEvidence(0, 0, strict, bridge, (0, 0), (0, 0), (0, 0), votes, summarize_votes(votes, offset))


def _frame(stamp, target):
    edge = ExpansionEdge(target, True, dict(witness_residual_m=800, repeat_both=0, episode_age_s=0), ())
    return ExpansionFrame(stamp, (), (), (edge,))


def _histories(pairs):
    return {key: [(stamp, (stamp + (100 if key[0] == "b" else 0), 0)) for stamp in (0, 10, 20)]
            for pair in pairs for key in pair}


class NeighborWritePolicyTests(unittest.TestCase):
    def test_eligible_writes_cannot_affect_same_frame_and_conflict_keeps_history(self):
        witness, target = _pair("witness"), _pair("target")
        raw, policy = HistoricalGeometryWitnesses(), NeighborWritePolicy("conflict_veto")
        histories = _histories((witness,))
        ring = {target: _evidence(strict=(100, 0))}
        unknown = {witness: _evidence(strict=(100, 0))}

        for stamp in (0, 10):
            samples = raw.observe(stamp, (witness,), histories)
            selected, history, writes = policy.decide(_frame(stamp, target), ring, samples, unknown)
            self.assertEqual((selected, history[target].count, writes), ((), 0, (witness,)))

        samples = raw.observe(20, (witness,), histories)
        conflict = {witness: _evidence(strict=(800, 0), support=True)}
        selected, history, writes = policy.decide(_frame(20, target), ring, samples, conflict)
        self.assertEqual((selected, history[target].offsets, writes), ((target,), ((witness, (100, 0)),), ()))
        _, history, _ = policy.decide(_frame(21, target), ring, (), {})
        self.assertEqual(history[target].witnesses, (witness,))

    def test_positive_only_and_veto_treat_bridge_as_unknown(self):
        positive, conflict, bridge, unknown = (_pair(name) for name in ("positive", "conflict", "bridge", "unknown"))
        pairs = positive, conflict, bridge, unknown
        raw = HistoricalGeometryWitnesses()
        samples = raw.observe(0, pairs, _histories(pairs))
        writes = {
            positive: _evidence(strict=(100, 0), support=True),
            conflict: _evidence(strict=(800, 0), support=True),
            bridge: _evidence(bridge=(800, 0), support=True),
            unknown: _evidence(strict=(100, 0)),
        }
        frame = ExpansionFrame(0, (), (), ())

        veto = NeighborWritePolicy("conflict_veto")
        trusted = NeighborWritePolicy("positive_only")
        self.assertEqual(veto.decide(frame, {}, samples, writes), ((), {}, (positive, bridge, unknown)))
        self.assertEqual(trusted.decide(frame, {}, samples, writes), ((), {}, (positive,)))


if __name__ == "__main__":
    unittest.main()
