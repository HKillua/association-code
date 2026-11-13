import unittest

from online_observation.expansion_confirmation import ConfirmationState, ExpansionEdge, ExpansionFrame


PAIR = (("a", "one"), ("b", "two"))


def frame(age, *, residual=None, repeat=False, nominated=True):
    edge = ExpansionEdge(PAIR, nominated, dict(episode_age_s=age, repeat_both=repeat,
                                               witness_residual_m=residual), ())
    return ExpansionFrame(age, (), PAIR, (edge,))


class ConfirmationTest(unittest.TestCase):
    def test_evidence_can_confirm_early_and_conflict_only_blocks_waiting_arms(self):
        state = ConfirmationState()
        at_start = state.decide(frame(0, residual=400))
        self.assertEqual(at_start, dict(dual_report=(), witness_veto=(), witness_confirm=(PAIR,)))
        conflicting = state.decide(frame(10, residual=401, repeat=True))
        self.assertEqual(conflicting, dict(dual_report=(PAIR,), witness_veto=(), witness_confirm=(PAIR,)))
        missing = state.decide(frame(60, repeat=True))
        self.assertTrue(all(pairs == (PAIR,) for pairs in missing.values()))
        late = state.decide(frame(61, residual=900, repeat=True))
        self.assertEqual(late, missing)

    def test_expired_nomination_stays_empty_until_a_new_nomination(self):
        state = ConfirmationState()
        self.assertTrue(all(not pairs for pairs in state.decide(frame(0)).values()))
        self.assertTrue(all(not pairs for pairs in state.decide(frame(61, residual=0, repeat=True)).values()))
        self.assertTrue(all(not pairs for pairs in state.decide(frame(62, nominated=False)).values()))
        new_nomination = state.decide(frame(0, residual=399))
        self.assertEqual(new_nomination, dict(dual_report=(), witness_veto=(), witness_confirm=(PAIR,)))
        state.decide(frame(1, nominated=False))
        self.assertTrue(all(not pairs for pairs in state.decide(frame(0)).values()))
