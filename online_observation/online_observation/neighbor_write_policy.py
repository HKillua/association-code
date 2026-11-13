"""Evaluate G witness write rules without changing nomination or publication logic."""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import replace

from .expansion_confirmation import ConfirmationState, ExpansionFrame
from .expansion_evidence import HistoricalGeometryWitnesses, WitnessEvidence, _OffsetSample
from .neighbor_witness import NeighborEvidence, evidence_signal
from .nemot_replay import Edge


WRITE_POLICY_ARMS = ("conflict_veto", "positive_only")


class NeighborWritePolicy:
    """Filter current G writes, with the legacy reader and confirmation latch."""

    def __init__(self, mode: str):
        if mode not in WRITE_POLICY_ARMS:
            raise ValueError(f"unknown neighbor witness write policy: {mode}")
        self.mode = mode
        self.witnesses = HistoricalGeometryWitnesses()
        self.confirmation = ConfirmationState()

    def _allows(self, evidence: NeighborEvidence) -> bool:
        signal = evidence_signal(evidence.consensus, evidence.strict_offset)
        if self.mode == "conflict_veto":
            return signal != "conflict"
        return signal == "compatible"

    def decide(
        self,
        expansion: ExpansionFrame,
        ring: Mapping[Edge, NeighborEvidence],
        new_samples: Iterable[tuple[Edge, _OffsetSample]],
        writes: Mapping[Edge, NeighborEvidence],
    ) -> tuple[tuple[Edge, ...], dict[Edge, WitnessEvidence], tuple[Edge, ...]]:
        """Read earlier evidence, publish C decisions, then apply this frame's writes."""
        history = {}
        nominees = []
        for edge in expansion.edges:
            if not edge.wide:
                continue
            readout = self.witnesses.lookup(edge.pair, expansion.stamp, ring[edge.pair].strict_offset)
            history[edge.pair] = readout
            features = {**edge.features, "witness_residual_m": readout.residual_m, "witness_count": readout.count}
            nominees.append(replace(edge, features=features, witnesses=readout.witnesses,
                                    witness_offsets=readout.offsets))
        selected = self.confirmation.decide(replace(expansion, edges=tuple(nominees)))["witness_confirm"]

        recorded = []
        for pair, sample in new_samples:
            if sample.observed_at != expansion.stamp:
                raise ValueError("witness write samples must come from the current decision")
            if self._allows(writes[pair]) and self.witnesses.record_sample(pair, sample):
                recorded.append(pair)
        return selected, history, tuple(recorded)
