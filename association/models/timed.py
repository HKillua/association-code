"""Explicit native-time adaptation of the TMTAFF-style association model.

The learned layers and parameter shapes are unchanged. Observed states carry
continuous physical-time encodings, and every decoder position is conditioned
on the next requested time. This is a disclosed adaptation, not author code.
The caller scales seconds to tau using the declared scene origin/time unit.
"""
from __future__ import annotations

import math

import torch
from torch import Tensor
from torch.nn import functional as F

from association.models.tmtaff import AssociationModel, ModelConfig, _causal_mask


def continuous_time_encoding(times: Tensor, dimension: int) -> Tensor:
    """Sinusoidal encoding at continuous tau; no learned or fitted parameters."""
    frequencies = torch.exp(torch.arange(0, dimension, 2, device=times.device, dtype=times.dtype)
                            * (-math.log(10000.0) / dimension))
    angles = times[..., None] * frequencies
    result = times.new_empty((*times.shape, dimension))
    result[..., 0::2] = torch.sin(angles)
    result[..., 1::2] = torch.cos(angles[..., :dimension // 2])
    return result


class TimedAssociationModel(AssociationModel):
    """One native history and one common query grid, using the existing TCM.

    Packed values have shape [N, M+L, 5]: M native rows followed by L query
    rows, each containing normalized four-state values and physical tau.
    ``observed`` is a true prefix of length >=2 in the native part only.
    Query values are observation-derived interpolation inside native support;
    outside support their four-state contents are ignored, including NaNs.
    Padding is ignored. Finite, increasing query times are always required.

    The inherited ``positions`` buffer is retained for state-dict shape
    compatibility but is unused. Existing weights must not be represented as
    already trained for this variant; callers record ``input_adapter`` and M.
    """
    input_adapter = 'native_time_v1'

    def __init__(self, config: ModelConfig, branch: str = 'ce', *, max_observations: int):
        if isinstance(max_observations, bool) or not isinstance(max_observations, int) or max_observations < 2:
            raise ValueError('max_observations must be an explicit integer >=2')
        if config.input_dim != 4:
            raise ValueError('native-time inputs require exactly four state features')
        super().__init__(config, branch)
        self.max_observations = max_observations

    def _validate_inputs(self, values: Tensor, observed: Tensor) -> Tensor:
        width = self.max_observations + self.config.length
        if values.ndim != 3 or tuple(values.shape[1:]) != (width, 5):
            raise ValueError(f'packed values must have shape [N, {width}, 5]')
        if observed.shape != values.shape[:2] or observed.dtype != torch.bool:
            raise ValueError('observed must be a boolean [N,M+L] mask')
        if values.device != observed.device or not values.is_floating_point():
            raise ValueError('floating values and observed must use the same device')
        m = self.max_observations
        native_mask = observed[:, :m]
        counts = native_mask.sum(1)
        if (counts < 2).any() or observed[:, m:].any():
            raise ValueError('native history requires >=2 observations; query mask must be false')
        prefix = torch.arange(m, device=values.device)[None, :] < counts[:, None]
        if not torch.equal(native_mask, prefix):
            raise ValueError('native observations must form a prefix without internal gaps')
        if not torch.isfinite(values[:, :m][native_mask]).all():
            raise ValueError('native observations and times must be finite')
        native_times = values[:, :m, 4]
        consecutive = native_mask[:, 1:]
        if (native_times[:, 1:][consecutive] <= native_times[:, :-1][consecutive]).any():
            raise ValueError('native times must be strictly increasing')
        query_times = values[:, m:, 4]
        if not torch.isfinite(query_times).all() or (query_times[:, 1:] <= query_times[:, :-1]).any():
            raise ValueError('query times must be finite and strictly increasing')
        last = native_times.gather(1, (counts - 1)[:, None])
        inside = (query_times >= native_times[:, :1]) & (query_times <= last)
        if not torch.isfinite(values[:, m:, :4][inside]).all():
            raise ValueError('inside-support query states must be finite observation-derived inputs')
        return observed

    def embed_inputs(self, values: Tensor, observed: Tensor) -> Tensor:
        """Embed valid native/query observations; all padding/outside states are ignored."""
        self._validate_inputs(values, observed)
        m = self.max_observations
        counts = observed[:, :m].sum(1)
        query_times = values[:, m:, 4]
        last = values[:, :m, 4].gather(1, (counts - 1)[:, None])
        inside = (query_times >= values[:, :1, 4]) & (query_times <= last)
        valid = torch.cat((observed[:, :m], inside), dim=1)
        states = torch.where(valid[..., None], values[..., :4], torch.zeros_like(values[..., :4]))
        times = torch.where(valid, values[..., 4], torch.zeros_like(values[..., 4]))
        embedded = F.relu(self.input_embedding(states + continuous_time_encoding(times, 4)))
        return torch.where(valid[..., None], embedded, torch.zeros_like(embedded))

    def _teacher_predictions(self, history: Tensor, next_times: Tensor) -> Tensor:
        """Each history slot predicts its explicit next time using only its prefix."""
        memory = self.tcm_encoder(history, mask=_causal_mask(history.shape[1], history.device))
        conditioned = history + continuous_time_encoding(next_times, self.config.d_model)
        return super()._decode_causal_history(conditioned, memory)

    def _extrapolate(self, sequence: Tensor, native_times: Tensor, query_times: Tensor) -> Tensor:
        """Autoregressive endpoint queries; fixed causal native memory in either direction."""
        if query_times.shape[1] == 0:
            return sequence[:, :0]
        memory = self.tcm_encoder(sequence, mask=_causal_mask(sequence.shape[1], sequence.device))
        history, previous_next_times = sequence, native_times[:, 1:]
        predictions = []
        for index in range(query_times.shape[1]):
            next_times = torch.cat((previous_next_times, query_times[:, index:index + 1]), dim=1)
            conditioned = history + continuous_time_encoding(next_times, self.config.d_model)
            prediction = super()._decode_causal_history(conditioned, memory)[:, -1:]
            predictions.append(prediction)
            history = torch.cat((history, prediction), dim=1)
            previous_next_times = next_times
        return torch.cat(predictions, dim=1)

    def completion_loss(self, values: Tensor, observed: Tensor) -> Tensor:
        """Observed native next-embedding MSE, both directions, detached targets.

        Exact denominator: sum_tracks 2*(native_count-1)*d_model. Query states,
        query spacing, padding and reference targets never enter this loss.
        """
        embedded = self.embed_inputs(values, observed)
        counts = observed[:, :self.max_observations].sum(1)
        total = embedded.sum() * 0 + next(self.tcm_encoder.parameters()).reshape(-1)[0] * 0
        elements = 0
        for count in counts.unique().detach().cpu().tolist():
            members = torch.nonzero(counts == count, as_tuple=True)[0]
            original = embedded[members, :count]
            times = values[members, :count, 4]
            sequence = torch.cat((original, original.flip(1)), dim=0)
            next_times = torch.cat((times[:, 1:], times.flip(1)[:, 1:]), dim=0)
            prediction = self._teacher_predictions(sequence[:, :-1], next_times)
            targets = sequence[:, 1:].detach()
            total = total + F.mse_loss(prediction, targets, reduction='sum')
            elements += targets.numel()
        return total / max(elements, 1)

    def complete(self, values: Tensor, observed: Tensor) -> Tensor:
        """Return [N,L,d] at physical queries, including tracks with zero query support."""
        embedded = self.embed_inputs(values, observed)
        m = self.max_observations
        completed = embedded[:, m:].clone()
        counts = observed[:, :m].sum(1)
        native_times, query_times = values[:, :m, 4], values[:, m:, 4]
        last = native_times.gather(1, (counts - 1)[:, None])
        before = (query_times < native_times[:, :1]).sum(1)
        end = (query_times <= last).sum(1)
        groups = {}
        for index, key in enumerate(torch.stack((counts, before, end), dim=1).detach().cpu().tolist()):
            groups.setdefault(tuple(key), []).append(index)
        for (count, start, stop), rows in groups.items():
            members = torch.tensor(rows, dtype=torch.long, device=values.device)
            sequence, times = embedded[members, :count], native_times[members, :count]
            if start:
                predicted = self._extrapolate(sequence.flip(1), times.flip(1), query_times[members, :start].flip(1))
                completed[members, :start] = predicted.flip(1)
            if stop < self.config.length:
                completed[members, stop:] = self._extrapolate(sequence, times, query_times[members, stop:])
        return completed
