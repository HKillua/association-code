"""Explicit native-time adaptation of the TMTAFF-style association model.

Observed states carry continuous physical-time encodings, and every decoder
position is conditioned on the next requested time. The default keeps the old
four-state projection; an explicit circular-heading variant adds a fifth input
column while preserving the original time encoding and other initialization.
This is a disclosed adaptation, not author code. The caller scales seconds to
tau using the declared scene origin/time unit.
"""
from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import replace

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from association.models.tmtaff import AssociationModel, ModelConfig, _causal_mask
from association.data.physical_units import normalizer_speed_policy


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

    def __init__(self, config: ModelConfig, branch: str = 'ce', *, max_observations: int,
                 normalizer: Mapping | None = None):
        if isinstance(max_observations, bool) or not isinstance(max_observations, int) or max_observations < 2:
            raise ValueError('max_observations must be an explicit integer >=2')
        if config.input_dim != 4:
            raise ValueError('native-time inputs require exactly four state features')
        heading_statistics = (self._normalizer_heading(normalizer)
                              if config.heading_representation == 'circular_sincos' else None)
        # Construct exactly the old model first: changing the projection width
        # before this point would also change every later random initialization.
        super().__init__(replace(config, heading_representation='linear_zscore',
                                 completion_policy='learned'), branch)
        self.config = config
        self.max_observations = max_observations
        self.speed_unit_policy = normalizer_speed_policy(normalizer)
        if self.speed_unit_policy is not None:
            self.register_buffer('speed_unit_contract', torch.tensor(1, dtype=torch.int64))
        if heading_statistics is not None:
            self.register_buffer('heading_mean_std', heading_statistics)
            original = self.input_embedding
            # The added frozen column must not advance the global random stream
            # used by later modules or training dropout.
            with torch.random.fork_rng(devices=[]):
                expanded = nn.Linear(5, config.d_model, bias=False)
                with torch.no_grad():
                    expanded.weight[:, :4].copy_(original.weight)
                    bound = math.sqrt(6.0 / (config.d_model + 4))
                    expanded.weight[:, 4].uniform_(-bound, bound)
            self.input_embedding = expanded

    @staticmethod
    def _normalizer_heading(normalizer: Mapping | None) -> Tensor:
        if not isinstance(normalizer, Mapping):
            raise ValueError('circular heading requires the frozen training normalizer')
        try:
            mean = torch.as_tensor(normalizer['mean'], dtype=torch.float64)
            std = torch.as_tensor(normalizer['std'], dtype=torch.float64)
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError('normalizer requires four finite means and positive scales') from error
        if (mean.shape != (4,) or std.shape != (4,)
                or not torch.isfinite(mean).all() or not torch.isfinite(std).all()
                or (std <= 0).any()):
            raise ValueError('normalizer requires four finite means and positive scales')
        return torch.stack((mean[3], std[3]))

    def validate_normalizer(self, normalizer: Mapping | None) -> None:
        """Reject a different physical interpretation of saved circular weights."""
        if normalizer_speed_policy(normalizer) != self.speed_unit_policy:
            raise ValueError('normalizer differs from model speed unit contract')
        if self.config.heading_representation == 'linear_zscore':
            return
        expected = self._normalizer_heading(normalizer).to(self.heading_mean_std)
        if not torch.equal(expected, self.heading_mean_std):
            raise ValueError('normalizer differs from saved circular heading mean/std')

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        incoming = state_dict.get(prefix + 'speed_unit_contract')
        if self.speed_unit_policy is None:
            if incoming is not None:
                raise ValueError('physical-unit weights cannot be loaded into a legacy-unit model')
        elif (not isinstance(incoming, Tensor) or incoming.shape != torch.Size([])
              or incoming.dtype != torch.int64 or incoming.item() != 1):
            raise ValueError('checkpoint is missing or contradicts the speed unit contract')
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                     missing_keys, unexpected_keys, error_msgs)

    def set_completion_trainable(self, trainable: bool, include_embedding: bool = True):
        if trainable and self.config.completion_policy == 'state_hold':
            raise ValueError('state_hold requires frozen completion, not TCM or joint training')
        return super().set_completion_trainable(trainable, include_embedding=include_embedding)

    def embed_states(self, states: Tensor, times: Tensor) -> Tensor:
        """Embed normalized physical states at explicit times in one shared space."""
        if states.shape[-1:] != (4,) or states.shape[:-1] != times.shape:
            raise ValueError('state inputs require four features and matching times')
        time_encoding = continuous_time_encoding(times, 4)
        if self.config.heading_representation == 'circular_sincos':
            mean, std = self.heading_mean_std.to(states).unbind()
            theta = torch.deg2rad(states[..., 3] * std + mean)
            states = torch.cat((states[..., :3], theta.sin()[..., None],
                                theta.cos()[..., None]), dim=-1)
            # Retain every original PE4 frequency and its projection column;
            # the added cosine state receives no extra time signal.
            time_encoding = F.pad(time_encoding, (0, 1))
        return F.relu(self.input_embedding(states + time_encoding))

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
        embedded = self.embed_states(states, times)
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
        if self.config.completion_policy != 'learned':
            raise ValueError('state_hold is not a trainable TCM objective')
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

    def complete_state_hold(self, values: Tensor, observed: Tensor) -> Tensor:
        """Use the same inside interpolation, re-embed endpoint state at query tau.

        Neither padding nor outside query state contents are read. The control
        never runs TCM or consumes random numbers. The old native input embedding
        path is reused so inside-support features remain exactly unchanged.
        """
        embedded = self.embed_inputs(values, observed)
        m = self.max_observations
        counts = observed[:, :m].sum(1)
        query_times = values[:, m:, 4]
        last_indices = (counts - 1)[:, None]
        last_times = values[:, :m, 4].gather(1, last_indices)
        before = query_times < values[:, :1, 4]
        after = query_times > last_times
        first_states = values[:, :1, :4]
        last_states = values[:, :m, :4].gather(1, last_indices[..., None].expand(-1, 1, 4))
        held_states = torch.where(before[..., None], first_states,
                                  torch.where(after[..., None], last_states, values[:, m:, :4]))
        held = self.embed_states(held_states, query_times)
        return torch.where((before | after)[..., None], held, embedded[:, m:])

    def complete(self, values: Tensor, observed: Tensor) -> Tensor:
        """Return [N,L,d] at physical queries, including tracks with zero query support."""
        if self.config.completion_policy == 'state_hold':
            return self.complete_state_hold(values, observed)
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
