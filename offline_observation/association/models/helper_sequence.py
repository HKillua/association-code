"""Matched helper representations for the fixed sequence position backbone.

The branch sees only the historical 27 observable pair features. Its weights
combine latent evidence, not physical states or calibrated helper reliability.
Original sequence and gate modules intentionally remain unchanged.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from association.models.evidence_fusion import fusion_inputs
from association.models.sequence_position import (
    SequencePosition, sequence_inputs, tangent_position,
)

SCOPE = 'helper-sequence-position-v1'
ROUTING_SEMANTICS = 'warm095-allquery-tangent-residual-helper-embedding-v1'
RAW_FEATURE_WIDTH = 76
HELPER_FEATURE_WIDTH = 27
BRANCH_WIDTH = 16
FEATURE_WIDTH = RAW_FEATURE_WIDTH + BRANCH_WIDTH
AGGREGATIONS = ('post_pool', 'pre_pool', 'weighted_pool')


def helper_sequence_inputs(motion, members, *, device='cpu'):
    """Keep historical physical/features input and append visible helper rows.

    Original node order fixes the anchor; helper node indices are sorted before
    either input builder performs reductions. Neither identity nor truth enters.
    """
    members = np.asarray(members)
    # The existing builder owns membership/physical-data validation. Call it
    # before conversion so invalid float members cannot be silently truncated.
    base, features, mean = sequence_inputs(motion, members, device=device)
    members = np.r_[members[:1], np.sort(members[1:])].astype(np.int64)
    _, helper_features, _ = fusion_inputs(motion, members, device=device)
    helper_support = torch.as_tensor(np.asarray(motion['support'])[members[1:]],
                                    dtype=torch.bool, device=device)
    return base, features, mean, helper_features, helper_support


class HelperSequencePosition(SequencePosition):
    """A/B matched capacity; C adds a bias-free 16-parameter latent scorer."""

    def __init__(self, *, aggregation='post_pool', backbone='transformer',
                 hidden_size=64, num_layers=2, initial_gate=.95):
        if aggregation not in AGGREGATIONS or backbone != 'transformer':
            raise ValueError('fixed transformer and a declared helper aggregation required')
        super().__init__(backbone=backbone, hidden_size=hidden_size,
                         num_layers=num_layers, initial_gate=initial_gate)
        self.aggregation = aggregation
        self.routing_semantics = ROUTING_SEMANTICS
        old_projection = self.input_projection[0]
        self.input_projection[0] = nn.Linear(FEATURE_WIDTH, hidden_size)
        with torch.no_grad():
            self.input_projection[0].weight[:, :RAW_FEATURE_WIDTH].copy_(old_projection.weight)
            self.input_projection[0].weight[:, RAW_FEATURE_WIDTH:].zero_()
            self.input_projection[0].bias.copy_(old_projection.bias)
        self.phi = nn.Sequential(nn.Linear(HELPER_FEATURE_WIDTH, 32), nn.SiLU(),
                                 nn.Linear(32, BRANCH_WIDTH))
        self.scorer = nn.Linear(BRANCH_WIDTH, 1, bias=False) if aggregation == 'weighted_pool' else None
        self._initialize_branch()

    def _initialize_branch(self):
        for module in self.phi.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        if self.scorer is not None:
            nn.init.zeros_(self.scorer.weight)

    def helper_representation(self, helper_features, support):
        """Return embedding[Q,16], latent weights[H,Q], and supported count[Q].

        Unsupported entries are discarded before the MLP, including any unused
        nonfinite sentinels. Empty sets are exactly zero but remain connected to
        every branch parameter, giving zero rather than absent gradients.
        """
        if (helper_features.ndim != 3 or helper_features.shape[-1] != HELPER_FEATURE_WIDTH
                or helper_features.dtype != torch.float32
                or support.shape != helper_features.shape[:2] or support.dtype != torch.bool
                or support.device != helper_features.device
                or not torch.isfinite(helper_features[support]).all()):
            raise ValueError('finite supported float32 helper[H,Q,27] and bool support[H,Q] required')
        safe = torch.where(support[..., None], helper_features, torch.zeros_like(helper_features))
        count = support.sum(0)
        divisor = count.clamp_min(1).to(torch.float64)
        weights = support.to(torch.float64) / divisor[None, :]
        if self.aggregation == 'post_pool':
            pooled = (safe.to(torch.float64).sum(0) / divisor[:, None]).to(torch.float32)
            embedding = self.phi(pooled)
            embedding = torch.where((count > 0)[:, None], embedding, torch.zeros_like(embedding))
        else:
            encoded = self.phi(safe)
            if self.scorer is not None:
                score = self.scorer(encoded).squeeze(-1)
                if len(score):
                    masked = torch.where(support, score, torch.full_like(score, -torch.inf))
                    maximum = masked.max(0).values
                    maximum = torch.where(count > 0, maximum, torch.zeros_like(maximum))
                    # Mask first, so an unused enormous score cannot overflow exp.
                    shifted = torch.where(support, score - maximum[None, :], torch.zeros_like(score))
                    mass = torch.where(support, shifted.exp(), torch.zeros_like(score))
                    mass = mass.to(torch.float64)
                    weights = mass / mass.sum(0).clamp_min(1.)[None, :]
                else:
                    weights = score.to(torch.float64) * 0.
            # Float64 reductions make B/C exactly equal for a zero score vector.
            embedding = (encoded.to(torch.float64) * weights[..., None]).sum(0).to(torch.float32)
        # H=0 and all-empty batches must still initialize Adam state for scorer
        # and phi. The zero connection does not alter the physical prediction.
        zero = sum((parameter.sum() * 0. for parameter in self.phi.parameters()), safe.sum() * 0.)
        if self.scorer is not None:
            zero = zero + self.scorer.weight.sum() * 0.
        embedding = embedding + zero
        if not torch.isfinite(embedding).all() or not torch.isfinite(weights).all():
            raise FloatingPointError('helper representation produced nonfinite evidence')
        return embedding, weights, count

    def forward(self, base_states, features, mean_states, helper_features, support):
        if (features.shape != (len(base_states), RAW_FEATURE_WIDTH) or features.dtype != torch.float32
                or not torch.isfinite(features).all() or mean_states.shape != base_states.shape
                or mean_states.dtype != torch.float64 or not torch.isfinite(mean_states).all()
                or not torch.equal(base_states[:, 2:], mean_states[:, 2:])
                or helper_features.ndim != 3 or helper_features.shape[1] != len(base_states)
                or helper_features.device != features.device):
            raise ValueError('matching visible helper/sequence features, base and ordinary Mean required')
        branch, weights, count = self.helper_representation(helper_features, support)
        embedded = self.input_projection(torch.cat((features, branch), -1)).unsqueeze(0)
        residual_m = self.output(self.encoder(embedded).squeeze(0)).to(torch.float64) * 1000.
        states = tangent_position(base_states, residual_m)
        if not torch.isfinite(states).all():
            raise FloatingPointError('helper sequence position produced nonfinite states')
        return dict(states=states, base_states=base_states, mean_states=mean_states,
                    residual_m=residual_m, helper_embedding=branch, helper_weights=weights,
                    helper_counts=count, helper_support_count=count, aggregation=self.aggregation)


def initialize_from_reference(model, reference_state, branch_seed):
    """Strictly inherit an original Transformer step0, then seed only new phi.

    Run on CPU before model.to(device). No checkpoint loading occurs here. The
    independent branch RNG leaves the caller's CPU RNG unchanged; the score
    vector and extra projection columns start at zero, while phi stays nonzero.
    """
    if (not isinstance(model, HelperSequencePosition) or type(branch_seed) is not int
            or branch_seed < 0 or any(p.device.type != 'cpu' for p in model.parameters())):
        raise ValueError('CPU helper model and nonnegative integer branch seed required')
    state = model.state_dict()
    common = {key for key in state if not key.startswith(('phi.', 'scorer.'))}
    if set(reference_state) != common:
        raise ValueError('reference must contain exactly the original Transformer state')
    for key in common:
        expected = state[key].shape
        if key == 'input_projection.0.weight':
            expected = (expected[0], RAW_FEATURE_WIDTH)
        value = reference_state[key]
        if (not isinstance(value, torch.Tensor) or value.shape != expected
                or value.dtype != state[key].dtype or not torch.isfinite(value).all()):
            raise ValueError(f'invalid reference tensor {key}')
    if torch.count_nonzero(reference_state['output.weight']) or torch.count_nonzero(reference_state['output.bias']):
        raise ValueError('reference must be the zero-output step0, not a trained model')
    with torch.no_grad():
        for key in common:
            if key == 'input_projection.0.weight':
                state[key][:, :RAW_FEATURE_WIDTH].copy_(reference_state[key])
                state[key][:, RAW_FEATURE_WIDTH:].zero_()
            else:
                state[key].copy_(reference_state[key])
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(branch_seed)
        model._initialize_branch()
    return dict(common_parameters_exact=True, extra_projection_zero=True,
                reference_output_zero=True, branch_seed=branch_seed,
                parameter_count=sum(p.numel() for p in model.parameters()))
