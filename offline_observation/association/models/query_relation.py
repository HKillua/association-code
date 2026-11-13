"""098 query-level set attention with one fixed observable relation bias.

Both modes preserve the original 76-feature Transformer and see exactly the
same supported helper[H,Q,27] rows. The candidate changes only attention logits:
compressed relative-position/OLS-velocity disagreement adds a nonpositive bias.
This is neither a metric distance nor a calibrated reliability probability.
"""
from __future__ import annotations

import torch
from torch import nn

from association.models.helper_sequence import helper_sequence_inputs
from association.models.sequence_position import SequencePosition, tangent_position

SCOPE = 'query-relation-position-v1'
ROUTING_SEMANTICS = 'source-final-query-set-relation-position-v1'
MODES = ('generic', 'relation')
RAW_FEATURE_WIDTH = 76
HELPER_FEATURE_WIDTH = 27
BRANCH_WIDTH = 16
RELATION_COLUMNS = (16, 17, 20, 21)
BRANCH_PREFIXES = ('phi.', 'set_query.', 'set_key.', 'set_value.', 'set_output.',
                   'set_norm.', 'pool_query.', 'branch_projection.')


def _masked_softmax(scores, mask):
    """Normalize the last axis; an empty supported set has exact zero mass."""
    safe = torch.where(mask, scores, torch.zeros_like(scores))
    maximum = safe.masked_fill(~mask, -torch.inf).amax(-1, keepdim=True)
    maximum = torch.where(mask.any(-1, keepdim=True), maximum, torch.zeros_like(maximum))
    # Mask before exponentiation: an ignored row must not create inf/nan gradients.
    shifted = torch.where(mask, safe - maximum, torch.zeros_like(safe))
    mass = torch.where(mask, shifted.exp(), torch.zeros_like(shifted))
    return mass / mass.sum(-1, keepdim=True).clamp_min(1.)


def observable_relation_bias(safe_features):
    """Return [Q,H,H] = -min(mean((z_i-z_j)^2),4), z uses four old columns.

    Called after unsupported rows have been zeroed. Float64 differences avoid
    overflow while forming this bounded, parameter-free observable quantity.
    Storage is Q*H*H*4, never Q*H*H*27. Diagonal biases are exactly zero.
    """
    coordinates = safe_features[..., list(RELATION_COLUMNS)].permute(1, 0, 2).to(torch.float64)
    difference = coordinates[:, :, None, :] - coordinates[:, None, :, :]
    return -difference.square().mean(-1).clamp_max(4.).to(torch.float32)


class QueryRelationPosition(SequencePosition):
    """Generic and relation modes have identical parameters and initialization."""

    def __init__(self, *, mode='generic', hidden_size=64, num_layers=2, initial_gate=.95):
        if mode not in MODES:
            raise ValueError('098 requires generic or relation set attention')
        super().__init__(backbone='transformer', hidden_size=hidden_size,
                         num_layers=num_layers, initial_gate=initial_gate)
        self.mode = mode
        self.aggregation = mode
        self.routing_semantics = ROUTING_SEMANTICS
        self._relation_enabled = mode == 'relation'
        self.phi = nn.Sequential(nn.Linear(HELPER_FEATURE_WIDTH, 32), nn.SiLU(),
                                 nn.Linear(32, BRANCH_WIDTH))
        self.set_query = nn.Linear(BRANCH_WIDTH, BRANCH_WIDTH)
        self.set_key = nn.Linear(BRANCH_WIDTH, BRANCH_WIDTH)
        self.set_value = nn.Linear(BRANCH_WIDTH, BRANCH_WIDTH)
        self.set_output = nn.Linear(BRANCH_WIDTH, BRANCH_WIDTH)
        self.set_norm = nn.LayerNorm(BRANCH_WIDTH)
        self.pool_query = nn.Linear(RAW_FEATURE_WIDTH, BRANCH_WIDTH)
        self.branch_projection = nn.Linear(BRANCH_WIDTH, hidden_size, bias=False)
        self._initialize_branch()

    def _initialize_branch(self):
        for name in ('phi', 'set_query', 'set_key', 'set_value', 'set_output', 'pool_query'):
            for module in getattr(self, name).modules():
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    nn.init.zeros_(module.bias)
        nn.init.ones_(self.set_norm.weight)
        nn.init.zeros_(self.set_norm.bias)
        nn.init.zeros_(self.branch_projection.weight)

    def _enabled(self, override):
        enabled = self._relation_enabled if override is None else override
        if type(enabled) is not bool or (self.mode == 'generic' and enabled):
            raise ValueError('relation bias can only be enabled for the declared relation mode')
        return enabled

    def set_relation_enabled(self, enabled):
        """Same-weight CPU ablation; no state-dict or parameter is modified."""
        self._relation_enabled = self._enabled(enabled)

    def _branch_zero(self):
        # Keep every new tensor in Adam even for H=0/all-masked training groups.
        return sum(parameter.sum() * 0. for name, parameter in self.named_parameters()
                   if name.startswith(BRANCH_PREFIXES))

    def helper_representation(self, helper_features, support, features, *, relation_enabled=None):
        """Return embedding[Q,16], query-pooling weights[H,Q], supported count[Q]."""
        if (helper_features.ndim != 3 or helper_features.shape[-1] != HELPER_FEATURE_WIDTH
                or helper_features.dtype != torch.float32
                or support.shape != helper_features.shape[:2] or support.dtype != torch.bool
                or support.device != helper_features.device
                or features.shape != (helper_features.shape[1], RAW_FEATURE_WIDTH)
                or features.dtype != torch.float32 or features.device != helper_features.device
                or not torch.isfinite(features).all()
                or not torch.isfinite(helper_features[support]).all()):
            raise ValueError('finite supported helper[H,Q,27], bool support and visible query[Q,76] required')
        enabled = self._enabled(relation_enabled)
        safe = torch.where(support[..., None], helper_features, torch.zeros_like(helper_features))
        count = support.sum(0)
        zero = self._branch_zero()
        if not len(helper_features):
            return (features.new_zeros((len(features), BRANCH_WIDTH)) + zero,
                    features.new_zeros((0, len(features)), dtype=torch.float64) + zero, count)
        encoded = self.phi(safe).permute(1, 0, 2)
        query, key, value = (layer(encoded) for layer in (self.set_query, self.set_key, self.set_value))
        scores = torch.matmul(query, key.transpose(-2, -1)) / 4.
        if enabled:
            scores = scores + observable_relation_bias(safe)
        visible = support.transpose(0, 1)
        attention = _masked_softmax(scores, visible[:, :, None] & visible[:, None, :])
        context = self.set_norm(encoded + self.set_output(torch.matmul(attention, value)))
        context = torch.where(visible[..., None], context, torch.zeros_like(context))
        pool_scores = (context * self.pool_query(features)[:, None, :]).sum(-1) / 4.
        pool_weights = _masked_softmax(pool_scores.to(torch.float64), visible)
        embedding = (context.to(torch.float64) * pool_weights[..., None]).sum(1).to(torch.float32) + zero
        weights = pool_weights.transpose(0, 1)
        if not torch.isfinite(embedding).all() or not torch.isfinite(weights).all():
            raise FloatingPointError('098 set attention produced nonfinite evidence')
        return embedding, weights, count

    def forward(self, base_states, features, mean_states, helper_features, support, *, relation_enabled=None):
        if (base_states.ndim != 2 or base_states.shape[-1] != 4 or not len(base_states)
                or base_states.dtype != torch.float64 or not torch.isfinite(base_states).all()
                or features.shape != (len(base_states), RAW_FEATURE_WIDTH)
                or features.dtype != torch.float32 or not torch.isfinite(features).all()
                or mean_states.shape != base_states.shape or mean_states.dtype != torch.float64
                or not torch.isfinite(mean_states).all()
                or not torch.equal(base_states[:, 2:], mean_states[:, 2:])
                or helper_features.ndim != 3 or helper_features.shape[1] != len(base_states)
                or base_states.device != features.device or mean_states.device != features.device
                or helper_features.device != features.device):
            raise ValueError('098 requires matching visible helper/query features, physical base and ordinary Mean')
        enabled = self._enabled(relation_enabled)
        branch, weights, count = self.helper_representation(helper_features, support, features,
                                                           relation_enabled=enabled)
        # Keep the old Linear[76,64] GEMM unchanged so zero branch initialization
        # preserves source-final outputs bit for bit, even for a nonzero head.
        projected = self.input_projection[0](features) + self.branch_projection(branch)
        embedded = self.input_projection[2](self.input_projection[1](projected)).unsqueeze(0)
        residual_m = self.output(self.encoder(embedded).squeeze(0)).to(torch.float64) * 1000.
        states = tangent_position(base_states, residual_m)
        if not torch.isfinite(states).all():
            raise FloatingPointError('098 query relation model produced nonfinite states')
        return dict(states=states, base_states=base_states, mean_states=mean_states,
                    residual_m=residual_m, helper_embedding=branch, helper_weights=weights,
                    helper_counts=count, helper_support_count=count, aggregation=self.aggregation,
                    relation_enabled=enabled)


def initialize_from_reference(model, reference_state, branch_seed):
    """Copy the complete original Transformer FINAL and independently seed the branch.

    The source retains its [64,76] projection and may have a trained nonzero
    output. No source optimizer is loaded here. New projection weights are zero.
    Caller CPU RNG is preserved; all model parameters remain trainable.
    """
    if (not isinstance(model, QueryRelationPosition) or type(branch_seed) is not int
            or branch_seed < 0 or any(parameter.device.type != 'cpu' for parameter in model.parameters())
            or not isinstance(reference_state, dict)):
        raise ValueError('098 requires a CPU query-relation model, source state and nonnegative branch seed')
    state = model.state_dict()
    common = {key for key in state if not key.startswith(BRANCH_PREFIXES)}
    if set(reference_state) != common:
        raise ValueError('098 reference must contain exactly the original Transformer tensors')
    for key in common:
        value = reference_state[key]
        if (not isinstance(value, torch.Tensor) or value.shape != state[key].shape
                or value.dtype != state[key].dtype or not torch.isfinite(value).all()):
            raise ValueError('098 incompatible reference tensor: ' + key)
    with torch.no_grad():
        for key in common:
            state[key].copy_(reference_state[key])
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(branch_seed)
        model._initialize_branch()
    return dict(source_final_parameters_exact=True, branch_projection_zero=True,
                branch_seed=branch_seed, parameter_count=sum(p.numel() for p in model.parameters()),
                trainable_parameter_count=sum(p.numel() for p in model.parameters() if p.requires_grad))
