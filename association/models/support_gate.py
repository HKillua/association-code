"""Bounded position correction toward the frozen supported-mean candidate.

Only observable motion enters this module. The sigmoid is a decision weight,
not a calibrated probability. Speed and heading remain ordinary mean outputs.
"""
from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn

from association.evaluation.fusion_controls import ordinary_fusion
from association.models.evidence_fusion import fusion_inputs, physical_errors, short_delta

SCOPE = 'support-gate-v1'
ROUTING_SEMANTICS = 'supported-mean-position-gate-v1'
FEATURE_WIDTH = 56
INITIAL_GATE = .95


def gate_inputs(motion, members, *, device='cpu'):
    """Return anchor, exact ordinary mean, 56 visible features, and eligibility.

    The first member is the anchor. Helpers are sorted by original node index,
    making helper permutation immaterial even to floating-point reductions.
    Each query pools the existing 27 pair features over supported helpers only:
    mean, max, log-count, and log local-metre RMS disagreement from the mean.
    Empty helper pools have zero features and cannot cause position correction.
    """
    members = np.asarray(members)
    states = np.asarray(motion['states'], np.float64)
    if (members.ndim != 1 or not len(members) or members.dtype.kind not in 'iu'
            or len(set(members.tolist())) != len(members)
            or (members < 0).any() or (members >= len(states)).any()):
        raise ValueError('members require a distinct valid anchor and optional helpers')
    members = np.r_[members[:1], np.sort(members[1:])].astype(np.int64)
    support = np.asarray(motion['support'])
    nearest = np.asarray(motion['nearest_seconds'])
    if (states.ndim != 3 or states.shape[-1] != 4 or support.shape != states.shape[:2]
            or support.dtype != bool or nearest.shape != support.shape):
        raise ValueError('motion states, support and nearest-query shapes disagree')
    local = states[members]
    k, q, _ = local.shape
    ordinary = ordinary_fusion(local, nearest[members], support[members], [list(range(k))])
    mean = ordinary['predictions']['mean_supported'][0]
    _, pair_features, _ = fusion_inputs(motion, members, device='cpu')
    pair_features = pair_features.numpy().astype(np.float64)
    helper_support = support[members[1:]]
    count = helper_support.sum(0)
    eligible = ~support[members[0]] & (count > 0)
    features = np.zeros((q, FEATURE_WIDTH), np.float64)
    if k > 1:
        features[:, :27] = np.where(helper_support[..., None], pair_features, 0.).sum(0) / np.maximum(count, 1)[:, None]
        maxima = np.where(helper_support[..., None], pair_features, -np.inf).max(0)
        features[:, 27:54] = np.where((count > 0)[:, None], maxima, 0.)
        # A fixed local N/E plane about the ordinary mean; no target is used.
        delta = local[1:, :, :2] - mean[None, :, :2]
        raw_lon = delta[..., 1].copy()
        delta[..., 1] = (raw_lon + 180.) % 360. - 180.
        delta[..., 1][(delta[..., 1] == -180.) & (raw_lon > 0.)] = 180.
        delta *= 6371000. * np.pi / 180.
        delta[..., 1] *= np.cos(np.deg2rad(mean[None, :, 0]))
        distance_squared = np.square(delta).sum(-1)
        rms = np.sqrt(np.where(helper_support, distance_squared, 0.).sum(0) / np.maximum(count, 1))
        features[:, 55] = np.log1p(rms / 1000.)
    features[:, 54] = np.log1p(count)
    if not np.isfinite(features).all():
        raise ValueError('support-gate features must be finite')
    return (torch.as_tensor(local[0], dtype=torch.float64, device=device),
            torch.as_tensor(mean, dtype=torch.float64, device=device),
            torch.as_tensor(features, dtype=torch.float32, device=device),
            torch.as_tensor(eligible, dtype=torch.bool, device=device))


def mix_gate_states(anchor, mean, gate, eligible):
    """Mix latitude and shortest-arc longitude; preserve both exact endpoints."""
    if (anchor.ndim != 2 or anchor.shape[-1] != 4 or mean.shape != anchor.shape
            or gate.shape != anchor.shape[:1] or eligible.shape != gate.shape
            or eligible.dtype != torch.bool or anchor.dtype != torch.float64
            or mean.dtype != torch.float64):
        raise ValueError('expected float64 anchor/mean[Q,4], gate[Q] and bool eligible[Q]')
    if (not torch.isfinite(anchor).all() or not torch.isfinite(mean).all()
            or not torch.isfinite(gate).all() or (gate < 0).any() or (gate > 1).any()):
        raise ValueError('states must be finite and gate must lie in [0,1]')
    fraction = gate.to(anchor.dtype)
    latitude = anchor[:, 0] + fraction * (mean[:, 0] - anchor[:, 0])
    longitude = anchor[:, 1] + fraction * short_delta(mean[:, 1] - anchor[:, 1])
    position = torch.stack((latitude, torch.remainder(longitude + 180., 360.) - 180.), -1)
    position = torch.where((fraction == 1)[:, None], mean[:, :2], position)
    position = torch.where((~eligible | (fraction == 0))[:, None], anchor[:, :2], position)
    return torch.cat((position, mean[:, 2:]), -1)


class SupportGate(nn.Module):
    def __init__(self, *, initial_gate=INITIAL_GATE):
        super().__init__()
        if isinstance(initial_gate, bool) or initial_gate != INITIAL_GATE:
            raise ValueError('068 fixes initial_gate=0.95; no V-selected initialization')
        self.initial_gate = INITIAL_GATE
        self.routing_semantics = ROUTING_SEMANTICS
        self.head = nn.Sequential(nn.Linear(FEATURE_WIDTH, 64), nn.SiLU(),
                                  nn.Linear(64, 64), nn.SiLU(), nn.Linear(64, 1))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.constant_(self.head[-1].bias, math.log(19.))

    def forward(self, anchor, mean, features, eligible):
        if (features.shape != (len(anchor), FEATURE_WIDTH) or features.dtype != torch.float32
                or not torch.isfinite(features).all()):
            raise ValueError('support-gate features must be finite float32 [Q,56]')
        logits = self.head(features).squeeze(-1)
        if not torch.isfinite(logits).all():
            raise FloatingPointError('support-gate logits must be finite')
        gate = torch.sigmoid(logits)
        states = mix_gate_states(anchor, mean, gate, eligible)
        return dict(states=states, gate=gate, eligible=eligible, logits=logits, mean_states=mean)


def support_gate_loss(result, target, valid):
    """T-only direct position loss on all valid queries, including R-exterior.

    Outside R the position is constant, so its loss contributes no gradient.
    Keeping its points in the denominator avoids changing the target population.
    """
    if target.shape != result['states'].shape or valid.shape != target.shape or valid.dtype != torch.bool:
        raise ValueError('one anchor target and per-attribute validity mask are required')
    selected = valid[:, :2].all(-1)
    if not torch.isfinite(target[:, :2][selected]).all():
        raise ValueError('valid position references must be finite')
    zero = result['logits'].sum() * 0.
    if not selected.any():
        return zero
    # Other attributes are not supervised by this head, even when their masks
    # are true. Do not let an unused speed/course value enter the loss graph.
    position_target = torch.where(selected[:, None], target[:, :2], torch.zeros_like(target[:, :2]))
    safe_target = torch.cat((position_target, torch.zeros_like(target[:, 2:])), -1)
    error = physical_errors(result['states'], safe_target)[:, 0]
    return torch.log1p(error[selected] / 1000.).mean() + zero
