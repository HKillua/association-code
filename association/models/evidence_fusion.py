"""Native-evidence utility routing around a fixed physical motion anchor.

Only visible motion evidence enters this module. Position shares one weight for
latitude/longitude; speed and heading have independent weights. No reference,
identity, source-specific reliability prior, association score or learned
completion checkpoint is accepted by the forward path.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

FEATURE_WIDTH = 27
ATTRIBUTE_NAMES = ('position', 'speed', 'heading')
ERROR_SCALES = (1000., 10., 30.)


def short_delta(value):
    result = torch.remainder(value + 180., 360.) - 180.
    return torch.where((result == -180.) & (value > 0.), -result, result)


def fusion_inputs(motion, members, *, device='cpu'):
    """Anchor is members[0]; build features exclusively from visible evidence.

    Physical states remain float64, preserving the ordinary OLS control.
    Only MLP features/parameters use float32. Fixed scales are unit conversion
    conventions, not fitted against V. Signed physical values use signed_log1p
    compression so retained short tracks/long extrapolations cannot dominate
    solely through an unbounded linear feature scale. Source IDs never enter.
    """
    members = np.asarray(members, dtype=np.int64)
    if members.ndim != 1 or not len(members) or len(set(members.tolist())) != len(members):
        raise ValueError('members must contain a distinct anchor and optional helpers')
    states = np.asarray(motion['states'], np.float64)[members]
    k, q, _ = states.shape
    def signed_log(value):
        value = np.asarray(value)
        return np.sign(value)*np.log1p(np.abs(value))

    rows = []
    for node in members:
        rows.append(np.column_stack((
            np.log1p(motion['nearest_seconds'][node]/20.),
            motion['support'][node], motion['exact_observation'][node],
            np.full(q, np.log1p(motion['native_count'][node])),
            np.full(q, np.log1p(motion['span_seconds'][node]/20.)),
            np.full(q, np.log1p(motion['fit_rmse_m'][node]/1000.)),
            np.broadcast_to(signed_log(np.asarray(motion['ols_velocity_mps'][node])/10.), (q, 2)),
            np.broadcast_to(signed_log((np.asarray(motion['endpoint_velocity_mps'][node])
                                       -motion['ols_velocity_mps'][node])/10.), (q, 2)),
        )))
    reliability = np.asarray(rows)
    delta = states[1:] - states[:1]
    raw_lon = delta[..., 1].copy()
    delta[..., 1] = (raw_lon + 180.) % 360. - 180.
    delta[..., 1][(delta[..., 1] == -180.) & (raw_lon > 0.)] = 180.
    heading = np.deg2rad(delta[..., 3])
    relative = np.stack((
        signed_log(delta[..., 0] * (6371000.*np.pi/180.) / 1000.),
        signed_log(delta[..., 1] * (6371000.*np.pi/180.)
        * np.cos(np.deg2rad((states[1:, :, 0]+states[:1, :, 0])/2.)) / 1000.),
        signed_log(delta[..., 2]/10.), np.sin(heading), np.cos(heading),
        np.log1p(states[1:, :, 2]/10.), np.broadcast_to(np.log1p(states[:1, :, 2]/10.), (k-1, q)),
    ), axis=-1)
    features = np.concatenate((np.broadcast_to(reliability[:1], (k-1, q, 10)),
                               reliability[1:], relative), axis=-1)
    if not np.isfinite(states).all() or not np.isfinite(features).all():
        raise ValueError('fusion evidence must be finite')
    return (torch.as_tensor(states, dtype=torch.float64, device=device),
            torch.as_tensor(features, dtype=torch.float32, device=device),
            torch.as_tensor(motion['exact_observation'][members[0]],
                            dtype=torch.bool, device=device))


class EvidenceFusion(nn.Module):
    def __init__(self, *, exact_observation_lock=False):
        super().__init__()
        if not isinstance(exact_observation_lock, bool):
            raise ValueError('exact_observation_lock must be boolean')
        self.exact_observation_lock = exact_observation_lock
        self.utility = nn.Sequential(nn.Linear(FEATURE_WIDTH, 64), nn.SiLU(),
                                     nn.Linear(64, 64), nn.SiLU(), nn.Linear(64, 3))
        nn.init.zeros_(self.utility[-1].weight)
        nn.init.zeros_(self.utility[-1].bias)

    def forward(self, states, features, exact_observation):
        if (states.ndim != 3 or states.shape[-1] != 4 or len(states) < 1
                or features.shape != (len(states)-1, states.shape[1], FEATURE_WIDTH)
                or exact_observation.shape != (states.shape[1],)
                or exact_observation.dtype != torch.bool):
            raise ValueError('expected states[K,Q,4], features[K-1,Q,27], exact[Q]')
        utility = self.utility(features)
        positive = torch.relu(utility).to(states.dtype)
        if self.exact_observation_lock:
            positive = positive.masked_fill(exact_observation[None, :, None], 0.)
        denominator = 1. + positive.sum(0, keepdim=True)
        weights = torch.cat((torch.ones_like(denominator), positive), 0)/denominator
        delta = states[1:] - states[:1]
        delta = torch.stack((delta[..., 0], short_delta(delta[..., 1]),
                             delta[..., 2], short_delta(delta[..., 3])), -1)
        weights4 = weights[1:, :, [0, 0, 1, 2]]
        prediction = states[0] + (weights4*delta).sum(0)
        prediction = torch.stack((prediction[:, 0],
                                  torch.remainder(prediction[:, 1]+180., 360.)-180.,
                                  prediction[:, 2],
                                  torch.remainder(prediction[:, 3], 360.)), -1)
        # Preserve exact anchor bit patterns when no update was accepted.
        active = (weights4 > 0).any(0)
        prediction = torch.where(active, prediction, states[0])
        return dict(states=prediction, utility=utility, weights=weights)


def physical_errors(prediction, target):
    """Differentiable physical errors; exact coincidence has finite zero gradient."""
    lat = torch.deg2rad(prediction[..., 0])
    target_lat = torch.deg2rad(target[..., 0])
    dlon = torch.deg2rad(short_delta(prediction[..., 1]-target[..., 1]))
    a = (torch.sin((lat-target_lat)/2.).square()
         +torch.cos(lat)*torch.cos(target_lat)*torch.sin(dlon/2.).square()).clamp(0., 1.)
    bounded = a.clamp(1e-30, 1.-torch.finfo(a.dtype).eps)
    distance = 2.*6371000.*torch.atan2(bounded.sqrt(), (1.-bounded).sqrt())
    distance = torch.where(a == 0, torch.zeros_like(distance), distance)
    return torch.stack((distance, (prediction[..., 2]-target[..., 2]).abs(),
                        short_delta(prediction[..., 3]-target[..., 3]).abs()), -1)


def evidence_fusion_loss(result, base_states, target, valid):
    """Reference is only a loss argument, and all helper errors use anchor truth.

    Validity is the anchor's per-attribute reference mask. A wrong helper must
    never be supervised against its own identity's reference.
    """
    if target.shape != base_states.shape[1:] or valid.shape != target.shape or valid.dtype != torch.bool:
        raise ValueError('one anchor reference and per-attribute mask are required')
    mask = torch.stack((valid[:, :2].all(-1), valid[:, 2], valid[:, 3]), -1)
    safe_target = torch.where(valid, target, torch.zeros_like(target))
    # Keep a computational graph for empty groups/masks without invented labels.
    zero = result['utility'].sum()*0.
    scales = base_states.new_tensor(ERROR_SCALES)
    errors = torch.log1p(physical_errors(base_states, safe_target)/scales)
    gain_target = (errors[:1]-errors[1:]).detach()
    fused = torch.log1p(physical_errors(result['states'], safe_target)/scales)
    utility_terms, direct_terms = [], []
    for attribute in range(3):
        selected = mask[:, attribute]
        if selected.any():
            direct_terms.append(fused[selected, attribute].mean())
            if len(base_states) > 1:
                utility_terms.append(torch.nn.functional.smooth_l1_loss(
                    result['utility'][:, selected, attribute],
                    gain_target[:, selected, attribute].to(result['utility'].dtype)))
    utility_loss = sum(utility_terms, zero)/max(len(utility_terms), 1)
    direct_loss = sum(direct_terms, zero)/max(len(direct_terms), 1)
    return dict(total=utility_loss+direct_loss, utility=utility_loss, direct=direct_loss,
                gain_target=gain_target, valid_attributes=mask)
