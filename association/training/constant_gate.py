"""Finite T-only scalar search on the actual successful 071B sample sequence.

This is an empirical constant-control fit, not a continuous-global-optimum claim.
Each sampled anchor keeps its original all-valid-query denominator, including
queries outside the opportunity region and anchors with no valid supervision.
"""
from __future__ import annotations

import numpy as np
import torch

from association.models.support_gate import gate_inputs, mix_gate_states
from association.models.evidence_fusion import physical_errors
from association.training.evidence_fusion import prepare_evidence_scenes, training_members


def candidate_logits():
    """1001 predeclared nominal fractions; finite logits include exact endpoints."""
    fractions = np.linspace(0., 1., 1001)
    logits = np.empty_like(fractions)
    logits[0], logits[-1] = -100., 100.
    logits[1:-1] = np.log(fractions[1:-1] / (1. - fractions[1:-1]))
    return torch.as_tensor(logits, dtype=torch.float32)


def collect_training_points(train_raw, rows, source_record, *, check=None):
    """Reconstruct only the recorded T samples; verify actual members and masks."""
    ids = [str(scene['scene_id']) for scene in train_raw]
    if (ids != source_record['training_scene_ids'] or len(set(ids)) != len(ids)
            or any(scene['split'] != 'T_core' for scene in train_raw)):
        raise ValueError('constant fit requires the complete original ordered T_core')
    train = prepare_evidence_scenes(train_raw, check=check, motion_observation_policy='canonical_all')
    lookup = {str(scene['scene_id']): scene for scene in train}
    anchors, means, targets, weights = [], [], [], []
    fixed_loss, sample_count, valid_count, eligible_count = 0., 0, 0, 0
    for row in rows:
        if check:
            check()
        for sample in row['batch']:
            scene = lookup[sample['scene_id']]
            node = sample['anchor_index']
            members = training_members(scene, node, row['mode'])
            if (members != sample['members'] or str(scene['node_ids'][node]) != sample['node_id']):
                raise ValueError('constant fit actual anchor/member reconstruction differs')
            anchor, mean, _, eligible = gate_inputs(scene['motion'], members)
            valid = torch.as_tensor(scene['valid'][node][:, :2].all(-1))
            target = torch.as_tensor(scene['target'][node], dtype=torch.float64)
            counts = (int(eligible.sum()), int((eligible & valid).sum()), int(valid.sum()))
            if counts != tuple(sample[k] for k in ('eligible_query_points',
                    'eligible_valid_position_points', 'valid_position_points')):
                raise ValueError('constant fit actual support/valid counts differ')
            sample_count += 1
            valid_count += counts[2]
            eligible_count += counts[1]
            if not counts[2]:
                continue  # Zero-loss anchors remain in the sampled-anchor denominator.
            if not torch.isfinite(target[valid, :2]).all():
                raise ValueError('constant fit valid target position is not finite')
            active = eligible & valid
            inactive = ~eligible & valid
            if inactive.any():
                fixed_loss += float(physical_errors(anchor[inactive], target[inactive])[:, 0].sum()) / (1000. * counts[2])
            if active.any():
                anchors.append(anchor[active]); means.append(mean[active]); targets.append(target[active])
                weights.append(torch.full((counts[1],), 1. / counts[2], dtype=torch.float64))
    if not sample_count or not anchors:
        raise ValueError('constant fit has no recorded samples or eligible valid points')
    return dict(anchor=torch.cat(anchors), mean=torch.cat(means), target=torch.cat(targets),
                weight=torch.cat(weights) / sample_count, fixed_loss=fixed_loss/sample_count,
                sampled_anchors=sample_count, valid_position_points=valid_count,
                eligible_valid_position_points=eligible_count)


def search_constant(points, *, logits=None, check=None):
    """Exhaustive declared finite grid, ties choose the smaller candidate index."""
    logits = candidate_logits() if logits is None else logits
    gates = torch.sigmoid(logits)
    anchor, mean, target, weight = (points[key] for key in ('anchor', 'mean', 'target', 'weight'))
    eligible = torch.ones(len(anchor), dtype=torch.bool)
    losses = []
    with torch.inference_mode():
        for gate in gates:
            if check:
                check()
            states = mix_gate_states(anchor, mean, gate.expand(len(anchor)), eligible)
            loss = float((physical_errors(states, target)[:, 0] * weight).sum()) / 1000. + points['fixed_loss']
            if not np.isfinite(loss):
                raise FloatingPointError('nonfinite constant-grid training objective')
            losses.append(loss)
    best = int(np.argmin(losses))
    return dict(train_only=True, optimizer_updates=0, candidate_count=len(logits),
        selected_index=best, selected_logit=float(logits[best]), selected_gate=float(gates[best]),
        selected_training_loss=losses[best], grid_logits=logits.tolist(),
        effective_gates=gates.tolist(), training_losses=losses,
        sampled_anchors=points['sampled_anchors'], valid_position_points=points['valid_position_points'],
        eligible_valid_position_points=points['eligible_valid_position_points'],
        objective='linear haversine_m/1000; all-valid mean per recorded anchor then sampled-anchor mean',
        limitation='Finite 0.001 nominal grid on071B observed synthetic training groups; not continuous optimum or deployment-distribution optimum.')
