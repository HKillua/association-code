"""Shared frozen fusion inference and bounded replay checks.

Forward consumes visible motion evidence and fixed groups only. References remain
outside prediction; replay checks preserve the original fixed CPU/GPU tolerance.
The historical 048 task wrapper is retired; Level1 system and062 use these helpers.
"""
from __future__ import annotations

import numpy as np
import torch

from association.evaluation.fusion_controls import ordinary_fusion, attribute_errors
from association.models.evidence_fusion import fusion_inputs

REPLAY_TOLERANCE = dict(anchor_weight_atol=1e-6, anchor_weight_rtol=1e-5,
    state_span_factor=1e-5, position_m=0.25, speed=0.001, heading_deg=0.001)


def read_scores(path):
    with np.load(path, allow_pickle=False) as z:
        ids = z['scene_ids'].tolist()
        if len(ids) != len(set(ids)):
            raise ValueError('duplicate score scenes')
        # Deliberately do not load labels: they cannot affect predicted groups.
        return {sid: {key: z[sid+'_'+key].copy() for key in ('pairs', 'scores', 'node_ids', 'sources')}
                for sid in ids}


@torch.inference_mode()
def predict_partition(model, motion, groups, device):
    """No target, identity, label or association score is accepted by forward."""
    controls = ordinary_fusion(motion['states'], motion['nearest_seconds'], motion['support'], groups)
    prediction = motion['states'].copy()
    n, q = prediction.shape[:2]
    own = np.ones((n, q, 3))
    anchors, helpers, utilities, weights = [], [], [], []
    for group in groups:
        for anchor in group:
            members = [anchor]+[node for node in group if node != anchor]
            output = model(*fusion_inputs(motion, members, device=device))
            prediction[anchor] = output['states'].cpu().numpy()
            own[anchor] = output['weights'][0].cpu().numpy()
            for i, helper in enumerate(members[1:]):
                anchors.append(anchor)
                helpers.append(helper)
                utilities.append(output['utility'][i].cpu().numpy())
                weights.append(output['weights'][i+1].cpu().numpy())
    routing = dict(helper_anchor=np.asarray(anchors, np.int64), helper_node=np.asarray(helpers, np.int64),
        utility=np.asarray(utilities, np.float32).reshape(-1, q, 3),
        helper_weights=np.asarray(weights, np.float64).reshape(-1, q, 3), anchor_weights=own,
        membership=controls['membership'])
    return dict(mean_supported=controls['predictions']['mean_supported'], learned=prediction), routing


def replay_check(saved, predictions, routing):
    """Fixed, bounded FP32 CPU/GPU tolerance; never widen it after observing V."""
    prefix = 'predicted_anchor_direct_'
    state, old = predictions['learned'], saved[prefix+'learned']
    delta = state-old
    for col in (1, 3):
        delta[..., col] = (delta[..., col]+180.) % 360.-180.
    physical = attribute_errors(state, old)
    maxima = {name: float(np.max(value, initial=0.)) for name, value in physical.items()}
    span = np.zeros_like(state)
    anchors, helpers = routing['helper_anchor'], routing['helper_node']
    if len(anchors):
        diff = saved['native_states'][helpers]-saved['native_states'][anchors]
        for col in (1, 3):
            diff[..., col] = (diff[..., col]+180.) % 360.-180.
        np.maximum.at(span, anchors, abs(diff))
    rounding = 32*np.finfo(np.float64).eps*np.maximum(abs(old), 1.)
    tolerance = REPLAY_TOLERANCE
    own_delta = abs(routing['anchor_weights']-saved[prefix+'anchor_weights'])
    checks = dict(
        membership_exact=np.array_equal(routing['membership'], saved[prefix+'membership']),
        ordinary_control_exact=np.array_equal(predictions['mean_supported'], saved[prefix+'mean_supported']),
        anchor_weights_close=np.allclose(routing['anchor_weights'], saved[prefix+'anchor_weights'],
            atol=tolerance['anchor_weight_atol'], rtol=tolerance['anchor_weight_rtol']),
        state_span_close=bool(np.all(abs(delta) <= tolerance['state_span_factor']*span+rounding)),
        physical_close=all(maxima[key] <= tolerance[limit]
                           for key, limit in (('position', 'position_m'), ('speed', 'speed'), ('heading', 'heading_deg'))))
    return dict(passed=bool(all(checks.values())), checks={k: bool(v) for k, v in checks.items()},
        max_physical_prediction_difference=maxima, max_anchor_weight_difference=float(own_delta.max(initial=0.)),
        tolerance=tolerance, rationale='Fixed FP32 cross-device rounding allowance plus absolute physical caps; '
        'not a theoretical error bound and never adjusted after replay.')
