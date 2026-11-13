"""Shared full-population evaluation for explicitly declared fusion outputs.

This is an evaluation boundary, never a predictor, source selector or aligner.
Each method keeps its own missing-output mask. Input arrays remain untouched.
"""
from __future__ import annotations

import numpy as np

from association.data.physical_units import KNOT_TO_MPS, SPEED_UNIT_POLICY
from association.evaluation.scene import evaluate_full_scene


def canonical_outputs(outputs, speed_unit):
    """Copy output records and convert their speed to m/s exactly once."""
    if speed_unit not in ('kn', 'mps'):
        raise ValueError('explicit prediction speed unit kn or mps required')
    result = []
    for output in outputs:
        row = dict(output)
        if row.get('states') is not None:
            values = np.asarray(row['states'], dtype=np.float64).copy()
            if values.ndim != 2 or values.shape[-1] != 4:
                raise ValueError('group states must be [Q,4]')
            if speed_unit == 'kn':
                values[:, 2] *= KNOT_TO_MPS
            row['states'] = values
        if 'prediction_valid' in row:
            row['prediction_valid'] = np.asarray(row['prediction_valid']).copy()
        result.append(row)
    return result


def evaluate_fusion_methods(method_outputs, *, prediction_speed_units,
                            node_identities, target_ids, target_states, target_valid,
                            support, nearest_seconds, target_speed_unit, split='V_select'):
    """Evaluate the same target registry without intersecting away misses.

    Prediction producers must be called before this function receives truth.
    m/s is the canonical target unit; original Level1 reference interpretation
    remains the caller's documented data contract, not evidence from this API.
    """
    if not method_outputs or set(method_outputs) != set(prediction_speed_units):
        raise ValueError('one explicit speed unit for every method required')
    if target_speed_unit != 'mps':
        raise ValueError('canonical reference speed must be explicitly mps')
    if split not in ('T_core', 'V_select'):
        raise ValueError('only declared development splits; T_test is sealed')
    reports, artifacts = {}, {}
    for name, outputs in method_outputs.items():
        report = evaluate_full_scene(canonical_outputs(outputs, prediction_speed_units[name]),
            node_identities=node_identities, target_identities=target_ids,
            target_states=target_states, target_valid=target_valid, node_observed=support,
            split=split, grouping_kind='predicted', correspondence_policy='anchor_identity_first',
            return_artifact=True, node_input_support=dict(support=support, nearest_seconds=nearest_seconds),
            speed_unit_policy=SPEED_UNIT_POLICY)
        artifacts[name] = report.pop('artifact')
        reports[name] = report
    return reports, artifacts
