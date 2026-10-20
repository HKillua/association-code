"""Unique-target view of frozen per-fragment predictions, without new inference."""
from __future__ import annotations

import numpy as np

from association.evaluation.scene import evaluate_full_scene


def unique_target_view(scene, saved, partitions, predictions, *, return_artifact=False):
    """Keep each group's first input node as its one physical output.

    ``saved`` contains the original 046 per-fragment targets and masks;
    ``scene.target_identity_ids`` is the complete original-label registry before
    invalid fragments were quarantined. Repeated fragment targets may be reduced
    only after checking that registry and their exact consistency. A target with
    no retained input cannot be reconstructed from this cache, so it fails here
    instead of disappearing or receiving invented all-invalid reference values.

    ``partitions[grouping]`` and ``predictions[grouping][arm][N,Q,4]`` are already
    fixed. No identity/reference chooses a group, representative or prediction.
    This additional view does not replace the original all-fragment pilot.
    Reports have the existing full-scene schema and can be passed unchanged to
    ``aggregate_fusion_reports``. Optional artifacts hold its actual metric arrays.
    """
    if scene['split'] != 'V_select':
        raise ValueError('frozen unique-target view requires V_select; T_test remains sealed')
    for key in ('node_ids', 'identities'):
        if not np.array_equal(np.asarray(scene[key], str), np.asarray(saved[key], str)):
            raise ValueError(f'{key} differs between motion inputs and saved targets')
    query = np.asarray(scene['motion']['query_seconds'])
    if not np.array_equal(query, np.asarray(saved['query_seconds'])):
        raise ValueError('saved target queries differ from motion inputs')
    identities = np.asarray(scene['identities'], str)
    registry = np.asarray(scene['target_identity_ids'], str)
    if registry.ndim != 1 or len(set(registry)) != len(registry):
        raise ValueError('original-label target registry must contain distinct identities')
    if set(registry) != set(identities):
        raise ValueError('original-label target registry cannot be reconstructed from fragment targets; '
                         'retain no-input targets by loading complete references before using this view')
    targets = np.asarray(saved['target'])
    valid = np.asarray(saved['target_valid'])
    expected = (len(identities), len(query), 4)
    if targets.shape != expected or valid.shape != expected or valid.dtype != bool:
        raise ValueError('saved per-fragment targets and boolean attribute masks must match [N,Q,4]')
    first = []
    for identity in registry:
        indices = np.flatnonzero(identities == identity)
        index = int(indices[0])
        for other in indices[1:]:
            if (not np.array_equal(targets[index], targets[other], equal_nan=True)
                    or not np.array_equal(valid[index], valid[other])):
                raise ValueError(f'inconsistent repeated fragment target/mask for {identity}')
        first.append(index)
    observed = np.asarray(scene['observed'])
    if (observed.shape != expected[:2] or observed.dtype != bool
            or not np.array_equal(observed, scene['motion']['support'])):
        raise ValueError('original registered observed mask must match the frozen native support')
    if set(partitions) != set(predictions):
        raise ValueError('partitions and prediction groupings must agree')
    reports = {}
    for grouping, groups in partitions.items():
        reports[grouping] = {}
        for arm, values in predictions[grouping].items():
            values = np.asarray(values)
            if values.shape != expected:
                raise ValueError('each frozen prediction must retain every original fragment and query')
            generated = [dict(node_indices=members, states=values[members[0]]) for members in groups]
            report = evaluate_full_scene(generated,
                node_identities=identities, target_identities=registry,
                target_states=targets[first], target_valid=valid[first],
                node_observed=observed, split=scene['split'],
                grouping_kind='oracle' if grouping == 'oracle_identity' else 'predicted',
                correspondence_policy='anchor_identity_first', return_artifact=return_artifact,
                node_input_support=dict(support=scene['motion']['support'],
                                        nearest_seconds=scene['motion']['nearest_seconds']))
            report.update(view='one_group_one_output_unique_target',
                representative_policy='first node in frozen input-order group; no reference selection',
                target_registry_policy='all original-label identities before quarantine, checked against cached targets',
                prediction_policy='copy existing fragment output; no inference or fusion is rerun',
                singleton_policy='existing complete OLS-based fragment prediction; differs from baseline observed-only singleton')
            reports[grouping][arm] = report
    return reports
