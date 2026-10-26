"""Development-only, full-target fusion evaluation on one explicit time grid.

Correspondence uses the true identity of each predicted group's first node,
only after prediction. The first group for an identity supplies its target row;
later groups remain extra predictions and also receive their own error report.
No geometry-based matching, group repair, threshold fitting or model call occurs.
This is an explicit development diagnostic, not the paper's mPCA/mPFA metric.
"""
from __future__ import annotations

import numpy as np

from association.evaluation.grouping import _partition, group_tracks_by_identity
from association.evaluation.physical import _check_valid_states, _mask, _states, evaluate_aligned_fusion, group_membership_diagnostics
from association.data.reference_states import ALLOWED_SPLITS
from association.evaluation.support import group_input_support
from association.evaluation.paper_fusion_metrics import paper_fusion_scene


INPUT_SUPPORT_REGIONS = ('anchor_support', 'helper_only_support', 'unsupported', 'unmatched_target')
INPUT_DISTANCE_REGIONS = ('le_300s', 'gt_300_le_600s', 'gt_600_le_1800s', 'gt_1800s', 'no_input')


def _array(value):
    if hasattr(value, 'detach'):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _identities(values, name):
    result = np.asarray(values, dtype=object)
    if result.ndim != 1 or any(not isinstance(x, str) or not x for x in result):
        raise ValueError(f'{name} must be a one-dimensional sequence of nonempty strings')
    return result


def _regions(prediction, truth, truth_valid, prediction_valid, observed):
    return {
        name: evaluate_aligned_fusion(prediction, truth, truth_valid,
                                      prediction_valid=prediction_valid, region_mask=mask)
        for name, mask in (('all', np.ones(observed.shape, bool)),
                           ('observed', observed), ('missing', ~observed))
    }


def evaluate_full_scene(group_outputs, *, node_identities, target_identities,
                        target_states, target_valid, node_observed, split,
                        grouping_kind, correspondence_policy, return_artifact=False,
                        node_input_support=None):
    """Evaluate all targets and all fixed-order predicted groups without I/O.

    ``group_outputs`` contains one prediction record per emitted group: each has
    ``node_indices`` and physical ``states[L,4]`` (or None for no output).
    Optional ``prediction_valid[L,4]`` explicitly marks available features;
    otherwise all output values are declared valid, so NaNs fail validation.
    Groups must partition every input node, including singleton/mixed groups.
    The caller may replace a singleton's missing states with its aligned raw
    observations and validity mask; this function does not invent that policy.

    Targets are unique identities with aligned arrays [T,L,4]. They may include
    targets having no input nodes; these remain in the coverage denominator.
    ``node_observed[N,L]`` is caller-declared original observation/support, not
    model availability. Its union over all nodes of each true identity defines
    observed versus missing regions, identically for every compared method.
    Reference validity remains a separate per-feature mask.

    ``grouping_kind`` is 'predicted' or 'oracle'. Oracle explicitly requires the
    full true-identity partition and is diagnostic only. Correspondence policy
    must be 'anchor_identity_first': no true identity changes a predicted group,
    and neither reference values nor prediction error choose a correspondence.
    The primary rows contain all T targets plus unmatched duplicate predictions.
    Additional per-group errors retain every group against its anchor identity;
    these repeat targets and must not be interpreted as full-target averages.
    No checkpoint selection rule is implemented or changed here. When explicitly
    requested, ``artifact`` retains the very arrays used for these metrics,
    including unmatched targets and duplicate prediction rows. It performs no
    further matching or prediction and is omitted by default.
    Optional ``node_input_support`` describes the actual visible encoder inputs,
    with support/nearest_seconds [N,L]. It adds diagnostic strata only: no true
    identity repairs group membership and no region changes the primary metric.
    """
    if not isinstance(return_artifact, bool):
        raise ValueError('return_artifact must be boolean')
    if split not in ALLOWED_SPLITS:
        raise ValueError('full-scene evaluation permits only explicit training/development splits; T_test is sealed')
    if grouping_kind not in ('predicted', 'oracle'):
        raise ValueError("grouping_kind must be 'predicted' or 'oracle'")
    if correspondence_policy != 'anchor_identity_first':
        raise ValueError("correspondence_policy must be 'anchor_identity_first'")
    nodes = _identities(node_identities, 'node_identities')
    targets = _identities(target_identities, 'target_identities')
    if len(set(targets)) != len(targets):
        raise ValueError('target_identities must be unique')
    target_index = {identity: index for index, identity in enumerate(targets)}
    if any(identity not in target_index for identity in nodes):
        raise ValueError('every node identity must have a declared target row, even if reference is unavailable')
    truth = _states(target_states, 'target_states')
    if len(truth) != len(targets):
        raise ValueError('target_states must have one row per target identity')
    truth_valid = _mask(target_valid, truth.shape, 'target_valid')
    _check_valid_states(truth, truth_valid, 'target_states')
    observed = _mask(node_observed, (len(nodes), truth.shape[1]), 'node_observed')
    if not isinstance(group_outputs, (list, tuple)) or any(not isinstance(r, dict) for r in group_outputs):
        raise ValueError('group_outputs must be a sequence of output records')
    if any('node_indices' not in r or 'states' not in r for r in group_outputs):
        raise ValueError('each output requires node_indices and explicit states (possibly None)')
    groups = _partition([record['node_indices'] for record in group_outputs], len(nodes))
    if grouping_kind == 'oracle' and groups != group_tracks_by_identity(nodes, node_order='input'):
        raise ValueError('oracle outputs must use the complete true-identity partition')

    target_observed = np.zeros(truth.shape[:2], bool)
    for index, identity in enumerate(nodes):
        target_observed[target_index[identity]] |= observed[index]
    shape = (len(groups), truth.shape[1], 4)
    prediction = np.full(shape, np.nan)
    valid = np.zeros(shape, bool)
    memberships = np.empty(len(nodes), dtype=int)
    first_group, anchor_rows, duplicate_groups = {}, [], []
    for index, (record, members) in enumerate(zip(group_outputs, groups)):
        memberships[members] = index
        anchor = target_index[nodes[members[0]]]
        anchor_rows.append(anchor)
        if anchor in first_group:
            duplicate_groups.append(index)
        else:
            first_group[anchor] = index
        if record['states'] is not None:
            states = _array(record['states'])
            if states.shape != shape[1:]:
                raise ValueError('group states must be aligned [L,4] physical values')
            prediction[index] = _states(states[None], 'group states')[0]
            valid[index] = _mask(_array(record.get('prediction_valid', np.ones(shape[1:], bool))),
                                 shape[1:], 'prediction_valid')
        elif 'prediction_valid' in record:
            mask = _mask(_array(record['prediction_valid']), shape[1:], 'prediction_valid')
            if mask.any():
                raise ValueError('missing group states cannot declare valid predictions')
    _check_valid_states(prediction, valid, 'group predictions')

    # Every target comes first; duplicates are appended with no target assigned.
    rows = len(targets) + len(duplicate_groups)
    aligned_prediction = np.full((rows, truth.shape[1], 4), np.nan)
    aligned_valid = np.zeros_like(aligned_prediction, dtype=bool)
    aligned_truth = np.full_like(aligned_prediction, np.nan)
    aligned_truth_valid = np.zeros_like(aligned_prediction, dtype=bool)
    aligned_observed = np.zeros((rows, truth.shape[1]), bool)
    aligned_truth[:len(targets)], aligned_truth_valid[:len(targets)] = truth, truth_valid
    aligned_observed[:len(targets)] = target_observed
    for target_row, group_index in first_group.items():
        aligned_prediction[target_row], aligned_valid[target_row] = prediction[group_index], valid[group_index]
    for row, group_index in enumerate(duplicate_groups, start=len(targets)):
        aligned_prediction[row], aligned_valid[row] = prediction[group_index], valid[group_index]
        aligned_observed[row] = target_observed[anchor_rows[group_index]]
    full = _regions(aligned_prediction, aligned_truth, aligned_truth_valid,
                    aligned_valid, aligned_observed)
    per_group = []
    for index, (members, anchor) in enumerate(zip(groups, anchor_rows)):
        per_group.append({
            'group_index': index, 'node_indices': members, 'anchor_target_id': str(targets[anchor]),
            'member_target_ids': list(dict.fromkeys(str(nodes[node]) for node in members)),
            'duplicate_for_anchor_target': index in duplicate_groups,
            'has_position_output': bool(valid[index, :, :2].all(axis=-1).any()),
            'metrics': _regions(prediction[index:index+1], truth[anchor:anchor+1],
                                truth_valid[anchor:anchor+1], valid[index:index+1],
                                target_observed[anchor:anchor+1]),
        })
    report = {
        'split': split, 'grouping_kind': grouping_kind,
        'correspondence_policy': correspondence_policy,
        'identity_usage': 'evaluation only; never repairs predicted groups or selects minimum error',
        'target_identity_count': len(targets), 'predicted_group_count': len(groups),
        'duplicate_prediction_group_count': len(duplicate_groups),
        'anchor_matched_target_count': len(first_group),
        'unmatched_target_ids': [str(identity) for index, identity in enumerate(targets) if index not in first_group],
        'selected_group_by_target': [first_group.get(index) for index in range(len(targets))],
        'reference_unavailable_target_count': int((~truth_valid.any(axis=(1, 2))).sum()),
        'group_diagnostics': group_membership_diagnostics(memberships, nodes),
        'full_target_metrics': full,
        'paper_fusion_metrics': paper_fusion_scene(
            aligned_prediction[:len(targets)], aligned_truth[:len(targets)],
            aligned_truth_valid[:len(targets)], prediction_valid=aligned_valid[:len(targets)]),
        'all_predicted_groups': per_group,
        'per_group_scope': 'all groups against their anchor identity; repeated targets, not a target-weighted metric',
        'region_definition': 'union of caller-declared original support over all nodes of each true identity',
        'test_opened': False,
    }
    aligned_groups = np.full(rows, -1, dtype=np.int64)
    for target_row, group_index in first_group.items():
        aligned_groups[target_row] = group_index
    aligned_groups[len(targets):] = duplicate_groups
    if node_input_support is not None:
        # This classification uses the supplied group nodes, never the union
        # of all nodes sharing their true identity. Missing targets stay rows.
        if np.asarray(node_input_support['support']).shape != (len(nodes), truth.shape[1]):
            raise ValueError('node_input_support must match all scene nodes and queries')
        group_support = group_input_support(node_input_support, groups)
        support_kind = np.full((rows, truth.shape[1]), 3, dtype=np.uint8)
        nearest = np.full((rows, truth.shape[1]), np.inf)
        for row, group_index in enumerate(aligned_groups):
            if group_index < 0:
                continue
            support_kind[row] = np.where(group_support['anchor_support'][group_index], 0,
                np.where(group_support['helper_only_support'][group_index], 1, 2))
            nearest[row] = group_support['nearest_seconds'][group_index]
        def metrics(mask):
            return evaluate_aligned_fusion(aligned_prediction, aligned_truth, aligned_truth_valid,
                                           prediction_valid=aligned_valid, region_mask=mask)
        report['input_support_metrics'] = {
            name: metrics(support_kind == index) for index, name in enumerate(INPUT_SUPPORT_REGIONS)}
        distance_masks = (nearest <= 300, (nearest > 300) & (nearest <= 600),
                          (nearest > 600) & (nearest <= 1800),
                          (nearest > 1800) & np.isfinite(nearest), ~np.isfinite(nearest))
        report['input_distance_metrics'] = {
            name: metrics(mask) for name, mask in zip(INPUT_DISTANCE_REGIONS, distance_masks)}
        report['input_support_definition'] = (
            'actual group encoder input time support; anchor is first group node; '
            'helper is another actual group node, regardless of true identity; '
            'interval support is not an exact measurement or proof of contribution; '
            'baseline singleton encoder visibility does not imply physical completion; '
            'unmatched targets remain separate; primary observed/missing metric unchanged')
        report['input_distance_definition'] = 'seconds to nearest actual visible group input point; no_input includes unmatched targets'
    if return_artifact:
        report['artifact'] = {
            'aligned_prediction': aligned_prediction,
            'aligned_prediction_valid': aligned_valid,
            'aligned_truth': aligned_truth,
            'aligned_truth_valid': aligned_truth_valid,
            'aligned_observed': aligned_observed,
            'target_ids': targets.astype(str),
            # Duplicate predictions have no assigned target in the full metric.
            'aligned_target_ids': np.concatenate((targets.astype(str), np.full(len(duplicate_groups), '', dtype=str))),
            'aligned_group_indices': aligned_groups,
            'group_states': prediction,
            'group_valid': valid,
            'group_anchor_target_indices': np.asarray(anchor_rows, dtype=np.int64),
            'group_offsets': np.concatenate(([0], np.cumsum([len(g) for g in groups], dtype=np.int64))),
            'group_node_indices': np.asarray([node for group in groups for node in group], dtype=np.int64),
            'node_group_indices': memberships,
            'split': split,
            'grouping_kind': grouping_kind,
            'correspondence_policy': correspondence_policy,
        }
        if node_input_support is not None:
            report['artifact'].update(aligned_input_support_kind=support_kind,
                aligned_nearest_input_seconds=nearest,
                input_support_region_names=np.asarray(INPUT_SUPPORT_REGIONS),
                input_distance_region_names=np.asarray(INPUT_DISTANCE_REGIONS))
    return report


def single_source_outputs(node_states, node_valid, node_identities, sources, *, source):
    """Oracle-identity single-source diagnostic on already aligned observations.

    Every input node remains in its true-identity group. Only the requested
    source contributes values. The first valid fragment in fixed input order
    supplies each point/feature; latitude/longitude are copied as a pair.
    No averaging, extrapolation, reference-assisted choice or new fusion occurs.
    Missing-source groups retain all-invalid outputs. Pass these records with
    grouping_kind='oracle'; this is not a predicted association baseline.
    """
    states = _states(node_states, 'node_states')
    valid = _mask(node_valid, states.shape, 'node_valid')
    _check_valid_states(states, valid, 'node_states')
    identities = _identities(node_identities, 'node_identities')
    sources = _identities(sources, 'sources')
    if len(states) != len(identities) or len(states) != len(sources):
        raise ValueError('states, identities and sources must describe the same nodes')
    if not isinstance(source, str) or not source:
        raise ValueError('source must be an explicit nonempty source ID')
    outputs = []
    for members in group_tracks_by_identity(identities, node_order='input'):
        values, mask = np.full(states.shape[1:], np.nan), np.zeros(states.shape[1:], bool)
        selected = [node for node in members if sources[node] == source]
        for node in selected:
            for columns in ((0, 1), (2,), (3,)):
                indices = list(columns)
                take = valid[node][:, indices].all(axis=-1) & ~mask[:, indices].all(axis=-1)
                values[np.ix_(take, indices)] = states[node][np.ix_(take, indices)]
                mask[np.ix_(take, indices)] = True
        outputs.append({'node_indices': members, 'states': values, 'prediction_valid': mask,
                        'status': 'single_source_oracle', 'source': source,
                        'source_node_indices': selected})
    return outputs


def simple_fusion_outputs(node_states, node_valid, node_identities, sources):
    """Equal-source observation fusion under oracle identity, diagnostic only.

    Each source first uses ``single_source_outputs``' fixed-fragment priority.
    At each point, only sources with valid features contribute. Coordinates
    require paired latitude/longitude: latitude is averaged arithmetically and
    longitude circularly across +/-180 degrees. Antipodal longitude cancellation
    leaves the whole coordinate pair unavailable. Heading is circularly averaged
    in [0,360), with opposing headings unavailable; speed is an arithmetic mean.
    No reference, extrapolation, precision weighting or fragment-count weighting
    is used. This is a simple physical diagnostic, not a geodesic estimator.
    """
    states = _states(node_states, 'node_states')
    valid = _mask(node_valid, states.shape, 'node_valid')
    _check_valid_states(states, valid, 'node_states')
    identities = _identities(node_identities, 'node_identities')
    sources = _identities(sources, 'sources')
    if len(states) != len(identities) or len(states) != len(sources):
        raise ValueError('states, identities and sources must describe the same nodes')
    source_ids = sorted(set(sources))
    per_source = [single_source_outputs(states, valid, identities, sources, source=source)
                  for source in source_ids]

    def circular_mean(values, mask, *, heading):
        radians = np.deg2rad(np.where(mask, values, 0))
        sine = np.where(mask, np.sin(radians), 0).sum(axis=0)
        cosine = np.where(mask, np.cos(radians), 0).sum(axis=0)
        count = mask.sum(axis=0)
        available = (count > 0) & (np.hypot(sine, cosine) > 1e-12 * count)
        angle = np.rad2deg(np.arctan2(sine, cosine))
        angle = (angle + 360) % 360 if heading else (angle + 180) % 360 - 180
        return np.where(available, angle, np.nan), available

    outputs = []
    for index, members in enumerate(group_tracks_by_identity(identities, node_order='input')):
        values = np.stack([rows[index]['states'] for rows in per_source])
        masks = np.stack([rows[index]['prediction_valid'] for rows in per_source])
        result = np.full(states.shape[1:], np.nan)
        available = np.zeros(states.shape[1:], bool)
        paired = masks[..., :2].all(axis=-1)
        longitude, position_valid = circular_mean(values[..., 1], paired, heading=False)
        count = paired.sum(axis=0)
        latitude = np.where(paired, values[..., 0], 0).sum(axis=0) / np.maximum(count, 1)
        result[:, 0] = np.where(position_valid, latitude, np.nan)
        result[:, 1] = longitude
        available[:, :2] = position_valid[:, None]
        speed_count = masks[..., 2].sum(axis=0)
        speed = np.where(masks[..., 2], values[..., 2], 0).sum(axis=0) / np.maximum(speed_count, 1)
        result[:, 2] = np.where(speed_count > 0, speed, np.nan)
        available[:, 2] = speed_count > 0
        result[:, 3], available[:, 3] = circular_mean(values[..., 3], masks[..., 3], heading=True)
        outputs.append({'node_indices': members, 'states': result, 'prediction_valid': available,
                        'status': 'simple_fusion_oracle', 'sources': source_ids})
    return outputs
