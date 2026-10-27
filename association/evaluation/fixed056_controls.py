"""Fixed-056 controls using only visible observations and the frozen partition."""
from pathlib import Path

import numpy as np

from association.data.motion_evidence import build_motion_evidence
from association.data.physical_units import KNOT_TO_MPS, SPEED_UNIT_POLICY, unit_metadata
from association.evaluation.fusion_controls import ordinary_fusion


NAMES = ('canonical_all_anchor_ols', 'canonical_all_mean_supported_ols',
         'selected_cap128_anchor_ols', 'selected_cap128_mean_supported_ols')


def controls_from_visible(visible, groups, fragment):
    """Return output rows and support; never accept a reference/target array."""
    if visible['physical_units'] != unit_metadata(SPEED_UNIT_POLICY):
        raise ValueError('fixed056 controls require the explicit new m/s contract')
    for key, source in (('node_ids', 'node_ids'), ('sources', 'sources'), ('grid', 'query_seconds')):
        if not np.array_equal(np.asarray(visible[key]), np.asarray(fragment[source])):
            raise ValueError('fixed056 visible population/grid differs: '+key)
    n = len(visible['node_ids'])
    flat = np.asarray([node for group in groups for node in group], dtype=np.int64)
    offsets = np.r_[0, np.cumsum([len(group) for group in groups])]
    if (not np.array_equal(offsets, fragment['learned_assoc_group_offsets'])
            or not np.array_equal(flat, fragment['learned_assoc_group_node_indices'])):
        raise ValueError('fixed056 original groups and reconstructed score partition differ')
    membership = np.empty(n, dtype=np.int64)
    for index, group in enumerate(groups):
        membership[group] = index
    if not np.array_equal(membership, fragment['learned_assoc_membership']):
        raise ValueError('fixed056 group membership differs')
    full_states = np.asarray(fragment['native_states'], dtype=np.float64).copy()
    full_states[..., 2] *= KNOT_TO_MPS  # canonical cache remains untouched
    full = dict(states=full_states, support=np.asarray(fragment['support']),
                nearest_seconds=np.asarray(fragment['nearest_seconds']))
    cap = build_motion_evidence(visible['native_values'], visible['native_times'],
        visible['native_observed'], visible['grid'], visible['sources'])
    result = {}
    for label, motion in (('canonical_all', full), ('selected_cap128', cap)):
        controls = ordinary_fusion(motion['states'], motion['nearest_seconds'], motion['support'], groups)
        for arm, suffix in (('anchor', 'anchor_ols'), ('mean_supported', 'mean_supported_ols')):
            predicted = controls['predictions'][arm]
            result[label+'_'+suffix] = dict(
                generated=[dict(node_indices=group, states=predicted[group[0]]) for group in groups],
                support=dict(support=motion['support'], nearest_seconds=motion['nearest_seconds']))
    return result


class Fixed056Controls:
    def __init__(self, source_system):
        self.source = Path(source_system)

    def __call__(self, visible, groups):
        path = self.source/'predictions'/f"states-{visible['scene_id']}.npz"
        keys = ('node_ids', 'sources', 'query_seconds', 'learned_assoc_group_offsets',
                'learned_assoc_group_node_indices', 'learned_assoc_membership',
                'native_states', 'support', 'nearest_seconds')
        with np.load(path, allow_pickle=False) as saved:
            fragment = {key: saved[key].copy() for key in keys}
        return controls_from_visible(visible, groups, fragment)
