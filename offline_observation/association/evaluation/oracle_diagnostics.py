"""Fixed-candidate Oracle selection and paired full-population comparisons.

These helpers serve 060/061. The closed 058 identity-grouping experiment lives
in Git history; reference-error selection is diagnostic, never deployable.
"""
from __future__ import annotations

import numpy as np

from association.evaluation.fusion_controls import attribute_errors
from association.evaluation.system_attribution import (
    ATTRIBUTES, _metrics,
)

ARMS = ('anchor', 'mean_supported', 'attribute_hybrid', 'oracle_selector')
POPULATION_KEYS = ('target_identity_count', 'predicted_group_count',
                   'duplicate_prediction_group_count', 'anchor_matched_target_count',
                   'reference_unavailable_target_count')


def select_candidates(anchor, mean, truth, truth_valid, prediction_valid=None):
    """Choose an entire position pair or scalar attribute; ties/invalid truth use Anchor.

    Both finite candidates have the same input availability. Reference validity
    controls selection/evaluation, never prediction coverage. Returned choice
    columns are position/speed/heading, not latitude/longitude separately.
    """
    anchor, mean, truth = (np.asarray(x, np.float64) for x in (anchor, mean, truth))
    truth_valid = np.asarray(truth_valid)
    if (anchor.shape != mean.shape or anchor.shape != truth.shape or anchor.ndim != 3
            or anchor.shape[-1] != 4 or truth_valid.shape != truth.shape
            or truth_valid.dtype != bool or not np.isfinite(anchor).all()
            or not np.isfinite(mean).all()
            or not np.isfinite(truth[truth_valid]).all()):
        raise ValueError('aligned finite candidates and explicit reference masks required')
    valid = np.ones(anchor.shape, bool) if prediction_valid is None else np.asarray(prediction_valid)
    if valid.shape != anchor.shape or valid.dtype != bool:
        raise ValueError('shared candidate prediction_valid must be boolean and aligned')
    result, choice = anchor.copy(), np.zeros(anchor.shape[:-1] + (3,), bool)
    errors = [attribute_errors(values, truth) for values in (anchor, mean)]
    for index, (attr, columns) in enumerate(ATTRIBUTES.items()):
        eligible = (truth_valid[..., list(columns)] & valid[..., list(columns)]).all(-1)
        choose = eligible & (errors[1][attr] < errors[0][attr])
        choice[..., index] = choose
        for column in columns:
            result[..., column] = np.where(choose, mean[..., column], anchor[..., column])
    return result, valid.copy(), choice


def _comparison(a, b, artifacts, representatives, truth, truth_valid, regions, arrays):
    """Paired common points plus gained/lost coverage; no survivor-only ranking."""
    t, q = truth.shape[:2]
    first, second = artifacts[a], artifacts[b]
    av, bv = (r['aligned_prediction_valid'][:t] for r in (first, second))
    common = truth_valid & av & bv
    ra, rb = representatives[a], representatives[b]
    represented = (ra >= 0) & (rb >= 0)
    same, changed = represented & (ra == rb), represented & (ra != rb)
    paired_regions = dict(regions,
        same_representative=np.broadcast_to(same[:, None], (t, q)),
        changed_representative=np.broadcast_to(changed[:, None], (t, q)))
    result = dict(methods={name: _metrics(artifacts[name]['aligned_prediction'][:t],
        truth, truth_valid, common, paired_regions) for name in (a, b)},
        same_representative_target_count=int(same.sum()),
        changed_representative_target_count=int(changed.sum()),
        error_change_second_minus_first={}, coverage_transitions={})
    errors = {name: attribute_errors(artifacts[name]['aligned_prediction'][:t], truth) for name in (a, b)}
    gained, lost = np.zeros_like(common), np.zeros_like(common)
    for attr, columns in ATTRIBUTES.items():
        reference = truth_valid[..., list(columns)].all(-1)
        old = reference & av[..., list(columns)].all(-1)
        new = reference & bv[..., list(columns)].all(-1)
        gain, loss = new & ~old, old & ~new
        result['coverage_transitions'][attr] = dict(
            rescued_target_count=int((new.any(1) & ~old.any(1)).sum()),
            lost_target_count=int((old.any(1) & ~new.any(1)).sum()),
            gained_target_points=int(gain.sum()), lost_target_points=int(loss.sum()))
        for column in columns:
            gained[..., column], lost[..., column] = gain, loss
    for region, mask in paired_regions.items():
        result['error_change_second_minus_first'][region] = {}
        for attr, columns in ATTRIBUTES.items():
            use = common[..., list(columns)].all(-1) & mask
            delta = (errors[b][attr] - errors[a][attr])[use]
            better, worse = delta < -1e-9, delta > 1e-9
            result['error_change_second_minus_first'][region][attr] = dict(
                common_point_count=len(delta), second_better_point_count=int(better.sum()),
                second_worse_point_count=int(worse.sum()), tied_point_count=int((~better & ~worse).sum()),
                benefit_sum=float(-delta[better].sum()), harm_sum=float(delta[worse].sum()),
                signed_error_change_sum=float(delta.sum()),
                mean_error_change=float(delta.mean()) if len(delta) else None)
    result['newly_covered'] = _metrics(second['aligned_prediction'][:t], truth, truth_valid, gained, regions)
    result['lost_coverage'] = _metrics(first['aligned_prediction'][:t], truth, truth_valid, lost, regions)
    prefix = a + '__vs__' + b
    arrays[prefix + '_common_valid'] = common
    arrays[prefix + '_newly_covered_valid'] = gained
    arrays[prefix + '_lost_coverage_valid'] = lost
    return result
