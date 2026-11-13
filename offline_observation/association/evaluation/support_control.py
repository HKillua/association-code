"""062 observable support switch, before any identity or reference alignment."""
from __future__ import annotations

import numpy as np


def support_switch_prediction(anchor, mean, support, groups):
    """Return complete per-node states and the position-switch mask.

    Every node is evaluated as a potential anchor, including nodes in duplicate
    output groups. The caller later selects the unchanged first node of each
    group. Position copies the entire existing Mean candidate only when the
    anchor lacks temporal support and another group member has support. Speed
    and heading always copy Mean. No identity, reference or validity mask is an
    input; support denotes the observed time interval, not measurement quality.
    """
    anchor = np.asarray(anchor, dtype=np.float64)
    mean = np.asarray(mean, dtype=np.float64)
    support = np.asarray(support)
    if (anchor.ndim != 3 or anchor.shape[-1] != 4 or anchor.shape[1] == 0
            or mean.shape != anchor.shape or support.shape != anchor.shape[:2]
            or support.dtype != bool or not np.isfinite(anchor).all()
            or not np.isfinite(mean).all()):
        raise ValueError('support switch requires finite [N,Q,4] states and boolean [N,Q] support')
    n = len(anchor)
    flat = [node for group in groups for node in group]
    if (any(not isinstance(node, (int, np.integer)) or isinstance(node, (bool, np.bool_))
            for node in flat)
            or sorted(flat) != list(range(n))
            or any(not len(group) or list(group) != sorted(group) for group in groups)):
        raise ValueError('fixed groups must partition every node once in input order')
    switched = np.zeros(support.shape, dtype=bool)
    for group in groups:
        members = np.asarray(group, dtype=np.int64)
        supported_count = support[members].sum(axis=0)
        # With an unsupported anchor, any supported member is necessarily a helper.
        switched[members] = ~support[members] & (supported_count > 0)
    prediction = mean.copy()
    prediction[..., :2] = np.where(switched[..., None], mean[..., :2], anchor[..., :2])
    return prediction, switched
