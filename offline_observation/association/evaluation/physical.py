"""CPU metrics for explicitly aligned physical trajectories and node groups.

This module does not align clocks, assign predictions to truth targets, choose
thresholds/checkpoints, or create a combined penalty. The caller supplies the
full aligned row population, including missed targets (prediction_valid=False)
and unmatched predictions (target_valid=False). Do not drop these rows first.
Observed/missing regions are separate calls with caller-provided region_mask.
"""

from collections import Counter, defaultdict

import numpy as np


EARTH_RADIUS_METRES = 6_371_000.0  # Explicit spherical approximation, not an ellipsoid.
STATE_FIELDS = ("lat", "lon", "vel", "cou")


def _states(values, name):
    values = np.asarray(values)
    if values.ndim != 3 or values.shape[-1] != 4 or values.dtype.kind not in "iuf":
        raise ValueError(f"{name} must be a numeric [N,L,4] array in lat/lon/vel/cou order")
    return values.astype(np.float64, copy=False)


def _mask(values, shape, name):
    values = np.asarray(values)
    if values.dtype != np.bool_ or values.shape != shape:
        raise ValueError(f"{name} must be a boolean array with shape {shape}")
    return values


def _check_valid_states(values, valid, name):
    # Check all values declared valid, even outside this call's region. Region
    # selection must not hide invalid model output or reference declarations.
    if not np.isfinite(values[valid]).all():
        raise ValueError(f"{name} contains nonfinite values declared valid")
    for column, low, high in ((0, -90, 90), (1, -180, 180)):
        selected = values[..., column][valid[..., column]]
        if ((selected < low) | (selected > high)).any():
            raise ValueError(f"{name} valid {STATE_FIELDS[column]} is out of range")
    if (values[..., 2][valid[..., 2]] < 0).any():
        raise ValueError(f"{name} valid native speed must be nonnegative")
    # Heading outputs may be unwrapped (e.g. 361 degrees); circular error is
    # invariant to complete turns. No heading clipping or unit conversion.


def _coverage(target_available, prediction_available):
    intersection = target_available & prediction_available
    target_count = int(target_available.sum())
    target_trajectories = int(target_available.any(axis=1).sum())
    covered_trajectories = int(intersection.any(axis=1).sum())
    count = int(intersection.sum())
    return intersection, {
        "available_target_count": target_count,
        "available_prediction_count": int(prediction_available.sum()),
        "intersection_count": count,
        "missing_prediction_count": target_count - count,
        "prediction_without_target_count": int((prediction_available & ~target_available).sum()),
        "coverage": count / target_count if target_count else None,
        "available_target_trajectory_count": target_trajectories,
        "covered_target_trajectory_count": covered_trajectories,
        "missing_target_trajectory_count": target_trajectories - covered_trajectories,
        "trajectory_coverage": covered_trajectories / target_trajectories if target_trajectories else None,
    }


def _error_statistics(errors):
    if not len(errors):
        return {"mae": None, "mse": None, "rmse": None}
    # Masking has already happened. Never multiply a NaN error by zero.
    with np.errstate(over="raise", invalid="raise"):
        mean_absolute = float(np.mean(np.abs(errors)))
        mean_squared = float(np.mean(np.square(errors)))
    return {"mae": mean_absolute, "mse": mean_squared, "rmse": float(np.sqrt(mean_squared))}


def evaluate_aligned_fusion(prediction, target, target_valid, *, prediction_valid, region_mask):
    """Evaluate explicitly aligned [N,L,4] native state arrays, without fitting.

    Both feature-validity masks are required [N,L,4] bool arrays; region_mask is
    a required [N,L] bool array, for example an explicitly defined observed or
    missing interval. True masks require finite values and legal latitude,
    longitude and nonnegative native speed. Masked values can be NaN and are
    never substituted with zero. Heading is measured in degrees modulo 360.

    Each quantity reports point counts and trajectory counts. Coverage uses all
    available target points in the chosen region, including missing predictions.
    Errors are conditional on the reported intersection, and are null when it is
    empty. Position needs both lat/lon; speed and heading use their own masks.
    MAE/MSE/RMSE units are respectively unit/unit squared/unit (metres, native
    speed, degrees). Position errors are spherical great-circle distances.

    This function cannot infer target correspondence from an association graph.
    The caller must supply a declared, complete row alignment; no Hungarian
    matching, reference-assisted prediction, or research selection rule is used.
    """
    prediction = _states(prediction, "prediction")
    target = _states(target, "target")
    if prediction.shape != target.shape:
        raise ValueError("prediction and target must have identical shapes")
    target_valid = _mask(target_valid, target.shape, "target_valid")
    prediction_valid = _mask(prediction_valid, prediction.shape, "prediction_valid")
    region = _mask(region_mask, target.shape[:2], "region_mask")
    _check_valid_states(target, target_valid, "target")
    _check_valid_states(prediction, prediction_valid, "prediction")
    result = {
        "aligned_row_count": target.shape[0], "time_slot_count": target.shape[1],
        "region_point_count": int(region.sum()), "state_fields": list(STATE_FIELDS),
        "error_scope": "available target/prediction intersection; read errors together with coverage",
        "alignment": "caller supplied; no target matching performed",
    }
    for name, columns, unit in (("position", (0, 1), "metres"),
                                ("speed", (2,), "native_speed"),
                                ("heading", (3,), "degrees")):
        target_available = target_valid[..., list(columns)].all(axis=-1) & region
        prediction_available = prediction_valid[..., list(columns)].all(axis=-1) & region
        intersection, coverage = _coverage(target_available, prediction_available)
        actual = prediction[intersection]
        truth = target[intersection]
        if name == "position":
            lat1, lat2 = np.deg2rad(actual[:, 0]), np.deg2rad(truth[:, 0])
            dlat = lat1 - lat2
            dlon = np.deg2rad((actual[:, 1] - truth[:, 1] + 180.0) % 360.0 - 180.0)
            haversine = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
            haversine = np.clip(haversine, 0.0, 1.0)  # Roundoff near coincident/antipodal points.
            errors = 2 * EARTH_RADIUS_METRES * np.arctan2(np.sqrt(haversine), np.sqrt(1 - haversine))
        elif name == "speed":
            errors = actual[:, 2] - truth[:, 2]
        else:
            # Reduce each finite angle first, avoiding huge-value subtraction.
            errors = ((actual[:, 3] % 360) - (truth[:, 3] % 360) + 180) % 360 - 180
        result[name] = {**coverage, **_error_statistics(errors), "unit": unit}
    result["position"]["earth_radius_metres"] = EARTH_RADIUS_METRES
    return result


def _memberships(values, name, allow_unassigned):
    values = np.asarray(values, dtype=object)
    if values.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional membership vector")
    normalized = []
    unassigned = 0
    for index, value in enumerate(values):
        if value is None and allow_unassigned:
            normalized.append(("unassigned_node", index))
            unassigned += 1
        elif isinstance(value, (str, np.str_)) and value:
            normalized.append(("str", str(value)))
        elif isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_)):
            normalized.append(("int", int(value)))
        else:
            raise ValueError(f"{name} must contain nonempty strings or integer IDs"
                             + (" (None explicitly means unassigned)" if allow_unassigned else ""))
    return normalized, unassigned


def group_membership_diagnostics(predicted_groups, target_identities):
    """Split/merge counts on the same full ordered node population, evaluation only.

    Group IDs need not equal target IDs. A None prediction is an unassigned node:
    it is separate from every other node for pair decisions and explicitly
    counted. Known target identity is required for every node. Caller retains
    isolated/unassigned nodes; this function never chooses or matches groups.
    """
    predicted, unassigned = _memberships(predicted_groups, "predicted_groups", True)
    truth, _ = _memberships(target_identities, "target_identities", False)
    if len(predicted) != len(truth):
        raise ValueError("predicted_groups and target_identities must describe the same nodes")
    joint = Counter(zip(truth, predicted))
    target_counts, prediction_counts = Counter(truth), Counter(predicted)
    choose_two = lambda count: count * (count - 1) // 2
    tp = sum(choose_two(count) for count in joint.values())
    truth_pairs = sum(choose_two(count) for count in target_counts.values())
    predicted_pairs = sum(choose_two(count) for count in prediction_counts.values())
    split_pairs, merge_pairs = truth_pairs - tp, predicted_pairs - tp
    target_to_prediction, prediction_to_target = defaultdict(set), defaultdict(set)
    for target_id, group_id in joint:
        target_to_prediction[target_id].add(group_id)
        prediction_to_target[group_id].add(target_id)
    n = len(truth)
    return {
        "node_count": n, "pair_count": choose_two(n),
        "target_group_count": len(target_counts),
        "predicted_assigned_group_count": len(prediction_counts) - unassigned,
        "effective_partition_group_count": len(prediction_counts),
        "unassigned_node_count": unassigned,
        "assigned_node_coverage": (n - unassigned) / n if n else None,
        "correct_same_identity_pairs": tp,
        "split_pairs": split_pairs, "merge_pairs": merge_pairs,
        "correct_different_identity_pairs": choose_two(n) - tp - split_pairs - merge_pairs,
        "split_target_group_count": sum(len(groups) > 1 for groups in target_to_prediction.values()),
        "merged_predicted_group_count": sum(len(identities) > 1 for identities in prediction_to_target.values()),
        "scope": "full aligned node population; pair counts, not trajectory matching or a composite score",
    }
