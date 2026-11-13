"""Audit use of a rejoin report in the original MTAD graph and shape."""
from __future__ import annotations

from .continuous_tracker import FRESH_S
from .hypothesis_gap_diagnostic import _direction, _margin
from .interval_assignment import PAIR_NULL_COST
from .nemot_replay import GATE_SQUARED, _add, _distance_sq, _other
from .relative_motion_gap import _same


_SIDES = ("release", "retain")
_ZERO = (0.0, 0.0)


def _registered(point, key, shifts):
    return _add(point, shifts.get(key[0], _ZERO))


def _inside(cost, limit, inclusive=False):
    return cost is not None and (cost <= limit if inclusive else cost < limit)


def _transition(actual, masked):
    return dict(actual=sum(actual), masked=sum(masked),
                loses=sum(a and not m for a, m in zip(actual, masked, strict=True)),
                gains=sum(m and not a for a, m in zip(actual, masked, strict=True)))


def _graph(switch, trigger, stamp, priors, batch_points, shifts, frame, previous_frame, focal):
    time, updated, _, fresh, costs, selected = frame
    if time != stamp or set(updated) != batch_points.keys() or set(updated) & switch.keys != {trigger}:
        raise ValueError("the graph frame does not contain the fixed single-report boundary")
    if previous_frame[0] >= stamp:
        raise ValueError("the previous graph frame must be strictly before the boundary")
    costs, previous = dict(costs), dict(previous_frame[4])
    if any(not _inside(costs.get(edge), PAIR_NULL_COST) for edge in switch.edges):
        raise ValueError("the original switch has an ineligible rejoin edge")
    selected, previous_selected = set(selected), set(previous_frame[5])
    if not selected.intersection(focal):
        raise ValueError("the original rejoin did not select any focal edge")

    old = priors.get(trigger)
    if old is None or not 0 < stamp - old.stamp <= FRESH_S:
        raise ValueError("the trigger lacks its frozen batch-prior forecast")
    actual = _registered(batch_points[trigger], trigger, shifts)
    masked = _registered(old.position(stamp, True), trigger, shifts)
    peers = [key for key in fresh if key[0] != trigger[0]]
    by_edge = {}
    for peer in peers:
        if peer in batch_points:
            raw_peer = kf_peer = batch_points[peer]
        else:
            prior = priors.get(peer)
            if prior is None or not 0 < stamp - prior.stamp <= FRESH_S:
                raise ValueError("a fresh opposite-source peer lacks its original prior")
            raw_peer, kf_peer = prior.observation, prior.position(stamp, True)
        hold_peer = _registered(raw_peer, peer, shifts)
        kf_peer = _registered(kf_peer, peer, shifts)
        edge = tuple(sorted((trigger, peer)))
        hold_actual = _distance_sq(actual, hold_peer)
        hold_masked = _distance_sq(masked, hold_peer)
        kf_actual = _distance_sq(actual, kf_peer)
        kf_masked = _distance_sq(masked, kf_peer)
        actual_gate = _inside(hold_actual, GATE_SQUARED, inclusive=True)
        masked_gate = _inside(hold_masked, GATE_SQUARED, inclusive=True)
        if actual_gate != (edge in costs) or (actual_gate and not _same(kf_actual, costs[edge])):
            raise ValueError("the reconstructed raw HOLD gate or KF cost differs from the original graph")
        by_edge[edge] = dict(hold_actual_m2=hold_actual, hold_masked_m2=hold_masked,
                             kf_actual_m2=kf_actual, kf_masked_m2=kf_masked,
                             actual_gate=actual_gate, masked_gate=masked_gate,
                             actual_legal=actual_gate and _inside(kf_actual, PAIR_NULL_COST),
                             masked_legal=masked_gate and _inside(kf_masked, PAIR_NULL_COST),
                             previous_gate=edge in previous, previous_legal=_inside(previous.get(edge), PAIR_NULL_COST),
                             previous_cost_m2=previous.get(edge), selected=edge in selected,
                             previous_selected=edge in previous_selected)
    if {edge for edge in costs if trigger in edge} != {edge for edge, row in by_edge.items() if row["actual_gate"]}:
        raise ValueError("an original incident candidate is absent from the fresh-peer reconstruction")

    roles = {side: next(edge for edge in getattr(switch, side) if trigger in edge) for side in _SIDES}
    fixed = {side: by_edge[edge] for side, edge in roles.items()}
    local = list(fixed.values())
    incident = list(by_edge.values())
    hypothetical = costs | {edge: by_edge[edge]["kf_masked_m2"] for edge in roles.values()}

    def transitions(items):
        return {test: _transition([row[f"actual_{test}"] for row in items],
                                  [row[f"masked_{test}"] for row in items]) for test in ("gate", "legal")}

    def graph_changes(edges):
        return {
            "gate": _transition([edge in previous for edge in edges], [edge in costs for edge in edges]),
            "legal": _transition([_inside(previous.get(edge), PAIR_NULL_COST) for edge in edges],
                                 [_inside(costs.get(edge), PAIR_NULL_COST) for edge in edges]),
        }

    old_incident = {edge for edge in previous if trigger in edge}

    return dict(trigger_coordinate_read_by_gate=True, trigger_coordinate_read_by_kf_cost=bool(incident),
                peers=len(peers), peers_same_batch=sum(peer in batch_points for peer in peers),
                incident=transitions(incident), fixed_incident=transitions(local),
                incident_previous_to_actual=graph_changes(old_incident | by_edge.keys()),
                fixed_incident_previous_to_actual=graph_changes(tuple(roles.values())),
                fixed_edges=fixed,
                incident_costs_changed=sum(row["actual_gate"] and not _same(row["kf_actual_m2"], row["kf_masked_m2"])
                                           for row in incident),
                fixed_incident_costs_changed=sum(not _same(row["kf_actual_m2"], row["kf_masked_m2"]) for row in local),
                factual_cached_margin_m2=_margin(switch, costs.__getitem__),
                coordinate_mask_cached_margin_m2=_margin(switch, hypothetical.__getitem__),
                retain_selected=set(switch.retain) <= selected,
                release_selected=set(switch.release) <= selected,
                focal_selected=len(set(focal) & selected),
                previous_retain_selected=set(switch.retain) <= previous_selected)


def _shape(trigger, stamp, boundary, switch):
    shape = boundary["shape"]
    geometry = boundary["geometry"]
    if boundary["stamp"] != stamp:
        raise ValueError("the frozen shape uses a different decision time")
    support = shape["provenance"]
    used = support is not None and stamp in support[trigger]
    if bool(shape["current_used"]) != used:
        raise ValueError("the original shape flag disagrees with its report provenance")
    if support is not None and (set(support) != switch.keys or any(t > stamp for times in support.values() for t in times)
                                or shape["available_at"] != max(t for times in support.values() for t in times)):
        raise ValueError("shape provenance does not belong to the frozen causal window")
    actual = shape["actual"]["two_block"] if shape["actual"] is not None else None
    masked = shape["masked"]["two_block"] if shape["masked"] is not None else None
    if (actual is None) != (masked is None) or (actual is None) != (support is None):
        raise ValueError("original shape and coordinate-mask support differ")
    if actual is not None and not used and not _same(actual, masked):
        raise ValueError("masking an unused report changed the shape")

    def action(geometry_margin, shape_margin):
        return None if shape_margin is None else _direction(geometry_margin) > 0 and _direction(shape_margin) > 0

    return dict(status=shape["status"], available_at=shape["available_at"],
                trigger_coordinate_read_by_geometry=True, trigger_in_shape_provenance=used,
                shape_actual_m2=actual, shape_masked_m2=masked,
                geometry_actual_m2=geometry["actual"], geometry_masked_m2=geometry["masked"],
                current_action=action(geometry["actual"], actual),
                shape_only_mask_action=action(geometry["actual"], masked),
                geometry_only_mask_action=action(geometry["masked"], actual),
                full_coordinate_mask_action=action(geometry["masked"], masked))


def audit_boundary(switch, trigger, stamp, priors, batch_points, shifts, frame, previous_frame, focal, shape):
    """Measure trigger-coordinate dependence without changing the frozen graph.

    Gate and cost counterfactuals replace only the trigger coordinate at the
    same clock. Cached costs of nonincident edges and original selections are
    preserved. Candidate threshold changes and fixed four-edge margin changes
    describe local sensitivity, without re-solving the full assignment.
    """
    return dict(graph=_graph(switch, trigger, stamp, priors, batch_points, shifts, frame, previous_frame, focal),
                shape=_shape(trigger, stamp, shape, switch))
