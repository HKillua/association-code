"""Re-solve the original MTAD graph at one rejoin batch with the trigger masked."""
from __future__ import annotations

from collections import defaultdict
from math import fsum

from .continuous_tracker import FRESH_S
from .hypothesis_gap_diagnostic import _margin
from .interval_assignment import PAIR_NULL_COST
from .nemot_episodes import _published
from .nemot_replay import _add, _distance_sq, _other, _solve_groups
from .relative_motion_gap import _same


def _selected(costs, fresh):
    groups = _solve_groups(costs, fresh)
    return _published(tuple(enumerate(groups)), fresh)


def _forecast(priors, key, stamp):
    prior = priors.get(key)
    if prior is None or not 0 < stamp - prior.stamp <= FRESH_S:
        raise ValueError("a frozen incident endpoint lacks its strict fresh prior")
    return prior.position(stamp, True)


def _registered(key, point, shifts):
    return _add(point, shifts.get(key[0], (0.0, 0.0)))


def _replace_incident(costs, trigger, stamp, priors, batch_points, shifts):
    observed = _registered(trigger, batch_points[trigger], shifts)
    forecast = _registered(trigger, _forecast(priors, trigger, stamp), shifts)
    replaced = {}
    for edge, old_cost in costs.items():
        if trigger not in edge:
            continue
        peer = _other(edge, trigger)
        point = batch_points[peer] if peer in batch_points else _forecast(priors, peer, stamp)
        center = _registered(peer, point, shifts)
        if not _same(_distance_sq(observed, center), old_cost):
            raise ValueError("an original trigger-incident KF cost did not reproduce")
        replaced[edge] = _distance_sq(forecast, center)
    return replaced


def _component(edges, seeds):
    neighbors = defaultdict(set)
    for left, right in edges:
        neighbors[left].add(right)
        neighbors[right].add(left)
    reached = set(seeds)
    pending = list(seeds)
    while pending:
        adjacent = neighbors[pending.pop()] - reached
        reached.update(adjacent)
        pending.extend(adjacent)
    return reached


def _state(pairs, switch, focal):
    focal_count = len(pairs & focal)
    retain_count = len(pairs.intersection(switch.retain))
    release_count = len(pairs.intersection(switch.release))
    fixed = switch.keys
    touched = {key for edge in pairs for key in edge if key in fixed}
    pattern = ("retain" if retain_count == 2 else "release" if release_count == 2
               else "partial_retain" if retain_count else "external_or_null")
    return dict(rejoined=bool(focal_count), focal_count=focal_count, retain_count=retain_count,
                release_count=release_count, external_edges=sum(
                    not set(edge) <= fixed and not fixed.isdisjoint(edge) for edge in pairs),
                unmatched_endpoints=len(fixed - touched), pattern=pattern)


def _affected(before, after, fixed):
    removed, added = before - after, after - before
    endpoints = {key for edge in removed | added for key in edge}
    return dict(removed_edges=len(removed), added_edges=len(added), changed_endpoints=len(endpoints),
                changed_fixed_endpoints=len(endpoints & fixed), changed_outside_endpoints=len(endpoints - fixed),
                removed_external_edges=sum(not set(edge) <= fixed for edge in removed),
                added_external_edges=sum(not set(edge) <= fixed for edge in added),
                removed_fully_external_edges=sum(fixed.isdisjoint(edge) for edge in removed),
                added_fully_external_edges=sum(fixed.isdisjoint(edge) for edge in added))


def _published_cost(pairs, costs):
    if not pairs <= costs.keys():
        return None
    # Pairs withheld because of a tie are absent here, so this describes the
    # published pairing cost relative to all-null, not the latent optimum.
    return fsum(costs[edge] - PAIR_NULL_COST for edge in pairs)


def score_global(switch, focal, trigger, frame, priors, batch_points, shifts):
    """Compare observed, all-incident masked, and fixed-incident masked solves.

    The gate edge set, fresh vertices, and nonincident cached costs are shared.
    Only the original trigger coordinate is replaced by its batch-prior KF
    forecast. The original pair solver still applies its own null costs and
    abstention for exact objective ties, as well as cross-source group rules.
    Selected edges and component keys are private inputs to a later label join.
    """
    stamp, updated, _, fresh_keys, cost_items, selected = frame
    costs, fresh, focal = dict(cost_items), set(fresh_keys), set(focal)
    fixed = switch.keys
    if (set(updated) != batch_points.keys() or set(updated) & fixed != {trigger}
            or not fixed <= fresh or not focal or not focal <= set(switch.retain)):
        raise ValueError("the frozen rejoin trigger, freshness or focal edges changed")
    if any(edge not in costs or costs[edge] >= PAIR_NULL_COST for edge in switch.edges):
        raise ValueError("the frozen switch no longer has its four legal edges")
    if any(left not in fresh or right not in fresh or left[0] == right[0] for left, right in costs):
        raise ValueError("the original HOLD graph has an ineligible endpoint")

    factual = _selected(costs, fresh)
    if factual != set(selected) or focal.isdisjoint(factual):
        raise ValueError("the full original rejoin assignment did not reproduce")
    replaced = _replace_incident(costs, trigger, stamp, priors, batch_points, shifts)
    fixed_edges = tuple(edge for edge in switch.edges if trigger in edge)
    masked_costs = costs | replaced
    fixed_costs = costs | {edge: replaced[edge] for edge in fixed_edges}
    if masked_costs.keys() != costs.keys() or fixed_costs.keys() != costs.keys():
        raise ValueError("masking changed the frozen HOLD graph")
    masked = _selected(masked_costs, fresh)
    fixed_mask = _selected(fixed_costs, fresh)
    local_masked = _selected({edge: masked_costs[edge] for edge in switch.edges}, fixed)

    component = _component(costs, fixed)
    eligible = {edge for edge in costs if min(costs[edge], masked_costs[edge]) <= PAIR_NULL_COST}
    eligible_component = _component(eligible, fixed)
    incident_null_losses = sum(costs[edge] < PAIR_NULL_COST <= value for edge, value in replaced.items())
    incident_null_gains = sum(value < PAIR_NULL_COST <= costs[edge] for edge, value in replaced.items())
    states = {"actual": factual, "masked": masked, "fixed_mask": fixed_mask}
    evaluations = {"actual": costs, "masked": masked_costs, "fixed_mask": fixed_costs}
    graph = dict(fresh_vertices=len(fresh), sources=len({key[0] for key in fresh}), candidate_edges=len(costs),
                 trigger_incident_edges=len(replaced), trigger_incident_costs_changed=sum(
                     not _same(costs[edge], value) for edge, value in replaced.items()),
                 fixed_incident_edges=len(fixed_edges), incident_null_losses=incident_null_losses,
                 incident_null_gains=incident_null_gains,
                 incident_null_ties=sum(value == PAIR_NULL_COST for value in replaced.values()),
                 actual_margin_m2=_margin(switch, costs.__getitem__),
                 masked_margin_m2=_margin(switch, masked_costs.__getitem__),
                 hold_component_vertices=len(component), hold_component_edges=sum(
                     set(edge) <= component for edge in costs),
                 eligible_union_component_vertices=len(eligible_component), eligible_union_component_edges=sum(
                     set(edge) <= eligible_component for edge in eligible),
                 published_null_adjusted_cost_m2={name: {arm: _published_cost(pairs, evaluation)
                                                        for arm, pairs in states.items()}
                                                 for name, evaluation in evaluations.items()})
    return {name: _state(pairs, switch, focal) for name, pairs in states.items()} | dict(
        selected_actual=tuple(sorted(factual)), selected_masked=tuple(sorted(masked)),
        selected_fixed_mask=tuple(sorted(fixed_mask)), component_keys=tuple(sorted(component)),
        local_masked=_state(local_masked, switch, focal), affected=_affected(factual, masked, fixed),
        affected_fixed_mask=_affected(factual, fixed_mask, fixed), graph=graph)
