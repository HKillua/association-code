"""C146 exact bipartite assignment on the C145 cross-source scores."""
from __future__ import annotations

import heapq
import math
from numbers import Integral

from tools.analysis.offline_baseline_core import ACCEPT_METRES


def _weighted_graph(chain_sources, edges):
    sources = tuple(chain_sources)
    source_names = sorted(set(sources))
    if len(source_names) > 2:
        raise ValueError('C146 assignment requires at most two source names')
    canonical = {}
    for first, second, cost in edges:
        left, right = sorted((first, second))
        if not (0 <= left < right < len(sources)) or sources[left] == sources[right]:
            raise ValueError('cross-source chain indices must be distinct and in range')
        if (left, right) in canonical:
            raise ValueError('duplicate cross-source chain pair')
        canonical[left, right] = cost
    return sources, canonical


def _components(sources, weights):
    if not weights:
        return
    left_source = min(sources)
    adjacent = {}
    for i, j in weights:
        adjacent.setdefault(i, []).append(j)
        adjacent.setdefault(j, []).append(i)
    seen = set()
    for root in sorted(adjacent):
        if root in seen:
            continue
        pending, vertices = [root], []
        seen.add(root)
        while pending:
            vertex = pending.pop()
            vertices.append(vertex)
            for other in adjacent[vertex]:
                if other not in seen:
                    seen.add(other)
                    pending.append(other)
        left = sorted(i for i in vertices if sources[i] == left_source)
        right = sorted(i for i in vertices if sources[i] != left_source)
        edges = [(i, j, weights[tuple(sorted((i, j)))])
                 for i in left for j in sorted(adjacent[i])]
        yield root, left, right, edges


def _minimum_cost(left, right, edges, exact_count, stop_increment):
    """Find the fixed-size optimum and the cheapest marginal-cost stopping point.

    Each shortest augmenting path extends a minimum-cost k-matching to a
    minimum-cost (k+1)-matching. Its marginal cost cannot decrease. All costs
    and reduced costs are exact Python integers. The free-cardinality answer
    uses the first optimum, before any zero-benefit addition.
    """
    source, sink = 0, len(left) + len(right) + 1
    graph = [[] for _ in range(sink + 1)]
    left_indices = {node: i for i, node in enumerate(left, 1)}
    right_indices = {node: i for i, node in enumerate(right, len(left) + 1)}

    def add(start, end, cost):
        forward = [end, len(graph[end]), 1, cost]
        graph[start].append(forward)
        graph[end].append([start, len(graph[start]) - 1, 0, -cost])
        return forward

    for node in left_indices.values():
        add(source, node, 0)
    for node in right_indices.values():
        add(node, sink, 0)
    real_edges = [(tuple(sorted((i, j))), add(left_indices[i], right_indices[j], cost))
                  for i, j, cost in edges]

    def selected():
        return sorted(pair for pair, edge in real_edges if edge[2] == 0)

    potential = [0] * len(graph)
    fixed = [] if exact_count == 0 else None
    optional = None
    flows = 0
    while fixed is None or optional is None:
        # Stable vertex ordering and strict improvement provide deterministic ties.
        distance = [None] * len(graph)
        predecessor = [None] * len(graph)
        distance[source] = 0
        queue = [(0, source)]
        while queue:
            cost, vertex = heapq.heappop(queue)
            if distance[vertex] != cost:
                continue
            for edge_index, (other, _, capacity, price) in enumerate(graph[vertex]):
                if not capacity:
                    continue
                candidate = cost + price + potential[vertex] - potential[other]
                if distance[other] is None or candidate < distance[other]:
                    distance[other] = candidate
                    predecessor[other] = (vertex, edge_index)
                    heapq.heappush(queue, (candidate, other))
        if distance[sink] is None:
            if fixed is None:
                raise ValueError('fixed-size assignment infeasible on the declared candidates')
            optional = selected()
            break
        increment = distance[sink] + potential[sink] - potential[source]
        if optional is None and increment >= stop_increment:
            optional = selected()
            if fixed is not None:
                break

        for index, cost in enumerate(distance):
            if cost is not None:
                potential[index] += cost
        vertex = sink
        while vertex != source:
            before, edge_index = predecessor[vertex]
            edge = graph[before][edge_index]
            edge[2] = 0
            graph[vertex][edge[1]][2] = 1
            vertex = before
        flows += 1
        if flows == exact_count:
            fixed = selected()
    return fixed, optional, flows


def assign(chain_sources, candidates, greedy_links):
    """Return the two C146 decisions and per-component counts/objectives.

    Candidate distances are nonnegative float metres. The accepted C145 graph
    is distance <= 5000. The stored floating-point costs are converted exactly
    to scaled binary integers, with no cost rounding or tie epsilon. The fixed
    arm minimizes distance at the greedy cardinality of each connected component.
    The null arm maximizes total 5000-distance, with independent unmatched
    chains at zero utility and fewer real edges preferred in exact utility ties.
    Returned chain pairs are canonical, in original chain order.
    """
    sources, distances = _weighted_graph(chain_sources, (
        (i, j, float(distance)) for i, j, distance in candidates))
    if any(not math.isfinite(cost) or cost < 0 for cost in distances.values()):
        raise ValueError('assignment requires finite nonnegative distances')
    eligible = {pair: cost for pair, cost in distances.items() if cost <= ACCEPT_METRES}
    greedy = {tuple(sorted(pair)) for pair in greedy_links}
    if (len(greedy) != len(greedy_links) or not greedy <= eligible.keys()
            or len({i for pair in greedy for i in pair}) != 2 * len(greedy)):
        raise ValueError('greedy links must be a matching on the eligible graph')
    arms = {'GLOBAL_FIXED': [], 'EXPLICIT_NULL': []}
    audit = []
    for root, left, right, edges in _components(sources, eligible):
        pairs = [tuple(sorted((i, j))) for i, j, _ in edges]
        ratios = [cost.as_integer_ratio() for _, _, cost in edges]
        denominator = max(divisor for _, divisor in ratios)
        costs = {pair: value * (denominator // divisor)
                 for pair, (value, divisor) in zip(pairs, ratios, strict=True)}
        greedy_in_component = greedy.intersection(pairs)
        quota = len(greedy_in_component)
        cutoff = int(ACCEPT_METRES) * denominator
        fixed, optional, flows = _minimum_cost(
            left, right, [(i, j, costs[pair]) for (i, j, _), pair in zip(edges, pairs, strict=True)],
            quota, cutoff)
        arms['GLOBAL_FIXED'].extend(fixed)
        arms['EXPLICIT_NULL'].extend(optional)
        row = dict(component_min_chain=root, eligible_edges=len(edges), left_chains=len(left),
                   right_chains=len(right), augmentations=flows)
        totals, utilities = {}, {}
        for name, links in (('greedy', greedy_in_component), ('global_fixed', fixed), ('explicit_null', optional)):
            total = sum(costs[pair] for pair in links)
            utility = cutoff * len(links) - total
            totals[name], utilities[name] = total, utility
            row[f'{name}_count'] = len(links)
            row[f'{name}_cost_m'] = total / denominator
            row[f'{name}_utility_m'] = utility / denominator
        if (len(fixed) != quota or totals['global_fixed'] > totals['greedy']
                or utilities['explicit_null'] < utilities['global_fixed']):
            raise RuntimeError('assignment objective or fixed cardinality invariant violated')
        audit.append(row)
    return {name: sorted(pairs) for name, pairs in arms.items()}, audit


def maximum_pair_weight(chain_sources, weighted_edges):
    """Return a post-freeze one-to-one capacity bound for nonnegative integer truth weights."""
    sources, weights = _weighted_graph(chain_sources, weighted_edges)
    if any(not isinstance(weight, Integral) or weight < 0 for weight in weights.values()):
        raise ValueError('capacity bounds need nonnegative integer truth weights')
    positive = {pair: int(weight) for pair, weight in weights.items() if weight}
    total = 0
    for _, left, right, edges in _components(sources, positive):
        max_weight = max(weight for _, _, weight in edges)
        _, chosen, _ = _minimum_cost(left, right,
                                     [(i, j, max_weight - weight) for i, j, weight in edges],
                                     exact_count=0, stop_increment=max_weight)
        total += sum(positive[pair] for pair in chosen)
    return total
