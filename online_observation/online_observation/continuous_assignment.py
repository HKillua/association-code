"""Partial bipartite assignment for an online observation update."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
import math
from typing import Union


Cost = Union[int, float]


@dataclass(frozen=True)
class AssignmentResult:
    pairs: tuple[tuple[int, int], ...]
    unmatched_rows: frozenset[int]
    unmatched_columns: frozenset[int]
    ambiguous_rows: frozenset[int]
    ambiguous_columns: frozenset[int]


def _ratio(cost: Cost) -> tuple[int, int]:
    if not isinstance(cost, (int, float)) or (isinstance(cost, float) and not math.isfinite(cost)):
        raise ValueError("costs must be finite integers or floats")
    numerator, denominator = cost.as_integer_ratio()
    return numerator, denominator.bit_length() - 1


def _adjusted_costs(
    costs: Sequence[Sequence[Cost | None]],
    row_unmatched: Sequence[Cost],
    column_unmatched: Sequence[Cost],
) -> list[list[int | None]]:
    if len(costs) != len(row_unmatched) or any(len(row) != len(column_unmatched) for row in costs):
        raise ValueError("cost matrix shape must agree with unmatched costs")

    row_ratios = [_ratio(cost) for cost in row_unmatched]
    column_ratios = [_ratio(cost) for cost in column_unmatched]
    edge_ratios = [[None if cost is None or cost == math.inf else _ratio(cost) for cost in row]
                   for row in costs]

    # All finite binary floats have a power-of-two denominator. A shared integer
    # scale preserves objective ties and tiny differences through all sums.
    shift = max((shift for _, shift in (*row_ratios, *column_ratios)), default=0)
    for row in edge_ratios:
        shift = max(shift, max((ratio[1] for ratio in row if ratio is not None), default=0))

    row_penalties = [numerator << (shift - power) for numerator, power in row_ratios]
    column_penalties = [numerator << (shift - power) for numerator, power in column_ratios]
    adjusted = []
    for row, row_penalty in zip(edge_ratios, row_penalties):
        adjusted_row = []
        for ratio, column_penalty in zip(row, column_penalties):
            if ratio is None:
                adjusted_row.append(None)
                continue
            numerator, power = ratio
            delta = (numerator << (shift - power)) - row_penalty - column_penalty
            adjusted_row.append(delta if delta <= 0 else None)
        adjusted.append(adjusted_row)
    return adjusted


def _result(
    pairs: Iterable[tuple[int, int]],
    rows: int,
    columns: int,
    ambiguous_rows: Iterable[int] = (),
    ambiguous_columns: Iterable[int] = (),
) -> AssignmentResult:
    pairs = tuple(sorted(pairs))
    return AssignmentResult(
        pairs,
        frozenset(range(rows)) - {row for row, _ in pairs},
        frozenset(range(columns)) - {column for _, column in pairs},
        frozenset(ambiguous_rows),
        frozenset(ambiguous_columns),
    )


def _hungarian(costs: list[list[int | None]]) -> tuple[list[int], list[int], list[int]]:
    """Assign each row to a distinct column, returning primal and dual solutions."""
    rows, columns = len(costs), len(costs[0])
    row_potential = [0] * (rows + 1)
    column_potential = [0] * (columns + 1)
    owners = [0] * (columns + 1)
    parent = [0] * (columns + 1)

    # Index zero acts as a temporary column holding the next unassigned row.
    for row in range(1, rows + 1):
        owners[0] = row
        column = 0
        slack = [None] * (columns + 1)
        used = [False] * (columns + 1)
        while True:
            used[column] = True
            owner = owners[column]
            step = None
            next_column = 0
            for candidate, cost in enumerate(costs[owner - 1], start=1):
                if used[candidate]:
                    continue
                if cost is not None:
                    residual = cost - row_potential[owner] - column_potential[candidate]
                    if slack[candidate] is None or residual < slack[candidate]:
                        slack[candidate] = residual
                        parent[candidate] = column
                if slack[candidate] is not None and (step is None or slack[candidate] < step):
                    step = slack[candidate]
                    next_column = candidate

            # The private null column guarantees that an augmenting path exists.
            assert step is not None
            for candidate in range(columns + 1):
                if used[candidate]:
                    row_potential[owners[candidate]] += step
                    column_potential[candidate] -= step
                elif slack[candidate] is not None:
                    slack[candidate] -= step
            column = next_column
            if owners[column] == 0:
                break

        while column != 0:
            predecessor = parent[column]
            owners[column] = owners[predecessor]
            column = predecessor

    selected = [-1] * rows
    for column, owner in enumerate(owners[1:]):
        if owner != 0:
            selected[owner - 1] = column
    return selected, row_potential[1:], column_potential[1:]


def _reachable(graph: list[list[int]], seeds: Iterable[int]) -> set[int]:
    seen = set(seeds)
    stack = list(seen)
    while stack:
        for neighbor in graph[stack.pop()]:
            if neighbor not in seen:
                seen.add(neighbor)
                stack.append(neighbor)
    return seen


def _components(graph: list[list[int]], reverse: list[list[int]]) -> list[int]:
    visited = set()
    finished = []
    for vertex in range(len(graph)):
        if vertex in visited:
            continue
        visited.add(vertex)
        stack = [(vertex, iter(graph[vertex]))]
        while stack:
            current, outgoing = stack[-1]
            neighbor = next(outgoing, None)
            if neighbor is None:
                finished.append(current)
                stack.pop()
            elif neighbor not in visited:
                visited.add(neighbor)
                stack.append((neighbor, iter(graph[neighbor])))

    components = [-1] * len(graph)
    for vertex in reversed(finished):
        if components[vertex] != -1:
            continue
        components[vertex] = vertex
        pending = [vertex]
        while pending:
            for neighbor in reverse[pending.pop()]:
                if components[neighbor] == -1:
                    components[neighbor] = vertex
                    pending.append(neighbor)
    return components


def _ambiguous(
    costs: list[list[int | None]],
    selected: list[int],
    row_potential: list[int],
    column_potential: list[int],
    real_columns: int,
) -> tuple[set[int], set[int]]:
    graph: list[list[int]] = [[] for _ in column_potential]
    reverse: list[list[int]] = [[] for _ in column_potential]
    for row, old_column in enumerate(selected):
        for column, cost in enumerate(costs[row]):
            if cost is not None and column != old_column and cost == row_potential[row] + column_potential[column]:
                graph[old_column].append(column)
                reverse[column].append(old_column)

    if not any(graph):
        return set(), set()

    # An alternative optimum changes edges along a tight cycle or a tight path
    # from a zero-dual occupied column to a free column. A path would otherwise
    # abandon a column with negative dual potential and increase the objective.
    components = _components(graph, reverse)
    from_zero = _reachable(graph, (column for column in selected if column_potential[column] == 0))
    to_free = _reachable(reverse, set(range(len(graph))) - set(selected))
    ambiguous_rows = set()
    ambiguous_columns = set()
    for row, old_column in enumerate(selected):
        for column in graph[old_column]:
            if components[old_column] == components[column] or (old_column in from_zero and column in to_free):
                ambiguous_rows.add(row)
                if old_column < real_columns:
                    ambiguous_columns.add(old_column)
                if column < real_columns:
                    ambiguous_columns.add(column)
    return ambiguous_rows, ambiguous_columns


def _candidate_components(
    costs: list[list[int | None]], columns: int,
) -> Iterable[tuple[list[int], list[int]]]:
    column_neighbors: list[list[int]] = [[] for _ in range(columns)]
    row_neighbors: list[list[int]] = [[] for _ in costs]
    for row, edge_costs in enumerate(costs):
        for column, cost in enumerate(edge_costs):
            if cost is not None:
                column_neighbors[column].append(row)
                row_neighbors[row].append(column)

    visited_rows = set()
    for start, neighbors in enumerate(row_neighbors):
        if start in visited_rows or not neighbors:
            continue
        component_rows = []
        component_columns = []
        seen_columns = set()
        pending_rows = [start]
        visited_rows.add(start)
        while pending_rows:
            row = pending_rows.pop()
            component_rows.append(row)
            for column in row_neighbors[row]:
                if column in seen_columns:
                    continue
                seen_columns.add(column)
                component_columns.append(column)
                for neighbor in column_neighbors[column]:
                    if neighbor not in visited_rows:
                        visited_rows.add(neighbor)
                        pending_rows.append(neighbor)
        yield component_rows, component_columns


def _global_component(
    adjusted: list[list[int | None]],
) -> tuple[list[tuple[int, int]], set[int], set[int]]:
    rows, columns = len(adjusted), len(adjusted[0])
    if rows == columns == 1:
        if adjusted[0][0] == 0:
            return [], {0}, {0}
        return [(0, 0)], set(), set()

    transpose = rows > columns
    if transpose:
        adjusted = [list(column) for column in zip(*adjusted)]
    left, right = min(rows, columns), max(rows, columns)
    for row, edge_costs in enumerate(adjusted):
        null_columns = [None] * left
        null_columns[row] = 0
        edge_costs.extend(null_columns)

    selected, row_potential, column_potential = _hungarian(adjusted)
    ambiguous_left, ambiguous_right = _ambiguous(adjusted, selected, row_potential, column_potential, right)
    pairs = [(row, column) for row, column in enumerate(selected) if row not in ambiguous_left and column < right]
    if transpose:
        return [(column, row) for row, column in pairs], ambiguous_right, ambiguous_left
    return pairs, ambiguous_left, ambiguous_right


def solve_global(
    costs: Sequence[Sequence[Cost | None]],
    row_unmatched: Sequence[Cost],
    column_unmatched: Sequence[Cost],
) -> AssignmentResult:
    """Minimize paired costs plus each unmatched vertex's cost.

    ``None`` and positive infinity forbid an edge. Return only pairs shared by
    every optimal assignment. A vertex is ambiguous if its optimal partner can
    vary, including becoming unmatched. Unmatched sets include abstentions.
    """
    adjusted = _adjusted_costs(costs, row_unmatched, column_unmatched)
    rows, columns = len(row_unmatched), len(column_unmatched)
    pairs = []
    ambiguous_rows, ambiguous_columns = set(), set()
    for row_ids, column_ids in _candidate_components(adjusted, columns):
        component = [[adjusted[row][column] for column in column_ids] for row in row_ids]
        component_pairs, component_rows, component_columns = _global_component(component)
        pairs.extend((row_ids[row], column_ids[column]) for row, column in component_pairs)
        ambiguous_rows.update(row_ids[row] for row in component_rows)
        ambiguous_columns.update(column_ids[column] for column in component_columns)
    return _result(pairs, rows, columns, ambiguous_rows, ambiguous_columns)


def _update_best(best: int | None, count: int, candidate: int) -> tuple[int, int]:
    if best is None or candidate < best:
        return candidate, 1
    return best, count + (candidate == best)


def solve_greedy(
    costs: Sequence[Sequence[Cost | None]],
    row_unmatched: Sequence[Cost],
    column_unmatched: Sequence[Cost],
) -> AssignmentResult:
    """Accept mutual unique positive savings in one pass, without backfill.

    Report ties with another locally best partner or with staying unmatched.
    If such a tie blocks a mutually best edge, report both endpoints.
    """
    adjusted = _adjusted_costs(costs, row_unmatched, column_unmatched)
    rows, columns = len(row_unmatched), len(column_unmatched)
    row_best: list[int | None] = [None] * rows
    column_best: list[int | None] = [None] * columns
    row_counts, column_counts = [0] * rows, [0] * columns
    for row, edge_costs in enumerate(adjusted):
        for column, cost in enumerate(edge_costs):
            if cost is not None:
                row_best[row], row_counts[row] = _update_best(row_best[row], row_counts[row], cost)
                column_best[column], column_counts[column] = _update_best(column_best[column], column_counts[column], cost)

    ambiguous_rows = {row for row in range(rows) if row_counts[row] > 1 or row_best[row] == 0}
    ambiguous_columns = {column for column in range(columns) if column_counts[column] > 1 or column_best[column] == 0}
    pairs = []
    for row, edge_costs in enumerate(adjusted):
        for column, cost in enumerate(edge_costs):
            if cost is None or cost != row_best[row] or cost != column_best[column]:
                continue
            if cost == 0 or row_counts[row] > 1 or column_counts[column] > 1:
                ambiguous_rows.add(row)
                ambiguous_columns.add(column)
            else:
                pairs.append((row, column))
    return _result(pairs, rows, columns, ambiguous_rows, ambiguous_columns)
