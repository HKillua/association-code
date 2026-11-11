from itertools import product
import math
import unittest

from online_observation.continuous_assignment import AssignmentResult, solve_global, solve_greedy


class ContinuousAssignmentTests(unittest.TestCase):
    def test_global_resolves_a_collision_that_local_control_leaves_unmatched(self):
        costs = [[1, 2], [2, None]]
        self.assertEqual(solve_global(costs, [5, 5], [5, 5]).pairs, ((0, 1), (1, 0)))
        result = solve_greedy(costs, [5, 5], [5, 5])
        self.assertEqual(result, AssignmentResult(((0, 0),), {1}, {1}, set(), set()))

    def test_null_uses_both_endpoint_costs_and_flags_a_zero_saving(self):
        for solve in (solve_global, solve_greedy):
            self.assertEqual(solve([[5]], [2], [4]).pairs, ((0, 0),))
            self.assertEqual(solve([[5]], [2], [3]), AssignmentResult((), {0}, {0}, {0}, {0}))
            self.assertEqual(solve([[5]], [2], [2]), AssignmentResult((), {0}, {0}, set(), set()))

    def test_equal_optima_abstain_only_on_changed_pairs(self):
        costs = [[1, 1, None, None], [None, 1, 1, None], [None, None, None, 1]]
        expected = AssignmentResult(((2, 3),), {0, 1}, {0, 1, 2}, {0, 1}, {0, 1, 2})
        for solve in (solve_global, solve_greedy):
            self.assertEqual(solve(costs, [5] * 3, [5] * 4), expected)

        # The common third edge belongs to the same connected candidate component.
        cycle = [[1, 2, None], [3, 4, 8], [None, None, 0]]
        self.assertEqual(solve_global(cycle, [5] * 3, [5] * 3),
                         AssignmentResult(((2, 2),), {0, 1}, {0, 1}, {0, 1}, {0, 1}))

    def test_a_tight_edge_that_cannot_complete_an_optimal_change_does_not_cause_abstention(self):
        self.assertEqual(solve_global([[1, 1], [1, 2]], [5, 5], [5, 5]),
                         AssignmentResult(((0, 1), (1, 0)), set(), set(), set(), set()))

    def test_binary_float_objective_keeps_sub_ulp_difference_after_adding_edges(self):
        # The rounded sum of 1 and nextafter(1, +inf) equals 2 in Python floats.
        higher = math.nextafter(1, math.inf)
        self.assertEqual(solve_global([[1, 1], [1, higher]], [5, 5], [5, 5]).pairs,
                         ((0, 1), (1, 0)))
        self.assertEqual(solve_greedy([[1, higher]], [5], [5, 5]).pairs, ((0, 0),))

    def test_missing_edges_empty_sides_and_shape_errors(self):
        for solve in (solve_global, solve_greedy):
            self.assertEqual(solve([[None, math.inf]], [3], [1, 2]),
                             AssignmentResult((), {0}, {0, 1}, set(), set()))
            self.assertEqual(solve([], [], [1, 2]).unmatched_columns, {0, 1})
            self.assertEqual(solve([[], []], [1, 2], []).unmatched_rows, {0, 1})
            with self.assertRaisesRegex(ValueError, "shape"):
                solve([[1], []], [3, 3], [1])
            with self.assertRaisesRegex(ValueError, "finite"):
                solve([[math.nan]], [3], [1])
            with self.assertRaisesRegex(ValueError, "finite"):
                solve([[1]], [math.inf], [1])

    def test_both_solvers_are_invariant_to_transposing_and_permuting(self):
        costs = [[4, 2, 1, None], [4, None, None, 2], [None, None, None, 2]]
        row_costs, column_costs = [5, 5, 4], [5, 3, 2, 5]
        rows, columns = [2, 0, 1], [1, 3, 0, 2]
        for solve in (solve_global, solve_greedy):
            expected = solve(costs, row_costs, column_costs)
            actual = solve([[costs[row][column] for row in rows] for column in columns],
                           [column_costs[column] for column in columns], [row_costs[row] for row in rows])
            self.assertEqual(set(expected.pairs), {(rows[row], columns[column]) for column, row in actual.pairs})
            self.assertEqual(expected.ambiguous_rows, {rows[row] for row in actual.ambiguous_columns})
            self.assertEqual(expected.ambiguous_columns, {columns[column] for column in actual.ambiguous_rows})

    def test_global_matches_exhaustive_optimal_partners_on_small_rectangular_graphs(self):
        row_costs, column_costs = (1, 2), (2, 1, 1)
        choices = [(left, right) for left, right in product((None, 0, 1, 2), repeat=2)
                   if left is None or right is None or left != right]
        for values in product((None, 1, 2, 3), repeat=6):
            costs = [values[:3], values[3:]]
            best = []
            best_cost = sum(row_costs) + sum(column_costs)
            for assignment in choices:
                score = sum(column_costs[column] for column in range(3) if column not in assignment)
                for row, column in enumerate(assignment):
                    if column is not None and costs[row][column] is None:
                        break
                    score += row_costs[row] if column is None else costs[row][column]
                else:
                    if score <= best_cost:
                        if score < best_cost:
                            best = []
                        best_cost = score
                        best.append(assignment)

            actual = solve_global(costs, row_costs, column_costs)
            partners = [set(values) for values in zip(*best)]
            self.assertEqual(actual.pairs, tuple((row, next(iter(options))) for row, options in enumerate(partners)
                                                 if len(options) == 1 and None not in options), values)
            self.assertEqual(actual.ambiguous_rows, {row for row, options in enumerate(partners) if len(options) > 1},
                             values)
            self.assertEqual(actual.ambiguous_columns,
                             {column for column in range(3) if len({assignment.index(column) if column in assignment
                              else None for assignment in best}) > 1}, values)


if __name__ == "__main__":
    unittest.main()
