"""Software properties only; synthetic data are not MTAD effectiveness evidence."""
from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from tools.analysis.audit_predicted_registration import (
    analyze_pair, analyze_scene, component_split, fit_center, input_projection, main,
)


def row(group, delta, query=0, nodes=None, same=True, support=True):
    a, b = nodes or (2*group, 2*group+1)
    return dict(fixed_group_index=group, node_a=a, node_b=b, source_a='1', source_b='2',
        query_index=query, query_seconds=float(query), both_supported=support, delta_NE_m=delta,
        direction='source_b minus source_a', oracle_same_identity=same, target_a=group, target_b=group)


def public(rows):
    return dict(query_pairs=rows, query_sources=[dict(group_index=g, selected_unique_group=True)
                                                for g in {r['fixed_group_index'] for r in rows
                                                          if r['fixed_group_index'] is not None}])


class PredictedRegistrationTests(unittest.TestCase):
    def test_oracle_fields_and_truth_do_not_change_primary_results(self):
        rows = [row(g, [10.+g, -20.], q) for g in range(6) for q in (0, 40)]
        extra = row(30, [1e6, 1e6]); extra['fixed_group_index'] = None
        original = public(rows+[extra])
        changed = deepcopy(original)
        for r in changed['query_pairs']:
            r['oracle_same_identity'] = False
            r['target_a'], r['target_b'] = 'changed', 'different'
            r['error_m'] = 1e20
        changed['points'] = [{'truth_valid': False, 'error_m': 1e25}]
        a, b = analyze_scene(original), analyze_scene(changed)
        for result in (a, b):
            for pair in result['source_pairs']:
                pair.pop('oracle_postfit_sensitivity')
        self.assertEqual(a, b)
        projected = input_projection(changed)
        self.assertTrue(all('oracle_same_identity' not in r and 'target_a' not in r for r in projected))

    def test_group_weight_does_not_increase_when_long_track_repeats(self):
        rows = [row(1, [0., 0.], q) for q in range(60)]+[row(2, [10., 20.]), row(3, [1000., 1000.])]
        self.assertEqual(fit_center(rows), [10., 20.])
        self.assertEqual(fit_center(rows, 'mean'), [1010./3., 1020./3.])

    def test_shared_node_groups_are_not_split_and_support_does_not_change_split(self):
        rows = [row(0, [1., 2.], nodes=(0, 1)), row(1, [1., 2.], nodes=(1, 2)),
                row(2, [1., 2.], nodes=(4, 5), support=False)]
        split = component_split(input_projection(public(rows)))
        self.assertEqual(split['components'], [[0, 1], [2]])
        self.assertEqual(split['heldout_groups'], [0, 1])
        self.assertEqual(split['fit_groups'], [2])
        result = analyze_pair(input_projection(public(rows)))
        self.assertIsNone(result['center_NE_m'])
        self.assertEqual(result['main']['heldout_all']['status'], 'unavailable_no_fit_groups')

    def test_time_holdout_never_falls_back_to_late_data(self):
        result = analyze_pair(input_projection(public([row(g, [10., 20.], 40) for g in range(3)])))
        self.assertEqual(result['center_NE_m'], [10., 20.])
        self.assertIsNone(result['early_center_NE_m'])
        self.assertEqual(result['temporal']['heldout_late']['status'], 'unavailable_no_fit_groups')

    def test_different_identity_pairs_remain_in_primary_population(self):
        data = public([row(g, [1000. if g == 1 else 10., 20.], same=g != 1) for g in range(3)])
        result = analyze_scene(data)['source_pairs'][0]
        self.assertEqual(result['population']['supported_pair_query_count'], 3)
        self.assertEqual(result['center_NE_m'], [505., 20.])
        oracle = result['oracle_postfit_sensitivity']
        self.assertEqual(oracle['same_fit_groups_oracle_center_NE_m'], [10., 20.])
        self.assertEqual(oracle['strata']['fit']['different_identity']['pair_query_count'], 1)

    def test_duplicate_pair_query_and_reversed_orientation_stop(self):
        r = row(0, [10., 20.])
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            input_projection(public([r, r]))
        r['direction'] = 'source_a minus source_b'
        with self.assertRaisesRegex(ValueError, 'orientation'):
            input_projection(public([r]))

    def test_failed_input_is_retained_and_existing_output_is_not_overwritten(self):
        with TemporaryDirectory() as temp:
            output = Path(temp)/'audit.json'
            argv = ['audit', '--source', str(Path(temp)/'absent'), '--output', str(output)]
            with patch('sys.argv', argv), self.assertRaises(FileNotFoundError):
                main()
            result = json.loads(output.read_text())
            self.assertEqual(result['status'], 'FAILED')
            self.assertEqual(result['error_type'], 'FileNotFoundError')
            self.assertEqual(result['scenes'], {})
            original = output.read_bytes()
            with patch('sys.argv', argv), self.assertRaises(FileExistsError):
                main()
            self.assertEqual(output.read_bytes(), original)


if __name__ == '__main__':
    unittest.main()
