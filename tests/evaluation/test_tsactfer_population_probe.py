"""080 immutable-score comparison and noncausal population accounting."""
from copy import deepcopy
import unittest
from unittest.mock import patch

import numpy as np
import torch

from association.evaluation.tsactfer_input_probe import DEFAULT as OLD_DEFAULT, probe
from association.evaluation.tsactfer_population_probe import (
    DEFAULT, standardized_ap, pair_score, select_validation_roster,
    validate_079, analyze_population, _margin_statistics)
from baselines.tsactfer import TSActfer, TSAConfig
from tests.test_tsactfer_level1 import tiny_config, training_fixture, native_scene


def fixture():
    tsa = tiny_config()
    scenes, _ = training_fixture(tsa)
    model = TSActfer(TSAConfig(**tsa['model']))
    source, saved = probe(model, scenes, tsa, dict(OLD_DEFAULT, scene_count=2, pairs_per_head_scene=3))
    source.update(status='COMPLETED', scene_ids=['t1', 't2'])
    records = []
    for sid in ('v1', 'v2'):
        row = native_scene(sid, 'V_select')
        row.pop('tracks')
        row['scores'] = np.where(row['labels'], .85, .2).astype(np.float32)
        records.append(row)
    manifest = dict(splits={'T_core': ['t1', 't2'], 'V_select': ['v1', 'v2'], 'T_test': ['sealed']},
        components={sid: 'component-'+sid for sid in ('t1', 't2', 'v1', 'v2', 'sealed')})
    config = dict(DEFAULT, scene_count=2, pairs_per_head_scene=3)
    return records, manifest, source, saved, config


class TSAPopulationProbeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_standardized_ap_ties_prevalence_and_undefined_classes(self):
        self.assertEqual(standardized_ap([1, 1, 1, 1], [1, 0, 0, 0]), .5)
        self.assertEqual(standardized_ap([2, 1, 0], [1, 0, 0]), 1.)
        self.assertAlmostEqual(standardized_ap([0, 1], [1, 0]), .5)
        self.assertIsNone(standardized_ap([], []))
        self.assertIsNone(standardized_ap([1, 2], [1, 1]))
        self.assertIsNone(standardized_ap([1, 2], [0, 0]))
        # Duplicating every negative changes observed AP but not the declared
        # class-conditional reference, including exact score ties.
        a = standardized_ap([.8, .3, .7, .3], [1, 1, 0, 0])
        b = standardized_ap([.8, .3, .7, .3, .7, .3], [1, 1, 0, 0, 0, 0])
        self.assertAlmostEqual(a, b)
        with self.assertRaises(ValueError):
            standardized_ap([np.nan], [1])
        with self.assertRaises(ValueError):
            standardized_ap([.2], [2])
        with self.assertRaises(ValueError):
            standardized_ap([.2], [1], 0.)

    def test_deterministic_full_population_balanced_cache_analysis(self):
        records, manifest, source, saved, config = fixture()
        original = deepcopy(records)
        with (patch('torch.load', side_effect=AssertionError('no checkpoint')),
              patch('torch.optim.Adam', side_effect=AssertionError('no optimizer'))):
            result, selected = analyze_population(records, manifest, source, saved, .6, config)
        again, selections = analyze_population(records, manifest, source, saved, .6, config)
        self.assertEqual(result, again)
        self.assertEqual(selected, selections)
        self.assertEqual(result['validation_scene_ids'], ['v1', 'v2'])
        self.assertEqual(result['model_forwards'], 0)
        self.assertFalse(result['test_opened'])
        self.assertFalse(result['threshold_reselected'])
        self.assertEqual(result['validation_full_candidates']['node_count'], 18)
        self.assertEqual(result['validation_full_candidates']['pooled']['all']['pair_count'], 72)
        for task in ('ITSA', 'MSTSA'):
            metrics = result['validation_balanced_full'][task]['pooled']
            self.assertEqual(metrics['positive_count'], 6)
            self.assertEqual(metrics['negative_count'], 6)
            self.assertEqual(metrics['prevalence'], .5)
            self.assertEqual(metrics['tp'], 6)
            self.assertEqual(metrics['fp'], 0)
            self.assertEqual(metrics['average_precision_prior05'], metrics['average_precision'])
            for pair in selected[task]['mapping']:
                views = selected[task]['rows']
                a, n = [views[pair[key]] for key in ('anchor_view_index', 'negative_view_index')]
                self.assertNotEqual(a['identity'], n['identity'])
                self.assertEqual(a['scene_id'], n['scene_id'])
                self.assertEqual(a['left_source'] == n['right_source'], task == 'ITSA')
                self.assertTrue(pair['cross_scene_candidates_exist'])
            self.assertTrue(result['source079_verification'][task]['all_saved_metrics_reproduced'])
        for row, old in zip(records, original):
            for key in ('scores', 'pairs', 'labels', 'intervals'):
                np.testing.assert_array_equal(row[key], old[key])

    def test_lookup_symmetry_missing_pairs_and_sealed_roster(self):
        records, manifest, _, _, config = fixture()
        self.assertEqual(pair_score(records[0], 0, 8), pair_score(records[0], 8, 0))
        bad = deepcopy(records[0]); bad['pairs'] = bad['pairs'][:-1]
        with self.assertRaisesRegex(ValueError, 'complete'):
            pair_score(bad, 0, 8)
        with self.assertRaises(ValueError):
            pair_score(records[0], 1, 1)
        self.assertEqual(select_validation_roster(manifest, config), ['v1', 'v2'])
        bad = deepcopy(manifest); bad['splits']['T_test'].append('v1')
        with self.assertRaisesRegex(ValueError, 'disjoint'):
            select_validation_roster(bad, config)
        bad = deepcopy(manifest); bad['splits']['V_select'] = ['v1']
        with self.assertRaisesRegex(ValueError, 'enoughV'):
            select_validation_roster(bad, config)

    def test_insufficient_eligibility_and_cross_availability_are_not_replaced(self):
        records, manifest, source, saved, config = fixture()
        # No positive identities remain in v2. It remains in the fixed roster.
        records[1]['identities'] = np.array([f'unique-{i}' for i in range(9)])
        records[1]['labels'] = np.zeros(36, np.float32)
        result, _ = analyze_population(records, manifest, source, saved, .6, config)
        self.assertEqual(result['validation_scene_ids'], ['v1', 'v2'])
        for task in ('ITSA', 'MSTSA'):
            report = result['validation_balanced_full'][task]
            self.assertEqual(report['paired_anchors'], 0)
            self.assertEqual(report['cross_scene_availability_exclusions'], 3)
            self.assertIsNone(report['pooled']['average_precision'])
            self.assertEqual(report['sampling'][1]['excluded_reason'], 'fewer_than_two_eligible_identities')
            self.assertEqual(report['scenes']['v2']['metrics']['pair_count'], 0)
        # Missing same-scene matching source, not cross-scene global identity,
        # can exclude an anchor despite multiple available positive identities.
        records, manifest, source, saved, config = fixture()
        records[0]['sources'] = np.array(['s0', 's0', 'r0', 's1', 's1', 'r1', 's2', 's2', 'r2'])
        result, _ = analyze_population(records, manifest, source, saved, .6, config)
        self.assertEqual(result['validation_balanced_full']['ITSA']['scenes']['v1']['metrics']['pair_count'], 0)
        self.assertTrue(any('same_scene' in r for x in result['validation_balanced_full']['ITSA']['excluded_anchors']
                            for r in x['reason']))

    def test_079_tampering_and_validation_inventory_rejected(self):
        records, manifest, source, saved, config = fixture()
        validate_079(source, saved, manifest)
        bad = deepcopy(saved)
        bad['ITSA']['arrays']['mapping'][0, 1] = bad['ITSA']['arrays']['mapping'][0, 0]
        with self.assertRaisesRegex(ValueError, 'negative'):
            validate_079(source, bad, manifest)
        bad = deepcopy(source)
        bad['tasks']['ITSA']['conditions']['same_scene__full']['pooled']['roc_auc'] = .123456
        with self.assertRaisesRegex(ValueError, 'numeric'):
            validate_079(bad, saved, manifest)
        bad = deepcopy(saved)
        bad['MSTSA']['arrays']['same_scene__full_negative'][0] = np.nan
        with self.assertRaisesRegex(ValueError, 'finite'):
            validate_079(source, bad, manifest)
        bad = deepcopy(records); bad.reverse()
        with self.assertRaisesRegex(ValueError, 'V_select order'):
            analyze_population(bad, manifest, source, saved, .6, config)
        bad = deepcopy(records); bad[0]['pairs'] = bad[0]['pairs'][:-1]
        bad[0]['scores'] = bad[0]['scores'][:-1]; bad[0]['labels'] = bad[0]['labels'][:-1]
        with self.assertRaisesRegex(ValueError, 'complete original'):
            analyze_population(bad, manifest, source, saved, .6, config)

    def test_margin_means_are_additive_but_medians_are_not(self):
        p = np.array([.1, .1, .1], np.float32)
        n = np.array([.1, .1, .1], np.float32)
        cp = np.array([.1, .1, .9], np.float32)
        cn = np.array([0., .9, .1], np.float32)
        result = _margin_statistics(p, n, cp, cn)
        self.assertAlmostEqual(result['mean_positive_change']-result['mean_negative_change'],
                               result['mean_margin_change'])
        self.assertEqual(result['improvement_count'], 2)
        self.assertEqual(result['deterioration_count'], 1)
        self.assertEqual(result['tie_count'], 0)
        delta_p, delta_n = cp.astype(float)-p, cn.astype(float)-n
        self.assertNotEqual(float(np.median(delta_p)-np.median(delta_n)), float(np.median(delta_p-delta_n)))
        self.assertEqual(_margin_statistics([], [], [], [])['count'], 0)
        self.assertEqual(_margin_statistics([.1], [.1], [.1+1e-9], [.1])['tie_count'], 1)


if __name__ == '__main__':
    unittest.main()
