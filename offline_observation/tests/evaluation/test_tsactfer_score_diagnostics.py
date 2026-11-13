import json
import unittest

import numpy as np

from association.evaluation.tsactfer_score_diagnostics import analyze_scores, _ranking, SCOPES


def record(labels, scores, *, sid='s', component='c', n=None):
    """Independent pairs allow an arbitrary class pattern without label cheating."""
    labels, scores = np.asarray(labels, bool), np.asarray(scores, np.float64)
    pairs = np.arange(2*len(labels), dtype=np.int64).reshape(-1, 2)
    n = 2*len(labels) if n is None else n
    identities = np.array([f'id-{i}' for i in range(n)])
    for (a, b), positive in zip(pairs, labels):
        if positive:
            identities[b] = identities[a]
    return dict(scene_id=sid, component_id=component, pairs=pairs, labels=labels,
        scores=scores, node_ids=np.array([f'node-{i}' for i in range(n)]),
        identities=identities, sources=np.arange(n) % 2,
        intervals=np.tile([0., 240.], (n, 1)))


def brute_ranking(scores, labels):
    scores, labels = np.asarray(scores), np.asarray(labels, bool)
    positives, negatives = scores[labels], scores[~labels]
    auc = np.mean([float(p > n)+.5*float(p == n) for p in positives for n in negatives])
    ap, old_tp = 0., 0
    for threshold in sorted(set(scores), reverse=True):
        accepted = scores >= threshold
        tp = np.count_nonzero(labels & accepted)
        ap += (tp-old_tp)/len(positives)*tp/accepted.sum()
        old_tp = tp
    return auc, ap


class TSAScoreDiagnosticTests(unittest.TestCase):
    def test_tied_rankings_match_independent_pair_and_step_integrals(self):
        rng = np.random.default_rng(75)
        for n in (4, 19, 70):
            for _ in range(10):
                scores = rng.integers(-3, 4, n).astype(float)
                labels = rng.integers(0, 2, n).astype(bool)
                labels[0], labels[-1] = True, False
                actual = _ranking(scores, labels)
                expected = brute_ranking(scores, labels)
                np.testing.assert_allclose(actual, expected, atol=1e-14, rtol=0)
        report = analyze_scores([record([1, 0, 1, 0], [0., 0., 0., 0.])], 0)
        metric = report['pooled']['all']
        self.assertEqual(metric['roc_auc'], .5)
        self.assertEqual(metric['average_precision'], .5)
        self.assertEqual(metric['ap_over_prevalence'], 1.)
        self.assertEqual((metric['tp'], metric['fp']), (2, 2))
        self.assertEqual(metric['false_positive_rate'], 1.)
        self.assertEqual(metric['false_discovery_rate'], .5)

    def test_known_ap_is_noninterpolated_and_invariant_to_tied_order(self):
        scores, labels = np.array([.9, .8, .8, .2]), np.array([0, 1, 0, 1], bool)
        expected = (0.125, 5/12)
        np.testing.assert_allclose(_ranking(scores, labels), expected)
        permutation = [3, 2, 1, 0]
        np.testing.assert_allclose(_ranking(scores[permutation], labels[permutation]), expected)

    def test_one_class_empty_all_reject_and_defined_macro_counts(self):
        records = [record([1, 1], [.3, .3], sid='p'),
                   record([0, 0], [.9, .8], sid='n'),
                   record([], [], sid='empty', component='other', n=1)]
        report = analyze_scores(records, None)
        p, n, empty = [r['scopes']['all'] for r in report['scenes']]
        self.assertEqual(p['average_precision'], 1.)
        self.assertIsNone(n['average_precision'])
        self.assertIsNone(p['roc_auc']); self.assertIsNone(n['roc_auc'])
        self.assertEqual(p['fn'], 2); self.assertEqual(n['tn'], 2)
        self.assertIsNone(empty['f1']); self.assertEqual(n['f1'], 1.)
        self.assertIsNone(n['precision']); self.assertIsNone(n['false_discovery_rate'])
        self.assertEqual(n['false_positive_rate'], 0.)
        macro = report['scene_macro']['all']
        self.assertEqual(macro['unit_count'], 3)
        self.assertEqual(macro['nonempty_unit_count'], 2)
        self.assertEqual(macro['average_precision'], {'mean': 1., 'defined_unit_count': 1})
        self.assertEqual(macro['roc_auc'], {'mean': None, 'defined_unit_count': 0})
        self.assertEqual(report['component_macro']['all']['roc_auc']['defined_unit_count'], 1)
        self.assertEqual(report['pooled']['all']['average_precision'], 0.5)
        json.dumps(report, allow_nan=False)
        empty_report = analyze_scores([], None)
        self.assertEqual(empty_report['scene_count'], 0)
        self.assertIsNone(empty_report['pooled']['all']['prevalence'])
        self.assertEqual(empty_report['examples']['false_positive'], [])
        json.dumps(empty_report, allow_nan=False)

    def test_paper_eligibility_boundaries_and_exact_partitions(self):
        row = record([1, 0, 1, 0, 1, 0], [0., .1, .2, .3, .4, .5])
        row['sources'] = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1])
        row['intervals'] = np.array([
            [0, 121], [121, 242],       # ITSA touch: valid, gap 0
            [0, 121], [1321, 1442],     # gap exactly 1200: invalid
            [0, 120], [121, 242],       # duration exactly 120: invalid
            [0, 240], [100, 340],       # same-source overlap: invalid
            [0, 240], [120, 360],       # MSTSA overlap exactly 120: invalid
            [0, 240], [119, 359]], float)  # overlap 121: valid
        result = analyze_scores([row], .25)
        scopes = result['pooled']
        self.assertEqual(scopes['itsa_all']['pair_count'], 4)
        self.assertEqual(scopes['itsa_eligible']['pair_count'], 1)
        self.assertEqual(scopes['itsa_ineligible']['pair_count'], 3)
        self.assertEqual(scopes['mstsa_eligible']['pair_count'], 1)
        self.assertEqual(scopes['mstsa_ineligible']['pair_count'], 1)
        for parent in result['partition_checks'].values():
            self.assertTrue(parent['passed'])
            for counts in parent['counts'].values():
                self.assertEqual(counts['parent'], counts['children_sum'])
        self.assertFalse(result['examples']['false_positive'][1]['paper_eligible'])

    def test_strata_components_quantiles_and_no_mutation(self):
        records = [record([1, 0], [.6, .5], sid='a', n=31),
                   record([1, 0], [.8, .3], sid='b', n=32),
                   record([1, 0], [.1, .4], sid='c', component='z', n=128),
                   record([1, 0], [.9, .8], sid='d', component='z', n=129)]
        copies = [{k: v.copy() if isinstance(v, np.ndarray) else v for k, v in r.items()} for r in records]
        checks = []
        report = analyze_scores(records, .5, check=lambda: checks.append(1))
        self.assertEqual(len(checks), 6)
        strata = report['scene_size_strata']
        self.assertEqual([strata[k]['scene_count'] for k in ('lt32', '32_to128', 'ge129')], [1, 2, 1])
        self.assertEqual(report['component_count'], 2)
        self.assertEqual(report['node_count'], 320)
        for before, after in zip(copies, records):
            for key in before:
                np.testing.assert_array_equal(before[key], after[key])
        self.assertEqual(report['pooled']['all']['scores']['positive']['count'], 4)
        self.assertEqual(report['pooled']['all']['scores']['positive']['p50'], .7)
        self.assertEqual(report['top_fp_scenes'][0]['scene_id'], 'a')
        self.assertEqual(report['top_fp_scenes'][1]['scene_id'], 'd')
        self.assertEqual(report['examples']['false_positive'][0]['scene_id'], 'd')
        json.dumps(report, allow_nan=False)

    def test_scene_and_component_rankings_do_not_become_pooled_pseudoreplication(self):
        # Each scene ranks perfectly, but cross-scene score offsets cause a
        # pooled inversion. The macro records, not edge count, define units.
        rows = [record([1, 0], [.2, .1], sid='a', component='x'),
                record([1, 0], [.9, .8], sid='b', component='y')]
        result = analyze_scores(rows, .5)
        self.assertEqual(result['pooled']['all']['roc_auc'], .75)
        self.assertEqual(result['scene_macro']['all']['roc_auc'],
                         {'mean': 1., 'defined_unit_count': 2})
        self.assertEqual(result['component_macro']['all']['roc_auc'],
                         {'mean': 1., 'defined_unit_count': 2})
        for scope in SCOPES:
            self.assertEqual(sum(row['scopes'][scope]['fp'] for row in result['scenes']),
                             result['pooled'][scope]['fp'])

    def test_bounded_examples_and_invalid_labels_rejected(self):
        row = record([0]*50, [1.]*50)
        report = analyze_scores([row], 1.)
        examples = report['examples']['false_positive']
        self.assertEqual(len(examples), 20)
        self.assertEqual([p['pair_index'] for p in examples], list(range(20)))
        self.assertEqual(report['pooled']['all']['fp'], 50)
        row['labels'][0] = True
        with self.assertRaisesRegex(ValueError, 'labels differ'):
            analyze_scores([row], .5)
        row = record([1, 0], [.3, .4])
        row['scores'][0] = np.nan
        with self.assertRaisesRegex(ValueError, 'finite'):
            analyze_scores([row], .5)
        for threshold in (np.nan, np.inf, True, '0.5'):
            with self.assertRaisesRegex(ValueError, 'threshold'):
                analyze_scores([], threshold)
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            analyze_scores([record([], []), record([], [])], 0.)


if __name__ == '__main__':
    unittest.main()
