"""Independent126 arithmetic/eligibility/plan regressions, synthetic only."""
from copy import deepcopy
from pathlib import Path
import json
import tempfile
import unittest

import numpy as np

from tools.analysis import audit_ordered_shape_arrays as audit
from tools.analysis import ordered_shape_signal as producer


def records():
    pairs = np.column_stack(np.triu_indices(4, 1))
    items = []
    for i in range(3):
        labels = np.array([True, False, False, False, False, True]) if i == 0 else np.zeros(6, bool)
        sources = np.array(['a', 'a', 'b', 'b'])
        items.append(dict(scene_id=str(i), component_id='c'+str(i), pairs=pairs,
            labels=labels, sources=sources, node_ids=np.array(['0', '1', '2', '3']),
            identities=np.array(['A', 'A', 'B', 'B']),
            intervals=np.array([[0., 200.], [200., 400.], [0., 300.], [120., 500.]]),
            joint_support_queries=np.array([0, 3, 0, 20, 1, 64], np.uint8),
            scores=np.array([1., .5, .5, -.5, -.7, 0.])+i))
    items.append(dict(scene_id='empty', component_id='c3', pairs=np.empty((0, 2), np.int64), labels=np.empty(0, bool),
        sources=np.array([], str), node_ids=np.array([], str), identities=np.array([], str),
        intervals=np.empty((0, 2)), joint_support_queries=np.empty(0, np.uint8), scores=np.empty(0)))
    return items


class OrderedShapeAuditTests(unittest.TestCase):
    def test_132_requires_exact_zero_update_source_and_immutable_cache_norms(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); old, new = root/'126', root/'132'
            (old/'private/cache').mkdir(parents=True); (new/'private').mkdir(parents=True)
            (new/'private/cache').symlink_to(old/'private/cache', target_is_directory=True)
            manifest = root/'manifest.json'; manifest.write_text('{"same":true}')
            source = dict(idea_id=audit.IDEA, plan_version='1.1', status='FAILED', exit_code=1,
                completed_optimizer_updates=0, completed_arms=[], optimizer_updates_by_arm={a:0 for a in audit.ARMS},
                error='OSError: Too many open files', elapsed_seconds=10., manifest=str(manifest), source_system=str(root/'056'))
            (old/'run.json').write_text(json.dumps(source))
            for path in (old,new): (path/'normalizers.json').write_text('{"original":1}')
            run = dict(idea_id=audit.CONTINUATION_IDEA, plan_version='1.0', scientific_idea_id=audit.IDEA,
                scientific_plan_version='1.1', failed_run=str(old), manifest=str(manifest), source_system=str(root/'056'),
                original126_restarted=False, raw_cache_rebuilt=False, normalizers_refitted=False,
                cache_read_only_reference=True, archive_read=False, prior_failed_elapsed_seconds=10.)
            summary = dict(run)
            audit.inspect_run_identity(new, run, summary, audit.ScoreChecks())
            for field in ('raw_cache_rebuilt','normalizers_refitted'):
                bad = dict(run); bad[field] = True
                with self.subTest(field=field), self.assertRaises(ValueError):
                    audit.inspect_run_identity(new, bad, summary, audit.ScoreChecks())
            (new/'normalizers.json').write_text('{"original":2}')
            with self.assertRaisesRegex(ValueError, 'inherited normalizers'):
                audit.inspect_run_identity(new, run, summary, audit.ScoreChecks())
            (new/'normalizers.json').write_text('{"original":1}')
            source['completed_optimizer_updates'] = 1; (old/'run.json').write_text(json.dumps(source))
            with self.assertRaisesRegex(ValueError, 'zero-update FD'):
                audit.inspect_run_identity(new, run, summary, audit.ScoreChecks())

    def test_full_raw_paper_support_and_null_arithmetic_against_saved_schema(self):
        rows = records(); scores = [r['scores'] for r in rows]
        for theta in (0., None, 100.):
            expected = producer.describe(rows, theta)
            counts = audit.count_all(rows, scores, theta)
            checks = audit.ScoreChecks()
            audit.inspect_description(expected, list(range(len(rows))), rows, scores, counts, theta, checks, 'fixture')
            self.assertGreater(checks.numeric_checks, 0)
            self.assertGreater(checks.null_checks, 0)
        broken = deepcopy(producer.describe(rows, 0.))
        broken['raw']['micro_precision'] += 1e-7
        with self.assertRaisesRegex(ValueError, 'micro_precision'):
            audit.inspect_description(broken, list(range(len(rows))), rows, scores,
                audit.count_all(rows, scores, 0.), 0., audit.ScoreChecks(), 'bad fraction')

    def test_paper_boundary_masks_are_distinct_from_same_cross_and_joint(self):
        r = dict(pairs=np.array([[0, 1], [0, 2], [0, 3], [0, 4], [0, 5], [0, 6]]),
            sources=np.array(['a', 'a', 'a', 'a', 'b', 'b', 'a']),
            intervals=np.array([[0., 200.], [200., 400.], [1400., 1600.], [1399., 1600.],
                                [80., 400.], [79., 400.], [200., 320.]]),
            joint_support_queries=np.array([0, 1, 64, 0, 2, 0], np.uint8))
        masks, same = audit.eligibility(r)
        np.testing.assert_array_equal(masks['paper_itsa'], [True, False, True, False, False, False])
        np.testing.assert_array_equal(masks['paper_mstsa'], [False, False, False, False, True, False])
        self.assertEqual(int(same.sum()), 4)
        self.assertEqual(int(masks['joint_support'].sum()), 3)

    def test_plan_exposure_balancing_original_indices_and_no_new_sampling(self):
        with tempfile.TemporaryDirectory() as temporary:
            cache = Path(temporary)
            np.save(cache/'a-labels.npy', np.array([True, False, True, False]))
            np.save(cache/'a-pairs.npy', np.zeros((4, 2), np.int64))
            np.save(cache/'b-labels.npy', np.array([True]))
            np.save(cache/'b-pairs.npy', np.zeros((1, 2), np.int64))
            plan = np.zeros((2, 4, 2), np.int64); plan[:, :, 1] = [0, 2, 1, 3]
            exposure = audit.plan_exposure(plan, ['a', 'b'], cache, audit.ScoreChecks(), steps=2, batch=4)
            self.assertEqual(exposure['sampled_pair_count'], 8)
            self.assertEqual(exposure['sampled_positive_count'], 4)
            self.assertEqual(exposure['distinct_pair_count'], 4)
            self.assertEqual(exposure['repeated_pair_count'], 4)
            self.assertEqual(exposure['visited_scene_count'], 1)
            self.assertEqual(exposure['scenes'][1]['sampled_batch_count'], 0)
            wrong = plan.copy(); wrong[0, 0, 1] = 100
            with self.assertRaisesRegex(ValueError, 'original pair indices'):
                audit.plan_exposure(wrong, ['a', 'b'], cache, audit.ScoreChecks(), steps=2, batch=4)
            wrong = plan.copy(); wrong[0, 0, 1] = 1
            with self.assertRaisesRegex(ValueError, 'half positive'):
                audit.plan_exposure(wrong, ['a', 'b'], cache, audit.ScoreChecks(), steps=2, batch=4)

    def test_all_six_pair_directions_and_existing_output_protected(self):
        rows = records(); scores = {arm: [r['scores']+i for r in rows] for i, arm in enumerate(audit.ARMS)}
        thresholds = dict(zip(audit.ARMS, [0., 1.2, None, 2.]))
        reports = {}; macros = {}
        for arm in audit.ARMS:
            made = [dict(r, scores=x) for r, x in zip(rows, scores[arm])]
            macros[arm] = producer.describe(made, thresholds[arm])['raw']['macro_f1']
        for left, right in audit.combinations(audit.ARMS, 2):
            a = [dict(r, scores=x) for r, x in zip(rows, scores[left])]
            b = [dict(r, scores=x) for r, x in zip(rows, scores[right])]
            value = producer.compare_predictions(a, b, thresholds[left], thresholds[right])
            value.update(macro_f1_difference=macros[right]-macros[left], baseline_method=left, candidate_method=right)
            reports[right+'_vs_'+left] = value
        audit.inspect_pairs(reports, rows, scores, thresholds, macros, audit.ScoreChecks())
        corrupted = deepcopy(reports); corrupted[next(iter(corrupted))]['wrong_to_right'] += 1
        with self.assertRaisesRegex(ValueError, 'wrong_to_right'):
            audit.inspect_pairs(corrupted, rows, scores, thresholds, macros, audit.ScoreChecks())
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)/'existing'; output.write_text('preserve')
            with self.assertRaises(FileExistsError):
                audit.audit(Path(temporary)/'missing', output)
            self.assertEqual(output.read_text(), 'preserve')


if __name__ == '__main__':
    unittest.main()
