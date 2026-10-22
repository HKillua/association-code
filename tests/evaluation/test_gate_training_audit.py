"""069 log-only contracts; fixtures require neither Torch nor checkpoints."""
from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from association.evaluation.gate_training_audit import audit_training, MODES


def training_fixture(updates=3):
    config = dict(scope='support-gate-v1', routing_semantics='supported-mean-position-gate-v1',
        data=dict(length=64, max_observations=128, time_unit_seconds=20.,
                  reference_longitude_policy='shortest_arc'),
        motion_observation_policy='canonical_all', train_splits=['T_core'], model=dict(initial_gate=.95),
        training=dict(seed=20261021, updates=updates, batch_size=8, learning_rate=.001,
                      gradient_clip=1., advisory_seconds=14400.))
    rows = []
    for step in range(1, updates+1):
        index = (step-1) % 3
        rows.append(dict(step=step, mode=MODES[index], loss=step/100.,
            gradient_norm=(0., .5, 2.)[index], batch=[dict(scene_id=scene,
                anchor_index=0, node_id=scene+':fragment0', members=[0, 1],
                eligible_query_points=index*4, eligible_valid_position_points=index*4,
                valid_position_points=64) for scene in ['a', 'b']*4]))
    mode_updates = [(updates+2-index)//3 for index in range(3)]
    counters = {mode: dict(updates=count, anchors=8*count,
        eligible_query_points=index*32*count, eligible_valid_position_points=index*32*count,
        valid_position_points=512*count, nonzero_gradient_updates=count if index else 0,
        zero_eligible_updates=0 if index else count)
        for index, (mode, count) in enumerate(zip(MODES, mode_updates))}
    exposure = dict(successful_updates=updates, nonzero_gradient_updates=sum(mode_updates[1:]),
        zero_eligible_updates=mode_updates[0], sampled_anchors=updates*8,
        eligible_query_points=32*mode_updates[1]+64*mode_updates[2],
        eligible_valid_position_points=32*mode_updates[1]+64*mode_updates[2],
        valid_position_points=512*updates, scenes=dict(a=4*updates, b=4*updates),
        anchors={'a:fragment0':4*updates, 'b:fragment0':4*updates}, modes=counters)
    return dict(scope=config['scope'], routing_semantics=config['routing_semantics'],
        loss_semantics='mean_valid_position_log1p_haversine_m_div_1000_all_queries_no_auxiliary',
        status='COMPLETED', configuration=config, optimizer_updates=updates,
        test_opened=False, motion_observation_policy='canonical_all',
        training_scene_ids=['a', 'b', 'unused'], exposure=exposure,
        final_checkpoint='final.pt', trained_scene_count=2), rows


class GateTrainingAuditTests(unittest.TestCase):
    def audit(self, run, rows, expected_updates=None):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'run.json').write_text(json.dumps(run))
            (root/'training.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
            return audit_training(root, expected_updates=run['optimizer_updates']
                                  if expected_updates is None else expected_updates)

    def test_complete_exposure_zero_gradient_and_fixed_mode_windows(self):
        run, rows = training_fixture(103)
        report = self.audit(run, rows)
        self.assertEqual(report['exposure'], run['exposure'])
        self.assertTrue(report['run_exposure_exact'])
        self.assertEqual(report['sampled_training_scene_count'], 2)
        self.assertEqual(report['unsampled_training_scene_ids'], ['unused'])
        self.assertEqual(report['unique_sampled_fragment_count'], 2)
        self.assertEqual(report['repeated_fragment_draw_count'], 822)
        self.assertEqual([(r['first_step'],r['last_step']) for r in report['windows']], [(1,100),(101,103)])
        first = report['windows'][0]['statistics']
        self.assertAlmostEqual(first['all']['log_loss']['mean'], .505)
        self.assertEqual(first['clean']['log_loss']['count'], 34)
        self.assertEqual(first['clean']['preclip_gradient_norm']['maximum'], 0.)
        self.assertEqual(first['wrong_only']['preclip_norm_greater_than_configured_clip_count'], 33)
        self.assertEqual(report['exposure']['zero_eligible_updates'], 35)
        self.assertFalse(report['linear_training_position_error_available'])
        self.assertIn('different sampled scenes', report['interpretation']['windows'])
        self.assertIn('before clipping', report['gradient_norm_semantics'])

    def test_failed_incomplete_and_wrong_terminal_sources_rejected(self):
        source, rows = training_fixture()
        for mutate in (lambda r:r.update(status='FAILED'), lambda r:r.update(status='INTERRUPTED'),
                       lambda r:r.update(test_opened=True), lambda r:r.update(optimizer_updates=2),
                       lambda r:r['configuration']['training'].update(updates=2),
                       lambda r:r.update(routing_semantics='other')):
            with self.subTest(mutate=mutate):
                run = deepcopy(source)
                mutate(run)
                with self.assertRaises(ValueError):
                    self.audit(run, rows, expected_updates=3)
        with self.assertRaisesRegex(ValueError, 'incomplete'):
            self.audit(source, rows[:-1])
        with self.assertRaisesRegex(ValueError, 'beyond'):
            self.audit(source, rows+[rows[-1]])

    def test_nonfinite_negative_out_of_order_and_mode_cycle_rejected(self):
        run, source = training_fixture()
        changes = [dict(loss=float('nan')), dict(gradient_norm=float('inf')),
                   dict(loss=-1.), dict(gradient_norm=-1.), dict(step=2),
                   dict(step=True), dict(mode='wrong_only')]
        for change in changes:
            with self.subTest(change=change):
                rows = deepcopy(source)
                rows[0].update(change)
                with self.assertRaises(ValueError):
                    self.audit(run, rows)
        rows = deepcopy(source)
        rows[1]['batch'].pop()
        with self.assertRaisesRegex(ValueError, 'eight'):
            self.audit(run, rows)

    def test_exposure_and_node_population_mismatches_rejected(self):
        source, rows = training_fixture()
        for mutate in (lambda r:r['exposure']['scenes'].update(a=11),
                       lambda r:r['exposure']['anchors'].update({'a:fragment0':11}),
                       lambda r:r['exposure']['modes']['clean'].update(updates=True),
                       lambda r:r['exposure'].update(eligible_valid_position_points=95),
                       lambda r:r.update(trained_scene_count=3)):
            with self.subTest(mutate=mutate):
                run = deepcopy(source)
                mutate(run)
                with self.assertRaises(ValueError):
                    self.audit(run, rows)
        for change in (dict(scene_id='V_select-scene'), dict(node_id='b:fragment0'),
                       dict(anchor_index=2, members=[2, 1]), dict(members=[1, 0]),
                       dict(members=[0,0]), dict(eligible_valid_position_points=1),
                       dict(valid_position_points=65)):
            with self.subTest(change=change):
                altered = deepcopy(rows)
                altered[2]['batch'][0].update(change)
                with self.assertRaises(ValueError):
                    self.audit(source, altered)

    def test_no_silent_compressed_summary_or_checkpoint_substitution(self):
        run, rows = training_fixture()
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'run.json').write_text(json.dumps(run))
            (root/'final.pt').write_text('not a checkpoint; must never be opened')
            with self.assertRaises(FileNotFoundError):
                audit_training(root, expected_updates=3)
            (root/'training.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
            self.assertTrue(audit_training(root, expected_updates=3)['run_exposure_exact'])
            text = (root/'training.jsonl').read_text()
            (root/'training.jsonl').write_text(text.replace('"step": 1', '"step": 1, "step": 1', 1))
            with self.assertRaisesRegex(ValueError, 'duplicate JSON'):
                audit_training(root, expected_updates=3)


if __name__ == '__main__':
    unittest.main()
