"""Synthetic V caches check wiring and scope; they are not MTAD evidence."""
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from tests.integration.test_source_risk_pilot import scene_arrays
from association.evaluation import history_candidates as history
from association.evaluation import native_motion_interventions as motion
from association.evaluation.local_source_evidence import ResidualScore, predict_residual
from tools.analysis import action_conditioned_risk_validation as pilot
from tools.analysis import action_conditioned_risk_validation_closeout as audit

p = pilot.p


def cache(root):
    parent, control, native = (root / name for name in ('178', '191', '127'))
    for path in (parent, control):
        (path / 'private').mkdir(parents=True)
        (path / 'scenes').mkdir()
    ids = [str(i) for i in range(32, 64)]
    tids = [str(i) for i in range(100, 228)]
    manifest_path = root / 'manifest.json'
    manifest = dict(splits=dict(T_core=tids, V_select=ids, T_test=['sealed']),
                    components={sid: 'V' + str(i % 9) for i, sid in enumerate(ids)})
    write = p.common.old.frozen.write
    write(manifest_path, manifest)
    record = dict(status='COMPLETED', exit_code=0, native_run=str(native),
                  idea_id='IDEA-HISTORY-CANDIDATE-ACTIONS-178',
                  training_scene_ids=tids[:32], fit_training_scene_ids=tids,
                  scene_ids=ids, manifest=str(manifest_path),
                  thresholds={'HARD': 1.7320103112387182, 'STRUCT': 1.7320103112387182})
    write(parent / 'run.json', record)
    control_record = dict(record, parent_run=str(parent),
          idea_id='IDEA-MEAN-VELOCITY-VALIDATION-191',
          training_scene_ids=[], evaluation_splits=['V_select'],
          completed_V_scenes=32, methods=[v + '_' + k for arms in pilot.ARMS.values()
                                         for v in arms[:7] for k in ('MEAN', 'NN')])
    del control_record['fit_training_scene_ids']
    write(control / 'run.json', control_record)
    # Only the full model exists. A mistaken leave-component load must fail.
    torch.manual_seed(20)
    net = ResidualScore(12)
    model = dict(dimension=12, mean=np.zeros(12), std=np.ones(12),
                 base=dict(mean=np.zeros(12), std=np.ones(12), beta=np.zeros(91)),
                 weights={k: v.detach().clone() * .01 for k, v in net.state_dict().items()},
                 parameter_count=1505, updates=256)
    model['base']['beta'][0] = .25
    model['base']['beta'][7] = -.01
    torch.save(model, parent / 'private' / 'model-full.pt')
    control_rows, control_arrays = [], []
    methods = tuple(v + '_' + k for arms in pilot.ARMS.values()
                    for v in arms[:7] for k in ('MEAN', 'NN'))
    for sid in ids:
        m, ev, member, _ = scene_arrays(sid)
        m['support'][:, -1] = False
        m['native_count'] = np.full(4, 2)
        m['span_seconds'] = np.full(4, 30.)
        groups = [[0, 1], [2], [3]]
        x, count = history.descriptors(m, groups)
        scores = predict_residual(model, x.reshape(-1, 12), 'cpu').reshape(m['support'].shape)
        base = m['states'][[0, 2, 3]].copy()
        outputs, weights = history.readouts(m, groups, count, scores, base)
        values = np.repeat(m['states'][:, 0], 2, axis=0)
        values[:, 2] /= 1852 / 3600
        values[1::2, 2] += 1
        events = dict(node_ids=m['node_ids'], query_seconds=m['query_seconds'],
                      event_offsets=np.arange(5) * 2,
                      event_times_seconds=np.tile([-10., 20.], 4),
                      event_values=values, event_valid=np.ones(values.shape, bool))
        mc, flags, _ = motion.mean_velocity_control(m, events, groups, count, weights['O'], outputs['O'])
        persist, _, _ = motion.persistence(m, events, groups, count, weights['O'], outputs['O'])
        parts = dict(pairs=m['pairs'], scores=np.full(len(m['pairs']), 2.))
        op, raw, oldraw, relations, group_map, control_policy = {}, {}, {}, {}, {}, {}
        for front, arms in pilot.ARMS.items():
            parts[front] = member
            parts[front + '_scores'] = parts['scores']
            op.update({front + '__features': x, front + '__scores': scores,
                       front + '__count': count, arms[1] + '__weights': weights['O']})
            control_policy[front + '__eligible'] = flags['eligible']
            rec = dict(scene_id=sid, component_id=manifest['components'][sid],
                       pairs=m['pairs'], node_ids=m['node_ids'], sources=m['sources'],
                       labels=ev['identities'][m['pairs'][:, 0]] == ev['identities'][m['pairs'][:, 1]],
                       joint_support_queries=(m['support'][m['pairs'][:, 0]] & m['support'][m['pairs'][:, 1]]).sum(1))
            for kind, decision, threshold in (
                    ('raw', parts['scores'], p.partition.THRESHOLD),
                    ('decoded', (member[m['pairs'][:, 0]] == member[m['pairs'][:, 1]]).astype(float), .5)):
                relations[front + '_' + kind] = p.partition.frozen.evaluate_scenes([dict(rec, scores=decision)], threshold)
            for prefix in arms[:7]:
                raw[prefix + '_MEAN'] = base
                raw[prefix + '_NN'] = (base if prefix == front else outputs['O'] if prefix == arms[1]
                                       else mc if prefix == arms[6] else persist[prefix[1:]])
                parts[prefix] = member
                parts[prefix + '_scores'] = parts['scores']
                group_map[prefix] = groups
                for kind in ('raw', 'decoded'):
                    relations[prefix + '_' + kind] = relations[front + '_' + kind]
            for prefix in (front, arms[1]):
                for suffix in ('_MEAN', '_NN'):
                    oldraw[prefix + suffix] = raw[prefix + suffix]
        for category, values in [('native-motion', m), ('native-events', events), ('native-evaluation', ev)]:
            dest = native / 'private' / category / 'V_select'
            dest.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(dest / f'scene-{sid}.npz', **values)
        for dest, name, values in [(parent, 'partitions', parts), (parent, 'policy', op),
                                    (parent, 'raw', oldraw), (control, 'partitions', parts),
                                    (control, 'policy', control_policy), (control, 'raw', raw)]:
            np.savez_compressed(dest / 'private' / f'{name}-V_select-{sid}.npz', **values)
        row, arr, artifact = p.evaluate(m, ev, group_map, {}, raw, 'V_select', method_names=methods)
        row.update(scene_id=sid, component_id=manifest['components'][sid], search={}, association=relations)
        write(control / 'scenes' / f'V_select-{sid}.json.gz', row)
        p.common.save_artifacts(control / 'private' / f'states-V_select-{sid}.npz', artifact)
        control_rows.append(row)
        control_arrays.append(arr)
    control_primary = tuple(arms[6] + '_NN_vs_' + arms[1] + '_NN' for arms in pilot.ARMS.values())
    summary = p.finish(control_rows, control_arrays, methods, control_primary)
    summary['status'] = 'COMPLETED'
    write(control / 'summary.json', summary)
    return SimpleNamespace(parent_run=parent, control_run=control, output=root / '196', time_policy='advisory')


POPULATION = dict(scene_count=32, component_count=9, target_count=96, query_count=64,
                  reference_points=6112,
                  **{front: dict(points=4064, covered_targets=64, duplicate_groups=32,
                                 eligible_points=64) for front in ('HARD', 'STRUCT')})


class ActionRiskValidationIOTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = TemporaryDirectory()
        cls.args = cache(Path(cls.tmp.name))

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_v_only_full_model_prediction_before_reference_and_audit(self):
        original_load = p.common.load_npz
        opened = []

        def guarded_load(path, *args, **kwargs):
            path = Path(path)
            self.assertNotIn('T_core', str(path))
            self.assertNotIn('T_test', str(path))
            if 'native-evaluation' in str(path):
                sid = path.stem.removeprefix('scene-')
                for kind in ('raw', 'policy', 'partitions'):
                    self.assertTrue((self.args.output / 'private' / f'{kind}-V_select-{sid}.npz').exists())
                opened.append(sid)
            return original_load(path, *args, **kwargs)

        with patch.object(pilot, 'EXPECTED_POPULATION', POPULATION), \
                patch.object(p.common, 'load_npz', guarded_load), \
                patch.object(p.common.source, 'load_npz', guarded_load):
            pilot.run(self.args)
        self.assertEqual(len(opened), 32)
        record = p.common.old.frozen.read(self.args.output / 'run.json')
        self.assertEqual(record['status'], 'COMPLETED')
        self.assertEqual(record['training_scene_ids'], [])
        self.assertEqual(record['evaluation_splits'], ['V_select'])
        self.assertEqual(record['completed_V_scenes'], 32)
        self.assertEqual(len(record['methods']), 40)
        self.assertFalse((self.args.output / 'T-summary.json').exists())
        cohorts = {front: dict(old_MC_harms=64, old_MC_rescues=0,
                              old_O_front_harms=0, old_MC_front_harms=64,
                              main_points=4064, eligible_points=64)
                   for front in ('HARD', 'STRUCT')}
        with patch.object(pilot, 'EXPECTED_POPULATION', POPULATION), \
                patch.object(audit, 'EXPECTED_POPULATION', dict(scenes=32, components=9, targets=96, reference_points=6112)), \
                patch.object(audit, 'EXPECTED_COHORTS', cohorts):
            audit.audit(self.args.output)
        self.assertEqual(p.common.old.frozen.read(self.args.output / 'independent-output-audit.json')['status'], 'COMPLETED')
        self.assertTrue((self.args.output / 'four-attribute-accounting.json').exists())
        self.assertEqual(p.common.old.frozen.read(self.args.output / 'action-opportunity-accounting.json')['status'], 'COMPLETED')

    def test_rejects_t_control_and_preserves_failure(self):
        path = self.args.control_run / 'run.json'
        original = path.read_bytes()
        record = p.common.old.frozen.read(path)
        record['evaluation_splits'] = ['T_core']
        p.common.old.frozen.write(path, record)
        target = self.args.output.with_name('rejected')
        try:
            with self.assertRaises((ValueError, AssertionError)):
                pilot.run(SimpleNamespace(**dict(vars(self.args), output=target)))
            failed = p.common.old.frozen.read(target / 'run.json')
            self.assertEqual(failed['status'], 'FAILED')
            self.assertEqual(failed['completed_V_scenes'], 0)
            self.assertEqual(list((target / 'private').iterdir()), [])
        finally:
            path.write_bytes(original)


if __name__ == '__main__':
    unittest.main()
