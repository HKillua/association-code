"""060 cache comparison: complete population, changed anchors and failed replay."""
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from association.evaluation.fusion_controls import ordinary_fusion
from association.evaluation.tmtaff import aggregate_fusion_reports
from association.utils.runtime import read_json, write_json
from tools.analysis.compare_association_duration import ARMS, METHODS, run
from tools.analysis.system_attribution import read
from tools.experiments.level1_system_evaluate import evaluate_unique_population, ordinary_predictions


class AssociationDurationTests(unittest.TestCase):
    @staticmethod
    def fixture(root):
        q = 64
        nodes = np.asarray(['x', 'a', 'a', 'b', 'a', 'd'])
        targets = np.asarray(['x', 'a', 'b', 'c', 'd'])
        truth = np.broadcast_to(np.array([[30., 120., 5., 1.], [30.01, 120.01, 6., 359.],
            [30.02, 120.02, 7., 2.], [30.03, 120.03, 8., 180.], [30.04, 120.04, 9., 90.]])[:, None],
            (5, q, 4)).copy()
        valid = np.ones_like(truth, bool)
        valid[4] = False  # Still in the target registry and output counts.
        valid[1, 0, 2] = False
        indices = [0, 1, 1, 2, 1, 4]
        native = truth[indices].copy()
        native[2, :, 0] += .002
        native[4, :, 0] -= .004
        support, nearest = np.ones((6, q), bool), np.full((6, q), 10.)
        support[:, 32:] = False
        nearest[:, 32:] = 700.
        exact = np.zeros((6, q), bool)
        exact[:, 0] = True
        nearest[:, 0] = 0.
        scene = dict(identities=nodes, node_ids=np.asarray(['n' + str(i) for i in range(6)]),
            sources=np.asarray(['9001', '9002', '9001', '9002', '9003', '9001']),
            target_identity_ids=targets.tolist(), target=truth[indices], valid=valid[indices],
            motion=dict(states=native, query_seconds=np.linspace(0., 1260., q),
                        support=support, nearest_seconds=nearest, exact_observation=exact),
            fusion_targets={identity: dict(states=truth[i], valid=valid[i]) for i, identity in enumerate(targets)})
        ids = ['01', '02']
        manifest = dict(dataset_level='Level1', split_unit='scene',
                        splits=dict(T_core=['train'], V_select=ids, T_test=['test']))
        manifest_path = root/'manifest.json'
        write_json(manifest_path, manifest)
        old, new = [[0, 1], [2, 3], [4], [5]], [[0], [1, 2, 4], [3], [5]]
        for name, step, partitions in (('reference', 1000, [old, new]), ('candidate', 3000, [new, old])):
            source = root/name
            (source/'predictions').mkdir(parents=True)
            (source/'system-predictions').mkdir()
            rows = []
            for sid, groups in zip(ids, partitions):
                grouping = dict(ols=[[0, 1, 2, 3, 4], [5]], learned_assoc=groups)
                controls = {key: ordinary_fusion(native, nearest, support, value) for key, value in grouping.items()}
                by_group = {key: ordinary_predictions(value, ARMS) for key, value in controls.items()}
                reports, artifacts = evaluate_unique_population(scene, grouping, by_group)
                fragment = dict(node_ids=scene['node_ids'], identities=nodes, sources=scene['sources'],
                    target_identity_ids=targets, query_seconds=scene['motion']['query_seconds'],
                    target=scene['target'], target_valid=scene['valid'], native_states=native,
                    support=support, nearest_seconds=nearest, exact_observation=exact)
                for key, members in grouping.items():
                    fragment.update({key + '_' + arm: values for arm, values in by_group[key].items()})
                    fragment[key + '_membership'] = controls[key]['membership']
                    fragment[key + '_group_offsets'] = np.r_[0, np.cumsum([len(group) for group in members])]
                    fragment[key + '_group_node_indices'] = np.asarray([i for group in members for i in group])
                np.savez_compressed(source/'predictions'/f'states-{sid}.npz', **fragment)
                np.savez_compressed(source/'system-predictions'/f'states-{sid}.npz',
                    **{method + '_' + key: value for method, artifact in artifacts.items() for key, value in artifact.items()})
                rows.append(dict(scene_id=sid, unique_target=reports))
            common = dict(status='COMPLETED', dataset_level='Level1', split='V_select', test_opened=False,
                optimizer_updates=0, association_endpoint='final', ordinary_controls=list(ARMS),
                method_names=METHODS, fusion_checkpoint_step=None)
            fit = root/(name + '-fit')
            fit.mkdir()
            write_json(fit/'run.json', dict(status='COMPLETED', optimizer_updates=step,
                       config=dict(training=dict(updates=step))))
            inputs = dict(association_run=str(fit/'run.json'))
            write_json(source/'run.json', dict(common, motion_observation_policy='canonical_all',
                threshold_reselected=False, normalizer_refitted=False, completed_scenes=2, total_scenes=2,
                inputs=inputs))
            write_json(source/'summary.json', dict(common, scene_ids=ids, scene_count=2, fragment_count=12,
                association_endpoint_step=step, association_threshold=-1., association={}, inputs=inputs,
                ols_working_point=dict(motion_observation_policy='canonical_all', validation_scene_ids=ids, threshold=-2.),
                per_scene=rows, unique_target={method: aggregate_fusion_reports([row['unique_target'][method] for row in rows])
                    for method in METHODS}))
        write_json(root/'candidate-fit'/'prefix-comparison.json', dict(comparison_status='MISMATCH',
            causal_attribution_valid=False, prefix_update=1000, source_endpoint='final', protocol_valid=True,
            source_directory=str(root/'reference-fit')))
        return SimpleNamespace(reference_system=root/'reference', candidate_system=root/'candidate',
                               manifest=manifest_path, output=root/'comparison', cpu_threads=1)

    def test_complete_paired_comparison_preserves_coverage_duplicates_and_representatives(self):
        with TemporaryDirectory() as temporary:
            args = self.fixture(Path(temporary))
            result = run(args)
            self.assertEqual(read_json(args.output/'run.json')['status'], 'COMPLETED')
            self.assertEqual(result['completed_scenes'], 2)
            self.assertEqual(result['fragment_count'], 12)
            self.assertEqual(result['endpoints']['candidate']['endpoint_step'], 3000)
            self.assertEqual(result['prefix_comparison']['comparison_status'], 'MISMATCH')
            self.assertFalse(result['causal_attribution_valid'])
            pair = result['pairwise']['anchor']
            self.assertEqual(pair['changed_representative_target_count'], 2)
            self.assertEqual(pair['same_representative_target_count'], 4)
            transition = pair['coverage_transitions']['position']
            self.assertEqual(transition, dict(rescued_target_count=1, lost_target_count=1,
                                             gained_target_points=64, lost_target_points=64))
            same = pair['error_change_second_minus_first']['same_representative']['position']
            self.assertEqual(same['mean_error_change'], 0.)
            self.assertEqual(same['tied_point_count'], 128)
            all_points = pair['methods']['reference']['all']['position']
            self.assertEqual(all_points['available_target_count'], 512)  # Includes no-input C.
            self.assertEqual(all_points['intersection_count'], 256)
            self.assertEqual(all_points['scene_macro_coverage'], .5)
            scenes = read(args.output/'scene-summary.json.gz')
            population = scenes[0]['methods']['reference']['learned_assoc_anchor']['population']
            self.assertEqual(population['predicted_group_count'], 4)
            self.assertEqual(population['anchor_matched_target_count'], 3)
            self.assertEqual(population['duplicate_prediction_group_count'], 1)
            self.assertEqual(population['reference_unavailable_target_count'], 1)
            changed = scenes[0]['pairwise']['anchor']['error_change_second_minus_first']['changed_representative']['position']
            self.assertLess(changed['mean_error_change'], 0.)
            with self.assertRaises(FileExistsError):
                run(args)

    def test_rejects_protocol_input_control_and_alignment_changes(self):
        for case in ('endpoint', 'input', 'ols', 'alignment'):
            with self.subTest(case=case), TemporaryDirectory() as temporary:
                args = self.fixture(Path(temporary))
                if case == 'endpoint':
                    path = args.candidate_system/'summary.json'
                    summary = read_json(path)
                    summary['association_endpoint_step'] = 2999
                    write_json(path, summary)
                else:
                    directory = 'system-predictions' if case == 'alignment' else 'predictions'
                    path = args.candidate_system/directory/'states-01.npz'
                    with np.load(path) as saved:
                        arrays = {key: saved[key] for key in saved.files}
                    if case == 'input':
                        arrays['target_valid'][0, 0, 0] = False
                    elif case == 'ols':
                        arrays['ols_mean_supported'][0, 0, 0] += .01
                    else:
                        arrays['learned_assoc_anchor_aligned_group_indices'][1] = 0
                    np.savez_compressed(path, **arrays)
                with self.assertRaises(ValueError):
                    run(args)
                self.assertEqual(read_json(args.output/'run.json')['status'], 'FAILED')

    def test_rejects_wrong_or_unfinished_training_sources(self):
        for case in ('source_disagreement', 'wrong_source', 'unfinished', 'actual_updates', 'configured_updates'):
            with self.subTest(case=case), TemporaryDirectory() as temporary:
                args = self.fixture(Path(temporary))
                if case in ('source_disagreement', 'wrong_source'):
                    names = ('run.json',) if case == 'source_disagreement' else ('run.json', 'summary.json')
                    for name in names:
                        path = args.candidate_system/name
                        document = read_json(path)
                        document['inputs']['association_run'] = str(Path(temporary)/'reference-fit'/'run.json')
                        write_json(path, document)
                else:
                    path = Path(temporary)/'candidate-fit'/'run.json'
                    document = read_json(path)
                    if case == 'unfinished':
                        document['status'] = 'RUNNING'
                    elif case == 'actual_updates':
                        document['optimizer_updates'] = 2999
                    else:
                        document['config']['training']['updates'] = 4000
                    write_json(path, document)
                with self.assertRaisesRegex(ValueError, 'training source'):
                    run(args)
                self.assertEqual(read_json(args.output/'run.json')['status'], 'FAILED')

    def test_interrupted_analysis_records_status_and_never_overwrites_sources(self):
        with TemporaryDirectory() as temporary:
            args = self.fixture(Path(temporary))
            with patch('tools.analysis.compare_association_duration._load_metadata', side_effect=KeyboardInterrupt):
                with self.assertRaises(KeyboardInterrupt):
                    run(args)
            self.assertEqual(read_json(args.output/'run.json')['status'], 'INTERRUPTED')
            args.output = args.reference_system/'nested-output'
            with self.assertRaisesRegex(ValueError, 'separate new output'):
                run(args)
            self.assertFalse(args.output.exists())


if __name__ == '__main__':
    unittest.main()
