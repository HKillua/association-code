"""103 software checks; synthetic arrays are NOT MTAD experiment evidence."""
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from association.data.archive import Scene, Track
from association.data.native import _record
from association.data.timed import prepare_timed_scene
from association.data.motion_evidence import scene_motion_evidence
from association.data.targets import align_reference_states
from association.evaluation.scene import evaluate_full_scene
from tests.data.test_fusion_targets import scene as reference_scene
from tools.analysis.diagnose_three_scenes import (
    METHODS, REGIONS, aggregate_points, analyze_scene, brackets, clean, cosine,
    local_delta, radial, read, read_cache, relative_summary, source_metadata, write,
)


def fixture(root):
    tracks = []
    for i, (source, times) in enumerate((('1', [0., 20.]), ('2', [5., 15.]), ('1', [0., 20.]))):
        values = np.array([[30.+t*.00001, 120.+i*.001, 2., 90.] for t in times])
        tracks.append(Track(f'3035:{source}:{i}', source, str(i), 'A-0', 'A', np.array(times), values))
    raw = Scene('3035', tracks, {})
    prepared = prepare_timed_scene(_record(raw, 'component', 'V_select', 128),
                                  length=3, max_observations=128, time_unit_seconds=20.)
    motion = scene_motion_evidence(prepared, motion_observation_policy='canonical_all')
    targets = ['A-0', 'B-0', 'C-0']
    reference = reference_scene([(identity, t+100., [30.+t*.00001, 120., 2., 90.])
        for identity in targets[:2] for t in (0., 10., 20.)], split='V_select')
    policy = dict(time_offset_seconds=100., max_gap_seconds=600.)
    aligned = [align_reference_states(reference, identity, prepared['grid'], **policy,
                                     longitude_policy='shortest_arc') for identity in targets]
    truth, valid = np.array([r['states'] for r in aligned]), np.array([r['valid'] for r in aligned])
    groups = [[0, 1], [2]]  # Second A group is a duplicate; B/C remain missing.
    artifacts = {}
    for m in METHODS:
        output = [dict(node_indices=g, states=motion['states'][g[0]].copy()) for g in groups]
        artifacts[m] = evaluate_full_scene(output, node_identities=prepared['identities'],
            target_identities=targets, target_states=truth, target_valid=valid,
            node_observed=motion['support'], split='V_select', grouping_kind='predicted',
            correspondence_policy='anchor_identity_first', return_artifact=True,
            node_input_support=dict(support=motion['support'], nearest_seconds=motion['nearest_seconds']))['artifact']
    base = artifacts['anchor']
    region = base['aligned_input_support_kind'][:len(targets)]
    fragment = dict(node_ids=np.array(prepared['node_ids']), identities=prepared['identities'],
        target_identity_ids=np.array(targets), native_states=motion['states'], support=motion['support'],
        nearest_seconds=motion['nearest_seconds'], exact_observation=motion['exact_observation'],
        query_seconds=motion['query_seconds'], learned_assoc_group_offsets=np.array([0, 2, 3]),
        learned_assoc_group_node_indices=np.array([0, 1, 2]), group_anchor_node_indices=np.array([0, 2]),
        region_all=np.ones((3, 3), bool),
        **{'region_'+name: region == i for i, name in enumerate(REGIONS)})
    for m in METHODS:
        fragment[m+'_group_states'] = artifacts[m]['group_states']
        mask = (base['aligned_truth_valid'][:3, :, :2] & base['aligned_prediction_valid'][:3, :, :2]).all(-1)
        fragment[m+'_position_error'] = np.where(mask, radial(base['aligned_prediction'][:3], truth), np.nan)
    (root/'final/predictions').mkdir(parents=True)
    (root/'final/system-predictions').mkdir()
    np.savez_compressed(root/'final/predictions/states-3035.npz', **fragment)
    np.savez_compressed(root/'final/system-predictions/states-3035.npz',
                        **{m+'_'+k: v for m, a in artifacts.items() for k, v in a.items()})
    return prepared, targets, reference, policy


class ThreeSceneDiagnosisTests(unittest.TestCase):
    def test_source_metadata_binds_configuration_and_original_system_without_weights(self):
        with TemporaryDirectory() as temp:
            root = Path(temp).resolve(); (root/'final').mkdir()
            config = read(Path(__file__).resolve().parents[2]/'configs/frozen-position-adapter-full-finetune-level1.json')
            run = dict(status='COMPLETED', configuration=config, optimizer_updates=4000,
                       test_opened=False, source_system='/original/056')
            final = dict(status='COMPLETED', configuration=config, optimizer_updates=0,
                test_opened=False, source_system='/original/056', initial=False,
                groups_recomputed=False, threshold_reselected=False,
                checkpoint=dict(step=4000, actual_updates=4000, path=str(root/'fit/final.pt')),
                scene_ids=['3035'])
            write(root/'run.json', run); write(root/'final/run.json', final); write(root/'final/summary.json', final)
            source_metadata(root, {'splits': {'V_select': ['3035']}})
            bad = deepcopy(run); bad['configuration']['training']['seed'] += 1
            write(root/'run.json', bad)
            with self.assertRaisesRegex(ValueError, 'configuration'):
                source_metadata(root, {'splits': {'V_select': ['3035']}})
            write(root/'run.json', run); final['source_system'] = '/different/system'
            write(root/'final/run.json', final)
            with self.assertRaisesRegex(ValueError, 'endpoint metadata'):
                source_metadata(root, {'splits': {'V_select': ['3035']}})

    def test_zero_vector_is_unknown_and_dateline_is_short_arc(self):
        self.assertTrue(np.isnan(cosine([0., 0.], [1., 1.])))
        self.assertAlmostEqual(float(cosine([1., 0.], [-1., 0.])), -1.)
        delta = local_delta([0., -179.9], [0., 179.9], 0.)
        self.assertAlmostEqual(delta[1], float(radial([0., -179.9], [0., 179.9])), places=6)
        self.assertEqual(clean(np.array(float('nan'))), None)

    def test_reference_endpoints_preserve_conflicting_duplicates_and_offset(self):
        ref = reference_scene([('A-0', 100., [30, 120, 2, 90]),
                               ('A-0', 100., [31, 120, 2, 90]),
                               ('A-0', 120., [32, 120, 2, 90])], split='V_select')
        rows = brackets(ref, 'A-0', [0., 10., 30.], dict(time_offset_seconds=100., max_gap_seconds=10.))
        self.assertEqual(len(rows[0]['left']['rows']), 2)
        self.assertEqual(rows[0]['left']['aligned_time'], 0.)
        self.assertEqual(rows[1]['kind'], 'gap_exceeded')
        self.assertEqual(rows[2]['kind'], 'outside_support')

    def test_real_evaluator_schema_preserves_missing_duplicates_and_raw_query_distinction(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            prepared, targets, ref, policy = fixture(root)
            cached = read_cache(root, '3035')
            summary, public, private = analyze_scene(prepared, targets, ref, policy, cached, {})
            self.assertEqual(summary['population']['duplicate_prediction_group_count'], 1)
            self.assertEqual(len(public['points']), 9)
            self.assertEqual(summary['strata']['all']['available'], 6)
            self.assertEqual(summary['strata']['all']['evaluated'], 3)
            self.assertEqual(sum(r['missing_reason'] == 'no_predicted_group' for r in public['points']), 3)
            self.assertEqual(sum(r['missing_reason'] == 'no_position_reference' for r in public['points']), 3)
            cross = [r for r in public['query_pairs'] if r['node_a'] == 0 and r['node_b'] == 1]
            self.assertTrue(any(r['both_supported'] for r in cross))
            same = [r for r in public['simultaneous_native_pairs'] if r['node_a'] == 0 and r['node_b'] == 1]
            self.assertEqual([r['kind'] for r in same], ['no_exact_common_native_time'])
            self.assertTrue(any(not r['selected_unique_group'] for r in public['query_sources']))
            # Serialization of unmeasured/undefined values stays null, never zero.
            write(root/'public.json.gz', public)
            self.assertTrue((root/'public.json.gz').exists())
            bad = deepcopy(ref); bad['rows'][0]['state'][0] += .1
            with self.assertRaisesRegex(ValueError, 'reference states'):
                analyze_scene(prepared, targets, bad, policy, cached, {})

    def test_tail_overlap_keeps_denominator_and_zero_error(self):
        rows = []
        for errors, pv, tv in (([3., 4., 5.], True, True), ([0., 0., 0.], True, True),
                               ([np.nan]*3, False, True), ([np.nan]*3, True, False)):
            rows.append(dict(truth_valid=tv, prediction_valid=pv,
                             methods={m: dict(error_m=e) for m, e in zip(METHODS, errors)}))
        result = aggregate_points(rows, {'fixed': {m: 2. for m in METHODS}})
        self.assertEqual((result['declared'], result['available'], result['evaluated']), (4, 3, 2))
        self.assertEqual(result['overlap']['fixed']['pattern_counts'], {'111': 1, '000': 1})
        self.assertEqual(result['methods']['anchor']['mean'], 2.)

    def test_relative_bias_is_target_equal_and_oracle_only(self):
        def row(target, north, same=True):
            return dict(source_a='1', source_b='2', oracle_same_identity=same,
                        distance_m=abs(north), both_supported=True, target_a=target, delta_NE_m=[north, 0.])
        rows = [row(0, 10.)]*9+[row(1, 30.), row(2, 1000., False)]
        result = relative_summary(rows, [])[0]
        self.assertEqual(result['matched_target_count'], 2)
        self.assertEqual(result['target_equal_mean_NE_m'][0], 20.)
        self.assertTrue(result['oracle_correspondence'])


if __name__ == '__main__':
    unittest.main()
