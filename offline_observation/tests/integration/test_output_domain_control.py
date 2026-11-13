"""Output-domain software checks on synthetic saved caches, not MTAD effects."""
from copy import deepcopy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from tests.integration import test_query_contribution_control as pilot_tests
from tests.integration import test_query_contribution_full_control as full_tests
from tools.analysis import output_domain_control as p


def scene_fixture():
    fragment, memberships, artifacts115, public115, _ = pilot_tests.scene_fixture()
    public116, _, all_artifacts, _ = p.q.analyze_scene(fragment, memberships, artifacts115, public115)
    retained = {r: all_artifacts[r+'_NEAREST_SUPPORTED'] for r in p.q.PARTITIONS}
    public117 = dict(scene_id='1', component_id='large', node_count=3, stratum='original32',
                     original056_A0_exact_verified=True, fixed_R_population_invariants_verified=True,
                     methods=public116['methods'])
    return fragment, memberships, retained, public117


def runner_fixture(root):
    args, meta, _ = full_tests.cache_fixture(root)
    full = root/'full117'
    (full/'scenes').mkdir(parents=True)
    public = p.old.frozen.read(args.pilot_run/'scenes'/'scene-1.json.gz')
    public.update(stratum='original32', original056_A0_exact_verified=True,
                  fixed_R_population_invariants_verified=True)
    p.old.frozen.write(full/'scenes'/'scene-1.json.gz', public)
    state = dict(status='COMPLETED', idea_id=p.prior.IDEA, plan_version='1.0',
                 source_system=str(args.source_system), partition_run=str(args.partition_run),
                 pilot_run=str(args.pilot_run), scene_ids=meta['all_ids'], original32_ids=meta['ids'],
                 completed_scenes=len(meta['all_ids']))
    summary = dict(status='COMPLETED', idea_id=p.prior.IDEA, plan_version='1.0',
                   scene_ids=meta['all_ids'], original32=dict(scene_ids=meta['ids']),
                   all_original056_A0_exact_verified=True, fixed_R_population_invariants_verified=True)
    p.old.frozen.write(full/'run.json', state)
    p.old.frozen.write(full/'summary.json', summary)
    selected = dict(meta, pilot_ids=meta['ids'], ids=meta['all_ids'])
    return SimpleNamespace(**vars(args), full_run=full), selected


class OutputDomainTests(unittest.TestCase):
    def test_input_only_union_interval_empty_group_and_no_mutation(self):
        support = np.array([[0,1,0,0,0], [0,0,0,1,0], [0,0,0,0,0]], bool)
        before = support.copy()
        domains = p.position_domains(support, [[0,1], [2]])
        np.testing.assert_array_equal(domains['UNION'], [[0,1,0,1,0], [0,0,0,0,0]])
        np.testing.assert_array_equal(domains['SPAN'], [[0,1,1,1,0], [0,0,0,0,0]])
        self.assertTrue(domains['FULL'].all())
        np.testing.assert_array_equal(support, before)
        for bad in (support.astype(float), support[:, :0]):
            with self.assertRaises(ValueError): p.position_domains(bad, [[0,1], [2]])
        with self.assertRaises(ValueError): p.position_domains(support, [[0,1], [1,2]])

    def test_real_evaluation_keeps_denominator_lost_risk_and_all_non_mask_arrays(self):
        fragment, memberships, retained, public = scene_fixture()
        original = deepcopy(retained)
        row, arrays, artifacts = p.analyze_scene(fragment, memberships, retained, public)
        self.assertEqual(set(row['methods']), set(p.METHODS))
        for r, covered, lost in (('A0', 128, 2), ('WS', 64, 1)):
            full = row['methods'][r+'_FULL']['position']['all']
            self.assertEqual(full['available_target_count'], 192)
            self.assertEqual(full['intersection_count'], covered)
            for domain in ('UNION', 'SPAN'):
                name = r+'_'+domain
                own = row['methods'][name]['position']['all']
                self.assertEqual(own['available_target_count'], 192)
                self.assertEqual(own['intersection_count'], covered-lost)
                key = name+'_vs_'+r+'_FULL'
                self.assertEqual(row['pairwise'][key]['coverage_transitions']['lost_target_points'], lost)
                self.assertEqual(row['pairwise'][key]['coverage_transitions']['gained_target_points'], 0)
                self.assertEqual(len(arrays['pairs'][key]['_lost']), lost)
                np.testing.assert_array_equal(arrays['pairs'][key]['all']['delta'], 0.)
                np.testing.assert_array_equal(artifacts[name]['group_states'], original[r]['group_states'])
                np.testing.assert_array_equal(artifacts[name]['group_valid'][...,2:], original[r]['group_valid'][...,2:])
        merged = p.aggregate([row], [arrays])
        self.assertEqual(merged['pairwise']['A0_UNION_vs_A0_FULL']['rejected_baseline_position']['point_count'], 2)
        self.assertGreater(merged['pairwise']['A0_UNION_vs_A0_FULL']['rejected_baseline_position']['mae'], 0.)
        self.assertTrue(merged['pairwise']['WS_UNION_vs_WS_FULL']['common_point_error_change_exact_zero'])
        for r in p.q.PARTITIONS:
            for key in retained[r]:
                np.testing.assert_array_equal(retained[r][key], original[r][key])

    def test_all_domains_are_frozen_before_reference_access_and_corrupt_baseline_fails(self):
        fragment, memberships, retained, public = scene_fixture()
        calls = []
        real_construct = p.construct_outputs
        class Guarded(dict):
            def __getitem__(self, key):
                if key in {'target_identity_ids','identities','target','target_valid'} and not calls:
                    raise AssertionError('reference reached before output domains were frozen')
                return super().__getitem__(key)
        def construct(*args):
            result = real_construct(*args)
            calls.append('all-six-frozen')
            return result
        with patch.object(p, 'construct_outputs', side_effect=construct):
            p.analyze_scene(Guarded(fragment), memberships, retained, public)
        corrupt = deepcopy(retained)
        corrupt['A0']['group_states'][0, 1, 0] += 1e-12
        with self.assertRaisesRegex(ValueError, '116 baseline artifact'):
            p.analyze_scene(fragment, memberships, corrupt, public)
        corrupt_public = deepcopy(public)
        corrupt_public['methods']['A0_NEAREST_SUPPORTED']['position']['all']['mae'] += 1.
        with self.assertRaisesRegex(ValueError, '117 baseline report'):
            p.analyze_scene(fragment, memberships, retained, corrupt_public)

    def test_runner_stops_at_original_pilot_and_only_writes_private_arrays(self):
        with tempfile.TemporaryDirectory() as temp:
            args, meta = runner_fixture(Path(temp).resolve())
            with patch.object(p.prior, 'metadata', return_value=deepcopy(meta)), \
                 patch.object(p.q, 'construct_predictions', side_effect=AssertionError('no recovery call')):
                summary = p.run(args)
            self.assertEqual(summary['scene_ids'], ['1'])
            self.assertEqual(summary['scene_count'], 1)
            self.assertEqual(set(summary['methods']), set(p.METHODS))
            self.assertTrue(summary['no_expansion'])
            self.assertIsNone(summary['without_largest_component'])
            self.assertEqual(len(list((args.output/'private').glob('*.npz'))), 1)
            state = p.old.frozen.read(args.output/'run.json')
            self.assertEqual((state['status'],state['completed_scenes'],state['exit_code']), ('COMPLETED',1,0))
            for field in ('models_constructed','model_forwards','optimizer_updates'):
                self.assertEqual(state[field], 0)
            for field in ('gpu_used','scores_recomputed','groups_recomputed','recovery_recomputed',
                          'threshold_reselected','normalizer_refitted','archive_read','csv_read','test_opened'):
                self.assertIs(state[field], False)
            forbidden = {'target_ids','target_identity_ids','node_ids','identities','sources','native_states',
                         'group_states','aligned_truth','aligned_prediction','node_group_indices'}
            def inspect(value):
                if isinstance(value,dict):
                    self.assertFalse(set(value)&forbidden)
                    for child in value.values(): inspect(child)
                elif isinstance(value,list):
                    for child in value: inspect(child)
            inspect(summary)
            with self.assertRaises(FileExistsError): p.run(args)

    def test_metadata_rejects_wrong_original32_and_failure_preserves_terminal(self):
        with tempfile.TemporaryDirectory() as temp:
            args, meta = runner_fixture(Path(temp).resolve())
            original = p.old.frozen.read(args.full_run/'run.json')
            with patch.object(p.prior, 'metadata', return_value=deepcopy(meta)):
                for key,value in (('status','RUNNING'),('original32_ids',['2']),('completed_scenes',2),
                                  ('pilot_run',str(Path(temp)/'other'))):
                    p.old.frozen.write(args.full_run/'run.json',dict(original,**{key:value}))
                    with self.subTest(key=key), self.assertRaisesRegex(ValueError,'original117'):
                        p.metadata(args.source_system,args.partition_run,args.pilot_run,args.full_run,{})
                p.old.frozen.write(args.full_run/'run.json', original)
            with patch.object(p.prior, 'metadata', return_value=deepcopy(meta)), \
                 patch.object(p.old.frozen, 'peak_rss_bytes', return_value=p.old.frozen.RAM_BUDGET+1):
                with self.assertRaises(MemoryError): p.run(args)
            state = p.old.frozen.read(args.output/'run.json')
            self.assertEqual((state['status'],state['completed_scenes'],state['exit_code']), ('FAILED',0,1))


if __name__ == '__main__':
    unittest.main()
