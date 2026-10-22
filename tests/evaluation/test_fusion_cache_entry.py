"""067 cache-only integration and immutable-source boundaries."""
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import os
import unittest
from unittest.mock import patch

import numpy as np

from association.evaluation.frozen_fusion_transfer import METHODS, LEARNED_METHODS, summarize_scene
from association.utils.runtime import write_json, write_json_gzip, read_json
from tests.evaluation import test_frozen_fusion_transfer as fixtures
from tools.analysis import fusion_cache_attribution as entry


class FusionCacheEntryTests(unittest.TestCase):
    def setUp(self):
        env = patch.dict(os.environ, {key: '1' for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS')})
        env.start()
        self.addCleanup(env.stop)

    def fixture(self, root):
        source, system, fusion = root/entry.TASK066, root/entry.TASK056/'system', root/entry.TASK057
        fragment, artifacts, routes = fixtures.FrozenFusionTransferTests.fixture()
        native = fragment['native_states']
        n, q = native.shape[:2]
        target_index = {identity: i for i, identity in enumerate(fragment['target_identity_ids'])}
        node_targets = [target_index[identity] for identity in fragment['identities']]
        fragment.update(node_ids=np.asarray([f'n{i}' for i in range(n)]), sources=np.ones(n, int),
            query_seconds=np.arange(q, dtype=float), exact_observation=fragment['support'].copy(),
            target=artifacts['anchor']['aligned_truth'][node_targets].copy(),
            target_valid=artifacts['anchor']['aligned_truth_valid'][node_targets].copy(),
            ols_group_offsets=fragment['learned_assoc_group_offsets'].copy(),
            ols_group_node_indices=fragment['learned_assoc_group_node_indices'].copy())
        fragment.pop('direct_only_prediction')
        for arm in entry.ORDINARY:
            # Use full fragment controls, including non-output fragments.
            from association.evaluation.fusion_controls import ordinary_fusion
            controls = ordinary_fusion(native, fragment['nearest_seconds'], fragment['support'], [[0, 1], [2, 3], [4]])['predictions']
            states = controls['mean_supported'].copy() if arm == 'attribute_hybrid' else controls[arm]
            if arm == 'attribute_hybrid':
                states[..., :2] = native[..., :2]
            fragment['learned_assoc_'+arm] = states
            fragment['ols_'+arm] = states
        original = deepcopy(fragment)
        saved056 = {prefix+arm+'_'+key: value for prefix in ('ols_', 'learned_assoc_')
                    for arm in entry.ORDINARY for key, value in artifacts[arm].items()}
        current = deepcopy(fragment)
        for arm in METHODS:
            current['output_'+arm] = (fragment['learned_assoc_'+arm] if arm in entry.ORDINARY else native.copy())
        # Cached support control differs at each actual first output only here;
        # unused node predictions are not used to fabricate target rows.
        anchors = np.array([0, 2, 4])
        current['output_support_switch'][anchors] = artifacts['support_switch']['group_states']
        old_systems = {}
        for arm in LEARNED_METHODS:
            oldroot = fusion/arm.replace('_', '-')/'system'
            old_systems[arm] = oldroot
            old = dict(original, ols_learned=native.copy(), ols_membership=routes[arm]['membership'])
            old.update({'ols_routing_'+key: value for key, value in routes[arm].items()})
            current.update({'source057_'+arm+'_prediction_learned': native.copy()})
            current['source057_'+arm+'_prediction_mean_supported'] = original['ols_mean_supported']
            current.update({'source057_'+arm+'_routing_'+key: value for key, value in routes[arm].items()})
            current.update({arm+'_routing_'+key: value for key, value in routes[arm].items()})
            self.save(oldroot, 'predictions', old)
            old_system = {key: value for key, value in saved056.items() if key.startswith('ols_')}
            old_system.update({'ols_learned_'+key: value for key, value in artifacts[arm].items()})
            self.save(oldroot, 'system-predictions', old_system)
        self.save(source, 'predictions', current)
        self.save(source, 'system-predictions', {arm+'_'+key: value for arm, a in artifacts.items() for key, value in a.items()})
        self.save(system, 'predictions', original)
        self.save(system, 'system-predictions', saved056)
        row, _ = summarize_scene(dict(current, direct_only_prediction=current['output_direct_only']), artifacts, routes, '2535', 'c1')
        row['source057_replay'] = {arm: dict(passed=True) for arm in LEARNED_METHODS}
        (source/'scenes').mkdir()
        write_json_gzip(source/'scenes/scene-2535.json.gz', row)
        write_json(source/'run.json', dict(status='FAILED', acceptance='REJECTED'))
        manifest = root/'manifest.json'
        write_json(manifest, dict(splits=dict(V_select=['2535'])))
        meta = dict(scene_ids=['2535'], components={'2535': 'c1'}, arm_systems=old_systems,
                    summary=dict(methods=row['methods']), audit=dict(rejected=[]))
        args = SimpleNamespace(source_066=str(source), source_system=str(system), source_fusion=str(fusion),
                               manifest=str(manifest), output=str(root/'067'), cpu_threads=1)
        return args, meta

    @staticmethod
    def save(root, directory, values):
        (root/directory).mkdir(parents=True, exist_ok=True)
        np.savez_compressed(root/directory/'states-2535.npz', **values)

    def test_cache_pipeline_no_model_or_archive_and_source_stays_rejected(self):
        with TemporaryDirectory() as temporary:
            args, meta = self.fixture(Path(temporary))
            source_record = Path(args.source_066)/'run.json'
            original = source_record.read_bytes()
            with patch.object(entry, 'bind_sources', return_value=meta), \
                    patch('torch.load', side_effect=AssertionError('checkpoint forbidden')), \
                    patch('association.evaluation.evidence_fusion_combo.predict_partition', side_effect=AssertionError('forward forbidden')), \
                    patch('tools.experiments.level1_system_evaluate.load_timed_dataset', side_effect=AssertionError('archive forbidden')):
                result = entry.run(args)
            self.assertEqual(result['run']['status'], 'COMPLETED')
            self.assertFalse(result['run']['method_result_accepted'])
            self.assertEqual(source_record.read_bytes(), original)
            self.assertFalse((Path(args.output)/'summary.json').exists())
            self.assertTrue((Path(args.output)/'scene-2535-numeric.json.gz').exists())
            report = (Path(args.output)/'report.md').read_text()
            self.assertIn('REJECTED', report)
            self.assertIn('触及最终有效点', report)
            self.assertIn('A_rule_better_model_worse', report)
            self.assertIn('beneficial_supported_helper_zero_mass_unchanged', report)

    def test_changed_cached_source_fails_and_preserves_failure(self):
        with TemporaryDirectory() as temporary:
            args, meta = self.fixture(Path(temporary))
            values = entry.load_arrays(Path(args.source_066), 'predictions', '2535')
            values['native_states'][0, 0, 2] += 1
            self.save(Path(args.source_066), 'predictions', values)
            with patch.object(entry, 'bind_sources', return_value=meta), self.assertRaisesRegex(ValueError, '066/056 fragment'):
                entry.run(args)
            self.assertEqual(read_json(Path(args.output)/'run.json')['status'], 'FAILED')

    def test_output_cannot_write_under_sources_or_overwrite(self):
        with TemporaryDirectory() as temporary:
            args, _ = self.fixture(Path(temporary))
            args.output = str(Path(args.source_fusion)/'analysis')
            with self.assertRaisesRegex(ValueError, 'separate'):
                entry.run(args)
            args.output = str(Path(temporary)/'exists')
            Path(args.output).mkdir()
            with self.assertRaises(FileExistsError):
                entry.run(args)

    def test_production_binding_rejects_small_population_and_accepted_source(self):
        with TemporaryDirectory() as temporary:
            args, _ = self.fixture(Path(temporary))
            roots = [Path(getattr(args, key)) for key in ('source_066', 'source_system', 'source_fusion')]
            with self.assertRaisesRegex(ValueError, 'fixed disjoint'):
                entry.bind_sources(*roots, dict(dataset_level='Level1', split_unit='scene', splits=dict(V_select=['2535'])))
            manifest = dict(dataset_level='Level1', split_unit='scene', splits=dict(
                T_core=[str(i) for i in range(2100)], V_select=[str(i) for i in range(2100, 2400)],
                T_test=[str(i) for i in range(2400, 3000)]))
            with patch.object(entry, 'read', return_value=dict(status='COMPLETED', acceptance='ACCEPTED')):
                with self.assertRaisesRegex(ValueError, 'completed rejected'):
                    entry.bind_sources(*roots, manifest)


if __name__ == '__main__':
    unittest.main()
