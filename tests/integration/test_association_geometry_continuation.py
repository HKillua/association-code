"""130 bounded-memory/import behavior; fixtures are implementation evidence only."""
from copy import deepcopy
import filecmp
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from tests.integration.test_query_contribution_control import cache_fixture
from tools.analysis import association_geometry_continuation as p


def cached_failure(root):
    old_args, meta = cache_fixture(root)
    ids = [str(i) for i in range(1, 8)]
    path = meta['association']/'final-scores.npz'
    with np.load(path, allow_pickle=False) as saved:
        payload = {key: saved[key].copy() for key in saved.files}
    for sid in ids[2:]:
        shutil.copyfile(old_args.source_system/'predictions/states-1.npz', old_args.source_system/'predictions'/f'states-{sid}.npz')
        for key in p.old.frozen.FIELDS:
            payload[sid+'_'+key] = payload['1_'+key].copy()
        private = p.old.frozen.read(old_args.partition_run/'private/scene-1.json.gz')
        p.old.frozen.write(old_args.partition_run/'private'/f'scene-{sid}.json.gz', private)
    payload['scene_ids'] = np.asarray(ids)
    np.savez_compressed(path, **payload)
    meta.update(ids=ids, all_ids=ids, components={sid: 'c' for sid in ids})
    failed = root/p.FAILED_RUN
    (failed/'private').mkdir(parents=True); (failed/'scenes').mkdir()
    with np.load(path, allow_pickle=False) as scores:
        for sid in ids[:p.REUSE_COUNT]:
            with np.load(old_args.source_system/'predictions'/f'states-{sid}.npz', allow_pickle=False) as saved:
                physical = {key: saved[key].copy() for key in p.p.PHYSICAL_KEYS}
                identities = saved['identities'].copy()
            membership = np.asarray(p.old.frozen.read(old_args.partition_run/'private'/f'scene-{sid}.json.gz')['memberships']['WS'])
            groups = p.old.groups_from_membership(membership, len(identities))
            features, audit = p.p.construct_features(physical['native_states'], physical['support'], physical['sources'],
                groups, scores[sid+'_pairs'], scores[sid+'_scores'], p.old.frozen.THRESHOLD)
            evidence = p.p.evaluate_features(features, identities)
            evidence['original'] = features['candidate_is_original']
            np.savez_compressed(failed/'private'/f'features-{sid}.npz', **features)
            np.savez_compressed(failed/'private'/f'evaluation-{sid}.npz', **evidence)
            row = dict(scene_id=sid, component_id='c', node_count=len(identities), feature_audit=audit,
                candidate_population=p.p.summarize(evidence), feature_file_written_before_identity_read=True,
                reference_array_fields_loaded=False)
            p.old.frozen.write(failed/'scenes'/f'scene-{sid}.json.gz', row)
    (failed/'private/features-7.npz').write_bytes(b'partial failed original file preserved')
    launch = Path(str(failed)+'-launch'); launch.mkdir()
    p.old.frozen.write(launch/'launch.json', dict(unit='mtad-fixture128.service'))
    state = dict(status='RUNNING', idea_id=p.p.IDEA, plan_version='1.0', scene_ids=ids,
        completed_scenes=6, last_scene_id='6', source_system=str(old_args.source_system),
        partition_run=str(old_args.partition_run), models_constructed=0, model_forwards=0, optimizer_updates=0,
        test_opened=False, reference_array_fields_loaded=False, partition_changed=False,
        physical_predictions_changed=False, peak_rss_bytes=12345)
    p.old.frozen.write(failed/'run.json', state)
    failure = dict(effective_status='FAILED', cause='systemd_oomd', completed_scenes=6,
        preserved_run_status='RUNNING', launcher_status_is_stale=True, payload_exit_code=None,
        unit='mtad-fixture128.service', systemd=dict(ExecMainStatus='9', Result='signal', ActiveState='failed'))
    p.old.frozen.write(failed/'terminal-synthesis.json', failure)
    args = SimpleNamespace(source_system=old_args.source_system, partition_run=old_args.partition_run,
        failed_run=failed, failure_evidence=None, manifest=old_args.manifest, output=root/'continuation130', cpu_threads=1)
    return args, meta


class GeometryContinuationTests(unittest.TestCase):
    def test_chunk_preserves_full_original_numeric_arrays_and_L1_round_mean(self):
        n = 100
        states = np.zeros((n, 64, 4)); states[..., 0] = np.arange(n)[:, None]*.001
        states[..., 1] = 120.+np.arange(64)[None, :]*.0001+np.arange(n)[:, None]*.0002
        support = np.random.default_rng(20).random((n, 64)) > .2
        pairs = np.stack(np.triu_indices(n, 1), axis=-1)
        args = (states, support, np.where(np.arange(n) % 2, 'a', 'b'), [list(range(n))], pairs,
                np.linspace(-1., 1., len(pairs)), p.old.frozen.THRESHOLD)
        original, original_audit = p.p.construct_features(*args)
        shapes = []
        function = p.p.distance
        def observed(a, b):
            if np.ndim(a) == 3:
                shapes.append(len(a))
            return function(a, b)
        with patch.object(p.p, 'distance', side_effect=observed):
            bounded, bounded_audit = p.p.construct_features(*args, pair_geometry_builder=p.chunked_pair_geometry)
        self.assertEqual(shapes, [4096, len(pairs)-4096])
        for key in original:
            np.testing.assert_array_equal(original[key], bounded[key])
        self.assertEqual(original_audit, bounded_audit)
        pos = np.zeros((2, 1, 2)); pos[:, 0, 1] = [-179., 179.]
        result, _ = p.p.position_mean(pos, np.ones((2, 1), bool))
        self.assertEqual(result[0, 1], 180.)  # Preserve128, do not wrap to -180.
        pos[:, 0, 1] = [-45., 135.-np.rad2deg(1.5e-12)]
        result, _ = p.p.position_mean(pos, np.ones((2, 1), bool))
        # This resultant passes the original L1 cutoff but would fail hypot.
        self.assertGreater(result[0, 1], 40.)
        self.assertLess(result[0, 1], 50.)
        with self.assertRaisesRegex(ValueError, 'complete finite same-query'):
            p.p.construct_features(*args, pair_geometry_builder=lambda *_: np.zeros((1, 1)))

    def test_imports_six_exact_and_computes_only_remaining_preserving_original_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            args, meta = cached_failure(Path(temporary).resolve())
            original_state = (args.failed_run/'run.json').read_bytes()
            partial = (args.failed_run/'private/features-7.npz').read_bytes()
            with patch.object(p.old, 'metadata', return_value=meta), \
                 patch.object(p.p, 'construct_features', wraps=p.p.construct_features) as constructor:
                result = p.run(args)
            self.assertEqual(constructor.call_count, 1)
            self.assertEqual((result['imported_feature_scenes'], result['newly_constructed_feature_scenes']), (6, 1))
            self.assertEqual(result['scene_ids'], meta['ids'])
            self.assertEqual(result['imported_complete128']['scene_ids'], meta['ids'][:6])
            self.assertEqual(result['newly_computed130']['scene_ids'], ['7'])
            for sid in meta['ids'][:6]:
                for relative in (f'private/features-{sid}.npz', f'private/evaluation-{sid}.npz', f'scenes/scene-{sid}.json.gz'):
                    self.assertTrue(filecmp.cmp(args.failed_run/relative, args.output/relative, shallow=False))
            self.assertEqual((args.failed_run/'run.json').read_bytes(), original_state)
            self.assertEqual((args.failed_run/'private/features-7.npz').read_bytes(), partial)
            self.assertFalse(result['imported_features_recomputed'])
            state = p.old.frozen.read(args.output/'run.json')
            self.assertIsNone(state['original128_true_peak_rss_bytes'])
            self.assertEqual(state['original128_last_reported_peak_rss_bytes'], 12345)
            self.assertEqual((state['status'], state['exit_code'], state['completed_scenes']), ('COMPLETED', 0, 7))
            with self.assertRaises(FileExistsError):
                p.run(args)

    def test_import_corruption_or_active_original_are_not_silently_accepted(self):
        for fault in ('label', 'active'):
            with self.subTest(fault=fault), tempfile.TemporaryDirectory() as temporary:
                args, meta = cached_failure(Path(temporary).resolve())
                if fault == 'label':
                    path = args.failed_run/'private/evaluation-6.npz'
                    with np.load(path, allow_pickle=False) as saved:
                        payload = {key: saved[key].copy() for key in saved.files}
                    payload['pure'][0] = ~payload['pure'][0]
                    np.savez_compressed(path, **payload)
                else:
                    path = args.failed_run/'terminal-synthesis.json'
                    evidence = p.old.frozen.read(path)
                    evidence['systemd']['ActiveState'] = 'active'
                    p.old.frozen.write(path, evidence)
                with patch.object(p.old, 'metadata', return_value=meta):
                    with self.assertRaises(ValueError):
                        p.run(args)
                state = p.old.frozen.read(args.output/'run.json')
                self.assertEqual(state['status'], 'FAILED')
                self.assertEqual(state['completed_scenes'], 5 if fault == 'label' else 0)
                self.assertFalse((args.output/'summary.json').exists())


if __name__ == '__main__':
    unittest.main()
