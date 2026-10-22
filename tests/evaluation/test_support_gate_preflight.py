"""Bounded raw Level1 reconstruction must never prepare another split/scene."""
from copy import deepcopy
import json
from pathlib import PurePosixPath
import unittest
from unittest.mock import patch
import zipfile

import numpy as np
import torch

from association.data.archive import DataError, validate_manifest
from association.data.dataset import load_timed_dataset
from association.evaluation.scene import evaluate_full_scene
from association.evaluation.support_gate_preflight import run_preflight, DATA_CONFIG
from tests.fixtures.native_zip import NativeZipFixture


class SupportGatePreflightTests(NativeZipFixture, unittest.TestCase):
    def setUp(self):
        super().setUp()
        with zipfile.ZipFile(self.path) as archive:
            members = [(i.filename.replace('level2', 'level1'), archive.read(i)) for i in archive.infolist()]
        with zipfile.ZipFile(self.path, 'w') as archive:
            for name, payload in members:
                archive.writestr(name, payload)
        splits = dict(T_core=['0', '1'], T_head=[], V_select=['2', '4'], T_test=['3'])
        self.manifest.update(dataset_level='Level1', split_unit='scene', split_origin='fresh_archive',
            splits=splits, caps={k: len(v) for k, v in splits.items()},
            scene_splits={sid: role for role, ids in splits.items() for sid in ids},
            historical_usage=dict(kind='preview_only', previewed_scenes=['0'], prior_measurement_splits=None),
            historical_protection_counts=dict(T_core=1, T_head=0, V_select=0, T_test=0))
        validate_manifest(self.manifest)
        self.source = self.root/'source056'/'system'
        for dirname in ('predictions', 'system-predictions'):
            (self.source/dirname).mkdir(parents=True)
        self.manifest_path = self.root/'manifest.json'
        self.alignment_path = self.root/'alignment.json'
        self.manifest_path.write_text(json.dumps(self.manifest))
        self.alignment_path.write_text(json.dumps(self.alignment))
        (self.source/'run.json').write_text(json.dumps(dict(status='COMPLETED', dataset_level='Level1',
            split='V_select', association_endpoint='final', motion_observation_policy='canonical_all',
            test_opened=False, completed_scenes=2,
            inputs=dict(manifest=str(self.manifest_path), alignment=str(self.alignment_path)))))
        (self.source/'summary.json').write_text(json.dumps(dict(status='COMPLETED', dataset_level='Level1',
            split='V_select', test_opened=False, scene_ids=['2', '4'], association_endpoint='final',
            association_endpoint_step=1000)))
        self.make_cache()

    def make_cache(self):
        scene = load_timed_dataset(self.path, self.manifest, train_splits=['T_core'], data_config=DATA_CONFIG,
            alignment_by_scene=self.alignment, validation_only=True,
            motion_observation_policy='canonical_all')['validation'][0]
        self.scene = scene
        motion = scene['motion']
        groups = [[0, 1], [2], [3]]  # Duplicate representatives are retained, no rematching.
        offsets = np.r_[0, np.cumsum([len(g) for g in groups])]
        flat = np.array([n for g in groups for n in g], np.int64)
        membership = np.empty(len(scene['node_ids']), np.int64)
        for gid, group in enumerate(groups):
            membership[group] = gid
        targets = scene['fusion_targets']
        registry = scene['target_identity_ids']
        truth = np.stack([targets[k]['states'] for k in registry])
        valid = np.stack([targets[k]['valid'] for k in registry])
        rows = [registry.index(k) for k in scene['identities']]
        self.cache = dict(node_ids=np.array(scene['node_ids']), identities=scene['identities'],
            sources=np.array(scene['sources']), target_identity_ids=np.array(registry),
            query_seconds=scene['grid'], native_states=motion['states'], support=motion['support'],
            nearest_seconds=motion['nearest_seconds'], exact_observation=motion['exact_observation'],
            target=truth[rows], target_valid=valid[rows], learned_assoc_group_offsets=offsets,
            learned_assoc_group_node_indices=flat, learned_assoc_membership=membership,
            learned_assoc_anchor=motion['states'].copy())
        artifact = evaluate_full_scene([dict(node_indices=g, states=motion['states'][g[0]]) for g in groups],
            node_identities=scene['identities'], target_identities=registry, target_states=truth,
            target_valid=valid, node_observed=motion['support'], split='V_select',
            grouping_kind='predicted', correspondence_policy='anchor_identity_first',
            node_input_support=dict(support=motion['support'], nearest_seconds=motion['nearest_seconds']),
            return_artifact=True)['artifact']
        self.system = {'learned_assoc_anchor_'+k: v for k, v in artifact.items()}
        self.save_cache()

    def save_cache(self):
        np.savez_compressed(self.source/'predictions/states-2.npz', **self.cache)
        np.savez_compressed(self.source/'system-predictions/states-2.npz', **self.system)

    def run_one(self, name='out', **kwargs):
        args = dict(archive_paths=self.path, manifest=self.manifest, alignment=self.alignment,
            source056=self.source, output=self.root/name, requested_scene_ids=('2',))
        args.update(kwargs)
        return run_preflight(**args)

    def test_only_requested_scene_payloads_no_forward_and_no_manifest_rewrite(self):
        before = deepcopy(self.manifest)
        reads = []
        original = zipfile.ZipFile.read

        def guarded_read(archive, member, *args, **kwargs):
            name = member.filename if isinstance(member, zipfile.ZipInfo) else member
            sid = PurePosixPath(name).stem.rsplit('-', 1)[1]
            self.assertEqual(sid, '2', 'nonrequested scene payload accessed')
            reads.append(name)
            return original(archive, member, *args, **kwargs)

        with patch.object(zipfile.ZipFile, 'read', guarded_read), \
             patch.object(torch, 'load', side_effect=AssertionError('checkpoint forbidden')), \
             patch.object(torch.nn.Module, '__call__', side_effect=AssertionError('forward forbidden')), \
             patch('association.data.dataset.load_timed_dataset', side_effect=AssertionError('full dataset forbidden')):
            result = self.run_one()
        self.assertEqual(result['status'], 'PASS')
        self.assertFalse(result['training_gate_passed'])  # A synthetic subset is not formal068.
        self.assertFalse(result['semantic_physical_plausibility_established'])
        self.assertEqual(self.manifest, before)
        self.assertEqual(len(reads), 3)
        self.assertEqual(result['completed_scenes'], 1)
        with np.load(self.root/'out/scene-2-evidence.npz') as evidence:
            np.testing.assert_array_equal(evidence['complete_truth'], self.system['learned_assoc_anchor_aligned_truth'][:3])
            self.assertEqual(evidence['canonical_times_0'].size, 4)
        self.assertTrue((self.root/'out/scene-2-raw-reference.json.gz').is_file())

    def test_wrong_fields_fail_and_preserve_failed_report(self):
        for key in ('native_states', 'support', 'nearest_seconds', 'target', 'target_valid', 'learned_assoc_membership'):
            original = self.cache[key].copy()
            self.cache[key].flat[0] = (not original.flat[0]) if original.dtype == bool else original.flat[0]+1
            self.save_cache()
            with self.subTest(key=key), self.assertRaises((DataError, ValueError)):
                self.run_one('bad-'+key)
            self.assertEqual(json.loads((self.root/('bad-'+key)/'report.json').read_text())['status'], 'FAILED')
            self.cache[key] = original
        self.save_cache()

    def test_sealed_wrong_level_and_changed_complete_manifest_rejected_before_payload_read(self):
        cases = [dict(requested_scene_ids=('3',)),
                 dict(manifest={**self.manifest, 'dataset_level': 'Level2'}),
                 dict(manifest={**self.manifest, 'components': {}}),
                 dict(manifest={**self.manifest, 'unrelated_metadata_change': True}),
                 dict(alignment={**self.alignment, '2': dict(time_offset_seconds=0., max_gap_seconds=10.)})]
        for index, args in enumerate(cases):
            with self.subTest(args=args), patch.object(zipfile.ZipFile, 'read', side_effect=AssertionError('payload forbidden')):
                with self.assertRaises((DataError, ValueError, KeyError)):
                    self.run_one('refuse'+str(index), **args)

    def test_large_legal_native_speed_is_reported_not_deleted(self):
        with zipfile.ZipFile(self.path) as archive:
            members = [(i.filename, archive.read(i)) for i in archive.infolist()]
        with zipfile.ZipFile(self.path, 'w') as archive:
            for name, payload in members:
                if name.endswith('量测场景/场景-2.csv'):
                    payload = payload.replace(b',5,45', b',80000,45')
                archive.writestr(name, payload)
        self.make_cache()
        result = self.run_one()
        self.assertEqual(result['status'], 'PASS')
        self.assertEqual(result['scene_summaries'][0]['focal_targets'][0]['tracks'][0]['native_speed_max'], 80000.)
        self.assertFalse(result['anomalies_removed'])

    def test_missing_cache_or_mapping_difference_cannot_fall_back(self):
        self.system['learned_assoc_anchor_aligned_group_indices'][0] = -1
        self.save_cache()
        with self.assertRaises(DataError):
            self.run_one('mapping')
        (self.source/'predictions/states-2.npz').unlink()
        with self.assertRaises(FileNotFoundError):
            self.run_one('missing')

    def test_original_source_cannot_be_output_or_overwritten(self):
        with self.assertRaises(DataError):
            self.run_one(output=self.source/'extra')
        self.run_one()
        with self.assertRaises(FileExistsError):
            self.run_one()


if __name__ == '__main__':
    unittest.main()
