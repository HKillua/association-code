"""089 real tiny successful logs, T-only ZIP replay and denominator rejection."""
from argparse import Namespace
from contextlib import ExitStack
from copy import deepcopy
import json
from pathlib import Path
import unittest
from unittest.mock import patch
import zipfile

import numpy as np
import torch

from association.data.archive import MTADArchive
from association.data.scene_manifest import build_fresh_scene_manifest
from association.evaluation import helper_prior_replay as audit
from association.training.sequence_position import train_sequence_position
from association.utils.runtime import write_json
from tests.fixtures.native_zip import make_native_zip_fixture
from tests.test_sequence_position import tiny_configuration
from tools.analysis import helper_prior_replay as entry


class HelperPriorReplayTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.fixture = f = make_native_zip_fixture(self)
        f.write_zip(missing_reference=('0', 'b'))
        with zipfile.ZipFile(f.path) as source:
            content = {i.filename: source.read(i) for i in source.infolist()}
        content['数据集level2/关联表/关联结果-3.csv'] = content[
            '数据集level2/关联表/关联结果-1.csv'].replace(b'target1', b'target3')
        with zipfile.ZipFile(f.path, 'w') as target:
            for name, payload in content.items(): target.writestr(name.replace('数据集level2/', 'Level1/'), payload)
        with MTADArchive(f.path) as source:
            f.manifest = build_fresh_scene_manifest(source, seed=46,
                caps=dict(T_core=2, T_head=0, V_select=2, T_test=1))
        self.config = c = tiny_configuration('transformer')
        c['training'].update(updates=6, batch_size=2)
        f.data = c['data']
        data = f.load(motion_observation_policy='canonical_all')
        self.control, self.drop = f.root/'linear', f.root/'drop'
        self.control.mkdir(); self.drop.mkdir()
        train_sequence_position(data['train'], c, self.control/'fit')
        self.dropout_config = d = deepcopy(c)
        d.update(scope=audit.SCOPE, augmentation=dict(kind=audit.KIND, probability=.5,
                                                     seed=c['training']['seed']+83000))
        train_sequence_position(data['train'], d, self.drop/'fit', paired_reference_fit=self.control/'fit')
        self.manifest_path, self.alignment_path = f.root/'manifest.json', f.root/'alignment.json'
        write_json(self.manifest_path, f.manifest); write_json(self.alignment_path, f.alignment)
        # Wrapper provenance is synthetic; actual successful fit logs above are
        # produced by the real trainer. No fake V prediction is consumed by089.
        for root, cfg, profile in ((self.control, c, 'extended4000'), (self.drop, d, 'helper-dropout4000')):
            cfgpath = root/'config.json'; write_json(cfgpath, cfg)
            run = dict(status='COMPLETED', configuration=cfg, optimizer_updates=6, test_opened=False,
                completed_endpoints=['initial', 'final'], retries=0, source_system=str(f.root/'056/system'),
                profile=profile, validation_selection=False, association_recomputed=False, groups_recomputed=False,
                threshold_reselected=False)
            args = dict(output=str(root), source_system=run['source_system'], profile=profile,
                manifest=str(self.manifest_path), alignment=str(self.alignment_path), config=str(cfgpath),
                archive=[str(f.path)])
            if root == self.drop: run['paired_control'] = str(self.control)
            write_json(root/'run.json', run)
            write_json(root/'execution.json', dict(arguments=args, configuration=cfg, test_opened=False))
        write_json(self.drop/'preparation.json', data['report'])
        self.args = Namespace(dropout_run=str(self.drop), linear_run=str(self.control),
            manifest=str(self.manifest_path), alignment=str(self.alignment_path), archive=[str(f.path)],
            output=str(f.root/'audit'), cpu_threads=4, time_policy='advisory')

    def invoke(self):
        return entry.run(self.args, expected_config=self.dropout_config,
            expected_manifest=self.fixture.manifest, expected_alignment=self.fixture.alignment)

    def test_real_training_logs_and_t_only_native_rebuild_no_model_checkpoint_or_v(self):
        read = zipfile.ZipFile.read
        seen = []
        def guarded(source, member, *args, **kwargs):
            name = member.filename if isinstance(member, zipfile.ZipInfo) else str(member)
            self.assertTrue(name.endswith('-0.csv') or name.endswith('-1.csv'), 'only T payloads permitted')
            seen.append(name)
            return read(source, member, *args, **kwargs)
        with ExitStack() as stack:
            stack.enter_context(patch.object(zipfile.ZipFile, 'read', guarded))
            stack.enter_context(patch.object(torch, 'load', side_effect=AssertionError('no checkpoint')))
            stack.enter_context(patch.object(torch.nn.Module, '__init__', side_effect=AssertionError('no model')))
            stack.enter_context(patch.object(torch.optim, 'Adam', side_effect=AssertionError('no optimizer')))
            stack.enter_context(patch('association.data.dataset.load_timed_dataset', side_effect=AssertionError('no mixed T/V loader')))
            stack.enter_context(patch('association.training.evidence_fusion.group_predicted_tracks', side_effect=AssertionError('no groups')))
            result = self.invoke()
        self.assertEqual(result['status'], 'COMPLETED')
        self.assertEqual(result['successful_sample_count'], 12)
        self.assertEqual(result['training_scene_count'], 2)
        self.assertTrue(seen)
        self.assertGreater(result['zero_supervision_sample_count'], 0)
        metrics = result['global_metrics']
        self.assertAlmostEqual(metrics['coefficient_mass'],
            1-result['zero_supervision_sample_count']/12)
        self.assertAlmostEqual(metrics['signed_absolute_error_change_m'],
            metrics['harm_sum_m']-metrics['improvement_sum_m'])
        self.assertEqual(sum(r['point_count'] for r in result['cross_table']), metrics['point_count'])
        self.assertEqual(len(result['cross_table']), 81)
        self.assertEqual(metrics['all_query_count'], 12*self.config['data']['length'])
        self.assertGreater(metrics['all_query_count'], metrics['point_count'])
        self.assertEqual(sum(r['all_query_count'] for r in result['cross_table']), metrics['all_query_count'])
        self.assertFalse(json.loads((Path(self.args.output)/'run.json').read_text())['test_opened'])
        with self.assertRaises(FileExistsError): self.invoke()

    def test_source_truncation_rng_and_input_membership_mutations_fail_and_preserve(self):
        log = self.drop/'fit/training.jsonl'; original = log.read_text()
        for label, mutate in (
                ('truncated', lambda rows: rows[:-1]),
                ('rng', lambda rows: self.mutate_row(rows, 'augmentation_rng_before', {})),
                ('identity_count', lambda rows: self.mutate_audit(rows))):
            with self.subTest(label=label):
                values = [json.loads(v) for v in original.splitlines()]
                values = mutate(values)
                log.write_text(''.join(json.dumps(v)+'\n' for v in values))
                self.args.output = str(self.fixture.root/label)
                with self.assertRaises(ValueError): self.invoke()
                state = json.loads((Path(self.args.output)/'run.json').read_text())
                self.assertEqual(state['status'], 'FAILED')
                self.assertFalse(state['checkpoints_opened'])
                self.assertEqual(state['model_forwards'], 0)
                log.write_text(original)

    @staticmethod
    def mutate_row(values, key, value):
        values[0][key] = value
        return values

    @staticmethod
    def mutate_audit(values):
        values[0]['augmentation_batch'][0]['retained_identity']['wrong_helpers'] += 1
        return values

    def test_original_frozen_config_alignment_and_preparation_must_match(self):
        value = json.loads(self.alignment_path.read_text())
        value['0']['max_gap_seconds'] += 1
        write_json(self.alignment_path, value)
        with self.assertRaisesRegex(ValueError, 'frozen Level1 manifest/alignment'): self.invoke()
        write_json(self.alignment_path, self.fixture.alignment)
        preparation = json.loads((self.drop/'preparation.json').read_text())
        preparation['scenes'][0]['native_point_count'] += 1
        write_json(self.drop/'preparation.json', preparation)
        self.args.output = str(self.fixture.root/'preparation-tamper')
        with self.assertRaisesRegex(ValueError, 'rebuilt T preparation differs'): self.invoke()

    def test_archive_must_match_both_original_ordered_resolved_lists(self):
        alternate = self.fixture.root/'other.zip'
        alternate.write_bytes(self.fixture.path.read_bytes())
        cases = ('request_changed', 'control_changed', 'dropout_changed', 'missing', 'reordered')
        originals = {root: (root/'execution.json').read_text() for root in (self.control, self.drop)}
        for label in cases:
            with self.subTest(label=label):
                for root, text in originals.items(): (root/'execution.json').write_text(text)
                self.args.archive = [str(self.fixture.path)]
                if label == 'request_changed':
                    self.args.archive = [str(alternate)]
                elif label == 'reordered':
                    self.args.archive = [str(self.fixture.path), str(alternate)]
                    for root in (self.control, self.drop):
                        value = json.loads(originals[root])
                        value['arguments']['archive'] = list(reversed(self.args.archive))
                        write_json(root/'execution.json', value)
                else:
                    root = self.control if label == 'control_changed' else self.drop
                    value = json.loads(originals[root])
                    value['arguments']['archive'] = None if label == 'missing' else [str(alternate)]
                    write_json(root/'execution.json', value)
                self.args.output = str(self.fixture.root/label)
                with patch.object(zipfile.ZipFile, 'read', side_effect=AssertionError('reject before reading raw data')):
                    with self.assertRaisesRegex(ValueError, 'archive'): self.invoke()
                self.assertEqual(json.loads((Path(self.args.output)/'run.json').read_text())['status'], 'FAILED')
        # Filesystem aliases to the original file are accepted, preserving order.
        for root, text in originals.items(): (root/'execution.json').write_text(text)
        alias = self.fixture.root/'original-link.zip'; alias.symlink_to(self.fixture.path)
        self.assertEqual(audit.archive_paths([str(alias)]), audit.archive_paths([str(self.fixture.path)]))

    def test_manual_coefficients_keep_zero_samples_and_do_not_equal_point_frequency(self):
        # Three sample slots: one valid query, three valid queries, no valid
        # queries. Both supervised anchors have identical mean error.
        result = audit.Totals()
        result.add(np.array([10.]), np.array([13.]), np.array([3.]), 1/(3*1))
        result.add(np.array([4., 10., 16.]), np.array([2., 10., 18.]), np.array([2., 0., 2.]), 1/(3*3))
        # Geometry also includes unsupervised queries, without creating any
        # supervised point or changing the original zero-sample denominator.
        result.add_geometry(np.array([3., 2., 0., 2., 9., 8.]))
        row = result.describe(3)
        self.assertEqual(row['point_count'], 4)
        self.assertEqual(row['all_query_count'], 6)
        self.assertEqual(row['all_query_displacement_sum_m'], 24.)
        self.assertAlmostEqual(row['coefficient_mass'], 2/3)
        self.assertAlmostEqual(row['sample_equal_linear_loss_full'], 20/3/1000)
        self.assertAlmostEqual(row['sample_equal_linear_loss_dropped'], 23/3/1000)
        self.assertAlmostEqual(row['harm_sum_m']-row['improvement_sum_m'], 3.)
        self.assertNotEqual(row['point_mae_full_m'], row['full_coefficient_m'])


if __name__ == '__main__': unittest.main()
