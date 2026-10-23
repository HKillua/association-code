"""075 saved-call audit: enumeration, exact reconciliation and corruption refusal."""
from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from association.evaluation.tsactfer_training_diagnostics import (
    analyze_training, negative_inventory, _schedule,
)

ROOT = Path(__file__).resolve().parents[2]


def training_log_fixture(folder, updates=4):
    folder = Path(folder)
    folder.mkdir()
    config = json.loads((ROOT/'configs/tsactfer-level1.json').read_text())
    config['model']['length'] = 8
    config['data']['max_points'] = 8
    config['training'].update(updates=updates, warmup_updates=1, pair_batch_size=4)
    ids = ['train-a', 'train-b']
    exposure = dict(successful_optimizer_calls=0, nonzero_lr_updates=0,
        positive_pair_exposures=0, track_exposures=0, token_exposures=0,
        task_calls={'ITSA': 0, 'MSTSA': 0}, scenes={}, identities={}, pairs={},
        cropped_pairs=0, fallback_pairs=0)
    rows = []
    for step in range(1, updates+1):
        task = 'ITSA' if step % 2 else 'MSTSA'
        samples = []
        for index in range(4):
            sid = ids[index//2]
            samples.append(dict(scene_id=sid, identity='ship-'+str(index),
                left_node=f'{sid}-{index}-left', right_node=f'{sid}-{index}-right',
                left_index=2*index, right_index=2*index+1,
                crop=dict(start=10., stop=20., fraction=.75,
                          fallback=task == 'MSTSA' and index == 3)))
        norm = (2., .5, 3., 1.)[(step-1) % 4]
        lr = _schedule(step, config['training'])
        augmentation = dict(pair_count=4, cropped_pairs=2,
            fallback_pairs=int(task == 'MSTSA'), input_points=32, output_points=28)
        row = dict(step=step, task=task, loss=4./step, learning_rate=lr,
            gradient_norm=norm, gradient_clip_scale=min(1., 1./(norm+1e-6)),
            seconds=.25, token_count=28, augmentation=augmentation, batch=samples)
        rows.append(row)
        for sample in samples:
            keys = dict(scenes=sample['scene_id'], identities=sample['identity'],
                pairs=json.dumps([sample['scene_id'], sample['left_index'], sample['right_index']], separators=(',', ':')))
            for field, key in keys.items():
                exposure[field][key] = exposure[field].get(key, 0)+1
        exposure['successful_optimizer_calls'] += 1
        exposure['nonzero_lr_updates'] += int(lr > 0)
        exposure['task_calls'][task] += 1
        exposure['positive_pair_exposures'] += 4
        exposure['track_exposures'] += 8
        exposure['token_exposures'] += 28
        exposure['cropped_pairs'] += 2
        exposure['fallback_pairs'] += augmentation['fallback_pairs']
    exposure.update(unique_scenes=2, unique_identities=4, unique_pairs=4,
        repeated_pair_exposures=4*updates-4)
    run = dict(scope=config['scope'], status='COMPLETED', configuration=config,
        optimizer_updates=updates, final_checkpoint='final.pt', training_scene_ids=ids,
        reference_read=False, test_opened=False,
        normalizer=dict(training_scene_ids=ids, training_splits=['T_core'],
            native_max_points=8, feature_order=['lon', 'lat'], observed_point_count=64,
            mean=[120., 30.], std=[1., 2.]), exposure=exposure,
        gradient_clipping={'clipped_updates': updates//2})
    save_fixture(folder, run, rows)
    return config, ids, run, rows


def save_fixture(folder, run, rows):
    (folder/'run.json').write_text(json.dumps(run))
    (folder/'training.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))


class TrainingDiagnosticsTests(unittest.TestCase):
    def test_negative_inventory_matches_explicit_ordered_infonce_enumeration(self):
        for scenes in (['a', 'a', 'b'], ['a']*4, ['a', 'b', 'c', 'd']):
            b = len(scenes)
            tracks = [(identity, scenes[identity]) for identity in range(b)]*2
            negative = [(left, right) for left in range(2*b) for right in range(2*b)
                        if tracks[left][0] != tracks[right][0]]
            same = sum(tracks[left][1] == tracks[right][1] for left, right in negative)
            actual = negative_inventory(scenes)
            self.assertEqual(actual['ordered_negative_comparisons'], len(negative))
            self.assertEqual(actual['same_scene_ordered_negatives'], same)
            self.assertEqual(actual['cross_scene_ordered_negatives'], len(negative)-same)
        self.assertEqual(negative_inventory(['a', 'a', 'b'])['same_scene_ordered_negatives'], 8)

    def test_exact_exposure_and_head_windows_read_only_two_files_once(self):
        with TemporaryDirectory() as temp:
            folder = Path(temp)/'fit'
            config, ids, run, _ = training_log_fixture(folder, updates=204)
            opened = []
            original_open = Path.open
            def restricted_open(path, *args, **kwargs):
                self.assertIn(path, (folder/'run.json', folder/'training.jsonl'))
                opened.append(path.name)
                return original_open(path, *args, **kwargs)
            with patch.object(Path, 'open', restricted_open):
                result = analyze_training(folder, config, ids)
            self.assertEqual(opened, ['run.json', 'training.jsonl'])
            self.assertEqual(result['status'], 'VERIFIED')
            self.assertEqual(result['actual_exposure']['nonzero_lr_updates'], 203)
            self.assertEqual(result['all_updates']['clipped_call_fraction'], .5)
            for task, first, last in (('ITSA', 1, 199), ('MSTSA', 2, 200)):
                windows = result['heads'][task]['windows']
                self.assertEqual([w['successful_optimizer_calls'] for w in windows], [100, 2])
                self.assertEqual((windows[0]['first_global_step'], windows[0]['last_global_step']), (first, last))
                self.assertEqual(sum(w['positive_pair_exposures'] for w in windows), 408)
            neg = result['all_updates']['negative_inventory']
            self.assertEqual(neg['ordered_negative_comparisons'], 204*48)
            self.assertEqual(neg['same_scene_ordered_negatives'], 204*16)
            self.assertEqual(result['all_updates']['crop']['token_retention_fraction'], 28/32)
            self.assertEqual(run['exposure']['scenes']['train-a'], 408)

    def test_rejects_wrong_step_task_rate_nonfinite_or_duplicate_identity(self):
        changes = (
            ('step', lambda rows: rows[0].update(step=2)),
            ('task', lambda rows: rows[0].update(task='MSTSA')),
            ('learning', lambda rows: rows[0].update(learning_rate=.2)),
            ('loss', lambda rows: rows[0].update(loss=float('nan'))),
            ('gradient', lambda rows: rows[0].update(gradient_norm=float('inf'))),
            ('clip', lambda rows: rows[0].update(gradient_clip_scale=.01)),
            ('distinct', lambda rows: rows[0]['batch'][1].update(identity=rows[0]['batch'][0]['identity'])),
            ('missing', lambda rows: rows.pop()),
        )
        for name, change in changes:
            with self.subTest(name=name), TemporaryDirectory() as temp:
                folder = Path(temp)/'fit'; config, ids, run, rows = training_log_fixture(folder)
                change(rows); save_fixture(folder, run, rows)
                with self.assertRaises(ValueError):
                    analyze_training(folder, config, ids)

    def test_rejects_failed_or_unbound_source_and_inexact_exposure(self):
        changes = (
            lambda run: run.update(status='FAILED'),
            lambda run: run.update(optimizer_updates=3),
            lambda run: run.update(test_opened=True),
            lambda run: run['configuration'].update(seed=0),
            lambda run: run['normalizer'].update(training_scene_ids=['validation']),
            lambda run: run['normalizer'].update(std=[1., 0.]),
            lambda run: run['exposure']['scenes'].update({'train-a': 9}),
            lambda run: run['exposure'].update(token_exposures=113),
            lambda run: run['exposure'].update(task_calls={'ITSA': 1, 'MSTSA': 3}),
            lambda run: run.update(gradient_clipping={'clipped_updates': 3}),
        )
        for index, change in enumerate(changes):
            with self.subTest(index=index), TemporaryDirectory() as temp:
                folder = Path(temp)/'fit'; config, ids, run, rows = training_log_fixture(folder)
                expected = deepcopy(config)
                change(run); save_fixture(folder, run, rows)
                with self.assertRaises(ValueError):
                    analyze_training(folder, expected, ids)

    def test_rejects_crop_token_scene_selfpair_corruption(self):
        changes = (
            lambda rows: rows[0]['batch'][0].update(scene_id='validation'),
            lambda rows: rows[0]['batch'][0].update(right_index=0),
            lambda rows: rows[0]['batch'][0]['crop'].update(fraction=.1),
            lambda rows: rows[0]['batch'][0]['crop'].update(stop=10.),
            lambda rows: rows[0]['batch'][0]['crop'].update(fallback=True),
            lambda rows: rows[0]['augmentation'].update(output_points=27),
            lambda rows: rows[0]['augmentation'].update(input_points=27),
            lambda rows: rows[0]['augmentation'].update(cropped_pairs=0),
            lambda rows: rows[1]['augmentation'].update(fallback_pairs=0),
        )
        for index, change in enumerate(changes):
            with self.subTest(index=index), TemporaryDirectory() as temp:
                folder = Path(temp)/'fit'; config, ids, run, rows = training_log_fixture(folder)
                change(rows); save_fixture(folder, run, rows)
                with self.assertRaises(ValueError):
                    analyze_training(folder, config, ids)

    def test_real_tiny_saved_trainer_log_without_weights_access(self):
        import torch
        from baselines.tsactfer_level1 import train_native
        from tests.test_tsactfer_level1 import tiny_config, training_fixture
        torch.set_num_threads(1)
        config = tiny_config(); records, normalizer = training_fixture(config)
        with TemporaryDirectory() as temp:
            folder = Path(temp)/'fit'
            run = train_native(records, normalizer, config, folder)
            (folder/'final.pt').unlink(); (folder/'step-0.pt').unlink()
            result = analyze_training(folder, config, normalizer['training_scene_ids'])
            self.assertEqual(result['actual_exposure']['token_exposures'], run['exposure']['token_exposures'])
            self.assertEqual(result['all_updates']['successful_optimizer_calls'], 4)
            self.assertFalse(result['verification']['model_weights_read'])


if __name__ == '__main__':
    unittest.main()
