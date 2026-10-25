"""090 real source-fit + three added fits + full CPU cache chain and invariants."""
from argparse import Namespace
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import torch

from association.models.frozen_position_adapter import FrozenPositionAdapter, initialize_from_teacher, trainable_parameters
from association.models.sequence_position import SequencePosition, sequence_inputs, sequence_objectives
from association.training.frozen_position_adapter import load_checkpoint
from association.utils.runtime import read_json, write_json
from tests.training.test_sequence_position import motion_fixture
from tests.fixtures.cache import support_gate_cache_fixture
from tests.fixtures.learning import tiny_train
from tools.experiments import frozen_position_adapter_fit as entry
from tools.analysis import compare_frozen_position_adapter as compare


class FrozenPositionAdapterTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1); torch.manual_seed(20261023)

    def test_teacher_origin_mask_denominator_gradients_and_same_branch(self):
        teacher = SequencePosition(backbone='transformer').eval()
        with torch.no_grad(): teacher.output.bias.copy_(torch.tensor([.02, -.03]))
        source = {k: v.clone() for k, v in teacher.state_dict().items()}
        a, b = [FrozenPositionAdapter(mode=mode) for mode in ('all_queries', 'anchor_unsupported')]
        for model in (a, b): initialize_from_teacher(model, source, branch_seed=20351023)
        self.assertTrue(all(torch.equal(v, b.state_dict()[k]) for k, v in a.state_dict().items()))
        self.assertEqual(sum(p.numel() for p in a.parameters()), 109732)
        self.assertEqual(sum(p.numel() for p in trainable_parameters(a)), 4578)
        motion = motion_fixture(); args = sequence_inputs(motion, [0, 1, 2])
        support = torch.from_numpy(motion['support'][0]); baseline = teacher(*args)['states'].detach()
        self.assertFalse(torch.equal(baseline, args[0]))
        for model in (a, b): self.assertTrue(torch.equal(model(*args, support)['states'], baseline))
        target = baseline.clone(); target[:, 0] += .01
        valid = torch.ones_like(target, dtype=torch.bool); valid[0, :2] = False
        optimizer = torch.optim.Adam(trainable_parameters(b), lr=.001)
        first_grad = None
        for step in range(5):
            b.train(); optimizer.zero_grad(set_to_none=True); result = b(*args, support)
            losses, diag = sequence_objectives(result, target, valid)
            self.assertEqual(diag['valid_position_points'], len(target)-1)
            losses['linear'].backward()
            self.assertTrue(all(p.grad is None for p in b.teacher.parameters()))
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in b.adapter.parameters()))
            if step == 0: first_grad = bool(torch.count_nonzero(b.adapter[0].weight.grad))
            torch.nn.utils.clip_grad_norm_(trainable_parameters(b), 1.); optimizer.step()
        self.assertFalse(first_grad)
        self.assertTrue(any(bool(torch.count_nonzero(p.grad)) for p in b.adapter[0].parameters()))
        self.assertTrue(all(torch.equal(v, source[k]) for k, v in b.teacher.state_dict().items()))
        final = b(*args, support)['states'].detach()
        self.assertTrue(torch.equal(final[support], baseline[support]))
        self.assertFalse(torch.equal(final[~support], baseline[~support]))
        self.assertFalse(b.teacher.training)
        with self.assertRaises(ValueError): b(*args, ~support)

    def test_production_contract(self):
        manifest = read_json(entry.ROOT/'configs/data/mtad-level1-scene3000.json')
        alignment = read_json(entry.ROOT/'configs/data/mtad-level1-alignment.json')
        for filename in entry.CONFIGS.values():
            config = read_json(entry.ROOT/'configs'/filename); entry.production_contract(config, manifest, alignment)
            config['training']['updates'] += 1
            with self.assertRaises(ValueError): entry.production_contract(config, manifest, alignment)

    def test_fully_supported_and_unsupervised_batches_preserve_teacher_and_denominator(self):
        teacher = SequencePosition(backbone='transformer').eval()
        model = FrozenPositionAdapter(mode='anchor_unsupported')
        initialize_from_teacher(model, teacher.state_dict(), branch_seed=20351023)
        motion = motion_fixture(); args = list(sequence_inputs(motion, [0, 1, 2]))
        support = torch.ones(len(args[0]), dtype=torch.bool)
        args[1][:, 57] = 1.
        with torch.no_grad(): model.adapter[2].bias.fill_(.02)
        expected = teacher(*args)['states'].detach()
        for has_reference in (True, False):
            model.train(); model.zero_grad(set_to_none=True)
            result = model(*args, support)
            target = expected.clone(); target[:, 0] += .01
            valid = torch.full_like(target, has_reference, dtype=torch.bool)
            losses, diagnostics = sequence_objectives(result, target, valid)
            self.assertEqual(diagnostics['valid_position_points'], len(target) if has_reference else 0)
            self.assertTrue(torch.equal(result['states'], expected))
            self.assertFalse(result['update_allowed'].any())
            losses['linear'].backward()
            self.assertTrue(all(p.grad is None for p in model.teacher.parameters()))
            self.assertTrue(all(p.grad is not None and torch.count_nonzero(p.grad) == 0
                                for p in model.adapter.parameters()))
            self.assertEqual(float(losses['linear'].detach()) > 0, has_reference)

    def test_successful_route_audit_rejects_mask_and_summary_misreporting(self):
        config = dict(model=dict(mode='anchor_unsupported'), data=dict(length=4))
        sample = dict(valid_position_points=3, eligible_query_points=2, eligible_valid_position_points=1)
        route = dict(all_queries=4, anchor_supported_queries=2, valid_queries=3, valid_anchor_supported_queries=2,
                     increment_allowed_queries=2, valid_increment_allowed_queries=1)
        rows = [dict(batch=[sample], route_batch=[route])]
        run = dict(configuration=config, route_exposure=route.copy())
        exposure, observed = compare.verify_route_exposure(run, rows)
        self.assertEqual(exposure, route)
        self.assertEqual(observed[0][0]['valid_anchor_supported_queries'], 2)
        changed = deepcopy(run); changed['route_exposure']['valid_queries'] += 1
        with self.assertRaisesRegex(ValueError, 'recorded route exposure'):
            compare.verify_route_exposure(changed, rows)
        changed = deepcopy(rows); changed[0]['route_batch'][0]['increment_allowed_queries'] = 4
        with self.assertRaisesRegex(ValueError, 'correction scope'):
            compare.verify_route_exposure(run, changed)
        changed = deepcopy(rows); changed[0]['route_batch'][0]['valid_anchor_supported_queries'] = 0
        with self.assertRaisesRegex(ValueError, 'intersections are impossible'):
            compare.verify_route_exposure(run, changed)
        changed = deepcopy(rows); changed[0]['route_batch'] = []
        with self.assertRaisesRegex(ValueError, 'missing samples'):
            compare.verify_route_exposure(run, changed)

    def test_real_source_three_fits_save_load_complete_endpoints_and_cached_table(self):
        from tools.experiments import sequence_position_fit as original
        with TemporaryDirectory() as directory:
            root = Path(directory); source, metadata, validation = support_gate_cache_fixture(root)
            train = tiny_train()
            manifest = dict(splits={'T_core': [str(s['scene_id']) for s in train], 'V_select': ['3035']}, components={'3035': 'shared'})
            base = read_json(entry.ROOT/'configs/sequence-position-transformer-extended-level1.json')
            base['training'].update(updates=4, batch_size=2)
            for name, value in (('config', base), ('manifest', manifest), ('alignment', {})): write_json(root/(name+'.json'), value)
            args = Namespace(source_system=str(source), output=str(root/'linear'), paired_control=None, archive=['unread.zip'],
                manifest=str(root/'manifest.json'), alignment=str(root/'alignment.json'), config=str(root/'config.json'),
                device='cpu', cpu_threads=4, time_policy='advisory', profile='extended4000')
            dataset = lambda *a, **k: dict(train=train, validation=validation, report={'fixture': True})
            def meta(path, arm=None):
                path = Path(path)
                return dict(path=path, method='learned_sequence', source_system=str(source.resolve()), scene_ids=['3035'],
                    components={'3035': 'shared'}, summary=read_json(path/'final/summary.json'))
            with patch('association.evaluation.sequence_position.frozen_motion', side_effect=lambda raw, *a: raw):
                with patch.object(original, 'production_contract'), patch.object(original, '_bind_source'), \
                        patch.object(original, '_load_metadata', return_value=metadata), patch.object(original, 'load_timed_dataset', side_effect=dataset):
                    original.run(args)
                args.paired_control = str(root/'linear')
                for mode, filename in entry.CONFIGS.items():
                    config = read_json(entry.ROOT/'configs'/filename); config['training'].update(updates=4, batch_size=2)
                    write_json(root/'config.json', config); args.output = str(root/mode)
                    with patch.object(entry, 'production_contract'), patch.object(entry, 'frozen_configuration', side_effect=lambda c: c), \
                            patch.object(entry, '_bind_source'), patch.object(entry, '_load_metadata', return_value=metadata), \
                            patch.object(entry, 'load_timed_dataset', side_effect=dataset), patch.object(compare, '_metadata', side_effect=meta), \
                            patch.object(compare, 'declared_manifest', return_value=manifest):
                        result = entry.run(args)
                    self.assertEqual(result['status'], 'COMPLETED')
                    model, saved = load_checkpoint(root/mode/'fit/final.pt')
                    self.assertEqual(saved['step'], 4)
                    self.assertEqual(result['trainable_parameter_count'], 105154 if mode == 'full_finetune' else 4578)
            params = Namespace(linear_run=str(root/'linear'), all_run=str(root/'all_queries'),
                missing_run=str(root/'anchor_unsupported'), finetune_run=str(root/'full_finetune'), output=str(root/'combined'), cpu_threads=4)
            with patch.object(entry, 'frozen_configuration', side_effect=lambda c: c), patch.object(compare, '_metadata', side_effect=meta), \
                    patch.object(compare, 'declared_manifest', return_value=manifest):
                summary = compare.run(params)
                arms = {'linear': meta(root/'linear')}
                for mode in entry.CONFIGS:
                    name, arm = compare.adapter_metadata(root/mode); arms[name] = arm
            self.assertTrue(summary['pairing']['ab_initial_parameters_exact'])
            self.assertTrue(all(arm['successful_route_exposure_recomputed'] for arm in summary['pairing']['arms'].values()))
            self.assertTrue(summary['protected_inside_teacher_exact'])
            self.assertEqual(summary['position_population']['covered_target_points'], 127)
            self.assertIn('posthoc_mask_A', summary['methods'])
            self.assertEqual(summary['model_forwards'], 0)
            row, data = compare.scene_pair(arms, '3035')
            inside = data['region_anchor_inside_support'] & np.isfinite(data['linear_position_error'])
            outside = ~data['region_anchor_inside_support'] & np.isfinite(data['linear_position_error'])
            np.testing.assert_array_equal(data['posthoc_mask_A_position_error'][inside], data['linear_position_error'][inside])
            np.testing.assert_array_equal(data['posthoc_mask_A_position_error'][outside], data['adapter_all_position_error'][outside])
            values = data['posthoc_mask_A_position_error'][np.isfinite(data['linear_position_error'])]
            self.assertAlmostEqual(summary['position_weightings']['all']['posthoc_mask_A']['point_equal']['p99'], float(np.quantile(values, .99)), places=8)
            self.assertFalse(summary['pairing']['arms']['finetune']['teacher_unchanged'])
            cache = root/'anchor_unsupported/final/predictions/states-3035.npz'
            with np.load(cache, allow_pickle=False) as z: original_cache = {k: z[k].copy() for k in z.files}
            changed = {k: v.copy() for k, v in original_cache.items()}; changed['group_source_teacher_states'][0, 0, 0] += 1e-5
            np.savez_compressed(cache, **changed)
            with self.assertRaisesRegex(ValueError, 'source final full-group'): compare.scene_pair(arms, '3035')
            np.savez_compressed(cache, **original_cache)
            checkpoint = root/'anchor_unsupported/fit/final.pt'; corrupt = torch.load(checkpoint, weights_only=False)
            key = next(k for k in corrupt['model'] if k.startswith('teacher.'))
            corrupt['model'][key].view(-1)[0] += 1.; torch.save(corrupt, root/'corrupt.pt')
            with self.assertRaisesRegex(ValueError, 'teacher tensor'): load_checkpoint(root/'corrupt.pt')


if __name__ == '__main__': unittest.main()
