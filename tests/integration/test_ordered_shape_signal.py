"""126 software regressions only; no synthetic research qualification."""
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from tools.analysis import ordered_shape_signal as p


def cached(root, sid, split='T_core'):
    rng = np.random.default_rng(int(sid))
    n = 6
    pairs = np.column_stack(np.triu_indices(n, 1))
    labels = np.array([True, True, False, False, False, False, False, False, False, False,
                       False, False, False, False, False])
    features = rng.normal(size=(len(pairs), 46)).astype(np.float32)
    base = -np.ones(len(pairs))
    fields = dict(features=features, base=base, pairs=pairs, labels=labels)
    meta = dict(scene_id=sid, split=split, pair_count=len(pairs))
    for name, value in fields.items():
        path = root/f'{sid}-{name}.npy'
        np.save(path, value)
        meta[name+'_path'] = path
    return meta


class OrderedShapeTests(unittest.TestCase):
    def test_pair_symmetry_chunk_stability_and_query_order_control(self):
        state = np.zeros((3, 64, 4)); state[..., 0] = 30.; state[..., 1] = 120.
        state[1, :, 1] += np.linspace(0, .1, 64)
        state[2, :, 1] += np.linspace(.03, .02, 64)
        pairs = np.array([[0, 1], [0, 2], [1, 2]])
        values = p.extra_features(state, pairs, split_index=0, scene_index=3, pair_start=0)
        reverse = p.extra_features(state, pairs[:, ::-1], split_index=0, scene_index=3, pair_start=0)
        np.testing.assert_array_equal(values, reverse)
        chunks = np.vstack([p.extra_features(state, pairs[:1], split_index=0, scene_index=3, pair_start=0),
                            p.extra_features(state, pairs[1:], split_index=0, scene_index=3, pair_start=1)])
        np.testing.assert_allclose(values, chunks, rtol=0., atol=1e-14)
        backwards = p.extra_features(state[:, ::-1], pairs, split_index=0, scene_index=3, pair_start=0)
        np.testing.assert_allclose(values[:, 8:16], backwards[:, 8:16], atol=1e-14)
        self.assertGreater(np.max(np.abs(values[:, :8]-backwards[:, :8])), .01)
        permutation = p.keyed_permutations(0, 3, np.arange(3))
        for row in permutation:
            np.testing.assert_array_equal(np.sort(row), np.arange(64))
        self.assertFalse(np.array_equal(permutation[0], permutation[1]))
        self.assertFalse(np.array_equal(permutation, p.keyed_permutations(1, 3, np.arange(3))))

    def test_streaming_training_normalizer_and_disk_batch_plan(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            scenes = [cached(root, '1'), cached(root, '2')]
            normalizers = p.fit_normalizers(scenes)
            for arm in p.ARMS:
                self.assertEqual(normalizers[arm]['mean'][:22], normalizers['SUMMARY']['mean'][:22])
                self.assertEqual(normalizers[arm]['std'][:22], normalizers['SUMMARY']['std'][:22])
                x = p.read_features(scenes[0], np.array([0, 2]), arm, normalizers[arm])
                self.assertEqual(x.shape, (2, 30))
            np.testing.assert_array_equal(p.read_features(scenes[0], np.array([0, 2]), 'SUMMARY',
                                                         normalizers['SUMMARY'])[:, 22:], 0.)
            with self.assertRaisesRegex(ValueError, 'T_core'):
                p.fit_normalizers([dict(scenes[0], split='V_select')])
            plan, audit = p.build_plan(scenes, steps=3)
            repeated, _ = p.build_plan(scenes, steps=3)
            np.testing.assert_array_equal(plan, repeated)
            self.assertEqual(plan.shape, (3, 256, 2))
            self.assertEqual(audit['sampled_pair_count'], 768)
            self.assertEqual(audit['sampled_positive_count'], 384)
            for batch in plan:
                reconstructed = np.column_stack((
                    np.concatenate([np.full(len(indices), sid) for sid, indices in p.groups_from_plan(batch)]),
                    np.concatenate([indices for _, indices in p.groups_from_plan(batch)])))
                np.testing.assert_array_equal(reconstructed, batch)

    def test_initial_function_and_new_channels_can_receive_gradients(self):
        p.runtime.seed_everything(20)
        model = p.ShapeProbe()
        self.assertEqual(sum(q.numel() for q in model.parameters()), 6209)
        x = torch.randn((16, 30)); base = torch.linspace(-2., -1., 16, dtype=torch.float64)
        target = torch.arange(16).remainder(2).to(torch.float64)
        np.testing.assert_array_equal(model(x, base).detach().numpy(), base.numpy())
        optimizer = torch.optim.Adam(model.parameters(), lr=.001)
        for _ in range(2):
            optimizer.zero_grad()
            loss = torch.nn.functional.binary_cross_entropy_with_logits(model(x, base), target)
            loss.backward(); optimizer.step()
        self.assertGreater(float(model.network[0].weight.grad[:, 22:].abs().sum()), 0.)

    def test_shared_initial_and_failed_training_keep_partial_without_retry(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            scenes = [cached(root, '1')]
            normalizers = p.fit_normalizers(scenes)
            plan, _ = p.build_plan(scenes, steps=2)
            p.runtime.seed_everything(20)
            model = p.ShapeProbe(); initial = {k: v.clone() for k, v in model.state_dict().items()}
            trained = p.train_arm(scenes, plan, 'SUMMARY', normalizers['SUMMARY'], initial,
                                  torch.device('cpu'), root/'normal', lambda **_: None)
            saved = torch.load(root/'normal/final.pt', weights_only=False)
            self.assertEqual(saved['update'], 2)
            for k, v in initial.items():
                np.testing.assert_array_equal(v, model.state_dict()[k])
            def fail(**values):
                if values.get('active_arm_updates') == 1:
                    raise RuntimeError('intentional software interruption')
            with self.assertRaisesRegex(RuntimeError, 'intentional'):
                p.train_arm(scenes, plan, 'ORDERED', normalizers['ORDERED'], initial,
                            torch.device('cpu'), root/'failed', fail)
            saved = torch.load(root/'failed/partial.pt', weights_only=False)
            self.assertEqual(saved['update'], 1)
            self.assertFalse((root/'failed/final.pt').exists())
            with self.assertRaises(FileExistsError):
                p.train_arm(scenes, plan, 'ORDERED', normalizers['ORDERED'], initial,
                            torch.device('cpu'), root/'failed', lambda **_: None)

    def test_v_binding_reads_no_reference_arrays(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); (root/'predictions').mkdir()
            motion = dict(states=np.zeros((1, 64, 4)), support=np.ones((1, 64), bool),
                          query_seconds=np.arange(64.), nearest_seconds=np.zeros((1, 64)),
                          exact_observation=np.ones((1, 64), bool))
            prepared = dict(scene_id='1', node_ids=np.array(['node']), sources=np.array(['source']),
                            identities=np.array(['identity']))
            values = dict(node_ids=prepared['node_ids'], sources=prepared['sources'], identities=prepared['identities'],
                native_states=motion['states'], support=motion['support'], query_seconds=motion['query_seconds'],
                nearest_seconds=motion['nearest_seconds'], exact_observation=motion['exact_observation'],
                target=np.array([object()], dtype=object))
            np.savez_compressed(root/'predictions/states-1.npz', **values)
            p._bind_v(motion, prepared, root)
            with self.assertRaisesRegex(ValueError, 'states'):
                p._bind_v(dict(motion, states=np.ones((1, 64, 4))), prepared, root)


if __name__ == '__main__':
    unittest.main()
