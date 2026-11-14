"""Regression evidence for batched, terminal-only decoding of frozen TFM."""
import unittest
from unittest.mock import patch

import numpy as np
import torch

from association.data.timed import fit_timed_normalizer, prepare_timed_scene
from association.models.fusion import PairFusion, PairFusionConfig
from association.models.timed import TimedAssociationModel
from association.models.tmtaff import ModelConfig
from association.training.common import normalize_scene, score_reference
from tools.analysis.tmtaff_own_chain import predict_scenes


def _scene(scene_id, split):
    nodes = np.array(['a', 'b', 'c', 'd', 'e', 'f'])
    identities = np.array(['x', 'x', 'x', 'y', 'y', 'z'])
    sources = np.array(['1', '2', '1', '2', '1', '2'])
    endpoints = [(i % 2, 5 - i % 2) for i in range(len(nodes))]
    tracks = [dict(node_id=node, identity_id=identity, source=source,
                   canonical_times=np.array(times, dtype=np.float64),
                   canonical_values=np.array([[0.03*i, 0.02*i, 2+i, 20+i],
                                              [0.03*i+0.1, 0.02*i+0.1, 2+i, 21+i]]))
              for i, (node, identity, source, times) in enumerate(
                  zip(nodes, identities, sources, endpoints, strict=True))]
    pairs = np.array([(0, 1), (0, 2), (1, 2), (3, 4)])
    native = dict(scene_id=scene_id, component_id=scene_id, split=split, node_ids=nodes,
                  sources=sources, identities=identities, tracks=tracks, pairs=pairs,
                  intervals=np.array(endpoints, dtype=np.float64), labels=np.ones(len(pairs), dtype=bool))
    return prepare_timed_scene(native, length=6, max_observations=2, time_unit_seconds=1.)


class TMTAFFOwnChainTests(unittest.TestCase):
    def test_independent_groups_match_sequential_terminal_states(self):
        old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, old_threads)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(147)
            normalizer = fit_timed_normalizer([_scene('train', 'T_core')])
            model = TimedAssociationModel(ModelConfig(d_model=8, nhead=2, num_layers=1,
                                                       dim_feedforward=16, length=6),
                                          'contrastive', max_observations=2, normalizer=normalizer)
            fusion = PairFusion(PairFusionConfig(fc_hidden=10, length=6, d_model=8, nhead=2,
                                                 num_layers=1, dim_feedforward=16), normalizer)
        snapshot = dict(model=model, fusion=fusion, normalizer=normalizer)
        originals = [_scene('dev1', 'V_select'), _scene('dev2', 'V_select')]
        blind = [{key: value for key, value in scene.items() if key not in {'labels', 'identities'}}
                 for scene in originals]
        with patch.object(fusion, 'decode_features', wraps=fusion.decode_features) as decode:
            latest, runtime = predict_scenes(snapshot, blind, threshold=-100., device='cpu',
                                             batch_tracks=3, pair_batch_size=2, fusion_batch_size=4)
            self.assertEqual(decode.call_count, 2)
        self.assertEqual(runtime['fusion_decodes'], 2)
        self.assertEqual(runtime['fusion_calls'], 2)
        cached = score_reference(model, normalizer, originals, 'cpu', batch_tracks=3, pair_batch_size=2)
        for scene, old, new in zip(originals, cached, latest, strict=True):
            self.assertEqual(new['groups'], [[0, 1, 2], [3, 4], [5]])
            self.assertEqual(new['steps'], 3)
            np.testing.assert_array_equal(new['scores'], old['scores'])
            with torch.inference_mode():
                values, observed = normalize_scene(scene, normalizer)
                latents = torch.cat([model.complete(torch.from_numpy(values[i:i+3]),
                                                     torch.from_numpy(observed[i:i+3]))
                                     for i in range(0, len(values), 3)])
                for group_index, group in enumerate(new['groups'][:2]):
                    state = latents[group[0]:group[0]+1]
                    for node in group[1:]:
                        prediction = fusion(state, latents[node:node+1])
                        state = prediction['fused_features']
                    np.testing.assert_allclose(new['states'][group_index], prediction['states'][0].numpy(),
                                               atol=2e-6, rtol=2e-6)
                    self.assertTrue(new['valid'][group_index].all())
            np.testing.assert_array_equal(new['states'][2], scene['values'][5])
            np.testing.assert_array_equal(new['valid'][2], np.repeat(scene['observed'][5, :, None], 4, axis=1))
