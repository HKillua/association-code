import copy
import unittest

import numpy as np

from association.fusion_targets import build_scene_fusion_targets


class SceneTargetTests(unittest.TestCase):
    def fixture(self):
        scene = dict(scene_id='7', split='V_select', identities=['a-b'], grid=np.array([0., 10.]))
        reference = dict(scene_id='7', split='V_select', state_fields=['lat', 'lon', 'vel', 'cou'],
                         rows=[dict(identity_id='a-b', time=100., state=[20., 130., 3., 359.], line=1),
                               dict(identity_id='a-b', time=110., state=[21., 131., 5., 1.], line=2)], issues=[])
        return scene, reference

    def test_unavailable_and_no_node_targets_remain_without_changing_inputs(self):
        scene, reference = self.fixture()
        before = copy.deepcopy(reference)
        targets = build_scene_fusion_targets(scene, reference, target_identity_ids=['a-b', 'c-d'],
                                             time_offset_seconds=100, max_gap_seconds=10)
        self.assertEqual(list(targets), ['a-b', 'c-d'])
        self.assertTrue(targets['a-b']['valid'].all())
        self.assertFalse(targets['c-d']['valid'].any())
        np.testing.assert_array_equal(targets['a-b']['query_times'], scene['grid'])
        self.assertFalse(np.shares_memory(targets['a-b']['query_times'], scene['grid']))
        np.testing.assert_array_equal(targets['a-b']['states'][:, 0], [20., 21.])
        self.assertNotIn('fusion_targets', scene)
        self.assertEqual(reference, before)

    def test_wrong_scene_split_or_population_cannot_drop_targets(self):
        for change in ('scene_id', 'split', 'identities'):
            scene, reference = self.fixture()
            if change == 'scene_id': reference[change] = '8'
            elif change == 'split': scene[change] = reference[change] = 'T_test'
            else: scene[change] = ['absent-target']
            with self.subTest(change=change), self.assertRaises(ValueError):
                build_scene_fusion_targets(scene, reference, target_identity_ids=['a-b'],
                                            time_offset_seconds=100, max_gap_seconds=10)

    def test_empty_scene_still_validates_time_policy(self):
        scene, reference = self.fixture()
        scene['identities'] = []
        result = build_scene_fusion_targets(scene, reference, target_identity_ids=[],
                                           time_offset_seconds=100, max_gap_seconds=10)
        self.assertEqual(result, {})
        with self.assertRaises(ValueError):
            build_scene_fusion_targets(scene, reference, target_identity_ids=[],
                                       time_offset_seconds=100, max_gap_seconds=-1)


if __name__ == '__main__':
    unittest.main()
