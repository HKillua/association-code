"""Native-event identity, anonymous grouping and opt-in data integration."""
import copy
import unittest
from unittest.mock import patch
import zipfile

import numpy as np

from association.data.motion_evidence import prepare_motion_scenes
from association.data.source_events import group_events, source_events
from association.data.timed import prepare_timed_scene
from association.training.evidence_fusion import prepare_evidence_scenes
from tests.fixtures.native_zip import make_native_zip_fixture
from tests.fixtures.scenes import native_scene


def _event_set(blocks):
    return {frozenset(block['event_ids']) for block in blocks}


class SourceEventsTests(unittest.TestCase):
    def test_all_canonical_points_and_real_seconds_are_preserved(self):
        scene = native_scene('s', 'T_core', shift=123456.)
        times = 123456. + np.arange(257, dtype=np.float64)*.125
        track = scene['tracks'][0]
        track.update(canonical_times=times,
                     canonical_values=np.tile([31., 120., 8., 90.], (257, 1)))
        events = source_events(scene)
        self.assertEqual(len(events[0]['times']), 257)
        np.testing.assert_array_equal(events[0]['times'], times)
        self.assertEqual(events[0]['event_ids'][-1], ('s:0', 256))
        self.assertEqual(events[0]['times'].dtype, np.float64)
        self.assertTrue(events[0]['valid'].all())
        events[0]['values'][0, 0] = 0.
        self.assertEqual(track['canonical_values'][0, 0], 31.)

    def test_duplicate_reference_removed_equal_other_node_retained(self):
        scene = native_scene('s', 'T_core')
        left, right = scene['tracks'][0], scene['tracks'][2]
        right.update(canonical_times=left['canonical_times'].copy(),
                     canonical_values=left['canonical_values'].copy())
        motion = dict(native_events=source_events(scene))
        blocks = group_events(motion, [2, 0, 0, 2])
        self.assertEqual(len(blocks), 1)
        self.assertEqual(len(blocks[0]['times']), 4)
        self.assertEqual(set(blocks[0]['event_ids']), {('s:0', 0), ('s:0', 1), ('s:2', 0), ('s:2', 1)})
        self.assertNotIn('source', blocks[0])

    def test_anonymous_source_renaming_and_member_order_preserve_event_partition(self):
        scene = native_scene('s', 'T_core')
        motion = dict(native_events=source_events(scene))
        original = group_events(motion, [0, 1, 2, 3])
        renamed = copy.deepcopy(motion)
        for record in renamed['native_events']:
            record['source'] = {'9001': 'zz', '9002': 'aa'}[record['source']]
        changed = group_events(renamed, [3, 2, 1, 0])
        self.assertEqual(_event_set(original), _event_set(changed))
        for block in original:
            match = next(item for item in changed if item['event_ids'] == block['event_ids'])
            for key in ('times', 'values', 'valid'):
                np.testing.assert_array_equal(block[key], match[key])

    def test_resegmentation_keeps_original_event_ids_and_rejects_conflicting_copies(self):
        scene = native_scene('s', 'T_core')
        original = group_events(dict(native_events=source_events(scene)), [1])
        track = scene['tracks'][1]
        pieces = []
        for name, indices in (('part-a', [0, 1]), ('part-b', [1, 2])):
            pieces.append(dict(node_id=name, source=track['source'],
                canonical_times=track['canonical_times'][indices],
                canonical_values=track['canonical_values'][indices],
                canonical_event_ids=[(track['node_id'], i) for i in indices]))
        split = dict(tracks=pieces, node_ids=['part-a', 'part-b'], sources=[track['source']]*2)
        motion = dict(native_events=source_events(split))
        rebuilt = group_events(motion, [1, 0])
        self.assertEqual(original[0]['event_ids'], rebuilt[0]['event_ids'])
        for key in ('times', 'values', 'valid'):
            np.testing.assert_array_equal(original[0][key], rebuilt[0][key])
        motion['native_events'][1]['values'][0, 0] += .01
        with self.assertRaisesRegex(ValueError, 'conflicting'):
            group_events(motion, [0, 1])

    def test_invalid_attributes_masked_without_dropping_real_events(self):
        scene = native_scene('s', 'T_core')
        track = scene['tracks'][1]
        track['canonical_values'] = np.array([[91., 120., 5., 90.],
            [30., np.nan, -1., 360.], [30., 120., 8., -1.]])
        track['canonical_valid'] = np.ones((3, 4), bool)
        track['canonical_valid'][0, 3] = False
        record = source_events(scene)[1]
        np.testing.assert_array_equal(record['valid'], [[False, True, True, False],
            [True, False, False, True], [True, True, True, False]])
        self.assertEqual(len(record['event_ids']), 3)
        self.assertTrue(np.isnan(record['values'][1, 1]))

    def test_training_opt_in_does_not_change_motion_targets_or_original_scene(self):
        scene = prepare_timed_scene(native_scene('s', 'T_core'), length=64,
                                    max_observations=3, time_unit_seconds=20.)
        scene['fusion_targets'] = {str(identity): dict(
            states=np.tile([30., 120., 5., 40.], (64, 1)), valid=np.ones((64, 4), bool))
            for identity in scene['identities']}
        old = prepare_evidence_scenes([scene], motion_observation_policy='canonical_all')[0]
        new = prepare_evidence_scenes([scene], motion_observation_policy='canonical_all',
                                      retain_native_events=True)[0]
        self.assertNotIn('native_events', old['motion'])
        self.assertNotIn('native_events', scene)
        for key, value in old['motion'].items():
            np.testing.assert_array_equal(value, new['motion'][key])
        for key in ('target', 'valid', 'identities', 'distances'):
            np.testing.assert_array_equal(old[key], new[key])
        self.assertEqual(sum(len(item['times']) for item in new['motion']['native_events']), 11)

    def test_dataset_retains_events_before_track_release_without_changing_targets(self):
        fixture = make_native_zip_fixture(self)
        old = fixture.load(motion_observation_policy='canonical_all')
        new = fixture.load(motion_observation_policy='canonical_all', retain_native_events=True)
        for previous, current in zip(old['train']+old['validation'], new['train']+new['validation']):
            self.assertNotIn('tracks', current)
            self.assertNotIn('native_events', previous['motion'])
            for key, value in previous['motion'].items():
                np.testing.assert_array_equal(value, current['motion'][key])
            for identity in previous['fusion_targets']:
                for key in ('states', 'valid'):
                    np.testing.assert_array_equal(previous['fusion_targets'][identity][key],
                                                  current['fusion_targets'][identity][key])
        self.assertEqual(len(new['train'][0]['motion']['native_events']), 4)
        self.assertEqual(new['validation'][1]['motion']['native_events'], [])
        self.assertFalse(new['report']['test_opened'])

    def test_zip_opt_in_preserves_empty_scene_and_does_not_read_targets_or_test(self):
        fixture = make_native_zip_fixture(self)
        original_read = zipfile.ZipFile.read
        def read(archive, member, *args, **kwargs):
            name = member.filename if hasattr(member, 'filename') else str(member)
            if '真实场景' in name or name.endswith('-3.csv'):
                raise AssertionError('native motion must not read reference or test')
            return original_read(archive, member, *args, **kwargs)
        with patch.object(zipfile.ZipFile, 'read', read):
            old = prepare_motion_scenes(fixture.path, fixture.manifest, 'V_select',
                                        motion_observation_policy='canonical_all')
            new = prepare_motion_scenes(fixture.path, fixture.manifest, 'V_select',
                                        motion_observation_policy='canonical_all', retain_native_events=True)
        self.assertEqual(new[1]['motion']['native_events'], [])
        self.assertEqual(sum(len(item['times']) for item in new[0]['motion']['native_events']), 12)
        for previous, current in zip(old, new):
            self.assertNotIn('native_events', previous['motion'])
            for key, value in previous['motion'].items():
                np.testing.assert_array_equal(value, current['motion'][key])


if __name__ == '__main__':
    unittest.main()
