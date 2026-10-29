"""131 fixed-head/import/native regressions; synthetic software evidence only."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from association.data.archive import Scene, Track
from tools.analysis import conditional_source_value_full as full
from tests.integration.test_source_mix_shrink_control import saved121

p = full.pilot


def original_scene():
    fragment, membership, retained, calibration, public = saved121()
    groups = p.old.groups_from_membership(membership, len(fragment['native_states']))
    n = len(fragment['native_states']); pairs = np.column_stack(np.triu_indices(n, 1))
    motion = dict(states=fragment['native_states'], support=fragment['support'],
        nearest_seconds=fragment['nearest_seconds'], sources=fragment['sources'],
        native_count=np.ones(n)*7., fit_rmse_m=np.ones(n)*3., ols_velocity_mps=np.zeros((n, 2)))
    _, _, mixed, _ = p.mix.analyze_scene(fragment, membership, retained, calibration, public)
    x, both, hard, soft = p.make_actions_features(**motion, groups=groups, calibration=calibration,
        pairs=pairs, scores=np.ones(len(pairs))*2., mean_states=retained['WS_MEAN']['group_states'],
        imported_hard=retained['WS_FIXED_LOW_SOURCE_SUPPORTED']['group_states'],
        imported_soft=mixed[p.SOFT]['group_states'])
    models = p.fit_heads(np.empty((0, 15)), np.empty(0))
    added, _, _ = p.apply_heads(models, x, both, hard, soft)
    controls = {name: retained[name] for name in p.BASES}; controls[p.SOFT] = mixed[p.SOFT]
    predictions = {name: artifact['group_states'] for name, artifact in controls.items()}; predictions.update(added)
    row, arrays, artifacts = p.evaluate_v(fragment, groups, predictions, controls, both)
    row.update(scene_id=public['scene_id'], component_id='c', node_count=n,
               fixed_R_population_invariants_verified=True)
    return fragment, membership, calibration, motion, pairs, models, both, row, arrays, artifacts


class ConditionalValueFullTests(unittest.TestCase):
    def test_frozen_models_exact_roundtrip_and_schema_rejected(self):
        rng = np.random.default_rng(15)
        models = p.fit_heads(rng.normal(size=(24, 15)), rng.normal(size=24))
        payload = {h+'_'+k: model[k] for h, model in models.items() for k in ('mean', 'std', 'beta', 'permutation')}
        x = rng.normal(size=(11, 15))*100.
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'head.npz'; np.savez_compressed(path, **payload)
            with patch.object(p, 'fit_heads', side_effect=AssertionError('must not fit')):
                restored = full.load_frozen_heads(path)
            for h in p.HEADS:
                np.testing.assert_array_equal(p.predict_head(models[h], x), p.predict_head(restored[h], x))
            corrupt = dict(payload, QUADRATIC_beta=np.zeros(135))
            np.savez_compressed(path, **corrupt)
            with self.assertRaisesRegex(ValueError, 'shape'):
                full.load_frozen_heads(path)
            corrupt = dict(payload, LINEAR_std=payload['LINEAR_std']*2.)
            np.savez_compressed(path, **corrupt)
            with self.assertRaisesRegex(ValueError, 'common frozen'):
                full.load_frozen_heads(path)

    def test_original_import_preserves_all_public_values_and_never_predicts(self):
        fragment, _, _, _, _, _, both, row, arrays, artifacts = original_scene()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary); (path/'private').mkdir(); (path/'scenes').mkdir()
            sid = row['scene_id']; p.old.frozen.write(path/'scenes'/f'scene-{sid}.json.gz', row)
            np.savez_compressed(path/'private'/f'states-{sid}.npz',
                **{name+'_'+k: v for name, artifact in artifacts.items() for k, v in artifact.items()})
            np.savez_compressed(path/'private'/f'V-features-{sid}.npz', both=both)
            with patch.object(p, 'make_actions_features', side_effect=AssertionError('must not construct')), \
                 patch.object(p, 'apply_heads', side_effect=AssertionError('must not infer')), \
                 patch.object(p.old, 'evaluate_full_scene', side_effect=AssertionError('must not emit/evaluate')):
                imported_row, imported_arrays, imported = full.import_pilot_scene(path, sid, fragment)
            self.assertEqual(imported_row, row)
            for name in p.METHODS:
                for key in artifacts[name]:
                    np.testing.assert_array_equal(imported[name][key], artifacts[name][key])
                for region in arrays['methods'][name]:
                    np.testing.assert_array_equal(imported_arrays['methods'][name][region], arrays['methods'][name][region])
            self.assertEqual(len(imported_row['pairwise']), 55)
            bad = deepcopy(row); bad['pairwise'][next(iter(bad['pairwise']))]['strata']['all']['A0']['mae'] += 1.
            p.old.frozen.write(path/'scenes'/f'scene-{sid}.json.gz', bad)
            with self.assertRaisesRegex(ValueError, 'imported pair'):
                full.import_pilot_scene(path, sid, fragment)

    def test_new_scene_only_fixed_candidates_and_strict_physical_api(self):
        fragment, membership, calibration, motion, pairs, models, _, _, _, artifacts = original_scene()
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary)/'source.npz'
            payload = {name+'_'+k: value for name in p.BASES for k, value in artifacts[name].items()}
            np.savez_compressed(source, **payload, **calibration)
            with patch.object(p, 'fit_heads', side_effect=AssertionError('must not fit')):
                groups, predictions, _, both, _, gates, _ = full.new_scene_predictions(motion,
                    fragment['node_ids'], pairs, np.ones(len(pairs))*2., membership, source, models)
            self.assertEqual(set(predictions), set(p.METHODS))
            for name in p.BASES:
                np.testing.assert_array_equal(predictions[name], artifacts[name]['group_states'])
            np.testing.assert_array_equal(predictions[p.SOFT], artifacts[p.SOFT]['group_states'])
            for name in p.HEAD_METHODS:
                np.testing.assert_array_equal(predictions[name], artifacts['WS_FIXED_LOW_SOURCE_SUPPORTED']['group_states'])
            self.assertEqual(sum(int(x.sum()) for x in gates.values()), 0)
            self.assertEqual(both.shape, (len(groups), 64))
            _, _, _, _, old_public = saved121()
            # 123's extra source strata are preserved at source, while131 uses
            # exactly125's common-region schema for old and new scenes.
            _, _, _, _, _, _, _, new_public, _, _ = original_scene()
            full.compare_imported_controls(new_public, old_public)
            with self.assertRaisesRegex(ValueError, 'allowlist'):
                full.new_scene_predictions(dict(motion, identities=['forbidden']), fragment['node_ids'],
                    pairs, np.ones(len(pairs)), membership, source, models)

    def test_native_factory_uses_all_events_and_binds_exact056_fields(self):
        t = np.linspace(0., 2600., 131)
        values = np.column_stack((20.+t*1e-5+np.sin(t/100.)*.0001,
                                  110.+t*2e-5, np.ones(131)*5., np.ones(131)*10.))
        scene = Scene('9', [Track('node', 'A', '1', 'identity', 'ancestor', t, values)], {})
        manifest = dict(splits=dict(V_select=['9']), components={'9': 'c'})
        class Archive:
            def load_scene(self, sid):
                return scene
            def read_labels(self, sid):
                raise AssertionError('nonempty grid does not need label bounds')
        native = full._record(scene, 'c', 'V_select', full.DATA['max_observations'])
        prepared = full.prepare_timed_scene(native, **full.DATA)
        motion = full.scene_motion_evidence(prepared, motion_observation_policy='canonical_all')
        payload = dict(node_ids=np.asarray(prepared['node_ids'], str), native_states=motion['states'],
            **{key: motion[key] for key in ('sources', 'support', 'nearest_seconds', 'query_seconds', 'exact_observation')})
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary); (source/'predictions').mkdir()
            path = source/'predictions'/'states-9.npz'; np.savez_compressed(path, **payload)
            physical, _, _, audit = full.rebuild_native_motion(Archive(), manifest, '9', source)
            self.assertEqual(physical['native_count'][0], 131)
            self.assertEqual(audit['canonical_event_count'], 131)
            self.assertGreater(physical['fit_rmse_m'][0], 0.)
            self.assertEqual(set(physical), set(full.ACTION_INPUTS))
            changed = dict(payload, native_states=payload['native_states']+.00001)
            np.savez_compressed(path, **changed)
            with self.assertRaisesRegex(ValueError, '056 exact native states'):
                full.rebuild_native_motion(Archive(), manifest, '9', source)


if __name__ == '__main__':
    unittest.main()
