"""Cache import/extension checks only; synthetic values are not MTAD evidence."""
from copy import deepcopy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from tests.integration import test_query_contribution_full_control as full_tests
from tests.integration.test_query_contribution_control import scene_fixture
from tools.analysis import shared_component_full_control as p


def cache_fixture(root):
    args117, meta, _ = full_tests.cache_fixture(root)
    args117.output = root/p.CONTROL_RUN
    with patch.object(p.old, 'metadata', side_effect=lambda *a: deepcopy(meta)):
        p.full.run(args117)
    pilot = root/p.PILOT_RUN
    (pilot/'scenes').mkdir(parents=True)
    (pilot/'private').mkdir()
    fragment, memberships, retained, public, _ = scene_fixture('1')
    public['component_id'] = 'large'
    row, _, artifacts, private = p.p.analyze_scene(fragment, memberships, retained, public)
    p.old.frozen.write(pilot/'scenes/scene-1.json.gz', row)
    payload = {name+'_'+key: value for name, a in artifacts.items() for key,value in a.items()}
    payload.update(private)
    np.savez_compressed(pilot/'private/states-1.npz', **payload)
    run = dict(status='COMPLETED', idea_id=p.p.IDEA, plan_version='1.0',
        source_system=str(args117.source_system), partition_run=str(args117.partition_run),
        scene_ids=['1'], completed_scenes=1, methods=list(p.p.METHODS))
    summary = dict(status='COMPLETED', idea_id=p.p.IDEA, plan_version='1.0', scene_ids=['1'],
        fixed_R_population_invariants_verified=True, original115_exact_replays_verified=True, no_expansion=True)
    p.old.frozen.write(pilot/'run.json', run)
    p.old.frozen.write(pilot/'summary.json', summary)
    return SimpleNamespace(source_system=args117.source_system, partition_run=args117.partition_run,
        pilot_run=pilot, control_run=args117.output, manifest=args117.manifest,
        output=root/'full-out', cpu_threads=1), meta


class SharedComponentFullTests(unittest.TestCase):
    def test_import_never_constructs_pilot_and_stratifies_new_components(self):
        with tempfile.TemporaryDirectory() as temporary:
            args, meta = cache_fixture(Path(temporary))
            with patch.object(p.old, 'metadata', side_effect=lambda *a: deepcopy(meta)), \
                 patch.object(p.p, 'construct_predictions', wraps=p.p.construct_predictions) as construct:
                result = p.run(args)
            self.assertEqual(construct.call_count, 2)
            self.assertEqual(result['scene_prediction_arms_imported'], 8)
            self.assertEqual(result['scene_prediction_arms_computed'], 16)
            self.assertEqual(result['original32']['scene_ids'], ['1'])
            self.assertEqual(result['remaining268_seen_components']['scene_ids'], ['2'])
            self.assertEqual(result['remaining268_unseen_components']['scene_ids'], ['3'])
            self.assertEqual(result['without_largest_component']['scene_ids'], ['3'])
            self.assertEqual(len(result['pairwise']), 22)
            self.assertFalse((args.output/'private/states-1.npz').exists())
            self.assertTrue((args.output/'private/states-2.npz').exists())
            state = p.old.frozen.read(args.output/'run.json')
            self.assertEqual((state['status'], state['exit_code'], state['completed_scenes']), ('COMPLETED', 0, 3))
            with self.assertRaises(FileExistsError):
                p.run(args)

    def test_corrupt_control_preserves_partial_run_without_new_attempt(self):
        with tempfile.TemporaryDirectory() as temporary:
            args, meta = cache_fixture(Path(temporary))
            path = args.control_run/'private/states-3.npz'
            with np.load(path, allow_pickle=False) as saved:
                payload = {key: saved[key] for key in saved.files}
            payload['WS_ANCHOR_group_states'][0, 1, 0] += 1e-6
            np.savez_compressed(path, **payload)
            with patch.object(p.old, 'metadata', side_effect=lambda *a: deepcopy(meta)):
                with self.assertRaisesRegex(ValueError, '117 exact ordinary control'):
                    p.run(args)
            state = p.old.frozen.read(args.output/'run.json')
            self.assertEqual((state['status'], state['completed_scenes']), ('FAILED', 2))
            self.assertTrue((args.output/'private/states-2.npz').exists())
            self.assertFalse((args.output/'summary.json').exists())

    def test_metadata_rejects_incomplete_or_reordered_pilot(self):
        with tempfile.TemporaryDirectory() as temporary:
            args, meta = cache_fixture(Path(temporary))
            run = p.old.frozen.read(args.pilot_run/'run.json')
            p.old.frozen.write(args.pilot_run/'run.json', dict(run, scene_ids=['2']))
            with patch.object(p.old, 'metadata', side_effect=lambda *a: deepcopy(meta)):
                with self.assertRaisesRegex(ValueError, 'original118/117'):
                    p.metadata(args.source_system, args.partition_run, args.pilot_run, args.control_run, {})


if __name__ == '__main__':
    unittest.main()
