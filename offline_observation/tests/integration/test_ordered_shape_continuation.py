"""132 descriptor/cached-input regressions; synthetic software evidence only."""
from copy import deepcopy
from pathlib import Path
import json
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from tools.analysis import ordered_shape_continuation as new

p = new.p


def scene(root, sid, split='T_core'):
    root.mkdir(parents=True, exist_ok=True)
    n = 6; pairs = np.column_stack(np.triu_indices(n, 1)); identities = np.array(['A', 'A', 'B', 'C', 'D', 'E'])
    labels = identities[pairs[:, 0]] == identities[pairs[:, 1]]
    sources = np.array(['low']*3+['high']*3)
    intervals = np.tile([0., 100.], (n, 1)); joint = np.full(len(pairs), 64, np.uint8)
    meta = dict(node_ids=np.asarray([str(i) for i in range(n)]), sources=sources, identities=identities,
                intervals=intervals, joint_support_queries=joint)
    item = dict(meta, scene_id=sid, split=split, component_id='c', pair_count=len(pairs), positive_count=int(labels.sum()))
    rng = np.random.default_rng(int(sid))
    for key, value in dict(pairs=pairs, labels=labels, features=rng.normal(size=(len(pairs), 46)).astype(np.float32),
                            base=np.zeros(len(pairs), np.float64)).items():
        item[key+'_path'] = root/f'{sid}-{key}.npy'; np.save(item[key+'_path'], value)
    np.savez_compressed(root/f'{sid}-metadata.npz', **meta)
    return item


class OrderedShapeContinuationTests(unittest.TestCase):
    def test_same_sampler_plan_exposure_and_original_norm_values(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); scenes = [scene(root, '1'), scene(root, '2')]
            old_plan, old_audit = p.build_plan(scenes, steps=7)
            new_plan, new_audit = new.build_plan(scenes, steps=7)
            np.testing.assert_array_equal(old_plan, new_plan); self.assertEqual(old_audit, new_audit)
            norms = p.fit_normalizers(scenes); p.runtime.write_json(root/'normalizers.json', norms)
            restored = new.restore_normalizers(root, sum(s['pair_count'] for s in scenes))
            self.assertEqual(restored, norms)
            broken = deepcopy(norms); broken['ORDERED']['mean'][0] += 1.
            p.runtime.write_json(root/'normalizers.json', broken)
            with self.assertRaisesRegex(ValueError, 'shared22'):
                new.restore_normalizers(root, sum(s['pair_count'] for s in scenes))

    def test_low_descriptor_limit_original_fails_new_transport_passes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for i in range(64): scene(root, str(i))
            script = r'''
import gc, json, resource, sys
from pathlib import Path
import numpy as np
from tools.analysis import ordered_shape_continuation as n
root=Path(sys.argv[1]); scenes=[]
for i in range(64):
 with np.load(root/(str(i)+'-metadata.npz'),allow_pickle=False) as z:
  meta={k:z[k] for k in z.files}
 scenes.append(dict(meta,scene_id=str(i),pairs_path=root/(str(i)+'-pairs.npy'),labels_path=root/(str(i)+'-labels.npy')))
soft,hard=resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE,(min(32,hard),hard))
failed=False
try:n.p.build_plan(scenes,steps=2)
except OSError as error:
 failed=error.errno==24
assert failed,'old transport did not reproduce descriptor failure'
gc.collect()
plan,audit=n.build_plan(scenes,steps=2)
assert plan.shape==(2,256,2) and audit['sampled_positive_count']==256
print('old Errno24 reproduced; ordinary arrays passed same low limit')
'''
            result = subprocess.run([sys.executable, '-c', script, str(root)], capture_output=True, text=True,
                cwd=Path(__file__).resolve().parents[2], timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('ordinary arrays passed', result.stdout)

    def test_all_cache_headers_metadata_and_small056_support_are_bound(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); failed = root/'126'; source = root/'056/system'
            train = scene(failed/'private/cache/T_core', '1'); val = scene(failed/'private/cache/V_select', '2', 'V_select')
            (source/'predictions').mkdir(parents=True)
            np.savez_compressed(source/'predictions/states-2.npz',
                **{k: val[k] for k in ('node_ids', 'sources', 'identities')},
                query_seconds=np.linspace(0., 100., 64), support=np.ones((6, 64), bool),
                target=np.array(['this field must not be used']))
            manifest = dict(splits=dict(T_core=['1'], V_select=['2']), components={'1': 'a', '2': 'b'})
            cached, audit = new.load_cache(failed, manifest, source)
            self.assertEqual((audit['cache_scene_count'], audit['cache_file_count']), (2, 10))
            self.assertFalse(audit['feature_payload_rescanned'])
            self.assertEqual(cached['V_select'][0]['positive_count'], 1)
            path = train['features_path']; path.write_bytes(path.read_bytes()+b'extra')
            with self.assertRaisesRegex(ValueError, 'payload length'):
                new.load_cache(failed, manifest, source)
            path.write_bytes(path.read_bytes()[:-5])
            with np.load(failed/'private/cache/V_select/2-metadata.npz', allow_pickle=False) as z:
                meta = {k:z[k] for k in z.files}
            meta['joint_support_queries'][0] = 63
            np.savez_compressed(failed/'private/cache/V_select/2-metadata.npz', **meta)
            with self.assertRaisesRegex(ValueError, 'joint input support'):
                new.load_cache(failed, manifest, source)

    def test_only_zero_update_original_descriptor_failure_is_eligible(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); source = root/'056/system'; (root/'private/cache').mkdir(parents=True)
            manifest = {'splits': {'T_core': [], 'V_select': []}}
            p.runtime.write_json(root/'manifest.json', manifest)
            run = dict(status='FAILED', exit_code=1, idea_id=p.IDEA, plan_version='1.1', phase='normalizer',
                completed_arms=[], completed_optimizer_updates=0, optimizer_updates_by_arm={a:0 for a in p.ARMS},
                arms=list(p.ARMS), fixed_updates_per_arm=1000, total_fixed_updates=4000,
                motion_observation_policy='canonical_all', source_system=str(source), manifest=str(root/'manifest.json'),
                reference_arrays_read=False, reference_csv_read=False, test_opened=False,
                error='OSError: [Errno 24] Too many open files')
            p.runtime.write_json(root/'run.json', run)
            self.assertEqual(new.verify_failed_source(root, source, manifest), run)
            p.runtime.write_json(root/'run.json', dict(run, completed_optimizer_updates=1))
            with self.assertRaisesRegex(ValueError, 'zero-update126'):
                new.verify_failed_source(root, source, manifest)
            p.runtime.write_json(root/'run.json', run); np.save(root/'private/shared-batch-plan.npy', np.zeros(1))
            with self.assertRaisesRegex(ValueError, 'no prior plan'):
                new.verify_failed_source(root, source, manifest)


if __name__ == '__main__': unittest.main()
