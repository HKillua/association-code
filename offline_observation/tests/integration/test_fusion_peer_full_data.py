"""Real ZIP reader/cache boundary test, not baseline performance evidence."""
from pathlib import Path
from types import SimpleNamespace
import unittest,zipfile
from unittest.mock import patch
import numpy as np
from tests.fixtures.native_zip import make_native_zip_fixture
from tools.experiments import fusion_peer_full_data as p

class FullPeerDataTests(unittest.TestCase):
    def test_streaming_T_reference_only_and_V_frozen_registry_and_units(self):
        fixture=make_native_zip_fixture(self);root=fixture.root;native=root/'127';native.mkdir()
        write=p.old.old.frozen.write
        write(native/'native-cache.json',dict(status='CACHE_COMPLETED',native_motion_policy='canonical_all',scene_ids=dict(T_core=[],V_select=[])))
        write(native/'run.json',dict(status='COMPLETED',exit_code=0))
        manifest=root/'manifest.json';write(manifest,fixture.manifest)
        alignment=root/'alignment.json';write(alignment,{s:dict(time_offset_seconds=1000.,max_gap_seconds=600.) for s in fixture.manifest['all_scenes']})
        source=root/'056';(source/'predictions').mkdir(parents=True);(source/'system-predictions').mkdir()
        with p.MTADArchive(fixture.path) as archive:
            for sid in fixture.manifest['splits']['V_select']:
                labels=archive.read_labels(sid);raw=p._record(archive.load_scene(sid),sid,'V_select',128)
                query=None if raw['tracks'] else np.linspace(5,25,64)
                scene=p.prepare_timed_scene(raw,**p.DATA,query_times=query);motion=p.scene_motion_evidence(scene,motion_observation_policy='canonical_all')
                targets=np.asarray(list(dict.fromkeys('-'.join(r['mmsi'].split('-')[:2]) for r in labels)),str)
                np.savez_compressed(source/'predictions'/f'states-{sid}.npz',query_seconds=motion['query_seconds'],
                    native_states=motion['states'],support=motion['support'],nearest_seconds=motion['nearest_seconds'],
                    node_ids=np.asarray(scene['node_ids'],str),identities=np.asarray(scene['identities'],str),target_identity_ids=targets)
                np.savez_compressed(source/'system-predictions'/f'states-{sid}.npz',
                    learned_assoc_mean_supported_aligned_truth=np.full((len(targets),64,4),np.nan),
                    learned_assoc_mean_supported_aligned_truth_valid=np.zeros((len(targets),64,4),bool))
        original=zipfile.ZipFile.read;seen=[]
        def guarded(archive,member,*args,**kwargs):
            name=member.filename if hasattr(member,'filename') else str(member);seen.append(name)
            if name.endswith('-3.csv') or ('真实场景' in name and not name.endswith('-0.csv')):raise AssertionError('V reference/test read')
            return original(archive,member,*args,**kwargs)
        args=SimpleNamespace(output=root/'new151',native_run=native,source_system=source,manifest=manifest,
                             alignment=alignment,archive=[str(fixture.path)],time_policy='advisory')
        with patch.object(zipfile.ZipFile,'read',guarded):p.run(args)
        summary=p.old.old.frozen.read(args.output/'summary.json')
        self.assertEqual(summary['scene_counts'],dict(T_core=1,V_select=2))
        self.assertTrue(any('真实场景' in name for name in seen))
        motion=p.load(args.output/'private/motion/T_core/scene-0.npz')
        np.testing.assert_allclose(motion['states'][...,2],5*1852/3600)
        self.assertEqual(str(motion['state_speed_unit']),'mps')
        ev=p.load(args.output/'private/evaluation/T_core/scene-0.npz')
        np.testing.assert_allclose(ev['target_states'][...,2][ev['target_valid'][...,2]],5.)
        empty=p.load(args.output/'private/evaluation/V_select/scene-4.npz')
        self.assertEqual(len(empty['target_ids']),1)
