"""Independent125 T audit regressions; synthetic software evidence only."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from tools.analysis import audit_conditional_T_arrays as a
from tools.analysis import conditional_source_value as p


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False), encoding='utf-8')


def save_heads(path, models):
    np.savez_compressed(path, **{head+'_'+key: model[key] for head,model in models.items()
                               for key in ('mean','std','beta','permutation')})


def label_fixture(seed=20):
    ids=np.array(['a','a','b','unknown']); targets=np.array(['a','b','no-input'])
    hard=np.zeros((4,64,4)); hard[...,0]=.01
    soft=np.zeros_like(hard); both=np.ones((4,64),bool);both[2,32:]=False
    valid=np.ones((3,64,4),bool);valid[0,0,:2]=False;truth=np.zeros((3,64,4))
    gain,labeled,audit=p.gain_labels(hard,soft,both,[[0],[1],[2],[3]],ids,targets,truth,valid)
    nodes=np.array(['n0','n1','n2','n3']);query=np.arange(64,dtype=float)
    feature=dict(features=np.random.default_rng(seed).normal(size=(4,64,15)),both=both,
                 hard_states=hard,soft_states=soft,membership=np.arange(4),scores=np.ones(6))
    evaluation=dict(node_ids=nodes,identities=ids,target_ids=targets,target_states=truth,target_valid=valid,query_seconds=query)
    motion=dict(node_ids=nodes,query_seconds=query)
    return feature,dict(gain=gain,labeled=labeled),evaluation,motion,audit


class ConditionalTAuditTests(unittest.TestCase):
    def test_independent_labels_duplicates_uncovered_population_and_gain_corruption(self):
        feature,labels,ev,motion,public=label_fixture()
        row=a.label_scene(feature,labels,ev,motion,a.core.Checks())
        self.assertEqual(row['audit'],public)
        self.assertEqual((len(row['y']),public['all_matched_reference_valid_query_count']),(95,127))
        self.assertEqual((public['duplicate_anchor_groups_without_training_labels'],public['unknown_anchor_groups'],
                          public['target_without_group_anchor_count']),(1,1,1))
        for kind in ('gain','duplicate'):
            broken=deepcopy(labels)
            if kind=='gain':broken['gain'][0,1]+=1.
            else:broken['labeled'][1,1]=True
            with self.subTest(kind=kind),self.assertRaises(ValueError):
                a.label_scene(feature,broken,ev,motion,a.core.Checks())

    def test_auc_ties_zero_gain_and_two_denominators(self):
        y=np.array([3.,-5.,0.,1e-10]); prediction=np.array([1.,1.,-1.,0.])
        result=a.gain_report(y,prediction,100)
        self.assertEqual(result['gain_auc'],.625)
        self.assertEqual((result['selected_count'],result['positive_count']),(2,2))
        self.assertEqual(result['full_common_point_mae_change_m'],.02)
        self.assertAlmostEqual(result['mean_regret_m'],1.25)
        self.assertEqual(result['near_zero_abs_observed_gain_le_1e_6_m_count'],2)
        self.assertIsNone(a.gain_report(np.ones(2),np.zeros(2),5)['gain_auc'])
        self.assertEqual(a.gain_report(np.empty(0),np.empty(0),4)['full_common_point_mae_change_m'],0.)

    def test_train_only_normalizer_ridge_and_shuffle_cannot_be_substituted(self):
        rng=np.random.default_rng(15);x=rng.normal(size=(50,15));y=3.+x[:,0]*7.
        models=p.fit_heads(x,y)
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/'heads.npz';save_heads(path,models)
            _,solves=a.inspect_heads(path,x,y,a.core.Checks());self.assertEqual(solves,3)
            for kind in ('normalizer','coefficient','permutation'):
                broken=deepcopy(models)
                if kind=='normalizer':broken['LINEAR']['mean'][0]+=1.
                elif kind=='coefficient':broken['QUADRATIC']['beta'][2]+=.1
                else:broken['SHUFFLED']['permutation']=np.arange(len(y))
                save_heads(path,broken)
                with self.subTest(kind=kind),self.assertRaises(ValueError):
                    a.inspect_heads(path,x,y,a.core.Checks())

    def test_complete_T32_saved_folds_OOF_main_and_heldout_leak_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);run_path=root/'run';native=root/'native';(run_path/'private').mkdir(parents=True)
            for folder in ('native-motion','native-evaluation'):(native/'private'/folder/'T_core').mkdir(parents=True)
            ids=[str(i) for i in range(32)];manifest=dict(splits=dict(T_core=ids),components={sid:'a' if int(sid)%2==0 else 'b' for sid in ids})
            write(root/'manifest.json',manifest)
            write(native/'native-cache.json',dict(status='CACHE_COMPLETED',scene_ids=dict(T_core=ids)))
            rows=[];pop=[]
            for sid in ids:
                feature,labels,ev,motion,audit=label_fixture(100+int(sid));cid=manifest['components'][sid]
                np.savez_compressed(run_path/'private'/f'T-features-{sid}.npz',**feature)
                np.savez_compressed(run_path/'private'/f'T-labels-{sid}.npz',**labels)
                np.savez_compressed(native/'private/native-motion/T_core'/f'scene-{sid}.npz',**motion)
                np.savez_compressed(native/'private/native-evaluation/T_core'/f'scene-{sid}.npz',**ev)
                rows.append(dict(scene_id=sid,component_id=cid,x=feature['features'][labels['labeled']],y=labels['gain'][labels['labeled']],audit=audit))
                pop.append(dict(scene_id=sid,component_id=cid,audit=audit))
            folds=[];oof={h:[] for h in a.HEADS}
            for index,cid in enumerate(('a','b')):
                train=[r for r in rows if r['component_id']!=cid];held=[r for r in rows if r['component_id']==cid]
                x=np.concatenate([r['x'] for r in train]);y=np.concatenate([r['y'] for r in train]);models=p.fit_heads(x,y)
                save_heads(run_path/'private'/f'head-fold-{index}.npz',models)
                folds.append(dict(fold_index=index,heldout_component=cid,training_components=['b' if cid=='a' else 'a'],
                    training_scene_ids=[r['scene_id'] for r in train],heldout_scene_ids=[r['scene_id'] for r in held],
                    labeled_training_query_count=len(y),no_label_fallback=False))
                for row in held:
                    for h in a.HEADS:oof[h].append(dict(row,prediction=p.predict_head(models[h],row['x'])))
            save_heads(run_path/'private/full-T-heads.npz',p.fit_heads(np.concatenate([r['x'] for r in rows]),np.concatenate([r['y'] for r in rows])))
            def metrics(items):
                return p.gain_metrics(np.concatenate([r['y'] for r in items]),np.concatenate([r['prediction'] for r in items]),
                    sum(r['audit']['all_matched_reference_valid_query_count'] for r in items))
            public={h:dict(metrics(items),
                per_scene=[dict(scene_id=r['scene_id'],component_id=r['component_id'],**p.gain_metrics(r['y'],r['prediction'],r['audit']['all_matched_reference_valid_query_count'])) for r in items],
                components={cid:metrics([r for r in items if r['component_id']==cid]) for cid in ('a','b')}) for h,items in oof.items()}
            amount=dict(analytic_head_fits=12,ridge_system_solves=9,constant_means_fitted=3,no_label_fallback_heads=0)
            run=dict(status='COMPLETED',exit_code=0,idea_id=a.IDEA,plan_version='1.0',training_scene_ids=ids,
                completed_T_scenes=32,total_T_scenes=32,feature_names=list(a.FEATURE_NAMES),native_run=str(native),
                training_component_ids=['a','b'],T_training_component_count=2,**amount)
            summary=dict(status='COMPLETED',idea_id=a.IDEA,plan_version='1.0',T_training_component_count=2,
                T_label_populations=pop,T_oof=public,**amount)
            for name,value in [('run.json',run),('summary.json',summary),('T-oof.json',public),('T-fold-index.json',folds)]:write(run_path/name,value)
            result=a.audit_run(run_path,root/'manifest.json',a.core.Checks())
            self.assertEqual((result['T_scene_count'],result['T_component_count'],result['audit_ridge_parameter_recomputations']),(32,2,9))
            self.assertEqual(result['T_labeled_both_query_count'],32*95)
            self.assertFalse(result['V_payload_read'])
            folds[0]['training_scene_ids'].append('0');write(run_path/'T-fold-index.json',folds)
            with self.assertRaisesRegex(ValueError,'LOCO'):
                a.audit_run(run_path,root/'manifest.json',a.core.Checks())
            output=root/'original-audit.json';output.write_text('original failed evidence')
            with self.assertRaises(FileExistsError):a.audit(run_path,root/'manifest.json',output)
            self.assertEqual(output.read_text(),'original failed evidence')


if __name__=='__main__':unittest.main()
