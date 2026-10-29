"""131 audit lineage/membership regressions; synthetic software evidence only."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

import numpy as np

from tools.analysis import audit_conditional_full_arrays as audit
from tests.integration.test_audit_conditional_value_arrays import fixed_scene, write_json, producer


def metadata_fixture(root):
    path, prior, control = root/'131', root/'125', root/'123'
    ids = [str(i) for i in range(300)]; original = ids[:64:2]
    training = ['T'+str(i) for i in range(32)]
    components = {sid:'v'+str(i % 38).zfill(2) for i,sid in enumerate(ids)}
    components.update({sid:'v'+str(i % 9).zfill(2) for i,sid in enumerate(original)})
    components.update({sid:('v'+str(i % 6).zfill(2) if i % 12 < 6 else 't'+str(i % 12))
                       for i,sid in enumerate(training)})
    manifest = dict(splits=dict(V_select=ids, T_core=training), components=components)
    t_components = {components[sid] for sid in training}; v_components = {components[sid] for sid in original}
    rows=[]
    for sid in ids:
        imported = sid in original; cid=components[sid]
        row = dict(scene_id=sid, component_id=cid, stratum='original32' if imported else 'remaining268',
            component_seen_in_original32=cid in v_components, component_seen_in_T32_head_training=cid in t_components,
            imported_prediction_methods=list(audit.METHODS if imported else audit.BASES),
            newly_computed_prediction_methods=[] if imported else list(audit.METHODS[6:]),
            original125_public_and_artifacts_imported_exact=imported, node_count=1,
            methods={audit.METHODS[0]:dict(population=dict(predicted_group_count=2))})
        if not imported:
            row['native_input_audit']=dict(node_count=1, canonical_event_count=5, canonical_all=True,
                physical_binding_exact=True, native_quality_reconstructed_from_events=True)
        rows.append(row)
    lineage=dict(source_system=str(root/'056/system'), partition_run=str(root/'114'))
    zero=dict(models_constructed=0,teacher_forward_calls=0,model_forwards=0,neural_model_forwards=0,
        optimizer_updates=0,analytic_head_fits=0,ridge_system_solves=0)
    run=dict(status='COMPLETED',exit_code=0,idea_id=audit.IDEA,plan_version='1.0',
        scene_ids=ids,original32_ids=original,training_scene_ids=training,training_component_ids=sorted(t_components),
        feature_names=['f'+str(i) for i in range(15)], methods=list(audit.METHODS),
        completed_scenes=300,total_scenes=300,pilot_run=str(prior),source_control_run=str(control),
        restored_head_count=4,declared_primary_head='QUADRATIC',restored_T_label_count=3,
        restored_head_coefficient_counts=dict(TMEAN=1,LINEAR=16,QUADRATIC=136,SHUFFLED=136),
        imported_prediction_scene_arms=1960,newly_computed_prediction_scene_arms=1340,
        completed_native_scenes=268,canonical_event_count=1340,analytic_head_application_count=1072,
        analytic_query_prediction_count=4*268*2*64,**lineage,**zero)
    for key in ('scores_recomputed','groups_recomputed','threshold_reselected','normalizer_refitted',
                'centers_refitted','head_reselected','test_opened'):run[key]=False
    largest=max(sorted(set(components[sid] for sid in ids)), key=lambda cid:sum(r['component_id']==cid for r in rows))
    labels=('original32','remaining268','remaining268_seen_components','remaining268_unseen_components',
        'T32_head_training_seen_components','T32_head_training_unseen_components',
        'remaining268_T32_head_training_seen_components','remaining268_T32_head_training_unseen_components',
        'without_largest_component')
    summary=dict(status='COMPLETED',idea_id=audit.IDEA,plan_version='1.0',scene_ids=ids,scene_count=300,
        original32_ids=original,training_scene_ids=training,methods={k:None for k in audit.METHODS},
        components={components[sid]:None for sid in ids},largest_component_id=largest,
        restored_head_count=4,declared_primary_head='QUADRATIC',scene_prediction_arms_imported=1960,
        scene_prediction_arms_computed=1340,completed_native_scenes=268,canonical_event_count=1340,
        analytic_head_application_count=1072,analytic_query_prediction_count=4*268*2*64,**zero,
        **{key:None for key in labels})
    for key in ('all125_fullT_head_arrays_imported_exact','original125_public_and_artifacts_imported_exact',
                'fixed_R_population_invariants_verified','no_expansion_beyond_V300','position_only'):summary[key]=True
    write_json(path/'run.json',run);write_json(path/'summary.json',summary);write_json(path/'scene-summary.json.gz',rows)
    write_json(prior/'run.json',dict(status='COMPLETED',exit_code=0,idea_id=audit.fixed.IDEA,plan_version='1.0',
        completed_V_scenes=32,scene_ids=original,training_scene_ids=training,methods=list(audit.METHODS),
        feature_names=run['feature_names'],threshold=1.7320103112387182,source_control_run=str(control),**lineage))
    write_json(prior/'summary.json',dict(status='COMPLETED',idea_id=audit.fixed.IDEA,plan_version='1.0',
        scene_ids=original,full_T_models_frozen_before_V=True))
    write_json(control/'run.json',dict(status='COMPLETED',exit_code=0,idea_id=audit.CONTROL_IDEA,
        plan_version='1.0',scene_ids=ids,original32_ids=original,**lineage))
    return path, manifest, run, summary, rows, prior, control


class ConditionalFullAuditTests(unittest.TestCase):
    def test_completed131_manifest_subsets_and_wrong_seen_metadata_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path,manifest,run,summary,rows,_,_=metadata_fixture(Path(temporary))
            result=audit.metadata(path,manifest,audit.core.Checks());subsets=result[3]
            self.assertNotIn('completed_T_scenes',run)
            self.assertEqual(len(subsets),9)
            self.assertEqual([rows[i]['scene_id'] for i in subsets['original32']],run['original32_ids'])
            self.assertEqual(len(subsets['remaining268']),268)
            self.assertEqual(set(subsets['remaining268_seen_components'])|set(subsets['remaining268_unseen_components']),
                             set(subsets['remaining268']))
            self.assertTrue(subsets['T32_head_training_seen_components'])
            self.assertTrue(subsets['T32_head_training_unseen_components'])
            changed=deepcopy(rows);changed[0]['component_seen_in_original32']=False
            write_json(path/'scene-summary.json.gz',changed)
            with self.assertRaisesRegex(ValueError,'manifest subset membership'):
                audit.metadata(path,manifest,audit.core.Checks())
            write_json(path/'scene-summary.json.gz',rows)
            bad=deepcopy(summary);del bad['remaining268_T32_head_training_unseen_components']
            write_json(path/'summary.json',bad)
            with self.assertRaisesRegex(ValueError,'nine declared subset'):
                audit.metadata(path,manifest,audit.core.Checks())
            write_json(path/'summary.json',summary)
            bad=deepcopy(manifest);bad['components'][run['original32_ids'][0]]='wrong'
            with self.assertRaises(ValueError):audit.metadata(path,bad,audit.core.Checks())
            write_json(path/'run.json',dict(run,completed_scenes=299,completed_T_scenes=32,completed_V_scenes=300))
            with self.assertRaisesRegex(ValueError,'completed_scenes schema'):
                audit.metadata(path,manifest,audit.core.Checks())

    def test_original11_new6_and_all16_head_arrays_are_exact_not_recomputed(self):
        saved,_,_=fixed_scene()
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);prior=root/'125';control=root/'123';current=root/'131'
            for directory in (prior,control,current):(directory/'private').mkdir(parents=True)
            payload={name+'_'+key:value for name,art in saved.items() for key,value in art.items()}
            np.savez_compressed(prior/'private/states-original.npz',**payload)
            np.savez_compressed(control/'private/states-new.npz',**{k:v for k,v in payload.items()
                if any(k.startswith(name+'_') for name in audit.BASES)})
            audit.inspect_imports(current,prior,control,'original',True,saved,audit.core.Checks())
            audit.inspect_imports(current,prior,control,'new',False,saved,audit.core.Checks())
            bad=deepcopy(saved);bad['WS_QUADRATIC_VALUE_GATE']['group_states'][0,0,0]+=.01
            with self.assertRaisesRegex(ValueError,'exact imported131'):
                audit.inspect_imports(current,prior,control,'original',True,bad,audit.core.Checks())
            # New six-arm lineage binds an actual retained arm, not all11 at123.
            bad=deepcopy(saved);bad['WS_ANCHOR']['aligned_prediction'][0,0,0]+=.01
            with self.assertRaisesRegex(ValueError,'exact imported131'):
                audit.inspect_imports(current,prior,control,'new',False,bad,audit.core.Checks())
            heads={}
            for h in audit.HEADS:
                heads.update({h+'_mean':np.zeros(15),h+'_std':np.ones(15),
                    h+'_beta':np.ones(dict(TMEAN=1,LINEAR=16,QUADRATIC=136,SHUFFLED=136)[h]),
                    h+'_permutation':np.arange(3)})
            for directory in (prior,current):np.savez_compressed(directory/'private/full-T-heads.npz',**heads)
            audit.inspect_heads(current,prior,dict(restored_T_label_count=3),audit.core.Checks())
            corrupt=deepcopy(heads);corrupt['QUADRATIC_beta'][135]+=1e-12
            np.savez_compressed(current/'private/full-T-heads.npz',**corrupt)
            with self.assertRaisesRegex(ValueError,'exact125 frozen head'):
                audit.inspect_heads(current,prior,dict(restored_T_label_count=3),audit.core.Checks())
            corrupt=deepcopy(heads);corrupt['LINEAR_mean']=corrupt['LINEAR_mean'].astype(np.float32)
            np.savez_compressed(current/'private/full-T-heads.npz',**corrupt)
            with self.assertRaisesRegex(ValueError,'exact dtype'):
                audit.inspect_heads(current,prior,dict(restored_T_label_count=3),audit.core.Checks())

    def test_aggregate_signed_risk_and_import_counters_cannot_drift(self):
        saved, row, arrays = fixed_scene()
        row.update(imported_prediction_methods=list(audit.BASES),
                   newly_computed_prediction_methods=list(audit.METHODS[6:]))
        result = audit.fixed.inspect_fixed_scene(saved, row, audit.core.Checks())
        report = producer.qc.aggregate([row], [arrays], method_names=audit.METHODS)
        report.update(scene_prediction_arms_imported=6, scene_prediction_arms_computed=5)
        audit.inspect_aggregate(report, [0], [result], [row], audit.core.Checks(), 'component')
        key = next(iter(report['pairwise']))
        for field in ('risk', 'counter'):
            corrupt = deepcopy(report)
            if field == 'risk':
                corrupt['pairwise'][key]['strata']['all']['candidate_minus_A0']['mean_error_change'] = 123456.
            else:
                corrupt['scene_prediction_arms_imported'] = 7
            with self.subTest(field=field), self.assertRaises(ValueError):
                audit.inspect_aggregate(corrupt, [0], [result], [row], audit.core.Checks(), 'component')

    def test_incomplete_run_and_existing_audit_not_touched(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);path,manifest,run,_,_,_,_=metadata_fixture(root)
            write_json(path/'run.json',dict(run,status='RUNNING'))
            with self.assertRaisesRegex(ValueError,'completed131 only'):
                audit.metadata(path,manifest,audit.core.Checks())
            target=root/'old-failed-audit.json';target.write_text('preserve failed evidence')
            with self.assertRaises(FileExistsError):audit.audit(path,root/'manifest.json',target)
            self.assertEqual(target.read_text(),'preserve failed evidence')


if __name__=='__main__':unittest.main()
