"""195: T-only O/MC risk transfer with source/node/motion decision separation."""
from __future__ import annotations
import argparse, sys, time
from pathlib import Path
from datetime import datetime, timezone
if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import torch
from association.evaluation import action_conditioned_risk as action
from association.evaluation.local_source_evidence import predict_residual
from tools.analysis import association_context_pilot as p
from tools.analysis.drift_memory_pilot import ARMS as OLD_ARMS

ARMS = {front: values + tuple(values[1][0] + x for x in ('CF', 'RM', 'JM')) for front, values in OLD_ARMS.items()}
METHODS = tuple(v + '_' + kind for values in ARMS.values() for v in values for kind in ('MEAN', 'NN'))
PRIMARY = tuple(values[b] + '_NN_vs_' + values[a] + '_NN' for values in ARMS.values() for a, b in ((1,9),(6,9),(7,9),(8,9),(6,8),(6,7)))


def run(a):
    old = p.common.old.frozen
    parent = old.read(a.parent_run / 'run.json')
    control = old.read(a.control_run / 'run.json')
    assert parent['status'] == control['status'] == 'COMPLETED'
    assert control['parent_run'] == str(a.parent_run) and control['evaluation_splits'] == ['T_core']
    assert len(parent['training_scene_ids']) == 32
    out = a.output
    out.mkdir(exist_ok=False)
    (out / 'private').mkdir(); (out / 'scenes').mkdir()
    started = time.monotonic()
    state = dict(status='RUNNING', idea_id='IDEA-ACTION-CONDITIONED-RISK-195', plan_version='1.0', mode='action_conditioned_risk', parent_run=str(a.parent_run), control_run=str(a.control_run), native_run=parent['native_run'], manifest=parent['manifest'], training_scene_ids=parent['training_scene_ids'], scene_ids=[], evaluation_splits=['T_core'], methods=METHODS, optimizer_updates=0, neural_head_fits=0, analytic_head_fits=0, native_likelihood_fits=0, teacher_forward_calls=0, frontend_optimizer_updates=0, native_mean_computations=0, frozen_risk_forward_calls=0, test_opened=False, reference_used_for_prediction=False, started_at=datetime.now(timezone.utc).isoformat())
    def check(**kw):
        state.update(kw, elapsed_seconds=time.monotonic()-started, peak_rss_bytes=old.peak_rss_bytes())
        old.write(out / 'run.json', state)
        if state['peak_rss_bytes'] > 8*1024**3:
            raise MemoryError('RSS8GiB')
    try:
        torch.set_num_threads(4)
        manifest = old.read(Path(parent['manifest']))
        rows, arrays, diagnostics = [], [], []
        for sid in parent['training_scene_ids']:
            n = Path(parent['native_run']) / 'private'
            m = p.common.load_npz(n / 'native-motion/T_core' / f'scene-{sid}.npz')
            events = p.common.load_npz(n / 'native-events/T_core' / f'scene-{sid}.npz')
            p.common.old._same(m['node_ids'], events['node_ids'], 'native nodes')
            p.common.old._same(m['query_seconds'], events['query_seconds'], 'native queries')
            p.common.old._same(m['native_count'], np.diff(events['event_offsets']), 'native counts')
            op = p.common.load_npz(a.parent_run / 'private' / f'policy-T_core-{sid}.npz')
            q = p.common.load_npz(a.parent_run / 'private' / f'partitions-T_core-{sid}.npz')
            raw = p.common.load_npz(a.control_run / 'private' / f'raw-T_core-{sid}.npz')
            mc, velocity, bounds, known, means = action.candidates(m, events)
            state['native_mean_computations'] += means
            fold = 'leave_' + str(manifest['components'][sid])
            model = torch.load(a.parent_run / 'private' / f'model-{fold}.pt', map_location='cpu', weights_only=False)
            parts = dict(pairs=q['pairs'], scores=q['scores']); groups = {}; policy = {}
            for front, arms in ARMS.items():
                g = p.common.old.groups_from_membership(q[front], len(m['states']))
                x, eligible = action.features(m, g, op[front+'__count'], op[front+'__features'], mc, velocity, bounds, known)
                score = predict_residual(model, x.reshape(-1, 12), 'cpu').reshape(m['support'].shape)
                state['frozen_risk_forward_calls'] += 1
                # An undefined alternative is exactly the original state AND score.
                score[~known] = op[front+'__scores'][~known]
                outputs, weights = action.readouts(m, g, op[front+'__scores'], score, mc, eligible, raw[arms[1]+'_NN'])
                for gid, nodes in enumerate(g):
                    for j in np.flatnonzero(eligible[gid]):
                        w = op[arms[1]+'__weights'][nodes,j]
                        np.testing.assert_allclose(action.spherical_position(mc[nodes,j,:2],w),raw[arms[6]+'_NN'][gid,j,:2],rtol=0,atol=1e-10)
                for prefix, key in zip(arms[7:], ('CF','RM','JM')):
                    raw[prefix+'_NN'] = outputs[key]; raw[prefix+'_MEAN'] = raw[front+'_MEAN']
                    policy[prefix+'__weights'] = weights[key]
                policy.update({front+'__'+k:v for k,v in dict(features=x,scores=score,eligible=eligible,known=known).items()})
                for prefix in arms:
                    groups[prefix]=g; parts[prefix]=q[front]; parts[prefix+'_scores']=q[front+'_scores']
                mask = np.zeros(m['support'].shape,bool)
                for gid,nodes in enumerate(g):mask[np.ix_(nodes,np.flatnonzero(eligible[gid]))]=True
                zz = (x[mask]-model['mean'])/model['std']
                diagnostics.append(dict(scene_id=sid,front=front,eligible_group_queries=int(eligible.sum()),eligible_node_queries=int(mask.sum()),new_feature_clipped_cells=int((np.abs(zz)>8).sum()),new_feature_clipped_rows=int((np.abs(zz)>8).any(1).sum()),rows=int(len(zz))))
            # Prediction and policy are saved before any reference is opened.
            np.savez_compressed(out/'private'/f'raw-T_core-{sid}.npz',**raw)
            np.savez_compressed(out/'private'/f'partitions-T_core-{sid}.npz',**parts)
            np.savez_compressed(out/'private'/f'policy-T_core-{sid}.npz',**policy)
            ev = p.common.evaluation(Path(parent['native_run']), 'T_core', sid, m)
            row, arr, artifact = p.evaluate(m,ev,groups,{},raw,'T_core',method_names=METHODS)
            original = old.read(a.control_run/'scenes'/f'T_core-{sid}.json.gz')
            row.update(scene_id=sid,component_id=str(manifest['components'][sid]),search={},association={prefix+'_'+kind:original['association'][front+'_'+kind] for front,arms in ARMS.items() for prefix in arms for kind in ('raw','decoded')})
            p.common.save_artifacts(out/'private'/f'states-T_core-{sid}.npz',artifact)
            old.write(out/'scenes'/f'T_core-{sid}.json.gz',row); rows.append(row); arrays.append(arr)
            check(phase='T_core',completed_T_scenes=len(rows))
        result = p.finish(rows,arrays,METHODS,PRIMARY)
        result.update(status='COMPLETED',idea_id=state['idea_id'])
        old.write(out/'T-summary.json',result); old.write(out/'T-scene-summary.json.gz',rows); old.write(out/'action-feature-accounting.json',diagnostics)
        check(status='COMPLETED',exit_code=0,finished_at=datetime.now(timezone.utc).isoformat())
    except BaseException as e:
        check(status='FAILED',exit_code=1,error=f'{type(e).__name__}: {e}')
        raise

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for k in ('parent-run','control-run','output'):parser.add_argument('--'+k,type=Path,required=True)
    parser.add_argument('--time-policy',choices=('advisory',),required=True)
    run(parser.parse_args())
