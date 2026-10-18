"""Read-only 016 speed/case reduction. No models, inference, ZIP or test data.

Run from any cwd with numpy. Add --plot when matplotlib is available.
"""
from pathlib import Path
import argparse
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT/'HandOver/results/COOPERATIVE-ARRAYS-20261018-016/files'
OUT = Path(__file__).resolve().parent
ARMS = ('plain', 'identity', 'utility', 'learned')


def haversine(a, b):
    d = np.deg2rad(b-a)
    lat = np.deg2rad(a[:, 0]); latb = np.deg2rad(b[:, 0])
    q = np.sin(d[:, 0]/2)**2 + np.cos(lat)*np.cos(latb)*np.sin(d[:, 1]/2)**2
    return 6371000*2*np.arcsin(np.sqrt(np.clip(q, 0, 1)))


def main(plot=False):
    inspection = json.loads((SOURCE/'inspection.json').read_text())
    speed = {}
    for arm in ARMS:
        speed[arm] = {}
        for kind in ('predicted', 'oracle'):
            x = inspection['speed'][arm][kind]
            pred = x['prediction_on_reference_intersection']; ref = x['reference_on_prediction_intersection']
            speed[arm][kind] = dict(
                all_output=x['prediction_valid'], prediction_intersection=pred,
                reference_intersection=ref,
                intersection_mean_signed_error=pred['mean']-ref['mean'],
                ratio_of_intersection_medians=pred['quantiles']['0.5']/ref['quantiles']['0.5'],
                all_outputs_above_reference_p99=x['prediction_valid']['quantiles']['0.0'] > x['reference_valid']['quantiles']['0.99'],
                ranges_overlap=not (x['prediction_valid']['quantiles']['0.0'] > x['reference_valid']['quantiles']['1.0'] or x['prediction_valid']['quantiles']['1.0'] < x['reference_valid']['quantiles']['0.0']))
    cases=[]; plot_cases=[]
    for case in inspection['case_selection']['cases']:
        entry=next(x for x in inspection['copied_cases'] if x['scene_id']==case['scene_id'] and x['arm']=='plain')
        idx=f"{entry['scene_index']:06d}"
        arrays={a:dict(np.load(SOURCE/'cases'/idx/a/f'{idx}.npz',allow_pickle=False)) for a in ARMS}
        base=arrays['plain']; prefix='m001__'
        for arm,a in arrays.items():
            for k in base:
                if k.startswith('scene__') or (k.startswith(prefix) and any(s in k for s in ('truth','observed','target_ids','prediction_valid'))):
                    assert np.array_equal(base[k],a[k],equal_nan=True) if base[k].dtype.kind not in 'US' else np.array_equal(base[k],a[k]), (arm,k)
        rows=[]
        for target_row, target_id in enumerate(base[prefix+'target_ids']):
            truth=base[prefix+'aligned_truth'][target_row]
            valid=base[prefix+'aligned_truth_valid'][target_row,:,:2].all(-1)
            for arm,a in arrays.items():
                pred=a[prefix+'aligned_prediction'][target_row]
                mask=valid & a[prefix+'aligned_prediction_valid'][target_row,:,:2].all(-1)
                t,p=truth[mask,:2],pred[mask,:2]
                scale=np.array([111195.,111195.*np.cos(np.deg2rad(t[:,0].mean()))])
                error=(p-t)*scale; bias=error.mean(0)
                centered=error-bias
                e=haversine(t,p)
                rows.append(dict(target_id=str(target_id),target_row=target_row,arm=arm,n=int(mask.sum()),
                    position_mae_metres=float(e.mean()),position_rmse_metres=float(np.sqrt((e*e).mean())),
                    tangent_bias_north_east_metres=bias.tolist(),
                    tangent_bias_squared_fraction=float((bias*bias).sum()/(error*error).sum(1).mean()),
                    tangent_residual_rms_metres=float(np.sqrt((centered*centered).sum(1).mean())),
                    prediction_centered_rms_metres=float(np.sqrt((((p-p.mean(0))*scale)**2).sum(1).mean())),
                    reference_centered_rms_metres=float(np.sqrt((((t-t.mean(0))*scale)**2).sum(1).mean()))))
        byarm={}
        for arm in ARMS:
            ar=[x for x in rows if x['arm']==arm]
            byarm[arm]=dict(target_count=len(ar),points=sum(x['n'] for x in ar),
                point_weighted_position_mae_metres=sum(x['position_mae_metres']*x['n'] for x in ar)/sum(x['n'] for x in ar),
                median_target_bias_squared_fraction=float(np.median([x['tangent_bias_squared_fraction'] for x in ar])),
                minimum_target_bias_squared_fraction=min(x['tangent_bias_squared_fraction'] for x in ar))
        cases.append(dict(**case,scene_index=entry['scene_index'],summary=byarm,targets=rows))
        plot_cases.append((case,arrays))
    result=dict(scope='016 server full-V scalar distributions; independent local arrays only for four predetermined case scenes. Oracle groups avoid conflating association errors with decoding.',
        no_models_loaded=True,test_opened=False,normalizer_inverse_not_computed='015 run supplies error_scales but not state_mean. Do not treat 014 mean as independently returned 015 evidence.',
        speed=speed,cases=cases,
        geometry_scope='Lat/lon errors use haversine metres. Bias decomposition is a local equirectangular approximation per target, descriptive only, not the official metric.',
        plot_target_policy='First stored target row in each predetermined case; all target rows retained in JSON. No target selected for performance.')
    (OUT/'speed_cases.json').write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    if plot:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        colors=dict(plain='#0072B2',identity='#E69F00',utility='#009E73',learned='#CC79A7')
        fig,axes=plt.subplots(4,2,figsize=(12,14),layout='constrained')
        for row,(case,arrays) in enumerate(plot_cases):
            b=arrays['plain']; p='m001__'; target=0; tid=b[p+'target_ids'][target]
            truth=b[p+'aligned_truth'][target]; ok=b[p+'aligned_truth_valid'][target,:,:2].all(-1)
            center=truth[ok,:2].mean(0); scale=np.array([111.195,111.195*np.cos(np.deg2rad(center[0]))])
            def xy(pos):
                v=(pos[:,:2]-center)*scale
                return v[:,1],v[:,0]
            ax=axes[row,0]
            for source,marker in [('9001','.'),('9002','x')]:
                nodes=np.flatnonzero((b['scene__identities']==tid)&(b['scene__sources']==source))
                for j,node in enumerate(nodes):
                    mask=b['scene__native_observed'][node]; vals=b['scene__native_values'][node,mask]
                    ax.scatter(*xy(vals),s=8,marker=marker,color='0.65' if source=='9001' else '0.35',alpha=.6,label=f'native {source}' if j==0 else None)
            ax.plot(*xy(truth[ok]),color='black',lw=2,label='reference')
            for arm,a in arrays.items():
                valid=a[p+'aligned_prediction_valid'][target,:,:2].all(-1)&ok
                ax.plot(*xy(a[p+'aligned_prediction'][target,valid]),color=colors[arm],lw=1.2,label=arm)
            ax.set(title=f"Scene {case['scene_id']}, target {tid} (oracle)",xlabel='East from reference mean (km)',ylabel='North from reference mean (km)')
            ax.grid(alpha=.2); ax.set_aspect('equal',adjustable='datalim')
            ax=axes[row,1]; times=b['scene__grid']/60
            mask=b[p+'aligned_truth_valid'][target,:,2]
            ax.plot(times[mask],truth[mask,2],color='black',lw=2,label='reference')
            for source,marker in [('9001','.'),('9002','x')]:
                for j,node in enumerate(np.flatnonzero((b['scene__identities']==tid)&(b['scene__sources']==source))):
                    m=b['scene__native_observed'][node]
                    ax.scatter(b['scene__native_times'][node,m]/60,b['scene__native_values'][node,m,2],s=8,marker=marker,color='0.65' if source=='9001' else '0.35',alpha=.6,label=f'native {source}' if j==0 else None)
            for arm,a in arrays.items():
                m=a[p+'aligned_prediction_valid'][target,:,2]&mask
                ax.plot(times[m],a[p+'aligned_prediction'][target,m,2],color=colors[arm],lw=1.2,label=arm)
            ax.set(title=f"Scene {case['scene_id']}: speed on reference-supported queries",xlabel='Relative time (minutes)',ylabel='Speed (native dataset units)');ax.grid(alpha=.2)
        handles,labels=axes[0,1].get_legend_handles_labels();fig.legend(handles,labels,loc='outside lower center',ncol=7)
        fig.suptitle('015 saved outputs: four fixed cases, same target/time, correct identity grouping\nCase selection: most improved / most worsened / lower median / smallest scene ID; first stored target shown',fontsize=12)
        fig.savefig(OUT/'fixed_cases.png',dpi=160);fig.savefig(OUT/'fixed_cases.svg');plt.close(fig)
        svg = OUT/'fixed_cases.svg'
        svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines()) + '\n')
    print(json.dumps({c['scene_id']:c['summary']['utility'] for c in cases},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--plot',action='store_true')
    main(parser.parse_args().plot)
