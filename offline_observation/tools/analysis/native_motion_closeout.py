"""Independent new-head forward/native arithmetic and complete output accounting."""
from __future__ import annotations
import argparse,sys
from pathlib import Path
if __package__ in (None,''):sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
import torch
from tools.analysis.history_followup_closeout import read,load,account,center,selected_weights,weighted_positions
from tools.analysis.native_motion_pilot import design
from association.evaluation import native_compatibility as nc

def numpy_correction(model,x):
    z=np.clip((x-model['mean'])/model['std'],-8,8).astype(np.float32)
    if model['kind']=='OLDINFO':z[:,12:]=0
    w={k:v.numpy() for k,v in model['weights'].items()}
    for layer in (0,2):z=z@w[f'net.{layer}.weight'].T+w[f'net.{layer}.bias'];z=z/(1+np.exp(-z))
    return (z@w['net.4.weight'].T+w['net.4.bias']).ravel()

def verify_training(run,r):
    parent=Path(r['parent_run']);rows=[];pop=[];manifest=read(Path(r['manifest']))
    for sid in r['fit_training_scene_ids']:
        z=load(parent/'private'/f'training-{sid}.npz');ex=load(run/'private'/f'training-native-{sid}.npz')
        for front in ('HARD','STRUCT'):
            use=np.zeros(z[front+'__labeled'].shape,bool);member=z[front+'__membership']
            for gid,label in enumerate(sorted(set(member.tolist()))):
                nodes=np.flatnonzero(member==label)
                if len(nodes)>1:use[nodes]=z[front+'__labeled'][nodes]&(z[front+'__count'][gid]==1)
            x=np.concatenate([z[front+'__features'],ex[front]],axis=-1)[use];w=.5*z[front+'__w'][use];rows.append((sid,str(manifest['components'][sid]),x,w));pop.append(dict(scene_id=sid,front=front,rows=int(use.sum()),weight=float(w.sum())))
    assert pop==read(run/'training-population.json');curves=read(run/'training-curves.json.gz');folds=read(run/'training-folds.json');assert len(folds)==13 and len(curves)==26
    for f in folds:
        use=[v for v in rows if v[1]!=f['held_component']];x=np.concatenate([v[2] for v in use]);w=np.concatenate([v[3] for v in use]);w/=w.sum();mean=w@x;std=np.sqrt(w@((x-mean)**2));std[std<1e-6]=1
        assert f['rows']==len(x) and f['training_scene_ids']==list(dict.fromkeys(v[0] for v in use));draws=load(run/'private'/f'sampling-{f["fold"]}.npz')['draws'];assert np.array_equal(draws,np.random.default_rng(20).integers(len(x),size=(256,4096)))
        models=torch.load(run/'private'/f'model-{f["fold"]}.pt',map_location='cpu',weights_only=False)
        for key,model in models.items():
            assert model['parameter_count']==1921 and model['updates']==256 and sum(v.numel() for v in model['weights'].values())==1921;np.testing.assert_allclose(model['mean'],mean,atol=1e-12);np.testing.assert_allclose(model['std'],std,atol=1e-12);assert model['curve']==next(c['curve'] for c in curves if c['fold']==f['fold'] and c['kind']==key)
    assert r['optimizer_updates']==6656 and r['neural_head_fits']==26
    return dict(fits=26,updates=6656,folds=13,rows=sum(x['rows'] for x in pop),scope='independent masks, populations, component exclusions, normalization, shared draws, saved heads and curves; no optimizer replay')

def forward_position(p,delta):
    d=np.linalg.norm(delta)/6371000.;b=np.arctan2(delta[1],delta[0]);lat,lon=np.deg2rad(p);phi=np.arcsin(np.clip(np.sin(lat)*np.cos(d)+np.cos(lat)*np.sin(d)*np.cos(b),-1,1));lam=lon+np.arctan2(np.sin(b)*np.sin(d)*np.cos(lat),np.cos(d)-np.sin(lat)*np.sin(phi));return np.array([np.rad2deg(phi),(np.rad2deg(lam)+180)%360-180])

def reconstruct(run,r,split,sid,front,m,events,groups,count,op,oraw,raw,policy):
    values=design(r['kind'])[0][front];base=oraw[values[1]+'_NN'];gap=0.;eligible=np.zeros(count.shape,bool);expected={v:base.copy() for v in values[2:]}
    if r['kind']=='feature':
        manifest=read(Path(r['manifest']));fold='leave_'+str(manifest['components'][sid]) if split=='T_core' else 'full';models=torch.load(run/'private'/f'model-{fold}.pt',map_location='cpu',weights_only=False);extra,_=nc.descriptors(m,events,groups,speed_unit='kn');x=np.concatenate([op[front+'__features'],extra],axis=-1);np.testing.assert_allclose(policy[front+'__features'],x,atol=1e-10)
        for prefix,key in zip(values[2:],('OLDINFO','NATIVE')):
            score=op[front+'__scores']+numpy_correction(models[key],x.reshape(-1,25)).reshape(m['support'].shape);gap=max(gap,float(np.max(np.abs(score-policy[prefix+'__scores']))));assert gap<1e-4
            for gid,nodes in enumerate(groups):
                active=count[gid]==1;w=selected_weights(policy[prefix+'__scores'],nodes,m['sources'],active);np.testing.assert_allclose(w,policy[prefix+'__weights'][nodes],atol=1e-12);expected[prefix][gid,active,:2]=weighted_positions(m['states'][nodes],w)[active]
        eligible=count==1
    elif r['kind']=='bridge':
        b=load(Path(r['bridge_run'])/'private'/f'raw-{split}-{sid}.npz');short='H' if front=='HARD' else 'S';assert np.array_equal(raw[values[2]+'_NN'],b[short+'B_NN']) and np.array_equal(raw[values[3]+'_NN'],b[short+'E_NN']);expected={v:b[short+'B_NN'].copy() for v in values[4:]}
        for gid,nodes in enumerate(groups):
            streams=[(nodes[m['sources'][nodes]==s],nc.stream(events,nodes[m['sources'][nodes]==s],speed_unit='kn')) for s in set(m['sources'][nodes].tolist())]
            for j in np.flatnonzero(count[gid]==0):
                q=m['query_seconds'][j];items=[]
                for own,s in streams:
                    l=np.searchsorted(s['t'],q,side='left')-1;rr=np.searchsorted(s['t'],q,side='right')
                    if l<0 or rr>=len(s['t']):continue
                    p0,p1=s['p'][l].copy(),s['p'][rr].copy();p1[1]=p0[1]+(p1[1]-p0[1]+180)%360-180;h=s['t'][rr]-s['t'][l];a=(q-s['t'][l])/h;scale=6371000*np.pi/180*np.array([1.,np.cos(np.deg2rad((p0[0]+p1[0])/2))]);valid=bool(s['known'][l] and s['known'][rr] and abs(scale[1])>1e-9);results=[]
                    if valid:
                        d0,d1=s['v'][l]/scale,s['v'][rr]/scale;sec=(p1-p0)/h;ratios=[np.divide(d,sec,out=np.zeros(2),where=sec!=0) for d in (d0,d1)];v0,v1=[np.clip(ratio,0,3)*sec for ratio in ratios]
                        A=2*a**3-3*a*a+1;B=a**3-2*a*a+a;C=-2*a**3+3*a*a;D=a**3-a*a
                        results=[p0+(p1-p0)*a,A*p0+B*h*d0+C*p1+D*h*d1,A*p0+B*h*v0+C*p1+D*h*v1];valid=all(np.isfinite(v).all() and abs(v[0])<=90 for v in results)
                        for v in results:v[1]=(v[1]+180)%360-180
                    items.append((float(op[front+'__scores'][own,j].min()),valid,results))
                if not items:continue
                eligible[gid,j]=True;best=min(v[0] for v in items);chosen=[v for v in items if v[0]==best]
                if all(v[1] for v in chosen):
                    for index,key in enumerate(values[4:]):expected[key][gid,j,:2]=center(np.stack([v[2][index] for v in chosen]),np.ones(len(chosen)))
    else:
        diagnostic=read(run/'private'/f'diagnostics-{split}-{sid}.json.gz')[front];models=diagnostic['models'];native=[]
        for node in range(len(m['states'])):
            l,rr=events['event_offsets'][node:node+2];t=events['event_times_seconds'][l:rr];p=events['event_values'][l:rr,:2];theta=np.deg2rad(events['event_values'][l:rr,3]);speed=events['event_values'][l:rr,2]*1852/3600;v=np.column_stack([speed*np.cos(theta),speed*np.sin(theta)]);valid=events['event_valid'][l:rr,2:4].all(1);native.append((t,p,v,valid))
            for key,model in models[node].items():
                vv=v[valid];dt=np.diff(t[valid]);mu=vv.mean(0) if key=='PG' else np.zeros(2);np.testing.assert_allclose(model['mean'],mu,atol=1e-12);grid=np.r_[0.,np.logspace(-6,2,65)/np.median(dt)];assert model['beta']==grid[model['grid_index']]
                if not model['constant_limit']:
                    decay=np.exp(-grid[:,None]*dt);q=np.empty_like(decay);q[0]=dt;q[1:]=-np.expm1(-2*grid[1:,None]*dt)/(2*grid[1:,None]);u=vv-mu;res=u[1:][None]-decay[:,:,None]*u[:-1][None];s2=np.mean(np.sum(res*res,axis=-1)/q,axis=1)/2;cost=len(dt)*np.log(np.maximum(s2,1e-24))+np.log(q).sum(1);assert cost[model['grid_index']]<=cost.min()+1e-10*max(1.,abs(float(cost.min())))
        for gid,nodes in enumerate(groups):
            first=min(native[i][0][0] for i in nodes);last=max(native[i][0][-1] for i in nodes)
            for j in np.flatnonzero((count[gid]==0)&((m['query_seconds']<first)|(m['query_seconds']>last))):
                eligible[gid,j]=True;chosen=nodes[op[values[1]+'__weights'][nodes,j]>0];mass=op[values[1]+'__weights'][chosen,j];outputs=[[] for _ in range(4)]
                for node in chosen:
                    t,p,v,valid=native[node];edge=0 if m['query_seconds'][j]<t[0] else -1;dt=m['query_seconds'][j]-t[edge]
                    if not models[node] or not valid[edge]:
                        for a in outputs:a.append(m['states'][node,j,:2])
                        continue
                    outputs[0].append(p[edge]);outputs[1].append(forward_position(p[edge],v[edge]*dt))
                    for ii,key in enumerate(('PD','PG'),start=2):
                        model=models[node][key];beta=model['beta'];h=dt if beta==0 else np.sign(dt)*(-np.expm1(-beta*abs(dt)))/beta;mu=np.array(model['mean']);outputs[ii].append(forward_position(p[edge],mu*dt+(v[edge]-mu)*h))
                for key,array in zip(values[2:],outputs):expected[key][gid,j,:2]=center(np.stack(array),mass)
    assert np.array_equal(eligible,policy[front+'__eligible'])
    for key,output in expected.items():np.testing.assert_allclose(raw[key+'_NN'][...,:2],output[...,:2],atol=1e-10,rtol=0)
    for key in values[2:]:assert np.array_equal(raw[key+'_NN'][~eligible],base[~eligible])
    return eligible,gap

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',type=Path,required=True);a=parser.parse_args();r=read(a.run/'run.json');assert r['status']=='COMPLETED';arms,methods,primary=design(r['kind'])
    from tools.analysis.association_context_closeout import closeout
    closeout(a.run);report=verify_training(a.run,r) if r['kind']=='feature' else dict(fits=0,updates=0,native_likelihood_fits=r['native_likelihood_fits']);account(a.run,a.run/'history-followup-accounting.json',(arms,primary,reconstruct,report))
