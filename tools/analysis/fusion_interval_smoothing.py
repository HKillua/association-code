"""C140: hold group and native events fixed while changing observation weights."""

import argparse
from collections import Counter
from pathlib import Path
import time

import numpy as np

from association.data.archive import MTADArchive
from tools.analysis import fusion_native_donor_evidence as c133
from tools.analysis.offline_association_tail import read_motion, summary_state, write


c130, c127 = c133.c130, c133.c130.c127
read = c127.read
MEAN, EQUAL, WEIGHTED = ARMS = ('MEAN', 'EQUAL', 'SOURCE_INTERVAL')
FEATURES = ('latitude', 'longitude', 'speed', 'heading', 'position')
REGIONS = ('all', 'exterior', 'interior_gap', 'one_source', 'multiple_sources', 'exact', 'source_conflict')
HUBER_M = 250.
KNOT_S = 60.
INTERVAL_S = 120.
STEPS = 5


def interval_weights(times, coordinates, sources):
    intervals = np.floor((times-times.min())/INTERVAL_S).astype(int)
    source = np.unique(sources, return_inverse=True)[1]
    _, distinct, row_count = np.unique(np.column_stack((source,times,coordinates)),
                                      axis=0, return_inverse=True, return_counts=True)
    _, index = np.unique(np.column_stack((source,intervals)), axis=0, return_inverse=True)
    row_mass = 1./row_count[distinct]
    weight = row_mass/np.bincount(index,weights=row_mass)[index]
    weight *= len(weight)/weight.sum()
    np.testing.assert_allclose(weight.sum(),len(weight),rtol=1e-12)
    return weight, len(np.unique(index)), len(times)-len(row_count)


def basis(knots, time):
    left = np.searchsorted(knots,time,side='right').clip(1,len(knots)-1)-1
    fraction = (time-knots[left])/(knots[left+1]-knots[left])
    return left, 1.-fraction, fraction


def smooth(times, positions, query, weights):
    span = float(times.max()-times.min())
    assert span>0. and len(times)==len(positions)==len(weights)
    knots = np.linspace(0.,span,min(257,int(np.ceil(span/KNOT_S))+1))
    centered = times-times.min()
    i, a, b = basis(knots,centered)
    n = len(knots)
    change = np.zeros((n-2,n))
    r = np.arange(n-2)
    step = KNOT_S/np.diff(knots)
    change[r,r] = step[:-1]
    change[r,r+1] = -step[:-1]-step[1:]
    change[r,r+2] = step[1:]
    penalty = change.T @ change
    strength = weights.copy()
    for count in range(STEPS):
        diagonal = np.bincount(i,weights=strength*a*a,minlength=n)
        diagonal += np.bincount(i+1,weights=strength*b*b,minlength=n)
        off = np.bincount(i,weights=strength*a*b,minlength=n-1)
        normal = penalty+np.diag(diagonal)+np.diag(off,1)+np.diag(off,-1)
        rhs = np.column_stack([np.bincount(i,weights=strength*a*y,minlength=n)
                           + np.bincount(i+1,weights=strength*b*y,minlength=n) for y in positions.T])
        state = np.linalg.solve(normal,rhs)
        if count+1<STEPS:
            residual = np.linalg.norm(a[:,None]*state[i]+b[:,None]*state[i+1]-positions,axis=1)
            strength = weights*HUBER_M/np.maximum(HUBER_M,residual)
    k, u, v = basis(knots,query-times.min())
    return u[:,None]*state[k]+v[:,None]*state[k+1], float(strength.sum())


def group_inputs(members, tracks, motion):
    native = [tracks[i] for i in members]
    times = [t['canonical_times'] for t in native]
    geo = np.concatenate([t['canonical_values'][:,:2] for t in native])
    offsets = np.r_[0,np.cumsum([len(t) for t in times])]
    longitude, positions, scale = c133._coordinates(geo,offsets)
    origin = longitude[0]-positions[0]/scale
    source = np.repeat(motion['sources'][members],np.diff(offsets))
    query = motion['query_seconds']
    span = np.concatenate(times)
    inside = (query>=span.min()) & (query<=span.max())
    raw_support = (query>=np.array([t[0] for t in times])[:,None]) & (
        query<=np.array([t[-1] for t in times])[:,None])
    np.testing.assert_array_equal(raw_support,motion['support'][members])
    source_count = np.zeros(len(query),dtype=int)
    for s in np.unique(motion['sources'][members]):
        source_count += raw_support[motion['sources'][members]==s].any(axis=0)
    exact = np.isin(query,span)
    conflict = np.zeros(len(query),bool)
    for k,n in enumerate(members):
        other = members[k+1:]
        across = other[motion['sources'][other]!=motion['sources'][n]]
        if len(across):
            gap = c130.DISTANCE(motion['states'][n,:,:2],motion['states'][across,:,:2])
            conflict |= ((gap>1000.) & motion['support'][across] & motion['support'][n]).any(axis=0)
    flags = np.stack((~inside, inside & (source_count==0), source_count==1, source_count>=2, exact, conflict),axis=-1)
    np.testing.assert_array_equal(flags[:,:4].sum(1),1)
    return span,geo,positions,source,origin,scale,inside,flags


def freeze(args, scenes, data):
    out, runs = args.output,args.runs
    c98 = c127.c98
    private = out/'private'
    private.mkdir()
    rows = {}
    with MTADArchive(read(runs/c98.DATA/'run.json')['archive']) as archive:
        c133.validate_measurement_archive(runs,archive)
        for sid in scenes:
            motion = read_motion(data[sid])
            tracks = c133._canonical_tracks(archive._rows(sid,'measurements'),sid,motion['node_ids'],motion['sources'])
            np.testing.assert_array_equal([len(t['canonical_times']) for t in tracks],motion['native_count'])
            with np.load(runs/c127.c124.C121/'private'/f'{sid}.npz',allow_pickle=False) as base:
                np.testing.assert_array_equal(base['node_ids'],motion['node_ids'])
                np.testing.assert_array_equal(base['query'],motion['query_seconds'])
                partition = base['current']
                mean = base['MEAN']
                np.testing.assert_array_equal(mean,c127.c107.mean_groups(motion,partition))
                groups = c98.group_nodes(partition)
                estimates = np.repeat(mean[None],len(ARMS),axis=0)
                flags = np.zeros((*mean.shape[:2],len(REGIONS)-1),bool)
                counts = Counter(groups=len(groups),native_rows=int(motion['native_count'].sum()))
                for g,members in enumerate(groups):
                    span,geo,xy,source,origin,scale,inside,flags[g] = group_inputs(members,tracks,motion)
                    w,cells,duplicates = interval_weights(span,geo,source)
                    counts.update(source_intervals=cells, duplicate_rows=duplicates,
                                  multisource_groups=int(len(np.unique(source))>1))
                    for j,weights in enumerate((np.ones(len(w)),w),start=1):
                        fit, effective = smooth(span,xy,motion['query_seconds'][inside],weights)
                        predicted = fit/scale+origin
                        predicted[:,1] = (predicted[:,1]+180.)%360.-180.
                        assert np.isfinite(predicted).all() and (np.abs(predicted[:,0])<=90.).all()
                        estimates[j,g,inside,:2] = predicted
                        counts['effective_weight_'+ARMS[j]] += effective
                np.testing.assert_array_equal(estimates[:,flags[...,0]],np.repeat(mean[flags[...,0]][None],len(ARMS),axis=0))
                np.testing.assert_array_equal(estimates[...,2:],np.repeat(mean[None,...,2:],len(ARMS),axis=0))
                np.savez_compressed(private/f'{sid}.npz', partition=partition, estimates=estimates, flags=flags)
                counts.update({k:int(flags[...,j].sum()) for j,k in enumerate(REGIONS[1:])})
                rows[sid] = dict(counts)
        assert [r['scene_id'] for r in archive.members_read]==scenes
        assert all(r['kind']=='measurements' for r in archive.members_read)
    write(out/'freeze.json',dict(scenes=len(scenes),archive_measurements=len(scenes),evaluation_before_freeze=False,
        counts=dict(sum((Counter(row) for row in rows.values()),Counter()))))
    return rows


def score(motion, frozen, truth):
    np.testing.assert_array_equal(motion['node_ids'],truth['node_ids'])
    np.testing.assert_array_equal(motion['query_seconds'],truth['query_seconds'])
    groups = c127.c98.group_nodes(frozen['partition'])
    ids = truth['identities']
    first = {}
    for g,members in enumerate(groups):
        first.setdefault(ids[members[0]],g)
    matched = np.array([first.get(t,-1) for t in truth['target_ids']],dtype=int)
    visible = matched>=0
    target = truth['target_states'][visible]
    valid = truth['target_valid'][visible]
    pred = frozen['estimates'][:,matched[visible]]
    mask = frozen['flags'][matched[visible]]
    lat = abs(pred[...,0]-target[...,0])
    lon = abs((pred[...,1]-target[...,1]+180.)%360.-180.)
    speed = abs(pred[...,2]/(3600./1852.)-target[...,2])
    heading = abs((pred[...,3]-target[...,3]+180.)%360.-180.)
    distance = c130.DISTANCE(pred[...,:2],target[...,:2])
    errors = np.stack((lat,lon,speed,heading,distance),axis=-1)
    legal = np.concatenate((valid,valid[...,:2].all(-1,keepdims=True)),axis=-1)
    errors = np.where(legal[None],errors,np.nan)
    usable = legal.any(-1)
    return errors[:,usable].transpose(1,0,2), mask[usable], dict(targets=len(matched),matched=int(visible.sum()),
        groups=len(groups),duplicate_groups=len(groups)-len(first))


def statistics(errors, flags, tail):
    result = {}
    for r,region in enumerate(REGIONS):
        x = errors if r==0 else errors[flags[:,r-1]]
        z = x[np.isfinite(x[:,:,4]).all(1),:,4]
        metrics = {arm:{feature:c127.c107.attribute_stats(x[np.isfinite(x[:,j,k]),j,k])
                        for k,feature in enumerate(FEATURES)} for j,arm in enumerate(ARMS)}
        changes = {}
        for a,b in ((1,0),(2,0),(2,1)):
            delta = z[:,a]-z[:,b]
            changes[ARMS[a]+'/'+ARMS[b]] = dict(harm250=int((delta>250.).sum()),
                non_tail_harm250=int(((delta>250.) & (z[:,b]<tail)).sum()),
                benefit250=int((delta<-250.).sum()), new_tail=int(((z[:,b]<tail)&(z[:,a]>=tail)).sum()),
                exit_tail=int(((z[:,b]>=tail)&(z[:,a]<tail)).sum()), net_delta_m=float(delta.sum()))
        result[region] = dict(metrics=metrics,changes=changes)
    return result


def run(args):
    runs, out, c98 = args.runs,args.output,c127.c98
    scenes = read(runs/c98.PREP/'frozen-split.json')['held']
    manifest = read(args.manifest)
    assert len(set(scenes))==177 and set(scenes)<=set(manifest['splits']['T_core'])
    data = read(runs/c98.DATA/'private/index.json')['T_core']
    counts = freeze(args,scenes,data)
    prior = read(runs/c127.c124.C121/'summary.json')
    scene_rows, errors, flags, scene_index = [],[],[],[]
    for i,sid in enumerate(scenes):
        motion = read_motion(data[sid])
        with np.load(out/'private'/f'{sid}.npz',allow_pickle=False) as cached, \
             np.load(data[sid]['evaluation'],allow_pickle=False) as truth:
            values,mask,coverage = score(motion,cached,truth)
        errors.append(values)
        flags.append(mask)
        scene_index.extend([i]*len(values))
        scene_rows.append(dict(scene=sid,component=manifest['components'][sid],input=counts[sid],**coverage))
    errors, flags, scene_index = np.concatenate(errors),np.concatenate(flags),np.array(scene_index)
    components = np.array([s['component'] for s in scene_rows])
    expected = np.isfinite(errors[:,0]).sum(0)
    np.testing.assert_array_equal(expected,[123051,123056,123044,122934,123051])
    for j in (1,2):
        np.testing.assert_array_equal(np.isfinite(errors[:,j]),np.isfinite(errors[:,0]))
        np.testing.assert_array_equal(errors[:,j,2:4],errors[:,0,2:4])
    tail = float(np.nanquantile(errors[:,0,4],.99))
    result = statistics(errors,flags,tail)
    np.testing.assert_allclose(result['all']['metrics'][MEAN]['position']['mae'],1772.1818826547121,rtol=0.,atol=1e-7)
    np.testing.assert_allclose(tail,17985.623358578603,rtol=0.,atol=1e-7)
    assert prior['scenes']==177
    assert len(np.unique(components))==4
    for i,row in enumerate(scene_rows):
        use = scene_index==i
        row['result'] = statistics(errors[use],flags[use],tail)
    by_component = {}
    leave = {}
    for key in np.unique(components):
        use = components[scene_index]==key
        by_component[key] = statistics(errors[use],flags[use],tail)
        leave[key] = statistics(errors[~use],flags[~use],tail)
    write(out/'summary.json',dict(scenes=len(scenes),coverage=dict(zip(FEATURES,expected.tolist(),strict=True)),
        baseline_tail_m=tail,results=result,components=by_component,leave_one_component_out=leave))
    write(out/'scenes.json',scene_rows)
    np.savez_compressed(out/'public-errors.npz',errors=errors,flags=flags,scene_index=scene_index,
        scene_ids=scenes,arms=ARMS,features=FEATURES,regions=REGIONS[1:])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('runs','manifest','output'):
        parser.add_argument(f'--{key}',type=Path,required=True)
    parser.add_argument('--time-policy',choices=('advisory',),required=True)
    args = parser.parse_args()
    args.output.mkdir()
    start = time.monotonic()
    try:
        run(args)
    except Exception as error:
        summary_state(args.output,start,status='FAILED',error=repr(error))
        raise
    summary_state(args.output,start,status='COMPLETED',native_archive_reads=177,optimizer_updates=0,
                  model_forwards=0,test_opened=False)


if __name__=='__main__':
    main()
