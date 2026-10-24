"""087 frozen temporal correspondence and strong-control cache comparison."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch

from association.evaluation.helper_sequence_position import helper_population, BUCKETS, POPULATIONS
from association.evaluation.sequence_position import REGIONS
from association.evaluation.support_gate_objectives import _read
from association.evaluation.system_attribution import _same
from association.training.helper_temporal import load_checkpoint
from association.utils.runtime import read_json, write_json, write_json_gzip
from tools.analysis.compare_position_benchmarks import _metadata, _pool, _sequence_sampling, CONTROLS
from tools.analysis.compare_position_harm import scene_pair, harm_statistics
from tools.experiments.helper_temporal_fit import CONFIGS, ROOT, bind_reference, frozen_configuration
from tools.analysis.compare_helper_sequence import helper_metadata as pre_metadata
from association.models.helper_temporal import permutation_plan, validate_plan
from association.training.sequence_position import load_checkpoint as sequence_checkpoint
from copy import deepcopy


def recorded_plan(value, helpers, queries):
    array=np.asarray(value)
    # JSON has no integer dtype for an empty Q-by-zero list.
    if not array.size: array=array.astype(np.int64)
    return validate_plan(array,helpers,queries)


def temporal_metadata(path):
    path = Path(path).resolve()
    run, summary, fit = [read_json(path/name) for name in ('run.json', 'summary.json', 'fit/run.json')]
    config = run['configuration']
    mode = config['model']['correspondence']
    frozen_configuration(config)
    steps = config['training']['updates']
    for record in (run, summary, fit):
        if (record.get('status') != 'COMPLETED' or record.get('configuration') != config
                or record.get('optimizer_updates') != steps or record.get('test_opened') is not False):
            raise ValueError('087 requires completed actual updates/configuration')
    if (run.get('completed_endpoints') != ['initial', 'final'] or run.get('validation_selection') is not False
            or summary.get('source_system') != run['source_system']
            or summary.get('paired_control') != run.get('paired_control') or fit.get('final_checkpoint') != 'final.pt'):
        raise ValueError('087 final endpoint/provenance is incomplete')
    manifest = read_json(ROOT/'configs/data/mtad-level1-scene3000.json')
    if fit.get('training_scene_ids') != manifest['splits']['T_core']:
        raise ValueError('087 complete training population differs')
    bind_reference(run['paired_control'], config, run['source_system'], manifest)
    for label, step, filename in (('initial', 0, 'step-0.pt'), ('final', steps, 'final.pt')):
        with torch.random.fork_rng(devices=[]):
            model, checkpoint = load_checkpoint(path/'fit'/filename, device='cpu')
        del model
        if (checkpoint['configuration'] != config or checkpoint['step'] != step or checkpoint['actual_updates'] != step):
            raise ValueError('087 actual endpoint checkpoint differs')
        final = read_json(path/label/'summary.json')
        for record in (read_json(path/label/'run.json'), final):
            provenance = record.get('checkpoint', {})
            if (record.get('status') != 'COMPLETED' or record.get('configuration') != config
                    or record.get('source_system') != run['source_system'] or record.get('device') != 'cpu'
                    or record.get('optimizer_updates') != 0 or record.get('test_opened') is not False
                    or record.get('groups_recomputed') is not False or record.get('threshold_reselected') is not False
                    or provenance.get('step') != step or provenance.get('actual_updates') != step
                    or provenance.get('configured_final_updates') != steps
                    or Path(provenance.get('path', '')).resolve() != path/'fit'/filename):
                raise ValueError('087 endpoint output does not bind actual checkpoint')
        if final['scene_ids'] != manifest['splits']['V_select']:
            raise ValueError('087 complete validation population differs')
        for directory in ('predictions', 'system-predictions'):
            if {p.name for p in (path/label/directory).glob('*.npz')} != {f'states-{s}.npz' for s in final['scene_ids']}:
                raise ValueError('087 endpoint cache inventory differs')
    components = {sid: str(cid) for cid, group in final['components'].items() for sid in group['scene_ids']}
    if components != {sid: str(manifest['components'][sid]) for sid in final['scene_ids']}:
        raise ValueError('087 component registry differs')
    return mode, dict(path=path, method='learned_sequence', configuration=config,
        source_system=run['source_system'], scene_ids=final['scene_ids'], components=components, summary=final)


def verify_pairing(arms):
    original,old_rows,old_exposure=_sequence_sampling(arms['linear']['path']/'fit')
    canonical=lambda rows:[{k:r[k] for k in ('step','mode','batch')} for r in rows]
    with torch.random.fork_rng(devices=[]): _,old=sequence_checkpoint(arms['linear']['path']/'fit/step-0.pt')
    reports={}; snapshots={}
    for name,arm in arms.items():
        if name=='linear': continue
        fit,rows,exposure=_sequence_sampling(arm['path']/'fit')
        if canonical(rows)!=canonical(old_rows) or exposure!=old_exposure or fit['training_scene_ids']!=original['training_scene_ids']:
            raise ValueError('087 original successful batches/population differ from072C')
        with torch.random.fork_rng(devices=[]): _,saved=sequence_checkpoint(arm['path']/'fit/step-0.pt')
        for key,value in old['model'].items():
            candidate=saved['model'][key]
            if key=='input_projection.0.weight':
                if torch.count_nonzero(candidate[:,76:]): raise ValueError('087 added projection must start zero')
                candidate=candidate[:,:76]
            if not torch.equal(candidate,value): raise ValueError('087 common initial tensor differs/'+key)
        if saved['numpy_rng']!=old['numpy_rng']: raise ValueError('087 original sampler RNG differs')
        if name=='pre_pool':
            reports[name]=dict(actual_batches_exact=True,common_initial_parameters_exact=True)
            continue
        rng=np.random.default_rng(fit['configuration']['training']['seed']+87000)
        if saved['temporal_route_rng']!=rng.bit_generator.state: raise ValueError('087 initial route RNG differs')
        route_exposure=dict(sampled_anchors=0,multihelper_anchors=0,helper_query_slots=0,
                            planned_changed_slots=0,applied_changed_slots=0)
        for row in rows:
            if row.get('temporal_rng_before')!=rng.bit_generator.state: raise ValueError('087 route RNG discontinuity')
            plans=row.get('temporal_route_batch',[])
            applications=row.get('temporal_applied_batch',[])
            if len(plans)!=len(row['batch']) or len(applications)!=len(plans): raise ValueError('087 route/sample count differs')
            for sample,plan,application in zip(row['batch'],plans,applications):
                expected=permutation_plan(len(sample['members'])-1,fit['configuration']['data']['length'],rng)
                actual=recorded_plan(plan,*expected.shape[::-1])
                if not np.array_equal(actual,expected): raise ValueError('087 actual temporal route does not replay')
                applied=recorded_plan(application,*expected.shape[::-1])
                identity=np.broadcast_to(np.arange(expected.shape[1]),expected.shape)
                if not np.array_equal(applied,expected if name=='shuffled' else identity):
                    raise ValueError('087 applied temporal route differs from declared mode')
                route_exposure['sampled_anchors']+=1
                route_exposure['multihelper_anchors']+=int(expected.shape[1]>=2)
                route_exposure['helper_query_slots']+=expected.size
                route_exposure['planned_changed_slots']+=int((expected!=identity).sum())
                route_exposure['applied_changed_slots']+=int((applied!=identity).sum())
            if row.get('temporal_rng_after')!=rng.bit_generator.state: raise ValueError('087 final batch RNG differs')
        if fit.get('temporal_routing_exposure')!=route_exposure:
            raise ValueError('087 successful-update route exposure does not close')
        with torch.random.fork_rng(devices=[]): _,final=sequence_checkpoint(arm['path']/'fit/final.pt')
        if final['temporal_route_rng']!=rng.bit_generator.state: raise ValueError('087 final route checkpoint differs')
        snapshots[name]=saved['model']
        reports[name]=dict(matched_updates=len(rows),actual_batches_exact=True,common_initial_parameters_exact=True,
            route_rng_replayed=True,applied_routes_verified=True,routing_exposure=route_exposure,
            helper_gradients=fit['helper_gradients'],gradient_clipping=fit['gradient_clipping'])
    values=list(snapshots.values())
    if any(set(x)!=set(values[0]) or any(not torch.equal(x[k],values[0][k]) for k in x) for x in values[1:]):
        raise ValueError('087 A/B all initial tensors must be identical')
    return dict(status='PAIR_VERIFIED',matched_updates=len(old_rows),matched_anchors=old_exposure['sampled_anchors'],
        arms=reports,shared_added_initial_parameters_exact=True)


def replay_routes(arm):
    config=arm['configuration']; seed=config['training']['seed']+87100
    mode=config['model']['correspondence']; endpoints={}
    for label in ('initial','final'):
        rng=np.random.default_rng(seed)
        exposure=dict(group_count=0,multihelper_groups=0,helper_query_slots=0,
                      planned_changed_slots=0,applied_changed_slots=0)
        for sid in arm['scene_ids']:
            a=_read(arm['path']/label/'predictions'/f'states-{sid}.npz')
            route=a['group_helper_query_route_plan']; offsets=a['group_helper_offsets']
            applied=a['group_helper_query_applied_route']
            if route.shape != (int(offsets[-1]),a['support'].shape[1]) or applied.shape!=route.shape:
                raise ValueError('087 V route inventory differs')
            for lo,hi in zip(offsets[:-1],offsets[1:]):
                expected=permutation_plan(int(hi-lo),route.shape[1],rng)
                if not np.array_equal(validate_plan(route[lo:hi].T,int(hi-lo),route.shape[1]),expected): raise ValueError('087 V route replay differs')
                actual=validate_plan(applied[lo:hi].T,int(hi-lo),route.shape[1])
                identity=np.broadcast_to(np.arange(int(hi-lo)),expected.shape)
                if not np.array_equal(actual,expected if mode=='shuffled' else identity):
                    raise ValueError('087 V applied route differs from declared mode')
                exposure['group_count']+=1;exposure['multihelper_groups']+=int(hi-lo>=2)
                exposure['helper_query_slots']+=expected.size
                exposure['planned_changed_slots']+=int((expected!=identity).sum())
                exposure['applied_changed_slots']+=int((actual!=identity).sum())
        expected_record=dict(evaluation_seed=seed,correspondence=mode,exposure=exposure,final_rng=rng.bit_generator.state)
        for filename in ('run.json','summary.json'):
            if read_json(arm['path']/label/filename).get('routing')!=expected_record:
                raise ValueError('087 V route exposure/RNG record does not close')
        endpoints[label]=expected_record
    if endpoints['initial']!=endpoints['final']: raise ValueError('087 V endpoint route plans differ')
    return endpoints


def helper_cohorts(fragment, system):
    counts = fragment['group_supported_helper_count']
    t = len(system['anchor_target_ids'])
    selected = system['anchor_aligned_group_indices'][:t]
    shape = system['anchor_aligned_truth_valid'][:t].shape[:2]
    target_counts = np.zeros(shape, np.int64)
    exists = selected >= 0
    target_counts[exists] = counts[selected[exists]]
    return {name: exists[:, None] & test for name, test in zip(
        BUCKETS, (target_counts == 0, target_counts == 1, target_counts >= 2))}


def summarize(rows, arrays, methods):
    result = _pool(rows, arrays, methods)
    result['relative_harm'] = {}
    for region in REGIONS:
        use = lambda a: a['region_'+region] & np.isfinite(a['anchor_position_error'])
        values = {name: np.concatenate([a[name+'_position_error'][use(a)] for a in arrays]) for name in methods}
        result['relative_harm'][region] = {name: harm_statistics(values[name], values['fixed_prior095'])
                                          for name in methods if name not in (*CONTROLS, 'fixed_prior095')}
        pairs = [(name, 'linear') for name in methods if name not in (*CONTROLS, 'fixed_prior095', 'linear')]
        pairs += [(a, b) for a, b in (('preserved', 'shuffled'),)
                  if a in values and b in values]
        for new, old in pairs:
            result['relative_harm'][region][new+'__minus__'+old] = harm_statistics(values[new], values[old])
    result['helper_cohorts'] = {}
    for cohort in BUCKETS:
        local = [dict(a, region_all=a['helper_cohort_'+cohort]) for a in arrays]
        result['helper_cohorts'][cohort] = _pool(rows, local, methods)['position_weightings']['all']
    result['support_helper_cross']=cross_table(arrays,methods)
    return result


def cross_table(arrays,methods):
    cells={}; seen=[np.zeros_like(a['anchor_position_error'],dtype=np.int8) for a in arrays]
    valid=[np.isfinite(a['anchor_position_error']) for a in arrays]
    global_net={name:sum(float((a[name+'_position_error'][v]-a['linear_position_error'][v]).sum()) for a,v in zip(arrays,valid)) for name in methods}
    denominator=sum(int(v.sum()) for v in valid)
    for region in REGIONS[1:]:
        for cohort in BUCKETS:
            masks=[v&a['region_'+region]&a['helper_cohort_'+cohort] for a,v in zip(arrays,valid)]
            for total,mask in zip(seen,masks): total+=mask
            errors={name:np.concatenate([a[name+'_position_error'][mask] for a,mask in zip(arrays,masks)]) for name in methods}
            count=len(errors['linear']); cell={'point_count':count,'methods':{}}
            for name,e in errors.items():
                delta=float((e-errors['linear']).sum()); baseline=errors['fixed_prior095']
                cell['methods'][name]=dict(error_sum_m=float(e.sum()),squared_error_sum_m2=float(np.square(e).sum()),
                    mae=float(e.mean()) if count else None,mse=float(np.square(e).mean()) if count else None,
                    improvement_sum_m=float(np.maximum(baseline-e,0).sum()),harm_sum_m=float(np.maximum(e-baseline,0).sum()),
                    signed_delta_vs_linear_m=delta,global_mae_contribution_m=delta/denominator if denominator else None,
                    share_of_global_net_delta=delta/global_net[name] if global_net[name]!=0 else None)
            cells[region+'__'+cohort]=cell
    if any(not np.array_equal(total,v.astype(np.int8)) for total,v in zip(seen,valid)):
        raise ValueError('087 support/helper cells do not partition complete valid population')
    return dict(cells=cells,point_count=denominator,interpretation='Disjoint descriptive error accounting, not causal contribution.')


def scientific_screen(summary,rows,name):
    from tools.analysis.compare_helper_dropout import scientific_screen as screen
    def convert(item):
        result=dict(item); result['position_weightings']=dict(item['position_weightings'])
        result['position_weightings']['all']=dict(item['position_weightings']['all'])
        result['position_weightings']['all']['dropout']=item['position_weightings']['all'][name]
        return result
    value=convert(summary);value['components']={k:convert(v) for k,v in summary['components'].items()}
    value['remaining_components']=convert(summary['remaining_components']) if summary['remaining_components'] else None
    answer=screen(value,[convert(r) for r in rows]);answer['interpretation']='Frozen strong072C screening; single seed only licenses matched confirmation, never automatic method promotion.'
    return answer


def run(args):
    if args.cpu_threads != 4: raise ValueError('087 comparison fixes CPU4')
    torch.set_num_threads(4)
    paths={name:Path(value).resolve() for name,value in (('linear',args.linear_run),('pre_pool',args.pre_pool_run),
        ('shuffled',args.shuffled_run),('preserved',args.preserved_run)) if value}
    if not {'linear','pre_pool'} < set(paths): raise ValueError('087 requires references and at least one temporal arm')
    output=Path(args.output).resolve()
    if output.exists(): raise FileExistsError('087 comparison exists; no retry')
    for path in paths.values():
        source=Path(read_json(path/'run.json')['source_system']).resolve().parent
        # Only this candidate's exact comparison child can be embedded.
        embedded = output==path/'reference-comparison' and path in [paths.get('shuffled'),paths.get('preserved')]
        if any(output==p or output in p.parents or (p in output.parents and not (embedded and p==path)) for p in (path,source)):
            raise ValueError('087 comparison overlaps immutable source')
    output.mkdir(parents=True)
    state = dict(status='RUNNING', completed_scenes=0,
        model_forwards=0, optimizer_updates=0, test_opened=False)
    write_json(output/'run.json', state)
    try:
        arms={'linear':_metadata(paths['linear'],'072C_transformer4000')}
        mode,arms['pre_pool']=pre_metadata(paths['pre_pool'])
        if mode!='pre_pool': raise ValueError('087 requires078B pre_pool')
        route_audits={}
        for name,path in paths.items():
            if name in ('linear','pre_pool'): continue
            mode,arms[name]=temporal_metadata(path)
            if mode!=name: raise ValueError('087 temporal mode differs')
            route_audits[name]=replay_routes(arms[name])
        first = arms['linear']
        if len(arms) < 2 or any(arm[key] != first[key] for arm in arms.values()
            for key in ('scene_ids', 'source_system', 'components')):
            raise ValueError('087 requires shared source/population/components and at least one helper arm')
        pairing = verify_pairing(arms)
        methods = (*CONTROLS, 'fixed_prior095', *arms)
        rows, arrays = [], []
        helper_populations = []
        for sid in first['scene_ids']:
            row, data = scene_pair(arms, sid)
            row['component_id'] = first['components'][sid]
            ref = None
            for name in arms:
                if name == 'linear':
                    continue
                fragment = _read(arms[name]['path']/'final/predictions'/f'states-{sid}.npz')
                system = _read(arms[name]['path']/'final/system-predictions'/f'states-{sid}.npz')
                population = helper_population(fragment, system)
                if ref is None:
                    ref = fragment
                    data.update({'helper_cohort_'+k: v for k, v in helper_cohorts(fragment, system).items()})
                    helper_populations.append(population)
                else:
                    for key in ('group_supported_helper_count', 'group_helper_offsets', 'group_helper_node_indices'):
                        _same(fragment[key], ref[key], '087 paired helper membership/'+key)
            rows.append(row); arrays.append(data)
            state['completed_scenes'] = len(rows)
            write_json(output/'run.json', state)
        summary = summarize(rows, arrays, methods)
        for name, arm in arms.items():
            actual = summary['position_weightings']['all'][name]['point_equal']
            expected = arm['summary']['methods']['learned_sequence']['full_target_metrics']['all']['position']
            if actual['unit_count'] != expected['intersection_count'] or any(
                not np.isclose(actual[k], expected[k], rtol=1e-10, atol=1e-9) for k in ('mae', 'rmse')):
                raise ValueError('087 cache does not reproduce producer physical metrics/'+name)
        scene_rows = [dict(summarize([r], [a], methods), component_id=r['component_id']) for r, a in zip(rows, arrays)]
        components = {}
        for cid in sorted(set(first['components'].values())):
            ids = [i for i, row in enumerate(rows) if row['component_id'] == cid]
            components[cid] = summarize([rows[i] for i in ids], [arrays[i] for i in ids], methods)
        largest = min(components, key=lambda cid: (-components[cid]['position_population']['covered_target_points'], cid))
        rest = [i for i, row in enumerate(rows) if row['component_id'] != largest]
        population = {name: dict(total=sum(row[name]['total'] for row in helper_populations), counts={
            bucket: sum(row[name]['counts'][bucket] for row in helper_populations) for bucket in BUCKETS})
            for name in POPULATIONS}
        summary.update(status='COMPLETED', pairing=pairing, routing=route_audits, methods=list(methods), components=components,
            largest_component_id=largest, remaining_components=summarize([rows[i] for i in rest],
            [arrays[i] for i in rest], methods) if rest else None, helper_population=population,
            inputs={name: str(arm['path']) for name, arm in arms.items()}, single_seed_screen=True,
            single_seed_comparison=True, complete_pair=set(arms)=={'linear','pre_pool','shuffled','preserved'},
            model_forwards=0, optimizer_updates=0,
            test_opened=False, old066_acceptance='REJECTED',
            interpretation='One seed and reused V/shared identities; temporal correspondence destruction is not identity supervision or a sufficient competitiveness control.')
        summary['scientific_screen']={name:scientific_screen(summary,scene_rows,name) for name in ('shuffled','preserved') if name in arms}
        write_json_gzip(output/'summary.json.gz', summary)
        write_json_gzip(output/'scene-summary.json.gz', scene_rows)
        lines = ['#087 temporal helper comparison', '', '| Method | MAE m | RMSE m | P95 m | P99 m |',
                 '|---|---:|---:|---:|---:|']
        for name in methods:
            value = summary['position_weightings']['all'][name]['point_equal']
            lines.append('| '+name+' | '+' | '.join(f'{value[k]:.4f}' for k in ('mae', 'rmse', 'p95', 'p99'))+' |')
        lines += ['', 'All outputs, duplicate groups and valid target masks retained. H/I and fixed100/1000/10000m harm counts, four weightings, support/helper strata and components are in JSON.',
                  'This table contains one declared seed/profile only. No automatic promotion or retry.']
        (output/'report.md').write_text('\n'.join(lines)+'\n')
        state.update(status='COMPLETED', population=summary['population'])
        write_json(output/'run.json', state)
        return summary
    except BaseException as error:
        state.update(status='FAILED', error=f'{type(error).__name__}: {error}')
        write_json(output/'run.json', state)
        raise


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('linear-run','pre-pool-run','output'): parser.add_argument('--'+key,required=True)
    for key in ('shuffled-run','preserved-run'): parser.add_argument('--'+key)
    parser.add_argument('--cpu-threads',type=int,default=4)
    print(run(parser.parse_args(argv))['status'],flush=True)
    return 0


if __name__=='__main__': raise SystemExit(main())
