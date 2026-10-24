"""083 fixed-endpoint comparison and exact T-only dropout draw replay.

Uses saved predictions and checkpoints for binding only; no forward or training.
No new score, grouping, rule or validation-based parameter selection.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
from pathlib import Path
import sys
if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import torch
from association.evaluation.support_gate_objectives import _read
from association.training.helper_dropout import SCOPE, drop_helper_fragments, empty_exposure, commit_exposure
from association.training.sequence_position import load_checkpoint
from association.utils.runtime import read_json, write_json, write_json_gzip
from tools.analysis.compare_position_benchmarks import _metadata, _sequence_sampling, CONTROLS
from tools.analysis.compare_position_harm import scene_pair
from tools.analysis.compare_helper_sequence import summarize


def verify_dropout_pair(new_fit, old_fit):
    new, nr, ne = _sequence_sampling(new_fit)
    old, rows, oe = _sequence_sampling(old_fit)
    expected = deepcopy(old['configuration'])
    expected.update(scope=SCOPE, augmentation=dict(kind='helper_fragment_dropout', probability=.5,
        seed=expected['training']['seed']+83000))
    if (new['configuration'] != expected or old['configuration']['scope'] != 'sequence-position-v1'
            or old['configuration']['position_loss'] != 'linear'
            or new['training_scene_ids'] != old['training_scene_ids']
            or ne != oe or new.get('augmentation_applied_only_to_training') is not True):
        raise ValueError('083 requires identical original linear configuration and exposure except T augmentation')
    if [{k: x[k] for k in ('step','mode','batch')} for x in nr] != [
            {k: x[k] for k in ('step','mode','batch')} for x in rows]:
        raise ValueError('083 original pre-drop sampling differs')
    initial = []
    for path, config in ((new_fit, expected), (old_fit, old['configuration'])):
        with torch.random.fork_rng(devices=[]):
            model, cp = load_checkpoint(Path(path)/'step-0.pt', device='cpu')
        del model
        if cp['configuration'] != config or cp['step'] != 0 or cp['actual_updates'] != 0:
            raise ValueError('083 initial checkpoint not bound')
        initial.append(cp)
    a, b = initial
    if (set(a['model']) != set(b['model']) or any(not torch.equal(a['model'][k], b['model'][k]) for k in a['model'])
            or a['numpy_rng'] != b['numpy_rng'] or not torch.equal(a['torch_rng'], b['torch_rng'])):
        raise ValueError('083 initial model or original RNG changed')
    rng = np.random.default_rng(expected['augmentation']['seed'])
    if a['augmentation_rng'] != rng.bit_generator.state:
        raise ValueError('083 initial augmentation RNG differs')
    exposure = empty_exposure()
    for row in nr:
        aug = row.get('augmentation_batch', [])
        if len(aug) != len(row['batch']) or row.get('augmentation_rng_before') != rng.bit_generator.state:
            raise ValueError('083 augmentation batch or RNG discontinuity')
        for plan, audit in zip(row['batch'], aug):
            retained, dropped = drop_helper_fragments(plan['members'], rng, .5)
            if (audit['original_members'] != plan['members'] or audit['retained_members'] != retained
                    or audit['drop_mask'] != dropped or any(type(x) is not bool for x in audit['drop_mask'])
                    or audit['original_helper_count'] != len(plan['members'])-1
                    or audit['retained_helper_count'] != len(retained)-1 or audit['dropped_helpers'] != sum(dropped)):
                raise ValueError('083 actual retained members do not replay frozen independent draws')
            for prefix, members in (('original',plan['members']),('retained',retained)):
                identity=audit[prefix+'_identity']
                if (set(identity) != {'correct_helpers','wrong_helpers'} or any(type(v) is not int or v < 0 for v in identity.values())
                        or sum(identity.values()) != len(members)-1):
                    raise ValueError('083 identity audit inconsistent')
                for pop, total in (('all_queries',expected['data']['length']),('valid_position_queries',plan['valid_position_points'])):
                    count = audit[prefix+'_support'][pop]
                    if (set(count) != {'zero','one','multiple'} or any(type(v) is not int or v < 0 for v in count.values())
                            or sum(count.values()) != total):
                        raise ValueError('083 helper support audit lost fixed query population')
                all_counts=audit[prefix+'_support']['all_queries']
                valid_counts=audit[prefix+'_support']['valid_position_queries']
                if (any(valid_counts[k] > all_counts[k] for k in all_counts)
                        or len(members)==1 and all_counts['zero'] != expected['data']['length']
                        or len(members)<=2 and all_counts['multiple'] != 0):
                    raise ValueError('083 helper support impossible for actual membership')
            eq=audit.get('retained_eligible_query_points'); ev=audit.get('retained_eligible_valid_position_points')
            ac=audit['retained_support']['all_queries']; vc=audit['retained_support']['valid_position_queries']
            if (type(eq) is not int or type(ev) is not int or not 0<=ev<=eq
                    or eq>plan['eligible_query_points'] or ev>plan['eligible_valid_position_points']
                    or eq>ac['one']+ac['multiple'] or ev>vc['one']+vc['multiple']):
                raise ValueError('083 retained eligible counts exceed original or supported population')
            if any(audit['retained_identity'][k] > audit['original_identity'][k] for k in identity):
                raise ValueError('083 retained identities exceed original')
            for pop in ('all_queries','valid_position_queries'):
                before, after = (audit[p+'_support'][pop] for p in ('original','retained'))
                if after['zero'] < before['zero'] or after['multiple'] > before['multiple']:
                    raise ValueError('083 removal increased supported helper count')
        if row.get('augmentation_rng_after') != rng.bit_generator.state:
            raise ValueError('083 augmentation RNG draw sequence differs')
        commit_exposure(exposure, aug)
    if exposure != new.get('augmentation_exposure'):
        raise ValueError('083 successful augmentation exposure differs from logs')
    with torch.random.fork_rng(devices=[]):
        model, final = load_checkpoint(Path(new_fit)/'final.pt', device='cpu')
    del model
    if (final['configuration'] != expected or final['step'] != len(nr) or final['actual_updates'] != len(nr)
            or final['augmentation_rng'] != rng.bit_generator.state):
        raise ValueError('083 final checkpoint does not bind completed dropout sequence')
    return dict(status='PAIR_VERIFIED', matched_updates=len(nr), matched_anchors=ne['sampled_anchors'],
        seed=expected['training']['seed'], initial_parameters_exact=True, initial_sampler_rng_exact=True,
        initial_torch_rng_exact=True, original_batches_exact=True, augmentation_draws_replayed=True,
        exposure=ne, augmentation_exposure=exposure)


def helper_strata(fragment, system):
    offsets, nodes = fragment['learned_assoc_group_offsets'], fragment['learned_assoc_group_node_indices']
    count = np.asarray([fragment['support'][nodes[a+1:b]].sum(0) for a,b in zip(offsets[:-1],offsets[1:])])
    t = len(system['anchor_target_ids']); selected = system['anchor_aligned_group_indices'][:t]
    valid = system['anchor_aligned_truth_valid'][:t,:,:2].all(-1) & system['anchor_aligned_prediction_valid'][:t,:,:2].all(-1)
    if ((selected < 0)[:,None] & valid).any():
        raise ValueError('083 valid target without representative')
    target_count=np.zeros(valid.shape,dtype=np.int64); chosen=selected>=0
    target_count[chosen]=count[selected[chosen]]
    return {'helper_cohort_'+name: valid & use for name,use in
        (('zero',target_count==0),('one',target_count==1),('two_or_more',target_count>=2))}


def scientific_screen(summary, scene_rows):
    table=summary['position_weightings']['all']; old=table['linear']; new=table['dropout']
    delta={k:new['point_equal'][k]-old['point_equal'][k] for k in ('mae','rmse','p95','p99')}
    fraction=1-new['point_equal']['mae']/old['point_equal']['mae']
    eps=1e-6
    other_weights=all(new[k]['mae'] <= old[k]['mae']+eps for k in old if k!='point_equal')
    def improved(item):
        tab=item['position_weightings']['all']
        return tab['dropout']['point_equal']['mae'] < tab['linear']['point_equal']['mae']-eps
    scenes=sum(improved(row) for row in scene_rows)
    components=sum(improved(row) for row in summary['components'].values())
    rest=summary['remaining_components']
    rest_ok=bool(rest and rest['position_weightings']['all']['dropout']['point_equal']['mae'] <=
        rest['position_weightings']['all']['linear']['point_equal']['mae']+eps)
    passed=(fraction>=.03 and delta['rmse']<=eps and delta['p99']<=eps and other_weights and rest_ok
        and scenes>len(scene_rows)/2 and components>len(summary['components'])/2)
    return dict(worth_matched_replication=bool(passed), relative_mae_reduction=fraction, required_mae_reduction=.03,
        deltas=delta, nonpoint_mae_nonworse=bool(other_weights), outside_largest_component_nonworse=rest_ok,
        improved_scenes=scenes,total_scenes=len(scene_rows),improved_components=components,
        total_components=len(summary['components']), numerical_guard_m=eps,
        interpretation='Predeclared resource screen, not statistical significance or method promotion. Failed recipe closes; no probability sweep or automatic repeat.')


def run(args):
    if args.cpu_threads != 4: raise ValueError('083 comparison requires CPU4')
    torch.set_num_threads(4)
    paths={'linear':Path(args.linear_run).resolve(),'dropout':Path(args.dropout_run).resolve()}
    output=Path(args.output).resolve()
    if output.exists(): raise FileExistsError('083 output exists; no overwrite/retry')
    for path in paths.values():
        source=Path(read_json(path/'run.json')['source_system']).resolve().parent
        for protected in (path,source):
            if output==protected or output in protected.parents or protected in output.parents:
                raise ValueError('083 output overlaps immutable input')
    output.mkdir(parents=True)
    state=dict(status='RUNNING',completed_scenes=0,model_forwards=0,optimizer_updates=0,test_opened=False)
    write_json(output/'run.json',state)
    try:
        arms={k:_metadata(paths[k],arm) for k,arm in (('linear','072C_transformer4000'),('dropout','083_helper_dropout4000'))}
        first=arms['linear']
        if any(arms['dropout'][k]!=first[k] for k in ('source_system','scene_ids','components')):
            raise ValueError('083 population/source/components differ')
        from tools.experiments.sequence_position_fit import bind_linear_control, ROOT
        config=read_json(paths['dropout']/'run.json')['configuration']
        if Path(read_json(paths['dropout']/'run.json').get('paired_control','')).resolve()!=paths['linear']:
            raise ValueError('083 paired source not the supplied072C')
        bind_linear_control(paths['linear'],config,first['source_system'],read_json(ROOT/'configs/data/mtad-level1-scene3000.json'))
        pairing=verify_dropout_pair(paths['dropout']/'fit',paths['linear']/'fit')
        methods=(*CONTROLS,'fixed_prior095',*arms)
        rows=[]; arrays=[]
        for sid in first['scene_ids']:
            row,data=scene_pair(arms,sid); row['component_id']=first['components'][sid]
            fragment=_read(paths['dropout']/'final/predictions'/f'states-{sid}.npz')
            system=_read(paths['dropout']/'final/system-predictions'/f'states-{sid}.npz')
            data.update(helper_strata(fragment,system)); rows.append(row); arrays.append(data)
            state['completed_scenes']=len(rows); write_json(output/'run.json',state)
        summary=summarize(rows,arrays,methods)
        for name,arm in arms.items():
            actual=summary['position_weightings']['all'][name]['point_equal']
            expected=arm['summary']['methods']['learned_sequence']['full_target_metrics']['all']['position']
            if actual['unit_count']!=expected['intersection_count'] or any(not np.isclose(actual[k],expected[k],rtol=1e-10,atol=1e-9) for k in ('mae','rmse')):
                raise ValueError('083 cached metrics differ from producer/'+name)
        scene_rows=[dict(summarize([r],[a],methods),component_id=r['component_id']) for r,a in zip(rows,arrays)]
        components={}
        for cid in sorted(set(first['components'].values())):
            ids=[i for i,r in enumerate(rows) if r['component_id']==cid]
            components[cid]=summarize([rows[i] for i in ids],[arrays[i] for i in ids],methods)
        largest=min(components,key=lambda cid:(-components[cid]['position_population']['covered_target_points'],cid))
        rest=[i for i,r in enumerate(rows) if r['component_id']!=largest]
        summary.update(components=components, largest_component_id=largest,
            remaining_components=summarize([rows[i] for i in rest],[arrays[i] for i in rest],methods) if rest else None)
        summary.update(status='COMPLETED',pairing=pairing,methods=list(methods),single_seed_comparison=True,
            scientific_screen=scientific_screen(summary,scene_rows),inputs={k:str(v) for k,v in paths.items()},
            model_forwards=0,optimizer_updates=0,test_opened=False,old066_acceptance='REJECTED',
            interpretation='T-only fragment-removal total effect: physical prior, information and contamination change together. V groups are complete. Reused V/shared identities, no independent generalization.')
        write_json_gzip(output/'summary.json.gz',summary); write_json_gzip(output/'scene-summary.json.gz',scene_rows)
        lines=['# 083 dropout vs original linear Transformer','','| Method | MAE m | RMSE m | P95 m | P99 m |','|---|---:|---:|---:|---:|']
        for name in methods:
            m=summary['position_weightings']['all'][name]['point_equal']
            lines.append('| '+name+' | '+' | '.join(f'{m[k]:.4f}' for k in ('mae','rmse','p95','p99'))+' |')
        lines+=['',str(summary['scientific_screen']),'','Full groups/masks, support/helper strata, H/I and four weightings retained. No automatic replication or method promotion.']
        (output/'report.md').write_text('\n'.join(lines)+'\n')
        state.update(status='COMPLETED',population=summary['population']); write_json(output/'run.json',state)
        return summary
    except BaseException as error:
        state.update(status='FAILED',error=f'{type(error).__name__}: {error}'); write_json(output/'run.json',state); raise


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('dropout-run','linear-run','output'): parser.add_argument('--'+name,required=True)
    parser.add_argument('--cpu-threads',type=int,default=4)
    result=run(parser.parse_args(argv)); print(result['status'],flush=True)
    return 0
if __name__=='__main__': raise SystemExit(main())
