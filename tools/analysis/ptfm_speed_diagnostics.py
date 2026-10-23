"""076A frozen074C speed diagnosis: CPU4, cache-only, no model or refitting."""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import signal
import sys
import time
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[key]='4'
if __package__ in (None,''):sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from association.evaluation.ptfm_speed_diagnostics import analyze, SOURCE_METHODS
from association.utils.runtime import read_json, write_json, write_json_gzip
ROOT=Path(__file__).resolve().parents[2]
ADVISORY_SECONDS=3600


def bind_source(source):
    source=Path(source).resolve()
    config=read_json(ROOT/'configs/ptfm-adapted-level1.json')
    manifest=read_json(ROOT/'configs/data/mtad-level1-scene3000.json')
    run,summary,fit,final,endpoint=[read_json(source/p) for p in
        ('run.json','summary.json','fit/run.json','final/summary.json','final/run.json')]
    updates=config['training']['updates']; ids=manifest['splits']['V_select']
    if not ids or len(ids)!=len(set(ids)):raise ValueError('076A complete unique V population required')
    for name,item in (('run',run),('summary',summary),('fit',fit),('final',final),('endpoint',endpoint)):
        if item.get('status')!='COMPLETED' or item.get('configuration')!=config:
            raise ValueError('076A completed frozen074C configuration required/'+name)
    for item in (run,summary,fit):
        if item.get('optimizer_updates')!=updates:raise ValueError('076A fixed final update mismatch')
    if (run.get('completed_endpoints')!=['initial','final'] or run.get('validation_selection') is not False
            or run.get('old066_acceptance')!='REJECTED'):
        raise ValueError('076A source must retain fixed final and old066 history')
    for item in (run,summary,final,endpoint):
        if item.get('test_opened') is not False:raise ValueError('076A source test-opened mismatch')
    for item in (run,summary,final,endpoint):
        if Path(item['source_system']).resolve()!=Path(summary['source_system']).resolve():
            raise ValueError('076A source056 binding differs')
    if (final.get('initial') is not False or endpoint.get('initial') is not False
            or final.get('device')!='cpu' or endpoint.get('device')!='cpu'
            or final.get('scene_ids')!=ids or endpoint.get('scene_ids')!=ids
            or final.get('scene_count')!=len(ids) or endpoint.get('completed_scenes')!=len(ids)
            or endpoint.get('total_scenes')!=len(ids) or final.get('method_names')!=list(SOURCE_METHODS)):
        raise ValueError('076A full completed CPU V endpoint required')
    if summary.get('normalizer')!=fit.get('normalizer') or summary['normalizer'].get('truth_used') is not False:
        raise ValueError('076A T-visible normalizer binding differs')
    checkpoint=dict(path=str(source/'fit/final.pt'),step=updates,actual_updates=updates,
        configured_final_updates=updates,routing_semantics=config['routing_semantics'])
    for item in (final,endpoint):
        if item.get('checkpoint')!=checkpoint:raise ValueError('076A checkpoint metadata differs (weights are never opened)')
    if summary.get('endpoints',{}).get('final')!=checkpoint:raise ValueError('076A final endpoint provenance differs')
    components={str(sid):str(c) for c,part in final['components'].items() for sid in part['scene_ids']}
    expected={sid:str(manifest['components'][sid]) for sid in ids}
    if components!=expected or final.get('component_count')!=len(set(expected.values())):
        raise ValueError('076A component registry differs')
    inventory={f'states-{sid}.npz' for sid in ids}
    for folder in ('predictions','system-predictions'):
        if {p.name for p in (source/'final'/folder).glob('*.npz')}!=inventory:
            raise ValueError('076A full074C NPZ inventory differs')
    return final,expected


def render_report(summary):
    pooled=summary['pooled']; lines=['# 076A：冻结074C速度诊断','',
        '仅原缓存读数；零速为预先固定诊断控制，未训练／前向／选择阈值。完整人口、代表与mask不变。', '',
        f"场景 {pooled['scene_count']}，速度共同有效点 {pooled['common_speed_point_count']}；单位保持原生，不能直接解释成节或米/秒。", '',
        '| 方法 | MAE | RMSE | P95 | P99 | Pearson | 目标内中心化RMSE | 相邻增量MAE |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    def value(x):return 'null' if x is None else f'{x:.7g}'
    for name,row in pooled['strata']['all']['methods'].items():
        metrics=(row['mae'],row['rmse'],row['absolute_error']['p95'],row['absolute_error']['p99'],row['pearson'],
            row['dynamics']['centered_error']['rmse'],row['dynamics']['adjacent_increment_error']['mae'])
        lines.append('| '+name+' | '+' | '.join(value(x) for x in metrics)+' |')
    lines+=['','相关系数只有非零双方方差且至少两点时定义；总体、逐目标和逐场定义数分别写入summary。',
        '中心化RMSE使用每个目标自己的均值，仅用于事后识别动态，不生成新预测。增量只用相邻query且两端有效；速度变化率除以实际秒差。','',
        '| 真值原生速度分层 | 点数 | PTFM MAE | 零速 MAE |', '|---|---:|---:|---:|']
    for name,row in pooled['fixed_truth_speed_bins'].items():
        lines.append(f"| {name} | {row['point_count']} | {value(row['methods']['ptfm_adapted']['mae'])} | {value(row['methods']['zero_speed']['mae'])} |")
    lines+=['','这些边界预先固定，仅描述原生数值；不声称对应停航或航行。','',
        '## 如何判断下一步','',
        '- 若PTFM仅接近零速控制、目标内变化及相关性很弱，则当前低MAE主要支持压制异常值／低速先验解释，不能宣称恢复了运动。',
        '- 若在非零真值分层、目标内中心化和相邻增量上均改善，则支持真实动态恢复线索；仍不能从有无helper两个不同人口推导多源因果收益。',
        '- 无论结果如何，本轮不拟合V常数、不屏蔽helper再前向、不自动启动训练；位置修复由独立实验验证。','',
        '## 边界','']+['- '+text for text in summary['limitations']]
    lines+=['','完整可复算聚合见summary.json.gz，逐场见scene-summary.json.gz；原始074C缓存保留服务器不回传。']
    return '\n'.join(lines)+'\n'


def run(args):
    source,output=Path(args.source_074c).resolve(),Path(args.output).resolve()
    if output==source or source in output.parents or output in source.parents:raise ValueError('076A output must preserve source')
    if output.exists():raise FileExistsError('076A no overwrite or automatic retry')
    original056=Path(read_json(source/'summary.json')['source_system']).resolve().parent
    if output==original056 or original056 in output.parents or output in original056.parents:
        raise ValueError('076A output must preserve original056 source')
    if args.cpu_threads!=4:raise ValueError('076A fixed CPU4')
    output.mkdir(parents=True,exist_ok=False);started=time.monotonic()
    state=dict(scope='076A-ptfm-speed-cache-diagnostics',status='RUNNING',source_074c=str(source),
        cpu_threads=4,model_forwards=0,optimizer_updates=0,checkpoints_opened=False,raw_archives_opened=False,
        test_opened=False,groups_recomputed=False,threshold_reselected=False,validation_fitted=False,
        old066_acceptance='REJECTED',advisory_reached=False)
    write_json(output/'run.json',state)
    def check():
        if time.monotonic()-started>=ADVISORY_SECONDS and not state['advisory_reached']:
            state['advisory_reached']=True;write_json(output/'run.json',state)
            print('ADVISORY:076A continues same fixed read-only attempt',flush=True)
    try:
        final,components=bind_source(source)
        state['scene_count']=len(final['scene_ids']);state['phase']='read_and_validate_cache';write_json(output/'run.json',state)
        result,rows=analyze(source/'final',final,components,check)
        result.update(scope=state['scope'],source_074c=str(source),model_forwards=0,optimizer_updates=0,test_opened=False,
            checkpoint_loaded=False,old066_acceptance='REJECTED',diagnostic_zero_rule='all output groups get speed=0 before original target alignment')
        write_json_gzip(output/'summary.json.gz',result);write_json_gzip(output/'scene-summary.json.gz',rows)
        (output/'report.md').write_text(render_report(result),encoding='utf-8')
        state.update(status='COMPLETED',phase='completed',completed_scenes=len(rows),elapsed_seconds=time.monotonic()-started)
        write_json(output/'run.json',state);return result
    except BaseException as error:
        state.update(status='INTERRUPTED' if isinstance(error,(KeyboardInterrupt,InterruptedError)) else 'FAILED',
            error=f'{type(error).__name__}: {error}',elapsed_seconds=time.monotonic()-started)
        write_json(output/'run.json',state);raise


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-074c',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--cpu-threads',type=int,default=4)
    args=parser.parse_args(argv)
    def stop(signum,frame):raise KeyboardInterrupt('076A signal interruption')
    signal.signal(signal.SIGTERM,stop)
    print(run(args)['status'],flush=True);return 0
if __name__=='__main__':raise SystemExit(main())
