"""085 CPU4 cache/log qualification of 074C and 076B; no model execution."""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import signal
import sys
import time

for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '4'
if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from association.evaluation.ptfm_objective_audit import analyze, bind_source
from association.utils.runtime import read_json, write_json, write_json_gzip

ROOT = Path(__file__).resolve().parents[2]
ADVISORY_SECONDS = 3600


def render_report(summary):
    def f(value):
        return 'null' if value is None else f'{value:.8g}'
    lines = ['# 085：PTFM 既有缓存与日志资格封口', '',
        '当前两项 PTFM 配方继续关闭。本次仅核现有输出的目标与物理误差，未重新训练、前向或构造原生双源子任务。', '',
        f"完整 V：{summary['scene_count']} 场；原六臂物理人口、分组、对应和有效 mask 已重放。", '',
        '| 方法/端点 | 唯一采用组数 | 有位置监督组数 | 标准化位置目标（组等权） | 位置 MAE m（点等权） | 位置有效系数（组均值） |',
        '|---|---:|---:|---:|---:|---:|']
    for method, data in summary['methods'].items():
        for endpoint, values in data['endpoints'].items():
            row = values['unique_selected_groups']; p = row['tasks']['position']
            lines.append(f"| {method}/{endpoint} | {row['group_count']} | {p['valid_group_count']} | "
                f"{f(p['loss_group_mean'])} | {f(p['physical_mae_point_mean'])} | {f(p['coefficient_group_mean'])} |")
    lines += ['', '上述组等权目标与点等权 MAE 分母不同；summary 同时给出组等权物理误差、点等权原目标和逐组符号对应表。',
        '全部输出组诊断含重复组，按首代表身份取已有参考；它不是新增的唯一目标成绩。无真值组仍保留计数，原目标总项为零，不被当成零误差样本。', '',
        '## 解释边界与出口', '',
        '- 原位置目标下降而物理 MAE 上升，仅支持目标与指标的取舍线索；不能据此直接确定训练失败根因。',
        '- 若二者都变差，也不能仅凭 V 输出判断优化失败或 T/V 分布差异。',
        '- 固定 T 同批 step0/final 未保存在本轮允许的缓存中，结论保持 UNKNOWN；需要新冻结前向，085 不执行。',
        '- 训练日志按 mode 与固定 100 步窗口汇总；批次不同，不能代替固定 T 端点比较。',
        '- 任务有效系数为每组、每任务有效点上权重的几何均值。它不是 helper 权重或可信概率。',
        '- 每批只保存 task loss 和 sigma 的均值，无法恢复历史每样本加权任务项。不得用均值乘积替代。',
        '- V 缓存没有 LSTM 参数；V 目标不含 L2。训练日志 data loss 与 L2 单列，不能把两者之和同此 V 数据目标混算。',
        '- 不自动修复、重训、挑终点或晋升该适配；发现具体可区分问题后再单独设计。', '',
        '## 其他限制', ''] + ['- ' + item for item in summary['limitations']]
    return '\n'.join(lines) + '\n'


def run(args):
    direct, residual, output = [Path(p).resolve() for p in (args.direct, args.residual, args.output)]
    if direct == residual or direct in residual.parents or residual in direct.parents:
        raise ValueError('085 requires distinct independent source directories')
    if args.cpu_threads != 4:
        raise ValueError('085 fixed CPU4')
    protected = [direct, residual]
    for source in (direct, residual):
        meta = read_json(source / 'run.json')
        protected.append(Path(meta['source_system']).resolve().parent)
        if meta.get('paired_control'):
            protected.append(Path(meta['paired_control']).resolve())
    if any(output == p or p in output.parents or output in p.parents for p in protected):
        raise ValueError('085 output must preserve all sources')
    if output.exists():
        raise FileExistsError('085 no overwrite or automatic retry')
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    state = dict(scope='085-ptfm-objective-cache-audit', status='RUNNING', direct=str(direct), residual=str(residual),
        model_forwards=0, optimizer_updates=0, checkpoints_opened=False, raw_archives_opened=False,
        test_opened=False, cpu_threads=4, old066_acceptance='REJECTED', current_ptfm_recipes='CLOSED', advisory_reached=False)
    write_json(output / 'run.json', state)

    def check():
        if time.monotonic() - started >= ADVISORY_SECONDS and not state['advisory_reached']:
            state['advisory_reached'] = True
            write_json(output / 'run.json', state)
            print('ADVISORY:085 continues its one cache-only attempt; no retry', flush=True)

    try:
        manifest = read_json(ROOT / 'configs/data/mtad-level1-scene3000.json')
        a = bind_source(direct, read_json(ROOT / 'configs/ptfm-adapted-level1.json'), manifest)
        b = bind_source(residual, read_json(ROOT / 'configs/ptfm-residual-level1.json'), manifest)
        if Path(b['run'].get('paired_ptfm', '')).resolve() != direct:
            raise ValueError('085 residual source must bind this original074C')
        pair = b['fit'].get('paired_ptfm_control', {})
        if (pair.get('status') != 'PAIR_VERIFIED' or Path(pair.get('fit', '')).resolve() != direct / 'fit'
                or pair.get('matched_updates') != b['fit']['optimizer_updates']
                or pair.get('matched_anchors') != b['fit']['exposure']['sampled_anchors']
                or any(pair.get(k) is not True for k in ('initial_parameters_exact_except_position_rows',
                    'normalizer_exact', 'actual_successful_batches_exact', 'all_task_counts_exact', 'full_exposure_exact'))):
            raise ValueError('085 residual paired source/exposure metadata differs')
        state['phase'] = 'validate_cache_and_logs'; write_json(output / 'run.json', state)
        result, rows = analyze(a, b, check)
        result.update(scope=state['scope'], direct=str(direct), residual=str(residual),
            model_forwards=0, optimizer_updates=0, checkpoints_opened=False, test_opened=False,
            old066_acceptance='REJECTED', current_ptfm_recipes='CLOSED')
        write_json_gzip(output / 'summary.json.gz', result)
        write_json_gzip(output / 'scene-summary.json.gz', rows)
        (output / 'report.md').write_text(render_report(result), encoding='utf-8')
        state.update(status='COMPLETED', phase='completed', completed_scenes=result['scene_count'],
                     elapsed_seconds=time.monotonic() - started)
        write_json(output / 'run.json', state)
        return result
    except BaseException as error:
        state.update(status='INTERRUPTED' if isinstance(error, (KeyboardInterrupt, InterruptedError)) else 'FAILED',
                     error=f'{type(error).__name__}: {error}', elapsed_seconds=time.monotonic() - started)
        write_json(output / 'run.json', state)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('direct', 'residual', 'output'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--cpu-threads', type=int, default=4)
    args = parser.parse_args(argv)
    def stop(signum, frame):
        raise KeyboardInterrupt('085 signal interruption')
    signal.signal(signal.SIGTERM, stop)
    print(run(args)['status'], flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
