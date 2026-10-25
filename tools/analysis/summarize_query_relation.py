"""098 fixed two-seed closeout from completed cached comparison tables only."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from association.utils.runtime import read_json, read_json_or_gzip, write_json
from tools.analysis.compare_query_relation import SCOPE as TABLE_SCOPE, declared_manifest, scientific_screen

SCOPE = '098-query-relation-two-seed-closeout'
SEEDS = (20261023, 20261021)
METHODS = ('linear', 'finetune', 'generic', 'relation', 'relation_off')


def load_table(path, seed):
    run = read_json(path/'run.json')
    summary = read_json_or_gzip(path/'summary.json.gz')
    scenes = read_json_or_gzip(path/'scene-summary.json.gz')
    for record in (run, summary):
        if (record.get('status') != 'COMPLETED' or record.get('scope') != TABLE_SCOPE
                or type(record.get('seed')) is not int or record['seed'] != seed
                or record.get('model_forwards') != 0 or record.get('optimizer_updates') != 0
                or record.get('test_opened') is not False or record.get('default_promoted') is not False):
            raise ValueError('098 closeout requires completed, correctly bound seed tables')
    manifest = declared_manifest()
    expected = manifest['splits']['V_select']
    if (summary.get('scene_ids') != expected or summary.get('scene_count') != len(expected)
            or len(scenes) != len(expected) or [row.get('scene_ids') for row in scenes] != [[sid] for sid in expected]
            or any(row.get('component_id') != str(manifest['components'][sid]) for row, sid in zip(scenes, expected))
            or run.get('completed_scenes') != len(expected)
            or summary.get('pairing', {}).get('status') != 'PAIR_VERIFIED'
            or summary['pairing'].get('seed') != seed or summary.get('initial_output_source_final_exact') is not True
            or any(not isinstance(summary.get(key), dict) or not summary[key]
                   for key in ('population', 'position_population', 'helper_population'))
            or not isinstance(summary.get('source_system'), str) or not summary['source_system']):
        raise ValueError('098 closeout full V population, source or pairing is incomplete')
    decision = scientific_screen(summary, scenes)
    if (decision != summary.get('scientific_screen')
            or run.get('seed_screen_passed') is not decision['seed_screen_passed']
            or decision['versus_budget']['required_reduction'] != .03
            or decision['versus_generic']['required_reduction'] != .01
            or decision['required_seeds'] != list(SEEDS)
            or decision['default_promoted'] is not False or decision['test_opened'] is not False):
        raise ValueError('098 closeout rejects a changed or unrecomputed scientific gate')
    return summary, decision


def run(args):
    paths = [Path(args.seed23_table).resolve(), Path(args.seed21_table).resolve()]
    output = Path(args.output).resolve()
    if paths[0] == paths[1]:
        raise ValueError('098 closeout requires two distinct seed tables')
    if output.exists():
        raise FileExistsError('098 closeout exists; no overwrite/retry')
    if any(output == path or output in path.parents or path in output.parents for path in paths):
        raise ValueError('098 closeout output overlaps an input table')
    output.mkdir(parents=True)
    state = dict(scope=SCOPE, status='RUNNING', model_forwards=0, optimizer_updates=0,
        test_opened=False, default_promoted=False, input_tables={str(seed): str(path) for seed, path in zip(SEEDS, paths)})
    write_json(output/'run.json', state)
    try:
        loaded = [load_table(path, seed) for path, seed in zip(paths, SEEDS)]
        left, right = loaded[0][0], loaded[1][0]
        for key in ('source_system', 'scene_ids', 'population', 'position_population', 'helper_population'):
            if left.get(key) != right.get(key):
                raise ValueError('098 two-seed source/full population differs: '+key)
        eligible = all(decision['seed_screen_passed'] for _, decision in loaded)
        by_seed = {str(seed): dict(scientific_screen=decision,
            position_metrics={name: table['position_weightings']['all'][name]['point_equal'] for name in METHODS})
            for seed, (table, decision) in zip(SEEDS, loaded)}
        summary = dict(state, status='COMPLETED', seeds=list(SEEDS), by_seed=by_seed,
            source_system=left['source_system'], scene_ids=left['scene_ids'],
            population=left['population'], position_population=left['position_population'],
            gates_recomputed=True, eligible_for_independent_confirmation=eligible,
            outcome='CANDIDATE_FOR_INDEPENDENT_CONFIRMATION' if eligible else 'CURRENT_RELATION_RECIPE_NOT_ACCEPTED',
            interpretation='Both preregistered seeds must pass their own 3% budget and 1% generic gates, tails, robustness and fixed-weight intervention. Reused V/shared identities; no automatic default change or test access.')
        write_json(output/'summary.json', summary)
        lines = ['# 098 双 seed 位置融合结案', '', '| Seed | 方法 | MAE m | RMSE m | P95 m | P99 m |',
            '|---|---|---:|---:|---:|---:|']
        for seed in SEEDS:
            for name, metric in by_seed[str(seed)]['position_metrics'].items():
                lines.append(f'| {seed} | {name} | '+' | '.join(f'{metric[k]:.6f}' for k in ('mae', 'rmse', 'p95', 'p99'))+' |')
        lines += ['', '双 seed 出口：'+summary['outcome']+'。', '',
            *[f"- {seed}：{by_seed[str(seed)]['scientific_screen']['outcome']}。" for seed in SEEDS], '',
            '全部门槛从两张完成表及逐场数据重新计算；只有两个 seed 同时通过才进入独立确认。',
            '默认方法不变，T_test 保持封存；通用 set 的改善单独记录为任务基线证据。']
        (output/'report.md').write_text('\n'.join(lines)+'\n')
        state.update(status='COMPLETED', eligible_for_independent_confirmation=eligible, gates_recomputed=True)
        write_json(output/'run.json', state)
        return summary
    except BaseException as error:
        state.update(status='FAILED', error=f'{type(error).__name__}: {error}')
        write_json(output/'run.json', state)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('seed23-table', 'seed21-table', 'output'):
        parser.add_argument('--'+name, required=True)
    result = run(parser.parse_args(argv))
    print(result['status'], result['outcome'], flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
