"""Summarize an existing107 TCM feature report; no model, ZIP or score forward.

Despite the historical proposal name, this stage measures provided/generated
query support and frozen-feature error only. It cannot reconstruct pairwise
OO/OG/GG masks, TAM scores, physical error or a causal completion effect.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import random
import resource
import signal
import sys
import time

SEED, SCENE_COUNT = 20261029, 16
COUNTS = ('query_count', 'evaluable_query_count', 'missing_reference_query_count', 'feature_elements')
REGIONS = ('all', 'inside_support', 'before_support', 'after_support')
METHODS = ('model', 'state_hold')
POLICY = 'measurement_kn_reference_mps_v1'


def write_json(path, value):
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2)+'\n')
    temporary.replace(path)


def read_json(path, maximum):
    if path.stat().st_size > maximum:
        raise ValueError('source JSON exceeds the prespecified byte limit')
    with path.open(encoding='utf-8') as stream:
        return json.load(stream, parse_constant=lambda _: (_ for _ in ()).throw(ValueError('nonfinite JSON literal')))


def peak_rss_bytes():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value if sys.platform == 'darwin' else value*1024


def select_scenes(manifest):
    splits = manifest['splits']
    if (manifest.get('dataset_level') != 'Level1' or manifest.get('split_unit') != 'scene'
            or len(splits.get('V_select', [])) != 300 or len(splits.get('T_core', [])) != 2100
            or len(set(splits['V_select'])) != 300
            or set(splits['V_select']) & set(splits['T_core'])):
        raise ValueError('fixed Level1 T2100/V300 manifest required')
    return sorted(random.Random(SEED).sample(sorted(splits['V_select'], key=int), SCENE_COUNT), key=int)


def validate_metric(item, dimension):
    if any(type(item[key]) is not int or item[key] < 0 for key in COUNTS):
        raise ValueError('feature metric counts must be nonnegative integers')
    if (item['evaluable_query_count']+item['missing_reference_query_count'] != item['query_count']
            or item['feature_elements'] != dimension*item['evaluable_query_count']):
        raise ValueError('feature/reference denominators are inconsistent')
    square = item['squared_error_sum']
    if isinstance(square, bool) or not isinstance(square, (float, int)) or not math.isfinite(square) or square < 0:
        raise ValueError('feature squared error must be finite and nonnegative')
    count = item['feature_elements']
    expected = square/count if count else None
    actual = item['feature_mse']
    if ((actual is None) != (expected is None) or
            (expected is not None and (not math.isfinite(actual) or
             not math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-12)))):
        raise ValueError('saved feature MSE differs from its numerator/denominator')


def pool(rows):
    result = {key: sum(row[key] for row in rows) for key in COUNTS}
    square = math.fsum(row['squared_error_sum'] for row in rows)
    elements, queries = result['feature_elements'], result['query_count']
    return dict(**result, squared_error_sum=square, feature_mse=square/elements if elements else None,
                reference_coverage=result['evaluable_query_count']/queries if queries else None)


def summarize(report, selected, full_scene_ids, *, dimension, length):
    """Use every fragment in the selected scenes, retaining missing reference."""
    per_scene = report['per_scene']
    ids = [str(row['scene_id']) for row in per_scene]
    if (ids != list(full_scene_ids) or report['scene_count'] != len(ids)
            or len(set(ids)) != len(ids) or report.get('test_opened') is not False):
        raise ValueError('saved TCM report does not cover the complete original V scene registry')
    if report['fragment_count'] != len(report['per_fragment']):
        raise ValueError('saved fragment denominator differs from its full journal')
    expected = {str(row['scene_id']): row['fragment_count'] for row in per_scene}
    actual, seen, chosen = Counter(), set(), []
    for row in report['per_fragment']:
        sid, node = str(row['scene_id']), row['node_index']
        if sid not in expected or type(node) is not int or node < 0 or (sid, node) in seen:
            raise ValueError('TCM fragment journal contains unknown, duplicate or malformed nodes')
        seen.add((sid, node)); actual[sid] += 1
        if sid not in selected:
            continue
        for method in METHODS:
            regions = row['query'][method]['regions']
            for key in REGIONS:
                validate_metric(regions[key], dimension)
            if regions['all']['query_count'] != length:
                raise ValueError('a selected fragment is missing declared query positions')
            part = pool([regions[key] for key in REGIONS[1:]])
            for key in COUNTS:
                if part[key] != regions['all'][key]:
                    raise ValueError('O/before/after counts do not partition the full query denominator')
            if not math.isclose(part['squared_error_sum'], regions['all']['squared_error_sum'],
                                rel_tol=1e-10, abs_tol=1e-10):
                raise ValueError('O/G error numerators do not sum to all-query error')
        for region in REGIONS:
            if any(row['query']['model']['regions'][region][key] !=
                   row['query']['state_hold']['regions'][region][key] for key in COUNTS):
                raise ValueError('model/state-hold do not use identical reference/support denominators')
        chosen.append(row)
    if any(actual[sid] != count for sid, count in expected.items()):
        raise ValueError('per-scene and per-fragment population denominators differ')
    if any({node for sid_, node in seen if sid_ == sid} != set(range(count)) for sid, count in expected.items()):
        raise ValueError('saved fragment node indices are not the complete input order')
    if not set(selected).issubset(expected):
        raise ValueError('selected scene absent from saved complete report')

    def aggregate(rows):
        metrics = {method: {region: pool([row['query'][method]['regions'][region] for row in rows])
                           for region in REGIONS} for method in METHODS}
        for method in METHODS:
            metrics[method]['generated'] = pool([metrics[method]['before_support'], metrics[method]['after_support']])
        deltas = {}
        for region in (*REGIONS, 'generated'):
            model, hold = (metrics[method][region]['feature_mse'] for method in METHODS)
            deltas[region] = model-hold if model is not None and hold is not None else None
        queries = metrics['model']['all']['query_count']
        provided = metrics['model']['inside_support']['query_count']
        return dict(fragment_count=len(rows), declared_query_count=queries,
                    provided_query_count=provided, generated_query_count=queries-provided,
                    generated_query_fraction=(queries-provided)/queries if queries else None,
                    fragments_without_provided_queries=sum(row['query']['model']['regions']['inside_support']['query_count'] == 0
                                                          for row in rows),
                    query_feature_metrics=metrics, model_minus_hold_feature_mse=deltas)

    result = aggregate(chosen)
    result['scene_count'] = len(selected)
    result['per_scene'] = [dict(scene_id=sid, **aggregate([row for row in chosen if str(row['scene_id']) == sid]))
                           for sid in selected]
    # Source-wide aggregates are identified as reused source output, not a new
    # second population or an independent comparator to this overlapping V16.
    result['full_V_source_aggregate'] = dict(scene_count=report['scene_count'], fragment_count=report['fragment_count'],
        query={method: {region: report['query'][method]['regions'][region] for region in REGIONS} for method in METHODS},
        status='existing_source_output_only; overlaps selected V16; not independent validation')
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('tcm-run', 'manifest', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args(argv)
    started, owns_output = time.monotonic(), False
    state = dict(status='RUNNING', plan_version='1.1', source_tcm_run=str(args.tcm_run.resolve()),
                 cpu_only=True, gpu_forwards=0, optimizer_updates=0, checkpoint_read=False,
                 archive_read=False, reference_csv_read=False, model_constructed=False, test_opened=False,
                 source_report_reference_derived=True, source_report_reads=0)
    prior_signal = signal.getsignal(signal.SIGTERM)
    def terminate(*_):
        raise InterruptedError('SIGTERM received; preserve partial diagnostic')
    signal.signal(signal.SIGTERM, terminate)
    try:
        args.output.mkdir(parents=True, exist_ok=False); owns_output = True
        write_json(args.output/'run.json', state)
        if sys.platform.startswith('linux'):
            soft, hard = resource.getrlimit(resource.RLIMIT_AS)
            limit = min([8*1024**3]+[x for x in (soft, hard) if x != resource.RLIM_INFINITY])
            resource.setrlimit(resource.RLIMIT_AS, (limit, hard))
            state['address_space_limit_bytes'] = limit
        manifest = read_json(args.manifest, 16*1024**2)
        selected = select_scenes(manifest)
        run = read_json(args.tcm_run/'run.json', 32*1024**2)
        summary = read_json(args.tcm_run/'summary.json', 32*1024**2)
        norm = summary['normalizer']; adapter = norm['adapter_config']
        model = run['requested_model_config']
        if (run.get('status') != 'COMPLETED' or run.get('phase') != 'tcm'
                or run.get('final_evaluation_status') != 'COMPLETED' or run.get('test_opened') is not False
                or run['cursor']['completed_updates'] != 2424 or summary.get('phase') != 'tcm'
                or run['config']['updates']['TAM_FROZEN'] != 0 or run['config']['updates']['JOINT'] != 0
                or norm.get('input_adapter') != 'native_time_v1'
                or adapter.get('speed_unit_policy') != POLICY
                or norm.get('physical_units', {}).get('policy') != POLICY
                or (adapter.get('length'), adapter.get('max_observations'), adapter.get('time_unit_seconds')) != (64, 128, 20.)
                or model.get('d_model') != 128 or model.get('length') != 64):
            raise ValueError('completed fixed107 new-unit TCM and original adapter contract required')
        state.update(selected_scene_ids=selected, selection_seed=SEED, selected_scene_count=SCENE_COUNT,
                     source_completed_updates=2424, frozen_normalizer_refitted=False,
                     feature_dimension=128, source_report='completion-evaluation.json')
        write_json(args.output/'run.json', state)
        path = args.tcm_run/'completion-evaluation.json'
        state['source_report_bytes'] = path.stat().st_size
        report = read_json(path, 256*1024**2); state['source_report_reads'] = 1
        if peak_rss_bytes() > 4*1024**3:
            raise MemoryError('CPU summary exceeded its 4GiB RSS budget')
        result = summarize(report, selected, manifest['splits']['V_select'], dimension=128, length=64)
        result.update(selection=dict(seed=SEED, scene_ids=selected, scene_count=SCENE_COUNT,
                    policy='random.Random(seed).sample(numerically sorted original V_select,16), then numerical sort; no scores'),
                    source_report_reference_derived=True,
                    feature_units='squared Euclidean error per feature element in frozen normalized time-encoded ReLU embedding; not physical metres',
                    interpretation='O is existing observation-derived interpolation; G is before+after support. No TAM/OO-OG-GG pair masks, score validation or causal completion effect.',
                    source_target_policy=report['target_policy'], source_interpretation=report['interpretation'],
                    score_subproblem_status='deferred;107 TAM unused/untrained; do not combine with heterogeneous056 scores')
        write_json(args.output/'summary.json', result)
        state['status'] = 'COMPLETED'
        return 0
    except (Exception, KeyboardInterrupt) as error:
        state.update(status='FAILED', error_type=type(error).__name__, error=str(error))
        print(type(error).__name__+': '+str(error), file=sys.stderr)
        return 1
    finally:
        signal.signal(signal.SIGTERM, prior_signal)
        if owns_output:
            state.update(elapsed_seconds=time.monotonic()-started,
                         peak_rss_bytes=peak_rss_bytes())
            write_json(args.output/'run.json', state)


if __name__ == '__main__':
    raise SystemExit(main())
