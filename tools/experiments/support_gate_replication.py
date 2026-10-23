"""071 independent log/linear seed replication and post-completion CPU pairing."""
from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
import signal
import sys
import time

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from association.data.dataset import load_timed_dataset
from association.evaluation.support_gate import evaluate_support_gate
from association.evaluation.support_gate_replication import SCOPE, compare_cached_endpoints, compare_replica_runs
from association.training.support_gate import configuration, loss_semantics, train_support_gate
from association.utils.runtime import read_json, write_json
from tools.analysis.compare_association_duration import _load_metadata
from tools.analysis.learned_group_oracle import _bind_source
from tools.experiments.support_gate_fit import endpoint_model, production_contract as original_contract

ROOT = Path(__file__).resolve().parents[2]


def production_contract(config, manifest, alignment):
    configuration(config)
    kind = config.get('position_loss')
    if kind not in ('log1p', 'linear'):
        raise ValueError('071 explicitly declares log1p or linear loss')
    name = 'log' if kind == 'log1p' else 'linear'
    if config != read_json(ROOT/f'configs/support-gate-replica-{name}-level1.json'):
        raise ValueError('071 requires its frozen seed20261023/1000-update configuration')
    original = deepcopy(config)
    original.pop('position_loss')
    original['training']['seed'] = 20261021
    original_contract(original, manifest, alignment)


def annotate_endpoint(directory, config, step):
    """Publish checkpoint binding identically for both endpoint records."""
    directory = Path(directory)
    filename = 'step-0.pt' if step == 0 else 'final.pt'
    checkpoint = dict(path=str(directory.parent/'fit'/filename), step=step, actual_updates=step,
        configured_final_updates=config['training']['updates'], routing_semantics=config['routing_semantics'],
        loss_semantics=loss_semantics(config))
    for name in ('run.json', 'summary.json'):
        record = read_json(directory/name)
        record.update(scope=SCOPE+'-evaluation', checkpoint=checkpoint,
                      configuration=config, position_loss=config['position_loss'])
        write_json(directory/name, record)


def run(args):
    source, output = Path(args.source_system).resolve(), Path(args.output).resolve()
    protected = source.parent
    if output == protected or output in protected.parents or protected in output.parents:
        raise ValueError('output must be separate from immutable056')
    if output.exists():
        raise FileExistsError('output already exists; no reuse or automatic retry')
    if args.cpu_threads != 4 or args.time_policy != 'advisory':
        raise ValueError('071 fixes CPU4 and advisory-only wall time')
    torch.set_num_threads(4)
    if args.device == 'cuda:0' and not torch.cuda.is_available():
        raise ValueError('requested CUDA unavailable; no fallback')
    config, manifest, alignment = map(read_json, (args.config, args.manifest, args.alignment))
    production_contract(config, manifest, alignment)
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    state = dict(scope=SCOPE, status='RUNNING', phase='source_binding', configuration=config,
        source_system=str(source), loss_semantics=loss_semantics(config), train_device=args.device,
        inference_device='cpu', cpu_threads=4, optimizer_updates=0, test_opened=False, retries=0,
        old066_acceptance='REJECTED', association_recomputed=False, threshold_reselected=False,
        validation_groups_recomputed=False, validation_selection=False, completed_endpoints=[],
        pairing_status='NOT_YET_AUDITED', other_arm_dependency=False, advisory_reached=False)
    write_json(output/'run.json', state)
    def check():
        if time.monotonic()-started >= config['training']['advisory_seconds'] and not state['advisory_reached']:
            state['advisory_reached'] = True
            write_json(output/'run.json', state)
            print('ADVISORY: continuing fixed single-attempt budget, without retry.', flush=True)
    def phase(name):
        state.update(phase=name, elapsed_seconds=time.monotonic()-started)
        write_json(output/'run.json', state)
        print('071 phase: '+name, flush=True)
        check()
    try:
        _bind_source(source)
        metadata = _load_metadata(source, manifest, 1000)
        metadata['component_ids'] = {sid: str(manifest['components'][sid]) for sid in metadata['scene_ids']}
        phase('prepare_T2100_V300')
        dataset = load_timed_dataset(args.archive, manifest, train_splits=['T_core'],
            data_config=config['data'], alignment_by_scene=alignment, check=check,
            motion_observation_policy='canonical_all')
        for key, split in (('train', 'T_core'), ('validation', 'V_select')):
            if [str(s['scene_id']) for s in dataset[key]] != manifest['splits'][split]:
                raise ValueError('prepared complete split/order differs: '+split)
        write_json(output/'preparation.json', dataset['report'])
        phase('fit_independent_'+config['position_loss'])
        if args.device == 'cuda:0':
            torch.cuda.reset_peak_memory_stats()
        fit = train_support_gate(dataset.pop('train'), config, output/'fit', device=args.device,
                                 check=check, record_objectives=True)
        state['optimizer_updates'] = fit['optimizer_updates']
        resources = dict(fit_elapsed_seconds=fit['elapsed_seconds'],
            peak_cuda_memory_allocated_bytes=(torch.cuda.max_memory_allocated() if args.device == 'cuda:0' else 0))
        if torch.cuda.is_initialized():
            torch.cuda.empty_cache()
        for label, filename, step in (('initial', 'step-0.pt', 0), ('final', 'final.pt', config['training']['updates'])):
            phase('evaluate_cpu_'+label)
            model = endpoint_model(output/'fit'/filename, step, config, fit)
            resources['parameter_count'] = sum(p.numel() for p in model.parameters())
            evaluate_support_gate(model, dataset['validation'], source, metadata, output/label, check=check)
            annotate_endpoint(output/label, config, step)
            state['completed_endpoints'].append(label)
            del model
        phase('own_endpoint_comparison')
        comparison = compare_cached_endpoints(output/'initial', output/'final', metadata['component_ids'],
                                               output/'comparison', check=check)
        summary = dict(scope=SCOPE, status='COMPLETED', configuration=config, source_system=str(source),
            optimizer_updates=fit['optimizer_updates'], test_opened=False, inference_device='cpu',
            resources=resources,
            pairing_status='NOT_YET_AUDITED', other_arm_dependency=False,
            own_comparison_status=comparison['status'], comparison_directory=str(output/'comparison'),
            scientific_status='Independent arm on second seed; paired claim waits for actual-batch audit.',
            limitations=['Reused development V and cross-scene shared identities; no independent generalization.',
                'Frozen056 grouping, offline whole-fragment inputs and synthetic training groups.',
                'Loss changes amplitude weighting and optimization dynamics; neither is separately isolated.',
                'Original066 remains REJECTED. Speed/course remain ordinary Mean outputs.'])
        write_json(output/'summary.json', summary)
        state.update(status='COMPLETED', phase='completed', elapsed_seconds=time.monotonic()-started)
        write_json(output/'run.json', state)
        return summary
    except BaseException as error:
        if (output/'fit/run.json').exists():
            state['optimizer_updates'] = read_json(output/'fit/run.json').get('optimizer_updates', 0)
        state.update(status='INTERRUPTED' if isinstance(error, KeyboardInterrupt) else 'FAILED',
            error=f'{type(error).__name__}: {error}', elapsed_seconds=time.monotonic()-started)
        write_json(output/'run.json', state)
        raise
    finally:
        write_json(output/'execution.json', dict(arguments=vars(args), configuration=config,
            test_opened=False, other_arm_dependency=False, resumed=False, raw_preflight_repeated=False))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    fit = sub.add_parser('fit', help='one independent arm; no other arm input')
    fit.add_argument('--archive', nargs='+', required=True)
    for key in ('manifest', 'alignment', 'config', 'source-system', 'output'):
        fit.add_argument('--'+key, required=True)
    fit.add_argument('--device', choices=('cpu', 'cuda:0'), default='cuda:0')
    fit.add_argument('--cpu-threads', type=int, default=4)
    fit.add_argument('--time-policy', choices=('advisory',), default='advisory')
    paired = sub.add_parser('compare', help='CPU audit only after both complete arms exist')
    for key in ('log-run', 'linear-run', 'output'):
        paired.add_argument('--'+key, required=True)
    paired.add_argument('--cpu-threads', type=int, default=4)
    args = parser.parse_args(argv)
    def cancel(signum, frame):
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        raise KeyboardInterrupt(f'cancelled by signal {signum}')
    previous = signal.signal(signal.SIGTERM, cancel)
    try:
        if args.cpu_threads != 4:
            raise ValueError('071 fixes CPU threads to4')
        torch.set_num_threads(4)
        result = run(args) if args.command == 'fit' else compare_replica_runs(args.log_run, args.linear_run, args.output)
        print(result['status']+': '+str(Path(args.output).resolve()), flush=True)
        return 0
    except (Exception, KeyboardInterrupt) as error:
        print(f'{type(error).__name__}: {error}', file=sys.stderr, flush=True)
        return 130 if isinstance(error, KeyboardInterrupt) else 1
    finally:
        signal.signal(signal.SIGTERM, previous)


if __name__ == '__main__':
    raise SystemExit(main())
