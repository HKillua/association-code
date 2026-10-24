"""096: compose completed same-policy 094 TAM and 093 TFM for full V.

This is evaluation only. The original 094 final score cache and threshold are
retained, while the 093 final fusion module processes its actual predicted
groups. No synthetic training checkpoint, optimizer, selection or refit is made.
Exit 75 means dependencies are still active and no output/GPU attempt is made.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import signal
import sys
import time

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.experiments import completion_value_fit as arm_checks


MODE = 'completion_system_evaluation_only'
WaitingDependencies = arm_checks.source_checks.WaitingDependencies


def _require(value, message):
    if not value:
        raise ValueError(message)


def _same(left, right, label):
    _require(arm_checks._same_tree(left, right), 'system source differs: ' + label)


def _terminal(directory, phase):
    """Inspect terminal journals before *either* source checkpoint is opened."""
    directory = Path(directory).resolve()
    path = directory / 'run.json'
    if not path.is_file():
        raise WaitingDependencies(f'{phase} has no run.json yet')
    run = arm_checks.source_checks._json(path)
    if run.get('status') in ('FAILED', 'INTERRUPTED'):
        raise ValueError(f'{phase} source is {run["status"]}; no source substitution or retry')
    if run.get('status') != 'COMPLETED':
        raise WaitingDependencies(f'{phase} status={run.get("status")}')
    _require(run.get('phase') == phase and run.get('final_evaluation_status') == 'COMPLETED',
             f'{phase} completed run has the wrong phase or unfinished evaluation')
    path = directory / 'completion-value-verification.json'
    if not path.is_file():
        raise WaitingDependencies(f'{phase} final wrapper verification is not yet available')
    verification = arm_checks.source_checks._json(path)
    _require(verification.get('status') == 'COMPLETED', f'{phase} endpoint verification failed')
    return directory


def _tam_records(arm):
    """Bind the exact fixed-final cache through checkpoint/history/summary."""
    import numpy as np
    from association.utils import runtime
    from tools.analysis.compare_completion_policies import _path_inside, _numbers
    from association.evaluation.association import evaluate_scenes
    saved, summary = arm['final'], arm['summary']
    selected, last = saved['selected'], saved['history'][-1]
    _same({key: value for key, value in selected.items() if key != 'states'},
          summary['selection'], 'TAM selection metadata')
    _same(selected['states'], saved['states'], 'TAM fixed-final selected parameters')
    _require(selected['stage'] == 'TAM_FROZEN'
             and selected['update'] == last['update'] == saved['cursor']['completed_updates'],
             'TAM selection is not the fixed final endpoint')
    threshold = selected['threshold']
    _require(threshold is None or (type(threshold) in (int, float) and math.isfinite(threshold)),
             'TAM original threshold must be finite or None')
    for item in (last['selection']['threshold'], summary['evaluation']['threshold']):
        _same(threshold, item, 'TAM original threshold')
    for path in (last['score_records'], summary['endpoint_score_records'],
                 summary['evaluation']['score_records']):
        _same(selected['score_records'], path, 'TAM original score cache')
    path = _path_inside(selected['score_records'], arm['directory'])
    records = runtime.load_records(path, expected_binding=dict(
        phase='tam', stage='TAM_FROZEN', update=selected['update'], split='V_select',
        threshold=threshold, effective_margin=saved['effective_margin'], test_opened=False))
    population = saved['data_contract'][1]
    _require(len(records) == len(population), 'TAM cache omits validation scenes')
    for record, row in zip(records, population):
        for key in ('scene_id', 'component_id', 'node_ids', 'identities', 'sources', 'intervals'):
            expected = np.asarray(row[key]).reshape(-1, 2) if key == 'intervals' else row[key]
            _require(np.array_equal(np.asarray(record[key]), np.asarray(expected)),
                     'TAM cache input/candidate identity differs: ' + key)
    _numbers(evaluate_scenes(records, threshold), summary['evaluation']['association'],
             'original TAM final association metrics')
    return records, threshold, path


def check_sources(tam_run, tfm_run, policy, manifest_path, alignment_path, *, cpu_threads=4):
    """Read and verify two terminal original runs without constructing models."""
    _require(policy in arm_checks.POLICIES, 'unknown completion policy')
    directories = {phase: _terminal(directory, phase)
                   for phase, directory in (('tam', tam_run), ('tfm', tfm_run))}
    _require(directories['tam'] != directories['tfm'], 'TAM and TFM must be distinct original runs')
    import torch
    from association.utils import runtime
    from association.data.archive import validate_manifest
    from tools.analysis.compare_completion_policies import _bind, _tfm_endpoints
    torch.set_num_threads(cpu_threads)
    arms = {phase: _bind(directory, phase, policy) for phase, directory in directories.items()}
    tam, tfm = (arms[phase]['final'] for phase in ('tam', 'tfm'))
    for key in ('normalizer', 'data_contract', 'model_config', 'fusion_config',
                'input_adapter', 'max_observations', 'split_unit', 'completion_policy'):
        _same(tam[key], tfm[key], key)
    for key in ('source_run', 'source_phase', 'source_completed_updates'):
        if key in tam['initialization'] or key in tfm['initialization']:
            _same(tam['initialization'].get(key), tfm['initialization'].get(key), 'initialization/' + key)
    source = Path(tam['initialization']['source_run']).resolve()
    frozen_keys = [key for key in tam['states'][0] if key in ('positions', 'heading_mean_std')
                   or key.startswith(('input_embedding.', 'tcm_encoder.', 'tcm_decoder.'))]
    _require(frozen_keys, 'shared frozen completion parameters missing')
    for key in frozen_keys:
        _same(tam['states'][0][key], tfm['states'][0][key], 'frozen completion/' + key)
    for key in ('singleton_policy', 'validation_batches', 'branch'):
        _same(tam['config'][key], tfm['config'][key], 'evaluation/' + key)
    # Check the TFM final snapshot against its own terminal weights, without
    # substituting its untrained TAM or choosing its better-looking endpoint.
    tfm_endpoints = _tfm_endpoints(arms['tfm'])
    records, threshold, score_path = _tam_records(arms['tam'])
    manifest, alignment = runtime.read_json(manifest_path), runtime.read_json(alignment_path)
    validate_manifest(manifest)
    profile = arm_checks._experiment_contract('tam', policy)
    _require(manifest.get('dataset_level') == 'Level1' and manifest.get('split_unit') == 'scene'
             and not manifest['splits']['T_head'], '096 requires the original Level1 scene split')
    data = dict(profile['target']['data'], length=profile['target']['model']['length'])
    for rows, split, count in zip(tam['data_contract'], ('T_core', 'V_select'),
            (profile['train_scenes'], profile['validation_scenes'])):
        _require([row['scene_id'] for row in rows] == manifest['splits'][split]
                 and len(rows) == count, 'source/manifest population differs: ' + split)
        for row in rows:
            sid = row['scene_id']
            _require(row['split'] == split and row['component_id'] == manifest['components'][sid]
                     and alignment.get(sid) == {key: row['reference_alignment'][key]
                        for key in ('time_offset_seconds', 'max_gap_seconds')},
                     'source/manifest reference alignment differs: ' + sid)
    report = dict(status='READY_DEPENDENCIES', completion_policy=policy,
        source_runs=dict(tcm=str(source), **{key: str(value) for key, value in directories.items()}),
        threshold=threshold, original_score_records=str(score_path),
        source_tam_update=tam['cursor']['completed_updates'],
        source_tfm_update=tfm['cursor']['completed_updates'],
        source_tcm_updates=arms['tam']['verified']['source_tcm_updates'],
        source_verification={phase: arms[phase]['verified'] for phase in arms},
        normalizer_equal=True, frozen_completion_equal=True, frozen_completion_tensor_count=len(frozen_keys),
        full_data_contract_equal=True, final_tam_selection_bound=True, final_tfm_snapshot_bound=True,
        validation_scene_ids=manifest['splits']['V_select'],
        validation_component_ids=[row['component_id'] for row in tam['data_contract'][1]],
        normalizer=tam['normalizer'], data=data,
        optimizer_updates=0, threshold_reselected=False, test_opened=False)
    return dict(arms=arms, records=records, report=report, manifest=manifest,
                alignment=alignment, data=data, tfm_final_endpoint=tfm_endpoints[-1])


def _prepared_validation_exact(scenes, endpoint, check):
    """Prevent old TAM scores being combined with changed archive inputs.

    The 093 final artifact retains original native inputs and aligned reference
    arrays. A semantic data_contract alone cannot detect changed floating data.
    """
    import numpy as np
    from association.evaluation.artifacts import load_scene_prediction
    from tools.analysis.compare_physical_predictions import _equal
    _require(len(scenes) == len(endpoint['paths']), 'TFM source predictions omit prepared scenes')
    for index, (scene, path) in enumerate(zip(scenes, endpoint['paths'])):
        check()
        saved = load_scene_prediction(path)
        _require(saved['metadata']['scene_index'] == index, 'TFM source scene order differs')
        for key, value in saved['scene'].items():
            _require(key in scene and _equal(value, scene[key]),
                     'prepared validation input differs from original TFM: ' + key)
        artifact = saved['methods']['oracle']
        targets = scene['fusion_targets']
        _require(artifact['target_ids'].tolist() == list(targets), 'prepared target order differs')
        for row, target in enumerate(targets.values()):
            _require(_equal(artifact['aligned_truth'][row], target['states'])
                     and _equal(artifact['aligned_truth_valid'][row], target['valid']),
                     'prepared reference arrays differ from original TFM')
        _require(not artifact['aligned_truth_valid'][len(targets):].any(),
                 'duplicate prediction rows cannot introduce target truth')
        del saved
    return dict(scene_count=len(scenes), native_inputs_and_reference_arrays_exact=True,
                scope='complete V arrays against original 093 final artifacts; training floats not replayed')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('tam-run', 'tfm-run', 'manifest', 'alignment'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--completion-policy', choices=arm_checks.POLICIES, required=True)
    parser.add_argument('--check-source-only', action='store_true')
    parser.add_argument('--archive', type=Path, nargs='+')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--device')
    parser.add_argument('--cpu-threads', type=int, default=4)
    parser.add_argument('--max-seconds', type=float, default=28800)
    parser.add_argument('--time-policy', choices=('advisory',), default='advisory')
    args = parser.parse_args(argv)
    started, owns_output = time.monotonic(), False
    state = dict(mode=MODE, completion_policy=args.completion_policy,
        source_tam_run=str(args.tam_run.resolve()), source_tfm_run=str(args.tfm_run.resolve()),
        optimizer_updates=0, normalizer_refit=False, margin_recalibrated=False,
        threshold_reselected=False, association_scores_reused=True, tam_scoring_forwards=0,
        time_policy=args.time_policy, advisory_seconds=args.max_seconds,
        test_opened=False, old066_acceptance='REJECTED')
    previous_signal = signal.getsignal(signal.SIGTERM)

    def write_run(status, **extra):
        from association.utils import runtime
        state.update(status=status, elapsed_seconds=time.monotonic()-started, **extra)
        runtime.write_json(args.output / 'run.json', state)

    def check():
        if owns_output and not state.get('advisory_reached') and time.monotonic()-started >= args.max_seconds:
            from association.utils import runtime
            state['advisory_reached'] = True
            runtime.write_json(args.output / 'time_advisory.json', dict(
                elapsed_seconds=time.monotonic()-started, advisory_seconds=args.max_seconds,
                action='continue the same fixed evaluation; no new training or retry'))

    def terminated(signum, frame):
        raise InterruptedError('SIGTERM interrupted the evaluation')

    try:
        _require(args.cpu_threads > 0 and math.isfinite(args.max_seconds) and args.max_seconds > 0,
                 'CPU threads and advisory duration must be positive')
        _require(args.check_source_only or (args.archive and args.output is not None and args.device),
                 'evaluation requires --archive, --output and explicit --device')
        if args.output is not None:
            output = args.output.resolve()
            _require(not any(output.is_relative_to(path.resolve()) or path.resolve().is_relative_to(output)
                             for path in (args.tam_run, args.tfm_run)),
                     'output must be separate from both original runs')
            _require(not output.exists(), 'output already exists; investigate the original attempt')
        bound = check_sources(args.tam_run, args.tfm_run, args.completion_policy,
            args.manifest, args.alignment, cpu_threads=args.cpu_threads)
        if args.output is not None:
            source = Path(bound['report']['source_runs']['tcm'])
            _require(not (args.output.resolve().is_relative_to(source)
                          or source.is_relative_to(args.output.resolve())),
                     'output must be separate from original TCM source')
        if args.check_source_only:
            report = bound['report']
            print(json.dumps(dict(status=report['status'], completion_policy=args.completion_policy,
                source_runs=report['source_runs'], source_tam_update=report['source_tam_update'],
                source_tfm_update=report['source_tfm_update'], threshold=report['threshold'],
                validation_scene_count=len(report['validation_scene_ids'])), ensure_ascii=False), flush=True)
            return 0
        import torch
        from association.utils import runtime
        from association.data.dataset import load_timed_dataset
        from association.training.tmtaff import _scenes, load_complete_snapshot
        from association.evaluation.tmtaff import evaluate_complete
        from association.evaluation.artifacts import ScenePredictionWriter
        args.output.mkdir(parents=True, exist_ok=False)
        owns_output = True
        signal.signal(signal.SIGTERM, terminated)
        state.update(source_binding=bound['report'], validation_scene_ids=bound['report']['validation_scene_ids'],
                     stage='preparation', advisory_reached=False, cpu_threads=args.cpu_threads)
        write_run('RUNNING')
        runtime.write_json(args.output / 'source-binding.json', bound['report'])
        check()
        dataset = load_timed_dataset(args.archive, bound['manifest'], train_splits=['T_core'],
            data_config=bound['data'], alignment_by_scene=bound['alignment'], check=check)
        tam, tfm = (bound['arms'][phase]['final'] for phase in ('tam', 'tfm'))
        _same(_scenes(dataset['train'], dataset['validation'], bound['data']['length']),
              tam['data_contract'], 'actual prepared scene contract')
        input_verification = _prepared_validation_exact(dataset['validation'], bound['tfm_final_endpoint'], check)
        runtime.write_json(args.output / 'preparation.json', dict(report=dataset['report'],
            source_binding=bound['report'], elapsed_seconds=time.monotonic()-started,
            validation_array_verification=input_verification,
            training_pool_purpose='original data contract check only; no normalization/refit/training'))
        validation = dataset['validation']
        del dataset
        del bound['tfm_final_endpoint']
        config = tam['config']
        state['stage'] = 'model_loading'
        write_run('RUNNING')
        device = runtime.device_from_string(args.device)
        loaded = load_complete_snapshot(args.tam_run / 'checkpoint.pt', selected=False, device=device)
        loaded['fusion'].load_state_dict(tfm['states'][1], strict=True)
        loaded['objective'].load_state_dict(tfm['states'][2], strict=True)
        for module, expected in zip((loaded['model'], loaded['fusion'], loaded['objective']),
                                   (tam['states'][0], tfm['states'][1], tfm['states'][2])):
            _same({key: value.detach().cpu() for key, value in module.state_dict().items()}, expected,
                  'composed actual module parameters')
            module.requires_grad_(False).eval()
        # The full successful training journals/initial snapshots are no
        # longer needed once actual module weights and V arrays are bound.
        # Keep compact source reports and final candidate records for output.
        del tam, tfm
        del bound['arms']
        state.update(stage='evaluation', device=str(device), provenance=runtime.provenance(device),
                     validation_array_verification=input_verification)
        write_run('RUNNING')

        def progress(item):
            check()
            runtime.write_json(args.output / 'evaluation-progress.json', item)

        writer = ScenePredictionWriter(args.output / 'predictions')
        evaluation = evaluate_complete(loaded['model'], loaded['fusion'], loaded['normalizer'], validation,
            device=device, batch_tracks=config['validation_batches']['tracks'],
            pair_batch_size=config['validation_batches']['pairs'], select_on_validation=False,
            threshold=bound['report']['threshold'], singleton_policy=config['singleton_policy'],
            association_records=bound['records'], predicted_controls=True,
            check=check, progress=progress, prediction_writer=writer)
        _require(evaluation['selection'] is None and evaluation['threshold'] == bound['report']['threshold']
                 and evaluation['association_scores_reused'] is True,
                 'evaluation changed original TAM selection')
        _require([row['scene_id'] for row in evaluation['per_scene']] == state['validation_scene_ids'],
                 'evaluation omitted validation scenes')
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        check()
        runtime.write_json(args.output / 'summary.json', dict(mode=MODE, completion_policy=args.completion_policy,
            source_binding=bound['report'], evaluation=evaluation, optimizer_updates=0,
            validation_array_verification=input_verification,
            threshold_reselected=False, test_opened=False,
            interpretation='same-policy frozen one-pass088-conditioned full-system pilot; '
                'not a sufficiently trained author-protocol reproduction; original V-selected TAM threshold retained'))
        write_run('COMPLETED', stage='completed', final_evaluation_status='COMPLETED',
            fusion_forward_count=evaluation['fusion_forward_count'])
        print(f'COMPLETED {args.completion_policy} full-system evaluation: {args.output.resolve()}', flush=True)
        return 0
    except WaitingDependencies as error:
        print(f'WAITING_DEPENDENCIES: {error}', flush=True)
        return 75
    except (Exception, KeyboardInterrupt) as error:
        interrupted = isinstance(error, (KeyboardInterrupt, InterruptedError))
        if owns_output:
            path = args.output / 'summary.json'
            if path.exists():
                path.replace(args.output / 'summary.incomplete.json')
            write_run('INTERRUPTED' if interrupted else 'FAILED', failure_type=type(error).__name__, failure=str(error))
        print(f'BLOCKED_OR_FAILED {type(error).__name__}: {error}', file=sys.stderr, flush=True)
        return 130 if interrupted else 1
    finally:
        if owns_output:
            signal.signal(signal.SIGTERM, previous_signal)


if __name__ == '__main__':
    raise SystemExit(main())
