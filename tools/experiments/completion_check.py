"""Read existing 020/021 weights and diagnose completion on four fixed V scenes.

No training, helper tracks, association, fusion, calibration or threshold search.
Time is advisory in launch_detached.py; this command has no wall-time kill.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import signal
import sys
import time

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

SCENE_IDS = ['1671', '2056', '290', '113']


def load_scenes(archive_path, manifest, alignment, data):
    """Read only the fixed development subset, retaining the original manifest."""
    import numpy as np
    from association.data.archive import MTADArchive, validate_manifest
    from association.data.native import _record
    from association.data.reference_states import iter_reference_states
    from association.data.targets import build_scene_fusion_targets
    from association.data.timed import prepare_timed_scene
    validate_manifest(manifest)
    if manifest['splits']['V_select'][:len(SCENE_IDS)] != SCENE_IDS:
        raise ValueError('fixed original development scene order differs')
    scenes = []
    with MTADArchive(archive_path) as archive:
        for sid in SCENE_IDS:
            native = _record(archive.load_scene(sid), manifest['components'][sid],
                             'V_select', data['max_observations'])
            labels = archive.read_labels(sid)
            query = None
            if not native['tracks']:
                query = np.linspace(min(row['t_s'] for row in labels),
                                    max(row['t_e'] for row in labels), data['length'])
            scene = prepare_timed_scene(native, **data, query_times=query)
            scene['target_identity_ids'] = list(dict.fromkeys(
                '-'.join(row['mmsi'].split('-')[:2]) for row in labels))
            scenes.append(scene)
        members = list(archive.members_read)
    references = []
    for scene, reference in zip(scenes, iter_reference_states(archive_path, manifest,
            split='V_select', scene_ids=SCENE_IDS)):
        sid = scene['scene_id']
        if reference['scene_id'] != sid:
            raise ValueError('reference scene order differs')
        scene['fusion_targets'] = build_scene_fusion_targets(scene, reference,
            target_identity_ids=scene['target_identity_ids'], longitude_policy='shortest_arc', **alignment[sid])
        references.append(dict(scene_id=sid, member=reference['member']))
    return scenes, dict(observation_members_read=members, reference_members_read=references)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('archive', 'manifest', 'alignment', 'reference-checkpoint',
                 'candidate-checkpoint', 'output', 'device'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--cpu-threads', type=int, default=4)
    args = parser.parse_args(argv)
    import torch
    from association.utils import runtime
    from association.models.tmtaff import ModelConfig
    from association.models.timed import TimedAssociationModel
    from association.training.cooperative import load_cooperative_checkpoint, resolve_device
    from association.evaluation.tmtaff import evaluate_tcm_completion
    from association.evaluation.cooperative import evaluate_cooperative_completion

    started, output = time.monotonic(), Path(args.output)
    run = dict(status='RUNNING', stage='loading', command=[sys.executable, __file__, *(argv or sys.argv[1:])],
        sources={name: str(Path(getattr(args, name)).resolve()) for name in
                 ('archive', 'manifest', 'alignment', 'reference_checkpoint', 'candidate_checkpoint')},
        device=args.device, cpu_threads=args.cpu_threads, scene_ids=SCENE_IDS,
        optimizer_updates=0, normalizer_refit=False, margin_recalibrated=False,
        threshold_reselected=False, test_opened=False)
    previous_signal = signal.getsignal(signal.SIGTERM)
    owns_output = False

    def progress(event):
        run.update(elapsed_seconds=time.monotonic()-started, progress=event)
        runtime.write_json(output/'run.json', run)

    def terminated(signum, frame):
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        raise InterruptedError('completion check interrupted by SIGTERM')

    try:
        output.mkdir(parents=True, exist_ok=False)
        owns_output = True
        signal.signal(signal.SIGTERM, terminated)
        if args.cpu_threads < 1:
            raise ValueError('cpu threads must be positive')
        torch.set_num_threads(args.cpu_threads)
        progress(dict(stage='loading'))
        device = resolve_device(args.device)
        saved = torch.load(args.reference_checkpoint, map_location='cpu', weights_only=True)
        normalizer = saved['normalizer']
        keys = ('length', 'max_observations', 'time_unit_seconds')
        data = {key: normalizer['adapter_config'][key] for key in keys}
        candidate, payload = load_cooperative_checkpoint(args.candidate_checkpoint, device='cpu')
        if (saved.get('input_adapter') != 'native_time_v1'
                or saved['cursor'] != dict(stage=1, update=300, completed_updates=1300)
                or payload['stage'] != 'teacher' or payload['completed_updates'] != 1000):
            raise ValueError('expected preserved TCM1000/TAM300 and teacher1000 endpoints')
        candidate_data = {key: payload['normalizer']['adapter_config'][key] for key in keys}
        if data != candidate_data or saved['max_observations'] != data['max_observations']:
            raise ValueError('checkpoint input adapters differ')
        manifest, alignment = runtime.read_json(args.manifest), runtime.read_json(args.alignment)
        if any(item['training_scene_ids'] != manifest['splits']['T_core']
               or item['training_splits'] != ['T_core'] for item in (normalizer, payload['normalizer'])):
            raise ValueError('saved normalization does not match original T_core population')
        contract = {row['scene_id']: row for row in saved['data_contract'][1]}
        for sid in SCENE_IDS:
            row = contract[sid]
            policy = row['reference_alignment']
            if (row['split'] != 'V_select' or row['component_id'] != manifest['components'][sid]
                    or policy.get('longitude_policy', 'linear') != 'shortest_arc'
                    or alignment[sid] != {key: policy[key] for key in ('time_offset_seconds', 'max_gap_seconds')}):
                raise ValueError('reference assembly differs from preserved training: ' + sid)
        run.update(data=data, reference_cursor=saved['cursor'], candidate_stage=payload['stage'],
                   candidate_updates=payload['completed_updates'], stage='preparation',
                   reference_normalizer=normalizer, candidate_normalizer=payload['normalizer'],
                   candidate_support_time_scale_tau=payload['support_time_scale_tau'])
        progress(dict(stage='preparation'))
        scenes, reads = load_scenes(args.archive, manifest, alignment, data)
        run.update(**reads, fragment_count=sum(len(scene['node_ids']) for scene in scenes))
        config = replace(ModelConfig(**saved['model_config']), margin=saved['effective_margin'])
        reference = TimedAssociationModel(config, saved['config']['branch'],
                                           max_observations=saved['max_observations'])
        reference.load_state_dict(saved['states'][0], strict=True)
        reference.to(device).eval()
        del saved
        run['stage'] = 'reference_completion'
        progress(dict(stage=run['stage']))
        reference_report = evaluate_tcm_completion(reference, normalizer, scenes, device=device, progress=progress)
        runtime.write_json(output/'reference.json', reference_report)
        del reference
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        run['stage'] = 'candidate_completion'
        progress(dict(stage=run['stage']))
        candidate.to(device).eval()
        candidate_report = evaluate_cooperative_completion(candidate, payload['normalizer'], scenes,
                                                           device=device, progress=progress)
        runtime.write_json(output/'candidate.json', candidate_report)
        runtime.write_json(output/'summary.json', dict(reference=reference_report, candidate=candidate_report,
            scope='four fixed development scenes; not full V or an industry benchmark',
            metric_warning='reference latent MSE and candidate physical error are not a common ranking',
            test_opened=False))
        run.update(status='COMPLETED', stage='completed')
        progress(dict(stage='completed'))
        print(f'COMPLETED: {run["fragment_count"]} fragments; {output}')
        return 0
    except BaseException as error:
        if owns_output:
            run.update(status='INTERRUPTED' if isinstance(error, (InterruptedError, KeyboardInterrupt)) else 'FAILED',
                       error=f'{type(error).__name__}: {error}')
            progress(run.get('progress', {}))
        print(f'{type(error).__name__}: {error}', file=sys.stderr)
        return 1
    finally:
        signal.signal(signal.SIGTERM, previous_signal)


if __name__ == '__main__':
    raise SystemExit(main())
