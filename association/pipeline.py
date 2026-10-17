"""Validation-locked pilot orchestration, with one held-out evaluation phase."""
from __future__ import annotations

import gc
from pathlib import Path
import time

import numpy as np
import torch

from .config import requested_seeds, validate_config, validate_manifest_config
from .data import validate_manifest
from .dataset import load_split
from .heads import load_heads, predict_heads, train_heads
from .metrics import (brier_scores, compare_predictions, evaluate_scenes,
                      grouping_diagnostics, paired_cluster_bootstrap,
                      third_node_diagnostics)
from .reference import load_reference, score_reference, train_reference
from .runtime import (Run, code_fingerprint, digest, load_records, object_hash,
                      provenance, read_json, save_records, semantic_binding,
                      validate_record_split, write_json)

ROLES = ('A0', 'A_full')
BRANCHES = ('ce', 'contrastive')


def _artifacts(path, names):
    return {name: digest(Path(path)/name) for name in names}


def _release():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _head_binding(manifest, config, reference, seed):
    return {'manifest_sha256': manifest.get('manifest_sha256'), 'config_sha256': object_hash(config),
            'reference_checkpoint_sha256': reference.get('checkpoint_sha256'),
            'code_sha256': code_fingerprint(), 'seed': seed,
            'reference_role': reference['role'], 'reference_branch': reference['branch'],
            'training_split': 'T_head', 'selection_split': 'V_select',
            'training_scene_ids': manifest['splits']['T_head'],
            'validation_scene_ids': manifest['splits']['V_select']}


def _compare(records, thresholds, base, candidate, config, seed):
    return {'baseline': base, 'candidate': candidate,
            'paired_component_difference': paired_cluster_bootstrap(
                records[base], records[candidate], thresholds[base], thresholds[candidate],
                n_bootstrap=config['evaluation']['bootstrap_samples'], seed=seed,
                min_clusters=config['evaluation']['minimum_components']),
            'edge_transitions': compare_predictions(records[base], records[candidate],
                                                    thresholds[base], thresholds[candidate])}


def evaluate_locked(records, thresholds, config, seed):
    """No threshold fitting here: only externally frozen validation thresholds."""
    metrics = {name: evaluate_scenes(rows, thresholds[name]) for name, rows in records.items()}
    comparisons = {f'B_minus_{base}': _compare(records, thresholds, base, 'B', config, seed)
                   for base in ('A_full', 'A_prime', 'EndpointMarginal', 'PairMLP', 'ScenePool',
                                'FixedRule', 'B_shuffle')}
    if 'B_inference_shuffle' in records:
        comparisons['B_inference_shuffle_minus_B'] = _compare(
            records, thresholds, 'B', 'B_inference_shuffle', config, seed)
    common_threshold = thresholds['A_prime']
    return {'seed': seed, 'metrics': metrics, 'comparisons': comparisons,
            'primary_comparison': 'B_minus_A_full',
            'mechanism_comparison': 'B_minus_EndpointMarginal',
            'common_threshold_sensitivity': {
                name: evaluate_scenes(rows, common_threshold) for name, rows in records.items()
                if name not in ('A0', 'A_full')},
            'brier_calibrated_arms': {name: brier_scores(rows) for name, rows in records.items()
                                      if name not in ('A0', 'A_full')},
            'grouping': {name: [grouping_diagnostics(row, thresholds[name]) for row in records[name]]
                         for name in ('A_prime', 'A_full', 'B')},
            'third_node_original_scores': third_node_diagnostics(records['A_prime'], common_threshold),
            'interpretation': 'Exploratory association pilot. Grouping is a common closure diagnostic; '
                              'not official MTAD V4 or TMTAFF/TFM full-system evaluation.'}


def _markdown(report):
    rows = ['# CDF1 pilot execution report', '',
            'These are held-out results under this implementation and split. They are not a comparison '
            'to published paper scores, a full TMTAFF reproduction, or evidence of novelty.', '',
            '| Seed | Arm | Macro F1 | Micro F1 |', '| --- | --- | --- | --- |']
    for result in report['seeds']:
        for name, metrics in result['metrics'].items():
            macro, micro = metrics['macro_f1'], metrics['micro_f1']
            rows.append(f'| {result["seed"]} | {name} | {macro if macro is not None else "undefined"} | '
                        f'{micro if micro is not None else "undefined"} |')
    rows.extend(['', 'Primary: B minus A_full. Mechanism: B minus EndpointMarginal. '
                 'Per-seed component bootstrap, coverage, false bridges, third-neighbor joint errors, '
                 'and common-threshold sensitivity are in report.json.', '',
                 'Do not promote a weak or unconverged reference based only on this table. '
                 'Review reference learning curves, conformance differences, CUDA/device provenance '
                 'and quarantine/registration coverage before making a research claim.', ''])
    return '\n'.join(rows)


def run_pilot(archive_path, manifest, config, device, output, reference_root=None, seeds=None):
    """Train/select all seeds before opening test measurements, then evaluate once.

    reference_root is an all-or-error reuse mode. Missing/stale/partial fits are
    never silently retrained, overwritten, or treated as completed references.
    """
    validate_manifest(manifest)
    config = validate_config(config)
    if config['reference'].get('purpose') == 'probe':
        raise ValueError('probe configuration cannot run the full pilot; use probe-training')
    validate_manifest_config(manifest, config)
    device = torch.device(device)
    seeds = requested_seeds(config, seeds)
    maximum_hours = ((4 + 2 * (len(seeds)-1)) * config['reference']['max_hours']
                     + len(seeds)*config['heads']['max_hours'] + 3)
    run = Run(output, {'kind': 'pilot', 'provenance': provenance(device), 'seeds': seeds,
                      'manifest_sha256': manifest.get('manifest_sha256'), 'config_sha256': object_hash(config),
                      'reference_reuse': reference_root is not None}, maximum_hours)
    write_json(run.path/'config.json', config)
    write_json(run.path/'manifest.json', manifest)
    selected, branch_choices, selection_trace = {}, {}, []
    references = Path(reference_root).resolve() if reference_root else run.path/'references'
    try:
        # Complete training and validation for every requested seed first.
        for position, seed in enumerate(seeds):
            selected[seed] = {}
            for role in ROLES:
                candidates = {}
                for branch in (BRANCHES if position == 0 else (branch_choices[role],)):
                    run.check()
                    path = references/f'seed{seed}'/role/branch
                    if reference_root is None:
                        train_reference(archive_path, manifest, config, role, branch, seed, device, path)
                    model, normalizer, summary = load_reference(path, manifest, config, role, branch, seed, device)
                    candidates[branch] = (path, summary)
                    selection_trace.append({'seed': seed, 'role': role, 'branch': branch,
                                            'path': str(path.resolve()),
                                            'checkpoint_sha256': summary.get('checkpoint_sha256'),
                                            'validation': summary['validation']})
                    del model, normalizer
                    _release()
                # Earlier predeclared branch wins exact ties; no test input.
                best = max(candidates, key=lambda b: candidates[b][1]['validation']['macro_f1'])
                if position == 0:
                    branch_choices[role] = best
                path, summary = candidates[best]
                selected[seed][role] = {'path': str(path.resolve()), 'branch': best,
                                        'summary': summary, 'threshold': summary['threshold'],
                                        'artifacts': _artifacts(path, ('checkpoint.pt', 'summary.json', 'run.json'))}
            a0 = selected[seed]['A0']
            head_records = load_records(Path(a0['path'])/'T_head', a0['summary']['cache_bindings']['T_head'])
            val_records = load_records(Path(a0['path'])/'V_select', a0['summary']['cache_bindings']['V_select'])
            validate_record_split(head_records, manifest, 'T_head')
            validate_record_split(val_records, manifest, 'V_select')
            binding = _head_binding(manifest, config, a0['summary'], seed)
            head_path = run.path/f'seed{seed}'/'heads'
            choices = train_heads(head_records, val_records, config, seed, device, head_path, binding)
            selected[seed]['heads'] = {'path': str(head_path.resolve()), 'binding': binding, 'arms': choices,
                                       'selection_sha256': digest(head_path/'selection.json'),
                                       'artifacts': _artifacts(head_path, ('heads.pt', 'calibrator.pt', 'selection.json', 'run.json'))}
            del head_records, val_records
            _release()
        frozen = {'schema': 1, 'phase': 'all_validation_selection_complete_before_test_read',
                  'manifest_sha256': manifest.get('manifest_sha256'), 'config_sha256': object_hash(config),
                  'code_sha256': code_fingerprint(), 'branch_choices': branch_choices,
                  'branch_candidates': selection_trace, 'selected': selected,
                  'seed_policy': 'first configured seed selects reference branches; later seeds inherit',
                  'threshold_policy': 'each fitted arm uses its own V_select threshold; ties prefer higher'}
        frozen['content_sha256'] = object_hash(frozen)
        write_json(run.path/'selection_frozen.json', frozen)
        # Single evaluation gate: everything that can be selected is now on disk.
        run.check()
        test_scenes = load_split(archive_path, manifest, 'T_test', config['model']['length'])
        write_json(run.path/'test_audit.json', [{'scene_id': scene['scene_id'], 'audit': scene['audit']}
                                              for scene in test_scenes])
        reports = []
        for seed in seeds:
            run.check()
            raw = {}
            thresholds = {}
            inference_times = {}
            for role in ROLES:
                entry = selected[seed][role]
                model, normalizer, loaded_summary = load_reference(entry['path'], manifest, config, role,
                                                                    entry['branch'], seed, device)
                if semantic_binding(loaded_summary) != semantic_binding(entry['summary']):
                    raise ValueError('Reference differs from frozen validation selection')
                if device.type == 'cuda':
                    torch.cuda.synchronize(device)
                started = time.monotonic()
                raw[role] = score_reference(model, normalizer, test_scenes, device,
                                             batch_tracks=config['reference']['batch_tracks'], check=run.check)
                if device.type == 'cuda':
                    torch.cuda.synchronize(device)
                inference_times[role] = time.monotonic()-started
                thresholds[role] = entry['threshold']
                del model, normalizer
                _release()
            head_info = selected[seed]['heads']
            bundle = load_heads(head_info['path'], device, expected_binding=head_info['binding'])
            if bundle['selection'] != head_info['arms']:
                raise ValueError('Head selection differs from frozen validation selection')
            started = time.monotonic()
            corrected = predict_heads(bundle, raw['A0'], seed, split='T_test')
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            inference_times['all_heads_and_context'] = time.monotonic()-started
            thresholds.update({name: choice['threshold'] for name, choice in head_info['arms'].items()})
            if 'B_inference_shuffle' in corrected:
                thresholds['B_inference_shuffle'] = thresholds['B']
            records = {**raw, **corrected}
            result = evaluate_locked(records, thresholds, config, seed)
            result['inference_seconds'] = inference_times
            result['selection'] = head_info['arms']
            for name, rows in records.items():
                save_records(run.path/f'seed{seed}'/'test_scores'/name, rows,
                             {'selection_sha256': frozen['content_sha256'], 'seed': seed, 'arm': name})
            write_json(run.path/f'seed{seed}'/'evaluation.json', result)
            run.check()
            reports.append(result)
            del bundle, raw, corrected, records
            _release()
        # Seeds share the test population. Do not pool edges or bootstrap seeds
        # as though they were independent data samples.
        effects = [r['comparisons']['B_minus_A_full']['paired_component_difference']['difference']
                   for r in reports]
        usable = [effect for effect in effects if effect is not None]
        report = {'protocol': config['protocol'], 'selection_sha256': frozen['content_sha256'],
                  'manifest_sha256': manifest.get('manifest_sha256'), 'seeds': reports,
                  'across_seed_primary': {'mean_difference': float(np.mean(usable)) if usable else None,
                                         'range': [min(usable), max(usable)] if usable else None,
                                         'note': 'Descriptive across seeds on the same test set; not an independent CI'},
                  'gpu_peak_allocated_bytes': torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else None,
                  'scientific_status': 'exploratory; requires reference convergence and conformance review'}
        run.check()
        write_json(run.path/'report.json', report)
        (run.path/'report.md').write_text(_markdown(report), encoding='utf-8')
        run.finish(report_sha256=digest(run.path/'report.json'))
        return report
    except BaseException as exc:
        run.finish('FAILED', error_type=type(exc).__name__, error=str(exc))
        raise
