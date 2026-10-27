"""One non-training TMTAFF data preparation with public count-only outputs.

Uses the production loader without fitting normalization, constructing a model,
reading weights, selecting a budget or opening sealed-test payloads.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import gzip
import json
import math
import os
from pathlib import Path
import re
import resource
import signal
import sys
import time
import traceback

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def write_gzip(path, value):
    with gzip.open(path, 'wt', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False)


def address_space_limit(maximum):
    """Constrain only this Linux process, retaining a lower inherited limit."""
    if not sys.platform.startswith('linux'):
        return None
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    available = [maximum] + [x for x in (soft, hard) if x != resource.RLIM_INFINITY]
    selected = min(available)
    resource.setrlimit(resource.RLIMIT_AS, (selected, hard))
    return selected


def member_inventory(paths, manifest, maximum):
    from association.data.archive import (_name, _scene_id, open_mtad_zip,
        validate_dataset_level, validate_manifest, zip_dataset_level)
    validate_manifest(manifest)
    requested = {sid: split for split in ('T_core', 'V_select') for sid in manifest['splits'][split]}
    patterns = {'labels': r'(?:^|/)关联表/关联结果-(\d+)\.csv$',
                'measurements': r'(?:^|/)量测场景/场景-(\d+)\.csv$',
                'reference': r'(?:^|/)真实场景/场景-(\d+)\.csv$'}
    inventory, seen = [], set()
    with open_mtad_zip(paths) as archive:
        validate_dataset_level(zip_dataset_level(archive), manifest)
        for info in archive.infolist():
            name = _name(info)
            if info.is_dir():
                continue
            for kind, pattern in patterns.items():
                match = re.search(pattern, name)
                if not match or _scene_id(match.group(1)) not in requested:
                    continue
                sid = _scene_id(match.group(1))
                if (sid, kind) in seen:
                    raise ValueError('Duplicate requested member')
                seen.add((sid, kind))
                inventory.append(dict(scene_id=sid, split=requested[sid], kind=kind, member=name,
                    compressed_bytes=info.compress_size, declared_uncompressed_bytes=info.file_size,
                    exceeds_size_limit=info.file_size > maximum))
    if len(inventory) != len(requested) * 3:
        raise ValueError('Missing requested members')
    return inventory


def summarize_scene(scene):
    """Aggregate the actual positive-pair supervision; never export identities."""
    import numpy as np
    from association.data.physical_units import SPEED_UNIT_POLICY, validate_target_units
    if validate_target_units(scene) != SPEED_UNIT_POLICY:
        raise ValueError('New-unit contract required')
    observed = np.asarray(scene['native_observed'])
    counts = observed.sum(axis=1)
    eligible = counts >= 2  # Same native masks as pack_timed_scene / TCM transitions.
    if not eligible.all():
        raise ValueError('Production native loader emitted a node with fewer than two observations')
    labels, pairs = np.asarray(scene['labels']), np.asarray(scene['pairs'])
    positive_indices = np.flatnonzero(labels == 1)
    fields = ('lat', 'lon', 'vel', 'cou', 'position_joint')
    length = len(scene['grid'])
    histograms = {key: [0] * (length + 1) for key in fields}
    target_counts = {key: 0 for key in fields}
    any_empty = all_empty = 0
    for target in scene['fusion_targets'].values():
        valid = np.asarray(target['valid'], dtype=bool)
        values = [int(valid[:, c].sum()) for c in range(4)] + [int((valid[:, 0] & valid[:, 1]).sum())]
        for key, count in zip(fields, values):
            target_counts[key] += count
    for index in positive_indices:
        left, right = pairs[index]
        identity = str(scene['identities'][left])
        if identity != str(scene['identities'][right]):
            raise ValueError('Positive pair identity mismatch')
        valid = np.asarray(scene['fusion_targets'][identity]['valid'], dtype=bool)
        values = [int(valid[:, c].sum()) for c in range(4)] + [int((valid[:, 0] & valid[:, 1]).sum())]
        for key, count in zip(fields, values):
            histograms[key][count] += 1
        any_empty += int(any(v == 0 for v in values[:4]))
        all_empty += int(all(v == 0 for v in values[:4]))
    audit = scene['audit']
    return dict(scene_id=scene['scene_id'], split=scene['split'], node_count=len(counts),
        native_point_count=int(counts.sum()), tcm_eligible_tracks=int(eligible.sum()),
        tcm_bidirectional_transitions=int(2 * (counts[eligible] - 1).sum()),
        candidate_count=len(pairs), positive_count=len(positive_indices), negative_count=len(pairs)-len(positive_indices),
        target_count=len(scene['fusion_targets']),
        target_without_input_count=len(set(scene['fusion_targets']) - set(scene['identities'])),
        target_without_reference_count=sum(int(t['report']['missing_identity']) for t in scene['fusion_targets'].values()),
        target_valid_points=target_counts, positive_pair_valid_query_histograms=histograms,
        positive_pair_valid_query_totals={key: sum(i*n for i,n in enumerate(h)) for key,h in histograms.items()},
        positive_pairs_with_any_empty_attribute=any_empty, positive_pairs_without_any_supervision=all_empty,
        quarantine_counts={key: int(audit.get(key, 0)) for key in (
            'candidate_nodes', 'valid_nodes', 'quarantined_nodes', 'exact_duplicate_rows_removed',
            'measurement_rows_outside_label_interval_diagnostic_only', 'positive_edges_lost_to_quarantine')},
        quarantine_reasons={key: int(value) for key,value in audit.get('quarantine_reasons', {}).items()})


def summarize_population(scenes, step_options, batch):
    result = {}
    scalar_keys = ('node_count', 'native_point_count', 'tcm_eligible_tracks', 'tcm_bidirectional_transitions',
        'candidate_count', 'positive_count', 'negative_count', 'target_count', 'target_without_input_count',
        'target_without_reference_count', 'positive_pairs_with_any_empty_attribute', 'positive_pairs_without_any_supervision')
    for split in ('T_core', 'V_select'):
        rows = [x for x in scenes if x['split'] == split]
        item = dict(scene_count=len(rows), **{key: sum(row[key] for row in rows) for key in scalar_keys})
        for field in ('target_valid_points', 'positive_pair_valid_query_totals', 'quarantine_counts', 'quarantine_reasons'):
            merged = Counter()
            for row in rows:
                merged.update(row[field])
            item[field] = dict(merged)
        if rows:
            item['positive_pair_valid_query_histograms'] = {key: [sum(row['positive_pair_valid_query_histograms'][key][i]
                for row in rows) for i in range(len(hist))]
                for key,hist in rows[0]['positive_pair_valid_query_histograms'].items()}
        pools = [row['positive_count'] for row in rows if row['positive_count']]
        item['positive_eligible_scenes'] = len(pools)
        item['positive_pool_smaller_than_pair_batch'] = sum(n < batch for n in pools)
        if split == 'T_core':
            expected = []
            for steps in step_options:
                if not pools:
                    expected.append(dict(steps=steps, pair_presentations=0,
                        expected_unique_scenes=0., expected_unique_pairs=0., expected_positive_pair_fraction=None))
                    continue
                nscene = len(pools)
                pair_cover = 0.
                for size in pools:
                    conditional = -math.expm1(batch * math.log1p(-1/size)) if 1 < size < batch else (1. if size == 1 else batch/size)
                    probability = conditional/nscene
                    covered = 1. if probability == 1 else -math.expm1(steps * math.log1p(-probability))
                    pair_cover += size*covered
                scene_cover = nscene if nscene == 1 else nscene * -math.expm1(steps*math.log1p(-1/nscene))
                expected.append(dict(steps=steps, pair_presentations=steps*batch,
                    expected_unique_scenes=scene_cover, expected_unique_pairs=pair_cover,
                    expected_positive_pair_fraction=pair_cover/sum(pools)))
            item['exposure_expectations_not_training'] = expected
        result[split] = item
    return result


def validate_config(config):
    from association.data.physical_units import SPEED_UNIT_POLICY
    required = {'scope','train_splits','data','expected','max_member_bytes','max_address_space_bytes',
                'pair_batch_size','exposure_step_options'}
    if set(config) != required or config['scope'] != 'TMTAFF-unit-population-inspection-v1':
        raise ValueError('Explicit population inspection config required')
    if config['train_splits'] != ['T_core'] or config['data'].get('speed_unit_policy') != SPEED_UNIT_POLICY:
        raise ValueError('Only new-unit T_core plus V_select preparation permitted')
    for key in ('max_member_bytes','max_address_space_bytes','pair_batch_size'):
        if isinstance(config[key], bool) or not isinstance(config[key], int) or config[key] <= 0:
            raise ValueError('Limits and batch size must be positive integers')
    if (not isinstance(config['exposure_step_options'], list) or not config['exposure_step_options']
            or any(isinstance(x,bool) or not isinstance(x,int) or x < 1 for x in config['exposure_step_options'])):
        raise ValueError('Exposure steps must be positive integers')
    if set(config['expected']) != {'T_core','V_select'}:
        raise ValueError('Expected split counts required')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, nargs='+', required=True)
    for name in ('manifest','alignment','config','output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--cpu-threads', type=int, default=4)
    args = parser.parse_args(argv)
    if not 1 <= args.cpu_threads <= 4:
        parser.error('cpu-threads must be 1..4')
    for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
        os.environ[key] = str(args.cpu_threads)
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    if args.output.exists():
        raise FileExistsError('Output already exists; never overwrite or retry in place')
    args.output.mkdir(parents=True)
    private = args.output/'private'
    private.mkdir(mode=0o700)
    os.chmod(private, 0o700)
    started = time.monotonic()
    run = dict(status='RUNNING', idea_id='IDEA-TMTAFF-UNIT-PREP-20261027', plan_version='1.0',
        start_utc=datetime.now(timezone.utc).isoformat(), argv=sys.argv if argv is None else argv,
        cpu_threads=args.cpu_threads, model_constructed=False, gpu_training_started=False,
        optimizer_updates=0, model_forwards=0, checkpoint_files_read=0, test_opened=False,
        normalizer_fitted=False, completed_observation_scenes=0, completed_reference_scenes=0)
    write_json(args.output/'run.json', run)
    prior_signal = signal.getsignal(signal.SIGTERM)
    def terminated(signum, frame):
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        raise InterruptedError('SIGTERM')
    signal.signal(signal.SIGTERM, terminated)
    try:
        config = json.loads(args.config.read_text())
        validate_config(config)
        manifest = json.loads(args.manifest.read_text())
        alignment = json.loads(args.alignment.read_text())
        run['configuration'] = config
        run['address_space_limit_bytes'] = address_space_limit(config['max_address_space_bytes'])
        run['address_space_limit_scope'] = 'this process virtual address space; not RSS'
        write_json(args.output/'run.json', run)
        inventory = member_inventory(args.archive, manifest, config['max_member_bytes'])
        write_gzip(args.output/'member-inventory.json.gz', inventory)
        if any(item['exceeds_size_limit'] for item in inventory):
            raise ValueError('Requested CSV exceeds fixed declared size limit')
        for split in ('T_core', 'V_select'):
            if len(manifest['splits'][split]) != config['expected'][split]['scene_count']:
                raise ValueError('Manifest scene count differs from fixed preparation contract')
        from association.data.dataset import load_timed_dataset
        with (args.output/'progress.jsonl').open('a') as journal:
            def progress(event):
                if event['status'] == 'COMPLETED':
                    key = 'completed_observation_scenes' if event['stage']=='observations' else 'completed_reference_scenes'
                    run[key] += 1
                run['last_event'] = event
                journal.write(json.dumps({**event,'elapsed_seconds':time.monotonic()-started})+'\n')
                journal.flush()
                write_json(args.output/'run.json',run)
            dataset = load_timed_dataset(args.archive, manifest, train_splits=config['train_splits'],
                data_config=config['data'], alignment_by_scene=alignment, progress=progress)
        write_gzip(private/'preparation.json.gz', dataset['report'])
        scenes = [summarize_scene(scene) for scene in [*dataset['train'],*dataset['validation']]]
        summary = summarize_population(scenes,config['exposure_step_options'],config['pair_batch_size'])
        mismatches = []
        for split, expected in config['expected'].items():
            for key,value in expected.items():
                actual = summary[split][key]
                if actual != value:
                    mismatches.append(dict(split=split,field=key,expected=value,actual=actual))
        actual_members = dataset['report']['observation_members_read'] + dataset['report']['reference_members_read']
        actual_names = [x['member'] for x in actual_members]
        if len(actual_names) != len(inventory) or set(actual_names) != {x['member'] for x in inventory}:
            mismatches.append(dict(field='member_scope_or_read_count',expected=len(inventory),actual=len(actual_names)))
        run.update(status='COMPLETED' if not mismatches else 'COMPLETED_WITH_MISMATCHES',
            mismatch_count=len(mismatches), read_member_count=len(actual_names),
            declared_uncompressed_bytes_successful_members=sum(x['declared_uncompressed_bytes'] for x in inventory
                if x['member'] in set(actual_names)))
        public = dict(populations=summary, mismatches=mismatches,
            limitations=['counts do not validate reference truth or training sufficiency',
                'exposure options are analytic expectations, not training',
                'private preparation contains identity-bearing audits; do not commit it'])
        write_gzip(args.output/'scenes.json.gz',scenes)
        write_gzip(args.output/'summary.json.gz',public)
        write_gzip(args.output/'read-members.json.gz',actual_members)
        print(run['status']+'; preparation only, no training',flush=True)
        return 0 if not mismatches else 2
    except (Exception, KeyboardInterrupt) as error:
        run.update(status='FAILED',error_type=type(error).__name__)
        (private/'traceback.txt').write_text(traceback.format_exc())
        print('FAILED '+type(error).__name__+'; details retained server-side',file=sys.stderr,flush=True)
        return 130 if isinstance(error,KeyboardInterrupt) else 1
    finally:
        run.update(end_utc=datetime.now(timezone.utc).isoformat(),elapsed_seconds=time.monotonic()-started,
            peak_rss_native_units=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            peak_rss_unit='KiB' if sys.platform.startswith('linux') else 'bytes')
        write_json(args.output/'run.json',run)
        signal.signal(signal.SIGTERM,prior_signal)


if __name__ == '__main__':
    raise SystemExit(main())
