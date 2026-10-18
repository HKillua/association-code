"""Validate proposed configs and simulate sampling from saved metadata only.

No ZIP/reference read, model construction, optimization, GPU or T_test access.
Output is proposed exposure and cost arithmetic, never measured model quality.
"""
import copy
import gzip
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
from association.complete_training import _configuration
from association.cooperative import CooperativeConfig
from association.cooperative_training import training_configuration
from association.fusion import PairFusionConfig
from association.models import ModelConfig


def read(path):
    return json.loads(path.read_text())


def validated():
    reference = read(HERE/'reference-development.template.json')
    candidate = read(HERE/'candidate-development.template.json')
    probe = read(HERE/'candidate-cost-proposal.json')
    model, fusion = ModelConfig(**reference['model']), PairFusionConfig(**reference['fusion'])
    assert (model.length, model.d_model) == (fusion.length, fusion.d_model) == (64, 128)
    original = read(ROOT/'configs/tmtaff-time-cost-probe.json')
    restored = copy.deepcopy(reference)
    for key in ('updates', 'validate_every_updates', 'max_hours'):
        restored['training'][key] = original['training'][key]
    assert restored == original
    for cfg, validator, budget in ((reference, _configuration, 'max_hours'),
                                  (candidate, training_configuration, 'max_seconds')):
        assert cfg['training'][budget] is None
        try:
            validator(cfg['training'])
        except ValueError:
            pass
        else:
            raise AssertionError('unapproved budget template must be rejected')
        # A temporary positive value checks every other field; no file, task or
        # runtime is created, and this value is not a resource recommendation.
        checked = dict(cfg['training'], **{budget: 1.})
        validator(checked)
    for cfg in (candidate, probe):
        assert cfg['scope'] == 'cooperative-physical-v1' and cfg['train_splits'] == ['T_core']
        assert cfg['data'] == {'time_unit_seconds': 20.}
        CooperativeConfig(**cfg['model'])
    training_configuration(probe['training'])
    assert reference['data'] == {'max_observations': 128, 'time_unit_seconds': 20.}
    return reference, candidate, probe


def main():
    reference, candidate, probe = validated()
    path = ROOT/'docs/results/frozen_input_audit_20261018/native_sampling/audit_native_sampling.py'
    spec = importlib.util.spec_from_file_location('saved_sampling_audit', path)
    audit = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(audit)
    check = audit.selfcheck()
    path = path.parent/'attempt2/scene_metadata.jsonl.gz'
    with gzip.open(path, 'rt') as stream:
        rows = [json.loads(line) for line in stream]
    rows = [r for r in rows if r['split'] == 'T_core']
    manifest = read(ROOT/'HandOver/results/EVIDENCE-20261018-012/prepared007/manifest.json')
    by_scene = {r['scene_id']: r for r in rows}
    rows = [by_scene[sid] for sid in manifest['splits']['T_core']]
    pools = [audit.CompactPools([t['identity_id'] for t in r['tracks']]) for r in rows]
    ids = [r['scene_id'] for r in rows]
    total_tracks = sum(p.n for p in pools)
    total_positive = sum(p.count[1] for p in pools)
    assert (len(rows), total_tracks, total_positive) == (512, 54792, 110733)

    def coverage(state):
        result = audit.covjson(state, ids)
        result['track_percent'] = 100*result['unique_tracks']/total_tracks
        result['unique_positive_pairs'] = sum(pools[s].ids[a] == pools[s].ids[b]
            for s, pair in state['pairs'] for a, b in pools[s].endpoints(np.array([pair])))
        result['positive_pool_percent'] = 100*result['unique_positive_pairs']/total_positive
        return result

    def candidate_plan(cfg):
        plan = {}
        for stage, count, seed in [('teacher', cfg['teacher_updates'], cfg['seed']),
                                   ('utility', cfg['utility_updates'], cfg['seed']+1),
                                   ('adapt_per_arm', cfg['adapt_updates'], cfg['seed']+2)]:
            sampler = audit.CompactSampler(pools, seed)
            sampled, used = audit.fresh(), audit.fresh()
            for _ in range(count):
                groups = sampler.sample(cfg['batch_size'])
                audit.coverage(stage, groups, pools, sampled)
                if stage == 'adapt_per_arm':
                    positives = []
                    for scene, pairs in groups:
                        endpoints = pools[scene].endpoints(pairs)
                        keep = pools[scene].ids[endpoints[:, 0]] == pools[scene].ids[endpoints[:, 1]]
                        if keep.any():
                            positives.append((scene, pairs[keep]))
                    groups = positives
                audit.coverage(stage, groups, pools, used)
            plan[stage] = dict(updates=count, seed=seed, sampled=coverage(sampled),
                               actually_used_by_loss=coverage(used))
        plan['four_arms_share_same_draws'] = True
        plan['total_optimizer_updates'] = cfg['teacher_updates']+cfg['utility_updates']+len(cfg['routes'])*cfg['adapt_updates']
        return plan

    track_rng, tfm_rng = np.random.default_rng(20), np.random.default_rng(21)
    sampler = audit.CompactSampler(pools, 20)
    track_map = [(s, i) for s, p in enumerate(pools) for i in range(p.n)]
    reference_coverage = {}
    for stage, count in reference['training']['updates'].items():
        state = audit.fresh()
        if stage == 'TCM':
            seen = set()
            for _ in range(count):
                seen.update(map(int, track_rng.choice(total_tracks, 64, replace=False)))
            reference_coverage[stage] = dict(updates=count, track_presentations=count*64,
                unique_tracks=len(seen), track_percent=100*len(seen)/total_tracks,
                unique_scenes=len({track_map[i][0] for i in seen}))
            continue
        for _ in range(count):
            if stage == 'TFM_FROZEN':
                scene = int(tfm_rng.choice(sampler.eligible[1]))
                pool = pools[scene]
                ranks = tfm_rng.choice(pool.count[1], 64, replace=pool.count[1] < 64)
                groups = [(scene, pool.select(1, ranks))]
            else:
                groups = sampler.sample(64)
            audit.coverage(stage, groups, pools, state)
        reference_coverage[stage] = dict(updates=count, **coverage(state))

    diagnostic = read(ROOT/'docs/results/tmtaff013_decisions/diagnostics.json')
    timing = diagnostic['timing']
    measured_validation = timing['validation_seconds']/2
    known = diagnostic['arithmetic_scenarios']['100_100_100_20']
    result = dict(status='CPU_PLAN_CHECKED_NOT_EXECUTED', native_data_read=False,
        model_constructed=False, test_opened=False, sampler_fixture_check=check,
        configs='Both budget-null development templates reject; all remaining fields and proposed cost config validate.',
        population=dict(train_scenes=len(rows), tracks=total_tracks, positive_pairs=total_positive),
        reference=dict(proposed_seed20_sampling=reference_coverage, stage_validation_passes=6,
            additional_final_association_passes=1, durable_checkpoint_writes=17,
            old013_known_training_calibration_stage_validation_hours=known['validation_scenarios']['20']['training_calibration_validation_hours'],
            old013_mean_single_validation_seconds=measured_validation,
            unknown_addends=['Full final evaluation (including its association pass)', 'Preparation, all checkpoint/report IO and process setup', 'Changed grouping/load and actual benefit from frozen-TFM caching'],
            cost_is_budget_bound=False),
        candidate_cost_probe=candidate_plan(probe['training']),
        candidate_development=candidate_plan(candidate['training']),
        candidate_cost=dict(measured_cuda_seconds=None, proposed_probe_outer_cap_seconds=7200,
            formula='P + U_teacher*t_teacher + U_utility*t_utility + U_adapt*sum(t_adapt_route) + V_identity + sum(V_physical_route) + IO',
            final_identity_scoring_passes=1, physical_evaluation_passes=4,
            warning='Each physical evaluation also rebuilds frozen native encodings. Step history excludes sampling/checkpoint IO; phase wall timings include them. Shared teacher/utility costs are counted once, not free.'),
        limitations=['Sampling-only deterministic simulation is not observed training coverage or convergence.',
            'Endpoint masks do not change pair sampler RNG; sampled exposure does not imply valid supervised queries.',
            'Four candidate arms repeat the same positive sequence; do not multiply independent coverage by four.',
            'Formal budgets intentionally remain null. Proposed two-hour candidate cap is not a runtime estimate or authorization.',
            'Cross-system training objectives, masking, optimizer, pair exposure and singleton policies differ.'])
    # Cast NumPy counts without allowing arbitrary objects or nonfinite values.
    def scalar(value):
        if isinstance(value, np.generic):
            return value.item()
        raise TypeError(f'unsupported analysis value: {type(value).__name__}')
    output = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False, default=scalar)
    (HERE/'analysis.json').write_text(output+'\n')
    print(json.dumps({k: result[k] for k in ('status', 'population', 'reference', 'candidate_cost')},
                     ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
