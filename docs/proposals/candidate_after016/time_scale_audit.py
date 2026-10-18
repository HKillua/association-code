"""CPU proposal evidence: saved training metadata and synthetic head inputs only."""
from pathlib import Path
import gzip
import json
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from association.cooperative import CooperativeConfig, CooperativeModel


def main():
    source = ROOT / 'HandOver/results/COOPERATIVE-COST-20261018-015/files/run.json.gz'
    saved = json.loads(gzip.decompress(source.read_bytes()))
    scenes = saved['scene_contract'][0]
    assert len(scenes) == 512 and all(s['split'] == 'T_core' for s in scenes)
    extrema, spans, counts = [], [], []
    for scene in scenes:
        assert scene['input_adapter'] == 'native_time_v1'
        assert scene['time_unit_seconds'] == 20.
        endpoints = np.asarray(scene['intervals'], dtype=np.float64)
        tau = ((endpoints - scene['time_origin_seconds']) / scene['time_unit_seconds']).astype(np.float32)
        query_tau = ((np.asarray(scene['grid']) - scene['time_origin_seconds']) / scene['time_unit_seconds']).astype(np.float32)
        extrema.append(float(max(np.abs(tau).max(), np.abs(query_tau).max())))
        spans.extend((tau[:, 1] - tau[:, 0]).tolist())
        counts.extend(scene['native_observed_counts'])
    scale = max(1., max(extrema))
    # Monotone native times and retained first/last points make endpoints exact
    # for this maximum. A future fit must derive it from its actual packed
    # training inputs, never copy this value as a population-independent rule.
    torch.set_num_threads(1)
    torch.manual_seed(20)
    model = CooperativeModel(CooperativeConfig(), dict(mean=[0.]*4, std=[1.]*4)).eval()
    cases = []
    with torch.inference_mode():
        for span in (1., 10., 100., 350., scale, 2. * scale):
            values = {}
            for mode, divisor in [('legacy', 1.), ('proposed', scale)]:
                # Only calls score_pairs/utility/ordinary_gate with supplied
                # zero semantic features. No encoder/decoder or real inference.
                encoded = dict(memory=torch.zeros(2, 2, 128),
                    observed=torch.ones(2, 2, dtype=torch.bool),
                    times=torch.tensor([[0., span], [.05*span, .85*span]]) / divisor,
                    pooled=torch.zeros(2, 128),
                    queries=torch.linspace(0, span, 64).expand(2, -1) / divisor)
                pairs = torch.tensor([[0, 1]])
                summaries = torch.zeros(2, 64, 128)
                score = model.score_pairs(encoded, pairs)[0]
                utility = model.utility(encoded, pairs, summaries=summaries)
                ordinary = model.ordinary_gate(model._directional_features(encoded, pairs, summaries))
                values[mode] = dict(identity_logit=float(score),
                    identity_sigmoid=float(score.sigmoid()),
                    utility_min=float(utility.min()), utility_max=float(utility.max()),
                    ordinary_logit_min=float(ordinary.min()), ordinary_logit_max=float(ordinary.max()))
            cases.append(dict(span_tau=float(span), **values))
    result = dict(status='PROPOSAL_CPU_EVIDENCE_ONLY', training_scene_count=len(scenes),
        training_track_count=len(counts), scale_tau=scale,
        scale_rule='max(1, absolute maximum of valid native/query FP32 tau on T_core)',
        endpoint_maximum_equivalence='native times monotone; cap retains first/last; queries included explicitly',
        time_unit_seconds=20., count_quantiles=np.quantile(counts, [0, .5, 1]).tolist(),
        native_span_tau_quantiles=np.quantile(spans, [0, .5, .9, 1]).tolist(),
        synthetic_initial_head_comparison=cases,
        unchanged='log1p(count); real temporal encoding; physical timestamps; speed output; losses; sampling',
        no_gpu_training_or_actual_model_inference=True,
        validation_or_test_used_for_scale=False,
        limitations=['Not fitted weights or real encoder features.',
                    'Reduced raw-time magnitude does not establish better accuracy or convergence.',
                    'Production model/config/CLI are not changed or authorized by this proposal.'])
    Path(__file__).with_suffix('.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(dict(scale_tau=scale, training_track_count=len(counts),
                         synthetic_span350=cases[3]), ensure_ascii=False))


if __name__ == '__main__':
    main()
