"""Read returned 013/014 evidence and derive TFM output bounds on CPU.

No data ZIP, checkpoint, training, optimizer update or model forward is used.
The fresh production-sized modules below only reproduce local initialization
order. They are not the server's saved selected weights.
"""
from collections import defaultdict
import gzip
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from association.models.fusion import FusionLoss, PairFusion, PairFusionConfig
from association.evaluation.physical import evaluate_aligned_fusion
from association.models.tmtaff import ModelConfig
from association.models.timed import TimedAssociationModel


def metric(prediction, target):
    p, t = np.array(prediction, dtype=float)[None, None], np.array(target, dtype=float)[None, None]
    valid = np.ones_like(p, dtype=bool)
    return evaluate_aligned_fusion(p, t, valid, prediction_valid=valid,
                                  region_mask=np.ones((1, 1), dtype=bool))['position']['mae']


def adam_coordinate_displacement(steps, lr, beta1=.85, beta2=.99):
    """Cauchy bound for fresh Adam, including bias corrections; eps only shrinks.

    Let actual finite (possibly clipped) gradients be g_i. Cauchy on the
    weighted sums m_t and v_t gives |mhat_t|/sqrt(vhat_t) <= K_t below.
    This is a real-arithmetic bound, not a bitwise FP32 error proof.
    """
    return sum(lr * (1-beta1)/(1-beta1**t) * math.sqrt(
        (1-beta2**t)/(1-beta2) * sum((beta1*beta1/beta2)**k for k in range(t)))
        for t in range(1, steps+1))


def main():
    with gzip.open(ROOT/'HandOver/results/TMTAFF-TIME-EVAL-20261018-014/files/summary.json.gz', 'rt') as stream:
        summary = json.load(stream)
    history = json.loads((ROOT/'HandOver/results/TMTAFF-TIME-COST-20261018-013/files/history.json').read_text())
    config = json.loads((ROOT/'configs/tmtaff-time-cost-probe.json').read_text())
    group_stats = {}
    for method in ('oracle', 'predicted'):
        buckets = defaultdict(list)
        for scene in summary['evaluation']['per_scene']:
            for group in scene['methods'][method]['all_predicted_groups']:
                row = group['metrics']['all']['position']
                if row['intersection_count']:
                    buckets[len(group['node_indices'])].append((row['intersection_count'], row['mae']))
        group_stats[method] = {str(size): dict(groups=len(rows), intersection_points=sum(n for n, _ in rows),
            point_weighted_mae_metres=sum(n*error for n, error in rows)/sum(n for n, _ in rows),
            min_group_mae_metres=min(error for _, error in rows),
            max_group_mae_metres=max(error for _, error in rows)) for size, rows in sorted(buckets.items())}

    # Match train_complete construction order, with CPU RNG only. No forwards.
    previous_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(config['training']['seed'])
            association = TimedAssociationModel(ModelConfig(**config['model']),
                config['training']['branch'], max_observations=config['data']['max_observations'])
            fusion = PairFusion(PairFusionConfig(**config['fusion'])).eval()
        norm = fusion.decoder.layers[-1].norm3
        assert not fusion.decoder.layers[-1].norm_first and fusion.decoder.norm is None
        assert not fusion.output_dropout.training
        d = fusion.config.d_model
        gamma, beta = norm.weight.detach().double(), norm.bias.detach().double()
        weights, bias = fusion.state_projection.weight.detach().double(), fusion.state_projection.bias.detach().double()
        assert torch.equal(gamma, torch.ones_like(gamma)) and not torch.count_nonzero(beta)
        intervals = {}
        for i, name in enumerate(('lat', 'lon', 'pre_softplus_speed', 'heading')):
            effective = weights[i]*gamma
            centre = float(weights[i]@beta + bias[i])
            radius = math.sqrt(d)*float(torch.linalg.vector_norm(effective-effective.mean()))
            intervals[name] = dict(centre=centre, radius=radius, lower=centre-radius, upper=centre+radius)
        a = math.sqrt(6/(d+4))
        assert float(weights.abs().max()) <= a
        bound_rows = {}
        for name, phases in (('fresh_TFM_2', [(2, .002)]),
                             ('hypothetical_fresh_TFM_100', [(100, .002)]),
                             ('hypothetical_TFM_100_then_fresh_JOINT_20', [(100, .002), (20, .0002)])):
            delta = sum(adam_coordinate_displacement(n, lr) for n, lr in phases)
            bound_rows[name] = dict(max_coordinate_displacement=delta,
                universal_raw_coordinate_abs_bound=delta+d*(a+delta)*(1+2*delta))
        # Evaluate only the loss on a constant synthetic physical target. No backward.
        reference_mean = summary['normalizer']['mean']
        synthetic_prediction = torch.tensor([[[0., 0., math.log(2), 0.]]])
        synthetic_target = torch.tensor([[reference_mean]], dtype=torch.float32)
        objective = FusionLoss(position_loss_mode='standard_smooth_l1')
        losses = objective(synthetic_prediction, synthetic_target, torch.ones_like(synthetic_target, dtype=torch.bool))
    finally:
        torch.set_num_threads(previous_threads)
    metric_checks = dict(coincident_metres=metric([20., 131., 1., 0.], [20., 131., 1., 0.]),
        equator_one_degree_metres=metric([0., 0., 1., 0.], [0., 1., 1., 0.]),
        origin_to_training_input_mean_metres=metric([0., 0., 1., 0.], [reference_mean[0], reference_mean[1], 1., 0.]))
    assert metric_checks['coincident_metres'] == 0
    assert math.isclose(metric_checks['equator_one_degree_metres'], math.pi*6371000/180, abs_tol=1e-8)
    result = dict(scope='CPU evidence reduction, fresh initialization and real-arithmetic bounds only; no selected weights/no forward/no training',
        torch_version=torch.__version__, selection={k:v for k,v in summary['selection'].items() if k != 'validation'},
        input_normalizer={k:summary['normalizer'][k] for k in ('mean', 'std', 'policy')},
        group_size_position_stats=group_stats,
        fusion_history=[{k:v for k,v in row.items() if k in ('stage','update','fusion_loss','fusion_component_losses',
            'gradient_norm','learning_rate','positive_pairs','fusion_valid_counts')} for row in history if row['stage'] in ('TFM_FROZEN','JOINT')],
        fresh_local_production_initialization=dict(seed=config['training']['seed'],
            construction_order='TimedAssociationModel then PairFusion; CPU only; not a server checkpoint reconstruction',
            parameter_count=sum(p.numel() for p in fusion.parameters()), eval_raw_coordinate_intervals=intervals,
            universal_initial_abs_bound=d*a),
        conditional_adam_bounds=bound_rows, metric_checks=metric_checks,
        synthetic_zero_coordinate_loss=dict(total=float(losses['total'].detach()),
            components={k:float(v.detach()) for k,v in losses['losses'].items()}))
    path = Path(__file__).with_name('model_geometry_diagnostic.json')
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
    print(json.dumps({k:result[k] for k in ('selection','fresh_local_production_initialization','conditional_adam_bounds','metric_checks')}, indent=2))
    print(f'Wrote {path}')


if __name__ == '__main__':
    main()
