"""Lightweight setup mixins shared by numeric and integration tests."""
import torch
from association.models.tmtaff import AssociationModel, ModelConfig
from association.models.fusion import FusionLoss, PairFusion, PairFusionConfig
from association.training.fusion import fusion_training_parameters
from association.evaluation.tmtaff import evaluate_complete
from association.training.common import fit_normalizer
from tests.fixtures.scenes import scene

class CompleteEvaluationFixture:
    def setUp(self):
        torch.manual_seed(42)
        self.model = AssociationModel(ModelConfig(length=4, d_model=8, nhead=2,
                                     num_layers=2, dim_feedforward=16, dropout=.1), 'ce')
        self.fusion = PairFusion(PairFusionConfig(fc_hidden=8, length=4, d_model=8,
                                 nhead=2, num_layers=2, dim_feedforward=16, dropout=.1))
        self.normalizer = fit_normalizer([scene()])

    def evaluate(self, scenes, threshold=-1e6, singleton='missing', select=False, **kwargs):
        return evaluate_complete(self.model, self.fusion, self.normalizer, scenes, device='cpu',
                                 batch_tracks=2, pair_batch_size=2, select_on_validation=select,
                                 threshold=threshold, singleton_policy=singleton, **kwargs)


class FusionTrainingFixture:
    def models(self, branch="contrastive", dropout=0):
        torch.manual_seed(42)
        model = AssociationModel(ModelConfig(length=4, d_model=8, nhead=2,
                                 num_layers=1, dim_feedforward=16, dropout=dropout), branch).double()
        fusion = PairFusion(PairFusionConfig(fc_hidden=8, length=4, d_model=8, nhead=2,
                            num_layers=1, dim_feedforward=16, dropout=dropout)).double()
        objective = FusionLoss(position_loss_mode="paper_printed").double()
        return model, fusion, objective

    def batch(self):
        values = torch.randn(5, 4, 4, dtype=torch.float64)
        observed = torch.tensor([[False, True, True, False], [True, True, True, False],
                                 [False, True, True, True], [True, True, False, False],
                                 [True, True, True, True]])
        values[~observed] = float("nan")
        # An unused fifth endpoint must not contribute auxiliary TCM targets.
        pairs = torch.tensor([[0, 2], [1, 3], [0, 1], [0, 1], [2, 3]])
        labels = torch.tensor([0., 0., 1., 1., 1.], dtype=torch.float64)
        target = torch.tensor([30., 120., 5., 359.], dtype=torch.float64).expand(3, 4, 4).clone()
        valid = torch.ones_like(target, dtype=torch.bool)
        valid[:2, :, 0] = False  # First positive micro has no latitude target.
        valid[:2, :2, 1] = False
        valid[2, :, 2] = False
        valid[2, 1:, 3] = False
        target[~valid] = float("nan")
        return dict(values=values, observed=observed, pairs=pairs, labels=labels, target=target, valid=valid)

    def optimizer(self, modules, stage):
        parameters = fusion_training_parameters(*modules, stage=stage)
        return torch.optim.SGD(parameters, lr=0.001)
