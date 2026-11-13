"""087 local implementation: explicit helper correspondence before set pooling.

The temporal branch is offline. Its20 channels are phi16 + interval-support1 + observable query
elapsed/spacing/span log-times3. Unsupported phi values are
zero, but time/support remain visible to the GRU across gaps. Only currently
supported rows enter the final mean. No identities or references are inputs.
"""
from __future__ import annotations
import numpy as np
import torch
from torch import nn
from association.models.helper_sequence import HelperSequencePosition, helper_sequence_inputs
from association.models.sequence_position import tangent_position

SCOPE = 'helper-temporal-position-v1'
ROUTING_SEMANTICS = 'warm095-allquery-helper-bidirectional-gru-correspondence-v1'
MODES = ('shuffled', 'preserved')


def permutation_plan(helpers, queries, rng):
    if type(helpers) is not int or helpers < 0 or type(queries) is not int or queries < 1 or not isinstance(rng, np.random.Generator):
        raise ValueError('nonnegative helper count, positive query count and independent Generator required')
    return np.asarray([rng.permutation(helpers) for _ in range(queries)], dtype=np.int64).reshape(queries, helpers)


def validate_plan(plan, helpers, queries):
    plan = np.asarray(plan)
    if plan.shape != (queries, helpers) or plan.dtype.kind not in 'iu' or not np.array_equal(np.sort(plan, axis=1), np.broadcast_to(np.arange(helpers), (queries, helpers))):
        raise ValueError('each query route must be a complete helper permutation')
    return plan


class HelperTemporalPosition(HelperSequencePosition):
    def __init__(self, *, correspondence='preserved', hidden_size=64, num_layers=2, initial_gate=.95):
        if correspondence not in MODES:
            raise ValueError('declared helper correspondence required')
        super().__init__(aggregation='pre_pool', hidden_size=hidden_size, num_layers=num_layers, initial_gate=initial_gate)
        self.correspondence = correspondence
        self.aggregation = 'temporal_'+correspondence
        self.routing_semantics = ROUTING_SEMANTICS
        self.temporal = nn.GRU(20, 8, num_layers=1, bidirectional=True, batch_first=True)
        self.route_rng = None
        self.route_log = []
        self._initialize_temporal()

    def _initialize_temporal(self):
        for name, parameter in self.temporal.named_parameters():
            if 'weight' in name: nn.init.xavier_uniform_(parameter)
            else: nn.init.zeros_(parameter)

    def reset_route(self, seed):
        if type(seed) is not int or seed < 0: raise ValueError('route seed required')
        self.route_rng = np.random.default_rng(seed)
        self.route_log = []

    def temporal_representation(self, features, support, time_features, plan):
        h, q = features.shape[:2]
        if (features.shape != (h,q,27) or features.dtype != torch.float32 or support.shape != (h,q)
                or support.dtype != torch.bool or support.device != features.device
                or time_features.shape != (q,3) or not torch.isfinite(time_features).all()
                or not torch.isfinite(features[support]).all()):
            raise ValueError('finite supported27-feature rows, masks and observable query times required')
        plan = validate_plan(plan,h,q)
        used = plan if self.correspondence == 'shuffled' else np.broadcast_to(np.arange(h),(q,h))
        index = torch.as_tensor(np.array(used,copy=True),dtype=torch.long,device=features.device).T
        qi = torch.arange(q,device=features.device)[None,:].expand(h,q)
        safe = torch.where(support[...,None],features,torch.zeros_like(features))
        routed_support = support[index,qi]
        routed = safe[index,qi]
        latent = torch.where(routed_support[...,None],self.phi(routed),torch.zeros((h,q,16),dtype=features.dtype,device=features.device))
        inputs = torch.cat((latent,routed_support[...,None].to(features.dtype),time_features[None].expand(h,-1,-1)),-1)
        counts=support.sum(0)
        if h:
            encoded,_=self.temporal(inputs)
            embedding=(encoded.to(torch.float64)*routed_support[...,None]).sum(0)/counts.clamp_min(1)[:,None]
            embedding=embedding.to(features.dtype)
        else:
            embedding=torch.zeros((q,16),dtype=features.dtype,device=features.device)
        # Zero connections preserve complete optimizer state even for empty groups.
        zero=sum((p.sum()*0. for p in (*self.phi.parameters(),*self.temporal.parameters())),safe.sum()*0.)
        embedding=embedding+zero
        weights=support.to(torch.float64)/counts.clamp_min(1)[None,:]
        if not torch.isfinite(embedding).all(): raise FloatingPointError('nonfinite temporal helper representation')
        return embedding,weights,counts,np.array(used,dtype=np.int64,copy=True)

    def forward(self, base_states, features, mean_states, helper_features, support, *, plan=None):
        if (features.shape != (len(base_states),76) or features.dtype != torch.float32 or not torch.isfinite(features).all()
                or mean_states.shape != base_states.shape or mean_states.dtype != torch.float64
                or not torch.isfinite(mean_states).all() or not torch.equal(base_states[:,2:],mean_states[:,2:])):
            raise ValueError('same frozen physical base, Mean and76 visible features required')
        h,q=helper_features.shape[:2]
        record_route=plan is None
        if plan is None:
            if self.route_rng is None: raise ValueError('explicit route plan or reset evaluation route required')
            plan=permutation_plan(h,q,self.route_rng)
        branch,weights,counts,applied=self.temporal_representation(helper_features,support,features[:,71:74],plan)
        if record_route: self.route_log.append(dict(planned=np.asarray(plan).copy(),applied=applied.copy()))
        embedded=self.input_projection(torch.cat((features,branch),-1)).unsqueeze(0)
        residual=self.output(self.encoder(embedded).squeeze(0)).to(torch.float64)*1000.
        states=tangent_position(base_states,residual)
        if not torch.isfinite(states).all(): raise FloatingPointError('nonfinite temporal position states')
        return dict(states=states,base_states=base_states,mean_states=mean_states,residual_m=residual,
            helper_embedding=branch,helper_weights=weights,helper_counts=counts,helper_support_count=counts,
            aggregation=self.aggregation,route_plan=np.asarray(plan,dtype=np.int64).copy(),effective_route=applied)


def initialize_from_reference(model, reference_state, branch_seed):
    """Inherit common step0; seed phi/GRU independently, without RNG pollution."""
    if (not isinstance(model,HelperTemporalPosition) or type(branch_seed) is not int or branch_seed<0
            or any(p.device.type!='cpu' for p in model.parameters())):
        raise ValueError('CPU temporal model required for initialization')
    state=model.state_dict()
    common={k for k in state if not k.startswith(('phi.','temporal.'))}
    if (set(reference_state)!=common or any(not isinstance(v,torch.Tensor) or v.dtype!=state[k].dtype
            or not torch.isfinite(v).all() for k,v in reference_state.items())):
        raise ValueError('exact finite original reference state required')
    if torch.count_nonzero(reference_state['output.weight']) or torch.count_nonzero(reference_state['output.bias']):
        raise ValueError('only original step0 may initialize temporal control')
    with torch.no_grad():
        for key in common:
            if key=='input_projection.0.weight':
                if reference_state[key].shape!=(64,76): raise ValueError('reference projection must have76 features')
                state[key][:,:76].copy_(reference_state[key]);state[key][:,76:].zero_()
            else:
                if state[key].shape!=reference_state[key].shape: raise ValueError('reference shape mismatch/'+key)
                state[key].copy_(reference_state[key])
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(branch_seed); model._initialize_branch(); model._initialize_temporal()
    return dict(common_parameters_exact=True,extra_projection_zero=True,branch_seed=branch_seed,
        parameter_count=sum(p.numel() for p in model.parameters()))
