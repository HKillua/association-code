# Identity/utility candidate: local implementation contract

2026-10-18. User requested review followed by local CPU/GPU implementation. This authorizes the candidate implementation and bounded synthetic verification, not a new native-data/server training budget. Existing TMTAFF execution code/configuration and active 013 remain unchanged. Research rationale and sources: `/Users/a1/Projects/biye-ronghe/docs/plans/2026-10-18_complete_method_design/README.md`.

## Scope and behavior

- Independent `association/cooperative*.py` modules and an experiment CLI; reuse native-time packing, fixed splits, sampling, grouping and physical metrics. No test split, new dataset or general execution framework.
- `CooperativeConfig`: length=64, max_observations=128, d_model=128, nhead=8, num_layers=6, dim_feedforward=256, dropout=.1, head_hidden=64. Small dimensions permitted only for explicitly labelled synthetic checks, never as a formal replacement.
- Packed input `[N,M+Q,5]`: encoder reads only first M native four-state values and times under native mask. Last Q contains query times only; its state values are never read. Query times are physical tau supplied by existing 20-second adapter. Masked/padded content is ignored, even if NaN. Every fragment retains >=2 strictly increasing visible observations. No reference/identity is a forward argument.
- Shared native encoder and physical-time query decoder produce four physical states for one or multiple fragments. Output rescales by training normalizer; speed uses softplus; latitude is constrained to [-90,90], longitude and course wrap to their physical ranges. These explicit engineering choices are common to all candidate controls, not exact TMTAFF reproduction.
- Model API: `encode(values, observed)` returns a dictionary of `memory`, `observed`, `times`, `queries`, `pooled`; `summarize(encoded)` returns `[N,Q,D]`; `utility(encoded,pairs,summaries=None)` returns `[E,2,Q,4]`, direction 0 is left<-right; `score_pairs(encoded,pairs,use_utility=False)` returns symmetric identity logits; `decode_group(encoded,members,routing='plain',identity_logits=None,utility=None)` returns `[Q,4]`. Pair indices are explicit, no true identity input.
- Routes: plain (unit weights), identity, utility, support (visible time support), learned (ordinary trainable gate). First member is anchor with weight 1. Helper weights are `f+(1-f)*sigmoid(s0)` for identity, `f+(1-f)*sigmoid(s0)*clip(mean4(u),0,1)` for utility, with explicit numerical floor f=.001. External helper tensors are strictly `[K-1]` logits and `[K-1,Q,4]` utility. Per-token logit bias `log(w)-log(n_visible)` for every member, including anchor; padding excluded. Utility route fixed four-attribute mean, never reference mask. Negative predicted utility is retained in supervision and clipped only for routing. All queries get outputs, including singletons and both-missing regions. Pair marginal utility is not additive group utility.
- Public trainable parts: `input_projection`, `encoder`, `decoder`, `state_projection`, `role_embedding`, `query_pool`, `utility_head`, `identity_base`, `identity_with_utility`, `ordinary_gate`. Both identity heads have matching capacity; base substitutes zeros for utility features. Final identity utility features use detached tanh(u), averaged over directions/queries; identity gradients cannot redefine utility. Query pooling/utility head are independent of base inference.

## Training and comparison

1. Teacher trains shared encoder/decoder with single-track physical reconstruction, positive-pair physical reconstruction and base identity BCE. Input mask views are none/prefix/suffix with equal probability, remove floor(n/4) when possible and retain >=2 points. No cached interpolated/query state enters model. Original unmasked observation/support population is retained for evaluation.
2. Freeze the same teacher snapshot in eval mode. Compute self/cross errors against the anchor reference at identical queries; labels are per-attribute reference-valid only. Negative helper uses anchor's target, never a fake shared target. Fixed training-side scales are explicit config. Artificial observed-point recovery is not mixed into this utility target.
3. Train only query-pooling/utility head using Huber(`sg(e0-e1)-sg(max(e0,epsilon))*u`). Source: Gekko error comparison, adapted to visible-only physical trajectory input. No identity gradient updates the frozen utility module.
4. First comparison fixes base identity scores/groups. From identical teacher decoder states, adapt plain/utility/support/ordinary-gate decoders with identical sampled positive pairs and update counts. Encoder/base identity/utility freeze. Ordinary gate uses the same frozen utility-trained query-pooling features as the utility arm but a separate trainable gate without predicted gain inputs, a strong matched-feature control; shared feature training cost is disclosed. Save predetermined final update, not a nonexistent identity-F1 improvement. Optional final identity head training is separate, after utility; keep it out of first fixed-identity comparison.

All losses expose valid counts; reference NaNs masked before arithmetic. No silent resampling if supervision is unavailable. Stages, sampler seeds, optimizer state, normalizer, scale and route stored in checkpoint. Checkpoint reload is inference-only; optimizer resume is explicitly unsupported by this minimal first implementation. The fixed utility error scale is the four training-native normalizer standard deviations (same declared constant-feature floor), with shortest longitude/course residuals; it is not fitted on V. First local executable supports explicit counts/config and seeded synthetic check; no default native training schedule or READY dispatch. Formal config/epochs/cost remains separate.

## Verification

| Acceptance | Evidence |
|---|---|
| No query/ref/masked-content leakage; two-point fragment preserved | Counterfactual input tests, native-format synthetic integration |
| Directional utility, symmetric identity | Pair swap test; loss/gradient isolation |
| Real physical outputs for singleton/group/missing queries | Shapes, finite gradients, full-target evaluator and coverage |
| Length-normalized gate and disabled branch invariance | Different visible counts; identical output with unused u altered |
| Teacher deterministic and frozen | Dropout-on training config, eval repeated labels, no teacher gradients |
| Fixed identity comparison and final decoder selection | Exact common records/groups, matched adaptation sequence, checkpoint final state |
| CPU/MPS/CUDA device behavior | Real CPU and available MPS checks; CUDA check conditional, explicit unavailable failure, no silent CPU fallback |
| Existing path unaffected | Full repository CPU tests; no baseline execution edits |

CPU or synthetic GPU checks establish implementation behavior, not MTAD effectiveness, calibration, novelty or formal runtime. MPS results do not establish CUDA parity. No new GPU experiment allocation is inferred from this document.

Independent review added an identity-only gate control: utility uses both s0 and u, so plain versus utility alone cannot attribute an improvement to u. The built-in check exercises plain/identity/utility/learned. A custom reduced route list records missing attribution controls and cannot support a utility-specific scientific claim.
