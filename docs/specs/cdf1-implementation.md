# CDF1 implementation contract

> 2026-10-18已退役，仅保留历史规格。当前判断与路线见[CDF1退役决定](../decisions/retire_cdf1_and_focus.md)；旧代码及启动脚本可从[清理前版本](https://github.com/HKillua/association-code/tree/7766b86)恢复。本文不授权执行。

Scope: executable MTAD ZIP ingestion, identity-isolated split, TMTAFF-style TCM+TAM association reference (CE and contrastive branches), frozen-score contextual correction and required controls, validation-only selection and held-out evaluation. This is an association pilot, not the full TMTAFF/TFM system or official MTAD V4 scorer. Source design: IDEA-CDF1-20261016-v1.1, TMTAFF DOI 10.3390/app16147251; author implementation is unavailable. Implementation choices and differences must be recorded in the run configuration.

## Shared module interfaces

- `association.data`: `Track(node_id, source, batch, identity_id, ancestor_id, times, values)` with NumPy arrays times[L], values[L,4]; `Scene(scene_id, tracks, audit)`; `MTADArchive(path)` context manager with `scene_ids()`, `read_labels(scene_id)`, `load_scene(scene_id)`; `build_manifest(archive, seed, caps, development_scenes)` returns JSON-compatible metadata with `splits` mapping T_core/T_head/V_select/T_test to scene ID strings and `components` mapping scene ID to component ID. No reference files are read. `resample_scene(scene, length=64)` returns dict `values[N,L,4]`, `observed[N,L]`, `grid[L]`, `node_ids`, `identities`, `sources`, `audit`. Keep all valid nodes; short temporal support cannot silently remove nodes. Agent must document temporal registration policy.
- `association.models`: `ModelConfig` dataclass (`input_dim=4,d_model=128,nhead=8,num_layers=6,dim_feedforward=256,dropout=.1,length=64,margin=1.0`); `AssociationModel(config, branch='ce')` supports `completion_loss(values, observed)`, `encode_tracks(values, observed)` -> [N,L*D], `score_pairs(embeddings, pairs)` -> [E] (CE logit or negative Euclidean distance), `association_loss(scores, labels)` -> scalar. `complete(values, observed)` returns [N,L,D] preserving observed input embeddings and using learned bidirectional endpoint completion only. No identity/source/reference input. Configurable tiny CPU model for tests, production defaults from paper where known. Model agent documents deviations and exact TCM semantics.
- `association.context`: `context_features(z, mask, shuffle=False, seed=0)` returns dict containing `features[E,8]`, `pairs[E,2]`, `counts[E]`, `support[E]`, `contradiction[E]`, `shuffleable[E]`; matrix z[N,N] symmetric, mask excludes diagonal and missing edges. One pass over original scores. `MonotoneCalibrator` torch module forward(raw)->z; `FeatureScaler` fit only T_head; `ResidualHead(kind, delta)` with `forward(features, counts)` -> bounded residual (not corrected logits); kind B/PairMLP/EndpointMarginal/ScenePool/B_shuffle, zero output init. ScenePool needs separate feature helper with own score+scene mean/variance/10-bin histogram. Exact interface details to root before integration.
- `association.metrics`: pure NumPy evaluation, threshold selection and component bootstrap; module agent reports actual function interfaces. Fixed macro population (>=2 valid nodes), empty truth+prediction F1=1; no prediction precision undefined; same-source/cross-source reporting; common connected-component grouping and wrong-right deltas. Threshold sweep unique validation scores + all reject, ties higher threshold.
- Root owns CLI, trainer/configs/checkpoint/cache orchestration, run identity/provenance and documentation. Other agents own only assigned modules/tests. No agent commits/pushes or changes HandOver.

## Acceptance criteria

| ID | Required observable behavior | Verification |
|---|---|---|
| AC1 | Reads native UTF-8/GBK ZIP member names and CSVs; never uses reference state | Synthetic archive + existing dev scene read |
| AC2 | Shared A identities and previously seen development components never cross splits | Connected-component fixture and manifest audit |
| AC3 | Same-source multiple positive identity edges retained; anomalies quarantined explicitly | Dataset edge/coverage tests |
| AC4 | Actual differentiable TCM and shared TAM, both losses, checkpoint roundtrip | Gradient/identity/symmetry/roundtrip tests |
| AC5 | Third-node stats symmetric, absent edges excluded, K=0 identity, one-side shuffle preserves marginals | Analytic fixtures, node permutation and masked-edge tests |
| AC6 | Required controls and strong A_full present; no test selection or reference-derived features | End-to-end synthetic CLI smoke + independent review |
| AC7 | Frozen split/model/calibration/thresholds, file hashes, unique output, finite checks and time caps | CLI integration and corrupt/stale input tests |
| AC8 | Primary and mechanism comparisons, coverage, bootstrap and bridge risk are distinguishable from proof | Evaluation tests and generated report |

All new runnable code lives in this repository. Existing ENV-001 remains the live server task until its receipt; implementation and local CPU verification do not start server training. Do not overwrite an active handover or infer GPU allocation from a snapshot.

## Delivered entry points and validation

`association.cli` now exposes `prepare`, `profile`, `train-reference`, `run-pilot`, and `smoke`. `profile` executes at most200 real training steps without creating an eligible reference. `run-pilot` completes every requested seed's validation choices before a single held-out measurement-loading phase, freezes complete artifact hashes, and verifies them again before scoring. CUDA must be explicit and cannot fall back to CPU. Training and preparation interruptions leave terminal failed evidence.

Final evidence is recorded in `docs/verification-2026-10-16.md`:93 new-module tests plus15 existing audit tests, an actual synthetic end-to-end run, reference reuse, CPU profile, expected CUDA-unavailable rejection, and a native development-scene gradient check. CUDA and production scientific performance remain unverified. `docs/pilot_execution.md` specifies the actual stage budgets and explicitly documented model choices.
