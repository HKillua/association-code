# CDF1 pilot execution protocol

These interfaces are implemented. They are **not** the current server task: read `HandOver/Handover.md` before executing anything. The current handover names the authorized environment/preflight/experiment stage. The local AI makes research and implementation decisions; the server AI owns execution and monitoring.

## Reproducible inputs

Run from the repository root, with a Python environment containing the CUDA-enabled PyTorch build selected for that server. `pyproject.toml` declares Python >=3.10, NumPy >=1.26,<3 and PyTorch >=2.5,<2.11; this is a compatibility range, not a tested version matrix. The local functional check used Python 3.12.14, NumPy 2.5.4 and PyTorch 2.10.0 on CPU. Every run records actual versions, device, Git commit/dirty state, Python source hashes, config and manifest hashes. A production handover must name an immutable code commit and approved environment.

`configs/pilot.json` declares width128, 8 heads, 6 layers in each TCM encoder/decoder and TAM encoder, FF256, length64. It uses all eligible unordered pairs, including same-source positives; there is no GT-informed candidate gating or negative sampling. Quarantine, circular course interpolation, native velocity units and short-support registration are explicit in `data_contract.md`. Selected scene caps are 512/128/64/128 for T_core/T_head/V_select/T_test; all labels establish shared-A connected components before selection. Previously examined development components are forced to training. Scene caps do not move unused component members into another split.

```bash
python -m association.cli prepare \
  --archive /data/mtad/level2.zip --config configs/pilot.json \
  --output runs/PREP-001
```

Preparation writes immutable `manifest.json` and `audit.json`, checking training/validation measurements only. Test measurements are deferred. It does not read reference trajectories. Downstream stages reject a different archive SHA, mismatched split seed/caps, or incompatible preprocessing. `--prepared` also checks completed audit and registered length; `--manifest` accepts the frozen split directly and records current model length in the run config.

## GPU cost gate before full training

After environment and device allocation are confirmed, a bounded next handover should profile the production model first. Example below assumes physical GPU 2 has been allocated; within that process it is `cuda:0`.

```bash
CUDA_VISIBLE_DEVICES=2 python -m association.cli profile \
  --archive /data/mtad/level2.zip --prepared runs/PREP-001 \
  --config configs/pilot.json --role A0 --branch ce --seed 20 \
  --steps 200 --max-minutes 30 --device cuda:0 \
  --output runs/PROFILE-A0-ce-seed20
```

This performs real Adam steps on the training scene with the most valid nodes. It records synchronized GPU times, allocated/reserved peak memory, loss, finite gradients, changed parameters and planned stage step counts. The first 5 steps are warmup; fewer than 6 steps cannot produce a steady-state extrapolation. Fresh model initialization is used, not a trained TCM checkpoint. It never reads validation/test and never creates an eligible reference checkpoint. The TAM-only cost extrapolation excludes TCM pretraining, validation, caching and preparation; it is not a complete runtime guarantee. The local AI must assess this report before issuing a full-fit task; no automatic smaller model, reduced scene set or increased budget is allowed.

## Training and validation

```bash
CUDA_VISIBLE_DEVICES=2 python -m association.cli train-reference \
  --archive /data/mtad/level2.zip --prepared runs/PREP-001 \
  --config configs/pilot.json --role A0 --branch ce --seed 20 \
  --device cuda:0 --output runs/REFERENCES/seed20/A0/ce
```

A0 learns only from T_core. A_full learns from T_core+T_head and is the primary stronger comparison. Both fit their normalizers only to their own observed training points. TCM pretraining freezes the input embedding to keep its reconstruction target fixed; TAM optimization normally updates embedding, TCM and TAM with association loss plus 0.1 completion loss. This is a documented independent two-module objective, not the authors' joint TCM/TAM/TFM objective. The explicitly configured frozen-TCM option changes the experiment and requires a new decision.

There are 80 maximum TCM epochs and 80 maximum TAM epochs **per fit**, with a combined 12-hour cap including input preparation, validation and cache writes. This clarifies a previously ambiguous 80-epoch allocation; it does not claim the author's schedule. Adam uses betas(.85,.99), epsilon1e-6, LR .002. `warmup_epochs=20` means a constant-LR first20 epochs, followed by cosine decay, separately in each stage; it is not linear warmup. Pair minibatches are64, track chunks64, gradients clipped to norm1. All pair minibatches update in each scene, so larger scenes contribute more reference steps. Context heads instead use a mean edge loss per scene with one step per scene.

Validation macro F1 chooses the TAM checkpoint; exact ties retain the earliest epoch. Threshold selection considers every distinct validation score plus all-reject (`null`), choosing the higher threshold on ties. Contrastive scores are negative distances; CE scores are logits. Neither raw negative distances nor raw CE outputs are labeled calibrated probabilities.

Each complete reference writes `checkpoint.pt`, `history.json`, `summary.json`, `run.json`, and hash-bound T_head/V_select score caches. A failed or time-limited fit is **not eligible**. Model, normalizer, split, role, seed, code and config must all match for reuse. Budget checks are cooperative at batch/stage boundaries; the server AI remains responsible for hard process/driver stalls and runtime supervision.

## Complete pilot

```bash
CUDA_VISIBLE_DEVICES=2 python -m association.cli run-pilot \
  --archive /data/mtad/level2.zip --prepared runs/PREP-001 \
  --config configs/pilot.json --device cuda:0 \
  --output runs/PILOT-001
```

The default is serial on one assigned GPU. Seed20 evaluates CE and contrastive for A0 and A_full; the better validation branch for each role is frozen for seeds21/22. Exact ties prefer CE. This is eight reference fits at most, followed by one complete head sweep per seed. No seed or branch is selected using test results. A deliberately smaller first pilot can pass `--seeds 20`; requested seeds must be a prefix of the fixed config list and must be declared in the handover. Subsequent experiments on the same revealed test set must disclose that history; starting a new output directory does not create a fresh hold-out.

For separately dispatched references on several allocated GPUs, `--reference-root runs/REFERENCES` reuses a tree laid out as `seed20/A0/ce`, `seed20/A0/contrastive`, `seed20/A_full/ce`, `seed20/A_full/contrastive`, then selected branches for later seeds. All required fits must exist, be complete and match the exact config/manifest/code; missing fits fail. The serial mode needs no external reference tree. Four-GPU dispatch and branch-to-device mapping belong in a concrete future handover, not a background scheduler.

One shared positive-slope calibrator fits T_head only. Each residual arm receives its own frozen T_head feature scaler, the same three delta candidates [.5,1,2], and40 epochs; output-layer zero initialization and bounded `delta*tanh` corrections keep the comparison interpretable. B, B_shuffle, EndpointMarginal, PairMLP and ScenePool have825–843 active parameters. All heads preserve the calibrated score at K=0. FixedRule has no fitted head but searches the same delta list on validation. B_shuffle randomizes one endpoint's neighbor correspondence each training epoch while preserving endpoint marginals; validation/test interventions use fixed deterministic seeds. B_inference_shuffle is a diagnostic on the trained B at B's original threshold, without additional fitting.

All seeds finish reference and head selection **before** test measurements are opened. `selection_frozen.json` binds chosen branches, thresholds, head settings and hashes of complete model artifacts. Those files are checked before opening test and again before use. Test scoring has no fitting or threshold search. No supported resume mode overwrites a partial run; retain evidence and let the local AI issue a justified retry.

## Outputs and interpretation

- `report.json` / `report.md`: held-out metrics and primary B−A_full, mechanism B−EndpointMarginal, additional controls, per-seed component bootstrap, wrong↔right changes, Brier and common-threshold sensitivity.
- `seed*/evaluation.json`: full per-seed diagnostics; connected-component closure uses the same rule for all arms and reports mixed/split identities and false bridges.
- `seed*/test_scores/`: immutable per-scene numeric caches with checksums, stored on the server.
- `test_audit.json`: held-out quarantine/registration coverage, first generated after selection freezes.
- `third_node_original_scores`: highest-support third neighbor chosen using original calibrated scores only; identity labels are used afterward to count joint errors. Time coverage means support-interval union, not independently observed information.

Macro F1 uses the same eligible scene population for all arms. Truth+prediction empty F1 is1; no-pair scenes are coverage-only. Positive-only and all-negative scene behavior is separately reported. ITSA/MSTSA partition same-/cross-source edges. Bootstrap resamples shared-A components, not edges; fewer than10 components gives descriptive results without a CI. Across-seed mean/range is descriptive on the same test data, not a second independent confidence interval.

A software completion status does not establish reference convergence, scientific superiority, online causality or novelty. Scene-level context uses all supplied fragments in that scene; this implementation makes no online claim. Do not directly rank its values against paper tables or official MTAD V4 scores. Review learning curves, coverage, cost and conformance first.
