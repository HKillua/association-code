# Implementation verification — 2026-10-16

Scope: local software verification of the first executable CDF1 association pilot. No server execution, GPU performance measurement, formal MTAD training result, author-code reproduction, or superiority claim is reported here.

## Environment and source binding

Local macOS, Python3.12.14, NumPy2.5.4, PyTorch2.10.0, CUDA unavailable. Editable package installation with existing dependencies succeeded; both `python -m association.cli --help` and the installed `association-pilot --help` entry point worked.

Final functional smoke recorded association-source SHA256 `37c400dd5d4358f53a24f3e28fa77771024281677678e6bab208b181634e9517`; it was checked against the current source after all code fixes. This source digest is independent of the later Git commit containing this report. The run correctly recorded the precommit worktree as dirty. Model outputs and fixture archives remain in ignored local `runs/`, not in Git.

## Checks actually executed

| Check | Outcome |
| --- | --- |
| `python -m unittest discover -s tests` | 93 passed |
| `python -m unittest discover -s tools/pilot -p 'test_audit_mtad_level2.py'` | 15 passed |
| `python -m compileall -q association tests` | Completed |
| `python -m association.cli smoke --output runs/local-smoke-20261016-final --device cpu` | Completed:4 tiny reference fits, all head sweeps, checkpoint reload, frozen selection, held-out scoring/report |
| `run-pilot` using that synthetic archive/manifest and `--reference-root runs/local-smoke-20261016-final/pilot/references --seeds 20` | Completed; verified existing references reused, separate new output directory |
| `profile` with synthetic prepared input, smoke config, CE/A0/seed20, `--steps 2 --device cpu` | Completed real optimizer steps; no eligible checkpoint; no GPU speed projection |
| Same CLI requesting `--device cuda:0` on this host | Expected exit1: `CUDA requested but unavailable; CPU fallback is forbidden`; no output directory created |
| Native Level2 previously inspected development scene0 → tiny CE model → association+completion backward |15 valid tracks,105 pairs; finite loss and gradients; no reference table read; no held-out/generalization claim |
| Independent review + regression fixes | Stale manifest/config rejected, frozen model/head replacement rejected, interrupt writes FAILED, single-epoch/no-warmup training keeps a nonzero LR |

The final synthetic pipeline produced ten reported arms, including A0, A_full, A_prime, five learned controls, FixedRule and the B inference-shuffle diagnostic. Its four test components correctly yielded `descriptive_insufficient_clusters` with no confidence interval. Scores and elapsed CPU time are intentionally not interpreted as research performance or server speed.

## Acceptance evidence

| Contract | Evidence |
| --- | --- |
| AC1–3 native input, isolation, all eligible pairs, quarantine | `test_data.py`, `test_dataset.py`, native development input check |
| AC4 real TCM/TAM gradients, losses and reload | `test_models.py`, `test_reference.py`, full smoke |
| AC5 score-only context, missing edges, shuffle, K=0 | `test_context.py`, `test_heads.py` |
| AC6 full controls and no test selection | `test_pipeline_contracts.py` verifies all seeds finish selection before test read; full smoke runs actual fits |
| AC7 immutable binding/failure behavior | Runtime, reference, heads, profile and pipeline regression tests; CLI no-CUDA failure and reuse run |
| AC8 metrics, uncertainty, bridge/joint-error diagnostics | `test_metrics.py`, generated report and low-component behavior |

Unverified: CUDA/RTX3090 execution and memory feasibility, production-model throughput, complete real-data training/convergence, and the method's scientific value. The current ENV-001 handover remains unchanged; server environment confirmation and a separate bounded profile handover precede formal execution. Server AI supervises processes and returns evidence; local AI reviews results and decides changes.
