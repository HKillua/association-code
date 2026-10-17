# GPU execution notes

**This page contains examples, not a READY task.** The current execution authority is the separate round file pointed to by `HandOver/Handover.md`. The commands below are examples; the current handover identifies the authorized stage and frozen code.

The local AI writes and tests code, fixes defects, chooses protocols, and interprets results. The server AI executes an authorized frozen request, monitors processes/resources/logs, enforces its stop conditions, and returns evidence. A Git push does not start a run or transfer monitoring responsibility to the local AI.

## Device selection

The user's direct authorization on **2026-10-17** permits dynamic GPU selection and sharing with other users. It supersedes the idle-only/no-sharing restrictions in historical ENV-004/005 instructions. The server selects a compatible visible device using current free memory and load; existing compute processes do not veto selection. Prefer the candidate with the most free memory, breaking ties by lowest utilization. No GPU model or physical GPU index is fixed. Recheck the candidate before starting and bind its UUID for that attempt.

Formal model execution must explicitly request CUDA. The following is a later training interface example, using the UUID selected at runtime; it does not authorize training during environment verification:

```bash
CUDA_VISIBLE_DEVICES="$task_gpu_uuid" python -m association.cli run-pilot \
  --archive /data/mtad/level2.zip \
  --prepared runs/PREP-001 \
  --config configs/pilot.json \
  --output runs/PILOT-001 \
  --device cuda:0
```

Within that process, `cuda:0` refers to the first visible GPU identified by `task_gpu_uuid`, not a fixed physical index. CUDA absence is an execution failure; the formal run must not silently fall back to CPU. Record the actual device model/UUID, visible-device mapping, free memory/load, sharing status, PyTorch/CUDA versions, code SHA, config/manifest hashes, elapsed time, and the process's peak allocated memory. Distinguish process allocation from whole-device occupancy; measurements under contention are not exclusive-device benchmarks.

The implementation runs on one selected GPU per job and does **not** default to DistributedDataParallel or occupying every visible card. Sharing a card is allowed; independent parallel jobs still require the current handover's job-count and aggregate budget. Preserve separate output paths and never terminate other users' processes. Resource failures may trigger a different candidate within the same budget and attempt limit, without changing the model, batch, precision, dependencies, or data protocol.

ENV-005 has already passed CPU and CUDA verification on the server. GPU-POLICY-006 only synchronizes this policy and performs one bounded, read-only device query; it does not repeat that verification. Later GPU tasks supply their own execution and waiting budgets. Resource failures may use up to 3 attempts within the current task's explicit limit, including termination grace, without resetting its deadline. The separate READY request supplies executable commands and any required CPU evidence source.

## Preparation and local verification

Data preparation performs no GPU computation. A later authorized request may use:

```bash
python -m association.cli prepare \
  --archive /data/mtad/level2.zip \
  --config configs/pilot.json \
  --output runs/PREP-001
```

The output directory must be new. Preparation hashes the archive, creates the shared-A component manifest, checks training/validation coverage, and defers held-out test measurements until frozen evaluation. Training must use that saved manifest rather than generate a different split opportunistically.

Local unit tests and synthetic CPU smoke runs validate interfaces, gradients, caching, and report generation. They **do not** establish that CUDA training succeeds, provide runtime or memory estimates on the selected GPU, validate the real-data reference model, or measure research performance. The server must first satisfy the actual environment handover and then receive a separate bounded training request.

Keep data, weights, caches, and full logs on the server. Return small reports and artifact paths/hashes through the HandOver protocol. Preserve failed run directories; do not overwrite them or retry with a changed model/split/config without a new local decision.
