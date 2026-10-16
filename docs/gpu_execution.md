# GPU execution notes

**This page contains examples, not a READY task.** The current execution authority is the separate round file pointed to by `HandOver/Handover.md`. The commands below are examples; the current handover identifies the authorized stage and frozen code.

The local AI writes and tests code, fixes defects, chooses protocols, and interprets results. The server AI executes an authorized frozen request, monitors processes/resources/logs, enforces its stop conditions, and returns evidence. A Git push does not start a run or transfer monitoring responsibility to the local AI.

## Device selection

Formal model execution must explicitly request a CUDA device. After the user/server has confirmed a particular physical GPU is allocated, restrict visibility before invoking Python. For example, **only if physical GPU 2 is allocated**:

```bash
CUDA_VISIBLE_DEVICES=2 python -m association.cli run-pilot \
  --archive /data/mtad/level2.zip \
  --prepared runs/PREP-001 \
  --config configs/pilot.json \
  --output runs/PILOT-001 \
  --device cuda:0
```

Within that process, `cuda:0` refers to the first visible GPU, which is physical GPU 2 in this example. It does not mean physical GPU 0. CUDA absence is an execution failure; the formal run must not silently fall back to CPU. Record the actual device name, visible-device mapping, PyTorch/CUDA versions, code SHA, config/manifest hashes, elapsed time, and peak allocated memory.

Four RTX 3090 cards permit separately authorized independent jobs; the implementation does **not** default to DistributedDataParallel, four-GPU training, or occupying every visible card. Only dispatch parallel branches when a later handover explicitly assigns devices, separate immutable output paths, and an aggregate GPU-hour budget. One physical GPU should not accidentally receive concurrent independent jobs.

## Preparation and local verification

Data preparation performs no GPU computation. A later authorized request may use:

```bash
python -m association.cli prepare \
  --archive /data/mtad/level2.zip \
  --config configs/pilot.json \
  --output runs/PREP-001
```

The output directory must be new. Preparation hashes the archive, creates the shared-A component manifest, checks training/validation coverage, and defers held-out test measurements until frozen evaluation. Training must use that saved manifest rather than generate a different split opportunistically.

Local unit tests and synthetic CPU smoke runs validate interfaces, gradients, caching, and report generation. They **do not** establish that CUDA training succeeds, provide runtime or memory estimates on an RTX 3090, validate the real-data reference model, or measure research performance. The server must first satisfy the actual environment handover and then receive a separate bounded training request.

Keep data, weights, caches, and full logs on the server. Return small reports and artifact paths/hashes through the HandOver protocol. Preserve failed run directories; do not overwrite them or retry with a changed model/split/config without a new local decision.
