# ENV-002: isolated environment and CUDA acceptance

This document explains the decision; only `HandOver/Handover.md` authorizes execution.

ENV-001 confirms Python 3.9.18 / torch 2.4.1+cu121, driver 535.86.05 and four RTX 3090 cards. The repository requires Python >=3.10 and torch >=2.5. Use the existing base Python 3.11.7 to create a separate venv at `/home/wangguangjie/djs/mtad-env-py311-cu121`, then install NumPy 1.26.4 and torch 2.5.1 from the official cu121 index. Do not modify the old pytorch_env or upgrade the driver/toolkit.

The [official PyTorch previous-versions page](https://pytorch.org/get-started/previous-versions/) lists torch 2.5.1/cu121; the [official wheel index](https://download.pytorch.org/whl/cu121/torch/) includes CPython 3.11/Linux x86_64. [NVIDIA CUDA 12.1 release notes](https://docs.nvidia.com/cuda/archive/12.1.1/cuda-toolkit-release-notes/index.html) give the compatible driver information. These checks establish a reasonable installation target, not successful operation of this model on the server.

`tools/setup/env_002.sh` creates a new result directory, refuses an existing target venv, installs only into the new environment, records versions and package resolution, runs dependency checks and the CPU test suites. CUDA is hidden during this stage. An outer timeout of 45 minutes is required; pip network timeouts alone are not a wall-clock cap. Save the wrapper exit code even if a hard timeout prevents the script's EXIT trap from recording one. Preserve failed environment/output directories; no automatic retry or deletion.

The user authorizes FIFO use of any free lab GPU, up to 4 simultaneously. This task needs only one. The server AI chooses an idle physical card, records it, and runs the existing synthetic CLI smoke with CUDA_VISIBLE_DEVICES restricted to that card and --device cuda:0. The maximum is 10 minutes including a 30-second termination grace; an occupied card is not reclaimed. If no card is free, return environment results and GPU deferral without a background waiter. Formal parallel fits need explicit per-run device and output assignments in a later handover.

No real-data preparation, native-data training, 200-step profile, or full pilot is authorized by this environment task. The synthetic smoke uses tiny generated fixtures and real gradient updates; it tests the execution path but has no scientific performance meaning. Data transfer can happen in parallel and need not block installation/smoke.

## Local verification of this revision

- 39 focused tests passed: data 22, dataset 11, pipeline contracts 6.
- The new test proves a cross-level-exposed component stays training-side and a stale manifest is rejected before model execution.
- A labels-only native Level2 check read all 5000 association tables and verified that the affected 22-scene component is assigned T_core; no measurement/reference was loaded. Details: `level1_isolation_verification.json`.
- `bash -n tools/setup/env_002.sh` passed; installation and CUDA smoke have **not** been run locally. The installation script received a separate read-only review.
