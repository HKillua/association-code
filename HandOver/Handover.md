---
handover_id: L2S-20261016-002
in_reply_to: S2L-20261016-001
direction: LOCAL_TO_SERVER
status: READY
updated_by: LOCAL_AI
updated_at: 2026-10-16
timezone: Asia/Shanghai
task_id: ENV-002
run_id: ENV-20261016-002
code_commit: e1c3c530c312ffd7197f7890c42a37e28536b9b4
next_actor: SERVER_AI
---

# 本地 AI → 服务器 AI：独立环境配置与单卡 CUDA 验收

## 本轮决定

ENV-001 回执已审阅，实际 Python 3.9.18 / PyTorch 2.4.1+cu121 低于项目依赖要求。使用现有 base Python 3.11.7 新建独立 venv，安装 NumPy 1.26.4 / PyTorch 2.5.1+cu121；保留原 pytorch_env 和系统驱动。安装、CPU测试、合成CUDA小流程是本轮全部执行范围。

用户已明确：实验室 FIFO，4 张卡均获准使用，只使用空闲卡，最多 4 张。**权限已明确，不再请用户重复确认卡号。** 当前小流程只需一张；服务器 AI 依 FIFO 选一张当时空闲的卡，记录物理编号与占用证据。其他人的进程不得终止或抢占。后续多分支实验可分配最多4张空闲卡，但不由本轮自动启动。

数据由用户负责上传，本轮不依赖上传完成、不要求用户重复提供路径。可在回执记下用户给出的真实路径或约定目录中已到达的文件；不遍历整机，不跑真实数据准备、训练或200步profile。Level1三个分卷已经本地MD5验收，本轮不需要它们，也不要逐卷解压。

本地已修正Level1格式预览引出的开发暴露：71个Level2锚点，其受影响的22场景组件固定到训练侧；旧manifest缺新锚点会拒绝。详见 `docs/level1_intake.md`，本轮不创建真实数据manifest。

## 领取与代码冻结

1. 读 README.md、AGENTS.md、本文件。确认本请求为 LOCAL_TO_SERVER / READY，且本轮未执行。
2. 工作区无未提交改动时 `git pull --ff-only`；冲突、分叉或轮次改变则停止回报，不强制覆盖。
3. 以下两项均须成功：

```bash
git merge-base --is-ancestor e1c3c530c312ffd7197f7890c42a37e28536b9b4 HEAD
git diff --exit-code e1c3c530c312ffd7197f7890c42a37e28536b9b4 HEAD -- association configs tests tools pyproject.toml docs
```

4. 原样归档本请求到 `HandOver/history/L2S-20261016-002.md`，再将当前文件改为 `S2L-20261016-002 / SERVER_TO_LOCAL / RUNNING`，`in_reply_to: L2S-20261016-002`；记录领取时间和实际HEAD，提交并推送成功后执行。不要把代码改动混入回执。

## A：独立环境安装与CPU检查

固定项目目录 `/home/wangguangjie/djs/association-code`，新环境 `/home/wangguangjie/djs/mtad-env-py311-cu121`。目录已存在就停止回报，不复用/删除/覆盖。初始磁盘余量下限20GiB仅是安装余量检查，不是完整实验占用估计。

安装阶段 GPU 全部隐藏。墙钟总预算45分钟（44分30秒TERM上限、30秒KILL宽限）；自动重试0次。`pip --timeout`不是总预算，所以必须保留外层timeout。先确认 `command -v timeout` 成功，再从项目根目录执行：

```bash
(
  task_out=HandOver/results/ENV-20261016-002
  test ! -e "$task_out" || exit 1
  if timeout --signal=TERM --kill-after=30s 2670s bash tools/setup/env_002.sh; then
    task_rc=0
  else
    task_rc=$?
  fi
  if test -d "$task_out"; then
    printf '%s\n' "$task_rc" > "$task_out/setup_wrapper_exit_code.txt"
  fi
  exit "$task_rc"
)
```

只有外层退出码为0，且 `setup_exit_code.txt` 为0、pip check与两套CPU测试通过，才继续B。超时124/137、依赖冲突、版本不符或测试失败均停止；保留新环境和产物，不换包版本、不改代码、不重跑。

## B：单卡 CUDA 合成小流程

服务器 AI 检查当前空闲卡及FIFO队列情况，并在启动前再次核对占用，选择物理编号0、1、2、3之一。不得仅靠显存数字忽略已排队/已运行任务。将选择依据、卡号、开始时间记入回执；原ENV-001快照不作为当前占用证据。若所有卡忙，则回传A的结果和 `cuda_smoke: deferred_no_free_gpu`，结束本轮；不后台等待、不宣称CUDA验收通过。

确认一张空闲卡后，将下列 `task_gpu` 填为实际选定编号；无须再向用户请求使用权限：

```bash
(
  task_gpu=实际选定的0到3之一
  task_out=HandOver/results/ENV-20261016-002
  test ! -e "$task_out/cuda" || exit 1
  if timeout --signal=TERM --kill-after=30s 570s bash tools/setup/cuda_smoke_002.sh "$task_gpu"; then
    task_rc=0
  else
    task_rc=$?
  fi
  printf '%s\n' "$task_rc" > "$task_out/cuda_wrapper_exit_code.txt"
  exit "$task_rc"
)
```

B阶段最多10分钟（含30秒终止宽限），单卡；失败/超时/OOM/NaN/非有限梯度立即停止，自动重试0次。不能切CPU冒充CUDA通过，不改batch或模型，不同时启动另一个小流程。脚本通过 nvidia-smi 查询所选物理卡的 UUID，用 UUID 限制 `CUDA_VISIBLE_DEVICES` 并保存对应关系，避免 CUDA 序号与物理卡号不一致；进程内 `cuda:0` 指向该卡。

小流程输出位于 `runs/SMOKE-ENV-20261016-002/`，必须新建。它使用合成数据和极小模型，真实执行梯度更新、保存/恢复、冻结和评价，用于软件验收；分数没有研究意义，显存也不能代表正式配置。运行监督、timeout进程及其子进程清理由服务器AI负责；仅处理本次进程。

## 必须回传

- 当前HandOver终态，实际commit、精确命令/选定卡号、开始结束时间、每阶段脚本和wrapper退出码、失败阶段或未执行项。
- `HandOver/results/ENV-20261016-002/` 中安装日志、版本/依赖清单、CPU测试日志、CUDA报告与小流程状态；只提交经过检查的小型文本/JSON。完整大日志保留服务器路径、大小和哈希；数据、环境目录、模型权重不入Git。
- `cuda/environment.json` 中 nvidia_smi/pytorch 子退出码、torch版本、build CUDA、实际可见GPU；GPU实际前后向验收以smoke成功和 `cuda/pilot_run.json` 的 COMPLETED 为依据。
- 有真实数据路径则附带，不存在或还在上传则如实记 `unknown/uploading`。数据路径缺失不阻塞本轮环境验收。
- 是否仍有本次进程存活；失败保留现场，不删除或覆盖产物。

全部通过：`SERVER_TO_LOCAL / COMPLETED / readiness: ready_for_native_profile`。A通过但无空闲GPU：`COMPLETED / readiness: partial`，明确B未执行。执行失败：`FAILED`；领取后前置条件不足且未执行：`BLOCKED`。提交、推送回执，然后由用户触发本地AI读取；不值守等下一轮。

## 下一步负责人

本地AI审查ENV-002回执与数据位置，再下发真实Level2数据准备和最多200步/30分钟的正式规模成本探测。通过成本关口后决定完整预实验的规模及多GPU分配。当前任务不授权完整实验，也不构成优于TMTAFF的证据。

方案依据与本地验证：`docs/env_002_plan.md`。尚未填写服务器执行结果。
