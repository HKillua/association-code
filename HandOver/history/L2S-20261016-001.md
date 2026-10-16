---
handover_id: L2S-20261016-001
in_reply_to: none
direction: LOCAL_TO_SERVER
status: READY
updated_by: LOCAL_AI
updated_at: 2026-10-16
timezone: Asia/Shanghai
task_id: ENV-001
run_id: ENV-20261016-001
code_commit: 29df6de0bbfc72807095e75d77fa803276177e65
next_actor: SERVER_AI
---

# 本地 AI → 服务器 AI：首次环境与数据位置核验

## 研究状态与本轮任务

本地负责研究决策、代码与结果解释；服务器负责执行、运行监督和回传。本地不值守服务器。本轮请核对现有 Python/PyTorch/CUDA、GPU 和空间，回报 MTAD 数据所在位置，**不启动训练、不安装依赖、不运行数据审计或自行实现新方法**。

首周目标已更新为有依据的候选方法及其相对 TMTAFF 的潜力证据。本地正在继续读论文、比较机制和设计公平参照。距离基线仅作后续流程验收；本轮环境结果不能称为模型实验。

## 执行前检查与领取

1. 确认当前仍为本请求、状态 READY，且本任务此前没有完成记录。先读 README.md、AGENTS.md。
2. 检查工作区；无待处理修改时 `git pull --ff-only`。出现冲突/分叉/轮次改变则暂停并回报，不能强制覆盖。
3. 指定代码提交已经存在。以下两条须成功；若失败，不要运行：

```bash
(
  git merge-base --is-ancestor 29df6de0bbfc72807095e75d77fa803276177e65 HEAD || exit 1
  git diff --exit-code 29df6de0bbfc72807095e75d77fa803276177e65 HEAD -- tools/pilot docs/research_brief.md || exit 1
)
```

4. 将本请求原样保存为 `HandOver/history/L2S-20261016-001.md`。把当前文件改为回执 `S2L-20261016-001`、`in_reply_to: L2S-20261016-001`、`direction: SERVER_TO_LOCAL`、`status: RUNNING`、`updated_by: SERVER_AI`，保留任务/代码信息，记录实际领取时间和 `git rev-parse HEAD`。提交并推送领取记录，确认成功后再执行；无推送能力则保留现场并通知用户。
5. 后续只更新该回执；本地等待终态，服务器负责本轮运行监控。新服务器任务不由 README 示例自动生成。

## 环境与预算

- 在用户准备使用的已有 Python 环境中运行；无需为了检查安装 PyTorch。若没有 `python`，使用 `python3` 并记录实际命令差异。
- 可用硬件据用户为 RTX 3090 ×4；实际可见/可用/分配情况由本轮确认，不能默认占满4卡。
- 本轮无训练、无 GPU 计算配额申请，仅环境探测；Python/PyTorch探测可能初始化设备查询。
- 运行总预算5分钟；脚本内部 nvidia-smi 超时20秒、PyTorch子进程超时30秒。
- 本轮允许自动重试0次。失败保留输出并回传；不自行换环境、改代码、清理别人的进程或重启机器。

## 精确命令

从仓库根目录运行。输出目录必须未存在；若已存在，先检查是否已执行，不能覆盖或自动换名重跑。

```bash
(
  git merge-base --is-ancestor 29df6de0bbfc72807095e75d77fa803276177e65 HEAD || exit 1
  git diff --exit-code 29df6de0bbfc72807095e75d77fa803276177e65 HEAD -- tools/pilot docs/research_brief.md || exit 1
  mkdir HandOver/results/ENV-20261016-001 || exit 1
  if python tools/pilot/server_check.py --workspace . --output HandOver/results/ENV-20261016-001/environment.json > HandOver/results/ENV-20261016-001/environment.log 2>&1; then
    task_env_exit=0
  else
    task_env_exit=$?
  fi
  printf '%s\n' "$task_env_exit" > HandOver/results/ENV-20261016-001/exit_code.txt
  exit "$task_env_exit"
)
```

目录创建失败时停止，不继续运行后两行。记录实际代码提交、当前文档提交、Python环境、开始/结束时间和退出码；无需额外运行训练来验证 GPU。

## 必须回传

1. `environment.json`、`environment.log`、`exit_code.txt`（存在的全部保留）；若失败发生在文件生成之前，保留终端错误、实际命令和失败阶段。
2. 摘要列出：Python/PyTorch版本、构建CUDA、CUDA可用性、PyTorch可见设备数、每张卡总/已用显存、工作目录剩余空间。明确 **`nvidia_smi.exit_code` 和 `pytorch.exit_code`**。脚本整体exit 0不代表两项都成功。
3. 服务器项目绝对路径；用户已提供/指定位置中的 MTAD ZIP 或解压目录路径、是否存在及可见的文件大小。若位置未提供或找不到，写 `unknown`，不遍历整机猜测。已有哈希可附，未计算则写 `not_checked`，本轮不重复数据审计。
4. 用户允许使用的 GPU 编号和时段；只知道占用快照时写“分配未确认”，不要用空闲快照冒充已预约。
5. 未执行项、异常、命令差异和仍需用户补充的条件。

无需发送密码、私钥或环境变量全集。数据与权重不上传；本次小型报告经检查后可提交 `HandOver/results/ENV-20261016-001/`。

## 终态与下一步

- 检查完成并清楚记录缺失项：`SERVER_TO_LOCAL / COMPLETED`，另写就绪结论 `ready / partial / not_ready`。缺PyTorch可作为已完成检查的发现，不能写成已可训练。
- 启动或脚本运行失败：`FAILED`；前置条件不足而未执行：`BLOCKED`。说明是否还有本次进程存活、已如何停止；保留证据。
- 提交并推送回执和小型产物，通知用户可让本地 AI 读取。Git 不自动唤醒本地 AI；服务器 AI 不需要持续等待下一任务或自行扩展执行。
- **下一步负责人：本地 AI。** 读取事实后决定环境配置与正式方法预实验，并另发新交接。当前没有授权的模型训练任务。

## 服务器回执

尚未领取、尚未执行；本地不预填结果。
