# association-code

MTAD 航迹关联与融合研究的执行代码仓库。**所有需要在服务器运行的脚本、模型、配置、评价代码和依赖说明都在本仓库版本化维护。** 本地研究笔记和论文原件另存；每次执行所必需的协议与参数必须随代码提交，不能依赖服务器无法访问的本地文件。

## 研究目标

参考 TMTAFF 的任务构造，研究有证据支持的改进方法。首周需要提出至少一个有机制依据、可验证的候选，争取获得其相对 TMTAFF 有潜力的初步证据；完整实验窗口约两个半月，算力为 RTX 3090 ×4，以实际可用资源为准。

简单距离基线和数据检查只负责验收流程，不能充当“有机会超过 TMTAFF”的证据。正式方法在本地完成文献、迁移条件和公平对照审查后下发；不预先限定全局关联、学习平滑或离线。具体边界见 [研究目标与比较原则](docs/research_brief.md)。

## 谁负责什么

| 角色 | 责任 |
| --- | --- |
| **本地 AI（研究与开发）** | 检索/读论文、选择方法、研究决策、设计实验、编写和测试代码、固定配置/评价协议；读取回执后解释结果并决定下一轮 |
| **服务器 AI（执行与运行管理）** | 拉取代码、核对环境和数据、按交接执行命令、启动与监督进程、监控资源/日志、处理运行现场、保存结果并推送回执 |
| 用户 | 提供机器/数据访问条件、可用 GPU 与研究反馈；有需要时协助获取论文 |

**本地 AI 不负责在服务器上值守、轮询日志或监控运行。服务器 AI 负责执行全程。** 本地 AI 在收到回执、用户触发下一轮时读取仓库并作决定；Git 推送不是自动唤醒 AI 的机制，本仓库没有后台调度器。

服务器 AI 可以按交接选择允许的空闲设备、使用已有环境、保留日志和中止达到停止条件的本次进程。修改算法、修代码、变更数据处理/划分、阈值/指标、扩大训练/场景/种子预算，均由本地 AI 决定并提交新版本。代码错误回传现场，不自行修改后冒称原计划已成功。

## 唯一交接入口

每轮先读 **[HandOver/Handover.md](HandOver/Handover.md)**，它只登记当前待处理和已有轮次；真正的任务与回执是各自独立的文件，一律用[模板](HandOver/TEMPLATE.md)的固定字段。一方一个文件、创建后不覆盖成别的轮次，状态演变由 Git 历史保留，不设 `history/` 副本目录。

| direction | 谁给谁 | 主要内容 |
| --- | --- | --- |
| `LOCAL_TO_SERVER` | **本地 AI → 服务器 AI** | 决策、待执行代码提交、协议/配置、精确命令、资源预算、输出与停止条件 |
| `SERVER_TO_LOCAL` | **服务器 AI → 本地 AI** | 实际代码提交、环境/命令、进程状态、退出码、结果、异常与执行偏差 |

状态为 `READY`（可领取）、`RUNNING`（已领取运行中）、`COMPLETED`（已完成所列工作）、`FAILED`（执行失败）、`BLOCKED`（条件不足）。`COMPLETED` 不代表假设成立，也不代表 CUDA 可用；必须读具体结果。

### 每轮如何交接

1. **本地下发。** 先提交通过本地检查的代码，再新建 `HandOver/L2S-<YYYYMMDD>-<NNN>.md`，写 `LOCAL_TO_SERVER / READY`，并在 `HandOver/Handover.md` 索引登记该轮。`code_commit` 填代码提交的完整 SHA；交接文档另一次提交，避免让文档引用包含自身的未知 SHA。命令、配置、预算和验收标准必须可在服务器直接理解。下发后本地不再修改这个文件。
2. **服务器领取。** 拉取后读索引指向的 `L2S-*.md`，检查方向、状态、`handover_id`、`code_commit`、前置条件。同一任务已经完成则不重复运行。**不修改本地那份文件**（它本身就是存档，不再复制进 `history/`）；新建自己的 `HandOver/S2L-<YYYYMMDD>-<NNN>.md`，`SERVER_TO_LOCAL / RUNNING`，设置回执 ID 和 `in_reply_to`，更新索引并推送领取记录后再启动。
3. **服务器执行与回传。** 保持本次代码不变，记录实际命令和代码 SHA。运行中由服务器 AI 监督；完成/失败后**只更新自己那份 `S2L-*.md`**为终态并同步索引，提交小型报告、指标和必要日志。输出用唯一 `run_id` 目录，保留首次失败，不能覆盖后重跑。重试仅限交接已明确允许的范围。
4. **本地决策。** 收到终态回执后审阅证据，旧轮次文件原样留在 `HandOver/`，再**新建** `L2S-...-<下一个NNN>.md` 下发新任务并更新索引。未收到终态或双方明确取消前，不修改运行中的 `S2L-*.md`，也不默认开启下一实验。

每次读取/写入前核对工作区并同步：

```bash
git status --short
git pull --ff-only
```

若有未提交修改，先判断归属并保存，不直接拉取覆盖。若出现分叉、冲突、轮次已被另一方改变或代码不符合指定 SHA，停止执行并报告，不能强推、`reset --hard` 或静默合并交接语义。每份 `L2S-*.md` 只由本地写、每份 `S2L-*.md` 只由服务器写，索引 `Handover.md` 是唯一双方都会追加登记的公共文件；写索引前先拉取再改，不删除对方登记的轮次。运行中本地不向 `main` 推送下一实验的代码。代码冻结的核验方式见本轮交接。

回传可以使用：

```bash
git add HandOver/S2L-<YYYYMMDD>-<NNN>.md HandOver/Handover.md HandOver/results
git diff --cached --stat
git commit -m "handover: return ENV-20261016-001 results"
git push origin main
```

提交信息和输出路径按实际轮次修改。只提交已核对的小型产物；数据集、权重、完整大日志、缓存及凭据不进仓库，大产物回传服务器路径、大小和哈希。若没有推送权限，保留本地提交并请用户传回回执，不把“本地提交成功”称为“本地 AI 已收到”。

## 服务器 AI 首次开始

```bash
git clone https://github.com/HKillua/association-code.git
cd association-code
```

已有 checkout 则同步后读取 `README.md`、`AGENTS.md` 和 `HandOver/Handover.md` 索引，再打开索引指向的状态为 `READY` 的 `L2S-*.md`。**只执行该交接列出的任务**，不要把 README 的示例当成另一轮任务，也不要重跑已有终态回执的轮次。

第一轮 ENV-001（环境与数据位置核验）已由服务器回传终态 `COMPLETED` / readiness `partial`：计算环境可用，MTAD 数据尚未上传到服务器。后续代码与实验由本地逐轮新建交接下发；索引中没有 `READY` 时，服务器不自行开始任何工作。

## 已实现的模型与实验代码

已实现可训练的 **TMTAFF 风格 TCM + TAM 关联参考模型**（CE / contrastive 两个损失分支），以及 CDF1 单次上下文残差和对照实验。它是独立实现的关联预实验，尚未复现作者完整 TFM 融合系统，也没有真实数据训练结果或“超过 TMTAFF”的结论。具体对应与差异见 [模型实现说明](docs/model_conformance.md)。

| 路径 | 用途 |
| --- | --- |
| `association/data.py`、`dataset.py` | 原生 Level2 ZIP、A–B 身份标签、共享 A 组件隔离划分、异常记录与时间配准 |
| `association/models.py`、`reference.py` | 实际 TCM/TAM 模型、梯度训练、验证选权重、保存与恢复、冻结分数缓存 |
| `association/context.py`、`heads.py` | 单调校准、CDF1、EndpointMarginal / PairMLP / ScenePool / FixedRule / B_shuffle |
| `association/metrics.py`、`pipeline.py` | 全部种子先完成验证选择，再读取测试量测；逐场景指标、组件 bootstrap、误桥与第三节点诊断 |
| `association/cli.py`、`profile.py` | 数据准备、GPU 实测、单个参考训练、完整实验、合成流程检查 |
| `configs/pilot.json`、`configs/smoke.json` | 正式规模候选配置与独立的小模型检查配置 |
| `tests/` | 数据隔离、模型梯度、冻结选择、缓存、评价与失败路径测试 |
| `tools/pilot/` | 原有环境检查与 ZIP 审计脚本，仍可仅用标准库运行 |
| `HandOver/` | 按 `handover_id` 逐轮保存的请求与回执文件、交接索引和小型执行证据 |

正式模型在服务器明确使用 **`--device cuda:0`**；CUDA 不可用会失败，不会自动退到 CPU。4 张 RTX 3090 可以分配给独立实验；代码当前按单卡执行，不自动占满 4 卡，也不把多卡数当作已确认的配额。数据读取、统计和指标计算在 CPU，神经网络前后向和优化在所选 GPU。

安装时先由服务器环境回执确定兼容的 CUDA 版 PyTorch；本地 AI 再下发安装命令。不要根据 Mac 的 CPU 验证环境安装服务器 PyTorch。已有兼容 PyTorch 后可在仓库根目录执行 `python -m pip install -e .`。依赖范围见 `pyproject.toml`；实际 Python / PyTorch / CUDA 版本随运行报告记录。

### 使用与检查

以下是接口示例，不替代当前 HandOver 任务。完整命令与产物见 [执行协议](docs/pilot_execution.md) 和 [GPU 执行说明](docs/gpu_execution.md)。

```bash
python -m association.cli --help
python -m unittest discover -s tests
python -m unittest discover -s tools/pilot -p 'test_audit_mtad_level2.py'
python -m association.cli smoke --output runs/SMOKE-unique --device cpu
```

合成 smoke 会真实训练四个小参考模型和所有上下文对照，保存并重新载入权重，再生成测试报告；其分数没有科研意义。正式配置的耗时、显存与收敛必须在服务器验证。当前尚未验证 CUDA 实测和真实 MTAD 训练，本地检查结果见 [验证记录](docs/verification-2026-10-16.md)，下一步按 [实施状态](HandOver/implementation_status.md) 衔接。

所有输出使用新目录，已存在即拒绝覆盖；失败保留 `run.json`、已有曲线和缓存。数据、PDF、权重、大日志与 `runs/` 不进 Git。任何参数、划分或模型变更都由本地提交并在后续交接明确记录。
