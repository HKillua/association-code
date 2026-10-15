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

每轮先读 **[HandOver/Handover.md](HandOver/Handover.md)**。这是持续迭代的当前任务/回执；[模板](HandOver/TEMPLATE.md)固定字段，[history](HandOver/history/README.md)和 Git 历史保留旧轮次。

| direction | 谁给谁 | 主要内容 |
| --- | --- | --- |
| `LOCAL_TO_SERVER` | **本地 AI → 服务器 AI** | 决策、待执行代码提交、协议/配置、精确命令、资源预算、输出与停止条件 |
| `SERVER_TO_LOCAL` | **服务器 AI → 本地 AI** | 实际代码提交、环境/命令、进程状态、退出码、结果、异常与执行偏差 |

状态为 `READY`（可领取）、`RUNNING`（已领取运行中）、`COMPLETED`（已完成所列工作）、`FAILED`（执行失败）、`BLOCKED`（条件不足）。`COMPLETED` 不代表假设成立，也不代表 CUDA 可用；必须读具体结果。

### 每轮如何交接

1. **本地下发。** 先提交通过本地检查的代码，再写 `LOCAL_TO_SERVER / READY`。`code_commit` 填代码提交的完整 SHA；交接文档另一次提交，避免让文档引用包含自身的未知 SHA。命令、配置、预算和验收标准必须可在服务器直接理解。
2. **服务器领取。** 拉取后检查方向、状态、`handover_id`、`code_commit`、前置条件。同一任务已经完成则不重复运行。把原请求原样保存为 `HandOver/history/<handover_id>.md`；当前文件换成 `SERVER_TO_LOCAL / RUNNING`，设置自己的回执 ID 和 `in_reply_to`，推送领取记录后再启动。
3. **服务器执行与回传。** 保持本次代码不变，记录实际命令和代码 SHA。运行中由服务器 AI 监督；完成/失败后更新同一回执为终态，提交小型报告、指标和必要日志。输出用唯一 `run_id` 目录，保留首次失败，不能覆盖后重跑。重试仅限交接已明确允许的范围。
4. **本地决策。** 收到终态回执后审阅证据，将回执归档，再写新的 `LOCAL_TO_SERVER` 任务。未收到终态或双方明确取消前，不覆盖运行中的交接，也不默认开启下一实验。

每次读取/写入前核对工作区并同步：

```bash
git status --short
git pull --ff-only
```

若有未提交修改，先判断归属并保存，不直接拉取覆盖。若出现分叉、冲突、轮次已被另一方改变或代码不符合指定 SHA，停止执行并报告，不能强推、`reset --hard` 或静默合并交接语义。双方交替更新当前文件；运行中本地不向 `main` 推送下一实验的代码。代码冻结的核验方式见本轮交接。

回传可以使用：

```bash
git add HandOver/Handover.md HandOver/history HandOver/results
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

已有 checkout 则同步后读取 `README.md`、`AGENTS.md` 和 `HandOver/Handover.md`。**执行当前交接列出的任务**，不要把 README 的示例当成另一轮任务。

当前第一轮仅检查环境并回报数据路径/可用 GPU；不安装训练环境、不训练、不重复全量数据审计。后续代码与实验由本地逐轮下发。

## 目录与当前代码

| 路径 | 用途 |
| --- | --- |
| `tools/pilot/server_check.py` | 环境检查，输出 Python/PyTorch/CUDA/GPU/空间报告 |
| `tools/pilot/audit_mtad_level2.py` | 只读 ZIP 完整性与抽样内容检查；异常时保留报告并返回非零 |
| `tools/pilot/test_audit_mtad_level2.py` | 使用合成小 ZIP 的 15 项行为测试 |
| `HandOver/` | 双向交接、历史和小型执行证据 |
| `docs/` | 执行所需研究边界、版本迁移与本地检查记录 |

现有脚本只依赖 Python 标准库；环境脚本会探测已安装的 PyTorch，缺失也能生成报告。尚未实现 TMTAFF 参考模型或新方法训练程序，不把基础脚本当作模型成果。

本地检查命令：

```bash
python -m unittest discover -s tools/pilot -p 'test_audit_mtad_level2.py'
python tools/pilot/server_check.py --help
python tools/pilot/audit_mtad_level2.py --help
```

工具细节见 [tools/pilot/README.md](tools/pilot/README.md)。新增服务器代码继续进入本仓库，不能重新在聊天附件或研究笔记目录维护另一份可编辑副本。
