---
package_type: EXPERIMENT_RECORD
package_version: research-pipeline-contract v1.0
experiment_id: ENV-20261016-002
source_idea_id: none
plan_version: L2S-20261016-002
execution_status: interrupted
recorded_at: 2026-10-16
timezone: Asia/Shanghai
next_routes: ENV-003 环境恢复与验收
next_skill: root-cause-debugging
---

# ENV-002 环境安装失败记录

本记录由本地 AI 根据服务器终态回执和已提交原始产物核验。它是工程验收记录；没有模型实验结果，也没有关于 CDF1、TMTAFF 或研究方向优劣的科学结论。原请求、服务器回执和失败现场保留，本记录不代写或改写服务器回执。

## 目的与设计

问题是独立的 Python 3.11.7 / NumPy 1.26.4 / PyTorch 2.5.1+cu121 环境能否完成依赖检查、CPU 测试和单卡 CUDA 合成流程。通过标准为 A 阶段安装与全部 CPU 检查成功后，B 阶段在按 FIFO 分配的一张空闲 GPU 上成功完成合成 smoke。

实验单位为一次环境配置与验收任务，无直接对照，仅执行一次，自动整轮重跑为 0。未使用 MTAD 量测、标签或真实数据划分；随机种子不适用于已执行的安装阶段。计划的合成数据、模型训练、权重选择及评价均未执行。

原协议：[L2S-20261016-002](../HandOver/L2S-20261016-002.md)。执行者为服务器 AI，本地 AI 负责证据核验与下一轮决策。

## 计划与实际

| 项目 | 计划 | 实际与影响 |
| --- | --- | --- |
| 环境 | 新建 `/home/wangguangjie/djs/mtad-env-py311-cu121`，保留旧环境 | venv 已创建；NumPy 1.26.4 安装成功；torch 安装未完成 |
| A 阶段 | 安装依赖、editable 项目安装、pip check、导入断言、两套 CPU 测试、CLI 检查 | 在安装 torch 的依赖元数据请求处停止，后续检查全部未执行 |
| A 预算 | 外层 TERM 2670 秒，KILL 宽限 30 秒 | 实际 1959 秒；内外退出码均为 1，未触及总时限 |
| B 阶段 | A 全部通过后，单卡 CUDA 合成 smoke，最多 10 分钟 | A 前置条件未满足，整体未执行；没有 CUDA 通过或失败结论 |
| GPU 与数据 | 安装阶段隐藏全部 GPU；不读取真实数据 | 未分配或运行 GPU 验收，未生成真实数据结果 |
| 重跑与现场 | 失败停止，保留证据，不覆盖目录 | 本轮未重跑；下一轮不能直接使用原目录重新执行原脚本 |

时间文件记录开始为 `2026-10-16T07:39:59Z`、结束为 `2026-10-16T08:12:38Z`，对应北京时间 15:39:59—16:12:38。差值 1959 秒，与回执相符。

A 的执行入口为 `timeout --signal=TERM --kill-after=30s 2670s bash tools/setup/env_002.sh`。直接失败的命令是：

```bash
/home/wangguangjie/djs/mtad-env-py311-cu121/bin/python -m pip install --retries 0 --timeout 60 --no-cache-dir --only-binary=:all: torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
```

日志末尾错误为 `HTTPSConnectionPool(host='files.pythonhosted.org', port=443)` 对 `typing_extensions-4.16.0-py3-none-any.whl.metadata` 读取超时，`read timeout=60.0`。脚本 `set -e` 使失败的 pip 命令结束本阶段，退出 trap 记录状态。

## 证据来源与核验

原始产物目录为 [HandOver/results/ENV-20261016-002](../HandOver/results/ENV-20261016-002/)。血缘为：固定安装脚本 → pip 输出与退出状态 → 服务器回执 → 本地核验记录；没有产生模型、数据分析或指标文件。

| 证据等级 | 可确认内容 | 来源与限制 |
| --- | --- | --- |
| 已提交原始日志 | NumPy 安装成功；torch 主 wheel 下载进度达到 780.5/780.5 MB；随后依赖元数据请求读取超时 | [setup.log](../HandOver/results/ENV-20261016-002/setup.log)，2209 B；SHA256 `923d89c07f5fc989caeb85f742414dde2cc18a48bda9e4726630cbdb6a18b0e9`，本地重算一致 |
| 已提交原始状态 | 内外退出码均为 1，耗时 1959 秒 | `setup_exit_code.txt`、`setup_wrapper_exit_code.txt`、`setup_started_at.txt`、`setup_ended_at.txt` |
| 已提交代码与产物 | 安装失败后不会进入 pip check/CPU 测试；结果目录没有这些检查或 CUDA 产物 | [env_002.sh](../tools/setup/env_002.sh)、原请求和结果目录；与回执所述未执行一致 |
| 服务器文字报告 | 失败 venv 104 MB，pip list 无 torch；相关进程均已退出；随后对同一 URL 的 curl 返回 HTTP 200、耗时 0.888 秒 | [S2L-20261016-002](../HandOver/S2L-20261016-002.md)；未附对应原始命令输出，本地未独立重现 |

缺失产物包括 pip check、完整环境冻结、导入断言、两套 CPU 测试、CLI 检查和 CUDA 结果。它们因上游失败而未产生，不能视为检查失败，也不能根据原环境或本地 Mac 测试替代此次服务器验收。

## 实际代码版本记录差异

原请求冻结的代码提交为 `2e88f13587810f79cb358e3607d7361d6f55a049`。服务器回执 frontmatter 和正文把实际执行 HEAD 记为 `db5f59d5b3d7675664a66ce00d6c13dc811c9cb9`；但脚本在启动时执行 `git rev-parse HEAD` 保存的 [actual_commit.txt](../HandOver/results/ENV-20261016-002/actual_commit.txt) 为 **`a6d6a04a6360520a43d6de5f73ceb1d910201761`**，以该原始记录为准。

本地核验 Git 历史：`a6d6a04` 是服务器领取 ENV-002 为 RUNNING 的提交，相对 `db5f59d` 只修改 `HandOver/Handover.md` 和 `HandOver/S2L-20261016-002.md`；相对请求冻结提交，`association configs tests tools pyproject.toml docs` 的差异检查退出码为 0。因此属于回执把领取前 HEAD 与执行 HEAD 混淆，未发现执行代码越出冻结范围。后续由服务器 AI 在自己的记录中澄清，本地不修改其回执。

回执还将原脚本的目录保护引用为第 20 行；实际为第 13 行 `test ! -e "$task_env"`。保护行为存在，现有半成品目录会阻止原脚本按原路径再次安装，行号差异不影响这一判断。

## 观察、解释与待验证项

| 观察 | 当前解释与可信边界 | 区分或验证方式 |
| --- | --- | --- |
| 元数据读取超时，安装非零退出 | 直接失败点为网络读取；`--retries 0` 没有为这次请求提供重试机会。网络根因未知，不能把该参数本身称为网络根因 | 下一轮保持目标软件版本，记录有限重试下的各阶段输出与退出状态；成功只说明下一轮安装通过 |
| 主 wheel 显示完整下载 | 仅证明本次主文件传输完成，不能推出大文件通道长期稳定或剩余依赖一定能下载 | 新一轮分别记录主 wheel 和依赖下载结果 |
| 使用了 `--no-cache-dir`，没有主 wheel 保存路径或哈希 | **不能声称旧的 780.5 MB 文件仍可复用**；需按可能重下载安排预算 | 新一轮明确保存 wheel，记录文件大小和 SHA256 后才确认可用产物 |
| 服务器报告随后 curl 成功 | 与瞬时故障解释相容，但单个小请求成功不等于 pip 安装或 CUDA 验收成功 | 按安装、离线验收、CPU 测试、CUDA smoke 的完整顺序验证 |

按照双端分工，本地只核验回传产物，本次未在服务器重现原网络故障。没有重复运行、性能对照或统计估计，不计算置信区间。当前证据只能支持“ENV-002 安装阶段中断，CUDA 和真实数据试验尚未获得证据”；不能支持方法无效、硬件不足或超越 TMTAFF 等结论。

## ENV-003 决策摘要（待执行）

- 保留 ENV-002 的 venv、回执与全部产物。新环境使用 `/home/wangguangjie/djs/mtad-env-py311-cu121-env003`，新输出使用 `HandOver/results/ENV-20261016-003/`；不覆盖失败现场。
- 保持 Python 3.11.7、NumPy 1.26.4、PyTorch 2.5.1+cu121，系统驱动和研究配置不变。本轮处理安装与验收通路，不调整研究方法。
- 先下载 NumPy 与构建工具，再单独保存 torch 主 wheel，最后下载依赖并离线安装。保存成功的文件按大小与 SHA256 记录，只有实际落盘的完整文件才能作为可复用产物。
- 使用 pip 持久缓存、`--retries 2`、单次 socket `--timeout 120`，不宣称这些参数保证下载途中断流能够续传。下载失败停止并保留已保存产物；**自动整轮重跑 0 次**。
- A 阶段总墙钟上限 60 分钟，包含下载、安装及 CPU 检查；B 阶段按实验室 FIFO 使用一张空闲卡，最多 10 分钟。安装期间仍隐藏所有 GPU；失败或超时按新交接停止，由服务器 AI 监督和回传。
- 只有依赖检查、CPU 测试及 CUDA 合成 smoke 全部通过，才由本地 AI 审阅后另行下发真实数据准备与最多 200 步的成本探测。本记录不授权直接启动真实数据 profile 或完整训练。

此摘要是本地下一轮决定，不是 ENV-003 已执行结果。具体冻结提交、脚本、命令与验收清单以新一轮 L2S 交接为准。
