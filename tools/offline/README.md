# 离线全窗口入口

本目录承接完整窗口下的评价和同行融合训练编排。数据读取、模型、训练与指标模块继续共享 `association/`，共用分析工具仍在 `tools/analysis/`。在仓库根目录使用已安装项目依赖的 Python 环境。

```bash
python -m tools.offline --help
python -m tools.offline system-evaluate --help
python -m tools.offline peer-train --help
```

这是源代码 checkout 的工具入口，当前安装包不包含 `tools`。研究目标和实验执行安排分别见[研究状态](../../docs/research_status.md)与[交接](../../HandOver/Handover.md)。

## 全场系统评价

`system-evaluate` 使用已冻结的 Level1 场景划分和 OLS 工作点，读取全部 `V_select` 场景，组合关联分组与普通/已训练融合，写入新输出目录。默认只用普通平均融合。提供关联或融合权重时必须同时提供对应运行和审计文件，原入口的绑定检查继续生效。

```bash
python -m tools.offline system-evaluate \
  --archive "$ARCHIVE" \
  --manifest "$MANIFEST" \
  --alignment "$ALIGNMENT" \
  --ols-working-point "$OLS_WORKING_POINT" \
  --output "$NEW_OUTPUT" \
  --device cpu
```

| 输入/输出 | 行为 |
|---|---|
| `--archive` | 一个或多个原 MTAD 压缩包，保持既有清理和全窗 `canonical_all` 输入权限 |
| `--manifest` / `--alignment` / `--ols-working-point` | 使用既定场景划分、时间对齐与 OLS 工作点，保留冻结阈值和统计量 |
| `--association-*` | 可选的基线、验证选模或末步关联端点，权重、分数和完整运行记录必须匹配 |
| `--fusion-checkpoint` / `--fusion-run` | 可选的已完成融合模型端点，必须成对提供 |
| `--output` | 新目录，保留 `run.json`、`summary.json`、逐场预测和系统对齐数组 |

该命令不训练、不打开封存测试集。它评价项目的全窗口协议，不等于 TMTAFF 作者完整协议复现，也不直接运行147/178等历史专用候选。主要实现为 [system_evaluate.py](system_evaluate.py)。

## PTFM / DeepAF 固定预算训练与评价

`peer-train --method ptfm|deepaf` 读取已完成的151全量新单位缓存，按原152流程从头训练指定方法，训练结束后在既定WS分组上评价全部V场景。它会进行真实训练并写新结果，具体实验范围仍由[交接](../../HandOver/Handover.md)确定。

必需参数为 `--data-run`、`--partition-run`、`--qhard-run`、`--output`、`--expected-training-groups`、`--expected-updates`、`--device`、`--time-policy advisory`，与原工具一致。实际整数曝光量来自准备后的索引及具体实验合同。数据准备仍使用 [fusion_peer_full_data.py](../experiments/fusion_peer_full_data.py)。

- PTFM固定150轮、batch32，双源共同支持核心域与公共Mean扩展分开报告。没有原文验证早停/降学习率，因此是项目任务适配。
- DeepAF固定20轮、batch16，保留独立位置/速度网络及现有离线输入，输出速度/航向沿用Mean。不能称原论文完整在线系统。
- 原模型、损失、数据人口、末步选取、失败保全及输出格式保持不变，无自动重试、续训或测试集访问。

实现见 [peer_fit.py](peer_fit.py)，条件与差异见[既定训练规格](../../docs/proposals/fusion_peer_full_training_20261030.md)。作者数值复现、我们的同协议适配是另需明确数据/窗口/选模条件的工作，不能只换入口名称就认定完成。

## 兼容与验证

旧 `tools/experiments/level1_system_evaluate.py`、`fusion_peer_full_fit.py` 的直接脚本、`python -m`、导入及可patch模块路径保留，指向这里的同一实现。原合同命令不必迁改。

| 验收条件 | 检查方式 |
|---|---|
| 新入口保留参数、默认值和错误处理 | 新旧 `--help`、非法命令/缺参数及委派检查 |
| 旧调用与patch仍作用于实际实现 | 新旧模块对象同一性、既有Level1与同行训练测试 |
| 数值与数据边界保持 | 原Level1系统合同测试、同行双支梯度/全目标核心域测试 |
| 目录调整不启动研究实验 | 只运行软件夹具及命令帮助，不读真实研究数据/权重、不提交训练 |
