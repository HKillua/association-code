# MTAD航迹关联与状态恢复

唯一代码仓。当前目标和结论只读[研究状态](docs/research_status.md)，服务器只领取[交接索引](HandOver/Handover.md)里的任务。

| 路线 | 工作目录 | 在仓库根目录查看命令 |
|---|---|---|
| 在线观测 | [online_observation](online_observation/README.md)，独立在线观测代码与回放 | `python -m online_observation --help` |
| 离线观测 | [tools/offline](tools/offline/README.md)，全场系统评价与同行融合训练/评价 | `python -m tools.offline --help` |

| 需要什么 | 唯一入口 |
|---|---|
| 先理解MTAD数据如何变成输出 | [输入→预处理→关联→融合→输出→评价](docs/specs/evidence-association-fusion.md#从输入到输出) |
| 当前目标、已知结果、下一步 | [研究状态](docs/research_status.md) |
| 当前执行安排 | [HandOver](HandOver/Handover.md)；[历次合同/回执](HandOver/README.md) |
| 协作与维护规则 | [AGENTS](AGENTS.md) |
| 科研方法与完整五技能契约 | [科研流程](docs/research_workflow.md) |
| 项目常用技能、完整正文与调用 | [8 项技能](.agents/skills/README.md) |
| 数据、单位与指标 | [数据约定](docs/data_contract.md)、[论文指标](docs/mtad_data_protocols.md) |
| 完整报告、提案与模块说明 | [文档导航](docs/README.md)、[报告索引](docs/reports/README.md)、[提案索引](docs/proposals/README.md) |
| 实现接口与命令 | [流程与保留组件](docs/specs/evidence-association-fusion.md)、[工具导航](tools/README.md) |
| 结案结论、完整推导沿革与旧文件恢复 | [历史摘要](docs/history/README.md)、[过程原文](docs/history/2026-10-30-research-process.md) |

在线代码和验证放在 `online_observation`。离线编排在 `tools/offline`，相关数据、模型、训练和评价模块在 `association` 和 `baselines`。原计时合同和历史实验入口由[工具导航](tools/README.md)定位。

原论文、矩阵和逐篇笔记留在独立资料库（本机 `/Users/a1/Projects/biye-ronghe`，不在本GitHub仓内），本仓研究状态提供对应入口。两仓不复制当前计划；数据、真实权重、预测、服务器runs及Git历史不随文档整理删除。
