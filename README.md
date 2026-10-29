# MTAD航迹关联与状态恢复

唯一代码仓。当前目标和结论只读[研究状态](docs/research_status.md)，服务器只领取[交接索引](HandOver/Handover.md)里的任务。

| 需要什么 | 唯一入口 |
|---|---|
| 当前目标、已知结果、下一步 | [研究状态](docs/research_status.md) |
| 当前执行安排 | [HandOver](HandOver/Handover.md) |
| 协作与维护规则 | [AGENTS](AGENTS.md) |
| 科研方法与完整五技能契约 | [科研流程](docs/research_workflow.md) |
| 项目常用技能、完整正文与调用 | [8 项技能](.agents/skills/README.md) |
| 数据、单位与指标 | [数据约定](docs/data_contract.md)、[论文指标](docs/mtad_data_protocols.md) |
| 在用实现与命令 | [方法说明](docs/specs/evidence-association-fusion.md)、[运行入口](tools/experiments/README.md) |
| 结案结论与旧文件恢复 | [历史摘要](docs/history/README.md) |

代码按 `association/{data,models,training,evaluation,utils}` 分工；文献方法在 `baselines`，命令在 `tools`，配置在 `configs`，必要检查在 `tests`。存在源码或配置不等于待执行任务。

原论文与逐篇笔记留在[资料库](../biye-ronghe/README.md)，不在两仓复制当前计划。数据、真实权重、预测、服务器runs及Git历史不随文档整理删除。
