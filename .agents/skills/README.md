# 项目常用技能

这里维护离线工程使用的 8 项技能及其完整引用。当前研究和执行范围分别见[研究状态](../../offline_observation/docs/research_status.md)和[交接](../../offline_observation/HandOver/Handover.md)。

| 技能 | 用途与入口 |
|---|---|
| research-workflow | [研究总控](research-workflow/SKILL.md)：按现有材料选择阶段，衔接六技能及知识库；保留原五技能完整契约 |
| analyze-paper | [来源分析](analyze-paper/SKILL.md)：独立读懂论文、公式、证据、限制与适用条件 |
| deep-read | [矩阵分析](deep-read/SKILL.md)：本项目按 Mode C 构建问题—现象—机制及证据关系 |
| discover-innovation | [形成 Idea](discover-innovation/SKILL.md)：从连接点发展可检验方案、竞争解释与实验设计 |
| transfer-ml-methods | [跨领域迁移](transfer-ml-methods/SKILL.md)：核对目标问题、源机制、生效条件及迁移归因 |
| record-experiment | [实验记录](record-experiment/SKILL.md)：如实记录输入、命令、结果、成本、异常和计划偏差 |
| squeeze-experiment | [结果深挖](squeeze-experiment/SKILL.md)：分析异质性、失败、代价与可支持结论，回流矩阵 |
| spec-driven-coding | [实现规范](spec-driven-coding/SKILL.md)：从行为规格、仓库证据和必要验证推进代码修改 |

先按[离线 AGENTS](../../offline_observation/AGENTS.md)和[完整科研流程](../../offline_observation/docs/research_workflow.md)定位阶段，再读对应技能及需要的引用。用户未指定版本时，以本目录为准。

可以在提示中指定 `$research-workflow`、`$deep-read` 或 `$record-experiment`。离线工程和离线资料库的原技能路径都链接到本目录。

2026-10-29 纳入时保留 8 项技能的全部 16 个源文件，正文与源副本逐字节一致，没有缩写替换。`research-workflow` 和 `spec-driven-coding` 来自当前使用的 `~/.codex/skills/`，其余六项来自 `~/.agents/skills/`；其中五个原始科研技能及总控所带原契约已与用户提供的原件核对一致。随包保留 `references/` 与 `agents/openai.yaml`，不依赖本机绝对路径或个人配置。所选源目录未附 LICENSE/NOTICE，本次不另造许可声明。

后续更新应核对源版本、必要引用和项目约定，作为普通 Git 变更审阅；不自动跟随全局副本变化，也不在此复制当前研究计划或服务器队列。
