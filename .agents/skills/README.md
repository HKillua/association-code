# 项目常用技能

这里保存本项目实际使用的 8 项技能，技能正文及必要引用完整纳入 Git，GitHub 上可直接阅读，克隆仓库后也可使用。当前任务仍只读[研究状态](../../docs/research_status.md)和[交接](../../HandOver/Handover.md)。

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

先按[AGENTS](../../AGENTS.md)及[完整科研流程](../../docs/research_workflow.md)定位当前阶段，再读取对应技能及其实际需要的引用；不因技能已安装就认为相应研究步骤已经完成。用户最新要求优先，项目约定对通用技能作具体约束，包括创新价值判断、证据边界、实验授权及数据协议。用户未指定其他版本时，优先读取本目录，同名全局副本不混读。

可在提示中明确使用 `$research-workflow`，也可按阶段指定 `$deep-read`、`$record-experiment` 等。[Codex 官方说明](https://learn.chatgpt.com/docs/build-skills#where-codex-loads-local-skills)将仓库内 `.agents/skills/` 列为本地技能发现目录；从本仓或其子目录打开的会话可发现这些技能。当前资料库 `biye-ronghe` 的 `.agents/skills` 通过相对符号链接引用本目录，维护时只改主仓这一份。

2026-10-29 纳入时保留 8 项技能的全部 16 个源文件，正文与源副本逐字节一致，没有缩写替换。`research-workflow` 和 `spec-driven-coding` 来自当前使用的 `~/.codex/skills/`，其余六项来自 `~/.agents/skills/`；其中五个原始科研技能及总控所带原契约已与用户提供的原件核对一致。随包保留 `references/` 与 `agents/openai.yaml`，不依赖本机绝对路径或个人配置。所选源目录未附 LICENSE/NOTICE，本次不另造许可声明。

后续更新应核对源版本、必要引用和项目约定，作为普通 Git 变更审阅；不自动跟随全局副本变化，也不在此复制当前研究计划或服务器队列。
