# association-code

MTAD不完整多源航迹研究。与TMTAFF对齐输入、最终输出与评价任务，允许重设计中间流程；优先关联/融合贡献，补全可作为辅助能力融入状态重建，不强制三个独立网络。目标是形成可信完整对照、有实证增量的系统和可完成的硕士论文。接受成熟机制的任务适配；当前尚未复现完整TMTAFF或证明系统超过它。

## 当前工作

用户最新要求提高推进效率：baseline集中修复与完整候选并行，按“可信对照、完整输出、去留结论”交付；不再把逐个修好参考三模块当作研究串行前提。具体优先级见[目标校准与近期交付](docs/research_brief.md#目标校准与近期交付)。

**用户2026-10-21最新决定：正式实验改用数据集1（Level1）。** 数据集2结果只作历史开发证据。050已停止保全：3356步训练完成，V300终评中断于154场，运行FAILED。Level1三分卷和049快照已在服务器确认可用；无需重新传输。当前没有新训练READY。2100/300/600名单和数据路径见[数据入口](configs/data/README.md)。

053A/B及054A均已完成。Level1普通链有9,099个可评位置目标、覆盖5,797个；3,302个未覆盖身份都有输入，却没有本身份anchor输出。固定同组同点，mean相对anchor位置MAE增加698.63m，速度/航向分别改善9.59原生单位/7.06°；不能将关联损失与融合污染直接视为两个独立根因。下一步[055A](docs/reports/2026-10-21-055a-design.md)只新增一种原组内部一致性拆分，评价覆盖、重复输出及同分组双融合。代码和CPU一次执行交接见[055交接](HandOver/L2S-20261021-055.md)；未取得055真实结果，不新增训练。

A完整系统、B参考TFM机制、C一个有条件的融合候选并行；[053B分析](docs/reports/2026-10-21-054-mechanism-note.md)确认训练后条件响应衰减，但未证明单一根因。融合研究不以学习关联F1先胜OLS为前提；最终贡献由同任务系统误差、覆盖与消融决定。本轮没有新训练额度。

本轮Level1适配：关联/融合候选使用全部合法观测构造普通运动证据，与同输入OLS公平对照；融合阈值绑定当前Level1的纯OLS验证工作点。补全/参考编码器仍保留cap128、全场Q64与真实秒，不将数值时间单位20秒误当采样周期。具体实现、未解决问题和验收见[方法规格](docs/specs/evidence-association-fusion.md)。

| 环节 | 当前已知 |
| --- | --- |
| 补全 | 043学习输出位置4815.48m，比自身OLS4737.76m差；学习增量未过关 |
| 关联 | 045在旧开发集F1 .83884 vs OLS .72717，净少764错边；048确认收益能传到下游 |
| 候选融合 | 046有部分抗污染收益，航向和长缺失保护不足；051符号监督未改善 |
| 参考baseline | TCM有特征收益、TAM仍弱；049的358个多片段目标输出同一位置，须定位信息在哪层丢失 |

以上全是Level2历史结果，不能当Level1成绩。目标与下一步见[研究说明](docs/research_brief.md)，详细证据见[近期分析](docs/proposals/repair_before_retraining/README.md)。

## 目录

| 目录 | 内容 |
| --- | --- |
| `association/` | 原生数据、模型、训练和评价 |
| `baselines/` | TSA-cTFER独立关联对照 |
| `tools/experiments/` | [训练、评价和托管入口](tools/experiments/README.md) |
| `tools/analysis/` | 当前预测比较与049/051复算 |
| `configs/` | 当前模型参数；配置存在不表示获准训练 |
| `tests/` | 数据/时间/mask/指标、更新恢复和小型端到端检查 |
| `HandOver/` | [当前交接](HandOver/Handover.md)、050回执和近期必要结果 |
| `docs/` | 数据与方法约定、近期分析、[统一历史摘要](docs/history/README.md) |

本地AI负责研究、代码和结果分析；服务器AI负责执行、环境和运行回执，通过HandOver联系。push不唤醒服务器，本地不SSH值守。规则见[AGENTS.md](AGENTS.md)和[现场授权](docs/server_autonomy.md)。

```sh
python tools/experiments/tmtaff_complete_fit.py --help
python tools/experiments/cooperative_fit.py --help
python tools/experiments/evidence_association_fit.py --help
python tools/experiments/evidence_fusion_fit.py --help
python -m unittest discover -s tests
```

历史原件由Git追溯，当前工作树只保留决策需要的材料；数据和权重留服务器。运行核实际配置、划分、数组和退出状态，不生成文件/代码/缓存校验摘要作为门槛。
