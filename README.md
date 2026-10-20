# association-code

MTAD 不完整量测下的**补全、关联、融合**研究。目标是关联、融合各形成一个主要创新点，补全形成一个小创新点，并验证完整系统收益；接受成熟机制的组合与海上任务适配。当前仍在建立可信 baseline 和验证候选，尚无完整系统超过 TMTAFF 的证据。

## 当前状态

截至2026-10-21；服务器回执与本地实施分开记录：

| 工作 | 状态 | 结论或下一步 |
| --- | --- | --- |
| Baseline | **042已完成并复核，047最近回执RUNNING** | 035训练距离被dropout噪声主导，关闭后确定性表示仍近零；047固定033补全、从头检验TAM无dropout训练，待终态 |
| 候选补全 | 031/032、034、038及039—041均完成；043最近回执RUNNING | 观测底座带来主要改善，学习时变残差尚无整体净收益；043检验OLS底座上的学习增量，位置/航向和速度分别评价 |
| 039—041消融 | **全部完成并复核** | pre-norm恢复动态但位置更差；两层更小且精度近似；去BCE改善速度但未恢复动态，尚无整体补全优势 |
| 关联候选 | **044/045完成并独立复算** | 045宏F1 .83884，相对OLS .72717净少764条错边、错误helper少33.33%；044虽F1 .79229却增错边。跨源误报仍待解决 |
| 融合候选 | **046完成并独立复算，保护条件未通过** | 对本轮支撑均值位置MAE4391.52→3846.96m、速度改善；航向、9002、长缺失及速度RMSE退步。抗污染有价值，正确helper利用仍弱 |
| 原生运动控制R2及关联衔接 | **本地完整V已完成** | OLS位置MAE8.85→4.74km，关联F1 .6156→.7272；共用保持阈值也改善，但14场F1退化。普通控制，不是已成立的学习贡献 |
| OLS模型接入、R3/R4本地控制 | **代码、CPU检查与完整V控制已完成** | 匿名一次校准F1仅.72717→.72845且错误helper增多；正确组最近观测位置4.74→1.93km，预测组支撑均值4.74→4.39km。学习增量仍待验证，详见[本轮分析](docs/proposals/repair_before_retraining/local-readiness-analysis.md) |

服务器领取、预算和状态只维护在 **[HandOver/Handover.md](HandOver/Handover.md)**。用户已在服务器端批准043—047并由服务器领取；044—046已完成并审计，043/047按最近回执在途，不重复启动。022—025继续HOLD，T_test封存。

**本轮结论与下一步见[044—046分析](docs/proposals/repair_before_retraining/044-046-analysis.md)。** 建议先冻结045/046权重，验证“OLS/045分组 × 普通/046融合”的零训练组合收益；尚未执行或新增任务。按[最新研究路线](docs/research_brief.md#接下来怎么推进)并行准备baseline独立TFM，不把TAM达到理想F1作为其前置；候选先修融合短板再做贡献消融，并补共同的唯一目标/覆盖系统评价。043补全、047 baseline修复仍各自审计，当前未发布新任务。

## 从哪里开始

| 文档 | 用途 |
| --- | --- |
| [一天学习清单](docs/reports/MTAD学习与研究计划清单.md) | 按数据、baseline、关键实验、指标和候选想法掌握课题，含阅读顺序、练习与自查 |
| [研究说明](docs/research_brief.md) | 目标、三项候选贡献、baseline复现程度与论文比较边界 |
| [数据契约](docs/data_contract.md) | 划分、原生时间、单位、mask、全人口和指标 |
| [论文协议对照](docs/mtad_data_protocols.md) | TMTAFF与其他MTAD方法的数据选择和任务差异 |
| [当前结果与分析](docs/proposals/repair_before_retraining/README.md) | 补全排查、对照结果、复算脚本和保留证据 |
| [候选方法](docs/proposals/candidate_revision_after018/README.md) | 当前候选结构、表示和训练阶段 |
| [参考实现规格](docs/specs/complete-training-refactor.md) | TMTAFF适配参考的四阶段训练、选择和恢复；各模块规格从此进入 |
| [历史摘要](docs/history/README.md) | 已结束实验、负结果、旧方案与Git恢复方法 |
| [协作约定](AGENTS.md)、[服务器权限](docs/server_autonomy.md) | 本地研究/代码/分析，服务器执行/监控 |

## 代码与运行

- `association/`：数据读取、模型、训练、关联与物理评价。
- `baselines/`：TSA-cTFER独立关联适配。
- `tools/experiments/`：[训练与评价入口](tools/experiments/README.md)；`tools/analysis/`：预测比较。
- `configs/`：当前方法及对照配置；配置存在不代表获准开跑。
- `tests/`：数据、模型、训练、评价及原生ZIP端到端检查。
- `HandOver/`：当前交接、必要原始回执与直接比较所需的小型结果。数据和权重留服务器。

```bash
python tools/experiments/tmtaff_complete_fit.py --help
python tools/experiments/tmtaff_complete_evaluate.py --help
python tools/experiments/cooperative_fit.py --help
python tools/experiments/tsactfer_fit.py --help
python tools/experiments/evidence_association_fit.py --help
python tools/experiments/evidence_fusion_fit.py --help
python -m unittest discover -s tests
```

整理不改变代码、配置、实验协议或在途任务。结案过程材料总结后从工作树移除，原件由Git追溯，不另建archive副本。
