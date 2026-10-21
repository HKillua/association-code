# association-code

MTAD 不完整量测下的**补全、关联、融合**研究。目标是关联、融合各形成一个主要创新点，补全形成一个小创新点，并验证完整系统收益；接受成熟机制的组合与海上任务适配。当前仍在建立可信 baseline 和验证候选，尚无完整系统超过 TMTAFF 的证据。

## 当前状态

截至2026-10-21；服务器回执与本地实施分开记录：

| 工作 | 状态 | 结论或下一步 |
| --- | --- | --- |
| Baseline融合 | **049完成300步并复算，输出坍塌** | 358个多片段目标的位置全部变为同一常量，恰两片段也如此；共同17062点位置MAE63.79km vs普通融合1.80km。优先定位条件信息在哪层丢失，不先加步或JOINT |
| Baseline关联 | **047完成并复核，未改善关联** | 无dropout新TAM最佳F1 .454858/AUC .675962，低于035 .512865/.753154；误报减少但漏报增加，不能只凭训练距离分离宣布修好 |
| 候选补全 | **043完成并复核，学习增量未过关** | 模型位置4815.48m vs自身OLS4737.76m，支撑内退化；RMSE略好、9002略有收益，不能统一清零残差。动态修正仍近片段常量 |
| 039—041消融 | **全部完成并复核** | pre-norm恢复动态但位置更差；两层更小且精度近似；去BCE改善速度但未恢复动态，尚无整体补全优势 |
| 关联候选 | **044/045完成并独立复算** | 045宏F1 .83884，相对OLS .72717净少764条错边、错误helper少33.33%；044虽F1 .79229却增错边。跨源误报仍待解决 |
| 融合候选 | **046有条件收益；051符号辅助监督未改善** | 051对046位置MAE3846.96→3848.60m，速度/航向也略差；局部路由改变但没有净收益，暂不晋升为默认配置。046航向、9002和长缺失保护仍未通过 |
| 冻结连接048 | **完成并复核，关联收益已传到下游** | 045换组使普通融合位置4391.52→3751.35m；新组学习融合3695.92m但航向/尾部仍差。一组一输出的共同人口三属性MAE改善，覆盖与保护项仍须分开看 |
| 原生运动控制R2及关联衔接 | **本地完整V已完成** | OLS位置MAE8.85→4.74km，关联F1 .6156→.7272；共用保持阈值也改善，但14场F1退化。普通控制，不是已成立的学习贡献 |
| OLS模型接入、R3/R4本地控制 | **代码、CPU检查与完整V控制已完成** | 匿名一次校准F1仅.72717→.72845且错误helper增多；正确组最近观测位置4.74→1.93km，预测组支撑均值4.74→4.39km。学习增量仍待验证，详见[本轮分析](docs/proposals/repair_before_retraining/local-readiness-analysis.md) |

服务器领取、预算和状态只维护在 **[HandOver/Handover.md](HandOver/Handover.md)**。043—047均已完成并审计，不重复启动或加步。022—025继续HOLD，T_test封存。

正式数据已按 **2100训练/300验证/600测试** 固定，2400训练/验证场景完成时钟核验和人口准备，600测试仍封存；训练/验证分别214,739/23,743有效片段。新协议允许跨场共享A，历史已使用场景不进测试，旧512/64继续承担开发。详见[正式数据约定](docs/data_contract.md#2026-10-21正式场景级协议)。

050已在独立checkout启动：新2100场景从头有序TCM3356步，完整遍历214739片段一遍，终点评价V300；最新回执仍为RUNNING，尚无终态指标。保留原目标，不因049/051负结果中途换配置。CL尺度诊断选项已实现但未运行，补全受约束运动修正仍待实现。

**最新分析：[049融合坍塌](docs/proposals/repair_before_retraining/049-analysis.md)、[051符号监督](docs/proposals/repair_before_retraining/051-analysis.md)、[研究说明](docs/research_brief.md)。** 已拉取至`86e0932`，049三档预测包已补齐并复算；049/051完成，050运行中，当前没有未领取READY。本轮只分析保存结果与代码，没有新增真实模型训练、前向或测试评价。

## 从哪里开始

| 文档 | 用途 |
| --- | --- |
| [一天学习清单](docs/reports/MTAD学习与研究计划清单.md) | 按数据、baseline、关键实验、指标和候选想法掌握课题，含阅读顺序、练习与自查 |
| [研究说明](docs/research_brief.md) | 目标、三项候选贡献、baseline复现程度与论文比较边界 |
| [数据契约](docs/data_contract.md) | 划分、原生时间、单位、mask、全人口和指标 |
| [论文协议对照](docs/mtad_data_protocols.md) | TMTAFF与其他MTAD方法的数据选择和任务差异 |
| [当前结果与分析](docs/proposals/repair_before_retraining/README.md) | 补全排查、对照结果、复算脚本和保留证据 |
| [候选补全与早期v2](docs/proposals/candidate_revision_after018/README.md) | 043补全及早期cooperative结构；其中旧身份/融合四臂不是045/046 |
| [证据关联与效用融合](docs/specs/evidence-association-fusion.md) | 044—046独立方法，冻结普通OLS；尚未接入043，045与046的连接已由048评价 |
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
python tools/experiments/evidence_fusion_evaluate.py --help
python -m unittest discover -s tests
```

整理不改变代码、配置、实验协议或在途任务。结案过程材料总结后从工作树移除，原件由Git追溯，不另建archive副本。
