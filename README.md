# association-code

MTAD 不完整量测下的**补全、关联、融合**研究。目标是关联、融合各形成一个主要创新点，补全形成一个小创新点，并验证完整系统收益；接受成熟机制的组合与海上任务适配。当前仍在建立可信 baseline 和验证候选，尚无完整系统超过 TMTAFF 的证据。

## 当前状态

截至2026-10-21；服务器回执与本地实施分开记录：

| 工作 | 状态 | 结论或下一步 |
| --- | --- | --- |
| Baseline | 035 TAM1000与037圆周TCM4000均完成 | 035最佳关联F1约.5129，仍低于位置控制.6156；037特征恢复优于本空间控制，不能据此判下游成功 |
| 候选补全 | 031/032、034、038及039—041均完成 | 观测底座带来主要改善，学习时变残差尚无整体净收益；位置/航向须对照可见观测，速度MAE与尾部误差分开看 |
| 039—041消融 | **全部完成并复核** | pre-norm恢复动态但位置更差；两层更小且精度近似；去BCE改善速度但未恢复动态，尚无整体补全优势 |
| 关联/融合候选 | **新评分头、效用头及独立训练入口已实现** | 044/045隔离几何与运动/支撑证据，046固定组检验可拒绝的属性效用；CPU更新和完整V零更新检查完成，真实学习增量待服务器实验 |
| 原生运动控制R2及关联衔接 | **本地完整V已完成** | OLS位置MAE8.85→4.74km，关联F1 .6156→.7272；共用保持阈值也改善，但14场F1退化。普通控制，不是已成立的学习贡献 |
| OLS模型接入、R3/R4本地控制 | **代码、CPU检查与完整V控制已完成** | 匿名一次校准F1仅.72717→.72845且错误helper增多；正确组最近观测位置4.74→1.93km，预测组支撑均值4.74→4.39km。学习增量仍待验证，详见[本轮分析](docs/proposals/repair_before_retraining/local-readiness-analysis.md) |

服务器领取、预算和状态只维护在 **[HandOver/Handover.md](HandOver/Handover.md)**。当前仅042零更新诊断READY，尚无领取回执；043补全及044—046关联/融合方案已备齐，均为DRAFT；用户明确批准所选编号后记录授权再领取。022—025继续HOLD，T_test封存。

**接下来按[执行路线](docs/research_brief.md#接下来怎么推进)推进：** 042真实权重检查由服务器执行；本地R2/R3/R4普通控制已完成，OLS已接入候选模型。[新方法规格](docs/specs/evidence-association-fusion.md)已把两项机制落实：几何评分上的运动/支撑证据修正，以及对anchor真值直接监督的属性效用与helper拒绝。下一步由服务器按获批编号实验；匿名迭代校准尚无净收益，暂缓。baseline同步完善，最终进行完整系统验证。

## 从哪里开始

| 文档 | 用途 |
| --- | --- |
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
