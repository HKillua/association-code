# association-code

MTAD不完整量测下的**补全、关联、融合**研究。完整TMTAFF时间/输出尺度适配近似参考与自有候选并行验证，目标是得到可信的实验比较。

**首要目标是推进实验，弄清结果。** 不以代码量、测试数或文档量代替研究进展；必要正确性检查通过即推进已批实验，不反复增加框架与准备轮次。长期约定见[AGENTS](AGENTS.md)。

## 当前状态

- **017已FAILED**：外部SIGTERM于约5小时27分中断，未到预算上限；完整融合终评未开始。原件已pull并独立复算，不能用不同batch首末loss宣称位置变差。见[017分析与修复](docs/proposals/repair_before_retraining/README.md)。
- **018候选已完成**：600更新、2371秒；基础恢复及关联仍弱，效用无稳定专属收益。见[实测分析](https://github.com/HKillua/association-code/blob/612afea/docs/results/cooperative018_local_review/README.md)。
- **候选修正版已完成本地实现**：观测底座＋位置残差、log速度/圆周航向、支撑补全、局部关联与四臂属性残差融合；[当前方法与配置](docs/proposals/candidate_revision_after018/README.md)。已知修复与共享/单臂独立入口均已实现。用户同意按原上限拆分：[当前执行队列](HandOver/Handover.md)先核服务托管，再并行参考与共享训练，随后四臂独立运行。
- CDF已退役，TSA保留独立关联对照。旧过程与负结果见[历史摘要](docs/history/README.md)，机器证据和旧交接在Git历史中追溯。21分钟窗口未采用，当前先验证已实现修正版。

## 代码与资料

| 目录 | 内容 |
| --- | --- |
| `association/data/` | 原包读取、划分、原生时间输入、参考监督与候选输入 |
| `association/models/` | TMTAFF、真实时间适配、物理融合与候选网络 |
| `association/training/` | 两套完整训练、校准、采样和共享优化 |
| `association/evaluation/` | 关联/物理指标、分组、人口覆盖、输入支撑和预测导出 |
| `association/utils/` | 配置与运行产物处理 |
| `baselines/` | TSA-cTFER独立关联适配 |
| `tools/experiments/` | [四个实验入口](tools/experiments/README.md) |
| `tests/` | 按数据、模型、训练、评价、集成分组的科研正确性检查 |
| `configs/`、`HandOver/` | 当前实验配置、任务交接和待审回执 |

数据和方法从[数据契约](docs/data_contract.md)、[参考时间适配](docs/specs/tmtaff-time-adaptation.md)、[候选实现](docs/proposals/candidate_revision_after018/README.md)开始。当前结论与下一步见[研究说明](docs/research_brief.md)。清理前旧入口可在Git历史中追溯，不继续维护废弃执行路径。

## 使用

```bash
python tools/experiments/tmtaff_complete_fit.py --help
python tools/experiments/tmtaff_complete_evaluate.py --help
python tools/experiments/cooperative_fit.py --help
python tools/experiments/tsactfer_fit.py --help
python -m unittest discover -s tests
```

正式实验按[交接索引](HandOver/Handover.md)中的READY及已批预算运行。Git历史中的旧配置和回执用于追溯，不是重跑授权；T_test继续封存。
