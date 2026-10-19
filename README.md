# association-code

MTAD不完整量测下的**补全、关联、融合**研究。当前按**补全→关联→融合**逐阶段核验；完整TMTAFF时间/输出尺度适配近似参考与自有候选在同阶段比较。

**首要目标是推进实验，弄清结果。** 不以代码量、测试数或文档量代替研究进展；必要正确性检查通过即推进已批实验，不反复增加框架与准备轮次。长期约定见[AGENTS](AGENTS.md)。

## 当前状态

- **020参考已保全后停止**：保存TCM1000＋TAM300权重，关联开发macro-F1 **.435672**；TAM301—400日志保留但参数未落盘，TFM/JOINT与物理终评未做。此次停止来自用户改变研究顺序，不是模型自然故障。
- **021身份原分数已复核**：teacher1000＋utility200完成，teacher端点开发macro-F1 **.630321**、P .301445、R .835305。比同人口位置控制仅高1.47个百分点，跨源F1没有优势；不能归因于utility或融合，T_test未开启。
- **029补全检查已完成，尚无稳定改善证据**：固定4场景86片段，候选支撑内位置MAE2047.43米，对底座1927.10米，四场均退化；参考一步特征有收益，但查询结果跨场景不稳。两端权重保留、没有继续训练；022—025仍HOLD，当前无新READY。[复核与原因边界](docs/proposals/repair_before_retraining/README.md#029及021原分数复核2026-10-19)、[研究说明](docs/research_brief.md)、[交接索引](HandOver/Handover.md)。
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
| `tools/experiments/` | [当前训练与评价入口](tools/experiments/README.md) |
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
