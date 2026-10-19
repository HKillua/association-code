# association-code

MTAD不完整量测下的**补全、关联、融合**研究。用户2026-10-20明确：**主要创新优先在融合或关联，一个核心机制即可，再验证完整系统收益；补全作为可靠支撑，可有小创新。** 认可既有机制组合与任务适配，具体贡献靠对照和消融说明。补全→关联→融合用于分阶段核验，不要求三阶段各自原创；完整TMTAFF适配参考仍须建立可信比较。目标与候选主线见[研究说明](docs/research_brief.md)。

**首要目标是推进实验，弄清结果。** 不以代码量、测试数或文档量代替研究进展；必要正确性检查通过即推进已批实验，不反复增加框架与准备轮次。长期约定见[AGENTS](AGENTS.md)。

## 当前状态

- **020参考已保全后停止**：保存TCM1000＋TAM300权重，关联开发macro-F1 **.435672**；TAM301—400日志保留但参数未落盘，TFM/JOINT与物理终评未做。此次停止来自用户改变研究顺序，不是模型自然故障。
- **021身份原分数已复核**：teacher1000＋utility200完成，teacher端点开发macro-F1 **.630321**、P .301445、R .835305。比同人口位置控制仅高1.47个百分点，跨源F1没有优势；不能归因于utility或融合，T_test未开启。
- **030残差取证已完成并复核**：固定4场景86片段，位置残差几乎为片段固定偏移，速度预测集中在41.86–42.34。支撑内9001由1.26米退化至241.62米，9002从4489.32米小幅改善至4449.93米；不能统一清零残差。损失计算未发现分母错误，具体训练根因未证实；022—025仍HOLD。[030复核](docs/proposals/repair_before_retraining/README.md#030逐点残差与损失复核2026-10-19)、[交接索引](HandOver/Handover.md)。
- **baseline须以论文结果校准**：已补充`paper_metrics`及逐场计数，原指标、阈值选择不变。最新核对020 TCM1000步仅覆盖37,862/54,792条训练片段（69.10%）；未发现next-time/逆序/mask确定错误，训练量不足与递推目标不匹配仍需区分。低分不能用于宣称超过TMTAFF。[协议差异](docs/research_brief.md#baseline与论文结果的校准2026-10-19)、[本地续审](docs/proposals/repair_before_retraining/030-repair-analysis.md#7-本地baseline续审与结果准备2026-10-20)。
- **030进一步分析已有具体改法**：原量测速度/航向控制优于当前teacher；9002留身份跨源校准oracle位置误差4488.71→722.79米，证明共享偏移有可利用信息，但真值配对收益不是算法成绩。片段平均运动外推整体有效、逐场不稳定；baseline特征MSE主要对航向敏感。推荐先做速度输出坐标单因素修正，再分别验证观测底座、场景校准与动态补全。[数据、方案与复算脚本](docs/proposals/repair_before_retraining/030-repair-analysis.md)。
- **第一项速度修改已实现，真实收益待验证**：新增可选`scaled_log`输出和`--phase teacher`完整V独立补全评价，旧配置保留原行为；相同采样/初始化、teacher端点一致性及小型CLI流程已检查。[031/032方案](docs/proposals/repair_before_retraining/030-repair-analysis.md#6-已实现的第一项与ab2026-10-20)为两组各1000步、仅改速度公式；用户已回复“继续”，独立READY见[交接索引](HandOver/Handover.md)，不自动恢复020/022—025。
- 两组返回后可用`tools/analysis/compare_completion_predictions.py`直接比较完整人口的速度、位置、航向及分区域结果；本地工具已做小型端到端检查，真实A/B结论尚未产生。
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
