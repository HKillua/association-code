# association-code

MTAD不完整量测下的**补全、关联、融合**研究。用户2026-10-20最新明确：**关联、融合各形成一个创新点，补全形成一个小创新点，再验证完整系统收益。** 以有实证价值的增量创新为标准，认可既有机制组合与任务适配，不要求颠覆性突破。各环节贡献分别通过对照和消融说明；完整TMTAFF适配参考仍须建立可信比较。目标与候选分工见[研究说明](docs/research_brief.md)。

**首要目标是推进实验，弄清结果。** 不以代码量、测试数或文档量代替研究进展；必要正确性检查通过即推进已批实验，不反复增加框架与准备轮次。长期约定见[AGENTS](AGENTS.md)。

## 当前状态

**本轮工作汇报：** [复现程度、论文差距、补齐问题与并行安排](docs/reports/2026-10-20-research-progress.md)。baseline仍为部分复现，候选尚无可靠端到端增益。[034候选已完成并复核](docs/proposals/repair_before_retraining/034-analysis.md)：航向89.9213°→56.6293°，但观测控制55.9361°更好，消去学习部分的时间变化几乎不影响误差。035已于北京时间12:23在隔离目录启动，冻结033独立训练TAM1000步；尚未收到终态成绩。

**最新038结果：** [逐层分析已复核](docs/proposals/repair_before_retraining/038-analysis.md)。冻结034、四场86片段，全部片段每一层的时间变化幅度均下降；decoder输入RMS均值.528352到第6层.000584，逐轨缩小倍数中位1059。变化在输出头前已被压低，具体attention/归一化/优化根因尚未确定。下一单因素建议只改decoder为pre-norm，保持034其余设置；尚未修改生产模型或新增训练。038正常完成，不重跑。

**Baseline继续并行：** [037圆周TCM](HandOver/S2L-20261020-037.md)已于北京时间14:02:43在隔离目录、GPU1启动，一次4000步及完整V，无自动TAM；035仍在途，当前无新增READY。新旧特征MSE不可直接排名，下游收益另议。[036](docs/proposals/repair_before_retraining/036-analysis.md)短程反馈未显示为主要故障；候选038现象不能直接套用到baseline。历次证据见[综合排查](docs/proposals/repair_before_retraining/tcm-diagnosis.md)。

**下一步三组消融已实现，具体预算待确认：** [039—041规格](docs/specs/cooperative-decoder-ablations.md)分别只改decoder pre-norm、只将decoder降为两层、只关闭teacher身份BCE，均对照已完成034，从头1000步及完整V，完成后各一次原四场CPU动态读取。三组配置和独立DRAFT交接已备齐，尚不可领取；累计粗估7—8个单卡小时，各4h仅提醒、失败不重试，最多两个单卡。微型训练验证了零物理头初始时BCE先影响共享encoder的可能路径，但未证明它导致034失效；动态幅度和真实误差分开判断。

- **020参考已保全后停止**：保存TCM1000＋TAM300权重，关联开发macro-F1 **.435672**；TAM301—400日志保留但参数未落盘，TFM/JOINT与物理终评未做。此次停止来自用户改变研究顺序，不是模型自然故障。
- **021身份原分数已复核**：teacher1000＋utility200完成，teacher端点开发macro-F1 **.630321**、P .301445、R .835305。比同人口位置控制仅高1.47个百分点，跨源F1没有优势；不能归因于utility或融合，T_test未开启。
- **030残差取证已完成并复核**：固定4场景86片段，位置残差几乎为片段固定偏移，速度预测集中在41.86–42.34。支撑内9001由1.26米退化至241.62米，9002从4489.32米小幅改善至4449.93米；不能统一清零残差。损失计算未发现分母错误，具体训练根因未证实；022—025仍HOLD。[030复核](docs/proposals/repair_before_retraining/README.md#030逐点残差与损失复核2026-10-19)、[交接索引](HandOver/Handover.md)。
- **baseline须以论文结果校准**：已补充`paper_metrics`及逐场计数，原指标、阈值选择不变。最新核对020 TCM1000步仅覆盖37,862/54,792条训练片段（69.10%）；未发现next-time/逆序/mask确定错误，训练量不足与递推目标不匹配仍需区分。低分不能用于宣称超过TMTAFF。[协议差异](docs/research_brief.md#baseline与论文结果的校准2026-10-19)、[本地续审](docs/proposals/repair_before_retraining/030-repair-analysis.md#7-本地baseline续审与结果准备2026-10-20)。
- **030进一步分析已有具体改法**：原量测速度/航向控制优于当前teacher；9002留身份跨源校准oracle位置误差4488.71→722.79米，证明共享偏移有可利用信息，但真值配对收益不是算法成绩。片段平均运动外推整体有效、逐场不稳定；baseline特征MSE主要对航向敏感。推荐先做速度输出坐标单因素修正，再分别验证观测底座、场景校准与动态补全。[数据、方案与复算脚本](docs/proposals/repair_before_retraining/030-repair-analysis.md)。
- **031/032已完成并深挖**：各1000步、完整V64。log速度输出使MAE从29.4878降至19.7287（−33.1%），但航向70.3884°→89.9213°、支撑内位置2105.70→2185.77米；片段内部速度仍近常量。不能判为补全通过或恢复四臂。全V原量测控制显示来源/支撑差异，9002支撑内速度反而是神经B更好；不统一复制观测。[完整分析与下一方案](docs/proposals/repair_before_retraining/031-032-analysis.md)。
- **033 baseline TCM已完成并复核**：4000步、实耗2.30小时；与029同四场相比，一步特征MSE下降22.74%、自然query下降16.08%，四场均改善。完整V单报，尚无新关联/物理融合结果。保留该端点，不自动恢复020/022—025。[033分析](docs/proposals/repair_before_retraining/033-analysis.md)、[交接索引](HandOver/Handover.md)。
- **034属于候选，不是baseline**：1000步及完整V已完成；观测底座避免航向绝对预测失效，但网络仍近似每轨固定修正。038已证实decoder内时变幅度持续衰减，下一改进以完整V误差和观测控制净收益验收，不凭activation变大宣布修复。[034分析](docs/proposals/repair_before_retraining/034-analysis.md)、[038分析](docs/proposals/repair_before_retraining/038-analysis.md)。未知对应下场景平移控制与固定分组效用融合仍可独立设计。
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
