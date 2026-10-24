# association-code

MTAD不完整多源航迹的关联与状态恢复研究。与TMTAFF对齐输入、最终输出及评价任务，允许重设计中间流程；以完整系统误差、目标覆盖和重复输出判断方法价值。**当前尚未成功复现完整TMTAFF，也未证明超过原论文。**

正式数据为 **Level1，2100训练／300开发验证／600封存测试**。[文件、名单与准备统计](configs/data/README.md) · [时间、单位、人口与评价约定](docs/data_contract.md)

日常开发和服务器回传统一使用 **`main`**；本地工作目录为 `/Users/a1/Projects/association-code`。历史研究分支已合入main并退役，实验按提交保留，不再维护多条长期分支。

最新已同步`aaffce5`：093B完成，本地逐场重聚通过，但oracle位置MAE仍约49.64km；等待093A及固定配对表，不能据此判断补全相对价值。夜间控制服务已真实领取，**096B／097A运行，096A等来源；连同093A／094A共四个单卡领取**（最近回执，非实时监控）。[队列交接](HandOver/L2S-20261025-096-097-night.md)新冻结`8b4705d`，当前队列不得重复启动。复核与回执勘误见[研究状态](docs/research_status.md#093b终态复核与夜间队列领取2026-10-25)，实际执行只认[交接索引](HandOver/Handover.md)。默认056＋原linear、T_test封存、066 REJECTED保持。

## 现在看哪里

| 入口 | 用途 |
| --- | --- |
| [当前研究状态](docs/research_status.md) | 已确认的结果、未证明的机制、下一步决策 |
| [交接索引](HandOver/Handover.md) | 唯一当前执行状态；本地AI写代码和决策，服务器AI执行并回传 |
| [当前方法](docs/specs/evidence-association-fusion.md) | 学习关联、属性融合及公平对照 |
| [TMTAFF参考实现](docs/specs/tmtaff-reference.md) | 参考模型实现边界和复现限制 |
| [结果证据](HandOver/results/README.md) | 小型聚合/逐场证据与服务器原件路径 |
| [历史摘要](docs/history/README.md) | 退役实验的结论、负结果与Git恢复位置 |

## 代码与验证

- `association/`：数据、当前关联/融合、TMTAFF参考、评价和训练。
- `baselines/`：TSA-cTFER独立关联对照。
- `tools/experiments/`：保留的[训练与执行入口](tools/experiments/README.md)。
- `tools/analysis/`：缓存复核、归因诊断和071/072同人口比较。
- `configs/`：当前配置和固定数据名单；文件存在不代表可执行授权。
- `tests/`：数据划分、时间/单位/mask、参数更新/冻结、评价人口及小型贯通验证。

```sh
python tools/experiments/evidence_association_fit.py --help
python tools/experiments/evidence_fusion_fit.py --help
python tools/experiments/support_gate_fit.py --help
python tools/experiments/support_gate_linear_fit.py --help
python tools/experiments/support_gate_replication.py --help
python tools/experiments/sequence_position_fit.py --help
python tools/experiments/helper_sequence_fit.py --help
python tools/analysis/compare_helper_sequence.py --help
python tools/analysis/tsactfer_input_probe.py --help
python tools/experiments/ptfm_fit.py --help
python tools/experiments/ptfm_residual_fit.py --help
python tools/analysis/ptfm_speed_diagnostics.py --help
python tools/experiments/tsactfer_level1_fit.py --help
python tools/experiments/tsactfer_full_view_fit.py --help
python tools/experiments/constant_gate_fit.py --help
python tools/analysis/compare_position_benchmarks.py --help
python tools/analysis/compare_residual_shrink.py --help
python tools/analysis/frozen_fusion_transfer.py --help
python tools/analysis/support_gate_diagnostics.py --help
python tools/analysis/tsactfer_score_diagnostics.py --help
python -m unittest discover -s tests
```

本地AI负责代码和决策，服务器AI按交接执行；push不会唤醒服务器。权限、维护和回传规则见[AGENTS](AGENTS.md)及[现场执行规则](docs/server_autonomy.md)。

073已完成并复核：同4000更新、两个seed下Transformer的MAE均优于门控（590.744／606.624m对646.750／635.776m），满足预定晋级条件，采用其为后续位置恢复底座；RMSE和部分尾部仍有代价，门控保留为对照。固定056分组、97.21%覆盖与610重复；速度／航向统一Mean。领域PTFM／TSA及有效完整TMTAFF仍缺，T_test封存。074D TSA适配已完成，当前结果严重失效，尚未补齐可信强关联基线；074C已完成：位置1845.38m未达竞争水平，速度／航向有待进一步归因的改善，见[074C审查](docs/research_status.md#074c结果审查2026-10-24)；074A/B均完成：B四项改善，A以MAE/P95/P99退化换取伤害下降，未通过两seed晋级条件，保留原线性Transformer。详见[074D结果审查](docs/research_status.md#074d结果审查2026-10-24)，不将适配pilot失败宣称为击败原论文。详见[研究决策](docs/research_status.md#073审查与后续方向2026-10-23)与[交接](HandOver/Handover.md)。

[075只读诊断](HandOver/S2L-20261024-075.md)已完成，待深入科研复核。[076A/B](HandOver/L2S-20261024-076.md)均已完成并审查。A速度平均水平误差改善而目标内动态较差；B位置残差适配为有效负实验：MAE1609.060m虽优于074C的1845.384m，却在全部300场、38组件劣于自身固定.95初值721.174m。关闭该配方，不加步追榜；不据此否定原作者PTFM，见[076B审查](docs/research_status.md#076b结果审查2026-10-24)。

[074双seed深挖](docs/research_status.md#074ab两seed深挖与去留2026-10-24)确认累计伤害与有效改善同时减少，净收益方向跨seed翻转；A的RMSE净改善对3035敏感。[077缓存诊断](HandOver/S2L-20261024-077.md)已完成并审查：固定减半MAE640.247／624.993m，两seed均未超过原linear；A与harm互有取舍、B仅全局H/I由harm占优，不据此证明惩罚必要，见[综合审查](docs/research_status.md#076a077结果审查2026-10-24)。076A／076B／077均已结案，不重领。

078A/B/C和080全部完成并复核：[078/080审查与下一步](docs/research_status.md#078080完成复核与081082决定2026-10-24)。078C仅比原linear低1.60m，场景等权和部分尾部反向，未晋级；相对初值累计伤害下降有复验线索。080不支持“T强V弱”，类先验与候选组成需分别解释。下一轮081只补pre/weighted两臂匹配seed，082只改TSA训练视图crop→full；不为第四空卡追加任务。实现及实际领取状态见[交接索引](HandOver/Handover.md)，push不代表已启动。

081A/B与整表已完成并复核：weighted相对pre在第二seed的MAE增加16.381m、RMSE增加30.641m、H增加31.23%，四项预定出口均失败；关闭当前配方、不追加第三seed。082当前TSA适配亦暂停。**该轮已结案，不重领；默认仍为056关联＋原线性Transformer，新批次以顶部为准。** 见[081收尾](docs/research_status.md#081双seed收尾与阶段决定2026-10-24)；强领域baseline缺口与T_test封存状态保持。
