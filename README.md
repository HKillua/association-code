# association-code

MTAD不完整多源航迹的关联与状态恢复研究。与TMTAFF对齐输入、最终输出及评价任务，允许重设计中间流程；以完整系统误差、目标覆盖和重复输出判断方法价值。**当前尚未成功复现完整TMTAFF，也未证明超过原论文。**

正式数据为 **Level1，2100训练／300开发验证／600封存测试**。[文件、名单与准备统计](configs/data/README.md) · [时间、单位、人口与评价约定](docs/data_contract.md)

日常开发和服务器回传统一使用 **`main`**；本地工作目录为 `/Users/a1/Projects/association-code`。历史研究分支已合入main并退役，实验按提交保留，不再维护多条长期分支。

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
python tools/experiments/ptfm_fit.py --help
python tools/experiments/tsactfer_level1_fit.py --help
python tools/experiments/constant_gate_fit.py --help
python tools/analysis/compare_position_benchmarks.py --help
python tools/analysis/frozen_fusion_transfer.py --help
python tools/analysis/support_gate_diagnostics.py --help
python -m unittest discover -s tests
```

本地AI负责代码和决策，服务器AI按交接执行；push不会唤醒服务器。权限、维护和回传规则见[AGENTS](AGENTS.md)及[现场执行规则](docs/server_autonomy.md)。

073已完成并复核：同4000更新、两个seed下Transformer的MAE均优于门控（590.744／606.624m对646.750／635.776m），满足预定晋级条件，采用其为后续位置恢复底座；RMSE和部分尾部仍有代价，门控保留为对照。固定056分组、97.21%覆盖与610重复；速度／航向统一Mean。领域PTFM／TSA及有效完整TMTAFF仍缺，T_test封存。074A/B/C/D均已发布READY：两seed相对初值退化惩罚＋PTFM四状态融合适配＋TSA原生时间关联适配，可按资源独立并行。代码就绪不等于已取得领域基线成绩。详见[研究决策](docs/research_status.md#073审查与后续方向2026-10-23)与[交接](HandOver/Handover.md)。
