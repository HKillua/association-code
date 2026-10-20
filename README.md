# association-code

MTAD 不完整量测下的**补全、关联、融合**研究。目标是关联、融合各形成一个主要创新点，补全形成一个小创新点，并验证完整系统收益；接受成熟机制的组合与海上任务适配。当前仍在建立可信 baseline 和验证候选，尚无完整系统超过 TMTAFF 的证据。

## 当前状态

截至2026-10-20已收到的回执：

| 工作 | 状态 | 结论或下一步 |
| --- | --- | --- |
| Baseline | 033 TCM4000已完成；035 TAM1000、037圆周TCM4000在途 | 增训改善特征补全，仍有670/1635片段差于保持；TFM/联合训练及完整物理终评尚未完成 |
| 候选补全 | 031/032、034、038已完成并分析 | log速度输出降低速度MAE；034航向56.6293°仍差于观测控制55.9361°；038定位到decoder内部时间变化幅度持续衰减，具体原因待验证 |
| 039—041消融 | **已批准，READY** | 分别只改decoder pre-norm、decoder两层、关闭teacher身份BCE；各1000步＋完整V＋成功后一次原四场CPU动态检查 |
| 关联/融合创新 | 方案与部分原型阶段 | 共享偏差与对应判断相互修正、按时间/属性预测融合效用，尚缺可靠增益证据 |

服务器领取、预算和状态只维护在 **[HandOver/Handover.md](HandOver/Handover.md)**。039—041等待035/037释放名额，最多两个单卡作业；各4小时仅提醒，失败不自动重试。022—025继续HOLD，T_test封存。

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
python -m unittest discover -s tests
```

整理不改变代码、配置、实验协议或在途任务。结案过程材料总结后从工作树移除，原件由Git追溯，不另建archive副本。
