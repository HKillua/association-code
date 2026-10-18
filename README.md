# association-code

MTAD航迹研究的执行仓库。目标是在**不完整量测下完成补全、关联、融合**，以TMTAFF时间适配近似参考和自有候选的真实比较，判断课题潜力。

完整参考与当前候选均已实现。**018已完成600更新/四臂完整评价，2371秒，退出0；[本地复核](docs/results/cooperative018_local_review/README.md)确认物理误差较短训下降，但关联总体仍弱、基础恢复误差大，效用尚无稳定专属收益。** 017仍独立运行：最新原件TCM100+TAM100+TFM60，关联选中macro-F1 .432985，尚无完整物理终评。用户要求三个阶段都精进，对应[新研究方案](docs/proposals/tri_stage_support/README.md)已结合数据复审具体化为[本地实现规格](docs/proposals/candidate_revision_after018/README.md)，用户已批准并完成本地实现与CPU验证（见[实现记录](docs/proposals/candidate_revision_after018/implementation.md)），尚无新GPU授权/READY，不改017或挪用018额度。实际入口见[交接索引](HandOver/Handover.md)。

**CDF1本轮无专属机制收益，活动训练实现已退役；负结果、原缓存、历史交接与Git历史保留。** TSA保留为独立关联诊断对照，不充当完整融合baseline，也不自动启动新训练。旧A0、A_full和TSA关联分数不与新时间适配实验混排；`A_full`不代表完整TMTAFF。见[研究说明](docs/research_brief.md)与[退役及后续路线](docs/decisions/retire_cdf1_and_focus.md)。

## 从这里开始

10月19日新增[外部MTAD公开处理对照](docs/results/external_mtad_processing20261019/README.md)：核了公开代码与论文时间任务。整场景Q64是我们的适配选择，缺少作者确切窗口依据；尚未证明读错原始数据，未据此改协议或开训。

10月19日[数据与训练任务定点复审](docs/results/dataset_followup018/README.md)：查明速度字段/官方单位冲突、长时外推的任务条件及候选训练尺度风险；未发现需要立刻修改017的装配错误。分析保留原始标签和全部人口，没有新增实验。

| 需要了解 | 入口 |
| --- | --- |
| 当前任务与服务器回执 | [HandOver索引](HandOver/Handover.md) |
| 研究目标、方法角色与下一步 | [研究说明](docs/research_brief.md)、[CDF1退役与路线](docs/decisions/retire_cdf1_and_focus.md) |
| 双端协作与现场权限 | [AGENTS](AGENTS.md)、[现场授权](docs/server_autonomy.md)、[GPU说明](docs/gpu_execution.md) |
| 原生时间、数据与参考监督 | [时间适配规格](docs/specs/tmtaff-time-adaptation.md)、[数据契约](docs/data_contract.md)、[综合审计](docs/experiment_reaudit_20261018.md) |
| 完整参考与候选实现 | [参考重构](docs/specs/complete-training-refactor.md)、[候选规格](docs/specs/cooperative-completion-fusion.md)、[候选核验](docs/results/cooperative_local_20261018/README.md) |
| 既有实测与成本 | [012完整复算](docs/results/evidence012_local_review/README.md)、[013诊断](docs/results/tmtaff013_decisions/analysis.md)、[后续成本方案](docs/proposals/tmtaff_after013/README.md) |
| 当前配置、覆盖与费用提案 | [完整任务比较方案](docs/proposals/first_full_comparison/README.md)：015测量已完成并关闭；[并行配置与成本](docs/proposals/parallel_after015/README.md)：017运行中，018已完成并关闭 |
| 当前执行入口 | [实验工具](tools/experiments/README.md)、[旧pilot说明](docs/pilot_execution.md) |

014只评价短训参考，015是候选两步成本测量，均不代表充分训练。完整参考与候选的正式训练量、选模配置和总预算仍待确定；旧轮次额度关闭，T_test封存。TMTAFF时间适配改变了模型函数，明确称为近似参考，不能声称作者精确实现或把测试通过当作研究有效。

## 代码导航

| 路径 | 用途 |
| --- | --- |
| `association/data.py`、`dataset.py`、`native_inputs.py` | 原包读取、身份隔离与既有划分；旧scene64只服务历史关联路径 |
| `association/timed_inputs.py`、`timed_models.py`、`timed_dataset.py` | 已批准的原生真实时间输入、短轨保留和显式参考装配 |
| `association/models.py`、`reference.py`、`staged.py` | TCM/TAM共享模型与既有关联训练、验证和缓存 |
| `association/reference_states.py`、`fusion_targets.py`、`fusion.py`、`fusion_training.py` | 对齐参考、正配对监督、物理TFM及有效batch更新 |
| `association/fusion_inference.py`、`fusion_metrics.py`、`full_scene_evaluation.py` | 分组与递归融合、物理误差、全目标及覆盖评价 |
| `association/complete_training.py`、`complete_evaluation.py` | 完整参考四阶段训练/恢复与预测、真实分组、单源、简单融合评价 |
| `association/cooperative.py`、`cooperative_training.py`、`cooperative_evaluation.py` | 共享补全/融合候选与固定身份的四臂机制比较 |
| `association/prediction_artifacts.py`、`tools/analysis/compare_physical_predictions.py` | 可选保存实际物理预测；离线比较完整人口覆盖与共同有效点误差，保留改善/恶化案例索引 |
| `association/metrics.py`、`runtime.py`、`sampling.py` | 共享指标、设备/产物处理与采样；保留历史结果只读复算能力 |
| `association/cli.py`、`profile.py` | 保留prepare、train-reference、profile、probe-training；CDF run-pilot/smoke已移除 |
| `baselines/tsactfer*.py` | TSA-cTFER离线关联适配与原生输入支持 |
| `tools/experiments/` | 完整参考训练、仅评价、候选训练与TSA关联入口 |
| `configs/`、`tests/`、`tools/pilot/`、`HandOver/` | 配置、测试、原生数据审计和独立轮次证据 |

## 本地接口与检查

安装项目依赖后可从仓库根查看接口并运行CPU测试：

```bash
python -m association.cli --help
python tools/experiments/tmtaff_complete_fit.py --help
python tools/experiments/tmtaff_complete_evaluate.py --help
python tools/experiments/cooperative_fit.py --help
python -m unittest discover -s tests
python -m unittest discover -s tools/pilot -p 'test_audit_mtad_level2.py'
```

这些命令不授权真实训练或重跑已完成任务。历史交接保留原貌，旧预算、SHA准入或空闲整卡限制与现行授权冲突时，以用户最新指示和[AGENTS](AGENTS.md)为准。
