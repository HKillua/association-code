# association-code

MTAD 航迹研究的执行代码仓库。研究目标是**在不完整量测下，把补全、关联、融合一起做**：先完成TMTAFF的近似复现，再用同协议的候选预实验判断课题潜力。

010的A0与011的CDF1、A_full、TSA-cTFER均已完成，已有完整性能结果仍是关联预实验。用户已批准并已实现**TMTAFF真实时间适配版**：原生观测时间与下一预测时刻进入TCM，保留原模型参数规模；真实ZIP/参考装配、四阶段训练/checkpoint、全目标融合评价与TSA原生输入适配已接通。013已在真实数据上完成四阶段各2次更新，但在最终评价中超时，没有完整融合结果，也不是充分训练。

`A_full`表示扩大关联训练池，不是完整论文系统；CDF1不是预定的最终创新。新路径拒绝旧scene64实际/名义时刻错位，不读取T_test。时间适配改变了模型函数，明确作为近似参考；见[时间适配规格](docs/specs/tmtaff-time-adaptation.md)、[重构记录](docs/specs/complete-training-refactor.md)及[综合审计](docs/experiment_reaudit_20261018.md)。自有共享补全/融合候选已实现并做CPU/MPS合成核验，未做真实候选训练。013的1小时额度已关闭；012原件已回传，每六小时跟进保持。

## 从这里开始

| 需要了解 | 入口 |
| --- | --- |
| 当前任务、服务器回执、历史轮次 | [HandOver索引](HandOver/Handover.md) |
| 研究目标、已完成与缺项、最近交付 | [研究说明](docs/research_brief.md) |
| 双端协作、修改与执行权限 | [AGENTS](AGENTS.md)、[现场授权](docs/server_autonomy.md) |
| 论文对应、实现差异与数据协议 | [模型说明](docs/model_conformance.md)、[TSA适配](docs/tsactfer_adaptation.md)、[数据契约](docs/data_contract.md) |
| 实际比较结果及证据边界 | [011结果分析](docs/results/compare011_interim/README.md)、[010结果](docs/first_fit_010_result.md) |
| 完整配置与成本 | [013诊断](docs/results/tmtaff013_decisions/analysis.md)、[014已授权独立评价](HandOver/L2S-20261018-014.md)、[完整配置](docs/tmtaff_complete_plan.md)；正式训练预算仍待批 |
| 命令说明与运行环境 | [执行协议](docs/pilot_execution.md)、[GPU执行](docs/gpu_execution.md) |

**COMPARE-011三项均已完成。** A0/TSA/A_full的完整V_select分数已复算；CDF1汇总未显示收益，012已补回七臂逐边缓存、007 manifest和010完整history。013已终态FAILED/124，最终评价运行79.38秒后中断；独立014现获授权单次≤1GPU小时，只加载选中检查点评价，不重训/重校准/重选。具体状态以[交接索引](HandOver/Handover.md)为准。发布不表示服务器已开跑，完整V_select用于开发比较，T_test继续封存。

## 代码导航

| 路径 | 用途 |
| --- | --- |
| `association/data.py`、`dataset.py`、`native_inputs.py` | ZIP、身份标签与既有划分；新原生入口保留真实时间和全部有效片段，旧scene64仅供历史路径 |
| `association/models.py`、`reference.py` | TCM/TAM训练、验证选择、权重恢复与冻结分数缓存 |
| `association/timed_inputs.py`、`timed_models.py`、`timed_dataset.py` | 已批准的原生真实时间适配；短轨真实输入、显式查询时刻、既有划分与全目标参考装配 |
| `association/reference_states.py`、`fusion_targets.py`、`fusion.py` | 训练/开发侧参考状态、显式对齐与正配对监督装配、可微成对TFM；[实现约定](docs/specs/tmtaff-fusion.md) |
| `association/fusion_inference.py`、`fusion_metrics.py`、`fusion_training.py` | 显式分组/递归输出、已对齐轨迹与分组评价、精确归一化的有效batch更新；不自动启动原生训练 |
| `association/complete_training.py`、`complete_evaluation.py`、`full_scene_evaluation.py` | 四阶段训练/恢复，按原adapter恢复选定快照；完整V上的预测/oracle/单源/简单融合与覆盖 |
| `association/context.py`、`heads.py` | 校准、CDF1及机制对照 |
| `association/metrics.py`、`pipeline.py` | 既有关联评价与pilot；后者包含测试阶段，当前不执行 |
| `association/cli.py`、`profile.py` | 数据准备、单参考训练、成本探测和合成检查 |
| `baselines/tsactfer.py`、`tsactfer_data.py`、`tsactfer_native.py` | TSA-cTFER双头与离线适配；新入口直接取原生真实观测，显式128点上限，不经过scene64 |
| `tools/experiments/` | `tmtaff_complete_fit.py`完整训练、`tmtaff_complete_evaluate.py`仅评价已选检查点、`cooperative_fit.py`自有候选；均须遵守各自预算与数据范围；CDF1与TSA工具保留 |
| `configs/`、`tests/`、`tools/pilot/` | 配置、测试、原有环境与ZIP审计工具 |
| `HandOver/` | 每轮请求、回执与小型原始证据 |

## 本地接口与检查

已有兼容依赖后可用 `python -m pip install -e .` 安装；依赖范围见 `pyproject.toml`，服务器版本约束与实际命令以交接为准。

```bash
python -m association.cli --help
python tools/experiments/tmtaff_complete_fit.py --help
python -m unittest discover -s tests
python -m unittest discover -s tools/pilot -p 'test_audit_mtad_level2.py'
```

这些是开发检查，不授权重跑已完成的环境验收、smoke或训练。历史回执保留原貌，旧规则与现行授权冲突时以[AGENTS](AGENTS.md)和用户最新指示为准。
