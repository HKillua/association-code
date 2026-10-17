# association-code

MTAD 航迹研究的执行代码仓库。研究目标是**在不完整量测下，把补全、关联、融合一起做**：先完成TMTAFF的近似复现，再用同协议的候选预实验判断课题潜力。

010的A0与011的CDF1、A_full、TSA-cTFER均已完成，真实结果仍是关联预实验。用户已授权开始重构：原生真实时间输入、TCM因果一致性修复、完整四阶段训练/checkpoint、全目标融合评价与TSA原生输入适配已实现。**原生观测到TCM共同输出时间轴的适配尚未确定，完整原生三模块实验尚未启动**；代码内核与CPU合成集成通过不等于真实实验完成。

`A_full`表示扩大关联训练池，不是完整论文系统；CDF1不是预定的最终创新。新路径拒绝旧scene64实际/名义时刻错位，不读取T_test。见[本轮规格与边界](docs/specs/complete-training-refactor.md)及[综合审计](docs/experiment_reaudit_20261018.md)。012既有文件回传和每六小时跟进继续；新增GPU实验仍须先给完整配置与成本。

## 从这里开始

| 需要了解 | 入口 |
| --- | --- |
| 当前任务、服务器回执、历史轮次 | [HandOver索引](HandOver/Handover.md) |
| 研究目标、已完成与缺项、最近交付 | [研究说明](docs/research_brief.md) |
| 双端协作、修改与执行权限 | [AGENTS](AGENTS.md)、[现场授权](docs/server_autonomy.md) |
| 论文对应、实现差异与数据协议 | [模型说明](docs/model_conformance.md)、[TSA适配](docs/tsactfer_adaptation.md)、[数据契约](docs/data_contract.md) |
| 实际比较结果及证据边界 | [011结果分析](docs/results/compare011_interim/README.md)、[010结果](docs/first_fit_010_result.md) |
| 后续完整配置与成本 | [设计草案](docs/tmtaff_complete_plan.md)，尚非可执行计划或GPU授权 |
| 命令说明与运行环境 | [执行协议](docs/pilot_execution.md)、[GPU执行](docs/gpu_execution.md) |

**COMPARE-011三项均已完成。** A0/TSA/A_full的完整V_select分数已复算；CDF1汇总未显示收益，七臂逐边缓存仍待补回。012只复制既有CDF缓存、007 manifest和010完整history，不训练、不重新推理、不使用GPU。用户先看完整配置与成本再决定预算，当前没有新增GPU授权；具体任务以[交接索引](HandOver/Handover.md)为准。完整V_select用于开发比较，T_test继续封存。

## 代码导航

| 路径 | 用途 |
| --- | --- |
| `association/data.py`、`dataset.py`、`native_inputs.py` | ZIP、身份标签与既有划分；新原生入口保留真实时间和全部有效片段，旧scene64仅供历史路径 |
| `association/models.py`、`reference.py` | TCM/TAM训练、验证选择、权重恢复与冻结分数缓存 |
| `association/reference_states.py`、`fusion_targets.py`、`fusion.py` | 训练/开发侧参考状态、显式对齐与正配对监督装配、可微成对TFM；[实现约定](docs/specs/tmtaff-fusion.md) |
| `association/fusion_inference.py`、`fusion_metrics.py`、`fusion_training.py` | 显式分组/递归输出、已对齐轨迹与分组评价、精确归一化的有效batch更新；不自动启动原生训练 |
| `association/complete_training.py`、`complete_evaluation.py`、`full_scene_evaluation.py` | 显式已配准数据的四阶段训练/恢复，完整V上的预测/oracle/单源/简单融合与覆盖；不是原生fit CLI |
| `association/context.py`、`heads.py` | 校准、CDF1及机制对照 |
| `association/metrics.py`、`pipeline.py` | 既有关联评价与pilot；后者包含测试阶段，当前不执行 |
| `association/cli.py`、`profile.py` | 数据准备、单参考训练、成本探测和合成检查 |
| `baselines/tsactfer.py`、`tsactfer_data.py`、`tsactfer_native.py` | TSA-cTFER双头与离线适配；新入口直接取原生真实观测，显式128点上限，不经过scene64 |
| `tools/experiments/` | CDF1验证、错误迁移与缓存导出、TSA有界首训 |
| `configs/`、`tests/`、`tools/pilot/` | 配置、测试、原有环境与ZIP审计工具 |
| `HandOver/` | 每轮请求、回执与小型原始证据 |

## 本地接口与检查

已有兼容依赖后可用 `python -m pip install -e .` 安装；依赖范围见 `pyproject.toml`，服务器版本约束与实际命令以交接为准。

```bash
python -m association.cli --help
python -m unittest discover -s tests
python -m unittest discover -s tools/pilot -p 'test_audit_mtad_level2.py'
```

这些是开发检查，不授权重跑已完成的环境验收、smoke或训练。历史回执保留原貌，旧规则与现行授权冲突时以[AGENTS](AGENTS.md)和用户最新指示为准。
