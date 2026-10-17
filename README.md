# association-code

MTAD 航迹研究的执行代码仓库。研究目标是**在不完整量测下，把补全、关联、融合一起做**：先完成TMTAFF的近似复现，再用同协议的候选预实验判断课题潜力。

当前只完成两项真实关联短训，**TFM融合仍未实现**。`A_full`表示扩大关联训练池，不是完整论文系统；CDF1也不是预定的最终创新。模型、配置、执行入口和必要协议在本仓库维护，论文原件与研究笔记另存。

## 从这里开始

| 需要了解 | 入口 |
| --- | --- |
| 当前任务、服务器回执、历史轮次 | [HandOver索引](HandOver/Handover.md) |
| 研究目标、已完成与缺项、最近交付 | [研究说明](docs/research_brief.md) |
| 双端协作、修改与执行权限 | [AGENTS](AGENTS.md)、[现场授权](docs/server_autonomy.md) |
| 论文对应、实现差异与数据协议 | [模型说明](docs/model_conformance.md)、[TSA适配](docs/tsactfer_adaptation.md)、[数据契约](docs/data_contract.md) |
| 实际比较结果及证据边界 | [011阶段分析](docs/results/compare011_interim/README.md)、[010结果](docs/first_fit_010_result.md) |
| 命令说明与运行环境 | [执行协议](docs/pilot_execution.md)、[GPU执行](docs/gpu_execution.md) |

**最近回执状态：COMPARE-011仍为RUNNING；TSA已完成，CDF1/A_full尚无终态回传。** 执行与预算以索引指向的独立交接为准，README不是运行任务。011收尾期间，本地在隔离分支补齐融合链路；当前没有新的融合训练授权。完整V_select用于开发比较，T_test继续封存。

## 代码导航

| 路径 | 用途 |
| --- | --- |
| `association/data.py`、`dataset.py` | 原生Level2 ZIP、A–B身份标签、共享A组件隔离、异常记录与时间配准 |
| `association/models.py`、`reference.py` | TCM/TAM训练、验证选择、权重恢复与冻结分数缓存 |
| `association/context.py`、`heads.py` | 校准、CDF1及机制对照 |
| `association/metrics.py`、`pipeline.py` | 评价与完整pilot；后者包含测试阶段，不属于当前011任务 |
| `association/cli.py`、`profile.py` | 数据准备、单参考训练、成本探测和合成检查 |
| `baselines/tsactfer.py`、`tsactfer_data.py` | TSA-cTFER双头模型及离线数据适配 |
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
