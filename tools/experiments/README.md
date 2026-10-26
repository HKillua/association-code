# 运行入口

**只按[当前交接](../../HandOver/Handover.md)执行。** 下表是代码职责索引，不是待运行清单。

| 职责 | 入口 |
|---|---|
| 已有预测的论文指标（102使用） | `../analysis/coordinate_metrics.py`，具体参数只看L2S |
| 数据准备 | `make_scene_manifest.py`、`prepare_scene_data.py`；已有冻结名单不重复生成 |
| 学习关联／融合／系统 | `evidence_association_fit.py`、`evidence_fusion_fit.py`、`level1_system_evaluate.py` |
| 位置恢复与既有端点加载 | `sequence_position_fit.py`、`frozen_position_adapter_fit.py`、`sequence_budget_replication.py` |
| 原生事件实现 | `native_query_fit.py`；100配方已关闭，保留在用共享实现，不是新增训练安排 |
| TMTAFF阶段与评价 | `tmtaff_complete_fit.py`、`tmtaff_complete_evaluate.py`、`tmtaff_downstream_fit.py`、`completion_value_fit.py`、`completion_system_evaluate.py` |
| 文献适配 | `deepaf_peer_fit.py`、`ptfm_fit.py`、`tsactfer_level1_fit.py`、`tsactfer_full_view_fit.py`；实现存在不表示可信复现已成立 |
| 一次性服务托管 | `launch_detached.py`；脱离网络继续运行，真实退出落盘，墙时仅建议 |

历史命名的loader/共享输入仍有真实权重或生产调用时保留；结案专用队列、比较CLI与其测试已退役。旧源码按[历史入口](../../docs/history/README.md)从Git读单文件，不恢复旧任务。

数据分卷与路径见[数据清单](../../configs/data/README.md)。默认普通控制及完整比较人口由具体合同决定，不用不同人口拼成绩。
