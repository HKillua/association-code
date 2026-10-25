# 运行与分析入口

只从[当前交接](../../HandOver/Handover.md)领取。模型/配置/CLI存在不表示需要运行，历史入口不恢复旧授权。

| 范围 | 入口 |
|---|---|
| 数据 | `make_scene_manifest.py`、`prepare_scene_data.py`；当前名单已冻结，不重复生成 |
| 学习关联/属性融合 | `evidence_association_fit.py`、`evidence_fusion_fit.py`、`level1_system_evaluate.py` |
| 位置恢复与预算控制 | `sequence_position_fit.py`、`frozen_position_adapter_fit.py`、`sequence_budget_replication.py` |
| 098已结案 | `query_relation_fit.py`、`../analysis/compare_query_relation.py`、`../analysis/summarize_query_relation.py`；只保复核能力，不续训 |
| TMTAFF参考及在途096 | `tmtaff_complete_fit.py`／`tmtaff_complete_evaluate.py`、`completion_value_fit.py`、`completion_system_evaluate.py` |
| 096原有限队列 | `post088_overnight_queue.py`；已有服务持有接续权，不重复领取 |
| DeepAF待真实适配 | `deepaf_peer_fit.py`、`../analysis/compare_deepaf_peer.py`；099 HOLD，未形成可信同行成绩 |
| PTFM/TSA参考 | `ptfm_fit.py`及位置适配入口、`tsactfer_level1_fit.py`／`tsactfer_full_view_fit.py`；已有负结果，不冒称作者复现 |
| 服务托管 | `launch_detached.py`；一次性服务、真实退出记录、墙时advisory |
| 共享比较 | `../analysis/compare_association_duration.py`、`compare_position_benchmarks.py`、`compare_position_harm.py`、`compare_frozen_position_adapter.py`、`compare_physical_predictions.py` |

旧名的`frozen_fusion_transfer`、`learned_group_oracle`及部分比较器仍供当前输入来源与缓存核验，不能按编号删除。其他门控/helper入口也保留被当前模块引用的实现；它们不是新任务清单。

已结案一次性审计和DeepAF自建模拟训练入口已删除，必要数值/梯度/评价测试保留。[历史与恢复](../../docs/history/README.md)

默认普通控制仅Mean；完整系统对照应按合同明确`anchor mean_supported attribute_hybrid`及实际关联终点，不用selected替代final。所有比较保完整目标、mask、覆盖和重复，不能删难例或混排不同人口。分卷按`--archive PART1 PART2 PART3`传入，数据路径见[数据清单](../../configs/data/README.md)。
