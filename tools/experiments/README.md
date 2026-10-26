# 运行入口

仅从[当前交接](../../HandOver/Handover.md)领取。当前102完整论文指标CPU补算READY；101与100/096已结案，099暂停，其余代码或配置存在不表示新授权。

| 工作 | 入口 |
|---|---|
| 102已有090C完整论文指标 | `../analysis/coordinate_metrics.py --paper-attributes --prediction-speed-unit kn --target-speed-unit mps`；原final缓存＋056计数，0训练/前向/GPU；无此开关保留101坐标模式 |
| 数据准备 | `make_scene_manifest.py`、`prepare_scene_data.py`；现有名单冻结，不重复生成 |
| 学习关联／属性融合／完整系统 | `evidence_association_fit.py`、`evidence_fusion_fit.py`、`level1_system_evaluate.py` |
| 位置网络与预算控制 | `sequence_position_fit.py`、`frozen_position_adapter_fit.py`、`sequence_budget_replication.py` |
| 100原生事件机会筛查 | `native_query_fit.py`、`../analysis/compare_native_query.py`；共享090C底座，T0/T1/T2固定预算，真实执行看交接 |
| TMTAFF阶段与独立评价 | `tmtaff_complete_fit.py`、`tmtaff_complete_evaluate.py`、`tmtaff_downstream_fit.py`、`completion_value_fit.py`、`completion_system_evaluate.py` |
| 096原有限队列 | `post088_overnight_queue.py`；已有服务持有接续权，不重复提交 |
| DeepAF真实适配准备 | `deepaf_peer_fit.py`、`../analysis/compare_deepaf_peer.py`；099 HOLD |
| PTFM／TSA参考 | `ptfm_fit.py`、`tsactfer_level1_fit.py`／`tsactfer_full_view_fit.py`；适配负结果不代表作者方法无效 |
| 服务器单次托管 | `launch_detached.py`；真实退出记录、墙时仅提醒 |

`tools/analysis`只保留当前完整系统／预算／同行比较及它们使用的缓存读取工具。共同来源绑定已移入`association/evaluation/position_sources.py`和`relation_sources.py`，不再为了几个函数保留整套旧实验CLI。

门控、constant、helper变体、098训练/专用比较及PTFM旧损失变体入口已退役。仍有历史命名的模型/loader用于既有真实权重和共享输入，不恢复为训练计划；`tsactfer_fit.py`保留Level1所用公共设置。全部旧源码可从[历史版本](../../docs/history/README.md)单文件读取。

默认普通控制仅Mean；完整对比由合同明确`anchor mean_supported attribute_hybrid`及真实权重终点。所有比较保留完整目标、mask、缺预测与重复。分卷按`--archive PART1 PART2 PART3`传入，数据路径见[数据清单](../../configs/data/README.md)。
