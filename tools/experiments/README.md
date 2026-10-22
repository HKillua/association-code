# 当前运行入口

先读[交接索引](../../HandOver/Handover.md)。正式数据为Level1，历史任务已结案；配置和命令存在不表示授权。

| 入口 | 用途 |
| --- | --- |
| `evidence_association_fit.py` | OLS证据关联训练；固定终点、普通/配对困难采样和实际曝光审计 |
| `evidence_fusion_fit.py` | 属性utility融合训练；`check`为合成检查，Level1真实fit绑定当前OLS工作点 |
| `support_gate_fit.py` | 068：3035／2535输入预核→T_core受限位置门控1000更新→固定056分组CPU初始化及终点评价；一条失败即停的链 |
| `support_gate_linear_fit.py` | 070：只改线性位置损失，逐批配对068，复用原CPU初始化，只做新final1000完整V300评价；按当前READY交接执行 |
| `level1_system_evaluate.py` | 普通/学习关联与普通/学习融合的完整系统评价；保存全部目标、分组、mask和路由 |
| `level1_ols_working_point.py` | Level1纯OLS开发工作点，不加载学习权重 |
| `tmtaff_complete_fit.py` / `tmtaff_complete_evaluate.py` | TMTAFF参考各阶段训练和冻结评价，保留完整baseline实现 |
| `tsactfer_fit.py` | TSA-cTFER独立关联对照的离线适配 |
| `make_scene_manifest.py` / `prepare_scene_data.py` | 数据名单生成和时间/人口准备；当前名单已冻结，不重复生成 |
| `launch_detached.py` | 服务器服务管理器托管单次作业，保存真实退出状态 |
| `../analysis/frozen_fusion_transfer.py` | 当前062：056固定组与三个057冻结融合的七臂比较 |
| `../analysis/compare_association_duration.py` | 保留共同人口、代表变化、增/失覆盖的跨轮缓存比较；亦提供062源重放函数 |
| `../analysis/learned_group_oracle.py` | 有限候选Oracle诊断及062来源检查；061已结案，不恢复执行额度 |
| `../analysis/compare_physical_predictions.py` | 通用物理预测比较 |
| `../analysis/support_gate_diagnostics.py` | 069：只读068两端点、训练日志及两场预检，重算log目标／曝光／固定reference段贡献，无模型前向 |
| `../setup/gpu_allocation.py` | 已获批任务的现场资源选择，不终止他人进程 |

所有CLI支持`--help`。分卷输入使用`--archive PART1 PART2 PART3`，顺序固定，无需合并或全面解压。[数据文件与固定人口](../../configs/data/README.md)

系统评价默认普通控制只包含Mean。需要完整对照时显式指定`--ordinary-controls anchor mean_supported attribute_hybrid`；学习关联显式使用`--association-endpoint final`，提供原run、checkpoint、scores、summary和data-audit，避免用selected替代预定终点。062有独立冻结入口及[具体交接](../../HandOver/L2S-20261022-062.md)。

模型、监督、配对采样、输入与指标定义集中在[方法规格](../../docs/specs/evidence-association-fusion.md)；参考模型见[TMTAFF实现](../../docs/specs/tmtaff-reference.md)。旧Cooperative候选、已结案一次性分析及专用测试已退役，历史原件见[摘要](../../docs/history/README.md)。不得按真实身份删难例；normalizer只拟合训练，T_test封存。
