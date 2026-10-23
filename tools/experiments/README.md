# 当前运行入口

先读[交接索引](../../HandOver/Handover.md)。正式数据为Level1，历史任务已结案；配置和命令存在不表示授权。

| 入口 | 用途 |
| --- | --- |
| `evidence_association_fit.py` | OLS证据关联训练；固定终点、普通/配对困难采样和实际曝光审计 |
| `evidence_fusion_fit.py` | 属性utility融合训练；`check`为合成检查，Level1真实fit绑定当前OLS工作点 |
| `support_gate_fit.py` | 068：3035／2535输入预核→T_core受限位置门控1000更新→固定056分组CPU初始化及终点评价；一条失败即停的链 |
| `support_gate_linear_fit.py` | 已完成070：只改线性位置损失，逐批配对068，复用原CPU初始化；保留复现入口，不重领 |
| `support_gate_replication.py` | 071A/B：独立启动新seed Log／线性1000更新及CPU两端点评价；compare仅在两臂完成后核实际配对 |
| `sequence_position_fit.py` | LSTM／Transformer二维位置恢复；extended4000为072，harm4000／harm-replica4000为074A/B，必须绑定同seed完整旧控制；CPU两端点 |
| `helper_sequence_fit.py` | 078A/B/C：三种helper集合表示，同seed／原072C公共初值和4000实际批次配对，CPU两端点全人口评价 |
| `../analysis/compare_helper_sequence.py` | 078单臂或三臂缓存比较，严格配对、H/I/尾部/组件和helper数量分层，无前向 |
| `../analysis/tsactfer_input_probe.py` | 079冻结074D，在固定32个T场比较裁窗/完整×同/跨场负例；零训练，非全候选成绩 |
| `ptfm_fit.py` | 074C：双流PTFM四状态任务适配，T-only归一化、与072C实际批次配对；CPU两端点及真实缓存对照 |
| `ptfm_residual_fit.py` | 076B：只改PTFM位置为固定.95物理初值＋零初始化残差；配对074C全部初参例外与实际批次，并保留072C曝光比较 |
| `../analysis/ptfm_speed_diagnostics.py` | 076A：只读074C及056缓存；固定零速控制、原生分布、逐目标动态与相邻有效查询变化，不加载权重或前向 |
| `tsactfer_level1_fit.py` | 074D：TSA原生时间128点Level1适配，双头固定final评分、全候选及论文资格子集、六臂系统与056共同点 |
| `../analysis/tsactfer_population_probe.py` | 080：只读074D/079缓存，同规则T/V平衡及V完整候选对照、固定先验AP、margin均值分解；零GPU/前向/权重加载 |
| `constant_gate_fit.py` | 072A：原071B实际T样本上的1001点全局标量搜索，固定后一次完整CPU V评价 |
| `../analysis/compare_position_benchmarks.py` | 071整表；显式三项072输入后核常数与4000更新终点、真实采样前缀及同人口；不做模型前向 |
| `../analysis/tsactfer_score_diagnostics.py` | 075：只读074D固定final评分及成功训练日志；两头排序／原阈值错误／时间资格与同场负例曝光，不加载权重或重新选阈值 |
| `../analysis/compare_position_harm.py` | 074A/B分别核同seed旧Transformer缓存、实际训练配对、累计改善／伤害与尾部，不做模型前向 |
| `../analysis/compare_residual_shrink.py` | 077：两seed既有CPU缓存上固定残差减半，复现旧配对指标、全组球面重放、I/H及残差几何；无权重加载、前向或系数搜索 |
| `level1_system_evaluate.py` | 普通/学习关联与普通/学习融合的完整系统评价；保存全部目标、分组、mask和路由 |
| `level1_ols_working_point.py` | Level1纯OLS开发工作点，不加载学习权重 |
| `tmtaff_complete_fit.py` / `tmtaff_complete_evaluate.py` | TMTAFF参考各阶段训练和冻结评价，保留完整baseline实现 |
| `tsactfer_fit.py` | 历史TSA离线入口，不作为074D的Level1固定终点入口 |
| `make_scene_manifest.py` / `prepare_scene_data.py` | 数据名单生成和时间/人口准备；当前名单已冻结，不重复生成 |
| `launch_detached.py` | 服务器服务管理器托管单次作业，保存真实退出状态 |
| `../analysis/frozen_fusion_transfer.py` | 062—066：056固定组与三个057冻结融合的七臂比较；旧验收失败，不恢复额度 |
| `../analysis/compare_association_duration.py` | 保留共同人口、代表变化、增/失覆盖的跨轮缓存比较；亦提供062源重放函数 |
| `../analysis/learned_group_oracle.py` | 有限候选Oracle诊断及062来源检查；061已结案，不恢复执行额度 |
| `../analysis/compare_physical_predictions.py` | 通用物理预测比较 |
| `../analysis/support_gate_diagnostics.py` | 069：只读068两端点、训练日志及两场预检，重算log目标／曝光／固定reference段贡献，无模型前向 |
| `../setup/gpu_allocation.py` | 已获批任务的现场资源选择，不终止他人进程 |

所有CLI支持`--help`。分卷输入使用`--archive PART1 PART2 PART3`，顺序固定，无需合并或全面解压。[数据文件与固定人口](../../configs/data/README.md)

系统评价默认普通控制只包含Mean。需要完整对照时显式指定`--ordinary-controls anchor mean_supported attribute_hybrid`；学习关联显式使用`--association-endpoint final`，提供原run、checkpoint、scores、summary和data-audit，避免用selected替代预定终点。062有独立冻结入口及[具体交接](../../HandOver/L2S-20261022-062.md)。

模型、监督、配对采样、输入与指标定义集中在[方法规格](../../docs/specs/evidence-association-fusion.md)；参考模型见[TMTAFF实现](../../docs/specs/tmtaff-reference.md)。旧Cooperative候选、已结案一次性分析及专用测试已退役，历史原件见[摘要](../../docs/history/README.md)。不得按真实身份删难例；normalizer只拟合训练，T_test封存。
