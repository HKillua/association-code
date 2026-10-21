# TMTAFF 单阶段 TCM 入口

本规格只定义执行能力，历史033已结束。当前数据为Level1，预算以交接索引为准。沿用`tmtaff_complete_fit.py`，显式
`--phase tcm`；默认 `full` 保持原四阶段流程、训练量、损失、采样和选模语义。

TCM 配置仍完整声明原配置字段，但 `updates.TCM > 0`，其他三个阶段更新数必须
为 0。只支持现有 `native_time_v1` 数据路径；训练与验证人口、标准化、输入投影
冻结、初始化顺序、Adam、梯度裁剪、调度均沿用原实现。无训练时间序列时失败，
不能跳过更新后声称完成。

达到指定 TCM 更新数后，在完整传入的 V_select 上调用现有
`evaluate_tcm_completion`，报告 native 正反向一步误差、自然 query 递推误差、
state-hold 控制及分区域/距离/片段人口，保存 `completion-evaluation.json`。
这是冻结特征空间诊断；支撑内查询沿用原观测插值，不是物理误差或关联收益。
保留无参考片段和无输入场景。该阶段不做 margin 校准、关联阈值选择、TAM/TFM/
JOINT 更新或完整系统评价；`--save-predictions` 不适用于该诊断并明确拒绝。

正常产物保留 `checkpoint.pt`、配置、normalizer、data_contract、权重、优化器/
RNG、更新游标及 coverage，另有 `summary.json` 和真实 `COMPLETED` 终态。
checkpoint 的 `selected` 为 null，只能用 `load_complete_snapshot(selected=False)`
读取实际末端权重；TAM/TFM通过各自显式初始化入口移交TCM，必须建立新run。
只允许同 phase/config 的 `INTERRUPTED` run 按原规则恢复到新目录；失败 020
不得恢复，也不改变其历史。保存和终点评价计入已有时间策略；评价失败不能报完成。

| 可观察要求 | 最小验证 |
| --- | --- |
| 原 TCM 计算完全不变 | 有 dropout 的 native 微型数据，与 full run 相同 TCM 端点权重/损失/RNG 逐项相等 |
| 独立 TCM 真实完成、无下游 | 实际 CLI 读取微型 ZIP、训练和完整 V 诊断；禁止校准/阈值/下游调用 |
| 所有 V 场景及参考缺失保留 | 诊断包含有输入和无输入场景，缺失参考片段保留且零交集为 null |
| 产物可追溯、同阶段可恢复 | 末端 checkpoint 可加载，微型中断恢复与同预算单次执行一致 |
| 明确拒绝混用 | 下游非零、full 的零更新、跨 phase 恢复、tcm 物理预测导出均在启动前拒绝 |

软件测试使用微型合成ZIP，不代表真实MTAD效果。

## 可选有序片段呈现

`training.tcm_sampling`缺省`"uniform_fragment"`，保持原随机抽样与RNG消费。显式`"ordered_fragment"`仅作用于TCM：按传入场景顺序、每场原node顺序遍历至少两个有效原生点的片段，跨场边界顺接；每批最多`batch_tracks`条，epoch尾批不足就保留小批，不回填、不丢弃，下一更新开始下一epoch。这是对论文公开顺序组织的局部对齐；具体尾批约定是本项目选择，不声称作者实现已完全复现。

一个epoch定义为全部eligible片段各呈现一次，不等于逐pair遍历或训练更新数。每步记录实际epoch、epoch内批次、输入池首末索引、轨次和双向相邻transition数`2*sum(n_i-1)`；transition不乘embedding维数，teacher loss的元素分母仍为该数乘`d_model`。既有coverage继续记录累计轨次与唯一覆盖，有序模式补累计transition及已完成epoch数。游标由已完成TCM更新数确定，checkpoint保留采样配置；旧缺字段按原随机模式恢复，恢复中切换sampling被拒绝。

本项不改变TAM正负pair采样、CL或其他阶段缓存，也未实现随机截短。现TCM对完整选点序列做双向全部下一点teacher forcing；随机内部crop会改变上下文并裁去部分监督，不能把它直接说成论文随机截短后的缺失段恢复。80epochs的阶段分配仍未知，不把本项epoch定义自行换算成论文训练量。

050旧Level2曾计划有序3356步，现已请求停止保全，不再READY。采样功能保留；Level1更新数按新有效片段人口计算，不能沿用214739片段或3356步。
