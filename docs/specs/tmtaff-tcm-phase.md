# TMTAFF 单阶段 TCM 入口

本规格描述执行能力；真实预算以当前交接为准，033已获用户批准。沿用 `tmtaff_complete_fit.py`，新增显式
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
读取实际末端权重；未来阶段使用须另立明确的新 run 方案，本次不实现迁移入口。
只允许同 phase/config 的 `INTERRUPTED` run 按原规则恢复到新目录；失败 020
不得恢复，也不改变其历史。保存和终点评价计入已有时间策略；评价失败不能报完成。

本次只做终点诊断，不加中间钩子或端点选优。未来若比较 020 的 1000 步与新 run，
须披露跨 run 差异和旧端点评价范围；新增 4000 更新预算已获用户批准，按独立033交接执行。

| 可观察要求 | 最小验证 |
| --- | --- |
| 原 TCM 计算完全不变 | 有 dropout 的 native 微型数据，与 full run 相同 TCM 端点权重/损失/RNG 逐项相等 |
| 独立 TCM 真实完成、无下游 | 实际 CLI 读取微型 ZIP、训练和完整 V 诊断；禁止校准/阈值/下游调用 |
| 所有 V 场景及参考缺失保留 | 诊断包含有输入和无输入场景，缺失参考片段保留且零交集为 null |
| 产物可追溯、同阶段可恢复 | 末端 checkpoint 可加载，微型中断恢复与同预算单次执行一致 |
| 明确拒绝混用 | 下游非零、full 的零更新、跨 phase 恢复、tcm 物理预测导出均在启动前拒绝 |

本地测试不读取真实 MTAD、不运行 GPU 训练，不改 README、HandOver 或研究结论。
