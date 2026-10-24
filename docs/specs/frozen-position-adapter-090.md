# 090：强模型输出保持与缺失区修正

## 问题与固定假设

087B 相对强 072C 的净 MAE 退化主要落在 anchor 已有支撑区；083/089 又表明删除正确 helper 会破坏可用信息。090 保留完整观测与强 072C final4000，只检验：**保持已有支撑区的已学输出，把新增修正限制到 anchor 无支撑区，是否比全域新增修正及相同追加预算的全模型微调更有效？** 残差 adapter 是成熟机制的任务适配，不声称发明新结构。support 只是观测首末时间范围，不代表无噪声、准确或安全。

072C 来源为 `sequence-position-transformer-extended-level1.json`，seed 20261023、T_core 2100、V_select 300、105154 参数、4000 更新。056 固定关联系统以运行参数绑定，具体路径见 L2S。旧 066 REJECTED，T_test 不打开。

## 已完成的零训练否证

来源：`HandOver/results/LEVEL1-HELPER-TEMPORAL-20261024-087-summary/summary.json.gz` 的 `position_weightings`，完整覆盖 442263 个有效目标位置点。三个互斥区域点数：inside 322953，anchor-outside/helper-supported 90769，group-no-support 28541。把 inside 全部替换成物理 fixed .95，另两个区域保留原方法输出，可由各区 MAE/MSE 及点数精确重聚（只受已有 JSON 浮点精度限制）：

| 保留 outside 的方法 | 原全局 MAE m | 替换后 MAE m | 替换后 MSE m² | 替换后 RMSE m | MAE 改变量 m |
|---|---:|---:|---:|---:|---:|
| 072C linear | 590.7435832803362 | 599.0629769960383 | 10823892.543918177 | 3289.9684715690178 | +8.319393715702063 |
| pre_pool | 591.0374630199375 | 600.8806714319327 | 11086639.759055229 | 3329.6606071873493 | +9.843208411995192 |
| shuffled | 592.5791927985551 | 601.563841400174 | 10857233.064744636 | 3295.0315726476183 | +8.984648601618915 |
| preserved | 612.6133294724413 | 600.4424335229655 | 10728179.934452508 | 3275.39004310212 | -12.170895949475835 |

公式为 `MAE=sum(n_r MAE_r)/N`、`MSE=sum(n_r MSE_r)/N`、`RMSE=sqrt(MSE)`。**现有聚合不能重构混合 P99，故没有伪造 P99。** 此控制否证直接退回物理 prior：090B 保护的是强模型输出。

若保持 072C inside 不变，全局相对其降低 3% MAE 所需 outside 改善为 `0.03*590.7435832803362*442263/119310=65.69374638479037 m`。outside 当前 MAE 1546.409937155145 m，需降至 1480.7161907703546 m，即 outside 相对改善 4.248145644074427%。这只是点权代数门槛，不是效果或 tail 可达性的预测。

## 三臂与唯一干预

| 臂 | 起点 | 可训练量 | 输出与路由 |
|---|---|---:|---|
| 090A all_queries | 完整 072C final | adapter 4578；teacher 逐值冻结 | 所有 query 的 teacher 预测位置 + 新二维切平面增量 |
| 090B anchor_unsupported | 同 A | 同 A，逐值相同初始张量 | 增量乘 `~motion.support[anchor]`，包括 helper-supported 和 group-no-support |
| 090C full_finetune | 同一完整 072C final | 原模型全部 105154；无 adapter | 追加预算全模型微调对照；原输出参数化 |

A/B 输入为原可见 76 维特征和冻结 Transformer 最终 64 维隐状态，拼接 140→32 SiLU→2。新输出层 weight/bias 置零；同一独立 branch seed=`training.seed+90000`。输出单位 km→m，在 teacher 预测位置的局部北/东切平面增量后 ECEF 归一化。teacher 使用 eval、requires_grad=False、无 optimizer 条目；首末全部参数逐值核验并拒绝 teacher 梯度。A/B 总参数 109732。C 不新增输出分支。

这里没有增加原始 helper 信息、逐 helper 表示或新的时序编码能力：64 维隐状态完全来自已有 teacher，76 维也与 072C 相同。它是**冻结表征的新读出与修正作用域实验**，不能把增益解释成获得了新的 helper 证据；C 的全模型参数更新能力与 A/B 不同，也不能把 C 的差异仅归因于作用域。

初始输出所有臂都必须与 072C **final** 每个原输出组完全一致，包括重复/未引用组；不是旧模型 step0 或 fixed .95。速度/航向保持原 Mean；身份、标签、真值误差不进入预测/路由。

## 训练与预算

各臂新增 4000 次成功更新，batch8、Adam lr .001、clip1、无权重衰减、linear Haversine km。**fresh Adam**，不加载旧 optimizer；C 必须称“追加预算全模型微调”，不能称恢复原协议续训。NumPy sampler 从原 seed 重启，实际每个 batch 的 scene、anchor、mode、members、有效点和机会点逐项匹配原 072C 的 4000 条成功更新日志；与 A/B/C 彼此的成功曝光也必须一致。

每个 anchor 仍按全部有效位置点平均，再按原 batch 平均。B 不删除已支撑样本、不以可修正点重归一化；已支撑部分对 adapter 只是常数损失。无参考点的 query 不凭空加入目标。原始 T/V 顺序、真实时间、单位和 masks 全保留。训练不做 V 选择或区域挑选。每次单卡、CPU4，墙时 advisory，失败不自动重试/降级。

成功更新日志逐样本记录全部/有效 query、anchor 支撑交集，以及实际允许修正的全部/有效 query 数。CPU 联表从成功 batch 重新累加，检查支持集合交集的合法范围、B 的实际允许量等于无支撑量，并检查 A/B/C 的可见支撑曝光一致；不只信任汇总 `route_exposure`。全支撑或零有效样本仍保留原分母，允许零梯度更新，计入实际采样和成功更新，但不能把零有效点报成额外有效监督。

预算解释：所有臂依赖已有 4000 更新 teacher，再各用 4000 新更新。不能拿 A/B 的两阶段收益当作与原 072C 等总预算。C 使用同追加批次/optimizer设置以控制追加曝光；参数容量和更新参数数仍不同，分别报告。

## 完整 CPU 缓存比较

每臂独立完成 initial/final 全 V300；A/B 单臂回执不等待 C。以真实 checkpoint、实际配置、成功日志、原始输入/分组/标签/masks/组件核验，不能只信 status 字符串或哈希。

对每个完整输出组记录 teacher 源输出、实际增量、可见 anchor support；A/B 验证 `tangent_position(teacher, applied_delta)`，B inside 输出必须逐值等于 teacher；C 验证原物理 base 的 tangent 几何。所有组映射到同一个完整 target 人口，缺失预测、缺参考、重复输出计数保留。

CPU 联表方法含物理控制、072C、可用的 A/B/C，以及 **posthoc_mask_A**：只用缓存 support 构造 teacher inside + A outside 的完整 group/aligned prediction，再重新计算所有误差数组及 P99，零训练、零新模型 forward。不做逐区择优。B 对此控制的优势才支持“训练阶段局部化”超过部署时硬屏蔽。

返回全局及三互斥支撑区（另列 no-predicted-group 人口）、四种 MAE/MSE/RMSE 权重、真实 point P95/P99、H/I 与净差分恒等式（相对 teacher 和 fixed .95；有 C 时再相对 C）、逐场、38 个身份组件、最大组件及其外部。完整 NPZ/checkpoints/JSONL 留服务器；聚合与必要逐场表回传。

## 固定筛查与解释

候选对 072C 全局点 MAE 至少降低 3%；RMSE/P99 不恶化（数值容差 1e-6 m）；其他三权重 MAE 和最大组件外 MAE 不恶化；超过半数场景及组件改善。有 C 时还必须对 C 点 MAE 严格改善（超过 1e-6 m），RMSE/P99、全部四权重 MAE 与最大组件外 MAE 不恶化；C 缺失/失败时只能报告暂定筛查，不能排除追加预算解释。A/B/C 全部报告，不后验删臂改阈值。

B<A 且 B≈posthoc_mask_A：主要是部署路由，不足以声称训练局部化有额外作用。B 和 A 都无益或 C 同样/更好：停止当前 adapter 主方法晋级；保留新增预算/表示能力解释。B 支撑区必然无损，但 outside 没收益或尾部变差：否证当前可见分工，不靠删除区域挽救。仅主筛查通过才由后续显式交接考虑独立 seed/全系统确认。

重复使用 V300、共享身份、单 seed 与原合成 T 组/部署固定关联组差异必须披露；支撑区代数不变性不是准确性或泛化保证。
