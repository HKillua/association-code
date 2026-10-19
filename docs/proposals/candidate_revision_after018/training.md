# 候选v2训练协议与四臂比较

2026-10-19。当前方法及训练安排已获同意，正式执行见[019—025队列](../../../HandOver/Handover.md)，候选合计4GPU小时。本文只定义训练与比较语义；成本、配置和分任务额度统一见[执行安排](../repair_before_retraining/README.md)。017/018已结案，旧额度不转移。

## 固定更新：1000共享训练＋200效用训练＋四臂各300，共2400更新

主要训练调整是：**共享encoder、补全/融合底座及身份支路持续联合训练1000次，之后才固定公共表征进行可归因的四臂比较。** 不再把018仅teacher100步的表征和身份作为整个后续实验的永久起点。1000是预先固定的开发训练档位，不是论文epoch或“足够收敛”的保证，不能把2400总更新说成身份训练了2400步。

保留T_core512/V64、原A-B标签和全部候选、全部64个query及合法reference监督（包括长外推）、cap128、unit20、既有offset/gap600、FP32、单seed20、batch64平衡scene-pair采样、原首尾留出比例0.25、标准SmoothL1与当前物理loss/速度/圆周航向语义。训练侧native normalizer和MLP时间缩放只拟合一次；不删高速度、短轨或长缺口、不改采样分布、不读T_test。具体张量来源仍是当前可见原生输入，支撑特征在端点留出后重算。

fresh初始化，不自动加载018旧teacher或旧u。既有模型尺寸d128、6层、8头、FFN256保持；新W_phi/局部身份证据头/W_c按本版方法规格实现。W_c固定Xavier初始化，所有对照相同；纬经/速度状态行零初始化、航向沿原初始化，具体见representation.md，不在运行中临时调整。

## 1. 阶段、更新对象与训练信号

| 阶段 | 更新数 / 种子 | 更新的参数 | 冻结的参数 | 训练信号 |
|---|---:|---|---|---|
| A：共享teacher | 1000 / 20 | input projection、encoder、支撑查询W_phi、query_pool、decoder、物理state projection、W_c、实际使用的role embedding、原identity_base＋新增局部身份残差头；原FusionLoss参数 | utility_head、普通learned gate、未启用的utility辅助身份头 | 保留原teacher组合：所有采样pair的两条self恢复；正pair的双向普通cross恢复；所有pair身份BCE。物理范围与权重策略不变。 |
| B：冻结teacher效用学习 | 200 / 21 | **仅utility_head** | 整个E、W_phi、query_pool、D、W_c、物理输出、全部身份分支、物理loss参数；另保留独立冻结teacher快照 | 同一新结构teacher在相同可见输入、同query上产生self/cross四属性误差；仍用停止梯度的误差残差监督，epsilon与物理尺度保持现值。 |
| C：四臂物理适配 | 每臂300 / 同22 | decoder、state projection、W_c、实际使用的role embedding及原FusionLoss参数；learned臂另训练ordinary_gate | encoder、input projection、W_phi、query_pool、身份分支、utility_head；各臂的teacher始终冻结 | 同一预生成采样序列和端点留出，仍仅正pair的双向物理恢复；四臂从完全相同的A末物理状态与B末u状态复制。 |

**关键兼容修正：** 018的query_pool只用于效用分支，故它在utility阶段训练；新融合的`r_j(t)`直接使用query_pool，query_pool成为物理信息底座，必须进入A阶段训练，在B/C阶段固定。若B继续更新query_pool，就会让新融合所读特征漂移，不能一边声称teacher固定一边改变其来源表示。B只训练u头，避免额外教师重标轮次。

效用标签是同一冻结teacher对锚轨自身恢复与“锚轨←辅助”恢复的逐query、逐属性误差。每方向均用锚轨自己的reference及逐属性合法mask，负辅助也不伪造共同目标。误差为物理绝对残差除以T_core原生normalizer的对应std，经度/航向取最短圆弧；不使用V拟合尺度。令停止梯度后的误差为e0/e1，效用损失对`r=(e0-e1)-max(e0,epsilon)*u`使用标准SmoothL1(r,0)，beta=1、epsilon=.001；各属性按有效监督数归一后四属性平均。保留负增益监督，仅在推理路由时clip效用，不对小分母直接做比值回归。

手工MLP时间尺度`S=max(1,max(abs(tau)))`仅从T_core打包FP32输入中所有有效native时刻与全部query时刻拟合一次；padding、状态、reference、V均不参与。各阶段/教师/四臂共享同一S，只缩放手工时间差/乘积与支撑距离；实际时间、20秒sin/cos和物理支持不变。

A阶段普通融合使用确定的无效用路由，不依赖未学好的u；先有普通物理恢复器，再比较其self/cross误差，所以不存在“必须先有u才能训练teacher”的循环。普通teacher按现定义不使用身份概率当物理监督标签。四属性标签由**新teacher**生成，不能把018旧decoder的收益当新残差融合器的真收益。

B结束后到C的冻结是为了同一身份、同一表征和同一效用语义下比较路由；不是声称这些模块永远无需再训练。若1000步后身份仍弱，原样报告并用oracle诊断拆开恢复问题，不自动加训、换seed或用V选一个特殊阶段规避负结果。

优化器沿018：Adam、lr=1e-4、clip_norm=1；每个阶段/每个臂重新建对应优化器，A阶段内部连续1000更新，不每100步重置。实际累计updates、阶段训练参数名单/数目、分项loss与裁剪前范数写入同一run/history。损失权重机制沿现实现，不改变任务配比；因此训练步数增加仍未保证消除梯度竞争。

## 2. 四臂定义：沿同一新结构只比较路由

四臂均使用`h_self + W_c × helper_summary_residual`，同参数规模、同输入、同选中身份分组、同eps floor；绝不让某个臂换回旧group decoder或另加监督。为确保参数账单一致，四臂实例都保留同尺寸u/ordinary模块，不使用的模块不影响输出。

| 臂 | 各属性k的辅助来源系数 | 回答什么 |
|---|---|---|
| plain | 所有辅助的w=1 | 普通多源残差融合能否已解决主要问题；保留用户要求的普通融合对照。 |
| scalar-u | `eps+(1-eps)×sigmoid(s')×clip(mean_k u_k,0,1)`，四属性共用 | 维持同新结构但不分属性。 |
| attribute-u | `eps+(1-eps)×sigmoid(s')×clip(u_k,0,1)` | 属性专属效用路由是否比scalar-u多提供收益。 |
| learned-attribute | `eps+(1-eps)×sigmoid(s')×sigmoid(g_k)`；g与u同输入、同隐藏宽度、同四输出 | 直接用物理任务训练的普通属性门控能否解释收益。 |

后三臂使用**相同身份因子和floor**，不能让learned臂缺身份项而把差异归给效用监督。g是新的普通门控，C阶段随物理loss训练；u则先经B的200次教师增益监督后固定。它们参数容量和C阶段采样/更新相同，但训练目标与额外学习机会不同，不能称总优化预算完全等价；B的成本单独披露。这里隔离的是“效用预训练路线与普通物理门控路线”的实际增量，不宣称单一因果机制被严格证明。

teacher快照、u头、身份分数和编码结果可共享；最终四套decoder/W_c为独立适配副本，不串行继承前一臂权重。保留四臂有清晰问题对应，不扩回旧plain/identity/utility/learned再额外叠四臂。没有单独identity-only臂的结论边界：plain与scalar-u的差同时含身份和效用路由，不能据此独占归因u；核心属性归因只看scalar-u与attribute-u。

## 3. 选择时点与评价

- A1000固定端点产生唯一teacher/身份版本；不按V挑teacher更新数，不以训练曲线最好的一批替换端点。
- A结束后一次完整V身份评分，按已有macro-F1阈值选择规则选一个阈值；B/C身份参数不变，此分数/阈值/分组四臂共用。除数值检查外，不每20步跑全V终评。
- B200、C各300均为固定端点，不按物理结果选最优臂checkpoint。四臂各一次完整V physical评价，保存全部预测和相同目标/point population。
- 日常训练日志按原更新保存；直接从日志每20步聚合分项曲线，不为“看趋势”增加训练、重新校准或开一个独立诊断轮次。失败保留原件，一次尝试，不续跑。
- 报预测组与oracle组、所有目标覆盖、真实缺失/已有支撑分区、位置MAE/RMSE、速度和圆周航向、关联P/R/F1及错并拆分；沿当前区域定义，同时披露单轨self监督含长外推。可以随运行记录u/g属性摘要，不能靠训练后临时选一个子集阈值证明收益。

整体baseline是020修复后的完整TMTAFF近似参考；它与本候选loss/优化量不同，属于完整系统比较，不是等预算结构消融。018可作历史学习进展参照；新结构、训练量、训练参数集合均变化，不能把新旧分数差全部归于某一模块。首轮不追加“关闭W_phi/局部身份头”的多种组合，总体有信号后再做阶段归因。

## 4. 独立执行

021只训练A/B并固定身份分数与阈值；完整COMPLETED后，022—025各从相同共享端点独立执行一个C阶段，使用同seed22、同初态和各自新Adam。各臂不重fit统计、不重训公共部分、不重新评分或选阈值，失败不改配置续跑。真实成本与是否学会以本次结果为准。
