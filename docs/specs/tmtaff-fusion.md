# TMTAFF融合组件实现约定

2026-10-18。目标：让既有TCM补全输出可训练地进入TFM，并提供显式参考对齐接口。仅隔离分支开发及合成软件验证，不改变活动011、不下发原生训练或选定新的评价协议。原文§2.4–2.5；论文未公开处如下明示。

## 数据接口

复用reference_states原始reader，不改它的原始记录。新增`align_reference_states(reference_scene, identity_id, query_times, *, time_offset_seconds, max_gap_seconds)`，返回`states[L,4]`、逐特征`valid[L,4]`与问题/覆盖计数。字段顺序lat/lon/vel/cou，时间统一为`raw_reference_time - time_offset_seconds`；offset和max_gap均必须显式给定，不自动从真值估计，max_gap=0只作精确时间匹配。

输入仅允许reader提供的训练/开发场景；T_test拒绝。query_times为调用方明确给定的有限、严格递增时间，不自动把scene64实际观测时间当名义网格。原生场景使用何种offset、gap、网格要在正式配置中另行确定。

适配复制数据，不更改reader或量测。有限时刻按身份排序；相同数值重复可用于一个监督时刻，冲突/无效属性逐特征屏蔽并计数，不静默挑第一行。lat/lon合法范围、非负native速度及0..360航向按已知数据契约校验；360归0仅在监督副本。缺失时刻不外推；仅在相邻原参考时刻且间隔≤max_gap、两端该特征有效时插值，不跨无效参考点桥接。航向按最短圆弧插值，恰好180度的方向歧义屏蔽；其他属性线性。速度不转换m/s或kn。无效输出用NaN并显式mask，不能当0真值。

## PairFusion模型

`PairFusionConfig(fc_hidden=..., length=64,d_model=128,nhead=8,num_layers=6,dim_feedforward=256,dropout=.1)`，fc_hidden必填正整数，其余对应既有TCM配置。论文未披露两FC中间宽度，不伪造作者值；正式配置和总参数成本在训练方案中显式给出。

`PairFusion.forward(left,right)`输入[B,L,d]有限浮点补全嵌入，左右各独立encoder，不共享参数；沿时间拼接、展平，两FC经LeakyReLU恢复[B,L,d]作为融合memory。独立decoder以learned BOS和自己生成的历史latent逐步输出，self attention因果，cross attention可见完整memory（离线）；函数不接收真值序列。BOS及latent递推是原文未披露细节的明确适配，不冒称作者实现。

返回字典`states[B,L,4]`与`fused_features[B,L,d]`，二者不混用；四维输出是原生状态尺度（度/native速度/度），不是标准化状态。speed用softplus保证非负，属于未公开输出约束的明确适配选择；lat/lon不clamp，航向loss按圆周处理。全层独立初始化沿现有模型惯例。

按§2.5在最终状态投影前另加配置中的dropout。长度64、维128时，仅两FC参数量为`24577×fc_hidden+8192`；中间宽128/256/8192分别约3.15M/6.30M/201.34M，不能把未知宽度随意当8192。当前自回归重复计算完整历史，真实成本须随后在完整配置上测量，不能以TSA的耗时外推。

`forward_pairs(completed_tracks,pairs)`从[N,L,d]按[E,2]索引，无detach；可直接接`AssociationModel.complete`。不改原model/trainer/CLI。递归反馈正文与Algorithm1不一致，本次不静默决定完整递归规则，也不将4维预测当成128维嵌入。

原文§2.4的TFM训练仅使用同一目标的真配对；此接口不接收身份，也不替调用方选择训练对。后续trainer必须以训练侧身份构建正配对，不能把TAM预测的正边直接当融合监督。预测关联分组属于完整推断/评价侧，用于检验错配对融合的影响。

## FusionLoss

`FusionLoss(position_loss_mode=...)`显式选择`paper_printed`或`standard_smooth_l1`；mode无默认。前者按Eq20字面阈值.001、二次0.5e²/线性|e|-.0005，保留不连续疑义；后者显式用SmoothL1 beta=.001，明确属于解释性修正，不冒称原公式。

`forward(prediction,target,valid)`的mask是[B,L,4]bool。只对有效监督取值，不把无效NaN送进log/乘零掩盖；逐特征均值，任何特征全无监督明确失败。有效prediction/target必须有限，负速度拒绝。

速度采用log(value+1e-6)差平方，航向采用1-cos(pi/180*差)。四项损失用Eq18可学习log variance s组合：sum(.5*exp(-s)*L+softplus(s))，初始s=0；不暗改单位或自动截断数值。返回total、分项loss和学习到的variance，保存重载必须一致。

## 必须验证

| 行为 | 检查 |
| --- | --- |
| 对齐无外推/无跨坏点插值、时间平移显式、圆周正确 | 手工小数组及冲突/缺失/边界测试，输入不变 |
| encoder独立、输出shape/有限性、无真值输入、eval可重现 | 小模型真实forward及重载 |
| TCM→TFM可微，TAM另有监督 | 含缺端点的合成输入；TCM/TAM/TFM三loss联合backward/optimizer，三模块梯度有限非零 |
| loss不吞坏值或全空监督 | 手算圆周/速度/位置边界与mask测试 |
| 活动行为不回归 | 新模块测试＋已有模型/数据/原始reader测试；不跑原生数据或GPU训练 |

合成通过只说明接口和数值路径可用；参考时钟/网格的原生配置、递归/目标分组、完整评价、正式训练入口及预算仍未完成。没有“完整TMTAFF已复现”的结论。
