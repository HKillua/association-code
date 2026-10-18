# TMTAFF融合组件实现约定

2026-10-19补充：[经度边界修复](longitude-boundaries.md)新增显式最短弧参考插值和经度损失选项；以下线性经度描述仍适用于默认历史模式。017保持冻结配置，新候选草案启用修正；这不改变纬度、速度、航向损失及其权重。

本页记录完整参考TCM→TFM的监督、分组、评价和训练行为。完整外层已接通，见[训练与恢复规格](complete-training-refactor.md)；013/014的真实短训/终评与修复见[历史摘要](../history/README.md)，当前017状态见[交接索引](../../HandOver/Handover.md)。原文§2.4–2.5未公开处均明示适配，不称作者精确实现。

## 数据接口

复用reference_states原始reader，不改它的原始记录。新增`align_reference_states(reference_scene, identity_id, query_times, *, time_offset_seconds, max_gap_seconds)`，返回`states[L,4]`、逐特征`valid[L,4]`与问题/覆盖计数。字段顺序lat/lon/vel/cou，时间统一为`raw_reference_time - time_offset_seconds`；offset和max_gap均必须显式给定，不自动从真值估计，max_gap=0只作精确时间匹配。

输入仅允许reader提供的训练/开发场景；T_test拒绝。query_times为调用方明确给定的有限、严格递增时间，不自动把scene64实际观测时间当名义网格。当前配置显式使用已核逐场景offset与gap600，Q64沿现原生adapter；更改规则仍需明确研究决定。

适配复制数据，不更改reader或量测。有限时刻按身份排序；相同数值重复可用于一个监督时刻，冲突/无效属性逐特征屏蔽并计数，不静默挑第一行。lat/lon合法范围、非负native速度及0..360航向按已知数据契约校验；360归0仅在监督副本。缺失时刻不外推；仅在相邻原参考时刻且间隔≤max_gap、两端该特征有效时插值，不跨无效参考点桥接。航向按最短圆弧插值，恰好180度的方向歧义屏蔽；其他属性线性。速度不转换m/s或kn。无效输出用NaN并显式mask，不能当0真值。

监督装配接口`build_fusion_targets(prepared_scene, reference_scene, pair_indices, *, split, nominal_grid, time_offset_seconds, max_gap_seconds)`只把调用方已选真配对接到上述接口，不采样、不读取文件、不决定正式offset/gap。核两份scene_id相同、reference split与显式split一致且只训练/开发；若prepared_scene含split也必须一致。pair_indices为现有candidate pairs的整数行索引（可重复，保留顺序和次数），每个所选对必须非自配对、标签为1且两端A-B身份相同；负对、身份/标签矛盾和越界直接报错，不自动筛选。nominal_grid显式传入，有限严格递增、长度与values/observed的L一致，且等于prepared_scene.grid；不使用observation_times代替。

返回pairs、identity_ids、query_times、states[P,L,4]、valid[P,L,4]及覆盖报告。按唯一identity仅调用一次align后复制给对应pair；无监督仍保留NaN/false，不能删除pair或改采样权重；输入不可变。测试必须覆盖真实prepared_scene接口、正负对/场景/split/时间轴错接、重复抽样次序、部分/零监督及TCM→TFM→loss的合成连通。它是监督数据，不能并入模型观测。

`build_scene_fusion_targets`的整场装配按显式目标身份顺序保留全部目标，包括缺参考身份。2026-10-18普通效率修复只减少重复过滤：首个目标仍经公开对齐入口完整校验，成功后在本次调用内按身份分组一次，其余目标只处理各自的原顺序行；非字符串reader身份保持原入口兼容路径。不在reference字典中加入或相信可由调用方伪造的索引/缓存，不按属性有效性丢行。公开`align_reference_states`签名、监督数值/NaN/mask/query、issue顺序、整scene的`reference_rows`/`source_issue_count`、缺身份与非法行/状态的错误行为均不变；空目标场景仍执行原校验。复杂度由每身份重扫全scene，降为首身份校验、一次分组和各身份相关行处理，不声称免除首次全scene校验。

此优化验收：合成多身份数据逐值/逐mask/逐report对比公开入口，含重复同刻冲突、坏属性/时间、缺身份和额外无关行；非法行/状态及参数报错保持；输入/返回值不别名；用可计数行列表确认全scene遍历次数不随身份数增加。只跑CPU合成测试，不扫描原生CSV或改变正式gap/offset。

## PairFusion模型

`PairFusionConfig(fc_hidden=..., length=64,d_model=128,nhead=8,num_layers=6,dim_feedforward=256,dropout=.1)`，fc_hidden必填正整数，其余对应既有TCM配置。论文未披露两FC中间宽度，不伪造作者值；正式配置和总参数成本在训练方案中显式给出。

`PairFusion.forward(left,right)`输入[B,L,d]有限浮点补全嵌入，左右各独立encoder，不共享参数；沿时间拼接、展平，两FC经LeakyReLU恢复[B,L,d]作为融合memory。独立decoder以learned BOS和自己生成的历史latent逐步输出，self attention因果，cross attention可见完整memory（离线）；函数不接收真值序列。BOS及latent递推是原文未披露细节的明确适配，不冒称作者实现。

返回字典`states[B,L,4]`与`fused_features[B,L,d]`，二者不混用；四维输出是原生状态尺度（度/native速度/度），不是标准化状态。speed用softplus保证非负，属于未公开输出约束的明确适配选择；lat/lon不clamp，航向loss按圆周处理。全层独立初始化沿现有模型惯例。

按§2.5在最终状态投影前另加配置中的dropout。长度64、维128时，仅两FC参数量为`24577×fc_hidden+8192`；中间宽128/256/8192分别约3.15M/6.30M/201.34M，不能把未知宽度随意当8192。当前自回归重复计算完整历史，真实成本须随后在完整配置上测量，不能以TSA的耗时外推。

`forward_pairs(completed_tracks,pairs)`从[N,L,d]按[E,2]索引，无detach；可直接接`AssociationModel.complete`。不改原model/trainer/CLI。递归反馈正文与Algorithm1不一致，下述独立推断接口要求调用方显式选择FC memory解释，不将4维预测当成128维嵌入；这是已公开的近似解释。

原文§2.4的TFM训练仅使用同一目标的真配对；此接口不接收身份，也不替调用方选择训练对。后续trainer必须以训练侧身份构建正配对，不能把TAM预测的正边直接当融合监督。预测关联分组属于完整推断/评价侧，用于检验错配对融合的影响。

## FusionLoss

`FusionLoss(position_loss_mode=...)`显式选择`paper_printed`或`standard_smooth_l1`；mode无默认。用户2026-10-18已批准`standard_smooth_l1(beta=.001)`，作为Eq20的明确解释性修正；lat/lon仍以原生度逐维计算，不冒称作者精确公式。`paper_printed`按本地科学记数解释采用阈值.001、二次0.5e²/线性|e|-.0005，仅保留兼容与历史用途，不将该记号解释认作作者已确认含义。此决定不包含CL margin规则或GPU预算。

`forward(prediction,target,valid)`的mask是[B,L,4]bool。只对有效监督取值，不把无效NaN送进log/乘零掩盖；逐特征均值，任何特征全无监督明确失败。有效prediction/target必须有限，负速度拒绝。

速度采用log(value+1e-6)差平方，航向采用1-cos(pi/180*差)。四项损失用Eq18可学习log variance s组合：sum(.5*exp(-s)*L+softplus(s))，初始s=0；不暗改单位或自动截断数值。返回total、分项loss和学习到的variance，保存重载必须一致。

微批累积可显式传`normalization_counts[4]`与`regularizer_weight`：每维按整个有效batch的监督数归一化，返回分项是本微批对全目标的贡献，日志相加；四项uncertainty正则全batch合计一次。局部微批某维无监督贡献可微零；全batch任一维无监督仍失败，不能重抽或删对。默认调用保持原完整batch语义。

## 分组与递归输出

`fusion_inference.py`的预测分组只接节点数、候选边、分数和显式阈值，不接身份或参考。必须显式选择`grouping_policy='anchor_direct'`和`node_order='input'`：依输入顺序取首个未分组节点为锚点，收纳与锚点直接判连的剩余节点，不做传递闭包。缺边不连，`>=`含同分，`None`沿既有指标表示全拒绝；重复无向候选、自边或非有限分数报错。所有节点恰出现一次。真身份分组是分离的评价helper，不能冒作预测关联。

`fuse_track_groups`要求模型已eval、完整固定序分区，显式`feedback_policy='fc_memory'`、`singleton_policy='missing'`。每次左输入为累计FC memory，右输入为下一片段TCM输出，最后一步四维状态为本组输出。单节点组保留记录、状态为singleton_missing、输出None，不自配对、不丢弃；上层须接单源轨迹。混合身份的预测组照常融合，不读取真身份纠正。空场景仍检查接口；不修改权重、模式或输入。

## 评价原语与边界

`evaluate_aligned_fusion`接完整、由调用方显式对齐的预测/真值[N,L,4]、两份逐特征mask及[N,L]区域mask。位置按球半径6371000米的haversine距离，速度保留native单位，航向按最短圆周度差；分别给MAE/MSE/RMSE、可评真值数、预测交集数、缺预测数、点及轨迹覆盖。漏预测留在真值分母；零交集误差null，零真值覆盖null。非法声明为有效的值报错，不能以region mask隐藏。已观测/缺失区间由调用方分别明确mask。

`group_membership_diagnostics`在同一完整节点人口上计算拆分/错误合并的配对数及受影响组数；未分组节点单独计数，不把None聚为同一目标。它不选择轨迹真值匹配、不做Hungarian、不生成新综合惩罚分数。上层全目标对应、单源轨迹装配和汇总已接入完整评价；正式配置/选模见具体轮次，软件验证与真实模型结果分开。

## 三模块训练更新

`fusion_training_parameters`显式设置TFM_FROZEN或JOINT阶段模式/可训练参数，提供新optimizer成员，包含FusionLoss四个参数。`fusion_training_step`只接受已抽取的有效batch；不采样、不读数据、不启动fit、不选择checkpoint。正对target按labels==1的原顺序装配，重复抽样不去重。TFM_FROZEN只接真配对，整个关联模型冻结/eval；JOINT分别计算TAM＋completion_weight×TCM＋TFM。

关联微批均值乘本微批配对数/全batch配对数；TCM辅助仅对全batch唯一端点，按`2*d_model*Σmax(observed_count−1,0)`分配监督元素权重；融合损失按每维全batch有效数，正则只在第一个正对微批加一次。先核全batch监督和optimizer完整参数集合，再反传；一次清梯度、裁剪、step。异常/已有run deadline中断清除部分梯度且不step，不跳过并补抽。

`pair_microbatch`的TCM前向最多含其2倍端点；`track_microbatch`只限制TCM辅助loss前向，不能声称控制所有TCM显存。每个配对微批内TCM结果共享给TAM/TFM，跨微批重算以释放反向图。开启dropout会改变随机mask实现，精确梯度等价只在dropout=0下验证；默认精度/模型规模不因此修改。原生监督装配、四阶段外层loop、完整checkpoint（association/fusion/loss及配置）、选择与输出入口已完成CPU集成验证，不能按这里的组件规格直接启动未批准的原生训练。

## 必须验证

| 行为 | 检查 |
| --- | --- |
| 对齐无外推/无跨坏点插值、时间平移显式、圆周正确 | 手工小数组及冲突/缺失/边界测试，输入不变 |
| encoder独立、输出shape/有限性、无真值输入、eval可重现 | 小模型真实forward及重载 |
| TCM→TFM可微，TAM另有监督 | 含缺端点的合成输入；TCM/TAM/TFM三loss联合backward/optimizer，三模块梯度有限非零 |
| loss不吞坏值或全空监督 | 手算圆周/速度/位置边界与mask测试 |
| 分组保留全部节点、非传递、FC递归无真值输入 | 链图、错组、None阈值/同分、singleton及真实小模型手工递归对照 |
| 评价不因缺预测缩小分母 | 日期线/圆周手算、特征缺失、全部漏目标、split/merge计数 |
| 微批不改损失分母、冻结正确、未丢loss权重 | CE/contrastive独立整批梯度对照，重复pair/缺局部监督/纯负微批，dropout有限反传，超时无部分step |
| 活动行为不回归 | 新模块测试＋已有模型/数据/原始reader测试；不跑原生数据或GPU训练 |

合成通过只说明接口和数值路径可用；监督时钟与完整入口已落实，真实短训结果和当前017运行见交接。仍无“作者完整TMTAFF已精确复现”的结论。
