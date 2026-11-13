# 航迹关联与恢复：流程及保留组件

本页只说明保留代码的数学行为与证据边界。数据定义见[数据约定](../data_contract.md)，成绩与选择见[研究状态](../research_status.md)，执行只看[交接索引](../../HandOver/Handover.md)。已结案配方、逐轮预算和测试清单从[历史摘要](../history/README.md)所列Git版本查阅，不在这里累积。

## 离线输入与输出

输入为完整场景量测，输出为历史状态及关联结果，关联标签和参考状态只在训练目标或事后评价使用。运行入口见[离线命令](../../tools/offline/README.md)。完整窗口和已见前缀重算的原计时说明见历史报告。

## 从输入到输出

MTAD的measurement是一条已带本地航迹编号的状态记录：`source、batch、time、lat、lon、vel、cou`。同一场景中相同`source/batch`的多条记录组成一段航迹；它不是单个雷达回波，片段数也不等于真实船数。项目从这些记录开始处理，不包含雷达检测及单传感器航迹形成。

|阶段|输入与输出|实现入口|
|---|---|---|
|读取与清洗|CSV记录→按`(scene, source, batch)`分组、排序、去重及合法性检查后的航迹段|[archive.py](../../association/data/archive.py)、[native.py](../../association/data/native.py)|
|时间与运动证据|合法原生观测→共同Q64时刻、插值/OLS外推状态及支撑信息|[timed.py](../../association/data/timed.py)、[motion_evidence.py](../../association/data/motion_evidence.py)|
|关联评分|全部`i<j`航迹对→056基础分数；HARD/STRUCT组合另加上下文残差|[evidence_association.py](../../association/models/evidence_association.py)、[association_context.py](../../association/models/association_context.py)|
|构组|边分数→互不重复、覆盖全部输入节点的组；下述整链组合使用WS聚合|[static_partition_pilot.py](../../tools/analysis/static_partition_pilot.py)的`agglomerate`|
|融合与恢复|每组运动证据→该组各query的状态；后端按组合接159＋178/O或PTFM/DeepAF|[source_risk.py](../../association/evaluation/source_risk.py)、[history_candidates.py](../../association/evaluation/history_candidates.py)、[同行组预测输入与实现](../../association/models/fusion_peer_inputs.py)|
|输出|组成员、query时刻和状态→可发送的JSON bytes|[group_prediction.py](../../association/evaluation/group_prediction.py)、[离线骨架的历史应用计时](../../tools/offline/README.md#历史整链前缀计时)|
|评价|保存预测与身份/reference→关联、覆盖及逐属性误差|[论文指标](../mtad_data_protocols.md)、[数据与评价约定](../data_contract.md)|

Q64是覆盖当前场景时间范围的计算/评价网格，不是传感器采样频率，也不代表每秒输出64次。完整窗口允许使用某个历史query之后的观测；按已到数前缀重算与维护增量状态是不同实现。

离线`MTADArchive`读取标签以核验节点和构造监督；评分特征不使用真实身份/reference。工程计时入口的`ReceivedMeasurements`由已收到的source/batch和时间边界生成匿名节点描述，不读取真实关联标签。两种读取边界须分别说明。

[完整链计时](../reports/2026-11-05-pipeline-online-timing.md)覆盖应用收到measurement CSV字节至JSON序列化完成，实测组合为056＋WS＋PTFM/DeepAF、HARD/STRUCT＋WS＋159＋178/O；不包含传感器物理采集、网络传输或屏幕显示。下面保留056基础关联、旧anchor分组及072等位置网络的数学说明，它们不是上述组合共用的默认后端。各组合是否保留及其证据以研究状态和原报告为准。

## 可见输入与普通运动底座

候选和普通控制使用`motion_observation_policy=canonical_all`，由全部合法原生观测构造Q64紧凑证据；TMTAFF参考仍使用cap128。`selected_cap128`仅为旧行为兼容，不混用其权重或人口。全部节点、i<j候选、A-B标签、逐属性mask及短轨遵守数据约定，不能按难度删除。

[motion_evidence.py](../../association/data/motion_evidence.py)用真实秒对纬度、展开经度做全片段OLS。位置支撑内插值，外部从真实首／尾观测沿OLS斜率延伸，不以拟合截距替换端点；速度线性、航向短弧插值，外部保持端点。它能使用查询之后的观测，因此是离线恢复，20秒只是时间编码单位。

证据包含支撑、最近观测时距、点数／跨度、局部米制OLS残差、运动及属性差。支撑不等于准确测量，残差不等于校准协方差。source只用于同源关系及分层，不以9001/9002编号预定哪源可信。预测不读取identity、reference或目标mask；这些只用于训练构组/监督和事后评价。标准化只拟合T_core，V冻结。

## 056基础关联与历史anchor分组

[EvidenceAssociation](../../association/models/evidence_association.py)使用

`score = -log1p(RMS_distance_m / 1000) + MLP(features)`。

MLP为两层64宽SiLU、无dropout、末层零初始化。base为FP64，FP32残差转FP64相加，step0保持OLS排序。特征交换pair顺序不变，包含几何统计、去均值运动、共同支撑距离、属性差及支撑/点数/跨度/残差/同源标志，不输入绝对经纬度；列序和T内标准化随checkpoint保存。分数不是校准概率。

056固定来源为seed20261021、batch256正负各半、Adam .001、clip1、1000成功更新。scene→pair抽样并非完整遍历数据；有效曝光和实际批次需随运行记录。其它历史采样配方不因代码尚在而成为默认。

完整V按场景Macro-F1选阈值，相同分数整体加入、平局取更高阈值。OLS与学习可有不同阈值数值，但使用同一选择规则。`baseline/selected/final`须绑定实际权重、normalizer、配置、步数、scores和全V人口；不能用文件名、seed或摘要计数代替核验。

056历史anchor控制的分组是固定输入顺序的`anchor_direct`：取首个未分配节点作anchor，只收纳与它直接过阈值的未分配节点，`>=`包含等号。不是传递闭包或Hungarian。全部节点各出现一次；组首节点输出，`anchor_identity_first`仅在外部评价对应，未覆盖和重复照报。关联边F1、共属关系、分组覆盖/重复及最终状态误差分别报告。WS组合使用上表所列聚合入口，不使用这条anchor分组规则。

## 物理控制与072等历史位置网络

普通Anchor是上述运动底座；Mean supported在当前有支撑成员间融合，并按既定规则回退；Hybrid使用Anchor位置与Mean速度/航向。

位置机会区为`R=(~anchor_support)&any(helper_support)`。支撑规则在R内复制Mean、其余保留Anchor；固定.95初值在R内沿Anchor—Mean短弧混合，R外仍Anchor。这些控制只读可见信息，不是学习贡献。全人口和重复组都必须保留。

此类位置网络以固定.95为物理初值，读取76维紧凑证据：56维helper池化证据、anchor可靠性、速度/圆周航向、Anchor—Mean相对位移、真实时间和anchor相对首查询位移。helper平均/最大池化按可见support处理，标准化或单位缩放不拟合V。没有helper也保留anchor/时间输入；所有Q64 query参与，不将目标mask作为padding。

通用学习控制包括两层64宽LSTM和两层64宽、4头、FFN256的Transformer（无dropout）。Transformer为105154参数，LSTM为71746参数；不是同容量控制，也不是某篇领域论文的复现。公共物理底座已使用全片段观测，因此二者均按离线恢复解释，不因LSTM单向就声称在线。

零初始化二维输出头在全部query预测北/东米制残差。经球面单位向量与切平面残差归一化映射为lat/lon；零残差保持初值且梯度有效。可表达范围受底座半球限制，不能称任意全球跳转。速度/航向统一保留Mean，当前位置收益不能包装成三属性学习成功。

训练主目标为每anchor有效query的`Haversine_m/1000`均值，再对batch等权；不删困难点或R外点。典型batch8、Adam .001、clip1及固定终点由具体配置确定。T中按场景/anchor抽样，轮换身份构造的`clean/clean_plus_wrong/wrong_only`；部署用预测组，两者分布差异需披露。保存实际成功样本、optimizer、RNG、配置与固定端点，不事后挑最佳V步。

072C/073C为4000更新的两个seed来源；090C/097A是其final上fresh Adam及原成功批次重放的预算对照，不能称恢复原Adam续训。新增helper集合结构及098关系偏置属于单独配方，不自动替换默认；其当前结论只由原合同的同预算、同人口和机制对照决定。

## 保留的属性效用模块

[EvidenceFusion](../../association/models/evidence_fusion.py)从27维可见helper证据预测三属性utility，`p=relu(u)`；helper权重`p/(1+sum(p))`、anchor权重`1/(1+sum(p))`。经度/航向沿anchor短弧混合，速度凸组合，全部utility非正时回退anchor。区间内插值不自动锁成准确观测。

其直接损失为混合误差的`log1p(error/scale)`，scale为1000m／10原生速度／30°，逐属性有效点均值再平均。旧helper_quality/mixed_action/direct_only辅助目标由配置绑定，不混名；单helper收益不等于多helper联合效用。这些代码用于历史解释和明确的独立实验，不是已证实的默认完整方法。

## 评价与可信度边界

全部组先预测，再读取真值对应；完整人口、缺参考、未覆盖、重复输出、共同有效点均保留。位置米、速度原生单位、圆周航向角分开。MAE/RMSE/P95/P99、四种位置加权、支撑层、逐场/38共享身份组件及最大组件外同时报告；全局分位数不能由场景分位数平均代替。

固定056位置融合的覆盖不变是控制条件，不能算融合贡献。常量/OLS/插值的收益不归学习创新；两个模型的不同人口条件MAE不能直接排名，骨干控制不能冒称领域同行，退化参考不能成为胜过作者论文的证据。

同seed不足以证明配对；核实际初值、成功批次、输入与输出人口。反复V开发和跨场共享身份不构成独立泛化；T_test继续封存。有限软件合成检查仅用于发现实现错误，不是独立科研成绩、原论文性能复现或真实MTAD同行资格，也不自动成为正式实验之前必须不断追加的阶段。
