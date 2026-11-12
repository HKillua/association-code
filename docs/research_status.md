# 当前研究状态

更新：2026-11-12。这里维护目标、结论和下一步；实际执行只看[交接](../HandOver/Handover.md)。结案全文见[报告索引](reports/README.md)，整理前推导和取舍已[完整保留](history/2026-11-02-research-process.md)，更早沿革见[历史入口](history/README.md)。

## 当前要解决什么

研究统一经纬度、速度、航向及时间/来源归属输入下的多源航迹关联、融合与恢复。按[AGENTS](../AGENTS.md)、[完整科研流程](research_workflow.md)和[项目skills](../.agents/skills/README.md)，从真实问题和证据矩阵形成方案；允许更换骨架，成熟跨域方法的合理适配及明确新增价值均可研究，不追求绝对无先例。

用户于11月10日进一步明确：**离线观测和在线观测都要持续推进**，替代此前“先核清差距，再定主线”的安排。两条路线分别建立可信对照与验收标准，不等其中一条完成才开展另一条。离线路线以作者协议或明确统一协议下超过可信强baseline的主要精度为目标，速度作次要成本；在线路线则须在因果输入、持续更新条件下比较精度、延迟和处理规模。用户 11 月 12 日进一步明确无需固定 1 Hz，只要解决在线预测的实际问题。输出节奏与最大允许结果延迟尚未指定。[路线与对照审查](reports/2026-11-10-offline-online-route-review.md)保存原论文条件、152既有聚合及在线评测缺口；不能将当前任务适配数值称为已达到原论文水平，也不继续默认以196局部修补或离线加速推进。

离线首要对照明确为 **TMTAFF 完整系统**，同一个冻结版本须在ITSA、MSTSA和四属性融合上同时领先。用户进一步明确离线优先追求精度，允许重选整体方法。原生点与片段图关联、组级全窗状态和有限分区候选进入重新筛选，原056→WS→位置选源作为开发对照。位置改进和PTFM／DeepAF条件融合比较继续保留为内部证据与补充对照。现有完整组合的关联和四属性差距、作者协议边界及验收条件见[离线TMTAFF核查](reports/2026-11-10-offline-tmtaff-review.md)。

最终方案支持可变K和源换序/增减，不依赖固定源编号优先或固定双源配方；真实数据主要K2，软件K3检查不等真实泛化。用户接受合理训练侧监督及新数据集重拟合，倾向少依赖后端reference系数拟合；旧骨架不构成路线约束。

## 已有依据与保留方案

|问题/范围|当前判断与证据|
|---|---|
|关联与覆盖|HARD保折中候选、STRUCT保均值强控；两者覆盖和跨源指标各有代价。[173结构监督](reports/2026-10-31-structured-association-review.md)、[175匹配难例](reports/2026-10-31-decision-error-mining-review.md)、[176工作点](reports/2026-10-31-decoded-coverage-review.md)完整保留。176当前网格三规则都选回原阈值，无新增收益。|
|历史候选恢复178，原V32|保留真实平均收益：HARD/STRUCT位置MAE355.760/328.916m，覆盖及关联保持各自原值。但相对对应旧base仍新增128/107个>1km增害，T尾部反证未解。[完整178记录](reports/2026-10-31-history-candidate-review.md)。|
|简单平均运动MC，原V32|191相对178 MAE−2.676/−2.386m，RMSE/P99两头降；H描述区间跨0、去最大组件H略差。原PG公里害11/0仍在，救回保70/71及42/42。保条件候选，未自动替换178。[191记录](reports/2026-11-01-mean-velocity-validation-review.md)。|
|动作条件风险195，仅T32|JM MAE391.235117/364.845327m，对178−.997387/−2.357123m，对MC−2.059982/−.132478m；MC新增公里害34/1→7/0、救回3/33全保，覆盖不变。描述CI跨0、H P95/P99及S P95略差178，原front总害仍107/71。相对O的JM均值改善，CF约保有93%/97%；相对MC则约97%/47%，不能混称。JM重选仅1混组2点；保条件候选，未替换178。[完整195及远程审查补充](reports/2026-11-02-action-conditioned-risk-review.md)。|
|冻结动作规则196，原V32|科学与规定独审均成功。CF−MC MAE+.987763/+1.053248m、RMSE/P99更差，H11旧害全解、S仍0害，但原MC救回70/42仅保24/7；JM池外只14/3点、H改善/S恶化，去最大组件均0。部分支持/明确取舍，当前不替换178/MC。[196完整Record/Squeeze](reports/2026-11-03-action-conditioned-risk-validation-review.md)。|
|阶段信息门控194，仅T32|真实39fit/4992GPU更新、44臂和规定独审完整完成。ORDER−MC MAE+.047849/+.330501m，ORDER−BAG两头区间跨0；原34/1公里害保28/1，原3/33救回仅保0/28。当前粗四块＋小网络配方不晋级，不否定整个时序方向。[194完整分析](reports/2026-11-01-temporal-gate-review.md)。|
|解释与反例|192原生片段内部近均匀采样，当前密度解释缺前提；190全段拟合端点两种运动条件均恶化。不能把阶段门控失败、纯组大错或拟合状态当噪声/参考错误已证。[192](reports/2026-11-01-sampling-measure-review.md)、[190](reports/2026-11-01-fitted-endpoint-review.md)、[189](reports/2026-11-01-drift-memory-review.md)。|
|全300场与同行比较|147/157为完整V300内部贡献基准，MAE590.760/582.808m，后者仍有稀有反转与尾部代价；178/191/194不能跨人口替换这些数字。152按原合同完成PTFM/DeepAF训练和V300评价。11月10日聚合初核：同WS全V300原147/PTFM＋Mean/DeepAF位置MAE为590.760/738.682/1361.695m；PTFM的RMSE/P99/速航更好，属于取舍，完整数组/规定收尾仍待审。比较对象仍147，非作者完整系统胜负。[初核及边界](reports/2026-11-10-offline-online-route-review.md)。TMTAFF常值/尺度失效尚未获得可信复现或唯一根因。[147](reports/2026-10-30-source-risk-full-benchmark-review.md)、[157](reports/2026-10-30-t128-full-scope-review.md)、[152合同](../HandOver/L2S-20261030-152.md)、[TMTAFF定位](reports/2026-10-31-tfm-gradient-review.md)。|

上表各自报告T/V、有效人口、覆盖、尾部和成本；不拼两前端或不同实验冠军，不宣已胜论文方法。193导入故障与194科学结果分开，[193失败原件与原因](reports/2026-11-01-temporal-gate-startup-failure.md)完整保全。

## 下一步

1. **离线继续建立高精度关联与可信 TMTAFF 全系统对照。** [C95](reports/2026-11-12-association-negative-coverage-review.md)同预算替换一半旧尾部负例后，混合BCE跨源90%召回FP由C92复现的1,196,033降至105,720，原056为44,754。V最终跨源FN/FP相对原WS由5,888/8,085降到4,291/4,372，held亦双降。混合BCE丢103个原代表和35,645个同船支持槽，四新臂未过总门。加排序对两种采样的跨源效果相反，当前混合BCE保留为候选分区。
   - C107固定C104规则和C98候选，在72个新开发留场零分区编辑，跨源漏/假176/129、同源66/61、四属性及覆盖逐值相同。72个跨源旧供给只6个可配准，0/3误拒真、1/3拒假。69块中17个候选事后可补漏去假且混合J为正。获反证假边所在可行候选另需切一条只具2个共时外锚的假边，未获准入。历史13纯异船槽映射6实际目标×Q，其中4点位置MAE由0.133升至4.100km，移除新异船后恢复。C110先查整块证据覆盖和输出供给保护，再固定隔离留场。 [C107报告](reports/2026-11-12-association-registered-holdout-fusion-review.md)
   - C108新57场在C105-T8/V48组件之外，固定纯物理残差只在双支撑点启用，其余Mean。全域MAE/P99由2.668/22.674降至2.345/22.290km，启用组P99由15.573升至15.705km。仍有25点增害>1km，最大11.410km，小间距启用的6个比较相对向量MAE由1.098升至1.621km，指派不变。保留Mean，C111先查尾部和近距几何的事前资格。 [C108报告](reports/2026-11-12-tmtaff-dual-position-fallback-review.md)
   - C109原生参考相邻区间，固定尾部最近供给同船6/6在发布CSV中位置割线已远超两端速度字段，5/6经统一Q仍可见。4393/0原59秒跨33.813km，割线573.105m/s、原字段9.723/8.643m/s。精确上游步骤未知。C112追母AIS、公开生成资产及参考字段写入，缺上游时先设计扩大人口和物理一致性诊断。C94仍为同V300/WS/目标/Q的四属性内部对照。 [C109报告](reports/2026-11-12-raw-reference-speed-origin-review.md) · [C94四属性](reports/2026-11-12-v300-position-far-review.md)
   - C107–C109回流轮定向核读13篇原文相关段落并核TMTAFF表3。TFWFB及MHT-EnKF的MTAD适配参数、传统基线缺测掩码，以及原Level、格宽、场景名单、时窗、配准和mESD实现仍未取得。C107–C109已回矩阵，C110–C112待筛。152规定独审待收尾，PTFM/DeepAF继续作融合补充。 [本轮矩阵](/Users/a1/Projects/biye-ronghe/docs/surveys/2026-10-26_mtad_actual_use/reassessment_20261027.md#mode-c-244) · [作者协议](/Users/a1/Projects/biye-ronghe/docs/surveys/2026-10-27_expanded_search/offline_fusion_support_reference_sources_20261112.md)
   - 2025–2026文献轮定向核读21项去重来源，其中19项以学习为研究方法，2项为物理关联/融合控制。Mode C2.45将C110–C112细化为C113–C117。C113已拟定固定池、两外锚分时核验的零训练诊断；C114保留同输入整块学习评分与接入供给风险；C115优先对固定位置残差核尾部与近距几何，C116另查物理时间和来源的增量，C117继续核参考字段来源。当前只到文献、矩阵及诊断设计。 [Mode C2.45](/Users/a1/Projects/biye-ronghe/docs/surveys/2026-10-26_mtad_actual_use/reassessment_20261027.md#mode-c-245) · [C113事前设计](proposals/offline_two_anchor_time_validation_20261112.md)
2. **在线 C-ON33/35 完成稀疏首报的邻轨证据诊断和写入留场。** 11 开发场与 22 个新留场按共享 A 组件隔离，12 个留场组件此前有其他 C-ON32 场。195 个稀疏首报有 6 个开放真边入射，均无双侧共时邻轨。原几何／8 km 宽门／原见证及两种写入臂当刻正确 42／48／42，错接均 5，查询端未见真对侧的错接均 2。 [两阶段 Record/Squeeze](reports/2026-11-12-online-neighbor-witness.md) · [在线 Mode C v30](/Users/a1/Projects/biye-ronghe/docs/surveys/2026-10-26_mtad_actual_use/online_causal_replay_mode_c_20261111.md#v30-见证写入留场回流)
   - 冲突禁写拦下 478 假、30 真样本，关联和延迟指标与现有见证逐场一致。正证才写在 11 个稀疏场等同纯等待，较现有见证少 784 真、多 44 假事件，主稀疏首次门外 10 个更晚、无提前。保留原几何为工作点，宽门和原见证作对照；后续设计须有能增加独立证据覆盖的来源，长期状态反馈另评。 [前轮固定环确认](reports/2026-11-12-online-expansion-confirmation.md) · [原边排序与错链证据](reports/2026-11-12-online-nemot-association-diagnostic.md)
   - C-ON34核实MTAD论文的Pd针对整轨抽取，V4评分说明2秒发送片、V3归档更新为10秒。完整空扫、物理可见性、收报时钟仍缺。CD-PMBM/TM-PMBM保留为条件系统候选，DetectorContext作者PR可作机制参照。 [报告生成与发送协议](data_contract.md) · [公开代码迁移筛选](reports/2026-11-12-online-fusion-transfer-review.md)
   - C-ON28在原28场找到成组改配的空间。114个反超仍错段中100个可事后强制2假换2真，17个起点真边已更低的长段中10个须2假换1真。两类动作在原目标均增代价。 [原生错链Record/Squeeze](reports/2026-11-12-online-competing-reports.md)
   - C-ON36在48个新留场／31组件完成固定原图五臂。相对同刻端点，等量区间少7,458原生错对秒，完整长错段多3,980对秒。留空区间少585原生错对秒，原长段只去掉2.853对秒。16稀疏场原8个长段均完整保留，1,424首报伙伴逐项相同。当前保留原KF为工作基线。 [留场Record/Squeeze](reports/2026-11-12-online-interval-assignment.md) · [Mode C v35](/Users/a1/Projects/biye-ronghe/docs/surveys/2026-10-26_mtad_actual_use/online_causal_replay_mode_c_20261111.md#v35-留场的长错段与配对留存)
   - C-ON37在原28场的482组冻结分歧中，用同一时刻屏蔽和实报比较。首个涉事报告43组变对／164变错，严格晚到30／56，稀疏13／26。稀疏60个短段间隙中42个后来有新报，12个出现撤边方向翻转，剩余43.204原错对秒。 [间隙Record/Squeeze](reports/2026-11-12-online-hypothesis-gap.md) · [Mode C v37](/Users/a1/Projects/biye-ronghe/docs/surveys/2026-10-26_mtad_actual_use/online_causal_replay_mode_c_20261111.md#v37-新报的区分度和长错段覆盖)
   - C-ON38沿用原图和原28场。159次支持保留翻到误撤均由单源触发，87次源内分离轴投影反向，正确翻转也有21／39次。晚到相对位移一致、连续异源两批一致相对同刻简单对照净真边+38／+23，较继续保留为−38／−54，稀疏−4／−20。两者在稀疏长错间隙中方向正确的信号后余69.392／33.218对秒，属触及量。 [相对运动Record/Squeeze](reports/2026-11-12-online-relative-motion.md) · [Mode C v39](/Users/a1/Projects/biye-ronghe/docs/surveys/2026-10-26_mtad_actual_use/online_causal_replay_mode_c_20261111.md#v39-单源翻转与一致性的取舍)
   - C-ON39在原28场冻结风险窗拆时钟和退出。稀疏236组里9个严格窗内新共时、222个原窗结束当刻或以后、5个60秒内无。222个里202个由KF接回结束、6个原候选退出，首次新共时均新鲜。60个稀疏目标间隙中56个由KF先接回假边，首次共时仍全部合法。邻轨见证起点／晚到在稀疏只有3／6组，留出误差上升，60目标间隙无合格。保留原KF和I22作对照、隔离新场待独立证据。后续优先寻找在接回前可用且可检验的非共用状态，避免把延长失鲜期或同源共量直接当修复。 [支撑与邻轨Record/Squeeze](reports/2026-11-12-online-shared-innovation.md) · [Mode C v41](/Users/a1/Projects/biye-ronghe/docs/surveys/2026-10-26_mtad_actual_use/online_causal_replay_mode_c_20261111.md#v41-支撑迟到和邻轨证据回流)
   - 个体历史与 HyperSSM 保留为在线预测对照，TSA-cTFER 同数据作者条件与复现缺项见[原文核对](/Users/a1/Projects/biye-ronghe/docs/surveys/2026-10-26_mtad_actual_use/online_tsactfer_protocol_20261112.md)。报文触发预测和关联，1 秒网格仅供可比评分。独立位置参考和真实网络到达仍缺。代码在 [online_observation](../online_observation/README.md) 独立维护。当前无待执行在线合同。
3. **两线采用同一数据资产与定义，独立实现和验收。** 离线优先精度、覆盖与尾部，在线比较同条件下的精度、延迟和规模。各自固定输入权限、状态时刻、模型选取和输出人口，各自发布有具体输入、训练量、终点和停止条件的独立READY。
4. 196已完成Record→Squeeze→Mode C2.10/F306/E209、SR4.11/OD1.24；区分“回O避害”和“保MC救回”的动作监督/特征条件仅保留为候选，交叉180反例，待对应任务和具体Idea成立再执行。M123仍AI，主eligible O/MC裁剪均0，不按V后验设margin/时间阈值，不自动重训、扩V300或改默认。[196报告](reports/2026-11-03-action-conditioned-risk-validation-review.md)、[固定plan1.1](proposals/action_conditioned_risk_validation_20261102.md)。192密度解释、190拟合端点、194粗阶段门控当前配方已结案，既有反例/未采纳依据保留，不用加步/seed续旧试验。
5. [11月5日整链计时](reports/2026-11-05-pipeline-online-timing.md)作为工程证据保留：72条记录齐备，但无在线精度/长期吞吐/P95。87片段prefix下PTFM约.895–1.004s，我们HARD约1.974–2.301s/STRUCT约2.234–9.931s；当前全量重算不能承诺中大场景1Hz，不能泛称比PTFM快。2s可以在线，串行吞吐与结果延迟分别评价；当前无加速优化READY或在线精度保证。

## 研究材料与推导入口

资料库全文在本机 `/Users/a1/Projects/biye-ronghe`，不在本GitHub仓内。下列索引只导航，完整矩阵、来源分析、Idea和否证原文保留。

|材料|入口|
|---|---|
|全局问题—现象—机制—证据|[Mode C2.45](/Users/a1/Projects/biye-ronghe/docs/surveys/2026-10-26_mtad_actual_use/reassessment_20261027.md#mode-c-245)，P14/F391/M175/E421/C117|
|在线字段、相对偏差与逐秒状态|[在线 Mode C](/Users/a1/Projects/biye-ronghe/docs/surveys/2026-10-26_mtad_actual_use/online_causal_replay_mode_c_20261111.md)，含首报、历史、拒配与留场验证回流|
|多源可信度/贡献条件|[多源可靠性矩阵](/Users/a1/Projects/biye-ronghe/docs/surveys/2026-10-27_expanded_search/multisource_reliability_matrix_20261030.md)|
|选源与集合读出|[SR矩阵](/Users/a1/Projects/biye-ronghe/docs/surveys/2026-10-27_expanded_search/selection_readout_matrix_20261030.md)，196实际回流至v4.11/180格|
|观测驱动与状态恢复|[OD矩阵](/Users/a1/Projects/biye-ronghe/docs/surveys/2026-10-27_expanded_search/observation_driven_matrix_20261030.md)，196实际回流至v1.24/96格|
|逐篇正文/补充/代码阅读范围|[深入阅读审计](/Users/a1/Projects/biye-ronghe/docs/surveys/2026-10-27_expanded_search/deepread_coverage_100_20261028.md)，充分109＝72领域/相近＋37跨域；不是作者复现|
|来源原件与连接|[资料库入口](/Users/a1/Projects/biye-ronghe/docs/README.md)、[FlowState完整来源](/Users/a1/Projects/biye-ronghe/docs/surveys/2026-10-27_expanded_search/flowstate_icml2026_full_source_analysis_20261101.md)|
|Idea与否证/Record与Squeeze|[提案索引](proposals/README.md)、[报告索引](reports/README.md)；未执行、阴性和失败材料同样保留|
|历史推导与取舍|[完整过程2026-11-02](history/2026-11-02-research-process.md)、[2026-10-30](history/2026-10-30-research-process.md)、[历史摘要](history/README.md)|

## 判断边界

- 正式Level1 T2100/V300/封存测试600；T/V共享6053个原A键，V最大两组件占63%，不能称身份独立或确认性泛化。遵守[数据约定](data_contract.md)与[指标定义](mtad_data_protocols.md)，不删难点、不混ME/MAE、raw/decoded或各自/共同人口；作者ESD实现仍未知。
- reference生成、物理准确性和长尾来源未查清；三场原链只能支持所查案例。[原链](reports/2026-10-27-three-case-chain-review.md)、[CSV核验](reports/2026-10-27-raw-case-chain-review.md)。当前后端位置试验速航沿用Mean，不宣四属性学习增量、真实K≥3泛化或在线端到端。
- 132完成1000次GPU更新后被oomd终止，用户要求保留现场，管理员日志缺失；不重启/重训。[失败记录](reports/2026-10-29-ordered-input-continuation-failure.md)。107/113及其他结案任务不续旧额度；执行边界见交接。
