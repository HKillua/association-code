# 当前研究状态

更新：2026-11-11。这里维护目标、结论和下一步；实际执行只看[交接](../HandOver/Handover.md)。结案全文见[报告索引](reports/README.md)，整理前推导和取舍已[完整保留](history/2026-11-02-research-process.md)，更早沿革见[历史入口](history/README.md)。

## 当前要解决什么

研究统一经纬度、速度、航向及时间/来源归属输入下的多源航迹关联、融合与恢复。按[AGENTS](../AGENTS.md)、[完整科研流程](research_workflow.md)和[项目skills](../.agents/skills/README.md)，从真实问题和证据矩阵形成方案；允许更换骨架，成熟跨域方法的合理适配及明确新增价值均可研究，不追求绝对无先例。

用户于11月10日进一步明确：**离线观测和在线观测都要持续推进**，替代此前“先核清差距，再定主线”的安排。两条路线分别建立可信对照与验收标准，不等其中一条完成才开展另一条。离线路线以作者协议或明确统一协议下超过可信强baseline的主要精度为目标，速度作次要成本；在线路线则须在因果输入、持续更新条件下比较精度、延迟和处理规模。用户已指定每秒更新一次，最大允许结果延迟仍未指定。[路线与对照审查](reports/2026-11-10-offline-online-route-review.md)保存原论文条件、152既有聚合及在线评测缺口；不能将当前任务适配数值称为已达到原论文水平，也不继续默认以196局部修补或离线加速推进。

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

1. **离线重选高精度骨架，并建立可信TMTAFF全系统对照。** [C70科学与独审](reports/2026-11-11-conditional-trajectory-review.md)完成，固定池位置留出评分比同池J少11条同源、43条跨源真边，关闭当前评分配方。主前端推进[可变容量候选诊断](proposals/offline_variable_partition_20261111.md)，固定拆并动作族和原始切口预算，比旧分数与原生运动的候选可达性，几何C72后续独立验证。固定WS与真组中位置估计读出明显改善速航、位置MAE升，按[C74对照](proposals/offline_trajectory_reselection_20261110.md#c70结果后的融合读出对照)准备冻结分组和位置后的读出验证。TMTAFF优先[固定16真对TFM资格](proposals/tmtaff_component_qualification_20261111.md)，已核107端点保存源normalizer与完整数据合同，准备独立T8入口比较原四属性与位置目标。TAM同配对尺度诊断并行准备。152按原合同收尾，PTFM/DeepAF保留为融合补充对照。
2. **在线当前处于文献阅读和问题发现阶段。** 按用户明确的顺序，先逐篇读海事实测、时空配准、异步后验融合和在线评价，再用来源与项目观察建立 Mode C。首批定向复核 13 篇论文和 1 份学位论文，整理了相对偏差与时差混淆、暂失与重现时的持续身份、融合相关历史、结果实际可用时间和评价人口五组问题。继续取得最近邻联合配准及现场级联原文，比较已知条件、反例、支持范围和本项目是否有相同现象。证据充分的连接再进入创新筛选与事前设计。[逐篇来源分析](/Users/a1/Projects/biye-ronghe/docs/surveys/2026-10-26_mtad_actual_use/online_literature_analysis_20261111.md) · [在线 Mode C v3](/Users/a1/Projects/biye-ronghe/docs/surveys/2026-10-26_mtad_actual_use/online_causal_replay_mode_c_20261111.md#v3-文献扩展与问题复核)
   - 已有 8 场字段审计、3 场关联/融合诊断及相对平移、源等权的结果属于开发证据，先前的接入设想回到候选比较。严格输入、每秒发布、跨次身份和可变 K 约束沿用，最大结果延迟待定。实际推理使用 [online_observation](../online_observation/README.md) 的独立代码。当前无在线执行合同。[字段权限](data_contract.md#在线字段可用时间) · [开发证据原报告](reports/2026-11-11-online-field-and-method-review.md)
3. **两线采用同一数据资产与定义，独立实现和验收。** 离线优先精度、覆盖与尾部，在线比较同条件下的精度、延迟和规模。各自固定输入权限、状态时刻、模型选取和输出人口，各自发布有具体输入、训练量、终点和停止条件的独立READY。
4. 196已完成Record→Squeeze→Mode C2.10/F306/E209、SR4.11/OD1.24；区分“回O避害”和“保MC救回”的动作监督/特征条件仅保留为候选，交叉180反例，待对应任务和具体Idea成立再执行。M123仍AI，主eligible O/MC裁剪均0，不按V后验设margin/时间阈值，不自动重训、扩V300或改默认。[196报告](reports/2026-11-03-action-conditioned-risk-validation-review.md)、[固定plan1.1](proposals/action_conditioned_risk_validation_20261102.md)。192密度解释、190拟合端点、194粗阶段门控当前配方已结案，既有反例/未采纳依据保留，不用加步/seed续旧试验。
5. [11月5日整链计时](reports/2026-11-05-pipeline-online-timing.md)作为工程证据保留：72条记录齐备，但无在线精度/长期吞吐/P95。87片段prefix下PTFM约.895–1.004s，我们HARD约1.974–2.301s/STRUCT约2.234–9.931s；当前全量重算不能承诺中大场景1Hz，不能泛称比PTFM快。2s可以在线，串行吞吐与结果延迟分别评价；当前无加速优化READY或在线精度保证。

## 研究材料与推导入口

资料库全文在本机 `/Users/a1/Projects/biye-ronghe`，不在本GitHub仓内。下列索引只导航，完整矩阵、来源分析、Idea和否证原文保留。

|材料|入口|
|---|---|
|全局问题—现象—机制—证据|[Mode C2.14](/Users/a1/Projects/biye-ronghe/docs/surveys/2026-10-26_mtad_actual_use/reassessment_20261027.md#mode-c-214)，P14/F313/M124/E215/C74|
|在线字段、相对偏差与逐秒状态|[在线 Mode C](/Users/a1/Projects/biye-ronghe/docs/surveys/2026-10-26_mtad_actual_use/online_causal_replay_mode_c_20261111.md)，含两项资格验证回流与文献扩展|
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
