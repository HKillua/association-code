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

1. **离线重选高精度骨架，并建立可信 TMTAFF 全系统对照。** [C88](reports/2026-11-12-external-supplier-translation-review.md)换场 16/12 组件中，187 个跨源供给仅 6 个有合格外锚且全为同船，四门均保持原 WS，跨源漏/假 43/36、同船逐时支持保持。无门 ABS 漏/假 49/37，丢 40 个同船支持槽。事后 68 个净补漏去假且保原同船支持的候选均被四门拦截，ABS 也无正分。42 个还同时严格降低跨源漏和假，集中于两场，分别出现接受和场内排序失效。外锚保留条件诊断，下一步在固定池上区分 T 侧接受校准与分区评分，用独立组件核实际输出。 [定向原文](/Users/a1/Projects/biye-ronghe/docs/surveys/2026-10-26_mtad_actual_use/offline_candidate_readout_sources_20261112.md)
   - [C89](reports/2026-11-12-v300-far-paper-bridge-review.md)完成全 V300 四属性与历史 113 的逐场目标和参考配对。远端规则保留为速度候选，全域 MAE/P99 由 1.410/28.535 降到 0.881/6.992 m/s，经纬、航向及覆盖一致。历史 113 缺已训练 TAM，旧 TFM 真组位置也退化。后续将 FAR 和强位置候选在同一冻结版本评价。[C83 近端追因](reports/2026-11-12-held-near-tail-audit.md)确认并列最大增害 22.505 m/s 源自同一次跨船取值，另外三点同船同源，0.603–3.000 秒的近端速度仍偏大。保留近端对照，供给者证据与同供给者速度估计分别检验。
   - 现有 V300 加权跨源正确率/误关联率 63.62/43.30%，TMTAFF 原表 91.38/6.87%，作者逐场和窗口未对齐。[C90-R1 真对 TFM 资格](reports/2026-11-12-tmtaff-true-pair-qualification-review.md)恢复原 Adam 后两臂各 32 步，四属性/仅位置的 MAE 36.692/36.022 km，普通 Mean 1.887 km，两臂均趋近常值、未过门。下一设计区分训练早期条件响应如何丢失，TAM 同配对尺度诊断待独立方案。152 按原合同收尾，PTFM/DeepAF 为融合补充。
2. **在线门外真边完成三场诊断和组件隔离留场。** 三个稀疏弃权场 3,824 门外真查询，宽门／共同位移／双外锚分别救回 3,774／27／0。新留场 28 场、19 组件，18 场稀疏。全体宽门救回 5,040/6,624，新增 13,740 真与 4,904 假事件边，去重 81 真／69 假。共同位移救回 29，事件边 54 真／64 假，稀疏为 54 真／2 假。双外锚救回 0。 [两阶段 Record/Squeeze](reports/2026-11-12-online-causal-recovery.md) · [冻结留场](proposals/online_causal_recovery_held_20261112.md) · [在线 Mode C v25](/Users/a1/Projects/biye-ronghe/docs/surveys/2026-10-26_mtad_actual_use/online_causal_replay_mode_c_20261111.md#v25-门外真边留场回流)
   - 136 个首次门外起点，240 秒原几何／宽门／双锚／位移确认 99／125／99／101。宽门有 26 个独有当刻确认，位移仅两例约 217–221 秒的独有确认，同一组件。保留原几何为工作点，宽门作候选可达对照。C-ON32 先核固定扩展候选里的配对确认、独立锚资格和空配证据，长期身份反馈另评。 [原边排序与错链证据](reports/2026-11-12-online-nemot-association-diagnostic.md)
   - 个体历史与 HyperSSM 保留为在线预测对照，TSA-cTFER 同数据作者条件与复现缺项见[原文核对](/Users/a1/Projects/biye-ronghe/docs/surveys/2026-10-26_mtad_actual_use/online_tsactfer_protocol_20261112.md)。报文触发预测和关联，1 秒网格仅供可比评分。独立位置参考和真实网络到达仍缺。代码在 [online_observation](../online_observation/README.md) 独立维护。当前无待执行在线合同。
3. **两线采用同一数据资产与定义，独立实现和验收。** 离线优先精度、覆盖与尾部，在线比较同条件下的精度、延迟和规模。各自固定输入权限、状态时刻、模型选取和输出人口，各自发布有具体输入、训练量、终点和停止条件的独立READY。
4. 196已完成Record→Squeeze→Mode C2.10/F306/E209、SR4.11/OD1.24；区分“回O避害”和“保MC救回”的动作监督/特征条件仅保留为候选，交叉180反例，待对应任务和具体Idea成立再执行。M123仍AI，主eligible O/MC裁剪均0，不按V后验设margin/时间阈值，不自动重训、扩V300或改默认。[196报告](reports/2026-11-03-action-conditioned-risk-validation-review.md)、[固定plan1.1](proposals/action_conditioned_risk_validation_20261102.md)。192密度解释、190拟合端点、194粗阶段门控当前配方已结案，既有反例/未采纳依据保留，不用加步/seed续旧试验。
5. [11月5日整链计时](reports/2026-11-05-pipeline-online-timing.md)作为工程证据保留：72条记录齐备，但无在线精度/长期吞吐/P95。87片段prefix下PTFM约.895–1.004s，我们HARD约1.974–2.301s/STRUCT约2.234–9.931s；当前全量重算不能承诺中大场景1Hz，不能泛称比PTFM快。2s可以在线，串行吞吐与结果延迟分别评价；当前无加速优化READY或在线精度保证。

## 研究材料与推导入口

资料库全文在本机 `/Users/a1/Projects/biye-ronghe`，不在本GitHub仓内。下列索引只导航，完整矩阵、来源分析、Idea和否证原文保留。

|材料|入口|
|---|---|
|全局问题—现象—机制—证据|[Mode C2.30](/Users/a1/Projects/biye-ronghe/docs/surveys/2026-10-26_mtad_actual_use/reassessment_20261027.md#mode-c-230)，P14/F340/M148/E293/C90|
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
