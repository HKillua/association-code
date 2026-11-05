# 当前研究状态

更新：2026-11-05。这里维护目标、结论和下一步；实际执行只看[交接](../HandOver/Handover.md)。结案全文见[报告索引](reports/README.md)，整理前推导和取舍已[完整保留](history/2026-11-02-research-process.md)，更早沿革见[历史入口](history/README.md)。

## 当前要解决什么

研究统一经纬度、速度、航向及时间/来源归属输入下的多源航迹关联、融合与恢复。按[AGENTS](../AGENTS.md)、[完整科研流程](research_workflow.md)和[项目skills](../.agents/skills/README.md)，从真实问题和证据矩阵形成方案；允许更换骨架，成熟跨域方法的合理适配及明确新增价值均可研究，不追求绝对无先例。

最终方案支持可变K和源换序/增减，不依赖固定源编号优先或固定双源配方；真实数据主要K2，软件K3检查不等真实泛化。精度优先，轻量、在线、端到端是加分目标。用户接受合理训练侧监督及新数据集重拟合，倾向少依赖后端reference系数拟合。

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
|全300场与同行比较|147/157为完整V300内部贡献基准，MAE590.760/582.808m，后者仍有稀有反转与尾部代价；178/191/194不能跨人口替换这些数字。152按原合同完成PTFM/DeepAF训练和V300评价，完整结果待审，比较对象仍147。TMTAFF常值/尺度失效尚未获得可信复现或唯一根因。[147](reports/2026-10-30-source-risk-full-benchmark-review.md)、[157](reports/2026-10-30-t128-full-scope-review.md)、[152合同](../HandOver/L2S-20261030-152.md)、[TMTAFF定位](reports/2026-10-31-tfm-gradient-review.md)。|

上表各自报告T/V、有效人口、覆盖、尾部和成本；不拼两前端或不同实验冠军，不宣已胜论文方法。193导入故障与194科学结果分开，[193失败原件与原因](reports/2026-11-01-temporal-gate-startup-failure.md)完整保全。

## 下一步

1. 196已完成Record→Squeeze→Mode C2.10/F306/E209、SR4.11/OD1.24。当前固定O风险迁移配方没有保住MC多数救回，下一问题是区分“回O避害”与“保MC救回”的可观测条件；交discover-innovation比较动作匹配监督/特征条件的可归因设计，交叉旧180不同配方反例，另立具体Idea后执行。M123仍AI，主eligible O/MC裁剪均0，不按V后验设margin/时间阈值，不自动重训、扩V300或改默认。178/O、简单MC与CF/JM的条件取舍证据完整保留。[196报告](reports/2026-11-03-action-conditioned-risk-validation-review.md)、[固定plan1.1](proposals/action_conditioned_risk_validation_20261102.md)。
2. 输入、时序、关联、融合、输出五条路线继续按矩阵分析；关联仍以覆盖、跨源mPCA/mPFA和完整位置尾部一起判断。成熟方案先明确输入、训练量、终点和停止条件，再独立发布READY。
3. 192密度解释、190拟合端点和194粗阶段门控的当前配方已结案，不以加步/seed/窗口/β小补丁续跑；既有反例及未采纳理由继续保留。
4. 152A/B科学均已按原整数训练预算与300场V评价完成，真实exit0；下一步按原合同核完整命令/退出、人口、指标/尾部/成本及规定收尾后再落实公平比较，当前不据科学日志预宣胜负。比较对象仍147，不把178/191跨人口代入。195/196均已结案，无自动全300、动作条件重训或新seed。
5. 用户询问部署耗时已完成[同机工程计时](reports/2026-11-05-deployment-timing.md)：两个固定共享WS组、每组64query，CPU后端PTFM约6–11ms、DeepAF约.98–.99s、178/O约33–51ms。这是常驻后端小样本，包含输入特征、不含关联前端/原生motion构造；用户进一步明确要求完整应用链路与1Hz在线；已另定[三场原始量测→完整关联/恢复→序列化、连续prefix回放](../HandOver/L2S-20261105-pipeline-timing.md)，软件检查通过，真实完整计时待执行。

## 研究材料与推导入口

资料库全文在本机 `/Users/a1/Projects/biye-ronghe`，不在本GitHub仓内。下列索引只导航，完整矩阵、来源分析、Idea和否证原文保留。

|材料|入口|
|---|---|
|全局问题—现象—机制—证据|[Mode C2.10](/Users/a1/Projects/biye-ronghe/docs/surveys/2026-10-26_mtad_actual_use/reassessment_20261027.md#mode-c-210)，P14/F306/M123/E209/C69|
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
