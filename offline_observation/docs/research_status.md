# 离线观测研究状态

更新：2026-11-14。这里维护目标、结论和下一步；实际执行只看[交接](../HandOver/Handover.md)。结案全文见[报告索引](reports/README.md)，整理前推导和取舍已[完整保留](history/2026-11-02-research-process.md)，更早沿革见[历史入口](history/README.md)。

## 当前要解决什么

研究统一经纬度、速度、航向及时间/来源归属输入下的多源航迹关联、融合与恢复。按[AGENTS](../AGENTS.md)、[完整科研流程](research_workflow.md)和[项目skills](../../.agents/skills/README.md)，从真实问题和证据矩阵形成方案；允许更换骨架，成熟跨域方法的合理适配及明确新增价值均可研究，不追求绝对无先例。

离线以统一协议下提高 ITSA、MSTSA 和四属性融合精度为目标，同一冻结系统须承担三项。**TMTAFF 完整系统是离线整链主对照，也是优先复现对象。** 旧056→WS→位置选源作为开发对照，C145/C146 提供传统链路及指派机制对照，PTFM／DeepAF 的条件融合保留为模块证据。 [TMTAFF 原文](https://www.mdpi.com/2076-3417/16/14/7251) · [离线TMTAFF核查](reports/2026-11-10-offline-tmtaff-review.md)

用户于11月13日明确当前验证仅使用 **MTAD**。可从其他论文与代码借鉴方法、选取对照，实际训练和评价沿用MTAD已有数据权限及划分。当前无其他数据集的验证安排。

最终方案支持可变K和源换序/增减，不依赖固定源编号优先或固定双源配方；真实数据主要K2，软件K3检查不等真实泛化。用户接受合理训练侧监督，倾向少依赖后端reference系数拟合；旧骨架不构成路线约束。

## 下一步

**回到基线来源筛选与复建。** 最早可证缺口在 10 月 17–18 日。已有 TSA-cTFER 离线适配和 TMTAFF 训练入口，但未建立多套可信基线的共同协议整链实测起点。原 107/113 属于开发诊断，152 保留为固定分组的条件融合实测。旧候选队列退出当前执行。C143、C144 和先前局部候选按原证据保留。 [回退审计与重新筛选](reports/2026-11-14-offline-baseline-restart-audit.md)

1. **TMTAFF 整链主对照。** 继续核作者实际场景、配准、各阶段训练和表3 mESD 实现，保留原刊表1–3作为作者协议目标。共同协议以同一冻结 MTAD 场景、可见量测、查询和参考并列评估双方完整输出。TMTAFF 使用自身 TCM、TAM 判边成组及 TFM，报告同源和跨源关联、经纬度／速度／航向误差、覆盖、重复及位置 P99。旧096A的关联退化、旧113借用056关联、C129固定组重融合各自保留其诊断范围。作者原报、可核复现和共同协议适配分别链接证据。 [协议核验](reports/2026-11-12-tmtaff-protocol-gap-audit.md) · [固定组诊断](reports/2026-11-13-tmtaff-same-group-support-review.md)
2. **模块。** 先核 MIS-GNN、MVW-GNN、MH-T2TA 的跨源候选范围、空配、训练与评价实际单位。TSA-cTFER 适配保留关联开发对照，PTFM／DeepAF 保留固定组融合对照。各方法只在有相应输出和实测资格的项目入表。
3. **共同实测起点。** 先按原文核明确切的输入和主比较，沿用冻结 Level1 开发数据和独立封存测试。对整链固定场景、量测、查询和评分，报告 ITSA、MSTSA、覆盖、重复、四属性和位置尾部。先完成来源分析、Mode C 和基线复建 Idea，再写对应 READY。获得多套可比实测后回到组合探索和问题分析。 [数据约定](data_contract.md) · [指标协议](mtad_data_protocols.md) · [完整研究顺序](research_workflow.md#从基线到研究问题和解法)

首轮 C145 已完成 MTAD 原 V300。端点＋DTW同源微F1 .7716、跨源 .5509，WS＋Mean 为 .6952／.6024。共同位置P99降低0.871公里，覆盖由96.965%降到96.575%。纯双端CV本版退化，同刻替换DTW存在取舍。保留WS和传统两锚。作者复现和本地共同协议各自记录。 [C145全文](reports/2026-11-14-offline-baseline-common-c145.md) · [执行回执](../HandOver/S2L-20261114-c145.md)

C146 固定同源链及 DTW，跨源三个门合计损失987个真片段对，固定链的一对一容量损失1,247个。同配额全局相对贪心同时少1,007漏、1,036误，去最大仍少174漏和177误，四属性全域有效点略减，保留作后续强关联对照。显式空配全域再少551漏、多17误，共同位置MAE增加42.79米、少648个有效位置点，现工作点不替换。评分、匹配容量和融合传导后续单独立项。 [C146全文](reports/2026-11-14-offline-baseline-assignment-c146.md) · [执行回执](../HandOver/S2L-20261114-c146.md)

C147 已用自身 TCM/TAM/TFM 重新跑完与 Qhard 的原 V300 同场对照。4,439,092个新分数和旧阈值决定逐值重现，290场只有一组，位置覆盖3.356%。同组同CAP128的支持Mean位置误差好于现有TFM，速航与位置的取舍见报告。论文原表保持复现目标，现模型仅为退化适配证据。完整适配的下一资格先查训练侧关联分数、时空配准与四属性位置读出。 [C147全文](reports/2026-11-14-offline-tmtaff-own-chain-c147.md) · [作者协议核验](/Users/a1/Projects/biye-ronghe/offline_observation/docs/surveys/2026-10-27_expanded_search/offline_c147_tmtaff_author_protocol_20261114.md)

## 已有依据

<details>
<summary>开发对照与历史结果</summary>

|问题/范围|当前判断与证据|
|---|---|
|关联与覆盖|HARD保折中候选、STRUCT保均值强控；两者覆盖和跨源指标各有代价。[173结构监督](reports/2026-10-31-structured-association-review.md)、[175匹配难例](reports/2026-10-31-decision-error-mining-review.md)、[176工作点](reports/2026-10-31-decoded-coverage-review.md)完整保留。176当前网格三规则都选回原阈值，无新增收益。|
|历史候选恢复178，原V32|保留真实平均收益：HARD/STRUCT位置MAE355.760/328.916m，覆盖及关联保持各自原值。但相对对应旧base仍新增128/107个>1km增害，T尾部反证未解。[完整178记录](reports/2026-10-31-history-candidate-review.md)。|
|简单平均运动MC，原V32|191相对178 MAE−2.676/−2.386m，RMSE/P99两头降；H描述区间跨0、去最大组件H略差。原PG公里害11/0仍在，救回保70/71及42/42。保条件候选，未自动替换178。[191记录](reports/2026-11-01-mean-velocity-validation-review.md)。|
|动作条件风险195，仅T32|JM MAE391.235117/364.845327m，对178−.997387/−2.357123m，对MC−2.059982/−.132478m；MC新增公里害34/1→7/0、救回3/33全保，覆盖不变。描述CI跨0、H P95/P99及S P95略差178，原front总害仍107/71。相对O的JM均值改善，CF约保有93%/97%；相对MC则约97%/47%，不能混称。JM重选仅1混组2点；保条件候选，未替换178。[完整195及远程审查补充](reports/2026-11-02-action-conditioned-risk-review.md)。|
|冻结动作规则196，原V32|科学与规定独审均成功。CF−MC MAE+.987763/+1.053248m、RMSE/P99更差，H11旧害全解、S仍0害，但原MC救回70/42仅保24/7；JM池外只14/3点、H改善/S恶化，去最大组件均0。部分支持/明确取舍，当前不替换178/MC。[196完整Record/Squeeze](reports/2026-11-03-action-conditioned-risk-validation-review.md)。|
|阶段信息门控194，仅T32|真实39fit/4992GPU更新、44臂和规定独审完整完成。ORDER−MC MAE+.047849/+.330501m，ORDER−BAG两头区间跨0；原34/1公里害保28/1，原3/33救回仅保0/28。当前粗四块＋小网络配方不晋级，不否定整个时序方向。[194完整分析](reports/2026-11-01-temporal-gate-review.md)。|
|解释与反例|192原生片段内部近均匀采样，当前密度解释缺前提；190全段拟合端点两种运动条件均恶化。不能把阶段门控失败、纯组大错或拟合状态当噪声/参考错误已证。[192](reports/2026-11-01-sampling-measure-review.md)、[190](reports/2026-11-01-fitted-endpoint-review.md)、[189](reports/2026-11-01-drift-memory-review.md)。|
|全300场与同行比较|147/157为完整V300内部贡献基准，MAE590.760/582.808m，后者仍有稀有反转与尾部代价；178/191/194不能跨人口替换这些数字。152按原合同完成PTFM/DeepAF训练和V300评价。11月10日聚合初核：同WS全V300原147/PTFM＋Mean/DeepAF位置MAE为590.760/738.682/1361.695m；PTFM的RMSE/P99/速航更好，属于取舍，完整数组/规定收尾仍待审。比较对象仍147，非作者完整系统胜负。[初核及边界](reports/2026-11-10-offline-peer-route-review.md)。TMTAFF常值/尺度失效尚未获得可信复现或唯一根因。[147](reports/2026-10-30-source-risk-full-benchmark-review.md)、[157](reports/2026-10-30-t128-full-scope-review.md)、[152合同](../HandOver/L2S-20261030-152.md)、[TMTAFF定位](reports/2026-10-31-tfm-gradient-review.md)。|

上表各自报告T/V、有效人口、覆盖、尾部和成本；不拼两前端或不同实验冠军，不宣已胜论文方法。193导入故障与194科学结果分开，[193失败原件与原因](reports/2026-11-01-temporal-gate-startup-failure.md)完整保全。

</details>

## 研究材料与推导入口

资料库全文在本机 `/Users/a1/Projects/biye-ronghe`，不在本GitHub仓内。下列索引只导航，完整矩阵、来源分析、Idea和否证原文保留。

|材料|入口|
|---|---|
|全局问题—现象—机制—证据|[Mode C2.64](/Users/a1/Projects/biye-ronghe/offline_observation/docs/surveys/2026-10-26_mtad_actual_use/reassessment_20261027.md#mode-c-264)，P14/F470/M207/E600/C147|
|多源可信度/贡献条件|[多源可靠性矩阵](/Users/a1/Projects/biye-ronghe/offline_observation/docs/surveys/2026-10-27_expanded_search/multisource_reliability_matrix_20261030.md)|
|选源与集合读出|[SR矩阵](/Users/a1/Projects/biye-ronghe/offline_observation/docs/surveys/2026-10-27_expanded_search/selection_readout_matrix_20261030.md)，196实际回流至v4.11/180格|
|观测驱动与状态恢复|[OD矩阵](/Users/a1/Projects/biye-ronghe/offline_observation/docs/surveys/2026-10-27_expanded_search/observation_driven_matrix_20261030.md)，196实际回流至v1.24/96格|
|逐篇正文/补充/代码阅读范围|[深入阅读审计](/Users/a1/Projects/biye-ronghe/docs/surveys/2026-10-27_expanded_search/deepread_coverage_100_20261028.md)，充分109＝72领域/相近＋37跨域；不是作者复现|
|来源原件与连接|[资料库入口](/Users/a1/Projects/biye-ronghe/offline_observation/README.md)、[FlowState完整来源](/Users/a1/Projects/biye-ronghe/docs/surveys/2026-10-27_expanded_search/flowstate_icml2026_full_source_analysis_20261101.md)|
|Idea与否证/Record与Squeeze|[提案索引](proposals/README.md)、[报告索引](reports/README.md)；未执行、阴性和失败材料同样保留|
|历史推导与取舍|[完整过程2026-11-02](history/2026-11-02-research-process.md)、[2026-10-30](history/2026-10-30-research-process.md)、[历史摘要](history/README.md)|

## 判断边界

- 正式Level1 T2100/V300/封存测试600；T/V共享6053个原A键，V最大两组件占63%，不能称身份独立或确认性泛化。遵守[数据约定](data_contract.md)与[指标定义](mtad_data_protocols.md)，不删难点、不混ME/MAE、raw/decoded或各自/共同人口；作者ESD实现仍未知。
- reference生成、物理准确性和长尾来源未查清；三场原链只能支持所查案例。[原链](reports/2026-10-27-three-case-chain-review.md)、[CSV核验](reports/2026-10-27-raw-case-chain-review.md)。当前后端位置试验速航沿用Mean，不宣四属性学习增量或真实 K≥3 泛化。
- 132完成1000次GPU更新后被oomd终止，用户要求保留现场，管理员日志缺失；不重启/重训。[失败记录](reports/2026-10-29-ordered-input-continuation-failure.md)。107/113及其他结案任务不续旧额度；执行边界见交接。
