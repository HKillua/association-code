# 离线观测研究状态

更新：2026-11-14。这里维护目标、结论和下一步；实际执行只看[交接](../HandOver/Handover.md)。结案全文见[报告索引](reports/README.md)，整理前推导和取舍已[完整保留](history/2026-11-02-research-process.md)，更早沿革见[历史入口](history/README.md)。

## 当前要解决什么

研究统一经纬度、速度、航向及时间/来源归属输入下的多源航迹关联、融合与恢复。按[AGENTS](../AGENTS.md)、[完整科研流程](research_workflow.md)和[项目skills](../.agents/skills/README.md)，从真实问题和证据矩阵形成方案；允许更换骨架，成熟跨域方法的合理适配及明确新增价值均可研究，不追求绝对无先例。

离线首要对照明确为 **TMTAFF 完整系统**，同一个冻结版本须在ITSA、MSTSA和四属性融合上同时领先。用户进一步明确离线优先追求精度，允许重选整体方法。原生点与片段图关联、组级全窗状态和有限分区候选进入重新筛选，原056→WS→位置选源作为开发对照。位置改进和PTFM／DeepAF条件融合比较继续保留为内部证据与补充对照。现有完整组合的关联和四属性差距、作者协议边界及验收条件见[离线TMTAFF核查](reports/2026-11-10-offline-tmtaff-review.md)。

用户于11月13日明确当前验证仅使用 **MTAD**。可从其他论文与代码借鉴方法、选取对照，实际训练和评价沿用MTAD已有数据权限及划分。当前无其他数据集的验证安排。

最终方案支持可变K和源换序/增减，不依赖固定源编号优先或固定双源配方；真实数据主要K2，软件K3检查不等真实泛化。用户接受合理训练侧监督，倾向少依赖后端reference系数拟合；旧骨架不构成路线约束。

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
|全300场与同行比较|147/157为完整V300内部贡献基准，MAE590.760/582.808m，后者仍有稀有反转与尾部代价；178/191/194不能跨人口替换这些数字。152按原合同完成PTFM/DeepAF训练和V300评价。11月10日聚合初核：同WS全V300原147/PTFM＋Mean/DeepAF位置MAE为590.760/738.682/1361.695m；PTFM的RMSE/P99/速航更好，属于取舍，完整数组/规定收尾仍待审。比较对象仍147，非作者完整系统胜负。[初核及边界](reports/2026-11-10-offline-peer-route-review.md)。TMTAFF常值/尺度失效尚未获得可信复现或唯一根因。[147](reports/2026-10-30-source-risk-full-benchmark-review.md)、[157](reports/2026-10-30-t128-full-scope-review.md)、[152合同](../HandOver/L2S-20261030-152.md)、[TMTAFF定位](reports/2026-10-31-tfm-gradient-review.md)。|

上表各自报告T/V、有效人口、覆盖、尾部和成本；不拼两前端或不同实验冠军，不宣已胜论文方法。193导入故障与194科学结果分开，[193失败原件与原因](reports/2026-11-01-temporal-gate-startup-failure.md)完整保全。

## 下一步

1. **离线继续建立高精度关联与可信 TMTAFF 全系统对照。** [C95](reports/2026-11-12-association-negative-coverage-review.md)同预算替换一半旧尾部负例后，混合BCE跨源90%召回FP由C92复现的1,196,033降至105,720，原056为44,754。V最终跨源FN/FP相对原WS由5,888/8,085降到4,291/4,372，held亦双降。混合BCE丢103个原代表和35,645个同船支持槽，四新臂未过总门。加排序对两种采样的跨源效果相反，当前混合BCE保留为候选分区。
   - C138只用MTAD原177场，固定4,133槽、逐场六格配额和每臂2,090次。118报文槽归112簇，90个可评去重代表78同号。原始方向换3次，较C137撤出上界错81→82、非尾害248→249、相对上界新增2个换入增害，真脱尾380、P99 13.651km和四属性覆盖保持。缓冲零换逐点重现稳定保护，P99 13.659km。收尾点位借用换槽，保留KEEP默认和J对照。 [C138](reports/2026-11-14-fusion-near-report-review.md) · [C137](reports/2026-11-13-fusion-swap-opportunity-review.md)
   - **关联 C139。** MTAD拟合1,748场／76组件中，1,912块事后可同时补漏、去假及保原真支持。双边有两次独立原生区间且同块有安全与有害候选的偏好块三折63／11／0，去最大后13块，放开原生门也仅16块。按合同原056 J复核，去最大无逆序或同分。停止该分区偏好训练配方，拟合更新0。 [C139终态](../HandOver/S2L-20261114-c139.md)
   - **关联 C141。** 原MTAD拟合1,748场／76组件，三臂各15,795个匹配新增分区。旧分数／仅时间重叠／持续分离的安全块2,200／2,211／2,192，去最大组件后378／382／380。固定旧J与供给证书，三臂实际各改2块、跨源漏假各减1，同船支持无毛损，去最大后零动作。去最大后79／80／80块存在正J安全候选，原证书放行的安全块均为零。结束持续分离切口配方。 [C141](reports/2026-11-14-association-native-separation-review.md)
   - **关联候选 C143。** 固定C141候选和旧J，先查两源各有两条可比原生轨迹时，两个报文不复用时段能否一致区分直配与交换，并覆盖完整改动所需的逐Q供给。保留原证书和C104反证作对照，未覆盖的槽沿原保护。 [C143](proposals/association_paired_supplier_certificate_20261114.md)
   - **融合 C140。** 固定 MTAD 原177场的关联组和原生输入，四属性覆盖及速航保持。全域位置 MAE/P99 为支持Mean 1,772.182m/17.986km、等权819.088m/16.275km、区间分权1,605.879m/16.267km。等权对Mean在四组件全域均进，但1,683个真实内缺口P99从8.567升至9.743km。区间分权对等权仅降全域P99 7.97m，超过250m的增害77,043点，结束该权重配方。保留等权强对照。 [C140](reports/2026-11-14-fusion-interval-smoothing-review.md)
   - **融合 C142。** 原177场全部具双块及锚资格，去最大仍109场。原等权／同锚绝对／同锚相对的全域位置P99 16.275／16.547／19.902km，76,064启用点P99 4.869／5.289／11.939km，四属性覆盖保持。相对对同锚绝对71,291点增害>250m，5,580个启用近距船对交换优于直配483→1,314。结束单组去均值残差配方，等权保留强对照。 [C142](reports/2026-11-14-fusion-source-offset-motion-review.md)
   - **融合候选 C144。** 固定原等权的双船中心位置，用独有原生双块核可见的同源两船位移，先查候选可用性和源间一致，再决定是否试双船差向量约束。保持关联与原覆盖，重点看近距相对几何及全域P99。 [C144](proposals/fusion_pair_displacement_constraint_20261114.md)
   - 双船位置沿用MTAD的C125证据边界，当前缺独立的近距相对方向参照。Autoferry的C128已归历史，后续C131外部验证暂停。 [C125](reports/2026-11-13-tmtaff-native-relative-witness-review.md) · [C128历史记录](reports/2026-11-13-autoferry-dual-gnss-eligibility-review.md)
   - C129在V300筛出61,097个同首组且逐节点支撑/最近观测时距一致的共同点，C94主臂位置MAE638.891m、历史113 54,498.162m。另4场真正固定WS及同一CAP128输入，113旧权重复现后，历史TFM相对Mean在1,133点的位置MAE 1.101→41.052km，速度23.944→8.425m/s，位置退化不能由分区差独自解释。C132优先查位置读出尺度与速航的异向收益。 [C129](reports/2026-11-13-tmtaff-same-group-support-review.md) · [C126](reports/2026-11-13-tmtaff-c94-paired-review.md)
   - Mode C2.59回填C141/C142结果、十篇2026原文和后继关联、融合候选。TMTAFF实际Level、表3场名单、时窗、格宽、配准、mESD实现和母AIS仍缺。历史113未完成TAM，152规定独审待收尾。 [本轮矩阵](/Users/a1/Projects/biye-ronghe/offline_observation/docs/surveys/2026-10-26_mtad_actual_use/reassessment_20261027.md#mode-c-259) · [原表协议](</Users/a1/Projects/biye-ronghe/offline_observation/docs/surveys/2026-10-27_expanded_search/offline_c126_tmtaff_table3_protocol_20261113.md>)
2. 196已完成Record→Squeeze→Mode C2.10/F306/E209、SR4.11/OD1.24；区分“回O避害”和“保MC救回”的动作监督/特征条件仅保留为候选，交叉180反例，待对应任务和具体Idea成立再执行。M123仍AI，主eligible O/MC裁剪均0，不按V后验设margin/时间阈值，不自动重训、扩V300或改默认。[196报告](reports/2026-11-03-action-conditioned-risk-validation-review.md)、[固定plan1.1](proposals/action_conditioned_risk_validation_20261102.md)。192密度解释、190拟合端点、194粗阶段门控当前配方已结案，既有反例/未采纳依据保留，不用加步/seed续旧试验。

## 研究材料与推导入口

资料库全文在本机 `/Users/a1/Projects/biye-ronghe`，不在本GitHub仓内。下列索引只导航，完整矩阵、来源分析、Idea和否证原文保留。

|材料|入口|
|---|---|
|全局问题—现象—机制—证据|[Mode C2.59](/Users/a1/Projects/biye-ronghe/offline_observation/docs/surveys/2026-10-26_mtad_actual_use/reassessment_20261027.md#mode-c-259)，P14/F454/M202/E583/C144|
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
