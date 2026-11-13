# 当前研究状态

更新：2026-11-14。这里维护目标、结论和下一步；实际执行只看[交接](../HandOver/Handover.md)。结案全文见[报告索引](reports/README.md)，整理前推导和取舍已[完整保留](history/2026-11-02-research-process.md)，更早沿革见[历史入口](history/README.md)。

## 当前要解决什么

研究统一经纬度、速度、航向及时间/来源归属输入下的多源航迹关联、融合与恢复。按[AGENTS](../AGENTS.md)、[完整科研流程](research_workflow.md)和[项目skills](../.agents/skills/README.md)，从真实问题和证据矩阵形成方案；允许更换骨架，成熟跨域方法的合理适配及明确新增价值均可研究，不追求绝对无先例。

用户于11月10日进一步明确：**离线观测和在线观测都要持续推进**，替代此前“先核清差距，再定主线”的安排。两条路线分别建立可信对照与验收标准，不等其中一条完成才开展另一条。离线路线以作者协议或明确统一协议下超过可信强baseline的主要精度为目标，速度作次要成本；在线路线则须在因果输入、持续更新条件下比较精度、延迟和处理规模。用户 11 月 12 日进一步明确无需固定 1 Hz，只要解决在线预测的实际问题。输出节奏与最大允许结果延迟尚未指定。[路线与对照审查](reports/2026-11-10-offline-online-route-review.md)保存原论文条件、152既有聚合及在线评测缺口；不能将当前任务适配数值称为已达到原论文水平，也不继续默认以196局部修补或离线加速推进。

离线首要对照明确为 **TMTAFF 完整系统**，同一个冻结版本须在ITSA、MSTSA和四属性融合上同时领先。用户进一步明确离线优先追求精度，允许重选整体方法。原生点与片段图关联、组级全窗状态和有限分区候选进入重新筛选，原056→WS→位置选源作为开发对照。位置改进和PTFM／DeepAF条件融合比较继续保留为内部证据与补充对照。现有完整组合的关联和四属性差距、作者协议边界及验收条件见[离线TMTAFF核查](reports/2026-11-10-offline-tmtaff-review.md)。

用户于11月13日明确当前验证仅使用 **MTAD**，适用于在线和离线。可从其他论文与代码借鉴方法、选取对照，实际训练和评价沿用MTAD已有数据权限及划分。当前无其他数据集的验证安排。

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
|全300场与同行比较|147/157为完整V300内部贡献基准，MAE590.760/582.808m，后者仍有稀有反转与尾部代价；178/191/194不能跨人口替换这些数字。152按原合同完成PTFM/DeepAF训练和V300评价。11月10日聚合初核：同WS全V300原147/PTFM＋Mean/DeepAF位置MAE为590.760/738.682/1361.695m；PTFM的RMSE/P99/速航更好，属于取舍，完整数组/规定收尾仍待审。比较对象仍147，非作者完整系统胜负。[初核及边界](reports/2026-11-10-offline-online-route-review.md)。TMTAFF常值/尺度失效尚未获得可信复现或唯一根因。[147](reports/2026-10-30-source-risk-full-benchmark-review.md)、[157](reports/2026-10-30-t128-full-scope-review.md)、[152合同](../HandOver/L2S-20261030-152.md)、[TMTAFF定位](reports/2026-10-31-tfm-gradient-review.md)。|

上表各自报告T/V、有效人口、覆盖、尾部和成本；不拼两前端或不同实验冠军，不宣已胜论文方法。193导入故障与194科学结果分开，[193失败原件与原因](reports/2026-11-01-temporal-gate-startup-failure.md)完整保全。

## 下一步

1. **离线继续建立高精度关联与可信 TMTAFF 全系统对照。** [C95](reports/2026-11-12-association-negative-coverage-review.md)同预算替换一半旧尾部负例后，混合BCE跨源90%召回FP由C92复现的1,196,033降至105,720，原056为44,754。V最终跨源FN/FP相对原WS由5,888/8,085降到4,291/4,372，held亦双降。混合BCE丢103个原代表和35,645个同船支持槽，四新臂未过总门。加排序对两种采样的跨源效果相反，当前混合BCE保留为候选分区。
   - C138只用MTAD原177场，固定4,133槽、逐场六格配额和每臂2,090次。118报文槽归112簇，90个可评去重代表78同号。原始方向换3次，较C137撤出上界错81→82、非尾害248→249、相对上界新增2个换入增害，真脱尾380、P99 13.651km和四属性覆盖保持。缓冲零换逐点重现稳定保护，P99 13.659km。收尾点位借用换槽，保留KEEP默认和J对照。 [C138](reports/2026-11-14-fusion-near-report-review.md) · [C137](reports/2026-11-13-fusion-swap-opportunity-review.md)
   - **关联 C139。** MTAD拟合1,748场／76组件中，1,912块事后可同时补漏、去假及保原真支持。双边有两次独立原生区间且同块有安全与有害候选的偏好块三折63／11／0，去最大后13块，放开原生门也仅16块。按合同原056 J复核，去最大无逆序或同分。停止该分区偏好训练配方，拟合更新0。 [C139终态](../HandOver/S2L-20261114-c139.md)
   - **关联候选 C141。** 在相同作用域和切口预算下，以同源近时报文持续分离提出新完整分区，对照旧分数和仅原生时间重叠的切口。先核去最大后是否新增能补漏、去假并保住原同船供给的分区。 [C141](proposals/association_native_separation_candidates_20261114.md)
   - **融合 C140。** 固定 MTAD 原177场的关联组和原生输入，四属性覆盖及速航保持。全域位置 MAE/P99 为支持Mean 1,772.182m/17.986km、等权819.088m/16.275km、区间分权1,605.879m/16.267km。等权对Mean在四组件全域均进，但1,683个真实内缺口P99从8.567升至9.743km。区间分权对等权仅降全域P99 7.97m，超过250m的增害77,043点，结束该权重配方。保留等权强对照。 [C140](reports/2026-11-14-fusion-interval-smoothing-review.md)
   - **融合候选 C142。** 固定组、原生输入与对称位置锚，在同一估计器比较同源区间的绝对残差和去均值运动残差。先盘点不共用原生报文的双块资格，比较原等权，查P99、增害和近距两船，真实缺口以等权回退做一致性检查。 [C142](proposals/fusion_source_offset_motion_20261114.md)
   - 双船位置沿用MTAD的C125证据边界，当前缺独立的近距相对方向参照。Autoferry的C128已归历史，后续C131外部验证暂停。 [C125](reports/2026-11-13-tmtaff-native-relative-witness-review.md) · [C128历史记录](reports/2026-11-13-autoferry-dual-gnss-eligibility-review.md)
   - C129在V300筛出61,097个同首组且逐节点支撑/最近观测时距一致的共同点，C94主臂位置MAE638.891m、历史113 54,498.162m。另4场真正固定WS及同一CAP128输入，113旧权重复现后，历史TFM相对Mean在1,133点的位置MAE 1.101→41.052km，速度23.944→8.425m/s，位置退化不能由分区差独自解释。C132优先查位置读出尺度与速航的异向收益。 [C129](reports/2026-11-13-tmtaff-same-group-support-review.md) · [C126](reports/2026-11-13-tmtaff-c94-paired-review.md)
   - Mode C2.58回填C139/C140结果、2026原文和后继关联、融合候选。TMTAFF实际Level、表3场名单、时窗、格宽、配准、mESD实现和母AIS仍缺。历史113未完成TAM，152规定独审待收尾。 [本轮矩阵](/Users/a1/Projects/biye-ronghe/docs/surveys/2026-10-26_mtad_actual_use/reassessment_20261027.md#mode-c-258) · [原表协议](</Users/a1/Projects/biye-ronghe/docs/surveys/2026-10-27_expanded_search/offline_c126_tmtaff_table3_protocol_20261113.md>)
2. **在线 C-ON33/35 完成稀疏首报的邻轨证据诊断和写入留场。** 11 开发场与 22 个新留场按共享 A 组件隔离，12 个留场组件此前有其他 C-ON32 场。195 个稀疏首报有 6 个开放真边入射，均无双侧共时邻轨。原几何／8 km 宽门／原见证及两种写入臂当刻正确 42／48／42，错接均 5，查询端未见真对侧的错接均 2。 [两阶段 Record/Squeeze](reports/2026-11-12-online-neighbor-witness.md) · [在线 Mode C v30](/Users/a1/Projects/biye-ronghe/docs/surveys/2026-10-26_mtad_actual_use/online_causal_replay_mode_c_20261111.md#v30-见证写入留场回流)
   - 冲突禁写拦下 478 假、30 真样本，关联和延迟指标与现有见证逐场一致。正证才写在 11 个稀疏场等同纯等待，较现有见证少 784 真、多 44 假事件，主稀疏首次门外 10 个更晚、无提前。保留原几何为工作点，宽门和原见证作对照；后续设计须有能增加独立证据覆盖的来源，长期状态反馈另评。 [前轮固定环确认](reports/2026-11-12-online-expansion-confirmation.md) · [原边排序与错链证据](reports/2026-11-12-online-nemot-association-diagnostic.md)
   - C-ON34核实MTAD论文的Pd针对整轨抽取，V4评分说明2秒发送片、V3归档更新为10秒。完整空扫、物理可见性、收报时钟仍缺。CD-PMBM/TM-PMBM保留为条件系统候选，DetectorContext作者PR可作机制参照。 [报告生成与发送协议](data_contract.md) · [公开代码迁移筛选](reports/2026-11-12-online-fusion-transfer-review.md)
   - C-ON28在原28场找到成组改配的空间。114个反超仍错段中100个可事后强制2假换2真，17个起点真边已更低的长段中10个须2假换1真。两类动作在原目标均增代价。 [原生错链Record/Squeeze](reports/2026-11-12-online-competing-reports.md)
   - C-ON36在48个新留场／31组件完成固定原图五臂。相对同刻端点，等量区间少7,458原生错对秒，完整长错段多3,980对秒。留空区间少585原生错对秒，原长段只去掉2.853对秒。16稀疏场原8个长段均完整保留，1,424首报伙伴逐项相同。当前保留原KF为工作基线。 [留场Record/Squeeze](reports/2026-11-12-online-interval-assignment.md) · [Mode C v35](/Users/a1/Projects/biye-ronghe/docs/surveys/2026-10-26_mtad_actual_use/online_causal_replay_mode_c_20261111.md#v35-留场的长错段与配对留存)
   - C-ON37在原28场的482组冻结分歧中，用同一时刻屏蔽和实报比较。首个涉事报告43组变对／164变错，严格晚到30／56，稀疏13／26。稀疏60个短段间隙中42个后来有新报，12个出现撤边方向翻转，剩余43.204原错对秒。 [间隙Record/Squeeze](reports/2026-11-12-online-hypothesis-gap.md) · [Mode C v37](/Users/a1/Projects/biye-ronghe/docs/surveys/2026-10-26_mtad_actual_use/online_causal_replay_mode_c_20261111.md#v37-新报的区分度和长错段覆盖)
   - C-ON38沿用原图和原28场。159次支持保留翻到误撤均由单源触发，87次源内分离轴投影反向，正确翻转也有21／39次。晚到相对位移一致、连续异源两批一致相对同刻简单对照净真边+38／+23，较继续保留为−38／−54，稀疏−4／−20。两者在稀疏长错间隙中方向正确的信号后余69.392／33.218对秒，属触及量。 [相对运动Record/Squeeze](reports/2026-11-12-online-relative-motion.md) · [Mode C v39](/Users/a1/Projects/biye-ronghe/docs/surveys/2026-10-26_mtad_actual_use/online_causal_replay_mode_c_20261111.md#v39-单源翻转与一致性的取舍)
   - C-ON39在原28场冻结风险窗拆时钟和退出。稀疏236组里9个严格窗内新共时、222个原窗结束当刻或以后、5个60秒内无。222个里202个由KF接回结束、6个原候选退出，首次新共时均新鲜。60个稀疏目标间隙中56个由KF先接回假边，首次共时仍全部合法。邻轨见证起点／晚到在稀疏只有3／6组，留出误差上升，60目标间隙无合格。保留原KF和I22作对照、隔离新场待独立证据。后续优先寻找在接回前可用且可检验的非共用状态，避免把延长失鲜期或同源共量直接当修复。 [支撑与邻轨Record/Squeeze](reports/2026-11-12-online-shared-innovation.md) · [Mode C v41](/Users/a1/Projects/biye-ronghe/docs/surveys/2026-10-26_mtad_actual_use/online_causal_replay_mode_c_20261111.md#v41-支撑迟到和邻轨证据回流)
   - C-ON40在原28场的56焦点间隙（28组、2组件）核异步单报。最后严格可用高斯分差识别9／28组，误拦87／142组正确接回；同刻高斯屏蔽为10／28、65／142，共用方差为10／28、89／142。接回批高斯16／28、36／142，提前量为0。跨源真边归一创新在≤10秒／10–30秒档超标61.9%／64.2%，原源内方差无法解释跨源残差。隔离新场未启用。若后续能提出可检验的跨源误差参照或原接回前的独立支持，再定同延迟试验。 [异步报告Record/Squeeze](reports/2026-11-12-online-async-report.md) · [Mode C v43](/Users/a1/Projects/biye-ronghe/docs/surveys/2026-10-26_mtad_actual_use/online_causal_replay_mode_c_20261111.md#v43-异步提前证据回流)
   - C-ON41在原28场核20个换源目标。跨源真边残差1.62→1.70 km，源内0.325→0.255 km，按组件和方向分层均值能量占97.7%。同组件正确接回也有同量级偏移。目标最新18组有单锚，严格接回前无双锚。事后单锚使18组真边RMSE 1.62→0.41 km，命中7→13，18组仅来自2个外部锚；全76正确换源按最近单锚误拦38组。隔离新场未启用，单锚仅保留为需独立验证的均值参照连接。 [跨源Record/Squeeze](reports/2026-11-12-online-cross-source-calibration.md) · [Mode C v45](/Users/a1/Projects/biye-ronghe/docs/surveys/2026-10-26_mtad_actual_use/online_causal_replay_mode_c_20261111.md#v45-相对均值与单锚证据回流)
   - C-ON42只读原28场冻结图。18个有锚换源窗按两个锚对等权，真边RMSE 1.54→0.37 km。58个有锚正确接回中38个真边残差下降，其中16个假边归一化代价下降更多。单锚新增10个误拦、解除14个，8个新误拦真边仍改善。锚后实报／同刻屏蔽与共用方差一致才干预，提前命中9／20、误拦21／76；同组件6／22、两个目标组件净+3／0。跨目标均值效果保留，正确接回保护未过全人口判据，隔离新场未启用。 [单锚Record/Squeeze](reports/2026-11-13-online-single-anchor-transfer.md) · [Mode C v47](/Users/a1/Projects/biye-ronghe/docs/surveys/2026-10-26_mtad_actual_use/online_causal_replay_mode_c_20261111.md#v47-单锚跨目标与排序回流)
   - C-ON43在原28场482窗核双报证据。稀疏目标28组中15个在原接回前有两候选下端点不交的双报、6个当批才有、7个无。全482窗618次提前异源互指读出都直接复用前报。最新双实报命中13／28、误拦28／142，遮晚报12／28、23／142；要求晚报净增量为7／28、21／142，同支持落后屏蔽及单锚。隔离新场未启用。 [双报Record/Squeeze](reports/2026-11-13-online-disjoint-report-closure.md) · [Mode C v49](/Users/a1/Projects/biye-ronghe/docs/surveys/2026-10-26_mtad_actual_use/online_causal_replay_mode_c_20261111.md#v49-双报供给与预测复用回流)
   - C-ON44固定原28场，618次严格窗内互指均可剔除最后一次前报更新。晚报增量62次改向，523次真假分支明确的读出中支持真方向242→236。最新同刻有互指8／28目标、53／142正确对照。去复用且晚报增加正向分差命中2／28、误拦17／142，原预测同规则3／28、15／142，只遮前报同规则3／28、13／142；同组件2／28、5／48。该直接复用可以定位，去复用读出未过晋级条件。 [去复用Record/Squeeze](reports/2026-11-13-online-reciprocal-report-downdate.md) · [Mode C v51](/Users/a1/Projects/biye-ronghe/docs/surveys/2026-10-26_mtad_actual_use/online_causal_replay_mode_c_20261111.md#v51-单报去复用的证据回流)
   - C-ON45重核原生速航由后段位置生成，严格回放沿用遮蔽。原28场最新共同历史覆盖20／28目标、135／142正确，历史右端均未越过分歧起点。两段运动加几何7／28、11／142，净位移加几何7／28、10／142，同组件7／28、5／48对7／28、3／48。要求当刻运动增量只剩2／28、10／142。接回批有局部正确保护，提前量为零。当前去留按提前条件，留场未启用。 [历史运动Record/Squeeze](reports/2026-11-13-online-past-motion-shape.md) · [Mode C v53](/Users/a1/Projects/biye-ronghe/docs/surveys/2026-10-26_mtad_actual_use/online_causal_replay_mode_c_20261111.md#v53-四端共同历史的排序回流)
   - C-ON46只读原MTAD 28场，416次完整合法单报接回。当前两段运动加几何5／28、1／142，遮当前形状坐标逐项相同，缓存加同批几何7／28、3／142，净值同为+4。更广真值明确人口为8／67、8／308，未过新留场线。接回当批未改变真实发布，持续ID、错链与首报未重测。 [当批Record/Squeeze](reports/2026-11-13-online-rejoin-batch.md) · [Mode C v55](/Users/a1/Projects/biye-ronghe/docs/surveys/2026-10-26_mtad_actual_use/online_causal_replay_mode_c_20261111.md#v55-mtad-当批形状与缓存回流)
   - C-ON47只读原MTAD 28场，新增2命中和2误拦均在同一场同一组件，缓存共同右端都约20秒旧并早于旧KF连续发布。367个有缓存接回全部通过60秒，36个右端在旧KF持续区间内，0个同时不短于I22。几何同向只保1新增命中及1新增误拦。三项组合对416次合法接回与当前逐项相同，未启用新MTAD留场。 [缓存Record/Squeeze](reports/2026-11-13-online-rejoin-cache.md) · [Mode C v57](/Users/a1/Projects/biye-ronghe/docs/surveys/2026-10-26_mtad_actual_use/online_causal_replay_mode_c_20261111.md#v57-缓存时距与连续性回流)
   - C-ON48在原MTAD 28场复现原8／8，除候选分离和用名义联合尺度均7／8，相对间距4／8；8对三端时距同档。两次缓存新增命中真边残差均升，假边升得更多。主修正臂5／28、1／142与当前相同，广义8／67、9／308，比当前多1误拦。未触发MTAD新留场。 [几何增量Record/Squeeze](reports/2026-11-13-online-rejoin-geometry.md) · [Mode C v59](/Users/a1/Projects/biye-ronghe/docs/surveys/2026-10-26_mtad_actual_use/online_causal_replay_mode_c_20261111.md#v59-几何尺度与真边残差回流)
   - C-ON49固定原 MTAD 28 场。131 个真边变远、121 个变近，各 62 个有严格批前历史支撑，严格排除假边／真边分别为 18／12、9／26。两次缓存新增命中只确认一次，对侧支撑全部与缓存重叠。稀疏主人口由当前 5／28、1／142 变为 6／28、1／142，广义由 8／67、8／308 变为 9／67、9／308，新增正确动作仅一共享 A 组件。未启用新 MTAD 留场。 [对侧历史Record/Squeeze](reports/2026-11-13-online-rejoin-witness.md) · [Mode C v61](/Users/a1/Projects/biye-ronghe/docs/surveys/2026-10-26_mtad_actual_use/online_causal_replay_mode_c_20261111.md#v61-对侧历史排除回流)
   - C-ON50固定原 28 场，焦点 10 窗均为正确接回，旧位移的 18 条分歧后对侧报文中 17 条在缓存。逐报剔除后 3 窗仍有可核双侧新报，均间隔超限。416 个合法窗中 71 窗有缓存外新报，均超 45 秒上限，合格位移为零。去缓存动作逐窗与当前形状一致。未启用新 MTAD 留场。 [去缓存位移Record/Squeeze](reports/2026-11-13-online-rejoin-fresh-displacement.md) · [Mode C v63](/Users/a1/Projects/biye-ronghe/docs/surveys/2026-10-26_mtad_actual_use/online_causal_replay_mode_c_20261111.md#v63-去缓存位移的覆盖回流)
   - C-ON51在原 28 场逐点复算 71 窗。141 个约 50 秒断口均由移除 4 条连续缓存原报形成，原始相邻间隔 10.013–10.046 秒。31 个双侧窗的 62 报均可用各自原生时刻批前预测。主动作双侧命中／误拦 5／13，对当前 4／2，稀疏主 5／7 对 5／1。唯一新增命中所在组件也新增误拦，未启用新 MTAD 留场。 [异步创新Record/Squeeze](reports/2026-11-13-online-rejoin-async-innovation.md) · [Mode C v65](/Users/a1/Projects/biye-ronghe/docs/surveys/2026-10-26_mtad_actual_use/online_causal_replay_mode_c_20261111.md#v65-缓存断口与异步增量回流)
   - C-ON52沿原 28 场和 C51 双侧 31 窗。28 窗的差分贡献较大。去自身共量后，两边变远误拦留 7／8、假边变近留 3／3、唯一真命中保留。62 条报告仅 1 条有同向双邻轨供体，留出误差 218→264 米，0 个双侧支持。主动作逐窗回到当前形状，未启用新 MTAD 留场。 [共同创新Record/Squeeze](reports/2026-11-13-online-rejoin-common-innovation.md) · [Mode C v67](/Users/a1/Projects/biye-ronghe/docs/surveys/2026-10-26_mtad_actual_use/online_causal_replay_mode_c_20261111.md#v67-共同创新与留出回流)
   - C-ON53在原 28 场冻结候选、双报和接回批。7／3 次原误拦分别有 4／7、1／3 在逐报屏蔽后失去错误方向，但均无同一报同时具备前两报异向大跳。31 双报窗均可算紧邻历史，仅 8 窗两轨均有分歧后前报。主动作全弃权，双报 4／2、稀疏主 5／1，与当前逐窗相同，唯一新增真命中丢失，未启用新 MTAD 留场。 [单轨历史Record/Squeeze](reports/2026-11-14-online-rejoin-track-history.md) · [Mode C v69](/Users/a1/Projects/biye-ronghe/docs/surveys/2026-10-26_mtad_actual_use/online_causal_replay_mode_c_20261111.md#v69-单轨历史与单报遮蔽回流)
   - C-ON54原 28 场保留 5 次新报推错、5 次屏蔽几何已错和唯一真命中。62 条涉事报无接回前同轨原生后报，各轨分歧后也仅一条缓存外报。晚于两条涉事报的对侧原报按窗去重为 7 条，全部在旧缓存。吸收与跳过无真实共同支撑，主动作逐窗回到当前，双报 4／2、稀疏主 5／1，未启用新 MTAD 留场。 [后报Record/Squeeze](reports/2026-11-14-online-rejoin-next-report.md) · [Mode C v71](/Users/a1/Projects/biye-ronghe/docs/surveys/2026-10-26_mtad_actual_use/online_causal_replay_mode_c_20261111.md#v71-后报供给与同延迟对照回流)
   - C-ON55原 28 场，31 条原接回触发均进原门、KF 边代价及当前形状。遮触发坐标后固定边资格不变，13 窗缓存局部排序翻向。5 次新报推错组的触发实报吸收后 5／5 指对，同时跳过只 1／5。唯一真命中在四条实报预测路径均被触发推错。主确认双报 4／3 对当前 4／2，稀疏主同为 5／1，未启用新 MTAD 留场。 [触发报Record/Squeeze](reports/2026-11-14-online-rejoin-boundary-report.md) · [Mode C v73](/Users/a1/Projects/biye-ronghe/docs/surveys/2026-10-26_mtad_actual_use/online_causal_replay_mode_c_20261111.md#v73-触发复用与预测路径回流)
   - 在线候选继续在MTAD的因果回放中筛选，沿用冻结输入权限，核持续身份、漏接、误接和提前量。Autoferry、Mariner的传感器协议作为来源笔记留档。 [外部方法来源](/Users/a1/Projects/biye-ronghe/docs/surveys/2026-10-26_mtad_actual_use/online_past_motion_shape_sources_20261113.md#外部数据能补哪项证据)
   - 个体历史与 HyperSSM 保留为在线预测对照，TSA-cTFER 同数据作者条件与复现缺项见[原文核对](/Users/a1/Projects/biye-ronghe/docs/surveys/2026-10-26_mtad_actual_use/online_tsactfer_protocol_20261112.md)。报文触发预测和关联，1 秒网格仅供可比评分。独立位置参考和真实网络到达仍缺。代码在 [online_observation](../online_observation/README.md) 独立维护。当前无待执行在线合同。
3. **两线采用同一数据资产与定义，独立实现和验收。** 离线优先精度、覆盖与尾部，在线比较同条件下的精度、延迟和规模。各自固定输入权限、状态时刻、模型选取和输出人口，各自发布有具体输入、训练量、终点和停止条件的独立READY。
4. 196已完成Record→Squeeze→Mode C2.10/F306/E209、SR4.11/OD1.24；区分“回O避害”和“保MC救回”的动作监督/特征条件仅保留为候选，交叉180反例，待对应任务和具体Idea成立再执行。M123仍AI，主eligible O/MC裁剪均0，不按V后验设margin/时间阈值，不自动重训、扩V300或改默认。[196报告](reports/2026-11-03-action-conditioned-risk-validation-review.md)、[固定plan1.1](proposals/action_conditioned_risk_validation_20261102.md)。192密度解释、190拟合端点、194粗阶段门控当前配方已结案，既有反例/未采纳依据保留，不用加步/seed续旧试验。
5. [11月5日整链计时](reports/2026-11-05-pipeline-online-timing.md)作为工程证据保留：72条记录齐备，但无在线精度/长期吞吐/P95。87片段prefix下PTFM约.895–1.004s，我们HARD约1.974–2.301s/STRUCT约2.234–9.931s；当前全量重算不能承诺中大场景1Hz，不能泛称比PTFM快。2s可以在线，串行吞吐与结果延迟分别评价；当前无加速优化READY或在线精度保证。

## 研究材料与推导入口

资料库全文在本机 `/Users/a1/Projects/biye-ronghe`，不在本GitHub仓内。下列索引只导航，完整矩阵、来源分析、Idea和否证原文保留。

|材料|入口|
|---|---|
|全局问题—现象—机制—证据|[Mode C2.58](/Users/a1/Projects/biye-ronghe/docs/surveys/2026-10-26_mtad_actual_use/reassessment_20261027.md#mode-c-258)，P14/F445/M200/E564/C142|
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
