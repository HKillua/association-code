# C133 · 借用位置前的原生供体证据

## Record

`EXPERIMENT_RECORD REC-OFFLINE-NATIVE-DONOR-EVIDENCE-20261113`，verified。消费 Mode C2.51、C130 整窗保护、F419–F420 和 C133 冻结合同。仅用 MTAD Level1 T_core 原 177 场／4 组件，固定 C130 的 4,133 个接收槽、每槽同一供体、REG_MIX 分区、首认领与每场 J 预算。六臂各实际替换 2,090 个位置，保持全部 290 个场×供体组的整窗四属性、其他位置和原速航。85 个有候选场每次各读一次原生量测。所有动作冻结后才读 C151 评价缓存，无新数据、原标签档、原参考档、V、封存、模型前向或更新。两次经 Campus SSH MCP 执行，科学与 worker 均退出 0。 [冻结设计](../proposals/fusion_native_donor_evidence_20261113.md) · [终态回执](../../HandOver/S2L-20261113-c133.md)

首次入口在借用时刻的原生括距超过 120 秒时未保留实际行号。修正入口 R1 先保留这些行供排除，并以缓存里的实际借用位置比较原生位置。私有只读审计确认，本池 4,133 槽均未出现 >120 秒实际括距，六臂动作及证据判定逐项相同。两次原件都保留，表中采用 R1。 [原件](</Users/a1/Projects/biye-ronghe/offline_observation/docs/evidence/20261113_fusion_native_donor_evidence>) · [动作审计](</Users/a1/Projects/biye-ronghe/offline_observation/docs/evidence/20261113_fusion_native_donor_evidence/r1/decision-audit.json>)

全域共同有效点为纬度 123,051、经度 123,056、速度 123,044、航向 122,934，球面位置交集 123,051。可评动作为有双坐标参考且落在首认领输出上的替换。原尾部以 KEEP 全域 P99（17,985.623 m）固定，非尾部增害以相对 KEEP 的误差增量 >250 m 计。

|规则|实际／可评|实际同源／跨源|真同船／异船|无真供体|同船脱尾／新进尾|非尾部增害（同船子集）|位置 MAE／P99，m|
|---|---:|---:|---:|---:|---:|---:|---:|
|原混合 J|2,090／1,443|1,293／797|1,344／99|94|380／0|248（185）|1,626.967／13,638.911|
|同源优先|2,090／1,456|1,906／184|1,398／58|58|395／2|98（60）|1,611.267／13,198.271|
|原生身份|2,090／1,356|1,430／660|1,297／59|56|414／1|74（67）|1,610.557／13,069.095|
|原生位置|2,090／1,350|1,311／779|1,272／78|75|332／1|222（175）|1,640.933／14,243.246|
|原生联合|2,090／1,350|1,383／707|1,266／84|81|362／1|161（109）|1,628.014／13,807.155|
|逐场按 J 匹配来源，再联合|2,090／1,371|1,293／797|1,276／95|90|355／1|214（162）|1,632.335／13,906.180|

## Squeeze

`SQUEEZE_REPORT SQ-OFFLINE-NATIVE-DONOR-EVIDENCE-20261113`，verified。公开逐点、逐场及全域 MAE 独立复算 161,659 项通过；Campus 从私有缓存重建全域位置误差，KEEP 和六臂 P99 均一致。 [R1公开复算](</Users/a1/Projects/biye-ronghe/offline_observation/docs/evidence/20261113_fusion_native_donor_evidence/r1/public-recomputation.json>) · [全域位置复算](</Users/a1/Projects/biye-ronghe/offline_observation/docs/evidence/20261113_fusion_native_donor_evidence/r1/position-metric-audit.json>)

- **DE，错接和位置增害有两个分母。** J 的 99 次错接跨 11 场和两个组件，全部通过旧互选，87 次跨源。94 次所在的保首候选集合没有在该 Q 活跃的真同船节点；余 5 次全在场 185。248 个非尾部增害跨 47 场，其中 185 个来自真同船；238 个跨源增害中 183 个来自真同船。原生身份反证覆盖 58／99 次错接、56／94 次无真供体、215／248 次非尾部增害和 11／380 个真脱尾。原生位置反证对应 61／99、59／94、141／248 和 161／380，显示它会压低许多原本有益的动作。联合反证覆盖 78／99 次错接、232／248 个增害，也覆盖 165／380 个真脱尾。 [C130回溯](</Users/a1/Projects/biye-ronghe/offline_observation/docs/evidence/20261113_fusion_supplier_matched_activation/trajectory/c133-prior-j-audit.json>) · [C133原生证据分层](</Users/a1/Projects/biye-ronghe/offline_observation/docs/evidence/20261113_fusion_native_donor_evidence/r1/public-recomputation.json>)
- **DE，现有身份反证的判别能力有明确边界。** J 的 1,443 个可评动作中，309 个有双时刻或端点桥的身份检验资格，309 个全部越过 1 km，含 251 个真同船动作。按来源，同源可检／不可检分别错 6／78、6／874；跨源分别错 52／231、35／260。全池有 825 个身份可检槽，其中 801 个越过阈值。原生时段能指出应优先审查的风险人口，当前固定阈值对这批 J 可评动作没有展示额外判别力。短时供体复算与接收端外推是代理。双方可能共享既有数据和来源误差，端点桥允许复用供体端点。 [全池资格审计](</Users/a1/Projects/biye-ronghe/offline_observation/docs/evidence/20261113_fusion_native_donor_evidence/r1/identity-availability-audit.json>)
- **DE，原生身份优先筛选值得继续拆。** 相对 J，同船脱尾 +34，异船 −40，非尾部增害 −174，P99 降 569.817 m。两个有错接的组件分别少 25、15 次错接，多 25、9 个真脱尾；三个组件的非尾部增害下降，另一个持平。换掉的 261 个可评 J 动作含 58 错、184 害、2 真脱尾；换入 174 个可评动作含 18 错、10 害、36 真脱尾。错接率随可评人口变动仍由 6.86% 降至 4.35%。共 1,182 个可评共同动作，位置逐项一致。同源优先达到 58 错及 395 真脱尾，原生身份相对它多 1 错、多 19 个真脱尾、少 24 个非尾部增害，P99 再低 129.176 m；其中同船增害多 7 个。两个对照都出现新进尾部点。
- **DE/AI，预设联合臂未过门。** 联合在全域少错和少增害，同时少 18 个真脱尾，P99 升 168.244 m。逐场匹配来源的联合仍少 25 个真脱尾、P99 升 267.268 m。原生身份臂实际同源动作比 J 多 137 次，可评动作少 87 次，来源混合的联合消融没有单独检验身份规则。固定 KEEP 默认和 J 同预算对照。C134 待筛的单变量是固定逐场来源与原生证据资格后的连续身份距离增量，对照仅用资格的排序，并分双时刻与桥接报告。资格不足时保留原预算并报告各层数量。当前 177 场用于开发，后续确认须另行冻结 MTAD 人口和协议。

三篇 2026 原文分别提供原始报文与插值谱系的海事证据、关联与定位指标可分离的学习跟踪消融，以及共享证据下保守置信仍可损失位置精度的分布式融合反例。另复核既有实船 AIS–RTK 时差案例。 [K-N-KM](https://www.mdpi.com/1424-8220/26/7/2255) · [CDKF-Track](https://arxiv.org/html/2609.25668v1) · [Joint-CI](https://arxiv.org/html/2610.31221v1) · [原文范围与迁移笔记](</Users/a1/Projects/biye-ronghe/offline_observation/docs/surveys/2026-10-27_expanded_search/offline_c133_native_donor_sources_20261113.md>)
