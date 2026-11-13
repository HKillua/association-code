# C92 同池高分负例与关联评分

## EXPERIMENT_RECORD

`REC-OFFLINE-ASSOCIATION-TAIL-CONTRAST-20261112`，source_idea_id 为 `IDEA-OFFLINE-ASSOCIATION-HIGH-PRECISION-20261112` v1.1，消费 Mode C2.32/C92、P05/P14×F337/F346、M151、E299/E301–E309。研究问题为相同高分负例曝光下，成对排序能否提高固定召回处的精度，并在原 WS 构组后同时补漏、去假和保留同船支持。 [事前设计](../proposals/offline_high_precision_association_20261112.md) · [正式合同](../../HandOver/L2S-20261112-c92.md) · [前置合同](../../HandOver/L2S-20261112-c92-prep.md)

### 前置人口与输入

SSH MCP 盘点原 056 的 baseline/selected/final 分数文件，均仅覆盖 V300。168 保存 T128 的 775,705 个候选分数。151 的 canonical_all 缓存覆盖完整 T_core 2,100 场、27,632,584 个场内无序候选。冻结原 056 full_evidence、u1000、normalizer 和原速度单位，对其余 1,972 场的 26,856,879 个候选补前向。原 168 每场最多 64 个等距探针对齐，最大绝对差 3.815e−6，容差 1e−4。新分数另存，复用部分与旧 168 全数组逐值相等。独立盘点核全 2,100 份分数有限、候选总数、复用总数、组件互斥及完整场景并集。 [前置原件](/Users/a1/Projects/biye-ronghe/offline_observation/docs/evidence/20261112_association_tail/prep/summary.json)

在读分数和身份前按共享 A 组件冻结 fit/calibration/held 为 1,748/175/177 场，组件为 76/4/4。旧 056 已训练全 T，原 V 用于其端点和工作点，因此 held 检验新增头的留组件表现，V 为已暴露开发评价。拟合尾部的同源/跨源负边 138,198/134,865，其中同一个最大组件占 93.039%/93.020%。同源真边没有实际共同支持，跨源真边 79,374/95,593 有共同支持。三个人口均保留有、无共同支持的尾部负边，不能从覆盖资格推出均衡或身份独立。 [冻结名单](/Users/a1/Projects/biye-ronghe/offline_observation/docs/evidence/20261112_association_tail/prep/frozen-split.json) · [逐组件](/Users/a1/Projects/biye-ronghe/offline_observation/docs/evidence/20261112_association_tail/prep/components.json)

### 实际训练与评价

两臂相同 22 个原特征、冻结 normalizer、5,697 参数残差评分头、零末层、seed20261112、Adam .001、clip1。原 056 分数作为固定底分。每步同源和跨源各 128 正例、最高 0.1% 的 64 负例、接下来至 1% 的 64 负例。样本有放回、同一批索引供两头，训练前权重及零残差输出相同。BCE 为分类 softplus，RANK 为同源关系内相同正负数目的 paired softplus。每臂实际 1,024 步、524,288 次边呈现，共 2,048 更新。固定末步，未按结果选快照。两步真实 T 软件接口检查只验证实现，未用于选择科学端点。 [实际曝光和曲线](/Users/a1/Projects/biye-ronghe/offline_observation/docs/evidence/20261112_association_tail/science/training.json)

只用 T 校准完整跨源池达到首次完整同分组的 90% micro 真边召回，θ 分别为原056 1.7320103112、056_T 1.7314559916、BCE −0.7130228539、RANK 4.5564956742。同一臂的同源和跨源共用 θ，按 score−θ 对同一 114 WS 的 sum-gain 和 tie 规则构组。056 为原实际 114 输出，V 完整 confusion 与旧汇总相等。held177 与 V300 的分数及实际分区全部先保存，后读各自身份评价。原阈值、T 校准工作点及事后固定召回 PR 分开报告。 [校准](/Users/a1/Projects/biye-ronghe/offline_observation/docs/evidence/20261112_association_tail/science/calibration.json) · [全结果](/Users/a1/Projects/biye-ronghe/offline_observation/docs/evidence/20261112_association_tail/science/summary.json)

### 同召回的假边

下表是各自完整候选池在首次达到固定 micro 召回的 FP。阈值仅作事后 PR 诊断，不进入最终分组。

|人口与源关系|原056，80% / 90%|同池BCE，80% / 90%|同池RANK，80% / 90%|
|---|---:|---:|---:|
|held177，同源|343 / 615|428 / 730|1,053 / 1,785|
|held177，跨源|2,186 / 3,265|58,333 / 69,229|7,816 / 13,616|
|V300，同源|3,972 / 6,898|3,184 / 5,211|4,234 / 7,349|
|V300，跨源|32,914 / 44,754|995,841 / 1,196,033|94,739 / 174,175|

V 跨源 90% micro 召回的 PFA 为 75.116%/98.776%/92.156%。跨源 AP 为 .31931/.01329/.33128，AUC 为 .99005/.69288/.97383。RANK 的整体 AP 比原056略高，80/90% 工作区均更差。T 留组件亦同向恶化，不能仅以 V 上变化解释。056_T 只变阈值，其 PR 曲线与原056完全相同。

### 固定 T 工作点的最终分区

每格为 FN / FP，完整人口和同一 WS 解码。RANK 对原056、056_T、BCE 的预定联合门在 held 和 V 均未通过，满足全部方向的组件数均为 0。

|人口|臂|同源 FN / FP|跨源 FN / FP|
|---|---|---:|---:|
|held177|056|388 / 345|816 / 609|
|held177|056_T|388 / 345|816 / 609|
|held177|BCE|783 / 473|3,340 / 3,367|
|held177|RANK|2,333 / 0|2,403 / 529|
|V300|056|1,441 / 4,487|5,888 / 8,085|
|V300|056_T|1,440 / 4,486|5,883 / 8,082|
|V300|BCE|3,230 / 6,645|13,860 / 18,588|
|V300|RANK|8,198 / 0|9,663 / 3,423|

V300 的论文同源 mPCA/mwPCA/mPFA/mwPFA 分别为原056 87.967/81.655/10.071/39.892%，BCE 73.612/60.413/14.535/57.201%，RANK .092/11.921/null/0%。RANK 同源只有3场产生正预测，故场景平均误关联率为 null。跨源对应原056 82.732/63.615/14.169/43.305%，BCE 30.003/13.759/67.992/87.675%，RANK 48.715/41.110/15.968/33.451%。缺失定义人口完整保留，没有以 defined-only 平均替代 null。

RANK 在 V 原代表身份无丢失、额外认领389个，代表总数9,100，但重复输出由1,068增至5,267，298场重复增加，同船节点×Q支持丢286,251、补18,791。BCE 丢291个原代表身份、同船支持丢150,220。held 的RANK同船支持丢76,216，173场重复增加。代表目标数增加与片段碎裂可以同时发生，验收需保留逐时支持与重复数。 [逐场V](/Users/a1/Projects/biye-ronghe/offline_observation/docs/evidence/20261112_association_tail/science/V_select-scenes.json) · [逐场held](/Users/a1/Projects/biye-ronghe/offline_observation/docs/evidence/20261112_association_tail/science/held-scenes.json)

去 V 最大130场组件后，原056/RANK 的跨源 FN/FP 为933/683与2,192/511；去held最大组件后为415/404与1,547/331。两处均保留去假与补漏之间的代价。

### 执行、复核与边界

前置冻结 checkout 1eba52d，科学572.656秒、worker575.245秒、RSS948,400,128B、0更新。正式科学冻结 b1e094f，223.790/226.895秒、RSS1,481,072,640B、2,048更新。均通过SSH MCP、GPU1/CPU20–23、systemd一次托管、Restart=no、Linger=yes，真实退出0。未打开封存测试、未更新原056、未读物理参考、无科学重试或额度变更。 [前置托管](/Users/a1/Projects/biye-ronghe/offline_observation/docs/evidence/20261112_association_tail/prep-launch/status.json) · [正式托管](/Users/a1/Projects/biye-ronghe/offline_observation/docs/evidence/20261112_association_tail/science-launch/status.json)

发布前独立核367个含同分排序案例和144个分区覆盖案例。逐场重复门在发布前修正为逐场增加数为0，避免跨场相消。正式独审独立重放1,908套完整WS，核held258,836对及V4,439,092对、逐场/全部组件/去最大、论文指标、AP/AUC、召回同分、覆盖、全部资格门，以及拟合池和seed曝光。176,146项精确及105,801项数值检查通过，89.203秒、RSS836,440,064B、真实退出0。 [独审实现](../../tools/analysis/audit_offline_association_tail.py) · [独审原件](/Users/a1/Projects/biye-ronghe/offline_observation/docs/evidence/20261112_offline_association_tail/audit/audit.json)

原科学未保存175场校准的新分数，只保存阈值与校准点。独审重新做冻结头的CPU校准前向，BCE/RANK重定阈值各与CUDA相差4.7684e−7，各自工作点的TP/FP完全相同。把原CUDA阈值直接用于CPU分数时，BCE边界少1TP，2930相对2931；RANK和056_T无边界计数差。该数值边界和校准原分数未落盘的缺口均保留，不以CPU重新阈值替换科学结果。模型、身份、完整分区和预测留服务器，公开聚合已回传。 [终态回执](../../HandOver/S2L-20261112-c92.md)

## SQUEEZE_REPORT

`SQ-OFFLINE-ASSOCIATION-TAIL-CONTRAST-20261112`，输入上列Record，next_skill为Mode C。结果限定于静态旧056尾部、相同新残差头、一个种子和当前预算。BCE 与 RANK 均不替换原评分。

**DE。** 同源/跨源之间的分数标尺发生明显漂移。V 中RANK同源真边分数中位−3.189、最大4.390，跨源真边中位6.362。T跨源校准θ4.556高于所有同源真边分数，因此同源raw真边接收为0，decoded剩4条真边由跨源路径共组引入。同源各自事后重定阈值的80/90%FP亦比旧056差，单纯校准解释不足。

**DE／AI。** 在冻结分数上分别给同源、跨源各加常数，同源关系内部paired softplus的变化小于9e−16。这验证了目标对各关系独立平移不敏感，而共同θ和WS需要跨关系可比较的数值权重。该缺少约束的方向是当前适配的实际缺口，尚不能据此量化全部退化的因果份额。

**DE。** 事后按每个V场、每种关系的原056负例最高1%分层，跨源90%召回的FP落在该尾部以外者，原056/BCE/RANK为23,872/1,183,166/161,690。固定T工作点新增于旧056的跨源FP中，BCE 1,181,610/1,182,371、RANK 146,089/146,741在旧V尾部以外。已有错边和新错边都大量位于该静态层之外。V标签仅用于冻结后归因，未用于训练选例。 [固定分数只读诊断](/Users/a1/Projects/biye-ronghe/offline_observation/docs/evidence/20261112_association_tail/fixed-score-squeeze.json) · [实现](../../tools/analysis/offline_association_tail_squeeze.py)

**AI／UN。** 单独集中旧高分负边可能失去对其余合法负边的约束，且76个拟合组件的尾部高度集中。当前结果没有分别干预这些条件，不能断言哪一项是唯一根因。后继应从全候选背景约束、跨关系共同标尺及组件曝光三个条件筛选可归因设计。可比较的评分必须同时接受完整候选PR和原WS实分区检验。更换全局阈值或只展示相对已退化BCE的收益均不足以满足当前目标。
