# 110片段身份容量诊断：真实回执复核与证据深挖

2026-10-29。`EXP-FRAGMENT-CAPACITY-110-20261029` → `SQ-FRAGMENT-CAPACITY-110-20261029`。本页记录已发生的一次诊断，不维护领取/当前状态。

**结论：在固定056 V300有效输入中，全部300场都存在不能由未经并段或闭包的双源单位容量硬边输出完整表达的合法身份关系。16,473条跨源真边的该条件容量为9,146，micro direct-edge recall上界55.5212%；另有8,202条同源真边和11,063条无共同Q64支撑的正边。该结果验证任务/输出空间的适配条件，不是MH-T2TA实测成绩、当前anchor分组失败归因或新方法效果。**

## 一、EXPERIMENT_RECORD：实际执行与独立复核

- 包版本：`research-pipeline-contract v1.0`；来源Idea：`IDEA-FRAGMENT-IDENTITY-CAPACITY-20261029`，事前plan1.0，[原计划](../proposals/fragment_identity_capacity_20261029.md)。来源矩阵为Mode C1.28 C53-Pa/P08/M35；C33旧计划只提供条件数学，不冒充另一次真实执行。
- 问题/H1：同一有效输入身份在两源中有a/b段时，全部直接真边`T=ab`与单位容量硬边上限`B=min(a,b)`在真实人口上差多少？H2：合法正边中无共同Q64候选支撑的规模和组成如何？
- 设计/对照：固定既有056 final1000的V300原缓存、全节点全候选和原manifest；θ固定为`1.7320103112387182`，只绑定raw计数，不选择身份或计算容量。没有阈值搜索、抽取成绩最差子集或对照方法训练。
- 实际仅一次attempt1；`run.status=COMPLETED`、launcher `COMPLETED/returncode=0/timed_out=false/failure=null`、`exit_code.txt=0`。实际读取300/300场，无失败/重试记录。原数据和原scores只读。
- 真实命令入口为服务器Python `tools/analysis/fragment_identity_capacity.py --scores …056-attempt-1/association/final-scores.npz --summary …056-attempt-1/association/summary.json --manifest configs/data/mtad-level1-scene3000.json --output …LEVEL1-FRAGMENT-CAPACITY-20261029-110-attempt-1`；cwd为独立`association-code-run110`。精确路径和argv保留于原`launch-request.json`，本页省略公共前缀。
- 时间：run起点`2026-10-28T20:34:07.661941+00:00`（北京时间10月29日04:34:07.661941）；payload自报4.684763秒；launcher自报5.130811秒，结束`2026-10-28T20:34:12.419603+00:00`。两时长覆盖口径不同，分别保留。
- 资源：CPU；CUDA不可见；OMP/MKL/OpenBLAS线程环境均4；记录的AS限制8,589,934,592B（8GiB），峰值RSS103,432,192B（98.640625MiB）。计划RAM预算4GiB、10min advisory，均未触线；未提供CPU核秒/实际峰值线程/主机型号，不补填。`model_forwards=0`、`optimizer_updates=0`；run记录`archive_read=false/reference_read=false/test_opened=false/scores_recomputed=false`，静态代码与此操作范围一致，未另做系统I/O追踪。

### 证据位置和核验层次

原回执目录：[110原件](/Users/a1/Projects/biye-ronghe/docs/evidence/20261029_small_validation/110/)。包含`run.json`、`summary.json`、`scene-summary.json.gz`、`launch-status.json`、`launch-request.json`、`exit_code.txt`；另补[capacity-size-histogram.json](/Users/a1/Projects/biye-ronghe/docs/evidence/20261029_small_validation/110/capacity-size-histogram.json)，为执行后从原private细行聚合的六行(a,b,count)，未回传任何identity/source编号。独立重聚合结果：[audit JSON](2026-10-29-fragment-capacity-audit.json)。代码定位：[分析入口](../../tools/analysis/fragment_identity_capacity.py)、[原raw指标定义](../../association/evaluation/association.py)。

1. **DE，独立重聚合**：本地仅读返回的300条逐场计数；未导入原程序的pool函数。逐项核scene顺序与本地冻结manifest V_select、38个component映射；20项原始整数及5项衍生人口计数的全局与38组件重聚合均与原summary一致，浮点比例/场景均值在`1e-12`内一致，0差异。独立检查每场`pair_count=N(N−1)/2`、正边同/异源分解、`T=B+下界遗漏边`、全部raw混淆和、正边=TP+FN、eligible+zero分母、noQ拆分及U公式，全部通过。无空场、无缺source场、无跨源真边零分母场；所有300场source_count=2。
2. **DE，运行记录+静态链**：程序检查原NPZ的完整唯一i<j候选、二值label等价性、有限scores及support0..64，并逐场绑定原056 raw计数后才写完成。返回300行均记录`labels_equivalence_verified=true`。这说明实际执行通过了这些检查；本地未再打开原NPZ，不能称为对4,439,092条原标签的第二次独立校验。
3. **DE，执行后匿名直方图补核**：最初本地`scene-summary.json.gz`删除了原`scenes.jsonl`的`identities/source_pairs`细行；随后服务器从这些private原行逐项核a×b/min，并仅导出六行匿名(a,b,count)直方图。本地独立用六行重算Σab=16,473、Σmin=9,146、Σ同源组合数=8,202、9,100类和24,599节点，以及受限人口/正边、零分母和单节点等计数，均与原summary0差异。identity等权上界独立精确重算为`(1819+5478/2)/7297 = 4558/7297 = 62.4640263%`。这是**执行后证据聚合核验**，没有新模型前向或原生量测读取、没有补造事前终点。private身份细行仍未回传，直方图不能独立证明每个类的场景/source对应，也不能从全局a/b分布重建源对candidate数2,195,509；后者仍为原summary报告值。

|a|b|身份次数n|n×ab|n×min(a,b)|有定义U_i|
|--:|--:|--:|--:|--:|--:|
|1|0|928|0|0|null|
|1|1|1819|1819|1819|1|
|1|2|1767|3534|1767|0.5|
|2|0|875|0|0|null|
|2|1|1862|3724|1862|0.5|
|2|2|1849|7396|3698|0.5|

六行保留回传顺序和a/b方向；本报告不从匿名a/b方向推断真实源质量或完整source优先。实际正分母的所有受限对象U_i恰为0.5，非受限(1,1)为1，这是本人口分布的描述，不能推广到片段数更多的总体。

### 全人口和主要结果

计数单位分别为节点、候选边、场景内身份等价类次数、identity×无序source-pair；比例均为无量纲fraction（表中乘100显示%）。**9,100是300场内匿名输入身份等价类次数之和，既非跨场去重身份数，也不证明它们与全标签声明9,100目标名单一一绑定。** 缓存只包含清洗后有效输入；无输入目标、被quarantine节点和原生时间区间未进入此诊断，不据恰好相等的总数扩展到全目标人口。

|实际结果|分子/分母或定义|值/边界|
|:--|:--|:--|
|输入节点、全部候选、全部真边|300场全有效输入|24,599；4,439,092；24,675|
|同源/跨源真边|全部24,675正边|8,202 / 16,473；和为24,675|
|输入身份等价类次数|所有场景均保留|9,100；其中928为单节点，1,803为单source，6,353含同源多段|
|跨源有定义identity/source-pair|a>0且b>0|7,297；另外1,803个T=0行保留、U=null|
|受限identity/source-pair|ab>min(a,b)|5,478/7,297=75.0719%（或5,478/9,100=60.1978%）；遍及300/300场|
|受限对象内跨源真边|Σab restricted / 全跨源真边|14,654/16,473=88.9577%|
|单位容量正确硬边上限|Σmin(a,b)|9,146；即使理想选择仍至少7,327/16,473=44.4788%直接真边不能输出|
|micro条件recall上界|9,146/16,473|55.5211558%；不是mPCA/mwPCA|
|场景等权条件上界|mean(B_s/T_s)|55.6847412%，300/300均有定义，strict全场均值相同|
|identity等权条件上界|mean(B_i/T_i), T_i>0|从执行后六行a/b直方图独立重算4558/7297=62.4640263%，7,297个有定义|
|无共同Q64正边|joint_support_queries=0 / 全正边|11,063/24,675=44.8348531%|
|其中同源/跨源|各自合法正边分母|8,202/8,202=100%；2,861/16,473=17.3678140%|
|受限对象的noQ跨源真边|restricted noQ cross / all noQ cross|2,860/2,861；另1条属于不受容量限制的对象|
|raw绑定混淆整数|固定原scores与θ|TP22,666；FP55,652；FN2,009；TN4,358,765；只是原人口绑定，不是本项改善结果|

全局仅两个source匿名等价类`source_0/source_1`，300场都有两源，只有一个源对；其T/B即16,473/9,146。原summary还报告该源对候选2,195,509，包含1,803个identity零真边行；全候选4,439,092还含同源边。没有按source可靠性排序或赋先验。未观察到多于两源，因此本批不测试“多源各源对上界能否同时达到”。

### 计划符合性、负结果与未知

按既定一次缓存/CPU诊断执行，未见改变名单/阈值、丢零分母或重算分数。模型训练、seed效果比较、分组干预、坐标融合/恢复评价、原作者复现、MH-T2TA实际硬输出均未进行。原scene输出含更细记录，首次回传删细行；后补执行后匿名a/b直方图，已补齐容量与identity等权的可重算部分，不倒写成事前计划改变。private逐identity场景对应仍未回传。固定开发人口含共享身份，不能将9,100类或300场当IID重复，不报p值、bootstrap置信区间或独立泛化。一次确定性计数没有优化收敛轨迹；未做阈值敏感性，因为容量不依赖score阈值。

## 二、SQUEEZE_REPORT：从分母和异质性解释

### 已回答的窄问题与正面路线

- **DE，任务适配需求成立**：双源一一硬边若直接充当本任务的全部跨源正关系，受限人口很大。最短合法适配可选同源先并段、输出多对多身份partition，或先给backbone再定义identity闭包；要明确输出是直接边、骨架边还是闭包。各路线可用成熟近邻作为起点，新增价值由同任务下身份完整性、覆盖、物理恢复与成本检验，不能仅因已有处理fragment的方法就否定研究价值。
- **AI，数学机制明确而非经验因果**：一个身份有a/b段，源对全边为ab，单位容量每节点最多配一次只能保留min(a,b)。该代数解释的是声明硬输出的表达能力；它没有观察当前模型损失从何而来。当前anchor_direct允许多节点组，根本不受这个单射界。软矩阵、联合身份簇、预并段、局部单轨窗口以及backbone闭包也不受原界直接约束；不能用本结果指控MH-T2TA作者任务错误或作者成绩低于55.52%。
- **DE+AI，关系与query贡献需分开**：全部8,202条同源合法身份边在现有Q64上无共同support；跨源也有2,861条。Q64共同支撑不是身份真关系的必要条件。noQ只说明离散公共网格上支撑交集为空，不等于原生时间区间绝对不重叠，不证明两段物理上不同时存在，也不自动给出任何query应采用哪条轨迹。保留身份关系，再由时间/状态可信度决定对某query的贡献，是有具体条件的适配路线；本项未验证其物理收益。

### 场景与共享身份组件的异质性

场景U从50%到77.7778%，线性插值Q25=52.9412%、中位数55.3397%、Q75=57.1429%；这些是固定总体描述，不能冒充置信区间。38组件micro-U范围50%—73.3333%，组件等权均值55.1883%；该新组件均值是探索性描述，不是计划主终点。

按既有最大两组件分层的探索性复核：`component-23608a7a22c5322f7254ffc1`130场、`component-c7082a697d934f0004db48b9`59场，合189/300场、14,063/16,473跨源真边，U=7,813/14,063=55.5571%；其余36组件111场，U=1,333/2,410=55.3112%。每场都受限，故“存在容量需求”并非只来自两个巨组件；但总体边权重明显受其主导，不能把该相近比例提升成跨身份泛化或排除所有选择偏差。两大组件noQ正边9,443/21,062=44.8343%；其余1,620/3,613=44.8381%。

以下保留全部38组件，T/B针对跨源正边，最后一列noQ含同源及跨源全部正边。匿名component ID仅用于对应原manifest，不代指真实独立对象。

|共享身份组件|场景|输入身份次数|跨源真边T|容量B|U=B/T|无共同Q64正边/全部正边|
|:--|--:|--:|--:|--:|--:|--:|
|`component-23608a7a22c5322f7254ffc1`|130|6805|12282|6814|55.4796%|8262/18400|
|`component-c7082a697d934f0004db48b9`|59|993|1781|999|56.0921%|1181/2662|
|`component-9ea441151d6468bd21826aea`|20|268|489|273|55.8282%|320/731|
|`component-75391b641c3cd7287e382451`|12|218|360|205|56.9444%|233/541|
|`component-805a0d28ca2328b9fac2c2f2`|8|57|106|58|54.7170%|70/159|
|`component-5acb181fd8d75db014b018e5`|6|66|144|77|53.4722%|100/217|
|`component-a077aa3eb774a8b670334e7c`|6|62|119|67|56.3025%|83/177|
|`component-e043376fcec558cddc17efc9`|6|50|97|52|53.6082%|68/147|
|`component-d091361fb95b75d6d2a1a9c5`|5|90|151|83|54.9669%|92/225|
|`component-e08b792544e4e00fc0ec7592`|5|52|110|62|56.3636%|73/160|
|`component-99bde609acdcfd5bcc1d7e7c`|4|61|115|61|53.0435%|82/174|
|`component-13bf8d8c05af98b86d432a08`|3|100|161|91|56.5217%|110/247|
|`component-6168d795e66f68b6ea6c3b7a`|3|60|109|60|55.0459%|77/164|
|`component-e0806a8adaabe25cf8917cc1`|3|22|38|20|52.6316%|26/57|
|`component-0b71187a53d6235c8f81020b`|2|11|26|13|50.0000%|19/40|
|`component-0b9d7d23108cbf0a74ed5828`|2|7|20|10|50.0000%|15/30|
|`component-568f84fe210a79894feac892`|2|12|26|14|53.8462%|24/41|
|`component-c8161c1679f5abb00c5cd1aa`|2|8|23|12|52.1739%|17/34|
|`component-d29cb6bc9ce95155a0bf2abe`|2|12|21|11|52.3810%|14/32|
|`component-e062a18f3a426eba715c16d2`|2|10|25|13|52.0000%|18/38|
|`component-1a97afb84b073a6e67c5bb00`|1|6|17|9|52.9412%|13/25|
|`component-1b730762e8dcc866d831d9a3`|1|5|8|5|62.5000%|5/11|
|`component-4ad4f4b15e8855fdb30bf269`|1|5|16|8|50.0000%|16/25|
|`component-4ce1762bf4e868830cf8a460`|1|14|20|11|55.0000%|13/32|
|`component-4eb65272866a82b14a0055d6`|1|11|15|11|73.3333%|6/20|
|`component-5ee24adf22c874beaf8d076d`|1|36|64|36|56.2500%|39/95|
|`component-5f2aedadc99ed44ea4633c75`|1|5|7|5|71.4286%|2/9|
|`component-5fb27466478758819ffefd1e`|1|7|10|6|60.0000%|8/16|
|`component-6fb7e5c36b795834bb97c592`|1|4|10|5|50.0000%|8/15|
|`component-74e6aaca08d871c01ac5c399`|1|4|8|5|62.5000%|5/11|
|`component-9b9449afa258e2e53c16d960`|1|4|11|6|54.5455%|7/16|
|`component-9ec9c2b8d5ac2b679aa6b8c0`|1|7|16|9|56.2500%|11/23|
|`component-c260c3bbf884e315f715f00b`|1|5|13|7|53.8462%|8/19|
|`component-c6b98d02b719285e3a0d16d7`|1|5|12|6|50.0000%|7/18|
|`component-e6c09979d5dfd52c480a1657`|1|4|10|5|50.0000%|6/15|
|`component-e897d9ac78e38d080ce57c79`|1|5|16|8|50.0000%|14/24|
|`component-fdbc675a9f9f6a7e1c67e3bb`|1|3|10|5|50.0000%|8/15|
|`component-fde3da10230624198088bd00`|1|6|7|4|57.1429%|3/10|

### 替代解释、停止线与最有区分的下一步

本项不区分“关联证据差”“decoder/order问题”“身份簇损失与物理任务损失不一致”，也不验证reference质量、源偏差或缺测可靠性。源重复fragment、预合并和闭包会改变容量条件；这既是边界也是可发展解法，不能把它们排除后再声称所有方法都不适配。源码/回执验证不能替代方法效应。

**下一步优先联合109同缓存固定score诊断**，先核当前anchor的顺序敏感性、group-comembership误差及全人口覆盖/重复，而不追加训练或再跑一遍容量计数。若要比较fragment-aware解法，应事前固定同一边证据、同一真实θ及合法输出定义，比较当前partition与一个能保留fragment身份关系的明示构造；direct-edge与backbone闭包分报。所有raw pair整数/原paper raw指标须保持不变；decoded关系及之后真正重算的物理输出才可能变化，不能承诺fixed-score decoder降低raw FP55,652。

最有区分的价值检验是：**同一scores下，合法partition/闭包是否改善身份恢复和覆盖/重复，同时在完整同人口物理重算中有净收益**。只有109关系结果时，先回答分组，物理效果保持UN；旧坐标prediction cache不能拼作新组物理效果。若关系无改善，则保留已确认的适配条件，不能称方案成功；若关系改善而物理不改善，必须继续区分query贡献/时间外推/错误reference，不把所有问题都归给容量。后继如需模型/物理前向另立具体合同，本报告不发布或启动。

### 供Mode C回流的候选事实（本页不改主矩阵或分配全局ID）

|候选|证据等级与精确定义|能改变的判断/不能扩展到哪里|
|:--|:--|:--|
|F-CAP110-1|DE：300/300有效输入场景有受限identity；5,478/7,297有跨源输入类受限，涵盖14,654/16,473跨源真边|支持C53-Pa/P08的输出适配需求；不是当前anchor错误或MH实测|
|F-CAP110-2|DE+条件数学：16,473跨源真边、B=9,146、下界遗漏7,327；U55.5212%|支持M35在本人口的规模；U只约束未经并段/闭包的单位容量硬边，不是任何paper指标|
|F-CAP110-3|DE：同源合法真边8,202；noQ正边11,063（同源8,202、跨源2,861）|身份关系与Q64共同支撑不能混为同一目标；不推出原生时域无交叠或融合收益|
|F-CAP110-4|DE探索性：38组件和全部300场保留；最大两组件与其余U55.5571%/55.3112%|表明存在性不只在巨组件；不认证身份独立泛化|
|M-CAP110-1|AI/代数：a,b片段的ab全边对min(a,b)硬容量；在全边任务下产生表达上限|数学条件得到真实人口覆盖；现有非单射decoder误差根因仍UN|
|E-CAP110|DE：110原run/launch/exit、summary、300场计数、独立audit JSON及本报告|逐场及六行a/b重聚合已核；private细行场景对应和原NPZ仍未本地独审|

```yaml
package_type: EXPERIMENT_RECORD
package_version: research-pipeline-contract v1.0
experiment_id: EXP-FRAGMENT-CAPACITY-110-20261029
source_idea_id: IDEA-FRAGMENT-IDENTITY-CAPACITY-20261029
plan_version: '1.0'
execution_status: COMPLETED_exit0
research_question: 当前有效fragment身份全边关系是否受双源单位容量硬输出限制，合法noQ正边规模如何
hypothesis: ab真边多于min(a,b)时条件硬边不能完整表达；身份关系不等query共同支撑
planned_vs_actual_design_and_controls: 固定056全V300缓存与原summary/manifest，原θ只绑定；一次CPU零模型，未见执行偏离；执行后匿名a/b直方图补核
actual_configuration: CPU线程环境4，AS8GiB，RSS98.640625MiB，payload4.684763s，launcher5.130811s，0forward/optimizer/test
samples_materials_or_data_provenance: 300场24599有效输入节点4439092候选24675正边；9100为场内匿名identity次数，不认证目标名单
results_with_uncertainty: 条件microU9146/16473；5478受限identity；noQ11063；有限开发总体无IID推断
all_runs_or_repetitions: 单attempt1；无模型效应重复或自动重试
observations_and_deviations: 全300场有受限对象；未做方法/物理效果；执行后六行a/b聚合已独立重算，不认证private逐类场景对应
evidence_locations: '110原件目录；本报告与audit JSON；原分析脚本和事前Idea'
known_limits_and_missing_evidence: 原NPZ/private逐类场景行未本地重审、全target名单未绑定；a/b聚合可重算部分已补齐；作者实际输出/原生时间/物理收益未知
next_routes: squeeze-experiment已在本页完成；回流Mode C并联合109关系诊断，不自动新训练
next_skill: squeeze-experiment
```

```yaml
package_type: SQUEEZE_REPORT
package_version: research-pipeline-contract v1.0
squeeze_id: SQ-FRAGMENT-CAPACITY-110-20261029
package_id: SQ-FRAGMENT-CAPACITY-110-20261029
input_experiment_record_ids: [EXP-FRAGMENT-CAPACITY-110-20261029]
evidence_status: partial
claim_scope: 056固定V300有效输入的条件全硬边容量与Q64支撑计数；非方法效应
data_quality_and_provenance_gaps: 场内等价类由完整标签恢复；本地重聚合逐场计数及执行后六行a/b直方图；private逐类场景行/原NPZ与全target名单缺独立核验
primary_result_with_uncertainty: 条件B/T=9146/16473=55.5212%；全部300场受限；固定有限人口不做IID区间
process_dynamics: 一次CPU4线程环境，4.684763s，RSS103432192B，0模型前向/更新，无异常
mechanism_and_intermediate_findings: ab与min(a,b)表达限制有实际人口；noQ不等身份为负；现decoder并非单射
errors_heterogeneity_and_boundary_conditions: 所有38组件保留，最大两组件与其余U近似但非泛化；软矩阵/簇闭包/并段/局部单轨不受原界
robustness_sensitivity_and_alternative_explanations: 逐场整数全局及38组件、六行a/b容量与identity等权重聚合0差异；未测decoder/score/融合原因；无阈值或新样本敏感性
new_P: none；细化既有P08/C53-Pa
new_F: [F-CAP110-1, F-CAP110-2, F-CAP110-3, F-CAP110-4]
new_M: M-CAP110-1为条件代数及AI；当前实际错误根因UN
evidence_index: E-CAP110，110原件及本报告audit JSON
open_questions: private逐类场景对应；合法fragment-aware输出在同scores下的净收益；物理/query贡献与全target绑定
next_routes: deep-read_Mode_C；先联合109关系诊断，物理重算或新方法另定合同
next_skill: deep-read Mode C
```
