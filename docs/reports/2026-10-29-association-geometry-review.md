# 130：固定候选几何／LOO诊断的真实记录与深挖

2026-10-29。EXPERIMENT_RECORD `EXP-130-20261029`＋SQUEEZE_REPORT `SQ-ASSOCIATION-GEOMETRY-130-20261029`，research-pipeline-contract v1.0。[独立后续Idea1.0](../proposals/association_geometry_continuation_20261029.md)承接[128科学合同](../proposals/association_geometry_signal_20261029.md)和OA-C6r；[128真实OOM失败](2026-10-29-association-geometry-failure.md)保持原状。130一次attempt-1真实completed/exit0，6场原件复用＋26场新构造，合并原32场/9组件。正式独立保存候选标签/人口/秩AUC审计已通过，具体范围见§7。

**主要结论：当前固定负LOO残差scorer没有呈现超过同候选静态强控的信号。** 全共同可定义3429候选，SCORE_MIN AUROC0.892691，LOO_MEDIAN0.448240、LOO_RMS0.527186；32场同人口比较LOO median相对MIN为0好/2同/30差。原WS组的能力更弱，扩展候选的高分不能当作能修复原组或提升系统F1。数值负控支持“当前LOO由已有几何重组”的限定事实，不证明一般时序模型都无价值。

## 1. Record：问题、固定设计和修复范围

本项为Pattern5表示/系统审查：在同候选上，静态pair/support几何、原pair分数与LOO的pure/mixed区分能力怎样，LOO有多少可定义覆盖，是否提供超出完整pair信息的新信息。它不训练新head、不改关联分数/分区/位置输出，不把候选AUROC称端到端效果。

固定056缓存Q64 `native_states/support/sources/node_ids`（这些是预测状态，非原生事件），原全部字典序i<j pair/scores、θ=1.7320103112387182、114 WS原membership及事前V32场序。原WS组含singleton全部保留；每个至少二成员组按组外节点与组内pair平均signed margin，取最高两个单节点扩展，平分原节点序，仅去相同成员集合。候选是可供诊断的假设，**不是同时输出的对象**，扩展候选并不构成新分区。

全部pair逐Q球面距离和joint support保留于private。shared静态统计只用两节点均有支持的query，full统计使用完整Q64预测；两种权限不能混为一种原生信息。LOO整成员留出，自己有support且其它成员至少一个有support才计算；其余支持成员纬均值/经圆均值，单成员exact复制，零圆resultant沿128原L1规则fallback、原经度表示不变。无支持为null，不赋坏分。十个事前scorer方向冻结：四种signed score聚合方向为正，shared/full pair与LOO距离median/RMS方向为负。

每场特征先落盘，之后才打开原身份及完整pair labels、核两种标签等价，评价pure/mixed；不读取reference数组。metadata有既有评价汇总用于输入人口绑定，不能称整个程序从未接触任何评价材料。身份只作评价，不用于选组/纯组筛选/特征。

128在6场完成后被systemd-oomd杀死，原run/launcher的RUNNING为陈旧状态；130先核真实失败，再逐文件byte-exact导入前6场features/evaluation/公共row，**不重算这6场候选/几何/特征/预测**。剩余26场只将同一球面distance分4096pair构造、及时释放跨场大tensor；候选、公式、原L1圆均值fallback、精度和容差保持。原128不重启、不修改，不触oomd/他人进程。六场复用不是第二次独立复验，分块是资源修复而非研究方法创新。

## 2. 候选人口、标签与不可定义范围

3851个原节点全部在候选集合中出现，因为原WS全组保留；这是100%节点**候选覆盖**，不是新位置coverage或目标召回。4025候选=1605原组＋2420扩展；完全重复删除0。1348 pure/2677 mixed，unknown0；pure仅表示成员身份相同，不保证对象完整、无重复或全局关联正确，395 singleton天然pure。

| 候选层 | 总数 | pure | mixed | LOO及全部scorer共同可定义 |
| all | 4025 | 1348 | 2677 | 3429 |
| original | 1605 | 1159 | 446 | 1040 |
| size1 | 395 | 395 | 0 | 0 |
| size2 | 518 | 395 | 123 | 352 |
| size3 | 1485 | 426 | 1059 | 1453 |
| size4plus | 1627 | 132 | 1495 | 1624 |
| single_source | 580 | 543 | 37 | 8 |
| cross_source | 3445 | 805 | 2640 | 3421 |
| no_shared_pair_support | 596 | 545 | 51 | 0 |
| has_shared_pair_support | 3429 | 803 | 2626 | 3429 |
| shared_fraction_0_to_half | 3041 | 642 | 2399 | 3041 |
| shared_fraction_over_half | 388 | 161 | 227 | 388 |

原组1159 pure/446 mixed；扩展189 pure/2231 mixed，后者92.19%为mixed，占全部候选60.12%。扩展是依据已有score生成，和原组的大小/类构成/难度不同，不能把合并AUC反推原组纠错能力，也不能由现有汇总唯一归因“高AUC全靠简单负例”。本次没有标签选候选。

共同定义3429/4025（85.19%），其中803 pure/2626 mixed。缺失596含395 singleton及201多成员无共同support；缺失545 pure/51 mixed。于是pure覆盖803/1348=59.57%，mixed覆盖2626/2677=98.09%，覆盖很不平衡。必须保留无定义人口，不能将其填为负类。单源层LOO仅8个可定义、全mixed，AUROC为null；不将null当0、0.5或证实无价值。

## 3. 完整十scorer：各自人口与共同人口分别比较

AUROC正类为pure、ties=.5；单类null。下表“各自”score/full几何为3630（953 pure/2677 mixed），shared/LOO为3429（803/2626）；只有“共同”列能作同候选直接比较。各自场景等权和共同场景等权均32场可定义；pooled跨场候选排序与场景等权不是同一个估计目标。

| Scorer | 各自pooled AUC | 全共同pooled AUC | 各自场景等权AUC | 全共同场景等权AUC | 去最大共同pooled AUC |
|---|---:|---:|---:|---:|---:|
| SCORE_SUM | 0.646855 | 0.660417 | 0.851311 | 0.855195 | 0.832224 |
| SCORE_MEAN | 0.815989 | 0.805195 | 0.925133 | 0.918245 | 0.904971 |
| SCORE_MIN | 0.900670 | 0.892691 | 0.962592 | 0.957773 | 0.949821 |
| SCORE_MEDIAN | 0.687762 | 0.664909 | 0.811179 | 0.795933 | 0.793474 |
| PAIR_SHARED_MEDIAN | 0.470753 | 0.470753 | 0.676254 | 0.676254 | 0.607579 |
| PAIR_SHARED_RMS | 0.559225 | 0.559225 | 0.716277 | 0.716277 | 0.756279 |
| PAIR_FULL_MEDIAN | 0.575235 | 0.536757 | 0.773305 | 0.760370 | 0.713792 |
| PAIR_FULL_RMS | 0.642721 | 0.619154 | 0.813832 | 0.803299 | 0.750235 |
| LOO_MEDIAN | 0.448240 | 0.448240 | 0.649267 | 0.649267 | 0.605379 |
| LOO_RMS | 0.527186 | 0.527186 | 0.690432 | 0.690432 | 0.728544 |

主LOO_MEDIAN全共同0.448240低于0.5，LOO_RMS0.527186仅略高；不据V后验翻转符号制造胜出。与同支持静态PAIR_SHARED_MEDIAN/RMS比，两种LOO pooled及场景等权均更差。32场共同人口LOO median对shared median为2好/9同/21差，LOO RMS对shared RMS为1好/14同/17差；对SCORE_MIN分别0好/2同/30差、0好/3同/29差。这只是固定scorer排序比较，没有学习权重或对象决策阈值。

SCORE_MIN合并AUC较高，但原组各自1210可定义上的AUC仅0.644581；LOO在原组1040上的0.358046/0.384862不能直接与1210人口比较。下面事后追加了真正相同的人口交集，消除这一分母偷换。

## 4. 明确事后诊断：原组、扩展、二成员各与common取交集

在首次130完成并看到上述分母问题后，另一次有界只读保存evaluation做本节，**不是执行前主分析、不是新实验/特征/模型或原run改写**。独立NumPy `unique`求平均秩，Mann–Whitney U公式计算AUC，没有导入生产评价函数；known且十scorer全部finite之后分别取original、非original、size2。保存`posthoc-common-candidate-comparison.json`，保留run/summary原件。pool与等权均在各自同候选人口内：

- original∩common：1040=614 pure/426 mixed；15场具双类，另外17场不纳入AUROC均值但不删除候选/场景记录。
- extension∩common：2389=189 pure/2200 mixed；19场具双类。
- size2∩common：352=246 pure/106 mixed；8场具双类。

| Scorer | 原组pool | 原组场景等权（15） | 扩展pool | 扩展场景等权（19） | 二成员pool | 二成员场景等权（8） |
|---|---:|---:|---:|---:|---:|---:|
| SCORE_SUM | 0.394909 | 0.295995 | 0.664892 | 0.777431 | 0.572826 | 0.486653 |
| SCORE_MEAN | 0.506335 | 0.443445 | 0.754750 | 0.824496 | 0.572826 | 0.486653 |
| SCORE_MIN | 0.610581 | 0.558548 | 0.908769 | 0.949730 | 0.572826 | 0.486653 |
| SCORE_MEDIAN | 0.466425 | 0.377517 | 0.539196 | 0.592366 | 0.572826 | 0.486653 |
| PAIR_SHARED_MEDIAN | 0.366958 | 0.235698 | 0.392256 | 0.610126 | 0.385450 | 0.389325 |
| PAIR_SHARED_RMS | 0.390673 | 0.222401 | 0.512441 | 0.663937 | 0.407156 | 0.366554 |
| PAIR_FULL_MEDIAN | 0.317616 | 0.300352 | 0.457193 | 0.531585 | 0.403436 | 0.352466 |
| PAIR_FULL_RMS | 0.402303 | 0.347224 | 0.528769 | 0.575910 | 0.434192 | 0.371343 |
| LOO_MEDIAN | 0.358046 | 0.225802 | 0.373721 | 0.568292 | 0.385450 | 0.389325 |
| LOO_RMS | 0.384862 | 0.215691 | 0.473985 | 0.624473 | 0.407156 | 0.366554 |

- 原组同人口MIN仅0.610581/场景等权0.558548，LOO median0.358046/0.225802、RMS0.384862/0.215691；目前不能称能稳定识别原WS错误组。原组full pair median0.317616也很弱，不能把所有静态几何视作更好的新方法。
- 扩展MIN0.908769明显高于原组；LOO median0.373721/RMS0.473985仍不佳。全候选高分不等原分区纠错，且扩展不含任意拆分/重组，不能说明其它对象假设容量足够。
- 二成员LOO median/RMS与shared pair median/RMS的pooled及场景等权数值完全相同。四score聚合也相同，因为只有一个内部pair；同信息比较时不能把这四个名字算四种新机制。

## 5. 代数负控：有信号不等有新信息

冻结两成员容差atol=1e−6m/rtol=1e−10，27284次**member-query**检查最大差0m（每个共同query可检查两个留出方向，不是27284个独立轨迹/候选）。指定局部欧氏chart、每query当前支持成员m≥3下，LOO-MSE与`Σpair squared_distance/(m−1)²`的30115次检查，最大差3.814697e−6m²，小于冻结atol1e−4m²/rtol1e−10。

这支持两个限定事实：两成员当前LOO就是同pair距离；多成员指定欧氏平方误差统计可由全部pair平方距离重构。**不把局部欧氏恒等扩为所有球面均值/median的精确恒等，也不把完整pair信息与一个median/RMS标量混同。** 多成员球面LOO与某个静态标量不同，不自动产生新信息；当前原22D压缩也不等完整pair/support，所以是否有学习表示增量仍须另立同信息、同容量控制，不能由本项未训练scorer证明。

## 6. 九组件、去最大组件与复用边界

以下全部为同3429共同人口在各组件内的子集，各行十scorer均同候选；完整十scorer/分层原件保留。本表列主要静态与LOO对照：

| 组件 | 场数 | 共同候选pure/mixed | SCORE_MIN | SHARED_MEDIAN | SHARED_RMS | LOO_MEDIAN | LOO_RMS |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0b9d7d23108cbf0a74ed5828 | 1 | 4/8 | 1.000000 | 0.625000 | 0.625000 | 0.625000 | 0.625000 |
| 23608a7a22c5322f7254ffc1 | 19 | 608/2217 | 0.883509 | 0.433708 | 0.506117 | 0.409114 | 0.474119 |
| 75391b641c3cd7287e382451 | 1 | 21/70 | 0.828571 | 0.298639 | 0.343537 | 0.273469 | 0.303401 |
| 9ea441151d6468bd21826aea | 1 | 5/4 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| a077aa3eb774a8b670334e7c | 1 | 11/24 | 1.000000 | 0.859848 | 0.899621 | 0.848485 | 0.899621 |
| c260c3bbf884e315f715f00b | 1 | 8/10 | 0.987500 | 0.650000 | 0.687500 | 0.650000 | 0.687500 |
| c7082a697d934f0004db48b9 | 6 | 114/216 | 0.979613 | 0.671296 | 0.807018 | 0.661915 | 0.785859 |
| d091361fb95b75d6d2a1a9c5 | 1 | 28/73 | 0.919765 | 0.525685 | 0.510519 | 0.477006 | 0.501223 |
| fdbc675a9f9f6a7e1c67e3bb | 1 | 4/4 | 1.000000 | 0.500000 | 0.625000 | 0.500000 | 0.625000 |

最大组件19场/3322候选（82.53%）；其共同2825候选608 pure/2217 mixed，LOO median/RMS0.409114/0.474119。去最大余13场、644节点、703候选（287 pure/416 mixed），共同604（195/409），LOO median/RMS0.605379/0.728544，shared0.607579/0.756279，MIN0.949821；几何排序改善但仍不超强控。去最大场景等权LOO0.651204/0.711120，MIN0.970212（各自人口），共同MIN/LOO见原件；不能只挑去最大后AUC>0.5宣成功。

在九组件共同人口内，MIN对两种LOO均8组件更高、1组件相同；LOO median有3组件<0.5、1组件=0.5，LOO RMS有2组件<0.5。最小组件共同仅8/9/12候选，AUC=1或0.5不提供大样本独立保证。合并LOO低于0.5而场景等权高于0.5显示场景规模/跨场排序会影响结论，不能任取一个聚合方式。

原128六场共有443候选/435节点、共同386（108/278），LOO median/RMS0.581168/0.684669；新26场3582候选/3416节点、共同3043（695/2348），LOO0.423470/0.495881。原六场不是完整范围也不是独立复验，早期结果更好不能抵消全32结果；两部分场景构成不同，不能直接把差异归因130分块或时间变化。

## 7. 真实执行、成本、核验与偏差

- 冻结64ac7d4、独立`/home/wangguangjie/djs/association-code-run130`；脚本`tools/analysis/association_geometry_continuation.py`，原056/system/114 WS/manifest、失败128路径和故障原件均在request/run。[独立合同与完整命令](../../HandOver/L2S-20261029-130.md)只作复核，不据本报告自动再次执行。
- 原run=`runs/LEVEL1-ASSOCIATION-GEOMETRY-CONTINUATION-20261029-130-attempt-1`；同名`-launch`、exit文件均0。真实23:00:46.193165至23:04:16.873857（UTC+8）；服务`mtad-79fbecfb79be.service`inactive/dead、Result=success、ExecMainStatus=0、Restart=no。
- CPU8–11/4线程、RSS4GiB/AS8GiB、900s advisory未超。最终程序**210.680747897s**、launcher**217.938652844s**、进程报告peak RSS **2010116096B**（约1.872GiB）；summary写盘前210.649606988s不作最终全程。0模型/前向/优化/GPU/score重算/分区或物理预测变化，0 reference数组/T_test读取；身份在特征冻结后评价时读取。
- 128失败的已消耗成本单列在原失败Record；130包含6场复制/绑定与26场新构造，不把复用当免费独立重复或将两次总成本只报130。128最后记录2010796032B不是已获真实终态MemoryPeak，其真实峰值仍未知。130这次完成不证明分块在任何宿主压力下均免于OOM。
- 生产导入支路核6场features/evaluation/public row byte-exact及原输入/标签/score/support绑定；原128不改、前6特征不重算。当前公开逐场所有候选/known/pure/mixed计数与summary逐层重聚一致，32场唯一、汇总32；这类JSON自洽不认证全部私有几何/身份/特征。
- 上节独立事后同人口排序只读取32个保存evaluation，不重构特征/改候选/读坐标或reference。一次CPU8、20s硬上界内0.199321s、RSS36438016B，独立平均秩产物已回传；没有守护、后台排队或失败自动重试。本补充审计范围仅该三mask的标签/scorer排名，不是原特征计算正确性、数学原件或完整主审计认证。
- 正式独立保存数组核验已通过：冻结46902b3，服务mtad-66c711f32911.service成功exit0、inactive/dead、Restart=no；独立平均秩统计、不调用生产评价函数。29221exact/4154numeric/1826null，逐场/全量/9组件/去最大/原6/新26、十scorer全部13分层及scene-equal核对；重新按056身份计算纯度、按候选成员核size/source_count，绑定保存feature列和scorer符号，并核原6场三文件byte-exact。程序2.064682s/launcher2.721990s/RSS40767488B。未重构几何/LOO特征或候选生成、未独立认证标签打开时序/reference物理语义/分区F1/源健康；所有冻结worktree干净。没有新增训练、阈值/方向搜索、随机重复或置信区间。候选重叠、同轨迹query和共享组件导致相关性，不作4025候选IID显著性。

| 计划项 | 实际值/偏差 | 解释边界 |
|---|---|---|
| 原128同32场/9组件、10 scorer | 完整恢复终点 | 只诊断固定原组＋单节点扩展，不覆盖任意对象集 |
| 前6完整导入、26新构造 | 按合同执行，4096pair分块 | 原6复用不是新复现；原失败保持 |
| 数值公式/原L1fallback/容差 | 负控最大差在冻结范围内 | 欧氏恒等不扩大为一般球面统计定理 |
| null与单类保留 | 596共同无定义、395 singleton、单源shared仅8全mixed | 几何覆盖依赖support，不能填坏分/删场 |
| 原组／扩展／二成员同人口交集 | 完成后追加一次独立保存evaluation计算 | 明确事后描述，不倒填为预注册主结论 |
| 系统效果 | partition/物理预测未变 | 无新raw/decoded F1、位置、coverage/duplicate效果 |

## 8. Squeeze：关闭当前固定LOO升级解释，保留明确发展问题

**DE：** 当前LOO可定义候选范围有限且纯类覆盖更低；二成员与pair完全同数，局部欧氏多成员平方统计可重构；全32固定LOO scorer不超静态score_min/shared控制，原组同人口中尤其弱。最初“以留出一致性形成新增信息/更强组判据”的强解释不获本项支持。128资源失败不是科学负结果；130的真实完整科学诊断才提供这些数值。

**AI：** 原组已由相同教师/WS筛选，几何接近可能仍对应平行异船、共享偏差或support选择；聚合与候选大小也可能改变AUC。这些解释与观察相容，尚未用独立替换/消融区分，不能说发现物理根因。将距离符号后验翻转、换候选或按V择阈值，均会形成另一配方，当前不做。

**UN与可发展空间：** 当前结果没有否定原生时序、运动残差或完整pair/support的学习表示价值，也不要求绝对原创。若后续矩阵认为原组错误识别仍有价值，需先明确目标是修原混组、拒配还是对象覆盖，再用合法T监督或冻结规则，配同信息/同容量强控并真实评价分区、覆盖、重复、位置和成本；不能仅堆LOO名字就算创新，不能据合并候选MIN高AUC直接部署。当前固定LOO直接升级配方到32场终点结束，不自动训练新头或扩场。

正式保存数组限定统计核验通过后回流Mode C，保留限制与原组/扩展差异，不自动重跑科学配方。本报告不改变研究状态、队列或发布新实验。

## 9. 原件与标准交接

[必要脱敏原件](/Users/a1/Projects/biye-ronghe/docs/evidence/20261029_small_validation/130/)保留run/summary/完整scene-summary、真实launcher/退出/服务终态、新事后common比较；私有feature/evaluation/identity/几何矩阵留原服务器130，原128失败另存且未覆盖。

```yaml
package_type: EXPERIMENT_RECORD
package_version: research-pipeline-contract v1.0
experiment_id: EXP-130-20261029
source_idea_id: IDEA-ASSOCIATION-GEOMETRY-CONTINUATION-20261029
plan_version: '1.0'
execution_status: completed_diagnostic_exit0
research_question: 固定候选LOO能否超过原score及完整pair/support静态控制且具有足够定义覆盖
hypothesis: 表征增量待验；二成员及指定欧氏平方统计信息冗余条件已限定
planned_vs_actual_design_and_controls: frozen128_design_V32_10scorers,6byte_exact_import+26chunk_construct,posthoc3_common_masks_separate
actual_configuration: frozen64ac7d4,CPU8-11_4threads,RSS4GiB_AS8GiB_900s_advisory,pair_block4096
samples_materials_or_data_provenance: frozen056_Q64_allpairs+114_WS+manifest+failed128_six_complete_artifacts
results_with_uncertainty: common3429_pure803_mixed2626;MIN_AUC0.892691_vs_LOOmedian0.448240_RMS0.527186;no_IID_CI
all_runs_or_repetitions: failed128_preserved;one130_continuation_once;posthoc_rank_read_only_once
observations_and_deviations: 596undefined,original_LOO_weak,extensions_predominantly_mixed,no_system_output_change
actual_cost: program210.680747897s,launcher217.938652844s,peakRSS2010116096B,0model_forward_optimizer;posthoc0.199321s_36438016B
known_limits_and_missing_evidence: saved_label_population_rank_audit_passed_feature_generation_unverified;no_full_feature_recomputation;no_independent_testing;no_matched_learned_heads
next_routes: squeeze_ModeC;no_automatic_new_experiment
next_skill: squeeze-experiment
```

```yaml
package_type: SQUEEZE_REPORT
package_version: research-pipeline-contract v1.0
squeeze_id: SQ-ASSOCIATION-GEOMETRY-130-20261029
input_experiment_record_ids: [EXP-130-20261029,EXP-128-20261029]
evidence_status: partial_saved_label_population_rank_audit_passed
claim_scope: fixed_V32_candidates_and_ten_prespecified_scorers;no_partition_or_physical_prediction_change
primary_result_with_uncertainty: same_common_MIN0.892691_LOOmedian0.448240_RMS0.527186;LOOmedian_vs_MIN_0better2tie30worse_scenes
process_dynamics: 6_imported26_new;bounded_pair4096;original128_failed_oomd_true_peak_unknown
mechanism_and_intermediate_findings: DE_two_member_identity_and_stated_euclidean_pair_reconstruction;AI_selection_support_and_geometry_aliasing;physical_root_cause_UN
errors_heterogeneity_and_boundary_conditions: pure_definition_coverage59.57percent_vs_mixed98.09percent;original_common_LOO0.358046;largest_component_LOO_below0.5
robustness_sensitivity_and_alternative_explanations: full_vs_scene_equal,9components,without_largest,all_strata;posthoc_original_extension_size2_common_independent_average_ranks
new_P: 如何在已有原组及合法信息下识别真实对象错误而非主要分离单节点扩展
new_F: fixed_LOO_not_above_same_candidate_static_controls;original_vs_extension_rank_difficulty_and_definition_coverage_differ
new_M: DE限定代数冗余；AI共享偏差/平行异船/支持选择；UN原生运动或学习表示是否有增量
evidence_index: 130run_summary_scene_launcher_terminal_and_posthoc_common_candidate_comparison;128_failure_record
open_questions: feature_generation;matched_learned_heads_if_independently_justified;actual_partition_risk_and_native_temporal_increment
next_routes: deep-read_ModeC;do_not_directly_publish_new_head
next_skill: deep-read Mode C
```
