# 126：强22维之外的时间形状可学信息验证

2026-10-29，执行前Idea/实现规格 **1.1**。用户本轮明确要求将成熟输入方案落实成实验。消费[输入/时序局部矩阵§8](/Users/a1/Projects/biye-ronghe/docs/surveys/2026-10-27_expanded_search/input_time_action_matrix_20261029.md#8-review后修订先测时间形状的可学增量再独立检验原生状态估计) IT-C4（IT-P1，IT-F15/16/19，IT-M8，IT-E14/15/18），沿[原设计§2](input_time_signal_pilot_20261029.md)收敛为一次可执行训练。真正执行仅认主仓HandOver，不由本文件自行发布。

## 问题、证据与候选机制

现有22维已含距离/运动/支撑等强摘要；没有mean_diff字段，也没有110上的该字段模型消融。这些既有事实不证明摘要损失了有用身份信息。固定node级证据时，部分query汇总不保留完整时间次序，是算子层条件事实；实际额外次序能否判别身份仍UN。

本项检验：同canonical_all/Q64、同全部i<j候选、同监督和训练预算下，保留一部分距离时间形状，能否超过强摘要及同维无序/置乱控制。候选链是“部分顺序未显式表示→有序形状含额外可学身份信号→同容量探针能利用”，标 **AI/待验**。额外分布统计、有效容量、抽样及V阈值开发都是竞争解释，不能先认输入缺陷、先立EQ-Reg或将成功自动写成海上首创。

近邻和阅读范围复用已完整AP及局部矩阵的真实22维实现、EQ-Reg与native时间来源，不新增完整精读计数。成熟DCT/MLP本身不是原创；若适配有效，价值须由目标任务的新增信息、条件及后续同协议强控确定。没有新证据时不反复读同类或追加频率。

## 固定数据与四臂

- 正式Level1当前manifest：T_core2100/V_select300，scene划分但共享身份；T_test封存。读取正式三分卷量测及身份标签，**不读reference CSV/数组**。identity仅供采样/BCE及评价，不进入输入特征。
- 原`_record→prepare_timed_scene→scene_motion_evidence(canonical_all)`链保持原节点/候选、Q64/20s编码单位、FP64物理预测与原22维`features_from_motion`。V侧重建的node_ids/sources/identities以及states/support/query/nearest/exact_observation必须与056原system保存白名单数组exact绑定；失配失败保全，不择样或按误差修复。20s不当真实采样间隔。
- 对pair在Q64计算`z_q=log1p(Haversine(p_i(q),p_j(q))/1000)`。有序通道为`z−mean(z)`前8个非DC正交DCT-II系数；无序通道为z的.1… .8分位数；置乱通道先逐pair固定重排z，再取相同8系数。

| 臂 | 输入30维 | 用途 |
|---|---|---|
| SUMMARY | 原22维＋8恒零 | 同新预算的强摘要控制 |
| ORDERED | 原22维＋有序DCT8 | 测适配总增量 |
| DISTRIBUTION | 原22维＋无序分位数8 | 区分次序与额外分布信息 |
| SHUFFLED | 原22维＋逐pair置乱DCT8 | 有条件次序阴性控制 |

四臂均`30→64→64→1`、SiLU、6209参数。共享同一真实初始state：前22列/后层完全相同，新8列首层权重0、末层权重及bias0，初始函数都是同FP64 OLS base。全部臂输入参数预算相同，SUMMARY八零列不激活，**不声称有效容量严格相同**。零初始化不会永久阻断新通道，必要软件回归已检查后续更新可得梯度。

## 1.1相对原概念设计的实现收敛（执行前）

1. **磁盘缓存，保留人口。** 已知T约27632584 pairs，不把四臂30列同时驻留RAM。逐场只保存共享原22＋三种8通道共46列float32 `.npy`及FP64 base、原pairs/labels；normalizer流式、训练按批mmap读行，用完关闭映射。Q64联合support计数仅0…64，存uint8不改变值。缓存约数GiB，实际字节和所有准备/写盘/读盘成本均记录；不是删场或缩训练预算。
2. **固定计数式置乱。** 为避免约2700万次Python RNG初始化，以seed20、T/V split序、完整manifest scene序和原pair序，配合64个query计数，通过明确uint64 SplitMix64混合产生rank键并逐行argsort。每pair固定不同置乱、每pair的z多重集合exact保持；chunk边界不改变pair键。没有label/误差/V效果参与，不用全数据同一个可逆置换。它是确定性伪随机控制，**不是精确置换检验或独立随机性定理**；单次置乱及截断统计的信息量/尺度与ORDERED不完全相同。该算法收敛已在看本实验V结果前固定。
3. **专用末步runner。** 既有通用fit会step0/中期V择checkpoint，本项不调用其训练loop；仅复用原采样器、feature/metric等纯函数。每臂1000成功步完成后，对全V进行一次推理、保存最终模型/完整scores，然后开发阈值；不在V选步，不恢复旧权重。
4. **单GPU顺序四臂。** 共享实际初值与预生成批次方便核对，总量仍4000成功步。RSS8GiB，CPU≤4，一张现场FIFO兼容GPU按UUID绑定、进程内cuda:0；CUDA虚拟地址空间不设8GiB硬AS限制，实际RSS及GPU内存另记。原2小时为全部准备/训练/评价的advisory，超建议记录进度继续固定任务，不自动增加训练或换配方。

## 训练、阈值与统计对象

seed20；原`BalancedScenePairSampler` scene→class内pair策略，batch256正负各128，scene和pair原序保持。执行前生成并保存同一1000×256×2的(scene_index,pair_index)实际计划；四臂逐批相同，不能以同seed替代实际绑定。只有1000批覆盖到的T样本参与更新，报告全部场的计划/实际暴露，不能说所有T pair训练了一遍。

每臂Adam lr.001、clip1、BCEWithLogits，FP32网络/缓存输入，FP64 base和最终相加/score。每臂固定1000成功更新，总4000；全部四臂final checkpoint/optimizer state保留。出错保存已完成history、partial checkpoint（保存错误另记）与原run，不自动重试或覆盖。

每臂只对最终模型进行一次全V推理；用既有exact `select_threshold` 最大化场景macro-F1，保全all-reject/≥/并列较高threshold约定。阈值V开发且已有反复暴露，不能称独立确认；不按V挑频率、宽度、步数、seed或置乱。所有阈值与全部候选scores留服务器。旧056若引用只能列背景，不将其不同头初值或旧分数直接归本轮方法收益。

主要估计量是ORDERED分别相对SUMMARY/DISTRIBUTION/SHUFFLED的完整V场景macro-F1差。六个无序配对全报告wrong→right/right→wrong/FP/FN及逐场。各臂并报完整P/R/FP/FN/AUC、同源/异源全候选、joint/no_joint support、全部38组件与去场景数最大组件；空场/空任务层按现有null与分母规则保留。

论文ITSA/MSTSA另用原生片段intervals及既有`paper_masks/scope_metrics`：ITSA同源gap∈[0,1200)、双方span>120s；MSTSA异源overlap>120s。与“全部同源/异源”的项目补充表分开，同一全人口阈值不在子任务重选，列mPCA/mwPCA/mPFA/mwPFA。两者条件不同，不能混称作者任务已复现。当前只验证raw身份信号，不改变分区/物理输出，不宣完整融合或同行胜出。

场景/组件差与去最大层只描述异质性，单seed、开发V、共享身份不支持普遍显著性。当前不把edge当IID做CI；所有负结果、空层、候选/节点/正负数及成本完整回流。

## 解释与停止规则

- ORDERED对三强控均有方向一致且不被最大组件独占的增量：支持进一步发展保形输入，仍不能证明某个物理根因或完整方法创新已成立。
- 只胜SUMMARY而不胜DISTRIBUTION：更多分布统计是竞争解释，不支持时间次序独占贡献。
- 与SHUFFLED相近：当前8系数探针未给稳定次序证据；单次伪随机控制不是统计零分布。
- 无增量/反向/强子组损害：当前8系数配方到终点结束，不自动加频率、seed、宽度或更新。负结果不否定原生间隔、方向或全部时序家族。
- 输入绑定失配、非有限、OOM、代码错误即失败保全，不改人口/clip/lr或降精度续跑；建议墙时超时不是自动失败。真实坐标、身份、权重、大缓存和scores留服务器，只回传必要脱敏原件与全逐场/组件/分母/成本。

## 实现与必要验收

| 文件/行为 | 验收 |
|---|---|
| `tools/analysis/ordered_shape_signal.py` | 正式T2100/V300单次准备、全部候选、磁盘共享cache、1000实际批次和6209初值共享；0ref/test |
| prediction-only形状API | pair交换对称、chunk键一致、有序对次序敏感而分位数不变、逐pair置乱保多重集合 |
| T-only normalizer/训练 | 共享22列统计exact，V拒拟合；固定实际256批次、四臂输入映射；末步final-only |
| 失败与来源保护 | 旧056白名单exact且不读reference数组；输出exist_ok=false，失败保留partial/history，不覆盖旧run |

5项针对性软件回归已通过；另核0行`.npy`/memmap可保留。它们只认实现证据，不能当真实MTAD效果或作者复现。真正执行和全部Record/Squeeze/Mode C仍由主仓协调。

## 真实CLI（路径已由主代理只读核验）

```bash
python tools/analysis/ordered_shape_signal.py \
  --archive /home/wangguangjie/djs/data/mtad/数据集level1.zip.001.zip \
            /home/wangguangjie/djs/data/mtad/数据集level1.zip.002.zip \
            /home/wangguangjie/djs/data/mtad/数据集level1.zip.003.zip \
  --manifest configs/data/mtad-level1-scene3000.json \
  --source-system /home/wangguangjie/djs/association-code/runs/LEVEL1-ASSOCIATION-20261022-056-attempt-1/system \
  --output /home/wangguangjie/djs/association-code/runs/LEVEL1-ORDERED-SHAPE-SIGNAL-20261029-126-attempt-1 \
  --device cuda:0 --cpu-threads 4
```

由主代理冻结新checkout、按systemd launcher和`--time-policy advisory --max-seconds 7200`托管一次；现场GPU UUID/CPU12–15等只写真实交接/原request，活动代码不pull。此稿不自行发布READY。

```yaml
package_type: IDEA_PACKAGE
package_version: research-pipeline-contract v1.0
idea_id: IDEA-ORDERED-SHAPE-SIGNAL-20261029
plan_version: '1.1'
source_matrix_id: MC-INPUT-TIME-ACTION-20261029@review-correction
selected_C_id: IT-C4
source_P_ids: [IT-P1-refined]
source_F_ids: [IT-F15, IT-F16, IT-F19]
source_M_ids: [IT-M8]
source_E_ids: [IT-E14, IT-E15, IT-E18]
research_question: 强22维之外的8系数有序距离形状是否含可学身份增量
hypothesis_and_mechanism: 只有在剩余次序含额外判别信号时有序通道应超过三强控；目标机制未证
testable_predictions: ORDERED对SUMMARY/DISTRIBUTION/SHUFFLED的raw场景macro-F1差及组件异质性
design_and_controls: 四臂同6209初值/实际1000批次/监督/参数预算，全部六配对与末步一次V
sample_materials_or_experimental_units: 正式T2100/V300原全部节点及i小于j候选，共享身份开发划分
primary_outcome_and_estimand: 完整V300场景macro-F1的三个主差
secondary_outcomes: 全P/R/FP/FN/AUC、正式ITSA/MSTSA八指标、同源支撑/组件、参数内存全链成本
analysis_plan_and_uncertainty: 单seed/开发V确定性配对，组件和去最大层；不作edge-IID显著性
resources_time_cost_and_risk: 单GPU/CPU4/RSS8GiB/2h advisory/4000总成功步，约数GiB磁盘cache，全部计成本
ethics_safety_privacy_or_compliance: 只读量测和identity标签，reference/test不读；私有cache权重scores留服务器
stopping_and_interpretation_rules: 本文出口；固定末步或真实故障，失败保全无自动重试扩量/改参
status: pilot
evidence_status: partial
missing_evidence: [actual_learned_increment, independent_generalization, matched_credible_peer]
next_skill: record-experiment_after_actual_execution
```
