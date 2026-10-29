# 129：原生连续性候选、T冻结拒配与实际输出合并

2026-10-29，执行前Idea/规格1.0。消费[OA矩阵§7](/Users/a1/Projects/biye-ronghe/docs/surveys/2026-10-27_expanded_search/association_output_action_matrix_20261029.md#7-2026-10-29-review后修订留出一致性先与完整pair几何对齐对象化保留拒配)的OA-C5r，P8/F16/19–22、M8/E16/19–22；[原设计](association_object_signal_pilot_20261029.md)§3的native/拒配条件由127冻结cache建立。用户本轮要求成熟方案实际运行；具体领取只认HandOver。

## 问题、工作假设与新增价值

全V1068个重复首anchor组不等1068条纯断轨。本项不按真身份并组、不保证补覆盖，而检验真实native时间连续性构造的有限合并假设，在保留拒配/原组与完整人口后，是否能减少重复并改善最终输出。B机制是有限轨迹假设、排他/未配分支，借鉴Draves/TPMBM/MeanTraj的条件而非复制原始量测后验；相同海上匿名fragment适配C仍未核，有效性待本项实测，不因已有先例自动否定。

明确的**工作假设AI/UN**：同源同时至多一个fragment解释同一对象；首末端点间恒速度可提供连续性排序。原生重叠可能来自重复航迹/切段等，不能宣重叠必异船；端点可被噪声/机动/既有滤波/时差污染。原WS里已有冲突不拆、不删、不靠标签修复；新union只在工作约束下允许。

## 固定输入与T侧拟合

- 127 `native-cache.json` 必须`CACHE_COMPLETED`并冻结，含manifest原前32个T_core与原114事前V32/9组件。消费`private/native-events/{split}/scene-{sid}.npz`中真实node/source/event谱系、秒/latlon/valid；不是Q64反插值。native-evaluation中`node_ids/identities`只在特征落盘后显式加载，129不读其reference字段。
- 每node起止为全部原生事件时间范围；端点位置/速度只用位置valid的原生事件。无原生事件/无合法位置/少于两点分别计数；少于两点用零速度工作回退，无位置保留节点但无连续性信号。不删节点或难场。
- 对每node，只在**同source且其native结束严格早于另一node开始**的完整候选中，按“最后两个合法位置的球面恒速度预测另一片段首个合法位置”的距离选top2，平分按时间间隔及原node序；不足两候选全保留。候选库含所有原节点singleton/原WS组，不让真身份筛候选；所有可见同源前后对、top2保留率与事后真链接召回分开。
- T32逐场先保存完整候选/距离/端点可见状态，再读身份定义同目标标签。只对有限候选的有标签有限距离，以`distance<=threshold`拟合**一个全局balanced-accuracy阈值**；扫描全部T唯一距离含reject-all，BA并列选接受边最少的阈值。选择/全阈值训练曲线保存，不读V效果选阈值。若T无候选/无正例/无负例/无可用标签，则固定all-reject并报告不可识别，不暗改尺度/删场/存JSON inf。
- BA只是此监督分类目标，未计对象误并/重复/尾部成本，不称最优集合风险或校准概率；T/V共享身份/组件明确披露，非独立身份泛化。

## V侧三分区、六臂与可归因控制

V仅用冻结T阈值，所有候选先冻结后评价。三分区：原WS；`NO_REJECT`原WS上不加阈值的连续性合并；`T_REJECT`同候选/排序/约束加冻结T拒配。候选node边映射为当前组之间的提案，按距离、间隔、原node序从小到大处理；已同组跳过，所有新union须无同源native闭区间重叠、无重复event谱系冲突。重复原事件记录如属性冲突须失败，不静默去重。

原组内部若已冲突完整保留并计数；不能据此拆纯组。union成员保持原node序、首代表仍最小原序；每原node恰归一组。可见约束/阈值拒配、已同组和实际合并均计数，private保存原候选和选择轨迹，身份在这些动作锁定后才用于评价。没有联合多假设最优求解/校准似然/在线权限主张。

每分区用同原056 Q64状态接**Mean或LOW**，共6臂15个无序配对。LOW仅按可见source排序/support选位置贡献，速度航向复制各自同组Mean，空支持回相应首anchor。不加入118旧center或收缩，避免把对旧组估得中心误用到新组再称输出增量。原WS的Mean/LOW全artifact逐字段exact重放121；失败即保全。FixedLOW只作历史参照，不拼当前六臂冠军。

预测分区/六臂输出全部冻结后才读056 reference/身份，按主协议完整目标、首anchor固定对应、FULL Q64评价；同场所有pair原scores/raw不变，三分区decoded全pair F1/P/R及ITSA/MSTSA另报。main不能偷换为新Hungarian匹配。

## 指标、资源与停止

主估计量：T_REJECT相对WS/NO_REJECT的完整分母重复数、目标点coverage、共同点径向MAE与全pair宏F1；同时报告各自全人口风险，不能只用相同点删覆盖损失。RMSE/P95/P99、gained/lost/rescued目标、mixed/切分/组数/首代表变化、同源重叠冲突、候选容量/真链接召回、拒配/误并、T阈值训练曲线、逐场/9组件/去最大组件与全部15配对完整保留。真身份/target support仅事后分层，不用于合并/拒配。

若重复少但误并/覆盖/尾部更坏，记录真实取舍；若无合并，区分候选/约束/阈值作用，不宣对象化无效。若NO_REJECT已解释全部效应，新增T门控不晋级；若T规则跨组件失败，结束当前配方，不由V重选阈值。原生重叠工作约束排掉真链接则报告条件限制，不改成从标签特许。

一次attempt-1，T32一次候选阈值拟合＋V32六臂固定终点，CPU≤4/RSS4GiB/AS8GiB/1800s advisory，0神经模型/前向/优化/GPU/新关联score/normalizer/ZIP/CSV/T_test。读取已完成native cache可与127后续评价并行，不能修改127活动目录/文件；本项与128各有独立run/合同。代码错误、绑定失配、非有限、event冲突即失败保全，无自动重试/择参/扩量。private坐标/身份/谱系/轨迹留server，真实输入/命令/退出/成本及完整公共逐场回传，Record→Squeeze→Mode C。

```yaml
package_type: IDEA_PACKAGE
package_version: research-pipeline-contract v1.0
idea_id: IDEA-OBJECT-HYPOTHESIS-PILOT-20261029
plan_version: '1.0'
source_matrix_id: MC-ASSOCIATION-OUTPUT-ACTION-20261029-REVIEW-R3
selected_C_id: OA-C5r
source_P_ids: [OA-P8]
source_F_ids: [OA-F16, OA-F19, OA-F20, OA-F21, OA-F22]
source_M_ids: [OA-M8]
source_E_ids: [OA-E16, OA-E19, OA-E20, OA-E21, OA-E22]
research_question: 合法native连续性假设与冻结T拒配能否改善完整对象输出且控制误并覆盖代价
hypothesis_and_mechanism: 有限候选/排他/拒配可能减少片段式解释，native重叠与CV仅工作假设
testable_predictions: 同候选NO_REJECT与T_REJECT的实际合并及全人口关联/重复/覆盖/位置尾部差
design_and_controls: 原WS/NO_REJECT/T_REJECT三分区乘Mean/LOW，六臂15配对，旧两臂exact
sample_materials_or_experimental_units: 127冻结native T32/V32和056/114/121原V32缓存，非独立身份测试
primary_outcome_and_estimand: T_REJECT对WS和NO_REJECT的重复/coverage/共同MAE/全pair宏F1
secondary_outcomes: 全人口尾部/首代表/候选/冲突/拒配/组件及真实成本
analysis_plan_and_uncertainty: T-only一个BA阈值，单类all-reject，V固定；无query-IID推断
resources_time_cost_and_risk: 单次CPU4/RSS4GiB/AS8GiB/1800s advisory/T32+V32/0神经训练
ethics_safety_privacy_or_compliance: 动作先冻结再身份评价，private留server，T_test封存
stopping_and_interpretation_rules: 固定终点或错误保全，无自动重试改阈值扩量
status: diagnostic
evidence_status: partial
missing_evidence: [native_input_binding, candidate_recall, rejection_transfer, full_output_effect, credible_peer]
next_skill: record-experiment_after_actual_execution
```
