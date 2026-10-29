# 125：T侧合法输入能否判断借高源的条件收益

2026-10-29，执行前Idea/规格1.0。消费Mode C1.44及本轮1.45的CC-C1；用户明确要求实际验证T侧条件收益。先固定研究问题/比较/资源，再由HandOver发布；本文本身不是READY，不把候选原因写成已证机制。

## 1. 问题、来源和机制边界

124固定soft=.75相对FixedLOW有20/32场改善、更多点改善，却总体MAE+83.951m，损害幅度超过收益；123的LOW在未见组件又输FixedMean。问题是**合法条件下借高源的预期误差变化是否可预测**，不是“证明低源更干净”或再扫描一个常量alpha。来源为真实123/124 Record/Squeeze、贡献/恢复代码及当前原生数据接口；创新价值若存在是可归因地改善条件贡献，未声称首创回归、门控或解决了概率可识别性。

令固定候选hard=FixedLOW、soft=低源.75/高源.25的同中心支持均值，监督`y=e_hard−e_soft`（径向米误差，正数表示借高源有益）。若知道真实条件均值E[y|x]，按其正负选动作可对应预期MAE决策；**此为条件代数，不是本项目新定理，也不保证有限样本ridge可估准或跨域转移**。分类正负的概率不足以反映损害幅度，因此主探针直接回归有符号收益，AUROC只辅助。错组、共同偏差、机动、reference关系与输入不足仍为AI/UN。

## 2. 固定数据与冻结上游

- T为manifest前32个T_core（当前12共享组件），V为原118/121/124的32场/9组件；T_test封存。125第一轮不扩268：056缓存没有完整native_count/fit_rmse/OLS速度，127已提供T32/V32完整合法motion，先闭合这个有界探针。此前实现前拟V300未发布/未运行，现因真实字段可用性冻结V32，不看125结果改范围；不从Q64拟合伪native质量。
- 只读127不可变`native-cache.json=CACHE_COMPLETED`及分开的native-motion/native-evaluation；不修改127活动目录，不需其全部九rho评价结束。物理文件不带身份/reference；评价文件在预测/特征落盘后才显式读取。
- T侧加载原056 `association/final.pt`和原normalizer，按其FP64几何base＋FP32 residual规则分块打分，原θ=1.7320103112387182；同114 signed-margin sum WS，身份不作分组。当前教师已用全T2100训练、θ历史在V开发，**LOCO只隔离新收益head及其normalizer，不是全链未见组件/身份验证**。
- T同合法预测组按原三fold heldout_centers估相对共享项，保持经验低源gauge、原回退及全部输出；源排序不是物理可靠性标签，m不称meff。
- V原六强控（Mean/FixedMean/LOW/FixedLOW/Anchor/NEAR）从123相同原32来源121导入，固定soft.75从124导入，共七旧臂完整artifact逐字段exact；只新增四个head动作，11臂55配对。不在V重新拟合中心、normalizer、回归、阈值或候选alpha。

## 3. 输入、监督与四个探针

15个特征均为预测时可见query或组量：log1p的校正后两源均值距离/km、未校正距离/km、当前支持成员两源几何OLS平均速度差/mps、低/高源各拟合残差均值/km、低/高源各nearest均值/min、低/高源各native_count均值、低/高源各支持成员数、当前冻结中心norm/km、其它fold eligible组数；以及组内全pair signed-margin均值与负比例。拟合残差不叫协方差，成员数不等独立样本数；不输入绝对坐标、身份、reference、真实目标support或V误差。

先生成全部组/query特征与hard/soft并持久化，再读取T身份/reference。按主评价的首anchor固定对应、同一target仅首组为代表赋标签；重复组/无参考/非both支持不进入收益拟合，但保留所有节点/组/缺标签原因，不能用身份挑纯组或改变预测。y不clip，保留风险幅度。只用both且ref位置有效的首代表query拟合。

四模型：T收益均值常量、线性ridge（16系数）、二次ridge（136系数）、同二次但T训练y固定seed20置乱负控。每个组件留出fold只在其它T组件上计算15列均/std，z裁[-8,8]，展开`[1,z,z_i z_j(i≤j)]`，不对展开项二次标准化。目标`mean((phi beta−y)^2)+1*||beta_nonintercept||²`，截距不罚，λ=1固定；没有超参搜索、SGD或V择头。常量为同训练y均值；无标签训练fold/model固定全0预测→hard并报告缺证据。

T按整个component leave-one-out（当前12fold），每fold同四探针，最后全T再拟合四模型并冻结。置乱仅各fit的训练标签、一次seed20，不读heldout/V，不当精确随机化检验。记录原fold/参数/normalizer/置乱于private。预测动作固定`predicted_gain>0`且both支持才选soft；其余域始终原hard exact。无标签的query也按合法特征推断，评价有无标签与动作统计分别报告。

## 4. 结果、反证和归因

主探针是二次head；常量/线性/置乱均完整强控，不按V挑最好头。T OOF分别报告eligible both/refvalid条件人数、所有首代表matched refvalid共同点分母、AUROC、实际净收益和oracle regret，逐场/组件完整保留。全T共同点MAE变化为`−sum(y*action)/all_matched_refvalid_count`，不能把eligible条件均值当全人口成绩。该反事实差只定义在两个固定候选和同一reference协议内，不认证物理源健康。

全T拟合冻结后，V32一次所有11臂55配对。报完整径向MAE/RMSE/P95/P99、坐标ME/MAE、coverage/缺失/重复/无reference、both/单源/no_support及真目标support事后层、32场/9组件/去最大组件和动作率/代价。真目标support只评价，不能用于gate。所有动作完成再打开V参考/身份，不把score当校准概率。

- OOF净收益不胜常量、V不利或只少数组件有利：如实限定该探针，无“质量机制已发现”。
- 只胜hard而不胜constant/linear/置乱：不能归因非线性条件信息；保留更简单控制。
- OOF有益但V不转移：保留上游教师/共享身份/标签缺失/分布变化等解释，不从V改λ、alpha、阈值或再选特征。
- 即使整体及尾部/组件都改善，也只支持本有监督条件贡献，物理方差/无偏锚点/可信同行及在线端到端仍未证。

## 5. 资源、实现与一次终点

CPU0–3/4线程、RSS6GiB/AS12GiB、1800s advisory；T32 LOCO＋全T四模型，V32十一臂终点，一次attempt-1。拟合量按实际component计数，当前12fold×4＋4＝52个解析head fit（constant和ridge另计）；0新增神经优化更新，但构造1个冻结056模型并进行真实T前向，必须报checkpoint/forward/参数成本，不能写0模型或0训练准备成本。

实际原生读取已由127发生，125计自身cache读取/teacher评分/特征/拟合/评价/写盘；不把共享准备成本藏掉。错误/OOM/输入或artifact绑定失配/非有限保全，无自动重试、删难例、调参数、扩V268或自动后继。坐标、身份、中心、teacher和head权重/大预测留服务器，仅必要脱敏原件回传。完成后Record→Squeeze→Mode C。

软件验收：合法特征API无identity/ref；整组件排除与T-only normalizer；独立岭回归正规方程/裁剪；gain符号/首代表/重复与两种分母；无标签回退；outside-both exact；T控制与V旧七artifact exact、十一臂55配对。合成检查不作为真实目标效果。

```yaml
package_type: IDEA_PACKAGE
package_version: research-pipeline-contract v1.0
idea_id: IDEA-CONDITIONAL-SOURCE-VALUE-20261029
plan_version: '1.0'
source_matrix_id: MC-RESTART-20261027@1.45
selected_C_id: CC-C1
source_P_ids: [P11, P14]
source_F_ids: [F221, F222, CC-F3, CC-F4]
source_M_ids: [M92, M93, CC-M1]
source_E_ids: [E119, E120, NATIVE-CACHE-127, evidence_association.py]
research_question: 合法条件能否在T预测借高源的有符号收益并冻结到V
hypothesis_and_mechanism: 条件收益可学仍AI；预测目标是参考风险差，不是源方差
sample_materials_or_experimental_units: T32/12组件head-LOCO及原V32/9组件，教师曾全T训练
primary_outcome_and_estimand: 二次head相对常量及hard的T OOF实际全共同点MAE差和V完整位置风险
secondary_outcomes: 线性/置乱/原七强控、尾部/覆盖重复/所有配对/组件/代价
analysis_plan_and_uncertainty: 原文完整分母/LOCO仅隔离head，单次开发，无query-IID显著性
resources_time_cost_and_risk: CPU4/RSS6GiB/AS12GiB/1800s advisory/一次，52解析head fits及冻结teacher真实前向
ethics_safety_privacy_or_compliance: T监督与预测分离、V冻结后评价，private留server，test封存
stopping_and_interpretation_rules: 本文出口，错误保全无自动重试/调参/扩量
status: diagnostic
evidence_status: partial
missing_evidence: [conditional_value_signal, V_transfer, physical_quality_identification, credible_peer]
next_skill: record-experiment_after_actual_execution
```
