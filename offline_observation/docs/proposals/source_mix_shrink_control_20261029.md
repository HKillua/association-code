# 源贡献软混合与共享中心收缩：分离验证

2026-10-29，执行前 Idea 1.0；本稿由用户要求和[提案审查](../reports/2026-10-29-proposal-precision-shrinkage-review.md)修订。**先完成123新268的Record→Squeeze→Mode C回流，再由主仓交接发布本项；本稿不是READY。** 下列参数在查看123结果之前确定，不根据扩展结果选权重、收缩强度或场景。原118/120/121运行保持冻结。

## 证据入口与问题

**执行发布补记：** 123的Record/Squeeze及Mode C1.43现已完成，新268校正增量延续而多数场景/P99冲突仍在。本设计的参数和人口此前已冻结（3de43f4），没有据123误差改参；首段“本稿不是READY”为设计时的状态记录，实际领取只见主仓HandOver的独立124条目。

领域/对象是MTAD Level1海上多源航迹位置恢复，当前是重复开发V、离线Q64的内部控制。采用discover-innovation Pattern 2，从风险异质性出发，比较两个独立可替换操作；未预选最终网络骨架。

- 上游Mode C1.41的F218/M89/E116对应120，F219/M90/E117对应121；选择原C10/C51的相对共享校正及[融合局部矩阵§6的SC-C3](/Users/a1/Projects/biye-ronghe/docs/surveys/2026-10-27_expanded_search/fusion_action_matrix_20261029.md#6-118回流后的下一层问题相对校正绝对锚定与真正多源增量)。SC-P1/P2、SC-F7–9、SC-M3/4、SC-E7–9保留原含义。本稿使用局部MX标签，不覆盖现有全局编号。
- **DE：** 121同71071点LOW/FixedLOW MAE为527.446/447.060m，FixedMean656.041m；FixedLOW相对FixedMean有20/32场较差，相对LOW P99差265.615m。最大组件19场/58707点；其他13场/12364点的LOW684.111m反而劣于FixedMean622.008m。全点风险、场景多数和组件均须分别报告，不能只选有利层。
- **DE：** FixedLOW与LOW会在high_only及首anchor来自高source的no_support处不同；both/low_only主要同值（不将低位浮点差当效果）。high_only存在624个受损点，并未证明损害来自中心小样本方差。
- **AI/UN：** 两源混合可能以增加偏差换取较低方差，也可能只把较差源重新引入；中心收缩可能减轻不可靠修正，也可能使原来有效的校正不足。共同偏差、错组、机动/插值及reference关系仍是竞争解释。源排序不是物理质量标签，m不是有效独立样本量。

## 局部P×F复核与单连接

此表消费已完成Mode C证据，明确本项要验证的两个操作及不支持格；不把表格填齐称为新精读或因果发现。

| 问题 | F219全点收益 | F219场景/组件反向 | F219 high_only损害 | F218更广FixedMean收益 |
|---|---|---|---|---|
| MX-P1：硬源优先是否丢失另一源的可用贡献？ | LOW已解释大量收益；支持设置强控，不证明soft更好 | 支持比较soft的异质效应；不证明源方差差异 | 不直接支持：hard与soft在high_only应完全相同 | 支持保留FixedMean强控，不可拼人口当soft结果 |
| MX-P2：中心收缩能否改善风险而不过度欠校正？ | 仍有80.386m校正增量，向零收缩可能牺牲它 | 需固定m规则、按fold/组件完整评价；m相关不证明因果 | 是可干预作用域；可验证实际误差变化，根因UN | 全V校正有益，不支持普遍缩小所有中心一定更好 |

**MX-C1（partial、需诊断）：** 保留相对共享中心与合法来源support，将“源混合”与“中心幅度”拆成2×2；新增价值若成立，是可归因地改善当前恢复风险/异质性，而非宣称首创凸组合或经验贝叶斯理论。当前没有对两操作的目标实测，也不要求完全无先例才开展。

## 冻结输入、操作与边界

原121的V32/9组件、WS membership/首代表/全Q64/输出valid及同一056输入、manifest保持。原121保存的十臂artifact直接导入，保存的118中心/source_order/group_fold/group_fit_eligible复用，**不拟合新中心、不重算旧预测或Huber**。123只核其完成与同一数据/原32来源；本项不从其误差选择参数。

预测函数只接受物理状态、support、可见source、groups、保存中心/fold/eligible和原Mean速度/航向。identity/reference/target support仅在所有预测完成之后用于评价。来源排序沿原118字符串顺序，不重新定义物理AIS/雷达角色。

1. **软贡献：固定低源总质量0.75、高源0.25。** 仅both_sources有新动作，各源内对支持成员等权；仅一源有support则该源总质量1；no_support总质量0，按既有恢复复制相应输入首anchor。0.75是硬优先1和源等权0.5之间的事前透明控制，**不是估得的最优精度比，也不是已完成选参的最终方法**。不使用MAD、σ0或新门控。
2. **中心收缩：** 对原每fold中心`c_f`，取其他fold中原`group_fit_eligible=true`的组数`m_f`，固定`lambda_f=m_f/(m_f+4)`，仅高source使用`-lambda_f*c_f`球面位移。原`m_f<2`中心已为零，继续零回退；整个场景仅一个source时不校正，双源场景内只有高源的组或singleton仍沿121校正。4是固定经验强度（m=4缩半），不估先验、采样方差或meff，**不称EB估计、不承诺P99变好**。收缩会改变high_only与可能的no_support首anchor，也会改变soft的both域；速度/航向始终复制同一原WS_MEAN。
3. 两操作使用同一球面位移、经度圆均值与单支持成员exact复制规则；圆均值零resultant仍按相应输入首anchor对应角回退。没有改变输出存在域或拒配，无删难例/新reference门控。

### 臂与归因

导入原121十臂；新增以下五臂，共15臂、105个无序配对。不能只报最好组合。

| 新臂 | 中心 | both_sources低源质量 | 对应问题 |
|---|---|---:|---|
| `WS_SOFT75_SOURCE_SUPPORTED` | 无 | 0.75 | 原LOW→soft，分清中心存在与贡献本身 |
| `WS_FIXED_SOFT75_SOURCE_SUPPORTED` | 原完整中心 | 0.75 | FixedLOW→soft，源贡献主比较 |
| `WS_SHRINK_LOW_SOURCE_SUPPORTED` | m/(m+4)中心 | 1 | FixedLOW→收缩，幅度主比较 |
| `WS_SHRINK_SOFT75_SOURCE_SUPPORTED` | m/(m+4)中心 | 0.75 | 2×2交互，不能独归因给某一个操作 |
| `WS_FIXED_EQUAL_SOURCE_SUPPORTED` | 原完整中心 | 0.5 | 常量源等权强控；与按fragment数Mean有区别 |

核心2×2为**无收缩hard（原FixedLOW）、无收缩soft、收缩hard、收缩soft**；这里“无收缩”明确指复用原完整中心，不是完全无校正。无校正LOW/soft75、原Mean/FixedMean、Anchor/NEAR以及其余原十臂全部保留。新增0.5与0.75均为事前常量基线，不从两者选出“估得精度”。

## 评价、可否证预测和资源

两个主估计量均在同固定完整位置人口上：`fixedSoft75−FixedLOW`径向MAE，以及`shrinkLOW−FixedLOW`径向MAE。交互为`(shrinkSoft75−shrinkLOW)−(fixedSoft75−FixedLOW)`；同步报告无中心soft效应和所有强控。不是统计显著性宣称，不把71k query当独立样本做置信区间。

次要量包括RMSE/P95/P99、原论文坐标ME/MAE、完整人口/coverage/重复/无参考、场景/9组件及去最大组件，both/low_only/high_only/no_support、真实目标support内外（仅事后）、m/lambda分布、实际改动率和真实程序/launcher成本。主要效应、尾部及场景/组件方向完整保留。

- soft只应改变both，hard/soft同中心的其他三层应数值一致；变更其它层即实现或定义偏差，须保全、查清。
- 收缩可能总体/尾部改善，也可能欠校正更差；即使成功，也只支持该固定经验干预，不能宣中心方差机制被识别。
- soft仅在最大组件获利或全点与场景反向：如实缩小范围，不选场景门控、不改0.75。两单因素不利而组合有利：报告交互，不将单因素称有效。
- 0.5源等权或按fragment Mean解释同等效果：保留简单强控，不把更多参数包装成创新。所有新臂不利：结束本配方，不自动搜权重/m0或扩V。
- 新操作/人口/来源/旧artifact绑定失配、非有限值、OOM/代码错误即失败保全，不自动重试、覆盖或放宽断言。

一次attempt-1，固定V32终点，CPU≤4、RSS4GiB、AS8GiB、600s **advisory**，0GPU/模型/前向/优化更新；预计成本以121的97.631s程序为参考，105配对及15臂可能更高，不承诺90秒或因超建议墙时加训练。无新scores/分组/normalizer/阈值/ZIP/CSV/T_test。原坐标/身份/中心/大预测留服务器，只回传脱敏汇总/逐场/输入配置/真实命令与退出成本。软件合成检查只认实现证据，实际执行后严格Record→Squeeze→Mode C。

## 实现规格与必要验收

| 行为 | 可观察验收 |
|---|---|
| 旧十臂不重算，保存artifact全绑定 | patch禁调用旧constructor/heldout fit；完整artifact逐字段exact及公开position绑定 |
| 两操作独立、作用域清楚 | 直接可计算例验证0.75/0.5质量、fold排除本组、收缩倍数、仅高源校正、单源/空support/日期线回退 |
| 人口/速度航向不变 | 全臂valid/membership/首代表/coverage/重复一致；新增速度航向与原Mean exact |
| 失败可保全且不覆盖输入 | 新输出目录exist_ok=false，run.json终态/已完成场景保留，无自动重试 |

```yaml
package_type: IDEA_PACKAGE
package_version: research-pipeline-contract v1.0
package_id: IDEA-SOURCE-MIX-SHRINK-CONTROL-20261029
idea_id: IDEA-SOURCE-MIX-SHRINK-CONTROL-20261029
plan_version: '1.0'
source_matrix_id: MODE-C-1.41_AND_FUSION-SC-C3
selected_C_id: MX-C1
source_P_ids: [SC-P1, SC-P2, MX-P1, MX-P2]
source_F_ids: [F218, F219]
source_M_ids: [M89, M90]
source_E_ids: [E116, E117]
research_question: 合法固定软贡献与共享中心经验收缩各自及交互能否改善固定恢复风险
hypothesis_and_mechanism: 软贡献可能保留另一源信息而收缩可能缓和幅度风险，两者均为未验证竞争解释
testable_predictions: 同人口固定soft75与shrinkLOW各对FixedLOW的MAE及尾部效应和2x2交互，作用域不越界
design_and_controls: 导入原十臂，新增五臂，15臂105配对；主2x2及无校正soft/源等权0.5/Mean/Anchor/NEAR
sample_materials_or_experimental_units: 原121开发V32/9组件和同056/114/118保存输入；123完成后进入
primary_outcome_and_estimand: fixedSoft75-FixedLOW与shrinkLOW-FixedLOW同完整人口径向MAE差
secondary_outcomes: 全RMSE/P95/P99/坐标ME及MAE/coverage/重复/支持分层/场景组件/代价及交互
analysis_plan_and_uncertainty: 固定确定性配对与完整组件异质性，不将query当IID，不根据123结果选参
resources_time_cost_and_risk: 单次CPU4/RSS4GiB/AS8GiB/600s advisory/原32终点，0训练
ethics_safety_privacy_or_compliance: T_test封存，reference仅评价，私有坐标身份中心留服务器
stopping_and_interpretation_rules: 固定终点或真实故障，失败保全无自动重试搜索扩量或换配方
status: diagnostic
evidence_status: partial
missing_evidence: [soft_mixing_effect, shrinkage_effect, mechanism_identification, matched_credible_peer]
next_skill: record-experiment_after_actual_execution
```
