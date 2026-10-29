# 观测驱动的贡献与原生状态：136/137候选合同

2026-10-30，设计版本1.0。**实现已获root审查确认；尚未运行、未发布READY。** 本文包含两个独立Idea，不预定组合；已消费观测驱动Mode C v1.1的18格与OD-C1/C2。135保留但本轮降优先级。本文件不维护实时队列，执行许可只见研究状态和HandOver。

本轮使用项目discover-innovation与spec-driven-coding：先承接已有实际问题，再冻结可反驳的最小干预。新增值若成立，是在统一四状态、无固定源角色且质量未知的输入下建立可用的观测代理与状态恢复；当前不主张理论原创、概率校准或国际同行胜出。

## 0. 证据起点与待补桥接

既有来源为[多源可靠性Mode C](../../../biye-ronghe/docs/surveys/2026-10-27_expanded_search/multisource_reliability_matrix_20261030.md)的MR-P1/P3、MR-F11/F12、MR-M2/M3/M6/M7、MR-E11/E12（全局F231/F232、M102/M103、E129/E130）；原生状态连接见[127事前Idea](native_cv_state_control_20261029.md)的IT-C5及[127正式结果](../reports/2026-10-29-native-state-review.md)。[133](../reports/2026-10-30-source-risk-review.md)与[134](../reports/2026-10-30-mixture-opportunity-review.md)是已经完成的实际证据，不是本轮新结果。

| 问题/现象（DE） | 候选解释与拟区分问题（AI/UN） | 两项设计能与不能回答的内容 |
|---|---|---|
| 133共享Q hard比equal/soft好，但尚未超过旧强控制；134仅2.66%误选形成47.264m平均条件损害 | 监督得分是否还需要复杂读出，或合法原生观测自身已有低成本质量信号？也可能只识别数据生成角色 | 136检验不经reference拟合的具体代理；不把成功称跨真实传感器泛化，也不证明角色混杂消失 |
| 134条件网格混合空间均值39.600m、中位0.346m；源内一致性低仍不能保障低风险 | 可预测性与绝对正确性可能不同；任何仅看自身残差的代理都看不到某些系统偏移 | 136并列报告代理与实际误差/损害，保留常值偏差反例；不把小残差叫已校准协方差 |
| 133无支持点仅3.9%，却承担Qhard约69.82% SSE；127更换原生状态在相同Mean下有MAE改善且仍有尾部恶化 | 有些风险不能靠支持域内选源解决，状态恢复值得独立研究；原127rho依赖T参考 | 137固定贡献，检验原生观测留出能否选到有效尺度；不自动归因于真实Q/R识别或物理最优 |

**本轮来源交接已完成：** [观测驱动Mode C v1.1](../../../biye-ronghe/docs/surveys/2026-10-27_expanded_search/observation_driven_matrix_20261030.md)提供OD-P1/P2/P3、OD-F1–F6、OD-M1–M5、OD-E1–E6及18格。OD-S4为TAC2026 MDM完整来源，明确已知模型/结构与秩条件，未知输入同通道可能令Q不可辨；OD-S5为SemiDANSE观测空间约束边界；OD-S6为[Chroma完整分析§2](../../../biye-ronghe/docs/surveys/2026-10-27_expanded_search/observation_driven_transfer_sources_20261030.md)，给出共享历史目标、集合WQL选择及单选更优反例。136消费OD-C1，137消费OD-C2，以下才将连接变为具体干预。Chroma历史窗口与本设计全片段双侧信息不同，不称作者方法/greedy复现；各源自身带噪残差不满足共同真实验证目标条件。137也不复现完整自适应Q/R估计器。未知项仍包括proxy目标风险价值、真实噪声可辨性和换源泛化，补齐来源不等于效果已验证。

## 1. 共用固定合同

- 正式MTAD Level1，原manifest的T_core前32场/12组件、原V_select32场/9组件；只读完成的127 canonical_all cache、125 T memberships、114 V WS、133已保存控制及127尺度/结果。不读原ZIP、不新增teacher前向/神经训练、不打开T_test。
- `native-events`具有每节点原始有效事件；`native-motion`提供原Q64四状态/support/nearest/OLS残差与速度。评价文件中的identities/target/reference与预测API分离；source字符串只定义等价类，不排序成优先级、不入分数。
- 节点/组/首代表/重复/缺失/有效性和全部目标人口沿既有规范。各组K可变，源内多片段不当成多个独立传感器；源贡献分配给本源当前支持节点后沿原圆周恢复器恢复。源间先平均、节点先混合不能随意互换。
- 本轮**仅改变位置**；所有新臂的speed/heading逐元素保持原WS_MEAN，输出仍是四状态接口。这不等于已解决四属性联合可靠性。
- 整段OLS、双侧留一和RTS有未来信息；137的FWD也沿全窗口chart。二者均为离线验证，不称在线。新模块无target监督不等于上游056/WS全系统无监督。
- T仅诊断、137原生留出尺度选择；V固定一次评价，不据V改floor/rho网格/代理/人口。T/V共享组件和重复开发V的限制保持。真实K仍为2；K≥3只是接口与软件性质检查。
- 每项独立新目录、一次attempt，真实失败保全、不换参数/删难例/自动重跑；超建议墙时只报告。所有坐标/身份/事件/权重和大预测留服务器private，回传脱敏完整逐场/组件/人口/成本。

## 2. Idea 136：无reference的观测质量代理能否分配可变K源贡献

### 元数据与假设

`IDEA-OBSERVATION-QUALITY-20261030`，Pattern 1问题驱动；source_matrix=`MC-OBSERVATION-DRIVEN-20261030-v1.1`，selected_C=OD-C1；P=[OD-P1,OD-P3]，F=[OD-F1,OD-F2,OD-F4,OD-F5,OD-F6]，M=[OD-M1,OD-M2,OD-M4,OD-M5]，E=[OD-E1,OD-E2,OD-E4,OD-E5,OD-E6]。证据partial，允许实现核验但未READY。主要假设：观测可预测性可能含有部分与位置风险相关的信息，透明共享规则能否在相同状态下优于源等权，应由完整结果判断。竞争解释包括机动、采样密度、相关滤波误差、同源重复和生成角色；常值偏差不可辨是已知边界。

### 固定计算，不用T参考选代理或floor

每fragment保留原全canonical事件。两种质量均为m²量纲的非负代理，不宣协方差：

1. **OLS：** `q_n = fit_rmse_m[n]²`，即原全部native位置的局部平面直线残差均方。沿原实现，不另作自由度校正、不除以native_count。其含机动/模型失配，少事件也不意味着观测独立性更好。
2. **LOO-interpolation：** 对每个内点i，令`a=(t_i-t_(i-1))/(t_(i+1)-t_(i-1))`，用相邻两点线性插值预测该内点；同原圆周展开与米制局部平面计算残差e_i。`q_n=mean_i(||e_i||² / [1+(1-a)²+a²])`。分母仅在局部线性、三个噪声独立同分布、零均值时对应插值噪声增益；实际未认证，不能称观测噪声方差的无偏估计。n=2时proxy未知，不能用0冒充高质量。经度展开/平面是几何表示，136不作严格交叉验证或预测误差无偏性主张；137的留出则必须全链移除。

组g、query t中，源s的质量`q_s(t)`为该源当前支持成员`q_n`的**算术平均**，不使用`1/n`当作独立样本带来的方差缩减。若一个当前支持成员proxy未知，则本源unknown；若任一可用源unknown，整个该query采取source-equal，并记录fallback。0源依旧原Mean，1源贡献1；不删除任何点。

对全部已知的可用源，`w_s ∝ 1/max(q_s, f)`；预固定`f∈{1,100}m²`，即1m/10m工作尺度下限。两个floor均作为独立臂完整保留，不选winner再称预定方法。没有拟合参数，但仍有预设工程常数和方法选择成本。

另设**SOURCE_NEAREST**：源内当前支持成员nearest_seconds最小值，源间取最小，精确平局源间等分；被选源内部仍均分支持成员。它不同于历史nearest-node控制，名称与预测不得混淆。常量proxy的无信息读出必须exact退化为source-equal；不另制造一份相同科学臂。

### 臂与归因

V侧全部133十个保存臂原样保留：五旧控制（Mean、nearest-node、LOW、FixedLOW、旧Qgate），source-equal、Qsoft/Qhard、linear-soft、shuffled-soft。新五臂为SOURCE_NEAREST、OLS_F1、OLS_F100、LOO_F1、LOO_F100，V共15臂/105对；源等权从133绑定而不是另造分母。133未保存T侧十臂完整artifact，**T只执行source-equal加五新臂，6臂/15对**；T参考仅预测后诊断，不伪造历史T控制。可另读已有OOF作辅助排序证据，但不混同完整T预测成绩。

- 四个主要描述性比较：每个inverse-proxy臂−SOURCE_EQUAL，同原native状态和support。
- SOURCE_NEAREST作为低成本非残差替代解释；同floor的OLS−LOO比较仅区分这两个完整代理，不宣单独识别“局部性”的因果效应。
- floor1/100成对报告含权重熵/最大权重/选择翻转/触发比例，不能隐藏敏感性。Qhard为有监督强控；LOW/FixedLOW只作历史对照，不返回固定角色架构。
- 不跨臂挑每点/每场最佳；不给四臂的探索性比较套预注册单一检验的显著性结论。

### 终点、量化与出口

完整T32/V32逐场预测和评价一次。每臂全人口MAE/RMSE/P95/P99、64query覆盖/无支持/重复/漏目标、逐场/组件/去最大组件、组件bootstrap描述性区间与T15/V105对差值均保留；主效果先看V完整人口而不是仅双源点。额外记录0/1/2+源、proxy未知、floor触发、原native_count/支持与组污染的事后分层；身份/污染只评价用。

支持条件是代理相对equal有可重复的组件/场景收益且尾部/人口风险可解释；是否超过监督/历史强控分别报告，不把“比equal好”称最终方案成熟。若代理不能减损，保留残差盲点、角色混杂和状态误差解释，不否定所有无target质量估计。若只某floor有效，记录尺度依赖，后续必须独立冻结选尺度规则，不能从本V继续调。

资源：CPU≤4、RSS6GiB、AS12GiB；0GPU/0神经更新/0reference回归拟合。逐场一遍事件，时间O(E + NQ + policy evaluation)，保存私有proxy/源贡献，不建跨全部场景大tensor。1800s advisory；固定64场与15臂终点，实际事件数/计算耗时/读写/峰值与105配对计数全部记录。此预算是候选合同估计，不是已经实测的137/136成本。

## 3. Idea 137：原生观测留出能否替代reference选择离线状态尺度

### 元数据、区别与机制

`IDEA-OBSERVATION-SCALE-20261030`，Pattern 5系统假设审查；source_matrix=`MC-OBSERVATION-DRIVEN-20261030-v1.1`，selected_C=OD-C2；P=[OD-P2,OD-P3]，F=[OD-F3,OD-F4,OD-F5,OD-F6]，M=[OD-M2,OD-M3,OD-M4,OD-M5]，E=[OD-E3,OD-E4,OD-E5,OD-E6]。证据partial，允许实现核验但未READY。原127选择了工作rho=1e-8并有条件收益，但这个尺度使用T参考。拟检验：对原生观测作固定留出后选择尺度，能否在同一CV/RTS模型和source-equal贡献下保持或改善完整位置风险。此为**观测驱动选择参数**，不是无参数、自适应Q/R全识别、物理最优或贝叶斯最优。

### T原生留出合同

1. 每fragment按合法canonical时间排列，仅有效lat/lon事件进入位置序列。保留首末；内点局部0-based索引`i%3==1`作为heldout，其余为retained。n=2无heldout，不参加选择但完整保留最终恢复。每个合格fragment至少2 retained且至少1heldout。固定一次pattern，不轮换mask/挑切片。
2. 对127原九个`rho=10^k, k=-10…-2`，只向现有CV/RTS核传retained位置/时间，以heldout原时间为query。heldout位置不得进入chart原点、平均纬度east scale、初始化、拟合或几何展开；全部从retained重建。heldout值仅在预测冻结后计算评分。heldout时间是已知query，不是位置观测；不把其加入观测更新。
3. criterion为**场景等权→合格fragment等权→heldout点径向平方误差均值**，单位m²；选择RTS criterion最小的一个rho，精确平局较小rho。FWD共用这个rho，不称FWD独立调优。无合格fragment的场景保留null及计数；若全部T无可选数据则失败保全。
4. 所有T候选/场景/fragment criterion、heldout/retained计数、时间间隔和选择保存。T reference无论是否可读均不进入本选择API；V不执行尺度搜索。
5. 该留出预测的是原观测：仅在留出测量噪声与retained预测误差独立、零均值等条件下，平方观测误差与潜在状态误差存在简单常数差关系。实际观测可能已经滤波/时序相关且有bias，不认证该关系；机动和缺口也是竞争解释。固定保留首末意味着主选择是双侧内插/平滑，不代表外推/在线性能。

### 全数据恢复与严格同贡献对照

选定obs-rho后，对T32和V32所有完整native重新运行原127的同一CV/RTS计算，保留原Q64/支持和速度航向Mean。**零支持保留相同回退规则，但作用于当前CV状态的Mean/首节点，不固定为原OLS位置数值；否则会人为禁止状态模型改善无支持域。** 136状态未变，故其回退仍原Mean exact。比较：

- A：原native_OLS＋SOURCE_EQUAL（133保存输出绑定）。
- A′：原127 **reference选择的固定rho**＋FWD/RTS＋SOURCE_EQUAL，两个必要新恢复控制；原rho从127 `scale-selection.json`读取，不重选、不再训练。
- B：本轮obs-rho＋FWD/RTS＋SOURCE_EQUAL，两臂。

**主要比较B_RTS−A、B_RTS−A′_RTS；** FWD对应差值、RTS−FWD为信息权限次比较。137不使用136新权重，防止混合两机制。保存并导入127已完成的六臂作为历史位置/Mean/LOW结果核对，其中A′重算得到的V普通Mean/LOW须与原127对应完整artifact exact绑定；这也是重建没有悄悄改变模型的校验，LOW不是新方法。

V侧为避免重复科学臂，共保留133十臂、127其余四个CV旧臂（Mean/LOW两个原臂与133重合须先exact绑定）、上述四个SOURCE_EQUAL新状态臂，共18臂/153对。**T侧只有native-equal、reference-rho的FWD/RTS equal、obs-rho的FWD/RTS equal五臂/10对**，不假定存在T侧完整历史控制。若obs-rho等于原rho，则复用同次原生fit输出，两组标签仍显式别名、预测exact，不重复执行相同拟合。主结论只基于同状态/同贡献可归因比较，不把18臂中最低数值称组合系统。

### 预算与判断

选择阶段最多`9*N_Teligible`次fragment CV/RTS计算；恢复阶段最多`2*(N_T+N_V)`次（obs与reference-rho），相同rho去重。`N_Teligible`是有heldout的原生片段数，非query数。一次fit同时得到FWD/RTS，不为两臂重复拟合。现有127只保存group outputs，不能以它恢复SOURCE_EQUAL所需的所有node状态，因此必要重建不是新增拟合配方。预测完整节点先完成，分组恢复只重算权重/输出，不重复滤波。

CPU≤4、RSS6GiB、AS12GiB，9000s advisory；借原127约2612s实际墙时估计为同阶至约两倍以上，非保证。0GPU/神经更新；真实解析fit/事件/更新/RTS步/归一化/读写全报。九rho仅逐场逐candidate流式，不能持有全部64场×9所有原生状态。

固定T32×九候选和T32/V32×两种已定rho的完整恢复终点；完整人口、MAE/RMSE/P95/P99、组件/去最大组件、尾部与支持域，同136规范。若obs规则改善则只支持此观测留出方式；若内部observed criterion更优但reference效果更差，这是代理目标失配证据，不自动扩大网格/改mask/读取V选参。若仅支持域内好而无支持域尾部差，保留其作用域；不得通过取消外推/删点美化结果。

## 4. 转READY前的最小缺项与后续

OD来源/矩阵和root行为与资源审查已完成，版本1.0允许实现纯计算模块、独立runner及有意义测试。实际执行仍待完成实现/审查后独立READY，本文不是自动开跑许可。

两项都必须独立Record→Squeeze→Mode C，出现成功也不直接把136和137冠军组合。本轮不是135重温度搜索，也不取消135；若未来组合，要先定义同状态/同贡献析因合同与固定全人口再独立尝试。没有独立跨传感器/K≥3实测之前，只能主张接口可扩展与当前K2内部效果。

## 5. 标准Idea交接索引

两包分别只消费一个主连接；字段指向本文已冻结合同，不把尚未运行写成结果。

```yaml
package_type: IDEA_PACKAGE
package_version: research-pipeline-contract v1.0
idea_id: IDEA-OBSERVATION-QUALITY-20261030
plan_version: '1.0'
source_matrix_id: MC-OBSERVATION-DRIVEN-20261030-v1.1
selected_C_id: OD-C1
source_P_ids: [OD-P1, OD-P3]
source_F_ids: [OD-F1, OD-F2, OD-F4, OD-F5, OD-F6]
source_M_ids: [OD-M1, OD-M2, OD-M4, OD-M5]
source_E_ids: [OD-E1, OD-E2, OD-E4, OD-E5, OD-E6]
research_question: 无reference拟合的源内观测代理能否在同状态下改善source中立贡献
hypothesis_and_mechanism: §2，可预测性可能携带风险信息，偏差/机动/角色为竞争解释
testable_predictions: 四个inverse_proxy与同状态sourceequal的完整风险差及floor敏感性
design_and_controls: §2，两proxy乘两固定floor，nearest替代，V133十旧臂；T只有六合法臂
sample_materials_or_experimental_units: §1，固定T32/V32完整Q64人口，当前真实K2
primary_outcome_and_estimand: 四比较的完整V径向MAE/RMSE差，描述性探索
secondary_outcomes: 尾部/人口/损害/组件/去最大及proxy未知/floor层
analysis_plan_and_uncertainty: §2，完整逐场及组件bootstrap，不宣query独立或外部测试
resources_time_cost_and_risk: CPU4/RSS6GiB/AS12GiB/1800s advisory；一次固定64场；T15/V105对
ethics_safety_privacy_or_compliance: §1，T_test封存，private留server，预测与reference隔离
stopping_and_interpretation_rules: §2，固定终点，失败保全，不扩floor/据V改规则
status: exploration
execution_status: implementation_authorized_not_READY
missing_evidence: [actual_proxy_effect, physical_noise_identification, independent_K3plus_transfer]
next_skill: record-experiment_after_actual_execution
```

```yaml
package_type: IDEA_PACKAGE
package_version: research-pipeline-contract v1.0
idea_id: IDEA-OBSERVATION-SCALE-20261030
plan_version: '1.0'
source_matrix_id: MC-OBSERVATION-DRIVEN-20261030-v1.1
selected_C_id: OD-C2
source_P_ids: [OD-P2, OD-P3]
source_F_ids: [OD-F3, OD-F4, OD-F5, OD-F6]
source_M_ids: [OD-M2, OD-M3, OD-M4, OD-M5]
source_E_ids: [OD-E3, OD-E4, OD-E5, OD-E6]
research_question: 原生观测留出能否在同贡献下替代reference选择离线状态尺度
hypothesis_and_mechanism: §3，留出误差可能保留任务风险排序，相关噪声/bias/插值范围未识别
testable_predictions: 同sourceequal下obs_RTS对nativeOLS及reference_rho_RTS的完整风险差
design_and_controls: §3，固定九rho和mask，T观测MSE选择，FWD/RTS及V历史控制
sample_materials_or_experimental_units: §1/3，固定T32/V32完整native，无holdout节点仍全部输出
primary_outcome_and_estimand: 两主比较的完整V径向MAE/RMSE差
secondary_outcomes: FWD/RTS权限、尾部/组件/覆盖/完整配对及真实拟合成本
analysis_plan_and_uncertainty: §3，完整逐场/组件/去最大，不据V搜索
resources_time_cost_and_risk: CPU4/RSS6GiB/AS12GiB/9000s advisory；9*N_Teligible+2*(N_T+N_V)fit上界
ethics_safety_privacy_or_compliance: §1，Traw-only选择，T_test封存，坐标身份留private
stopping_and_interpretation_rules: §3，无候选失败保全，不扩rho/mask、不自动重试
status: exploration
execution_status: implementation_authorized_not_READY
missing_evidence: [observation_selected_state_effect, heldout_noise_assumptions, independent_transfer]
next_skill: record-experiment_after_actual_execution
```
