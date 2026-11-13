# 冻结动作条件评分的原 V32 验证

## 交接元数据与当前状态

创建2026-11-02，修订2026-11-03；`IDEA_PACKAGE` / `research-pipeline-contract v1.0`，总控 `research-pipeline-contract v2.0`；Idea ID、package ID 均为 `IDEA-ACTION-RISK-V32-20261102`，`plan_version: 1.1`。

**执行记录补记（2026-11-05）：196科学与规定独审已完成。** [完整Record/Squeeze](../reports/2026-11-03-action-conditioned-risk-validation-review.md)、[终态回执](../../HandOver/S2L-20261103-196.md)。以下保留事前plan1.1定义及判定，不因结果更改设计；当前目标/后续只看研究状态。

主连接为 **C69**，消费本地已核的 [MC-MTAD-20261102-v2.09](/Users/a1/Projects/biye-ronghe/offline_observation/docs/surveys/2026-10-26_mtad_actual_use/reassessment_20261027.md#mode-c-209)，对应 **P12/P13/P14 / F305 / M123（AI）/ E208**，上游为[195 Record/Squeeze](../reports/2026-11-02-action-conditioned-risk-review.md)，另引用[191 固定 V32 记录](../reports/2026-11-01-mean-velocity-validation-review.md)的 F298/E201。2026-11-03定向核读了上述完整Mode C小节及[SR v4.10/177格](/Users/a1/Projects/biye-ronghe/offline_observation/docs/surveys/2026-10-27_expanded_search/selection_readout_matrix_20261030.md#v410195动作条件风险实际回流)、[OD v1.23/93格](/Users/a1/Projects/biye-ronghe/offline_observation/docs/surveys/2026-10-27_expanded_search/observation_driven_matrix_20261030.md#v123195动作条件风险实际回流)，复用其连接和反例，不生成新矩阵版本、P/F/M/E或文献计数。历史矩阵中的“多数收益”须按本方案和195本地复核注明比较分母。

证据状态 `partial`：195/191本地公共summary、动作机会和旧病例聚合已核，规则和代码接口可核；本方案的 V 输出、原数组重建和效应尚不存在。195/191的private原预测、权重和原生数组没有在此次本地复核中重建；对应历史独审不冒称本轮重新执行。按[项目实现技能](../../.agents/skills/spec-driven-coding/SKILL.md)完成独立入口与独审适配；实际运行后才生成本方案的 Record→Squeeze→Mode C 回流。

版本沿革：plan1.0是2026-11-02远程审查附件中的未实现、未READY、未执行设计；原稿保留于用户交付的`association-code-review-20261102 (1).zip`，不倒填成已执行合同。plan1.1补入2026-11-03本地来源核验、单节点动作平局混合语义、明确结果判定与实现验收规格；原V32、模型、40臂、12比较、资源和0新训练范围不变。READY须引用实际通过验收的本版，不能追认plan1.0已通过软件检查。

## 1. 研究问题、来源与可否证假设

领域/任务：在统一经纬度、速度、航向、真实时间和来源归属输入下，研究多源航迹关联后支撑外的位置恢复。主模式为系统接口假设挑战：**195 的冻结评分在原 V32 上，能否限制在原 O 最低分节点候选池内选择 O/MC，较直接应用 MC 保住平均收益并减少新增高害？开放原池以外的节点还有多少独立增量？**

这里的 O 是 178 的历史候选恢复，MC 是 189/191 的原生报告平均运动；两者的数学、候选、已选节点/平局权重与信息权限均沿用原件。本文验证现有选择规则的转移，不产生新的传播公式、特征、训练目标或风险阈值。

### 来源到假设的映射

|来源及证据层级|本方案使用的事实/边界|对设计的约束|
|---|---|---|
|[178 历史候选报告](../reports/2026-10-31-history-candidate-review.md)、[历史特征/标签实现](../../association/evaluation/history_candidates.py)|风险头用原 O 状态的 `log1p(位置误差/1000)` 监督；特征 12 维，训练域含零/单来源支持；部署参考不可见|旧风险分数不是动作收益、概率或已校准的动作可靠性；不能由名称认定迁移条件已成立|
|[195 方案](action_conditioned_risk_20261102.md)及[结果](../reports/2026-11-02-action-conditioned-risk-review.md)|T32 中 CF−MC MAE 为 H −1.989868m / S −0.061903m；JM−CF 为 −0.070114 / −0.070574m，后者全部来自同一混组的两点|CF−MC 是本轮主要判别；JM−CF 保留为开放原池外节点的附加判别。不能笼统把“CF 获多数收益”外推到每个比较基线/前端|
|195 已记录尾部|MC 对 O 新增公里害 34/1，经 CF/JM 为 7/0，救回 3/33 保留；H 剩余 7 点是同一纯组单节点病例|这些是 T 条件发现；小负风险差与时间范围只作后验线索，不据此增加 margin、时间或拒绝阈值|
|[191 V32 完整证据](../reports/2026-11-01-mean-velocity-validation-review.md)|MC 对 O 的原 V 新增公里害为 **11/0**，救回 **70/42**；H 11 害集中同一场同一纯组、**两个节点**|V 的案例结构与 T 剩余 7 点不同；不预断 CF 必有效或节点重排必无用。H 的 PG 曾有 71 救回，MC 已是 70，不能用 71 作为 MC 保留分母|
|[180 成本监督阴性结果](../reports/2026-10-31-cost-routing-review.md)|单支持候选域中，匹配训练后的物理 regret/高害成本目标没有超继续回归；新害增加|不能把“改物理 loss / 加尾部 hinge”重新命名为未经尝试的新方案。它也不否定不同动作监督条件的未来方案；若要训练须另立 Idea、明确与 180 的差异|

**H1（动作选择）**：在相同合法候选、固定前端和固定风险模型下，CF 相对 MC 的平均/尾部取舍可在原 V32 保留条件价值，且覆盖、分组、关联和速度/航向不变。

**H2（候选放宽，次级）**：JM 相对 CF 可能因开放原 O 最低分池外节点而增加可归因收益；其证据必须来自实际池外节点的使用、权重及完整指标，而不是把 CF 已产生的池内动作/并列选择收益重复归给 JM。

**机制假设 M123 仍为 AI**：动作改变后重算相关描述可能使旧评分更贴近被选状态。竞争解释包括评分对 MC 特征条件不可迁移、原生位置/SOG/COG 依赖、观测之外的机动不可见、分组/first-claim 对应、参考生成误差、候选不足与组件组成。原单节点 O 的相对自身位置/速度特征为零、MC 后相应特征可非零，这是代码结构事实；它不单独证明落在整个 T128 训练分布之外，更不证明 T 的 7 点或 V 的 11 点以此为根因。

## 2. 固定数据、模型与人口

### 2.1 输入血缘及不可混用项

|输入|本方案固定使用|核验内容|
|---|---|---|
|178 风险模型|原 T128 的 **`private/model-full.pt`**，包括二次基底、MLP 权重、12 维 normalizer 与所有既定推理常数|和 178 原 V `policy` 的 scores 绑定；不得使用 T2100 模型、T32 的 `model-leave_<component>.pt` 或重新拟合 normalizer|
|178 分组和旧策略|原 `private/policy-V_select-{sid}.npz`、`private/partitions-V_select-{sid}.npz`、原 V O 输出|节点顺序、query、HARD/STRUCT membership、score、阈值、count、features、原 O 选择权重逐项绑定|
|127 原生输入|原 `private/native-motion/V_select/scene-{sid}.npz` 与 `private/native-events/V_select/scene-{sid}.npz`|原节点、真实 query、事件 offset、native count、位置/SOG/COG valid、单位和观测端点保持；不重新裁输入或扫描新场景|
|191 控制|原 V32 的全部 28 臂 raw/partitions/评价产物及其 MC 状态|完整导入原 front/O/PH/PC/PD/PG/MC×HARD/STRUCT×MEAN/NN；重建 MC 后与原输出绑定，不按本次效果换控制|
|评价|原 manifest 的 V32 顺序、9 个共享身份组件、原 reference/first-anchor/first-claim 对应|预测和策略落盘后，才打开该场的冻结 V reference 做统一评价；原始标识/坐标不进入公开聚合|

178 的 T32 评价使用后端 leave-component 模型，V32 使用原 T128 full 模型；上游前端曾见 T，原流程不是全链独立 LOCO。V 本身已用于开发且与 T 共享身份；本轮是**冻结规则在既有开发 V 上的转移检验**，不称封存确认、身份独立新域或全 T2100 训练验证。

### 2.2 预定人口绑定

保留原 **32 场 / 9 组件 / 1427 个场内目标出现 / Q64**，有效位置 reference 共 **72,196** 点。名义 91,328 个目标查询中其余 19,132 点无有效位置 reference；仍记缺失，不能将有效参考人口覆盖率解读为全部 query 的准确性认证。

|前端|主有效位置点/可用参考|位置点覆盖|覆盖目标/1427|重复组|first-claim 且有有效 reference 的 eligible 点|
|---|---:|---:|---:|---:|---:|
|HARD|71396 / 72196|98.891905%|1401 / 1427|203|784|
|STRUCT|70775 / 72196|98.031747%|1380 / 1427|104|655|

这些是本次动作实验必须绑定的原控制人口，不是筛选新输出的目标阈值。`eligible=784/655` 是被主评价认领且有有效位置参考的子域；raw eligible 的组查询/节点查询另记，不把未认领或无参考的行当同一分母。若实际绑定不符，保全现场、查输入/实现；不能删点凑数。

## 3. 干预、40 臂与比较顺序

复用[195 动作纯函数](../../association/evaluation/action_conditioned_risk.py)。只在**零来源当前支持且 query 位于组内全部原生片段之前/之后**的既定 eligible 域改变位置。MC 由同片段全部有效原生 SOG/COG 的 NE 速度均值、真实秒和原生测量端点按球面推进构造；kn→m/s 一次。有效报告不足或端点报告缺失时，候选状态与 score 同时退回 O。内部间隙、1/多来源支持、速度/航向、分组与阈值保持。

MC 评分仅更新 12 维特征的 **6/7/8（0-based）**：MC 位置相对冻结原 O 组中心/同源中心的距离，以及 MC 速度相对冻结原 O 组速度的差；其余特征固定。原模型的标准化与 ±8 裁剪不变；不重算群组上下文，不把评分改成概率、margin 门控或新 loss。

|策略|候选节点|候选动作|归因作用|
|---|---|---|---|
|O|原178节点/平局集合|原状态|保持历史候选强控|
|MC|原178节点/原权重|合法 MC，缺失退 O|保持简单平均运动强控|
|CF|候选池限定为原178最低分节点集合|在该池×O/MC中全局选最低分|核池内动作选择；原O并列时允许最终节点/质量变化|
|RM|原组全部节点|MC（未定义则与 O 同状态/同分数）|对 MC 核节点重排|
|JM|原组全部节点|O / MC|对CF核开放原池外节点，对RM核O保留|

**CF 冻结的是候选池，不是最终节点集合或原节点权重。** 若原 O 最低分仅一个节点，CF 是同一节点的动作选择，O/MC同分时按原规则混合，不能无条件称硬二选一；若原 O 有并列节点，MC 条件评分可以让该池中的最终选中节点子集/平局质量发生变化。MC 对照才保持原 O 节点与权重。本地195公共机会统计记CF在主eligible域`node_changed=0`；该字段不等于完整的原池大小、动作平局和最终权重逐点证明，仍不能据此断言195所有权重都固定。本方案预先保存并列池和最终权重以分离这一情况。

最低 score 平局按源等价类→节点→动作分层等权；保持 195 规则，不增加浮点近似并列阈值、固定源名优先或固定双源比例。可变 K、源改名/换序与并列质量仅作必要软件一致性验收，真实科学结论仍限当前数据主要 K2。

**40 臂完整保留：** H 前缀 `HARD/HO/HPH/HPC/HPD/HPG/HMC/HCF/HRM/HJM`，S 前缀 `STRUCT/SO/SPH/SPC/SPD/SPG/SMC/SCF/SRM/SJM`；每个前缀各 `_MEAN` 和 `_NN`。原 191 的 28 臂完整导入；新增 12 臂中 MEAN 是各自原前端普通 Mean 的 exact 副本。40 臂是输出策略计数，不是 40 次独立训练或独立样本。

### 固定 12 项配对比较

全部比较均 candidate−control；HARD/STRUCT 分别报告，不拼前端冠军。

|比较|HARD 名称|STRUCT 名称|本轮地位|
|---|---|---|---|
|CF−MC|`HCF_NN_vs_HMC_NN`|`SCF_NN_vs_SMC_NN`|**主要问题：限制在原O候选池内选择，能否比固定原权重的MC更有价值**|
|JM−CF|`HJM_NN_vs_HCF_NN`|`SJM_NN_vs_SCF_NN`|关键附加对照：开放原池外节点的额外作用|
|JM−O|`HJM_NN_vs_HO_NN`|`SJM_NN_vs_SO_NN`|相对178强控的总净效果|
|JM−MC|`HJM_NN_vs_HMC_NN`|`SJM_NN_vs_SMC_NN`|相对简单运动强控的总净效果|
|JM−RM|`HJM_NN_vs_HRM_NN`|`SJM_NN_vs_SRM_NN`|相同全节点集合下保留 O 的作用|
|RM−MC|`HRM_NN_vs_HMC_NN`|`SRM_NN_vs_SMC_NN`|只在 MC 状态中重排节点的作用|

相对 O 的 CF/RM 结果、各策略相对原 front 的结果及 PH/PC/PD/PG 控制继续完整报告，作为尾部和历史对照，不在看结果后替换主要问题。

## 4. 终点、统计与机制核算

### 4.1 整体与完整人口

主要估计目标为每个前端 **CF−MC 的共同有效点位置 MAE 差（m）**，同时呈现各自完整 MAE/RMSE/P95/P99、位置 reference/预测/共同有效数与覆盖。固定分组和保持域成立时各动作人口应相同；仍核共同 mask，不只假定相同。

完整保存全部 32 场、9 组件及去最大组件的结果。12 项配对统一用 **2000 次、seed20、组件有放回重采样**，以所抽组件的有效点总误差除以有效点总数重算配对 MAE 差；不把组件均值无权平均误作点加权结果。沿现有冻结统计定义报告 95% 描述区间，分位数使用同一既定实现及插值规则；独审从保存数组/误差和分母独立核算。组件大小、复用开发和跨 split 身份依赖均披露；该区间不提供独立测试或动作因果保证。

所有 40 臂报告完整的缺预测、缺参考、漏目标、重复输出与条件分母，保持逐场原件；coverage、raw/decoded **all、ITSA、MSTSA** 的 mPCA/mwPCA/mPFA/mwPFA 全部保留，含未定义项/定义分母。mPFA=FP/(TP+FP)，不混成传统假阳性率。动作不改组，关联值应 exact 等于对应原前端，不能记为新关联增量。

按[数据约定](../data_contract.md)与[论文指标](../mtad_data_protocols.md)保存位置米制误差、速度/航向物理指标及经纬/速度/航向四属性论文 ME/MAE/absolute_ME/residual_SD、印刷 Eq29 诊断和各自有效人口。ME 有符号，不冒充 MAE；速度单位显式分输入/目标处理，论文单位另转换；作者 ESD 未知保留。速度/航向 exact 沿用普通控制，不宣四属性学习增量。

### 4.2 三套尾部定义分开记账

设同一点候选误差为 e，原 O 为 eO，原 front 为 eF。新增公里害指误差**差值**严格大于 1000m，救回指差值严格小于 −1000m；都不是绝对误差超过 1000m。

1. **策略之间的实际差值**：固定 12 比较及 CF/RM/JM 对 O，记录全部增害/救回点数、组数、场数、最大增害和改善，保完整误差差分分布与 P95/P99，不能用均值代替尾部。
2. **原 MC 相对 O 的固定集合**：H 原害 11 点、S 原害 0 点；H 原救回 70 点、S 原救回 42 点。对各新臂分别计旧害持续/解除、旧非害新增、旧救回保留/跌出门槛/反转成害。不能把“相对 MC 改善超过 1km”直接当作“解决原 MC 害”，也不能以 PG 的 H71 替代 MC 的 H70。
3. **相对更早 front 的总害集合**：原178 O 相对 front 旧害 H128/S107；原 MC 对 front 总害 H138/S92。逐项记相对原 O 旧集合的持续/解除/新增及当前总数，检验集合恒等式；不能把 MC 增量害减少写成整条链原高害已解决。

H 原 MC 的 11 害是同一纯组、两个节点、同一场中的 11 个 query；不按 11 个独立病例解释。S原MC害为0，不能要求或宣称“进一步减少旧害”；S的对应问题是**是否保持0新增公里害，并保留42个原救回**。T195 剩余 7 点的单节点条件与此不同；保留这项结构差异，不因此新设节点数/时间/margin 分支。

### 4.3 分层、动作与条件检查

保留全体、pure/mixed、0/1/多来源支持、eligible、单节点/多节点、全部场景/组件及去最大组件，每层报完整分母。层级由冻结输入和事后标签定义，纯度/reference 仅用于解释，不参与部署选择或排除。

对 CF/RM/JM 记录原 O 最低分候选池大小（包括并列）、最终节点集合与原权重的变化、O/MC 的实际质量、池外节点是否被采用、并列/回退和各节点动作。分别汇总原池仅单节点时的动作改变、并列池内节点/质量变化，以及 JM 开放池外节点的作用，不能把 CF 一律当作最终节点固定。保留原数据上相同节点同时进入两前端的关系，不当作独立复验。

若沿195保留机会核算，必须在预测冻结后，以原 first-claim/reference 分别算最佳 O 节点、最佳 MC 节点、联合最佳节点×动作及原选择节点集合内最佳动作；全部注明是 reference 条件诊断，**未包含平局混合输出，不是部署函数族严格下界**，不得用于挑节点、阈值或新方法。

条件核算同时保存 O 与 MC 的匹配标准化特征/裁剪统计、6/7/8 的变化、raw eligible 及有效主评价 eligible 的分母、动作翻转和对应结果。原特征和变换特征分别记超界行/单元及新增/共有超界，不把约9%的T统计直接移到V。这个只读核算不能证明裁剪致错或校准成立；任何动作标签训练、normalizer 重拟合、风险差阈值或时间拒绝规则都属于未来另立 Idea 的范围。

## 5. 实施范围与规定验收

[195 科学入口](../../tools/analysis/action_conditioned_risk_pilot.py)与[195 独审](../../tools/analysis/action_conditioned_risk_closeout.py)均硬编码 `T_core`、T32 循环/路径和 `model-leave_<component>`，独审还写死 T-only scope。**现有195命令不能直接运行本方案。** 按 spec-driven-coding 新建独立、显式 V-only 的科学入口及独审适配，先核调用链与实际输入，再给真实命令；不得只改 split 字符串或对旧合同作静默扩范围。

实现入口固定为科学侧[`tools/analysis/action_conditioned_risk_validation.py`](../../tools/analysis/action_conditioned_risk_validation.py)和独审侧[`tools/analysis/action_conditioned_risk_validation_closeout.py`](../../tools/analysis/action_conditioned_risk_validation_closeout.py)。二者显式限定`V_select`及178的`model-full.pt`，不借旧195合同扩大范围；真实参数和命令以完成必要验收后的独立合同为准。保留195/178/189/191原行为、活动152 checkout/config/runs及全部失败原件。共享纯函数可复用，必要抽取须保持原默认行为；新入口只能读取已列 V 输入、冻结 full 模型和旧控制，禁止调用 fit、teacher、优化器或重新准备 T 数据。

预测侧顺序为：核原模型/输入/控制绑定→重建无标签 O/MC 候选及条件评分→保存全40臂 raw、partitions、policy→再打开冻结 V reference 评价。事后机会统计在保存预测以后，不能反向改变预测或选择。所有新写入在新的独立 science/audit 目录，旧 run 只读；不能用运行期间 `ExecMainStatus=0` 代替真正完成。

|必须验收的风险|证据与验收范围|
|---|---|
|错误加载模型或人口|原 T128 `model-full.pt` 的权重/normalizer/二次基底和原 V scores 对应；完整 V32/9组件/1427对象、71396/70775主点及784/655 eligible 绑定；0 T 复评/0 test|
|旧控制被改写|原191全部28臂 raw/分区/分数/物理输出 exact；原O、MC重建与保存数组绑定；新增MEAN副本 exact|
|改变非目标域|内部间隙、1/多来源支持、非eligible、全部速度/航向、membership/阈值/覆盖/关联 exact 保持|
|动作/分数重建错误|独立 NumPy 读取原二次基底与MLP权重重算 score，最大绝对差`<1e-4`；独立由原native SOG/COG、kn→m/s、实际秒和球面公式复算MC，球面位置及共享定义的特征比对`rtol=0, atol=1e-10`；独立重建最低分集合、source→node→mode平局质量及输出，不因score容差新设近似平局|
|评价/尾部集合错接|从保存数组独核全部主位置、共同mask、coverage、完整12配对/组件区间/去最大、量化尾部和三套害/救回集合；场景/节点/模式计数可回溯|
|过度宣称独审范围|四属性用既有评价器完整记账，明确不是全部四属性双实现；特征提取复用同定义，不宣独立物理reference认证或重新审计旧训练|
|接口回归|只做新V-only入口、原T入口和必要共享链路的针对性检查：full与leave模型不得串用、V路径/原名单绑定、原28臂导入、真实事件offset/list/缺报告、单节点O/MC同分混合、并列池内改选、池外节点采用及K/源换序。必要轻量夹具仅验代码行为，不训练、不作MTAD效果证据|

上述score及球面重建容差沿195独审既有精度范围；仅用于不同实现的数值核验。原28臂、非eligible/保持域、速度/航向、分组/阈值、关联、覆盖和新MEAN副本的**exact项仍须exact**，不得放宽成近似以过关。原O分数与full模型独立重建须记录差值和通过条件；实际决策始终使用冻结的原精确分数/原精确平局规则。

必要验收记录须给出实际命令、退出码、覆盖的行为与仍未取得的服务器private证据；本地软件通过不预记V32科学成绩。只有两入口完成、必要检查通过且真实输入路径/资源/终点已写入独立合同后，才能将`IMPLEMENTING/NOT_READY`改为READY。任何绑定/非有限/OOM/写入冲突按一次技术失败保全，不能缩batch、删点、改公式或自动重试。

## 6. 固定资源、终点与结果解释

资源为**一次 science＋一次规定 audit**；各 CPU4 线程、RSS 上限8GiB、无 GPU，各 1800s **advisory**。具体 CPU 亲和按现场兼容资源/FIFO及[服务器规则](../server_autonomy.md)确定，不占用活动152资源或干预他人进程。墙时超建议仅记实际耗时与阶段，不增加训练/样本/配方；固定终点是原V32全部40臂和规定原件/独审完成。技术失败保全，不自动重试。

本轮新增量为 **0 神经/解析/normalizer/似然 fit，0 optimizer 更新，0 teacher forward，0 T 复评，0封存test**。MC构造、冻结风险前向、保存和评价的调用数/耗时/RSS按实际报告；0训练不等0推理或0全链成本。旧模型训练/输入拟合成本保留，缓存导入不作为新旧算法公平速度比较；不宣在线延迟或端到端成本优势。

|实际观察（先通过完整性审计，再分别判断两前端）|解释及后继范围|
|---|---|
|CF−MC MAE<0；H原11害减少、无旧非害新增且70救回全保；S保持0害且42救回全保；相对MC的RMSE/P95/P99均未变差|支持均值与已定义尾部同时保留的开发域迁移；仍报告相对O/front的代价、区间和组件依赖，不称已校准风险或独立泛化。是否进入更大域另立方案，不自动替换178/默认模型|
|CF−MC MAE<0，但害未减少、出现新害、救回丢失或某分位数更差；或仅尾部改善而MAE未改善|判为部分支持/明确取舍，逐项列出得失，不以单列冠军或“可接受”概括通过；先回Squeeze/Mode C定位条件，不沿V追加margin/时间阈值|
|JM相对CF有增量且可由实际池外节点采用解释|单列开放池外候选的价值、范围与成本；不能把CF池内选择收益再次归入JM，也不默认扩大候选网络|
|JM相对CF无清晰增量/仅极少数组点|保留动作与节点分离结论；当前证据下优先研究可区分条件，既不加容量凑收益，也不否定整类候选选择|
|CF无平均或尾部增量，或MAE与已定义尾部同时反向|记未观察到增量或反向，保178/MC原控制，结束当前固定迁移配方的自动扩展。未来可区分动作标签、特征条件、候选可观测性或参考对应；重新训练需另立Idea，不以已有授权默认追加|
|技术绑定/实现失败|只记技术失败，保全原件，不能当科学阴性或换配方续同attempt|

不预设无依据的最小效应、风险容忍量或“CI必须不跨0”的晋级硬门槛。报告效应大小、方向、区间、完整尾部、病例集中程度和工程含义，给条件支持/部分支持/未观察到/反向的判断；CI跨0不机械否决合理信号，点估计改善也不自动认证稳定机制。既有>1km定义只用于保持可比的尾部记账，不是新设允许损害预算。

无自动 V300、测试集、第三seed、新训练、改默认或活动152对照变更。保留完整阴性/失败/未采纳理由，结果先 Record（所有运行/偏差）→Squeeze（均值/尾部/机制/异质性）→Mode C，再决定下一具体动作。当前方案的新增价值是区分冻结动作评分与节点重排的可迁移条件，不宣新算法、绝对首创或已胜同行；合理适配的价值仍由实际结果检验。

## 7. 标准交接字段

```yaml
package_type: IDEA_PACKAGE
package_version: research-pipeline-contract v1.0
workflow_contract_version: research-pipeline-contract v2.0
package_id: IDEA-ACTION-RISK-V32-20261102
idea_id: IDEA-ACTION-RISK-V32-20261102
created_at: '2026-11-02'
updated_at: '2026-11-03'
plan_version: '1.1'
source_matrix_id: MC-MTAD-20261102-v2.09
selected_C_id: C69
source_P_ids: [P12, P13, P14]
source_F_ids: [F305, F298]
source_M_ids: [M123_AI]
source_E_ids: [E208, E201]
source_ids: [SQ-ACTION-CONDITIONED-RISK-195, SQ-MEAN-VELOCITY-VALIDATION-191, REC-HISTORY-CANDIDATES-178, SQ-COST-ROUTING-180]
source_transfer_id: none
research_question: original_O_pool_CF_vs_fixed_original_weights_MC_and_extra_pool_access_JM_vs_CF_on_V32
hypothesis_and_mechanism: frozen_O_risk_may_retain_conditional_ranking_after_only_action_features_6_7_8_change_M123_unverified
testable_predictions: [conditional_CF_mean_tail_value_vs_MC, no_coverage_group_attribute_change, separately_observed_JM_outside_original_pool_increment, original_ties_and_final_node_weights_accounted]
design_and_controls: HARD_STRUCT_ten_prefixes_each_MEAN_NN_40arms_original191_28_exact_12_fixed_pair_comparisons
sample_materials_or_experimental_units: original_V32_9components_1427targets_72196_valid_reference_points_H71396_S70775_eligible784_655
primary_outcome_and_estimand: front_specific_CF_minus_MC_common_point_position_MAE_with_complete_tail_and_preserved_cohorts
secondary_outcomes: JM_minus_CF_and_other_fixed_pairs_all_position_quantiles_harm_sets_coverage_raw_decoded_all_ITSA_MSTSA_four_attributes_costs
analysis_plan_and_uncertainty: point_weighted_component_bootstrap_2000_seed20_descriptive95CI_all_scenes_components_leave_largest_no_independent_confirmation
resources_time_cost_and_risk: one_CPU4_science_one_CPU4_audit_each_RSS8GiB_1800s_advisory_no_GPU_no_fits_no_T_reevaluation_no_test
ethics_safety_privacy_or_compliance: private_identifiers_coordinates_weights_predictions_stay_server_only_public_aggregates_return
stopping_and_interpretation_rules: fixed32_40arms_then_prespecified_audit_technical_failure_preserved_no_auto_retry_or_V300_or_default_change
status: pilot
readiness: COMPLETED_L2S_20261103_196
evidence_status: partial
evidence_locations: sections1_2_and_195_local_public_summary_opportunity_posthoc_191_old_cohorts_MC2.09_SR4.10_OD1.23_and_existing_code
missing_evidence: [independent_domain_and_action_reliability_mechanism]
implementation_stage: actual_science_and_prespecified_audit_complete_see_196_record
next_skill: record-experiment_after_actual_execution
```
