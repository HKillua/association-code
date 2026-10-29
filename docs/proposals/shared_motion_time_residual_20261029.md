# 122：共享差中是否存在可留出的速度相关剩余

2026-10-29，执行前Idea1.0，消费[输入/时序矩阵§7](/Users/a1/Projects/biye-ronghe/docs/surveys/2026-10-27_expanded_search/input_time_action_matrix_20261029.md#7-后续增量先核非平凡作用再分离共享修正的空间与时间解释)的IT-C3。本文只形成可审查方案及实现，不提供READY或执行状态。独立于120/121，不读取其结果择参；118旧配方不重跑。

## 问题及机制边界

118表明共享常量相对修正有用，但不能区分空间位移、时差、机动/插值与错关联。已读TSP2026在后验层动态预测再估相对时差，GPTR在原生时刻建因子；它们均不证明当前MTAD有真实clock误差，也不提供无额外条件的绝对空间锚。

本项只检验**离线重建运动的速度相关项**是否在留出组超越常量项。高源若在记录时刻`t`实际落后`τ`，局部一阶工作模型为`d_g=p_H−p_L≈b−v_gτ`，故`τ>0`表示高源落后。空间项b为北/东米，v为北/东米每秒，τ为秒。若各组速度相同，b和τ不可联合识别；不同目标即使分别恒速，速度向量不同也可提供识别信息。解析的`D_v=mean(||v−vbar||²)`与设计秩如实报告，不能把拟合出的秒数叫物理时钟。

## 冻结输入与操作

- 原118固定V32/9组件、原056完整Q64物理输入、原114 WS membership及原118保存membership/source_order/group_fold绑定。不得改组、代表、预测、输出valid或任何物理状态；不读ZIP/CSV/T_test。
- NPZ可能含target和identity，但122只提取`native_states/support/query_seconds/sources/node_ids`；原118只取WS membership、source_order和group_fold，不取预测、reference、旧中心或权重。node_ids仅作跨缓存绑定，不参与拟合；身份和reference字段不加载。
- 来源/人口绑定复用既有metadata读取链，会读取含既有参考误差汇总的056等摘要文件；这些摘要不参与拟合或选参。因此“未读取参考/身份”仅指相关数组字段未加载，不能扩大为整个进程完全没有读取含评价信息的文件；run中的false字段相应限定为`*_array_fields_loaded`。
- 按118保存的低/高源字符串顺序定义方向，每一组取全部跨源fragment pair。只有`q=1..62`且两成员在`q−1,q,q+1`都support时，取中心位置差和中心差分速度；原Q64的时间秒严格递增，航速属性不使用。
- 每个合法pair-query：纬差为高−低，经差为最短弧；每源导数由q−1到q+1的纬差/经度最短弧除真实秒差，两源导数平均。先在每组内平均d和v，再以组等权拟合，不能把密集pair/query当独立重复。
- 三fold=`group_rank%3`，与118保存值exact；当前fold全部组不进入其参数或chart。chart纬度只用其余fold中有导数的组的可见support纬度均值，按`Rπ/180*[1,cos(lat0)]`同时换算d和v；chart仅工作坐标。无训练合格组时全部模型不可用、残差记null，组仍在分母。
- 同组/同折四臂：`STATIC_LS`只拟合b；`TIME_LS`在b=0假设下只拟合τ；`JOINT_LS`拟合b和τ；`JOINT_PERMUTED_LS`同联合模型但打乱速度对应。全部采用FP64最小二乘，组权相同，不把118的coordinate-median换成LS收益算时间项。
- 置换规则事先冻结：**每场新建一个`default_rng(20)`，按fold0/1/2顺序，分别对该fold的训练合格组、留出合格组各调用一次`permutation`**。只换速度行，d/组/支持不动，训练和留出集合不交叉，不择置换、不拒绝恒等置换；报告实际换位数。该阴性对照不读任何reference，并保留各折速度边际分布。
- `lstsq(rcond=1e-8)`仅作固定数值秩判断。TIME/JOINT及置换联合至少两训练合格组且相应设计满秩；否则回退同折STATIC、τ为null并记原因。TIME在b=0限制下可能可估而JOINT不可辨，必须区分。没有有效训练组则STATIC也不可用。奇异值、D_v和回退人口均报，不把数值阈值称科学可辨界限。

## 主要结果、资源与出口

主比较为留出组`JOINT_LS−STATIC_LS`的`||d−b+vτ||`均值及RMSE差；同时保留四臂全部六配对、组残差中位/P95、逐场和组件异质性、去最大组件敏感性。训练残差降低由嵌套模型必然，不作为作用证据。完整组数、跨源共同支撑pair-query、三点共同支撑导数pair-query、有/无导数组、训练不足/秩不足/无chart及真正可评留出组分别保留；不按误差/身份筛样本。

公开只回传脱敏计数、残差统计、秩/奇异值、参数范数及跨fold稳定性摘要。实际chart、带符号b/τ、组位移/速度、置换行号与残差数组留服务器`private/`。没有任何物理输出或参考误差变化，不能称降低MTAD绝对恢复误差、发现时钟、复现TSP、在线端到端或胜同行。

一次attempt-1、原32终点、CPU≤4/RSS4GiB/AS8GiB、600s advisory，0GPU/神经训练/模型前向。解析拟合本身的次数和真实CPU成本记录；失败保全，不自动重试/扩量/改τ范围/应用时间补偿。预测/reference不读取为目标，不需要重播118指标。

若联合在留出组更好且阴性对照没有同量收益、不是少数组/最大组件/病态折主导，可推进**另一个**native时间补偿Idea；这仍只支持速度相关结构。若只训练好或置换同样好，不支持所提中介；若联合多不可辨，保留静态修正及条件边界。结果先Record→Squeeze→Mode C，不能自动继续物理补偿。

## 实现与验收

| 行为 | 必要检查 |
|---|---|
| 参数仅由合法可见输入、当前fold之外产生 | 显式allowlist；扰动留出fold不改变该fold参数/chart；原缓存含毒target/identity不读取 |
| 符号和可辨条件 | 已知b/τ解析构造可恢复，恒同速度的联合退化为静态、τ=null；TIME的条件性保留 |
| native时钟主张边界 | 只有Q64内三点支持差分，数值秒系数仍叫工作参数；不改状态或读取速度属性 |
| 原缓存与失败保全 | 056/114/118节点/source/组/fold exact绑定；真实schema的软件夹具、失败第二场保留第一场与FAILED记录；禁止覆盖 |
| 完整分母与隐私 | 所有组/无定义组计数保留；公开JSON无坐标/身份/带符号参数，数组只写private |

```yaml
package_type: IDEA_PACKAGE
package_version: research-pipeline-contract v1.0
package_id: IDEA-SHARED-MOTION-TIME-RESIDUAL-20261029
idea_id: IDEA-SHARED-MOTION-TIME-RESIDUAL-20261029
plan_version: '1.0'
source_matrix_id: MC-INPUT-TIME-ACTION-20261029@followup
selected_C_id: IT-C3
source_P_ids: [IT-P3]
source_F_ids: [IT-F10, IT-F11, IT-F12, IT-F13]
source_M_ids: [IT-M6]
source_E_ids: [IT-E10, IT-E11, IT-E12, IT-E13]
research_question: 共享相对差是否有留出组可重现的速度相关剩余
hypothesis_and_mechanism: 跨对象速度差异使时间列可与常量空间列分开；同速时拒绝联合解释
testable_predictions: JOINT相对STATIC的留出残差及置换/秩/Dv/跨fold稳定性
design_and_controls: 本文四臂六比较、固定三fold与一次seed20训练和留出独立置换
sample_materials_or_experimental_units: 原118固定V32/9组件的原WS组，组等权输入残差，非独立测试
primary_outcome_and_estimand: 留出组JOINT减STATIC的相对残差范数均值及RMSE差
secondary_outcomes: 全配对/中位/P95/无定义人口/组件/去最大组件/成本/拟合稳定性
analysis_plan_and_uncertainty: 组与场景组件分开；不把query作为IID，null不填零，不据V真值选择
resources_time_cost_and_risk: 单次CPU4/RSS4GiB/AS8GiB/600s advisory/32终点；0神经训练
ethics_safety_privacy_or_compliance: 只读物理字段；坐标组索引带符号参数留private；T_test封存
stopping_and_interpretation_rules: 本文出口；不自动重试扩量或应用物理补偿
status: diagnostic
evidence_status: partial
missing_evidence: [heldout_motion_structure, native_clock_identifiability, physical_fusion_increment, credible_peer_effect]
next_skill: record-experiment_after_actual_execution
```
