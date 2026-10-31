# 182：T侧原生运动相容性与传播条件

2026-10-31；REC/SQ-NATIVE-COMPATIBILITY-182，research-pipeline-contract v1.0，source_idea_id=IDEA-NATIVE-COMPATIBILITY-QUALIFICATION-182、plan1.0；[事前Idea/spec](../proposals/native_compatibility_qualification_20261031.md)、[L2S182](../../HandOver/L2S-20261031-182.md)。探索性条件核查，科学一次exit0，execution_status=completed/evidence_status=verified；不产生新方法分数、训练或因果证明。

## Record：实际输入、操作、核验和成本

原178/181固定组/输出与127完整原生事件，原T32/860对象；0V、训练、teacher、新部署或测试。相等源内合并原生时间、位置/速度报告，显式将canonical kn一次转m/s；从同源nearest事件及strict brackets测报告外推距离、节点OLS/局部secant/报告速度差与时间条件。特征函数不接身份/参考；分析层才用原标签定义纯度、原增害及后验候选oracle，不把oracle部署。

主有效T HARD37993/STRUCT37745、178相对原base新增>1km害104/75、181 B相对O新增10/5全部与既有独审exact。全0/1/多、纯/混与before/after/有无双侧分母/所有分位数保留。新测量13维，缺报告有known标记；multi字段不适用的零值不解读为可靠0残差。4项READY前检查通过，含物理单位/日期线、K3匿名换序、缺失/跳变、第一对应/固定输出统计链；只认证软件。

新2dd82b4 checkout182，原run `runs/LEVEL1-NATIVE-COMPATIBILITY-20261031-182-attempt-1`，实际命令/环境同L2S。CPU16–19×4/无GPU，900s advisory/RSS4GiB，Linger=yes/Restart=no；原unit`mtad-7aede4266a48.service`真实inactive/dead/Resultsuccess/ExecMainStatus0。launcher payload10:23:16.058354Z–10:24:06.586011Z、50.646276s；程序10:23:19.921743Z开始，46.026143s、RSS652668928B。无重试、改定义/原输出、配方或人口偏差。

45公共原件1393605B＋真实服务终态清单回[182证据](/Users/a1/Projects/biye-ronghe/docs/evidence/20261031_native_compatibility/182)，tar753615B、逐文件数/大小核齐。private逐query/坐标/身份留server。本项无新部署，未额外进行一套新方法全输出审计；原分母/增害绑定已完成的178/181独审，既有参考物理准确性仍未知。

## Squeeze：三类发生条件及反证

|T前端/域|全点|178新增害/改善>1km|181桥接新增害/改善>1km|
|---|---:|---:|---:|
|HARD纯单支持|11744|19/822|0/0|
|STRUCT纯单支持|10458|20/817|0/0|
|HARD零支持有同源双侧|639|9/34|10/24|
|STRUCT零支持有同源双侧|616|33/35|5/49|
|HARD零支持无双侧|469|5/13|0/0|
|STRUCT零支持无双侧|351|5/9|0/0|
|HARD零支持全组时间之前/之后|189/262|0/2；5/10|0/0|
|STRUCT零支持全组时间之前/之后|121/223|0/0；5/9|0/0|

纯单支持所选节点与nearest报告速度球面外推的距离log1p(km)，新增害中位H4.260587/S4.177596、改善点1.618866/1.623707；两组范围明显重叠（害最低.774759，改善最高4.977568），不能直接设阈值。所选−后验oracle候选同特征的害最小均+.719830，中位+4.206295/+4.068216；改善点中位0。这支持把**原生局部运动相容性**作为有价值的新增输入条件尝试，未证明报告速度就是真实速度或所有误选来自此项。

纯单支持害的所选节点OLS−native报告速度差log1p(m/s)中位均3.518990，改善点3.4461/3.4550，单独速度差不形成清晰门限；报告−局部位置secant差的害中位仅.0324、改善3.4139/3.4229，盲目偏爱“自一致报告”也可能排除有益历史。新特征须与原12维共同学习/控制，不能只选支持解释的一列。

双侧域B−O新增害的选中source gap中位H721.596/S1063.099秒，改善点1316.824/2174.005秒；对应secant−两端报告均值速度差中位14.863/8.990m/s，改善点26.120/24.179m/s。**大gap或大速度不一致并不是该批害的充分判别条件**。H害gap最小83.2秒，改善最小191.476；分布有交叉，不能直接采用长gap/高残差拒绝。

无双侧域469/351中，全组before/after合451/344，其余18/7为组内包络但各源均不夹住query。单侧及有双侧恢复是不同操作域；after仍有5/5个原178新害，不能以双侧桥接结案代替处理单侧。全组before的原新增害虽0，已有绝对大误差仍保留，不能只评价新增害病例。

DE：新增局部原生运动测量与纯单误选有关联，双侧端点不宜用单个时长/不一致阈值处理，单侧无观测域尚未干预。AI/UN：相容性信息缺失、真实机动、同源片段不连续、噪声/偏置、参考对应的作用未分离；未判唯一根因/数据错误。后续保178及两套强控，分别形成新证据输入、双侧运动模型、单侧持续性干预；先合独立来源/Mode C，不止于本诊断、不沿V设cap或回退固定源。

```yaml
package_type: SQUEEZE_REPORT
package_version: research-pipeline-contract v1.0
squeeze_id: SQ-NATIVE-COMPATIBILITY-182
input_experiment_record_ids: [REC-NATIVE-COMPATIBILITY-182]
source_idea_id: IDEA-NATIVE-COMPATIBILITY-QUALIFICATION-182
plan_version: '1.0'
evidence_status: verified
claim_scope: T32_all_fixed178_181_main_points_no_V_fit_or_deployment
primary_result_with_uncertainty: descriptive_native_compatibility_associations_with_overlapping_distributions_no_causal_or_threshold_claim
process_dynamics: one_CPU_run50.646276_seconds
errors_heterogeneity_and_boundary_conditions: pure_one19_20_bracket_harms10_5_unbracketed469_351_one_sided451_344
new_F: F288
new_E: E191
new_M: none
next_skill: deep-read_Mode_C_then_source_analysis_before_method_Ideas
```
