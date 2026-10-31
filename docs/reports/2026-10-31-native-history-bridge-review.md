# 181：同源原生双侧历史连接

2026-10-31；REC-NATIVE-HISTORY-BRIDGE-181 → SQ-NATIVE-HISTORY-BRIDGE-181，research-pipeline-contract v1.0。来源[Idea181/plan1.0](../proposals/native_history_bridge_20261031.md)、Mode C1.89/C62及[GPTR定向复核](/Users/a1/Projects/biye-ronghe/docs/surveys/2026-10-27_expanded_search/history_candidate_transfer_20261031.md)。探索性计算实验，execution_status=completed，科学与规定收尾各一次成功、证据verified。复用既有来源，充分深入108不增；不是全V300、同行复现、物理真值或新域确认。

## Record：问题、设计与实际执行

针对178零支持时的单片段传播，检验当前预测组中同一源的原生前后事件能否提供更好的位置。固定175 HARD/173 STRUCT的分数、阈值、分组及178 O；只改zero且同源有严格双侧native事件的query，单/多支持、无资格zero及全部速度航向exact沿178 O。无源名/编号优先、GT净化、跨组真身份借用、训练或新拟合；0 GPU/teacher/优化更新，T_test未访问。

同源同时间多片段事件等质量合并，取query严格前后的最近原生端点。R只在合资格源内用旧节点状态；N取同一选中源的最近native端点；B用同一选中源的双端时间权重连接；E用全部合资格源的等源质量连接。R/N/B共用冻结178风险分数选源，平分源间/源内分数tie；不从参考标签选择源。B为一般LTV GP的局部Brownian位置特例所启发的工作prior，经球面纬度/圆周经度适配；不是GPTR的WNOJ/SE(3)复现、校准后验或任意运动下的真实轨迹保证。

原T32/860对象、V32/1427对象，各前端base/O/R/N/E/B及各自普通Mean副本，共24臂。T raw资格H1012/S1007，合2019，非空后按事前规则完整评价全部V32，规则V前冻结；V raw资格H1657/S1486。中央主评价的合资格点T639/616、V1155/992，与含重复输出的raw组query不同。无T/V调门限或挑臂拼接，V仍为复用开发集。

科学与独审均fcef505，独立`association-code-run181`/`association-code-audit181`；原run `runs/LEVEL1-NATIVE-HISTORY-BRIDGE-20261031-181-attempt-1`，原命令/环境见[L2S181](../../HandOver/L2S-20261031-181.md)、[实际启动](../../HandOver/S2L-20261031-181-start.md)及launcher request。CPU16–19×4、无GPU，科学3600s/独审1800s均advisory，Linger=yes/Restart=no。科学payload09:38:39.859567Z–09:47:09.137651Z，launcher509.415630s；程序09:38:46.068767Z–09:47:07.468195Z、501.399487s、RSS2290364416B。unit`mtad-b64da3f9c141.service`真实exit0/inactive/dead/Resultsuccess。

## Record：完整位置结果与不变人口

单位米；所有完整/共同点、场/组件、全部四属性、raw-decoded原件均保留。

|前端/策略|T MAE|V MAE|V RMSE|V P99|
|---|---:|---:|---:|---:|
|HARD原base|454.615|452.992|1502.760|5166.150|
|HO原178|392.233|355.760|1329.100|5003.510|
|HR仅资格限制|392.428|356.753|1330.637|5003.510|
|HN最近native|401.474|363.446|1342.876|5008.696|
|HE全部合资格源等桥接|394.947|360.842|1334.263|5011.074|
|HB同源时间桥接|390.259|356.142|1330.886|5011.074|
|STRUCT原base|430.087|427.241|1447.519|4842.083|
|SO原178|367.202|328.916|1267.936|4359.930|
|SR仅资格限制|367.422|329.525|1269.366|4368.098|
|SN最近native|372.907|336.079|1282.425|4378.070|
|SE全部合资格源等桥接|366.202|333.472|1272.058|4369.058|
|SB同源时间桥接|361.545|328.403|1268.273|4368.416|
|HARD普通Mean（六副本相同）|1202.191|1201.217|1922.727|5147.138|
|STRUCT普通Mean（六副本相同）|1227.212|1290.534|1950.607|4918.742|

HARD家族V点覆盖71396/72196＝98.891905%、1401/1427对象、重复203，decoded跨源mPCA81.759255%/mPFA14.924344%；STRUCT家族70775/72196＝98.031747%、1380对象、重复104、81.765139%/15.573316%。全部策略的分组/分数、coverage、速度航向exact不变，不称关联/覆盖或四属性学习收益。不能拼HARD覆盖和SB误差成一条系统。

## Squeeze：桥接的条件价值与完整强控反证

|V共同点MAE差：候选−对照|HARD|STRUCT|
|---|---:|---:|
|B−O原178|+.381，CI[−1.985,2.428]|−.512，CI[−2.887,.207]|
|B−R仅资格|−.611，CI[−2.143,−.355]|−1.122，CI[−3.009,−.891]|
|B−N最近native|−7.305，CI[−7.710,−1.775]|−7.675，CI[−10.063,−3.979]|
|B−E等源桥接|−4.700，CI[−5.287,−1.451]|−5.069，CI[−5.807,−1.416]|

原T12/V9组件，2000次seed20描述bootstrap，不独立确认性检验。T的B−O分别−1.973/−5.657m、区间均低于0；V均跨0，去最大组件V为−.215/−1.592m，仍不能据后验去组件晋级。B相对R/N/E有条件平均价值，但**没有稳定超过完整原178**；资格mask自身可丢失较好旧候选，不能将B−R的改善全部当B−O增量。

主评价合资格域T B−O均值H−117.323/S−346.631m，V却H+23.574/S−36.541m；V该域P99 H3638.902→4054.227m、S4054.620→5815.624m。V全体P95改善H561.988→547.987/S503.993→493.061m，但RMSE及P99两头反而略升。时间资格不等物理可靠性，平均/尾部必须一起判断。

|B相对原178 O|T HARD|T STRUCT|V HARD|V STRUCT|
|---|---:|---:|---:|---:|
|新增增害>1km点|10|5|41|23|
|改善>1km点|24|49|16|33|
|其中纯组新增害|5|5|20|12|
|最大增害m|4921.504|1915.828|20941.250|20941.250|

以上增害为新旧误差之差，不是绝对误差。另以最初175/173的base作为固定门槛：V旧178的>1km害集合128/107，B后变159/111；旧害消除4/13、引入35/17、持续124/94，纯组旧28/19→41/24。这个集合跨门槛的变化不等于B−O新增41/23，两种定义并报，不能相减混用。T原104/75→108/73，亦不支持尾部普遍解决。

DE：双侧时间信息相对最近端点与mask可有平均价值；真实完整O控制及V纯组20/12新增大害证明“同源且双侧”不足以保证可靠恢复。AI/UN：噪声/系统偏置端点、运动弯曲、错误历史连续性、状态与reference对应均仍可解释，**本次未区分根因**；纯组也不能据此判reference错。单侧长传播/无合法双侧域完全保持旧输出，本轮没有解决这部分。

## Record：规定核验、原件与偏差

规定独审在181科学结束后、180独审前FIFO一次执行，unit`mtad-2dea828a45db.service`真实exit0/inactive/dead/Resultsuccess/Restartno；payload09:47:26.076346Z–09:51:14.108213Z，launcher228.158755s、RSS1182056448B。独立主位置/人口/覆盖/全部raw-decoded/尾部/组件/primary CI重算105.641475s；四属性从同保存数组用现有评价器完整记账32.972996s，**并非四属性另写算法的独立复现**；原生资格/同源端点/冻结选源/全部新位置及不干预域独立重建82.376226s。不得把结果JSON中概括的scope读作物理真值或所有属性均有独立双实现。

科学＋规定收尾737.574385s，不是部署延迟；0训练/拟合/teacher。全部95公共原件83483018B与真实服务终态清单回[181原件](/Users/a1/Projects/biye-ronghe/docs/evidence/20261031_history_followups/181)，tar11491574B，逐文件数/大小全部核齐。private坐标、身份、权重、完整预测/风险与原生事件留server。11项READY前软件检查均通过；科学/独审各一次，无失败重试、加步、选V门限、实际配方改变或测试访问。结案后主仓只澄清audit的scope描述，原fcef505冻结代码和原始JSON均不改，无数值重算。真实数据主要K2，K3仅接口测试，不认证异构多源泛化。

**保留178，181当前配方不晋级/不扩大。** 保存B相对R/N/E的条件证据及完整O反证；下一从合法T反例区分端点运动/噪声相容性、历史链接与单侧传播条件，再结合充分来源回矩阵形成新方案。不能直接按V设gap cap，不能与180拼局部冠军。先回Mode C1.90/F286/E189、SR v3.1，再决定后继；无自动新READY。

```yaml
package_type: SQUEEZE_REPORT
package_version: research-pipeline-contract v1.0
squeeze_id: SQ-NATIVE-HISTORY-BRIDGE-181
input_experiment_record_ids: [REC-NATIVE-HISTORY-BRIDGE-181]
source_idea_id: IDEA-NATIVE-HISTORY-BRIDGE-181
plan_version: '1.0'
evidence_status: verified
claim_scope: fixed_T32_V32_HARD_STRUCT_zero_same_source_native_brackets_24_arms_offline
primary_result_with_uncertainty: B_beats_R_N_E_but_V_vs178_plus0.381_minus0.512m_intervals_cross_zero
process_dynamics: zero_fit_updates_science509.415630_plus_audit228.158755_seconds
mechanism_and_intermediate_findings: time_bracket_condition_has_limited_value_not_physical_reliability_or_unique_cause
errors_heterogeneity_and_boundary_conditions: V_new_harms41_23_pure20_12_max20.941km_P99_worse_single_sided_unchanged
robustness_sensitivity_and_alternative_explanations: independent_positions_populations_relations_native_reconstruction_and_domains_same_evaluator_four_attribute_accounting
new_F: F286
new_E: E189
new_M: none
next_skill: deep-read_Mode_C
no_automatic_new_run: true
```
