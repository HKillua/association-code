# 178：同风险评分器开放组内历史候选

2026-10-31；REC-HISTORY-CANDIDATES-178 → SQ-HISTORY-CANDIDATES-178（research-pipeline-contract v1.0）。来源[Idea178/plan1.0](../proposals/history_candidates_20261031.md)、Mode C1.86/C60及F281/E184、[独立来源复核](/Users/a1/Projects/biye-ronghe/docs/surveys/2026-10-27_expanded_search/history_candidate_transfer_20261031.md)。科研类型为探索性计算实验，原T128训练、原T32/V32评价；不是完整全T训练/全V300成绩或同行复现。科学、规定独审与四属性/动作/训练记账均成功终态，execution_status=completed，证据verified；来源复用，充分精读107不增。

## Record：问题、实际设计与血缘

检验当前硬support排除组内支持外状态是否遗漏可部署价值。冻结175 HARD与173 STRUCT前端、原阈值/WS分组、127节点Q64原状态及159原输出；只改当前组query支持来源≤1的位置，全部速度/航向及多支持状态保留。原组内所有片段为候选，不跨组借真身份/兼容输出，不按源名/编号优先。

共享12维相对物理风险特征→二次ridge91系数→12→32→32→1残差网1505参数，训练目标是T参考监督的log1p(位置误差米/1000)，非概率/协方差。第一批32场用127、其余用151，位置/OLS物理单位及canonical_all权限固定；0前端优化/teacher forward。T128的HARD/STRUCT两分组视图各半质量，group-query等权、源等总质量/源内均分，原first-anchor-first只给首组标签，重复组不删评价。共190232候选标签行/46604.5加权group-query；两视图不是独立样本。

T32十二组件leave-out＋full共13ridge/13神经fit，每fit256末步、4096采样/seed20、Adam.001/wd1e−4/clip1；真实3328GPU更新。leave-out从全部T128排除同组件，但上游前端见过T标签，不能称全链留出。所有模型在V前冻结，V仅推理/评价，无选择阈值/候选cap/新seed。T_test未访问。

每种前端六策略：base原159；M新评分器只选受支持片段、零支持仍base；E全历史源均衡；Z只零支持开放；1只单支持开放；O零＋单联合开放。同一新模型供M/Z/1/O，O−M只变动作范围；O−base同时含新训练/特征，不能全部归开放。每策略配原普通Mean副本，共24臂；Mean副本没有新训练，不当独立实验或学习融合收益。

科学83d9607，新`association-code-run178`，原run `runs/LEVEL1-HISTORY-CANDIDATES-20261031-178-attempt-1`；实际命令/env见[L2S178](../../HandOver/L2S-20261031-178.md)、[启动回执](../../HandOver/S2L-20261031-178-start.md)及原launcher request。CPU0–3×4/GPU0 UUID兼容共卡，3600s/advisory；payload08:05:04.401668Z–08:24:31.913990Z，launcher1167.642590s、程序1160.412064s，RSS2924781568B、Torch显存79365120B。unit`mtad-55ff3b33b7f6.service`真实exit0/inactive/dead/Resultsuccess/Restartno，Linger=yes，无重投/配方改变。5项READY前验收只认证软件链，不是MTAD效果。

## Record：全部位置策略结果

单位米，原T32/V32全目标人口；所有完整/共同点、四属性、raw/decoded、覆盖/重复、场/组件原件一起保留。表中M/E/Z/1/O为H前缀或S前缀，E是源均衡历史位置简单控，非学习混合。

|分组/位置策略|T MAE|V MAE|V RMSE|V P99|
|---|---:|---:|---:|---:|
|HARD 原输出|454.615|452.992|1502.760|5166.150|
|HM 仅支持|454.593|452.992|1502.756|5166.150|
|HE 均衡历史|724.666|692.724|1691.799|7030.841|
|HZ 仅零支持|452.230|447.414|1462.455|5145.835|
|H1 仅单支持|394.617|361.338|1373.325|5021.982|
|HO 联合开放|392.233|355.760|1329.100|5003.510|
|STRUCT 原输出|430.087|427.241|1447.519|4842.083|
|SM 仅支持|430.085|427.238|1447.509|4842.083|
|SE 均衡历史|674.936|648.551|1634.960|7087.815|
|SZ 仅零支持|432.846|423.152|1411.652|4868.593|
|S1 仅单支持|364.443|333.005|1307.750|4348.867|
|SO 联合开放|367.202|328.916|1267.936|4359.930|
|HARD 普通Mean（六副本相同）|1202.191|1201.217|1922.727|5147.138|
|STRUCT 普通Mean（六副本相同）|1227.212|1290.534|1950.607|4918.742|

|V人口与关联（每分组的全部策略不变）|点覆盖|覆盖目标/1427|重复输出|decoded跨源mPCA|decoded跨源mPFA|
|---|---:|---:|---:|---:|---:|
|HARD家族|98.891905%（71396/72196）|1401|203|81.759255%|14.924344%|
|STRUCT家族|98.031747%（70775/72196）|1380|104|81.765139%|15.573316%|

不能把HARD覆盖与SO位置误差拼为一条方法；新阶段没有关联或coverage增量。原HARD与STRUCT交叉强控仍全部保留。

## Squeeze：均值、归因、尾部与边界

V共同点MAE差HO−原HARD=−97.231m、2000次seed20组件描述区间[−146.126,−88.466]；SO−原STRUCT=−98.325m、[−119.589,−82.126]。去最大组件仍分别−107.846/−95.743m。T共同点分别−62.383/−62.884m、区间[−94.816,−26.863]/[−99.953,−53.625]。T12/V9组件、共享开发域且V复用，不能当独立外域确认。

同模型O−M的V差分别−97.231/−98.323m，区间不跨0；相对均衡历史E的区间也均在0以下（H[−446.710,−320.788]、S[−474.582,−301.076]）。支持本设置下**学得的风险读出能够利用部分被硬support排除的历史状态**，不是“所有历史都可信”或盲均衡有效，也不认证唯一机制/传感器健康。

V整体RMSE/P99与原基线同向改善，但HO/SO分别仍新增128/107个相对各自base增害>1km点；最大增害6781.732/16984.346m。增害是误差之差，非绝对误差>1km、也非全模型最大绝对误差。T分别104/75个，最大增害12256.613/15120.432m；T的P99两头反而变差（H5825.207→6195.742、S5114.230→5251.667），必须保反证。

单支持域贡献主要平均收益；零支持在T的STRUCT均值/尾部为负、V的SZ P99也略差，SO相对S1的V P99略差。联合策略是事前列出的固定臂，不据V重新拼策略。候选质量、当前风险目标对高损害误选的惩罚、组内状态/中央对应条件仍为候选解释；本轮未做替代损失或原生状态重估，不能把其中任何一个写成已证实根因。

## Record：规定核验、原件与偏差

独立收尾代码ab5914f，新`association-code-audit178`，原`...178-audit-launch-attempt-1`，CPU16–19×4/无GPU/1800s advisory。unit`mtad-ff25edf9e699.service`真实exit0/inactive/dead/Resultsuccess/Restartno；payload08:24:56.531208Z–08:35:29.665132Z，launcher633.251629s/RSS1183125504B。内部独立主输出核验102.480666s、四属性30.243669s、动作与训练记账491.290390s；其余启动/读写/收尾计入launcher，科学＋规定收尾共1800.894218s，不能当单次部署延迟。

完整重算所有方法/全部raw-decoded、主位置/人口/coverage/重复/尾部/组件/原primary CI；四属性独立核齐。另独立复原每个候选权重/位置、源均衡tie、0/1/多域/速航不变、原HARD/STRUCT/Mean exact，T128两视图first-anchor/重复/未知/有效标签及全部13fold排除组件/标签数/权重/91＋1505架构/曲线与3328更新齐备。不是优化器逐步重放或物理真值认证。

全部98公共原件（83833344B）＋含真实服务终态的传输清单回[178证据](/Users/a1/Projects/biye-ronghe/docs/evidence/20261031_history_candidates/178)，逐文件大小核齐；tar9017252B。private原坐标/身份/权重/完整预测/候选分数与权重留server。初次下载因本地父目录未建被工具拒绝，建目录后成功；这是传输修复，不是科学重试。科学/审计各一次，无超时/失败重投/计划外模型或科学配方变化。完整命令/退出、时间/环境/预算、全部曲线与结果均可从原件复核。

## Squeeze：选取差距、反例与下一条件

|固定分组/单支持同一人口|T query数|M相对全历史oracle仍差>1km|1相对同oracle仍差>1km|V query数|M差距>1km|1差距>1km|
|---|---:|---:|---:|---:|---:|---:|
|HARD|13492|1062|181|29883|2655|337|
|STRUCT|11919|1085|155|25320|2582|297|

该量是相对**全历史候选**的误差差距，不等于M在其受支持合法集内排错；M/O同评分器，不能归为新训练消融胜出。零支持V另有2003/1662个有参考query，O仍有83/87个相对候选oracle差>1km；联合420/384是在31886/26982人口，不能与仅单支持不同分母直接比较。oracle只作后验，不用于推理。

V原始组输出层面，HO/SO分别改变12550/10512个位置query；28235/23896个query采用支持外正质量（许多与原状态数值相同，采用数不等改变数）。这些含重复输出的raw组query，与主评价人口不同，不混分母。V多支持域误差差严格0、全部速航不变；raw/decoded分数/分区及全部coverage也exact不变。

|V联合−对应base|全部共同点|0域点/新增害|1域点/新增害|多域点/新增害|纯组新增害|混组新增害|改善>1km点|
|---|---:|---:|---:|---:|---:|---:|---:|
|HO−HARD|71396|2003/32|29883/96|39510/0|28|100|2502|
|SO−STRUCT|70775|1662/47|25320/60|43793/0|19|88|2406|

所有新增害same-anchor。T联合新增害的0/1域为H14/90、S38/37；纯组H22/S20，不能全归混组。T STRUCT零域均值反而+107.693m；T最坏H纯零点由960.547→13217.160m。V最坏H纯单支持点（场2502/query40）3880.921→10662.653m；S混零点（场3772/query61）247.313→17231.659m。域/纯度是发生条件，不证明传播、混组或loss是唯一根因；旧177的122个原纯零害没有被本轮声明全部救回。

**保留当前历史候选风险读出的平均收益，不宣尾部已解或全部强控整体胜出。** 下一回流Mode C1.87/F283/E186、SR v2.9后，以纯单支持风险误选与纯/混零支持传播分别结合来源形成可区分干预，保护多支持域和整体收益；不直接沿V设时距cap、拼策略或无依据加同头训练。跨片段可部署历史链接及原生状态重估仍独立保留，不退固定源。真实异构K≥3及新域、在线/端到端未验证；canonical_all是离线权限，1505仅新增头参数，非全流水线总参数。

```yaml
package_type: SQUEEZE_REPORT
package_version: research-pipeline-contract v1.0
squeeze_id: SQ-HISTORY-CANDIDATES-178
input_experiment_record_ids: [REC-HISTORY-CANDIDATES-178]
source_idea_id: IDEA-HISTORY-CANDIDATE-ACTIONS-178
plan_version: '1.0'
evidence_status: verified
claim_scope: original_T128_fit_and_T32_V32_fixed_HARD_STRUCT_groups_offline_24_arms
primary_result_with_uncertainty: matched_action_opening_V_common_MAE_minus97_minus98m_component_intervals_below_zero_but_reused_development_V
process_dynamics: 13_analytic_plus13_neural_fits_3328_GPU_updates_science1167.643_plus_audit633.252_seconds
mechanism_and_intermediate_findings: historical_candidates_can_be_selected_without_reference_in_this_setting_not_calibrated_health_or_unique_cause
errors_heterogeneity_and_boundary_conditions: V_new_harms128_107_pure28_19_zero32_47_one96_60_T_P99_worse
robustness_sensitivity_and_alternative_explanations: M_E_Z_1_O_shared_model_controls_full_population_independent_saved_arrays_folds_curves_and_domains
data_quality_and_provenance_gaps: shared_T_V_identity_development_population_true_K3plus_new_domain_and_physical_reference_origin_unverified
new_F: F283
new_E: E186
new_M: none
next_skill: deep-read_Mode_C
no_automatic_new_run: true
```
