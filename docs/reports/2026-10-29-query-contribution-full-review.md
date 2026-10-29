# 117：最近选择的全V收益、强anchor取舍和覆盖反转

2026-10-29；先Record→Squeeze，再回Mode C。上游[Idea1.0](../proposals/query_contribution_full_control_20261029.md)消费C53/F214/M85/E112，[L2S](../../HandOver/L2S-20261029-117.md)。用户本轮新授权独立117，不改116固定终点；零训练，不是作者复现或可信同行胜负。

## Record：实际运行及完整性

`experiment_id=EXP-QUERY-CONTRIBUTION-FULL-117-20261029`，`source_idea_id=IDEA-QUERY-CONTRIBUTION-FULL-CONTROL-20261029`，plan1.0，COMPLETED/exploratory。冻结f333c56，独立服务器checkout `/home/wangguangjie/djs/association-code-run117`；run `/home/wangguangjie/djs/association-code/runs/LEVEL1-QUERY-CONTRIBUTION-FULL-20261029-117-attempt-1`，同一launcher后缀`-launch`，服务mtad-f8ed63393b81.service。

真实returncode0、exit_code.txt0；服务终态inactive/dead、Resultsuccess/ExecMainStatus0、Restartno，Linger=yes。UTC程序06:18:44.514087→06:25:59.739555，435.225501s；launcher06:18:33.371215→06:26:03.734565，450.512890s（北京时间14:18–14:26）。CPU affinity0–3/RSS峰2007044096B/AS8GiB，未超1200s advisory。0GPU/模型/前向/训练/新score/partition/阈值/normalizer/CSV/ZIP/test，无重试/删场/调参。

实际与计划一致：固定原V300/38共享identity组件，两R×四W=8臂、16比较；原116 V32的六臂预测192场景臂直接复用并**只重放评价绑定**，本次新算2208场景臂（原32×2＋新268×8）。原32旧预测未重算，原116及056/114输入只读。所有预测只接显式物理输入/source/support/group，identity/reference仅绑定/评价，cached_reference_read=true。Q64是缓存插值/OLS，不是新读原生CSV点。

A0_MEAN全300完整artifact逐值exact原056；原32导入六臂artifact及public摘要重放、原114分区raw/decoded/public/private绑定、各R内组/首代表/有效mask/人口/覆盖exact不变均通过。新输出包装防止静默zip截断；软件26项相关检查通过，只是合成软件证据。没有实质配方偏差或未完成场景。

### 八臂全300完整位置结果

单位m；同R可评点相同，跨R只看另列共同点。原registry9100目标出现、9099有参考、有效位置454952点（lat/lon同时valid），未参考目标1保留。各自结果非跨R公平共同集。

| R/W | 可评点 | MAE | RMSE | P95 | P99 |
|---|---:|---:|---:|---:|---:|
| A0 MEAN | 442263 | 1224.604 | 3470.074 | 3081.829 | 8188.864 |
| A0 ANCHOR | 442263 | 885.579 | 6307.486 | 3946.704 | 11661.259 |
| A0 NEAREST_ALL | 442263 | 1000.988 | 3526.382 | 4012.677 | 9210.348 |
| A0 NEAREST_SUPPORTED | 442263 | 1001.407 | 3562.795 | 4006.002 | 9256.338 |
| WS MEAN | 441144 | 1294.074 | 3475.752 | 2671.411 | 8363.727 |
| WS ANCHOR | 441144 | 906.787 | 6323.166 | 4103.138 | 11749.949 |
| WS NEAREST_ALL | 441144 | 970.141 | 3481.747 | 3876.666 | 8444.706 |
| WS NEAREST_SUPPORTED | 441144 | 964.568 | 3488.823 | 3866.366 | 8503.414 |

A0固定9353组/610重复/8743首组匹配/357未匹配，position coverage97.210915%；WS固定9779组/1068重复/8711匹配/389未匹配，coverage96.964955%。每R四W完全相同，W无覆盖/重复收益。**原32的WS覆盖优势没有延续到完整人口**：32为98.4417% vs97.7270%，新268为96.6864% vs97.1136%，全300 WS更差0.245960个百分点；重复相对A0增加458。不是删除难例或换population后报告冠军。

### 新268、全300、组件和强简单控制

| candidate−baseline | 人口 | 共同n | ΔMAE | ΔRMSE | 场景MAE好/差/平 |
|---|---|---:|---:|---:|---|
| WS NEAR−MEAN（主） | 新268 | 370073 | −325.783 | +14.186 | 268/0/0 |
| 同上 | 全300 | 441144 | −329.506 | +13.071 | 300/0/0 |
| WS NEAR−ANCHOR（主） | 新268 | 370073 | +53.000 | −3131.006 | 94/174/0 |
| 同上 | 全300 | 441144 | +57.781 | −2834.343 | 108/192/0 |
| WS NEAR−ALL（机制） | 新268 | 370073 | −6.115 | −.407 | 152/88/28 |
| 同上 | 全300 | 441144 | −5.573 | +7.075 | 171/99/30 |
| WS NEAR−A0 NEAR | 新268 | 363273 | −41.248 | −67.811 | 105/80/83 |
| 同上 | 全300 | 433110 | −47.388 | −79.523 | 124/85/91 |

全部16比较、原32、新268、全300、38组件、去最大组件原件均保留，表只压缩主要判断。NEAR比MEAN的MAE在38/38组件改善，但RMSE/P95/P99未同时改善；NEAR比ANCHOR的MAE仅16组件好/22差，比ALL为25好/7差/6平。相同NEAR下WS比A0在17组件好/16差/5平；共同点有收益却不能覆盖新增漏目标/重复代价。

去最大130场组件剩余170场：WS NEAR MAE1311.862/RMSE5032.296，对MEAN1671.815/5013.248（MAE好、RMSE略差）；对ANCHOR1365.815/5418.357则MAE/RMSE及P95/P99都好。**总体Anchor MAE较好不等于各条件总较好**；该分层是事前描述性稳健性，主全表未删最大组件，也未按V成绩选新策略。

跨R同NEAR共同433110点，WS的MAE/RMSE较A0低47.388/79.523m；但gained8034、lost9153，救回196目标、丢失228目标。净少1119个可评点、32个匹配目标，且重复1068>610。不能只凭共同点或F1宣布完整系统净增。

## Squeeze：最近选择、支持内筛选与无支撑回退分别回答了什么

`squeeze_id=SQ-QUERY-CONTRIBUTION-FULL-117-20261029`。支持NEAR整体透明控制的实际MAE增量，而非学习可靠性、物理独立性或唯一误差根因。

WS全300的MEAN→ALL已经取得323.933m MAE收益，ALL→SUPPORTED再取得5.573m；在这个固定同人口顺序分解中，**绝大部分NEAR相对MEAN的差距由普通最近选择取得**，不能把约25.5%的MAE降幅包装为新支持门控的独占创新。新门控相对ALL有真实但小的净MAE变化，且RMSE/P99更差；旧成熟控制应保留为强baseline。

### 门控两个不同操作的预先分层

掩码只由真实预测组的输入support/nearest/index固定，不用真目标support构造输出。WS group-query总625856：有support且异选620、无support且ALL非anchor39497、其余同选择585739。映射到同WS可评目标441144点后：

| 条件 | 可评共同点 | SUPPORTED−ALL MAE/m | ALL RMSE→SUPPORTED RMSE |
|---|---:|---:|---|
| 有support且两法选不同成员 | 541 | −658.122 | 2493.373→1963.921 |
| 全无support，ALL非首anchor而SUPPORTED回首anchor | 5755 | −365.321 | 7329.318→7597.448 |
| 同选择 | 434848 | 0 | 3402.852→3402.852 |

541＋5755＋434848恰为441144。少数query的局部收益不等总体大幅新增；空支撑回退还会恶化平方误差。输入group-query计数不作目标误差分母，全部失败/缺失仍在原总体保留。本节gate细分来自原评价器与预先固定分层，独立原数组审计未覆盖这些细分，不能扩大其独立核验范围。

### 下一步问题及非结论

1. 可信强控制已经更清楚：必须比较Anchor、Mean、ALL、SUPPORTED及必要固定标量组合，不再仅以弱均值作为方法成功门槛。来源/最近时距不自动是质量标签。
2. 全人口出现“关系F1较好而覆盖/重复更差”的具体反例，输出对象决定需要独立研究；固定R只换W无法解决。新方案不能通过真身份/参考删组。
3. 方向借用、可留出共享分量、未知相关风险集合继续分别按[候选矩阵](/Users/a1/Projects/biye-ronghe/docs/surveys/2026-10-27_expanded_search/query_quality_candidates_20261029.md)发展。方向性目前仍未验证；没有因为普通nearest已有先例就否定发展，也没有把最近控制本身叫完整创新。
4. 原32、新268都属反复开发的V、T/V共享身份，38组件不是38训练重复；未给pair/query IID显著性或独立泛化。三场大尾仍保留，reference生成/物理真实性未知。没有自动调参/新seed/扩量/训练。

### 论文数值：必须用新的全300修订局部印象

关联R不变，仍是114 WS的8项对TMTAFF原表均不利。117 WS_NEAR全300的经/纬有符号ME为**+.011772/−4.530651（10⁻⁴度）**，距0均小于TMTAFF表3的6.24/5.88：**这两项纸面同名数字确更有利，不能吞掉。** 但经/纬坐标MAE为71.142426/61.631481，目标内absolute_ME场景均值为56.460492/49.218091，显示总体bias接近0可掩盖正负抵消；原文没有可直接对齐的坐标MAE，作者ESD实现未知，原人口也不同。

本轮米制MAE964.568/RMSE3488.823均高于PTFM原表354.2/2618m。116 V32的“RMSE小于PTFM”只限那32场，在完整V不再成立；旧局部记录保留并明确范围，不能沿用为最新全体结论。两项bias更好不构成同协议或整体胜出。[原表与历史各口径完整对照](2026-10-29-paper-number-comparison.md)另追加本轮事实。

## 独立核验、原件和回流

[脱敏原件](/Users/a1/Projects/biye-ronghe/docs/evidence/20261029_small_validation/117/)包含run/summary/全部300场scene-summary.gz、同一launcher命令/真实退出/服务及stdout/stderr、事后公有压缩脚本/JSON、独立审计脚本/JSON。未重复下载300个分文件；其完整内容在原scene-summary.gz，服务器原逐场仍保留。原32六臂artifact仍指116；新坐标/身份/完整artifact留117 private，不把复用原件伪装新预测。

独立审计未导入生产函数，用三维单位球cross/dot atan2复算8臂、16pairs的主要own/common位置指标及分母，覆盖全300/原32/新268/38组件/去最大组件；anchor及两nearest为输入成员copy，A0 Mean全artifact exact原056。退出0，20160个数值统计核对通过，MAE/RMSE最大差1.528e−10m、P95/P99最大差2.416e−9m、MSE最大差2.161e−6m²，低于预设rel1e−9/米abs1e−5/MSEabs1e−3。事后审计39.551068s/RSS164831232B与主运行成本分开，0模型/新预测/训练/ZIP/test。

限定未独立核：gate/observed/support/anchor细分、paper位置统计、benefit/harm/change计数、逐场数值报告及candidate_minus_A0块、组件标签与独立manifest/114原件绑定（只核报告标签下的重聚合）、WS Mean预测重建、reference真实性。科学证据状态partial指这些解释/泛化缺口，不把全字段或物理真值称已认证。

```yaml
package_type: SQUEEZE_REPORT
package_version: research-pipeline-contract v1.0
package_id: SQ-QUERY-CONTRIBUTION-FULL-117-20261029
squeeze_id: SQ-QUERY-CONTRIBUTION-FULL-117-20261029
input_experiment_record_ids: [EXP-QUERY-CONTRIBUTION-FULL-117-20261029]
source_idea_id: IDEA-QUERY-CONTRIBUTION-FULL-CONTROL-20261029
plan_version: '1.0'
created_at: '2026-10-29'
execution_status: COMPLETED
planned_vs_actual_design_and_controls: '原V300/38组件/八臂16比较/192场景臂复用2208新算，0训练，无实质偏差/排除/重试'
actual_configuration: '冻结f333c56；原run117/launcher与记录正文'
samples_materials_or_data_provenance: '056输入、114固定R、116原32六臂、同一manifest/ref；T_test封存'
claim_scope: '透明最近规则的全开发人口控制，不是独立泛化/学习质量/可信同行胜负'
primary_result_with_uncertainty: '新268 NEAR比Mean MAE−325.783m/268场均好、RMSE+14.186m；比Anchor MAE+53m；全300同方向'
process_dynamics: '程序435.225501s/launcher450.512890s/RSS2007044096B；单次无训练'
mechanism_and_intermediate_findings: 'ALL已解释大部分相对Mean差距；门控与空支撑回退分开；W无覆盖/重复效果'
errors_heterogeneity_and_boundary_conditions: '强Anchor/Mean/ALL的不同loss取舍；全体WS覆盖比A0差且重复增加，去最大组件对Anchor方向翻转'
robustness_sensitivity_and_alternative_explanations: '完整32/268/300/38组件/留最大组件对照；参考与身份泛化未知，无选赢家/调参'
new_P: none
new_F: '全V验证最近控制MAE收益但强控制/尾部/覆盖有代价，pilot覆盖优势不延续'
new_M: '支持内筛选与空支撑首anchor回退的操作份额，限所测固定W干预'
evidence_index: '正文原件、独立保存数组审计及其not_audited范围'
known_limits_and_missing_evidence: [credible_peer_same_task_results, quality_identifiability, output_decision_effect, reference_generation, independent_generalization]
all_runs_or_repetitions: '原117 attempt1唯一一次；原116复用不是重跑'
next_routes: [deep-read_Mode_C, concrete_direction_or_output_candidate_design]
next_skill: deep-read Mode C
evidence_status: partial
```
