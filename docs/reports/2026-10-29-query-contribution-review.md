# 116：固定身份关系后，query贡献有实际作用，但仍有强控制取舍

记录日期2026-10-29；先记实际执行，再Squeeze，再回Mode C。上游[Idea1.0](../proposals/query_contribution_control_20261029.md)来自C53/P14/P11/P02、F212/M84/E110；[L2S](../../HandOver/L2S-20261029-116.md)。本项是用户新授权的独立一次零训练控制，不是重开113/115或完整ASF/Gelato复现。

## Record：实际发生了什么

`experiment_id=EXP-QUERY-CONTRIBUTION-116-20261029`，`source_idea_id=IDEA-QUERY-CONTRIBUTION-CONTROL-20261029`，plan1.0，`execution_status=COMPLETED/exploratory`。操作者为本地Codex经已配置ssh-campus托管执行。冻结代码80ed87e，独立checkout `/home/wangguangjie/djs/association-code-run116`；原输入和旧运行均未改。

唯一run：`/home/wangguangjie/djs/association-code/runs/LEVEL1-QUERY-CONTRIBUTION-20261029-116-attempt-1`；同一launcher后缀`-launch`，真实服务`mtad-ca3fb9ef7880.service`。Linger=yes、Restart=no、RuntimeMaxUSec=infinity；终态inactive/dead，Result=success、ExecMainStatus=0，`exit_code.txt=0`，launcher returncode0。没有重试、模型、GPU、前向、优化、重新评分/分区/阈值/normalizer、CSV/ZIP读取或封存测试。

实际输入链：原056逐片段Q64插值/OLS状态、source/support/nearest → 原114事前V32的A0/WS分区 → 八臂预测先固定 → 原115完整target/reference/valid和原评价对应 → 误差/覆盖/尾部/场景/组件。115私有输出本身没有逐片段输入，故本次明确复用它的原冻结056输入，不从组均值逆推。`cached_reference_read=true`，reference/identity只用于绑定与评价，不参与W/分区/代表选择。原生时间nearest的距离是线索，实际输出仍为缓存状态，不冒称读取原生点。

### 计划对实际与完整人口

| 事项 | 事前计划 | 实际 |
|---|---|---|
| 样本 | 原115的原114固定V32，不换场、不扩量 | 32场全部完成，9个共享identity组件；具体名单在run/scene-summary，无新随机抽样 |
| 方法 | A0/WS×MEAN/ANCHOR/SOURCE_BALANCED/NEAREST_SUPPORTED | 八臂全部完成；两Mean完整artifact exact重放115；同R四W的分区/代表/valid/全人口/覆盖exact一致 |
| 目标 | 完整registry，含无输入/未匹配，不以unique node重建 | 1427场内目标出现，72196个reference有效位置点，reference unavailable目标0；A0未匹配50、WS未匹配39 |
| 来源支撑 | 来源平衡只用source等价类，nearest仅在support内选 | 算法与接口一致；没有学习权重、独立性或校准概率假设；singleton/空支持回退保留 |
| 资源 | CPU≤4/RAM4GiB/AS8GiB，600s advisory，一次终点 | CPU affinity0–3，AS8589934592B，真实RSS峰2005962752B；程序最终79.939837s，summary计时79.810222s（终盘前），launcher99.682562s；未超建议 |
| 时间 | 独立一次运行 | launcher UTC05:16:10.553347→05:17:50.093820；程序UTC05:16:26.301067→05:17:46.240872，即北京时间13:16–13:17；差额含导入/准备/收尾 |
| 排除/失败 | 不删难例，错误失败保全不重试 | 无排除、无运行失败；软件审查先修schema/摘要层级/异常终态问题，未启动MTAD失败尝试 |

软件核验为7单元和6集成测试通过；最后补严summary版本绑定后单项复核通过。它们使用小合成缓存，仅软件证据。真实效果来自下面实际运行，不把测试数量当研究成绩。launcher退出文件名为`exit_code.txt`；取证最初试读不存在的`exit.json`返回No such file，随后按真实launcher代码定位正确文件；这不是任务失败或数据缺失。

### 全部八臂结果：各自完整目标人口

位置单位m；每个R内可评点相同，跨R必须另看下节共同点。P95/P99来自真实点池，RMSE从合并MSE开方；不是平均场景分位数或RMSE。

| R / W | 可评位置点 | MAE | RMSE | P95 | P99 |
|---|---:|---:|---:|---:|---:|
| A0 / MEAN | 70555 | 1215.514 | 2233.713 | 3359.146 | 7358.117 |
| A0 / ANCHOR | 70555 | 839.024 | 2479.305 | 4252.255 | 10746.665 |
| A0 / SOURCE_BALANCED | 70555 | 1166.893 | 2190.242 | 3255.019 | 7353.762 |
| A0 / NEAREST_SUPPORTED | 70555 | 996.707 | 2346.966 | 4084.559 | 8458.615 |
| WS / MEAN | 71071 | 1276.328 | 2168.266 | 3103.795 | 7194.825 |
| WS / ANCHOR | 71071 | 844.760 | 2457.607 | 4385.889 | 10460.984 |
| WS / SOURCE_BALANCED | 71071 | 1276.441 | 2168.137 | 3097.435 | 7194.825 |
| WS / NEAREST_SUPPORTED | 71071 | 927.435 | 2173.150 | 3873.502 | 7301.504 |

A0四W一律1469组、92重复、1377首组匹配、位置coverage97.7270209%；WS四W一律1605组、217重复、1388首组匹配、coverage98.4417419%。固定R没有gained/lost，也没有通过删输出制造MAE收益。跨R仍有原1234 gained/718 lost；改变W没有解决217重复组或输出对象决定。

### 两个预先主比较、强简单控制与异质性

| 比较，candidate−baseline | 共同n | ΔMAE/m | ΔRMSE/m | 场景MAE好/差/平 | 组件MAE好/差/平 |
|---|---:|---:|---:|---|---|
| WS SB − WS MEAN（主） | 71071 | +0.112594 | −0.128222 | 5/5/22 | 2/2/5 |
| WS NEAR − WS MEAN（主） | 71071 | −348.893739 | +4.884650 | 32/0/0 | 9/0/0 |
| WS NEAR − WS ANCHOR | 71071 | +82.674817 | −284.456498 | 14/18/0 | 5/4/0 |
| A0 SB − A0 MEAN | 70555 | −48.620953 | −43.471195 | 19/1/12 | 6/0/3 |
| A0 NEAR − A0 MEAN | 70555 | −218.807092 | +113.253477 | 32/0/0 | 9/0/0 |
| WS MEAN − A0 MEAN | 69837 | +48.471038 | −80.430610 | 10/14/8 | 3/5/1 |
| WS NEAR − A0 NEAR | 69837 | −79.325584 | −189.568639 | 19/5/8 | 5/3/1 |

共14个事先确定的比较全部存summary，包括两R内ANCHOR控制、SB/NEAR与ANCHOR、四W的WS−A0，不省略负面。上表压缩主要判断；全部逐场/9组件、各自与逐对共同集、inside/outside/far/no_input、same/changed anchor、gained/lost、点收益/损害和纬度/经度论文量均保留于原件。论文ME是有符号bias，residual_SD与印式Eq29诊断分列，作者ESD实现未知；未解释原speed单位或冒称四属性总体方法。

**直接观察DE：** 支持内nearest在两R下相对MEAN的MAE均32/32场、9/9组件改善；WS降348.894m，约27.3%。但WS RMSE比MEAN略差4.885m、P95/P99也更差；对首ANCHOR的MAE仍差82.675m，而RMSE和尾部更好。这不是全面净胜。相同NEAR规则下WS对A0的共同位置MAE/RMSE都改善，说明115的MAE代价并不是“更好的身份R必然带来”；其传递依赖恢复规则。仍是内部开发控制，不是国际同行胜负。

## Squeeze：机制、反例与下一份证据

`squeeze_id=SQ-QUERY-CONTRIBUTION-116-20261029`，输入上述EXP与原115；证据状态partial（实测已完成，独立原数组复核见下节追加，解释/泛化仍有限）。

### 输入机制量与解释边界

| R | 全group-query | 有support | source成员数不均／SB改变W | NEAR改变相对片段均值W | NEAR相对全成员nearest变选择（只支持query） |
|---|---:|---:|---:|---:|---:|
| A0 | 94016 | 69721 | 8208 | 32766 | 78 |
| WS | 102720 | 79415 | 724 | 41607 | 84 |

全group-query包含所有输出组，不限唯一目标可评点；不能直接把它们当MAE分母。SB在WS只有724/79415个有支持query改变数学权重（约0.91%），5场好/5场差、22场几乎不变，**本轮没有改善WS MAE**。A0却有较多source重数不均且SB有局部收益，因此结果限制当前WS上的固定配方，不能推所有来源平衡或provenance处理无效；也不能由此认定数据存在重复原始量测。

NEAR的透明时间线索实际有用，但本轮未加入“所有成员nearest”物理效果臂。84个query选择差异只说明输入规则接近，**不能单凭数量判定支持门控贡献很小，更不能把全部348.9m收益归这个新门控**。所证是完整NEAREST_SUPPORTED相对MEAN的干预效应；最近时间不等误差最小、误差方差或健康概率。仍可能有源偏差、错组、缓存插值/OLS、未来权限和reference语义影响。

ANCHOR在MAE上更强而NEAR尾部更好，给下一阶段明确的目标冲突：需理解何时应保守保留anchor、何时借helper，以及如何在不偷看reference的前提下保留尾部收益。不能直接用V参考挑两个控制的逐query最优或拼模块冠军。成熟的query条件恢复/有限候选输出值得发展；当前无证据要求继续旧TFM、先加attention或训练Gelato。

### 已回答、仍未知与出口

- 已回答：固定R后只改W可以实际改变位置风险，且无需牺牲覆盖；静态关系收益能否传到位置取决恢复规则。W不能改变当前R带来的重复数。
- 当前配方去留：WS来源等权不晋级；保留NEAR的真实MAE/跨场景收益与对ANCHOR的取舍，作为下一方案强透明控制。没有按赢家自动扩到全V、换权重或训练。
- 未知：支持门控与最近选择的分别份额、可用的质量/机动/偏差区分线索、重复输出的合法决定、可信国际同行的同信息实际比较、原reference生成、独立泛化与概率校准。
- 不确定性：开发V反复使用、T/V共享身份，9组件非独立训练seed；32场一致不等独立测试证明。没有新重复、pair/query IID p值或置信区间，不宣论文新问题已被国际领域忽略。
- 后续先回流Mode C；从贡献/输出或可留出共享分量中另选一项，先形成具体合法输入/对照/预算，不沿本次自动开训练。新候选来源分析与116不是互相替代。

### 证据与核验追加

[本地脱敏原件](/Users/a1/Projects/biye-ronghe/docs/evidence/20261029_small_validation/116/)含原run、summary、scene-summary、32逐场、launcher request/launch/status/exit_code/初始及终态服务、stdout/stderr；`derive_public_summary.py`和`terminal-synthesis.json`只由公有结果作事后压缩。完整身份/坐标/逐节点weights/八臂artifact留原服务器private，无预测/权重回传。

**独立原数组核验已完成，退出0。** `audit_saved_arrays.py`未导入生产association/tools函数，只读原116/115/056必要数组，用三维单位球cross/dot的atan2距离替代生产haversine公式。32场/9组件/8臂/14个共同集的MAE/MSE/RMSE/P95/P99、分母/固定支撑与同换代表分层、两Mean exact、W数学约束和覆盖/重复均通过。72188个numeric statistics核对无失败；MAE/RMSE最大差1.916e−9m、P95/P99最大差2.425e−9m、MSE最大差4.530e−6m²、W最大差4.441e−16，均低于事先固定容差（rel1e−9、米统计abs1e−5、MSE abs1e−3、W abs2e−12）。数组exact约束没有放宽。审计11.007336s、RSS171487232B是**事后核验成本**，不是116主运行峰值；0预测/模型/训练/ZIP/test，不是第二次效果实验。

核验范围严格限定：没有独立复核paper_position、delta benefit/harm/better/worse/tie计数、多成员SB物理预测重建或reference物理真实性；这些原字段仍保留，但不称全部字段均获独立核验。[完整独立审计JSON](/Users/a1/Projects/biye-ronghe/docs/evidence/20261029_small_validation/116/independent-saved-array-audit.json)保存verified_scope/not_audited、预定容差和全部脱敏差值。

```yaml
package_type: SQUEEZE_REPORT
package_version: research-pipeline-contract v1.0
package_id: SQ-QUERY-CONTRIBUTION-116-20261029
squeeze_id: SQ-QUERY-CONTRIBUTION-116-20261029
input_experiment_record_ids: [EXP-QUERY-CONTRIBUTION-116-20261029]
source_idea_id: IDEA-QUERY-CONTRIBUTION-CONTROL-20261029
plan_version: '1.0'
created_at: '2026-10-29'
claim_scope: '一次原固定开发V32的透明W干预；非学习融合/独立泛化/可信同行胜负'
planned_vs_actual_design_and_controls: '八臂/固定32/一次/0训练/无排除，均值exact及同R人口不变量通过；无实质执行配方偏差'
primary_result_with_uncertainty: 'WS SB ΔMAE+.112594m，NEAR−348.893739m/32场9组件均好；NEAR对ANCHOR仍差82.674817m但尾部更好'
process_dynamics: '程序79.939837s/launcher99.682562s/RSS2005962752B，0训练，非独立重复'
mechanism_and_intermediate_findings: '固定R隔离W效应；SB在WS仅724支持query改权，NEAR的支持门控与最近选择未各自隔离'
errors_heterogeneity_and_boundary_conditions: '全部8臂/14pairs/32场/9组件/尾部/人口，正文及完整原件'
robustness_sensitivity_and_alternative_explanations: 'Anchor强控制和相同W跨R；错误身份、source偏差、OLS/未来/参考仍可影响；无自动后继'
data_quality_and_provenance_gaps: '真实原始信息provenance、reference生成、可信同行/独立泛化未建立'
new_P: none
new_F: '固定R的query规则可使位置MAE改善，同时保留与Anchor/Mean不同的尾部取舍；来源权重干预强度依R而变'
new_M: '透明W构造→固定组状态变化→同人口物理风险变化；不认定唯一误差根因'
evidence_index: '本页Record/原服务器run与private/本地脱敏原件；独立数组复核另追加'
open_questions: [quality_identifiability, nearest_vs_support_effect, output_duplicates, credible_peers, reference_generation]
next_routes: [deep-read_Mode_C, concrete_following_design_only]
next_skill: deep-read Mode C
evidence_status: partial
```
