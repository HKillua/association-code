# 136/137观测驱动探索：初始行为规格

2026-10-30，版本1.0，**实现已确认，未READY**。对应[两个独立Idea](../proposals/observation_driven_exploration_20261030.md)，消费观测驱动Mode C v1.1的OD-C1/C2。root已审查本合同；此文不是READY、不执行服务器。以下API为实现合同，文件/函数在实现完成前不保证存在。

## 1. 共用数据入口与隔离

- 复用`tools/analysis/source_risk_pilot.py::metadata/physical/evaluation`的既有绑定规则：127/125/114成功终态、原T32/V32及12/9组件、125native_run和partition lineage、native-cache completed、节点/query一致。另核133完整成功终态与相同native/value/partition/manifest人口。
- 输入物理allowlist：`native-motion`的states[N,64,4]、support、nearest_seconds、sources、native_count、span_seconds、fit_rmse_m、ols_velocity_mps、node_ids、query_seconds；`native-events`的node_ids/sources/query_seconds/event_offsets/event_times_seconds/event_values/event_valid/event_node_ids/event_canonical_indices。位置事件严格按原`native_records`的有效lat/lon与时间规则读取；不读identities/target进预测函数。
- T memberships来自125私有T-features，V来自114 `memberships.WS`。相同source只做等价类，不将字符串/顺序编码为质量。所有新臂speed/heading与原WS_MEAN exact；节点/组/valid/首代表/重复/无支持fallback保持。
- 137去掉heldout位置后才调用`cv_native_positions`；保留的观测才决定经度展开、原点、east scale、P0和更新。不得先用全序列构chart再mask。
- 预测在独立评价文件读入前完成并保存；136T/V均不作label拟合。137完整T选择完成且冻结后才处理V；T原生评分数据不是reference标签。

## 2. 136纯计算模块建议API

新文件候选`association/evaluation/observation_quality.py`：

`native_quality(*, native_cache, motion) -> dict`

- 对齐node_ids/sources/query_seconds。返回`ols_proxy_m2[N]`、`loo_proxy_m2[N]`、`loo_known[N]`、`loo_residual_count[N]`和事件计数。OLS是缓存fit_rmse平方；LOO按Idea相邻原生内点插值，米制平面残差均方除以噪声增益。n=2返回known=false及有限0填充值；unknown不得当可靠0用。
- 非有限、shape/时间单调性/有效位置异常直接ValueError，保全run，不fallback到另一算法。所有输入只读。

`source_proxy(*, node_proxy_m2, node_known, support, pack) -> dict`

- pack沿133 `build_source_features`保存的source mapping/group_offsets；输出`quality_m2[S,Q]`、`known[S,Q]`、`available[S,Q]`。仅本源当前support成员参与，算术平均；任一成员unknown使该source unknown。无support不参与，也不影响其他已可用source。

`observation_weights(*, proxy, pack, nearest_seconds, support, kind, floor_m2=None) -> dict`

- kind为equal/nearest/inverse；inverse要求floor为本合同1或100，按inverse-max归一化。一个组query任一可用源unknown则equal，保存`uninformative_fallback[G,Q]`。0source全0，1source为1，所有精确平局对source等分。
- nearest先取每个source当前support成员最小nearest，再对可用source选最小；源贡献最后均分给被选source的所有support成员。不能误用历史nearest-node。
- 返回`source_weights[S,Q]`、`node_weights[N,Q]`、`floor_hit[S,Q]`与计数。不能接reference/identity；常量proxy产生equal用于无信息软件对照。

`recover_observation_policy(*, states, support, groups, pack, source_weights, mean_states) -> dict`

- 节点权与原`recover_group_states`链一致；136所有0support输出原Mean，最后`[...,2:]=mean_states[...,2:]`。137必须传入当前CV状态对应ordinary_fusion Mean作0support位置回退，speed/heading仍覆盖原Mean；相同回退规则不等于冻结原OLS位置。返回完整四状态及贡献，保留圆周退化audit。可复用133合法恢复逻辑，不修改133冻结模块。

## 3. 137纯计算模块建议API

新文件候选`association/evaluation/observation_scale.py`：

`split_native_observations(times, latlon) -> dict`

- 有效序列n≥2、递增；0-based内点i%3==1为heldout，首末永保留。返回retained_times/latlon、heldout_times/latlon、原索引、eligible。n=2 eligible=false；不丢原记录。

`heldout_rho_errors(*, times, latlon, rhos) -> dict`

- rhos固定127九项，不扩。对eligible序列每rho调用原`cv_native_positions(retained_times,retained_latlon,heldout_times,rho)`，预测完才用heldout latlon计算RTS径向平方误差均值，返回`mse_m2[9]`及counts。保留FWD预测作同次计算审计，不据其再选rho。无eligible返回有限占位与明确false，聚合不能当0误差。

`select_observation_rho(scene_fragment_errors, *, rhos) -> dict`

- 先每scene合格fragment等权平均，再scene等权；无合格场景不进入数值平均但保留。输出所有criterion、eligible/缺失计数、selected_rho/index、tie_count、selection说明。全无eligible失败。接口不接受reference数组。

最终完整恢复复用`native_cv_state_control.py::predict_scene`或其同一纯核。obs与127reference-rho只按实际唯一rho执行，每次得到FWD/RTS；source_equal按原support恢复，同时重建原Mean/LOW绑定127 V四臂，不借旧group output猜node状态。无新增137LOW候选。

## 4. 独立runner合同

拟新增`tools/analysis/observation_quality_pilot.py`与`tools/analysis/observation_scale_pilot.py`，一个Idea一个run。不混在可自动连跑的循环内。CLI至少native_run/value_run/partition_run/source_risk_run/manifest/output/cpu_threads；137旧127同native_run保存尺度和六臂。

- 136：V十旧臂exact复用＋五新臂，15臂/105对；T只有source-equal加五新臂，6臂/15对，不能伪造T十旧artifact。T32/V32各完整一次；1800s advisory、CPU4/RSS6GiB/AS12GiB，0参考拟合。T诊断不得从结果更改臂/floor。
- 137：V原十臂＋127四个非重复CV旧臂＋四个source-equal新状态臂，18臂/153对；T只有native-equal加四新状态臂，5臂/10对。V两共同历史臂必须exact一致。九候选只T；fit上界`9*N_Teligible+2*(N_T+N_V)`，rho相同dedup；9000s advisory、CPU4/RSS6GiB/AS12GiB。
- 逐场流式，九rho顺序释放，不跨全场收集raw native×rho数组。run.json记录输入路径、真实命令、schema、plan_version、population、policy固定值、phase/完成计数/fit数/全部退出与成本；新目录存在即拒绝覆盖。错误写failed终态和traceback，不重试。
- private保存proxy/观测heldout预测和index/criterion、每臂完整artifact及权重、所有完整population arrays。公网只有脱敏聚合、scene/component及计数，不含source名/identity/coordinates。
- summary必须覆盖all_components/by_component/without_largest_component、完整pair数、主比较bootstrap、0/1/2+source与unknown/floor层、尾部、覆盖/漏目标/duplicate/未绑定reference；null与0区分。报告共享身份和已有V开发属性。

## 5. 有意义验收，不建额外模拟训练链

| 行为 | 必须能观察到的核验 |
|---|---|
| 来源中立/一般K | K1/2/3/4下源重命名与行排列等变；同物理成员不因source字符串排序变权；新增不同源可改变输出，不声称增删源不变 |
| 质量代理与尺度 | 手算等时间及不等时间三点LOO/噪声增益；直线残差0；n2unknown触发equal；floor1/100可产生预期不同权重；不除源内成员数估方差 |
| 盲点反例 | 同样平滑但常值偏移不同的两条轨迹给相同自身残差，proxy不得借参考偷偷挑更准源；只作为反例测试，非MTAD效果证据 |
| 信息边界 | 修改/删除评价文件不能影响136预测或137选择；137heldout位置变化不能改变对应rho的留出预测，但可改变评分/最后selectedrho；检查chart构造仅retained |
| 留出选择 | n2/n3/不规则时间/日期线；单场多fragment人工例区分等scene/等fragment与point加权；精确tie低rho；全无eligible失败 |
| 固定状态/恢复链 | 136状态不改、137只位置改、所有speed/heading exact；支持域及输出population不变；0support仍全部恢复，不删尾部 |
| 真绑定/兼容 | 旧133十臂、127共同两臂/重算CV四臂逐数组exact；错误path/population/节点顺序使run失败并留下可审现场 |
| 成本/终态 | 同rho只一次完整fit、一次返回FWD/RTS；136 T6/V15臂和15/105对、137 T5/V18臂和10/153对完整，不以仍运行summary冒充完成 |

以上合成fixture只验证软件合同。真实终态后需用独立代码从保存数组重算主要指标、完整人口和关键proxy/留出criterion，不能只读production summary认证科学结果。

## 6. 尚未冻结

来源/Mode C已冻结为OD-C1/C2，root已确认实现；最终文件分工与parser由各实现者协调。科学主估计量、固定floor/九rho/mask/人口/资源按1.0执行；正式READY前修订必须更新Idea版本，不把事后更改伪装成原计划。
