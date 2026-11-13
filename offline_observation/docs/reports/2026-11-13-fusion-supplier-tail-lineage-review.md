# C124 · 新供给回退与全域位置尾部溯源

## Record

`EXPERIMENT_RECORD REC-OFFLINE-SUPPLIER-TAIL-LINEAGE-20261113`，verified，消费 Mode C2.48 / C124 / M182 / F404。固定C121的T_core 177场、REG_MIX、首组、151参考和Q64。三臂仅在原供给无支持但新增有支持时允许改变位置，速航逐值沿用原Mean。按已知7个改组场读7件量测，所有预测在评价缓存之前冻结。复现C121保旧位置的123,051个共同有效点、MAE、P99、最大值，并逐值核341个新增点的原误差。按冻结规则选尾部前三场，和原7场合计仅10件发布参考。零原标签、验证、封存、优化和模型前向。 [事前设计](../proposals/fusion_supplier_tail_lineage_20261113.md) · [终态](../../HandOver/S2L-20261113-c124.md)

|位置人口|N|KEEP MAE/P99 m|最近新片段 MAE/P99 m|原片段最近原生端点 MAE/P99 m|
|---|---:|---:|---:|---:|
|全域共同点|123,051|1,768.605/17,985.623|1,768.605/17,985.623|1,768.173/17,985.623|
|仅新增供给|53|3,414.479/4,556.384|3,414.479/4,556.384|2,411.308/9,561.621|

53个仅新增点均只有一个实际新供给，最近新片段与KEEP逐点一致。旧端点改善38点，但14点增加超过250m，最大误差9,810.484m。三个活动组件MAE均降，最大活动组件的P99翻倍。所有臂保持相同的有效人口，联合门未通过。53个仅新增点均未达到原全域P99，全部存在新供给原生内插，零发布参考局部超额。23点组外已有同船活跃片段，均来自场2377；这个诊断使用真实身份。 [公开聚合](</Users/a1/Projects/biye-ronghe/offline_observation/docs/evidence/20261113_fusion_supplier_tail_lineage/science/summary.json>)

## Squeeze

`SQUEEZE_REPORT SQ-OFFLINE-SUPPLIER-TAIL-LINEAGE-20261113`，verified。 [全域尾部明细](</Users/a1/Projects/biye-ronghe/offline_observation/docs/evidence/20261113_fusion_supplier_tail_lineage/science/tail-points.json>) · [发布参考核点](</Users/a1/Projects/biye-ronghe/offline_observation/docs/evidence/20261113_fusion_supplier_tail_lineage/science/tail-reference-points.json>) · [18,406项公开复算](</Users/a1/Projects/biye-ronghe/offline_observation/docs/evidence/20261113_fusion_supplier_tail_lineage/public-recomputation.json>)

- **DE，尾部更多是供给没有进入首组。** 1,231个达到原P99的点分布84场，场4064/2794/1929各81/69/66点，合计216。791个首组无活跃片段，782个同时在组外有活跃同船片段。902个尾部点存在组外同船活跃供给，87个首组含活跃异船。以固定参考事后从全场同船活跃片段中选择最小误差，870点有某个候选低于原P99，352点即使任选也仍在尾部，9点全场没有同船活跃片段。该量是可触及空间，不能作为前端预测成绩。零尾部点属于本轮新增供给的341点，三臂全域P99实测持平。
- **DE，两个收益点都不是新源单独更准。** 场698/Q46的旧、新源单独误差892.281/2,273.968m，Mean838.389m；发布参考插值括点114秒，割线18.352m/s，比两端最大SOG多11.973m/s，涉及反向回跳。场3256/Q35的旧、新源2,012.289/2,318.390m，Mean153.585m；发布参考214秒括点，超额4.952m/s，未过10m/s门。两侧缓存位置均可从相应短原生括点重现。两例说明在当前参考下混合可抵消位置偏差，真实位置仍缺独立证据。 [原生括点与误差](</Users/a1/Projects/biye-ronghe/offline_observation/docs/evidence/20261113_fusion_supplier_tail_lineage/science/new-supplier-points.json>)
- **DE/UN，参考回跳只解释已查到的部分形态。** 在冻结10场的305个尾部点中，304个为内插、1个恰为发布原时刻。28个所在发布区间的割线比两端字段最大值高逾10m/s，其中20个触及成对反向回跳。该抽样由尾部大场和既定改组场组成，不能外推84场占比。原参考位置复现建立血缘，独立物理准确性仍未确定。 [尾部和独立位置来源](</Users/a1/Projects/biye-ronghe/offline_observation/docs/surveys/2026-10-27_expanded_search/offline_c124_position_tail_sources_20261113.md>)
- **AI。** 保留KEEP工作对照。优先在训练侧用事前可得的时标和轨迹关联证据检查那782个首组无活跃、组外有同船的点，设等覆盖的候选回收对照。必须分开对真正同船的回收和异船误用，同时在非尾部核增害。对352个有同船活跃仍尾部的点，下一步定位原生状态、插值、配准和发布参考中的误差来源。
