# C121 · 训练侧新增位置供给资格

## Record

`EXPERIMENT_RECORD REC-OFFLINE-SUPPLIER-TRAIN-QUALIFICATION-20261113` v1.0，2026-11-13，verified。消费 `IDEA-OFFLINE-SUPPLIER-TRAIN-QUALIFICATION-20261113`、C121/P05/P14/P04、F398–F399、M179。固定C104的T_core `held` 全177场/4组件，C104校准组与本批组件分离，WS→REG_MIX、Q64、组首、参考和有效性共用。只读实际分区改变的7场各一次量测，先冻结全部预测，再读151评价缓存。0原参考、原标签、验证场、封存、训练、模型前向。四臂速度和航向逐值复用 Mean。Campus SSH MCP 科学与worker均退出0。 [冻结Idea](../proposals/fusion_supplier_train_qualification_20261113.md) · [实际终态](../../HandOver/S2L-20261113-c121.md)

|共同有效位置人口|N|Mean MAE / P99 m|保留原供给 MAE / P99 m|C118位置门 MAE / P99 m|原生时间门 MAE / P99 m|
|---|---:|---:|---:|---:|---:|
|177场全部|123,051|1,772.182 / 17,985.623|1,768.605 / 17,985.623|1,768.605 / 17,985.623|1,769.501 / 17,985.623|
|旧新同有支撑，新增皆真同船|288|1,743.388 / 2,566.939|214.962 / 1,629.240|214.962 / 1,629.240|597.858 / 2,322.624|
|只有新增供给|53|3,414.479 / 4,556.384|3,414.479 / 4,556.384|3,414.479 / 4,556.384|3,414.479 / 4,556.384|

2,622个声明目标、2,923个输出组。经纬有效123,056/123,051，速度123,044，圆周航向122,934。四臂全域速度 MAE/P99 35.218/262.704m/s，航向70.764/177.479°，组首、每场逐属性覆盖相同。全域纬/经 MAE 从0.010361/0.011709°降至保留原供给臂0.010337/0.011685°，时间门为0.010343/0.011691°，对应全域P99均持平。位置门与保留原供给臂的公开指标完全相同。本批强局部收益传到全域位置MAE约3.577m，全域长尾未降。 [科学聚合](</Users/a1/Projects/biye-ronghe/offline_observation/docs/evidence/20261113_fusion_supplier_train_qualification/science/summary.json>) · [公开复算](</Users/a1/Projects/biye-ronghe/offline_observation/docs/evidence/20261113_fusion_supplier_train_qualification/public-recomputation.json>)

## Squeeze

`SQUEEZE_REPORT SQ-OFFLINE-SUPPLIER-TRAIN-QUALIFICATION-20261113` v1.0，verified。 [逐点脱敏结果](</Users/a1/Projects/biye-ronghe/offline_observation/docs/evidence/20261113_fusion_supplier_train_qualification/science/points.json>)

- **DE，当前动作空间里有用的新增很少。** 新旧共同支撑的288Q全部真同船、分属3组件。Mean比保留原供给只改善2Q，分别在场698（减少53.892m）与3256（减少1,858.704m）。余286Q全部增害超过250m。使用参考逐点在保留原供给和Mean中择优，局部MAE仅214.962→208.321m，但P99由1,629.240降至933.759m，说明稀有点可支配条件尾部。择优结果仅是这两个动作的事后上界。
- **DE，最近原生点没有把有用的新供给筛出来。** 288Q均满足双侧原生括点，72Q的新观测更近。时间门命中2个有用点中的1个，同时纳入71个增害点，局部MAE/P99均差于保留原供给。全部53个仅新增支撑Q按设计原样保留，输出覆盖保持。晋级条件未通过。
- **DE，外锚可解释源间偏移，尚无证据识别绝对位置收益。** 119Q满足同刻三外锚及留一半径，配准全部缩小新观测与原组首的间距。119Q全部属于直接Mean增害点，两个实际获益点都无合格外锚。1,194次锚使用中13次身份不符，属重复的锚×时刻计数。这一步只评价可用性和方向，没有用配准后状态重新融合，不能推断其精度。
- **AI/UN。** 当前保留原位置＋Mean速航为简单对照。要继续判断稀有救回，先对两处有用点和同场有害点核原生两侧偏差、旧组来源与独立观测。如果还缺独立判据和事件数，维持简单对照，后续优先处理全域尾部和仅新增支撑的53Q。公开参考的不确定性仍是竞争解释。 [近期机制与反例](</Users/a1/Projects/biye-ronghe/offline_observation/docs/surveys/2026-10-27_expanded_search/offline_c121_c122_supplier_identity_sources_20261113.md>)
