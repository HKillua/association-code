# C88 候选外平移校正与实际分区的留场结果

## EXPERIMENT_RECORD

`REC-OFFLINE-EXTERNAL-SUPPLIER-TRANSLATION-20261112`，research-pipeline-contract v1.0，2026-11-12，completed。上游 `IDEA-OFFLINE-EXTERNAL-SUPPLIER-TRANSLATION-20261112` plan1.0、Mode C2.28、F332/F334/F335。主比较在相同原生树候选与ABS评分上，检验同锚平移是否比原距离多识别错误供给，并实际减少最终分区的误并、漏关联且保住同船逐时支持。[冻结Idea](../proposals/offline_external_supplier_translation_20261112.md) · [执行合同](../../HandOver/L2S-20261112-c88.md) · [终态](../../HandOver/S2L-20261112-c88.md)

C83 V48按原manifest排除与C87 T32/T18共享A组件相交的场，保留全部16场/12组件，283片段、112个原WS组、75,659个原生事件、2,628完整配对及14,299个冻结候选。原056评分、原WS和C83 Q逐项匹配。全部V16特征和候选先落盘，再读T32身份校准，并冻结全部V16分区后才读V16身份。原056曾用完整V300选模和阈值，留场只适用于本供给门。0新模型训练、参考与封存测试。T32的共同可用外锚849对中603真，对应30对5%拒真预算。原始距离和外锚平移的距离上限分别为4,205.475m和420.168m，均用严格超过才拒绝。科学与worker实际退出0，4.217/7.042秒。 [公共汇总](/Users/a1/Projects/biye-ronghe/docs/evidence/20261112_external_supplier_translation/science/summary.json) · [逐场](/Users/a1/Projects/biye-ronghe/docs/evidence/20261112_external_supplier_translation/science/scenes.json) · [冻结阈值](/Users/a1/Projects/biye-ronghe/docs/evidence/20261112_external_supplier_translation/science/calibration.json)

|真实输出分区|采纳编辑场|同源FN / FP|跨源FN / FP|原同船支持损失 / 恢复|原纯他船支持释放|重复代表组|
|---|---:|---:|---:|---:|---:|---:|
|WS|0|20 / 16|43 / 36|0 / 0|0|12|
|ABS无门|3|20 / 16|49 / 37|40 / 0|43|15|
|J_GATE|0|20 / 16|43 / 36|0 / 0|0|12|
|RAW_J|0|20 / 16|43 / 36|0 / 0|0|12|
|TRAN_J|0|20 / 16|43 / 36|0 / 0|0|12|
|TRAN_KEEP|0|20 / 16|43 / 36|0 / 0|0|12|

FN为漏掉的真实关联，FP为错误共组关系。支持按节点×Q去重，原支持来自至少一个同船成员时计入同船覆盖。ABS只释放场203的43个纯他船槽，同时在全人口引入46个新的纯他船槽。原同船支持12,971槽，ABS变为12,931，另外四门保留原值。原代表身份100个在全部实际输出均保留。作者同公式的跨源预测数加权正确率/误关联率，WS及四门为80.27%/18.85%，无门ABS为77.47%/19.89%。不同于作者独立测试人口。

## SQUEEZE_REPORT

`SQ-OFFLINE-EXTERNAL-SUPPLIER-TRANSLATION-20261112`，输入 `REC-OFFLINE-EXTERNAL-SUPPLIER-TRANSLATION-20261112`，verified。

1. **外锚可用性在V16显著缩窄。** 187个跨源原供给对中，146对有候选双方的直接共时观测。仅11对有至少3个候选外互选锚组，全部来自场1328。要求至少三个时刻同时有三锚后剩6对，六对全部通过非共线条件，也全为同船关系。RAW与TRAN均拒0对，没有真假共存的场，AUC不可计算。覆盖瓶颈首先来自锚对应数量，其次是共同时间，此人口没有观测到可区分的假供给。较C87的两个混合场信号，当前方案未取得留场辨识资格。 [冻结锚覆盖漏斗](/Users/a1/Projects/biye-ronghe/docs/evidence/20261112_external_supplier_translation/anchor-availability-squeeze.json)
2. **无锚回退改门，没改实际分区。** 外锚不可用的181个跨源供给对含146真、35假，原056回退拒10真、1假，暴露356/4个供给对×Q，表明该固定门在本人口不能可靠释放他船槽。TRAN_KEEP相对TRAN_J不再拒其中这些对。两关系合并共21个拒绝变化，只有场3331的37个候选因此改变允许性，全部实际分区仍为原WS。RAW与TRAN候选允许掩码逐场完全一致。剔除最大组件后结论一致。
3. **评分也是独立瓶颈。** 14,299个候选中，事后有72个同时净补真、净去假且保护原代表身份，但冻结ABS收益为正的有0个，连无门臂也无法通过正分接受。无门实际采纳的三个编辑各减少2条真跨源边，其中场203还增加1条假跨源边。仅改原时刻供给门不足以解决这个分母上的排序。保留外锚平移为少量有观察条件下的诊断证据，不升级当前接受或回退规则。下一步先在冻结候选上区分可达性、排序和接受，使用任何降低锚要求的平移版本前都需独立确认假供给可见性和不损害真边。

4. **候选逐时支持和场内排名的事后复核。** SSH MCP只读已冻结候选、量测支持和同一151缓存身份，逐一物化这72项，未重做预测或决策。68项保住每个原同船支持槽，42项还同时严格降低跨源FN和FP，集中在场929的41项和场3272的1项。四个门全部挡住这68项，ABS也没有给出正分。场929最高分的编辑本身合格，收益−0.039875，输给不改分区的0分。场3272的合格编辑得−0.432470，另有6个编辑排在其前，首选−0.055296。事后现象分别对应编辑接受和编辑间排序两种失效。两场证据用于选后续问题，门和排序都不在已看的V16重新调。 [支持复核](/Users/a1/Projects/biye-ronghe/docs/evidence/20261112_external_supplier_translation/candidate-support-squeeze.json) · [排名复核](/Users/a1/Projects/biye-ronghe/docs/evidence/20261112_external_supplier_translation/candidate-rank-squeeze.json)
