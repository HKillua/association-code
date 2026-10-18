# 018执行与身份关联独立复核

2026-10-18。基于回传原件的CPU只读复算；未加载模型、未推理、未训练、未重选阈值、未使用SHA/hash准入。可重算脚本：[identity_execution_audit.py](identity_execution_audit.py)；机器结果：[identity_execution_audit.json](identity_execution_audit.json)。执行命令为 `.venv/bin/python docs/results/cooperative018_local_review/identity_execution_audit.py`，退出0。

**结论：018确实完成已批600更新与四臂完整开发评价；身份关联仍以大量接受为主，跨源排序较弱，但不能简单说模型“完全没有判别信息”。** 本地用NPZ分数/候选与run中身份重新生成标签，独立计算ROC-AUC和AP；没有以precision接近先验代替排序检验。

## 执行、配置与人口

原件目录：`HandOver/results/COOPERATIVE-DEVELOPMENT-20261018-018/`。

- 实际命令使用独立run018目录的`cooperative_fit.py fit`，原manifest、已批alignment/config、`--device cuda:0 --cpu-threads 4 --save-predictions`；CUDA_VISIBLE_DEVICES指定GPU UUID。`actual_config.json`与仓库已批018配置逐字段一致，run中的model/training一致；execution_extract提取的model/training也一致。没有按提交或hash判定。
- 外层exit_code=0、2371秒，即0.658611 GPU小时；run与progress均COMPLETED、累计600更新。history真实600条，teacher、utility和四臂adapt各严格update1…100；所有数值有限、各run.phases均COMPLETED。阶段末progress的`update=0,total_updates=0`是最终评价阶段计数，不否定600累计更新。
- run原始scene_contract自行聚合：T_core512场景、54792轨；V_select64场景、1635轨、29008对、3297正对、377目标。NPZ各场景候选无重复、合法索引，按身份重建正例数与contract逐场景吻合。没有读原ZIP重新审计。
- 六份checkpoint的服务器CPU提取元数据均为train_absmax_v1，顶层scale与buffer同为有限FP32标量360，更新均100，与run一致。**本地拿到的是提取元数据，未拿到六个.pt；不能说已本地加载权重核验。** 360不是预设准入值。
- 四个scores NPZ每个128数组（64个scores＋64个pairs），四臂逐数组形状/类型/值完全相同；没有用hash代替内容比较。64场景的预测分组也逐列表相同。四份evaluation同一阈值、`selection=null`、`identity_score_source=fixed_records`、`use_utility_identity=false`。
- 四份预测index各列64场景，对应256份本地metadata均存在，场景顺序/ID与run一致、每种方法目标数合计377；每臂预测79组、oracle377组。**完整物理256NPZ仍留服务器，本地此次只能核索引/metadata/逐场景指标人口，不能独立确认它们逐点内容或当前远端文件是否仍存在。** 服务器inventory记载存在是回传证据，不等同本地文件访问。
- test_opened：run、四评价、四索引、256metadata、execution_extract.preparation均明确false；**progress.json并无这个字段**。服务器“run、progress等六处一致”的叙述须收窄，不能把缺失默认为false。现有材料没有发现T_test使用证据。

## 原阈值身份结果

固定使用run已有阈值 **−0.3961639404296875**，判定scores≥threshold。未搜索新阈值。标签来自run.scene_contract的identity与scores NPZ的pairs；没有调用生产指标函数。

| 总体 | 对数/正例 | Precision | Recall | Micro-F1 | ROC-AUC | AP | 正例先验 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 全部 | 29008 / 3297 | .113500 | .978465 | .203405 | **.486286** | **.111611** | .113658 |
| 同源 ITSA | 14609 / 1534 | .105300 | .983051 | .190224 | **.530244** | **.113220** | .105004 |
| 跨源 MSTSA | 14399 / 1763 | .121827 | .974475 | .216577 | **.448768** | **.111008** | .122439 |

全量TP3226、FP25197、FN71、TN514；macro-F1 **.347575252698**，与回传值逐项吻合。AP使用不同分数阈值处的precision加权recall增量，ROC-AUC用并列分数平均秩；只是计算排名指标，不改变执行阈值或重新选模。

全量分数范围[-.609943,.938979]，28972个唯一值；`identity_selection.candidate_count=28973`对应唯一值加全拒绝项。它不是候选对数，不存在35条候选缺失。

**如何解读：** 当前全局得分排序没有优于随机方向的证据，跨源尤其弱；这比“precision≈先验所以完全无判别力”更准确。同源AUC/AP略高于简单参照，64场景平均AUC为 **.535054**，43/64场景AUC>.5；场景平均AP **.276236**、对应场景平均先验 **.219423**。说明部分场景/同源存在弱信号，整体阈值和跨场景得分差异会掩盖局部排序。没有置信区间或独立多种子结果，不能称统计显著，更不能据此宣称关联已学好。

**阈值处precision接近先验仅说明这个操作点保留了几乎全部候选，不能单独证明排序完全随机。** 018接受28423/29008对（97.98%），故高召回同时保留大量错边；这是当前严重过并的直接证据。

## 分组与标签评价边界

四臂同一身份分组：64场景共有79个预测组；56个场景仍只有一个组。1635节点全被分配，未通过丢轨改善分数。可匹配到输出的目标75/377，其余目标作为未匹配人口保留；79组与75覆盖目标差异含重复目标组，不应混称。

逐场景group_diagnostics累计：同身份保留3196对、同身份拆分101对、异身份误并24868对、异身份正确分开843对；68个预测组含误并，19个目标发生拆分。这里的组内关系经过anchor_direct分组，**不能直接与pair分类TP3226/FP25197混用**。

64个场景都含正对，因此negative-scene false-acceptance字段为null。它只是“全负场景误触发率”未覆盖；并不意味着场景过并没有量测——上述groups、merge_pairs和全目标覆盖已经在量测过并。

## 本轮能支持和不能支持什么

- 能支持：一次已批600更新跑通，四臂固定身份比较条件成立，缩放状态在run和回传checkpoint元数据一致，完整V身份数组和预测索引人口齐备。
- 能支持：018当前身份端输出仍大量误并，跨源排序差；改进后数值尺度正常不等于学得有效身份规则。
- 不能支持：015→018全部改善由时间缩放引起（同时增加50倍更新）；“所有场景完全无身份信息”；utility融合效用可替代身份标签；CPU检查或执行COMPLETED等于研究成功。
- 本独审不复算物理逐点误差、损失曲线成因或017训练结果；交给对应分析，避免把局部身份结论扩成全模型病因。无需因此重跑018、重选阈值或增加诊断GPU轮次。
