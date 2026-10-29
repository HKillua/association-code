# 固定状态的输出位置域控制

只规定本次行为，不发布READY，不修改116/117或原预测。研究证据与局部矩阵见[关联/输出行动矩阵](/Users/a1/Projects/biye-ronghe/docs/surveys/2026-10-27_expanded_search/association_output_action_matrix_20261029.md)。

## 输入与输出

`tools/analysis/output_domain_control.py`只消费原116固定V32名单、114固定A0/WS组、116已保存的A0/WS `NEAREST_SUPPORTED`物理预测，以及117原32的评价报告。原117复用了这些预测，因此FULL必须逐数组exact重放原116完整artifact，并与117对应公开方法报告绑定。禁止重算状态、组、分数、阈值、归一化或模型。

`position_domains(support, groups)`只接收布尔`[N,Q]`输入支撑和完整有序partition，返回三个`[G,Q]`布尔域：FULL全真；UNION为组内支撑并集；SPAN为并集最早至最晚支撑query的闭区间，空支撑组全假。无identity、reference、目标mask、坐标、score或“质量真值”输入。

`construct_outputs`以显式support、groups、已保存group states/validity构造全部六臂。只对纬经位置两列做`original_valid & domain`；speed/heading两列不变，状态数组不变。每组仍输出，不重新选首代表，不静默丢组。输入不被修改，非法shape、非布尔mask、遗漏/重复/越界/乱序节点拒绝。

## 评价及副作用

预测/域锁定后，identity/reference只用于绑定及既有`anchor_identity_first`全人口评价。全部target行、reference有效分母、组成员/顺序/重复组数固定；原首组即使位置域变空，也不由真实身份转用后续重复组。每臂保留缺参考、无预测和失去覆盖的点及目标。

报告各臂自身MAE/MSE/RMSE/P95/P99、全部位置覆盖计数、论文经纬指标、原observed/missing和固定输入支撑层。对每个UNION/SPAN与同R FULL配对，报告共同点及gained/lost/救回/丢失目标、被拒点原FULL风险。共同点状态未变，误差差必须exact为0；缩域不允许新增原不可用位置。speed/heading的all/observed/missing完整指标逐场不变检查。

聚合沿场景/组件保存完整分子分母；分位数由误差向量汇总，不平均场景分位数。逐场还保存speed/heading不变指标；聚合仅位置主表及逐场不变核验。常规分组/原raw绑定只确认原输入没有变化，不产生新的关联成绩。

公共回传只含脱敏数值；原states、truth、identity、membership等留服务器private。原V32终点一次结束，无自动扩量、训练或重试。建议CPU≤4/RSS≤4GiB/AS≤8GiB、600秒advisory；新目录拒绝覆盖，真实异常保存FAILED/已完成场景，最终收尾内存超限不得保留COMPLETED摘要。

## 可核验行为

| 行为 | 聚焦检查 |
|---|---|
| UNION保留洞、SPAN填中间洞、全无支撑保持空域 | 手算两组布尔输入，检查首末区间与空组 |
| 域只影响位置mask；组/状态/速度航向不变 | 既有缓存小fixture通过真实评价API比较arrays/指标 |
| 丢失覆盖的baseline误差留在报告，而非删除分母 | 手算目标点数、lost与rejected风险；共同点差0 |
| FULL复用原artifact并绑定117报告 | 小缓存逐数组重放及损坏source/report拒绝 |
| 固定终点、新目录、失败保全 | 最小runner使用既有小缓存，不调用任何恢复器；模拟后场失败保留先前场 |

小fixture仅软件行为证据，不是MTAD效果。此控制只测声明输出可用域，不把support当真实存在/死亡、不宣去重复；当前Q64已读未来观测，所有结果仍为离线。用户轻量/在线偏好是附加目标，不能据本控制排除有精度价值的离线或较重路线。
