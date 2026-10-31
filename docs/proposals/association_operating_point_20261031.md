# 170：同一T-only规则下的关联工作点控制

IDEA_PACKAGE / plan1.0，2026-10-31，源Mode C1.75/C56/F268/E171，继承[168/169完整Record与Squeeze](../reports/2026-10-31-association-context-review.md)。新增前端CTX有共同域均值/召回信号却丢coverage，新raw误报上升，不能直接扩大。AP-TrackTrack给出不同数据集匹配阈值/代价与敏感性，是成熟先例；**AI假设**是换打分后的旧θ工作点可能贡献副作用，尚未证实；竞争解释为错误关系/图合并目标、T过拟合、人口/状态不相容，标定可能仍失败。

`IDEA-ASSOCIATION-OPERATING-POINT-20261031`：0训练/teacher/新输入信息，只读原168冻结T/V raw scores。A原056/旧θ；另外原056、OWN、CTX都按同一T32规则标定，各一个scalar，不能只给CTX额外调参。T32来自原前端已训练T，标定乐观性/共享开发域明确；不称独立校准集，也不以T-only保证新域泛化。V/test不参与选择。

三个scorer各以T32**跨源raw场宏mPFA不高于原056固定θ的T32值**为同一约束，最大化跨源raw场宏mPCA；平局先较低mPFA，再较大θ。候选=该scorer全部T32跨源分数的257个等分位点（含两端）与原θ，去重；不搜索V、不按V coverage/物理误差改阈值。各场必须正类数及接受数非零，不静默排除未定义场；无满足候选即以T_NOT_QUALIFIED结束，不V评价或放宽约束。全网格/选择依据保留。此控制不是以raw约束保证decoded/coverage，更不是标定理论创新。

部署有效score=原score＋旧θ−T拟合θ，等价改变signed WS工作点，沿原完整all-pair WS；无一对一约束、source优先级或更改网络参数。四前端`WS/WSCAL/OWNCAL/CTXCAL × Mean/冻结159NN`，原T32/V32八臂，原A绑168/159，全部真实重新构组/特征/状态；预测落盘后V评价。CTXCAL−OWNCAL/CTXCAL−WSCAL隔离同规则标定后的条件增量，WSCAL−WS标定自身效果；不能与原未标定CTX拼不同人口冠军。

主要结果仍全target coverage获失/重复，raw/decoded跨源mPCA/PFA及all/ITSA完整计数，物理MAE/RMSE/P95/P99/新增>1km、四属性、场景/组件/去最大、2000次seed20组件描述区间。T raw约束通过只证明T选择合法；若V召回/误报/覆盖不能联合改善，不晋级、不沿V追加网格或换规则。总体更好仍需核最坏损害、共同域与新/丢人口；可保条件信号，不把候选解释写为根因。

新170 checkout/science/launcher attempt1，CPU0–3×4/RSS8GiB/GPU0仅冻结159评分、兼容余量6GiB，1800s advisory/Restart=no/Linger=yes，最多四自己单卡含152A/B。完整V完成后另一次audit170 checkout/launcher CPU16–19×4无GPU，沿既有association_context_closeout；它核保存位置/人口/覆盖/尾部/组件/CI及raw/decoded关联，**不独立重选标定**。T标定独立从保存网格核最优和约束，记录浮点差；私有score/coords/identity留server。

实现只新增`tools/analysis/association_operating_point_pilot.py`与T-only选择测试，不改168/169活动/完成冻结目录。沿已验证predict→variable-group评价/finish/独审/属性接口；新选择算子和同rule/拒V输入/工作点代数另核，软件通过后才READY。一次科学故障保全，不自动重试；无新训练/更多T/seed/152改方。
