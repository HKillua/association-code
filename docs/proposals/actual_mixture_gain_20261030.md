# 144：实际坐标融合损失驱动的连续集合读出

执行记录：[Record/Squeeze](../reports/2026-10-30-actual-mixture-gain-review.md)、[S2L144](../../HandOver/S2L-20261030-144.md)。本文保留事前规格，不提供重复启动许可。

2026-10-30；IDEA_PACKAGE / research-pipeline-contract v1.0，idea_id `IDEA-ACTUAL-MIXTURE-GAIN-20261030`，plan_version1.0，pilot/partial。来源MC-SELECTION-READOUT-v1.2／Mode C1.59，SR-C2/P11/P14/F241/M109/E140，邻接134 oracle及F240/E139。复用Chroma ICLR2026完整AP关于同目标实际集成损失（原p5、AppD p15–16）的机制、MeanTraj几何/存在域边界、原node恢复代码；不新增论文/精读，不称忠实作者复现。

## A/B/C与待验机制

A为旧Qhard/Qsoft与source equal；140表明合法chord²标签加精确几何求解仍未获实际增量。候选AI：直接学习**实际原节点坐标恢复后的径向误差变化**，可绕开未满足的绝对平方风险/条件几何相容性，但有限表达和有限样本可能仍失败。B借鉴Chroma直接以集合输出损失组织选择；其有历史真实目标，MTAD的A′必须明确使用T参考监督，不能用源自残差冒充。原时序预测与海上物理坐标/无序来源不同，C只支持机制可行不支持本数据必胜；作者核心代码未得，实际接口自己实现。

## 固定方案与可归因对照

- 保持原133 log头及其OOF/fullT评分，不重训评分头。用任意K源的hard质量h与可用源等权u定义连续家族w(alpha)=(1−alpha)h+alpha*u，alpha∈[0,1]；不使用源编号/命名，不限制为两个固定动作，真实数据仍K2。该家族不覆盖整个K单纯形，明确只是集合读出的最小替代。
- T原32场同一合法首代表/多源reference可评query，固定alpha=0,1/8,…,1的9个实际原节点圆周恢复，响应g(alpha)=(radial_error(alpha)−radial_error(0))/1000。仅T形成监督loss曲线；不在V或test做oracle/权重搜索。每query总训练质量1，8个非零alpha均分。没有输出/人口删除。
- source对称context24维：可用源10维特征均值、hard质量加权10维特征、log1p(源数)、log1p(源端点3D弦长RMS/1000)、旧log分数前二差、可用源平均旧log分数。无绝对经纬度、source字符串、identity/role/true purity。折内query均值/std，std<1e−6置1、zclip8，加入常数得到25维z。
- A′强对照GLOBAL：每折仅训练侧9点实际loss总和最小的alpha，exact tie选最小alpha；fullT同样冻结，13次选择，不重训评分。A′+B CONDITIONAL：50系数ridge拟合g≈alpha*a(z)+alpha²*b(z)，归一化训练权，λ1，两个alpha截距系数不罚，其余罚；12LOCO/fullT共13fit。alpha0的gain严格锚0。推理若b>0取clip(−a/(2b),0,1)，否则在0/1选预测gain更低者，exact tie取0。这是连续解析动作；预测gain为负只表示相对基准改善估计，不宣绝对非负风险或Bayes保证。
- 两新臂`WS_GLOBAL_ACTUAL_MIX`和`WS_CONDITIONAL_ACTUAL_MIX`，主要比较GLOBAL−Qhard、CONDITIONAL−GLOBAL、CONDITIONAL−Qhard。相同原节点恢复；0/1源及0支持不变，速度/航向仍Mean。未将143头插入此项，三任务不拼模块冠军。
- T4/V11原控制完整，新T6/V13全比较。全部新预测与模型先存再开V评价；fullT冻结在V前。完整主指标、覆盖/重复、尾部、0/1/multi、场组件/去最大、三主量2000seed20组件描述bootstrap、actual>1km/max和alpha分布。T代理误差/拟合后曲率、实际获益与预测gain分开审查。

## 实现、预算、判断

模块`actual_mixture_gain.py`、共享入口`decision_followup_pilot.py --mode mixture`。核已知曲线最小点、source置换/重命名、任意K质量归一、alpha0还原Qhard/alpha1还原equal、训练组件隔离与先保存后评价。CPU4/RSS6/AS12GiB、1800s advisory，13个50维解析fit与13个全局9点选择；T32×9物理训练曲线计真实成本，0新神经/teacher/原生CV/组/T_test。

只赢Qsoft不赢hard不能称有效增量；条件臂未胜GLOBAL，不能归功context；局部收益/更低代理loss不替代全V与尾部。若整体增量稳定且代价可接受再另立扩大合同；否则保留表达、曲线及信息条件的未知，不追V选alpha/λ/新特征。相似集成方法提供发展空间，当前未验证前不包装新理论或可信同行胜出。下一skill record→squeeze→Mode C。
