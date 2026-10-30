# 143：相对代价监督的源对称选源

2026-10-30；IDEA_PACKAGE / research-pipeline-contract v1.0，idea_id `IDEA-RELATIVE-SOURCE-DECISION-20261030`，plan_version1.0，pilot/partial。来源MC-SELECTION-READOUT-v1.2／Mode C1.59，SR-C1/P11/P14/F240/M108/E139，邻接F241/M109/E140。复用139/140正式Record、SR原Chroma/CoCo完整AP及原133代码；不新增精读。主模式问题驱动。

## 问题、机制与差异

米制绝对风险目标139救回158却新增304，说明直接换绝对目标不足。候选AI：同query两个源共同的绝对误差水平可能占拟合容量，但选源只需要相对代价；同时错误次数与错误损害是不同目标。原10维合法信息未变，不声称知道真实噪声或有新增健康通道。将比较改成源对间偏好/损害监督，是否可减少罕见高损害而保持整体？该机制是标准代价敏感成对学习的本任务适配，不宣softplus或换loss本身新颖；信息不足/罕见模式/混组参考仍竞争。

## 固定实现及验证

- 原T32/12组件、V32/9组件；原133可评源对/标签，所有标签与原训练pack绑定。共享同10维、66维二次基；**normalizer直接复用原133每fold**。单query各可评源对总权1；K≥2取所有无序对，source名称/编号不进分数。
- 用d=phi(x_i)−phi(x_j)、t=sign(e_i−e_j)学习共享潜势beta，分数高表示更差，hard取最小。常数维抵消，65个可学系数；目标为归一化权重softplus(−t*d·beta)+||beta||²/2，λ=1。UNWEIGHTED对非零代价对等权；REGRET将同一权乘abs(e_i−e_j)/1000后重新归一化。零真实差对不含排序信息，训练排除数明确报；预测不排除query。不得据V改权/λ。
- 12LOCO＋fullT，两个头共26fit；确定Newton至多30步，梯度无穷范数≤1e−8，回溯最多20步。严格凸有唯一有限解；不收敛/非法数立即一次失败保全。它是有监督迭代凸拟合，不是闭式ridge或神经网络训练。13fold的原场景/normalizer清单保存。
- 两新臂`WS_PAIR_LOGIT_HARD`、`WS_REGRET_LOGIT_HARD`，相同原节点hard恢复、0/1源/零支持回退不变。主要比较REGRET−旧Qhard、REGRET−UNWEIGHTED；完整T4/V11旧控制，T6/V13臂全比较。
- 新模型/fullT、全部分数/贡献/物理状态先存，再开V reference做评价。训练参考明确允许，不称观测免拟合。原神经/teacher/状态/组/单位不改，T_test封存。角色、pure/mixed只做事后评价。
- 全MAE/RMSE/P95/P99、coverage/重复、0/1/multi、场景/组件/去最大；2000次seed20组件描述bootstrap、真实差值strict>1km/max及T误选救回/新增/代价、角色/pure层。少数救回不得掩盖新增错误或强控差距。

## 规格、资源与去留

模块`relative_source_decision.py`、共享入口`decision_followup_pilot.py --mode selection`；source方向翻转/共同label偏移不变性、代价权对解析冲突的影响、loss收敛及拒绝异常、实际list组与原pack字段核验。CPU4/RSS6/AS12GiB，1800s advisory；26×至多30个Newton更新，0GPU/神经/teacher，固定终点无自动加步/seed/扩量。原输入/模型/大预测留服务器，仅脱敏聚合回传。

若只降低训练代理loss或局部误选而完整V/尾部不利，当前配方不晋级；REGRET仅优于UNWEIGHTED但不优于旧Qhard，只记代价权作用。合理监督、相似工作与换域重拟合继续允许；本次不认证真实K≥3、外部泛化或可信同行胜出。新颖性沿既有来源/近邻比较是应用适配待定，实际新增价值由这些主控决定。下一skill record→squeeze→Mode C。
