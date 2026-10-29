# 跨组留出共享分量与局部重加权控制

本规格对应 `IDEA-SHARED-COMPONENT-CONTROL-20261029` 1.0；输入、固定V32与资源由独立Idea/交接冻结，本文不创建执行任务。它检验相对结构能否改善绝对位置，不认证物理偏差或原Dualβ复现。

## 预测行为

`construct_predictions(states, nearest, support, sources, groups)`只接受有限物理状态[N,Q,4]（纬度/经度/速度/航向）、最近观测秒数、布尔support、source字符串及覆盖全部节点一次的有序partition。它不接受身份、reference、评价mask或完整fragment。

- 两源按字符串排序，定义高source−低source相对差；单源场景允许但只能零校正，超过两源拒绝，不静默选子集。
- 分组rank mod3固定三fold。每个待评组只能使用其它fold的组拟合共享分量；每训练组先平均所有双方有support的跨源pair-query差，再对组均差逐坐标取中位数。chart纬度只由训练fold有support的输入纬度均值确定，N/E米制比例与C10相同；无support组不作拟合贡献，但保留输出。至少2个可拟合训练组才启用校正，否则中心零并报告。source差不是校准概率/协方差。
- 主gauge固定低source、高source减相对中心；敏感性gauge低source加半中心、高source减半中心。平移通过各节点球面切向exp实现，R=6371000m；零向量完整exact保留。相对NE局部近似与球面回映不是精确物理配准定理。
- 不足2个拟合组的fold没有合法训练chart，其留出相对残差标为不可定义，公开定义/未定义组和pair-query分母；零校正仍exact执行，不能把chart零哨兵当真实赤道纬度。
- 对原/校正后物理输入，普通Mean直接调用旧`ordinary_fusion`，不重写数值。Huber臂是δ=1000m、固定5轮的径向距离重加权物理Mean：权重support×min(1,δ/球面距离)，圆周经度按既有恢复器求均值。它是固定迭代估计，不宣精确求得球面Huber最优解。单成员、无support或有效成员权重相同的query复制对应普通Mean；速度/航向始终复制相应普通Mean，仅位置受重加权影响。
- 八臂为WS_MEAN、WS_HUBER、WS_FIXED_MEAN、WS_FIXED_HUBER、WS_SYMMETRIC_MEAN、WS_SYMMETRIC_HUBER、WS_ANCHOR、WS_NEAREST_SUPPORTED。预测全部完成后才能读取身份/reference评价。校正不得改变分组、组顺序、代表、支撑、人口或有效点mask；每组完整输出Q64。

## 证据与副作用

新工具复用原056/114/115绑定检查及全人口评价；原WS_MEAN保存artifact须exact。所有组的校正、重加权预测和拟合中心均保存在服务器private；公开输出只含计数/误差/中心范数，不回传原坐标、身份或源名称。0训练/GPU/新score/partition/阈值/normalizer/ZIP/CSV/T_test。

输入保护及失败：新output不得与immutable输入重叠，必须不存在；schema、数值、人口绑定失败保全run.json并退出，不自动换参数或重试。默认CPU4，Linux AS8GiB与RSS4GiB，600s仅advisory，固定V32终点。当前缓存全窗插值与分组有未来信息，结果只称离线。

## 必要验证

1. 手算已知两源位移，三fold中心仅由训练组确定；改变当前fold不会改变该fold拟合中心。
2. 单源/不足训练组零校正回退保持完整人口；原Mean/Anchor/NEAR exact，所有零中心预测不引入几何漂移。
3. 局部异常的重加权降低其贡献；两点相等权、单点、无support及圆周跨界保留合理行为；原始输入不变。
4. 预测函数不能读取identity/reference，评价阶段在同R下mask/人口/重复保持；非有限值与原115不一致失败，输出不覆盖。

这些是本工具行为检查，不是MTAD效果或校准证明。实际运行后的Record/Squeeze与独立数值核验另行完成。
