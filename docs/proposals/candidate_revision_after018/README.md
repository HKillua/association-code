# 当前候选v2：已完成实现与独立任务训练

2026-10-19。模型、训练与共享/单臂入口已完成；026修正托管核实后，按[021—025队列](../../../HandOver/Handover.md)执行，候选合计上限4GPU小时。尚无v2真实MTAD效果。本文固定当前方法和来源，历史动机与负结果见[阶段摘要](../../history/README.md)。

## 1. 明确保留的研究条件

- T_core512/V_select64、共享A隔离、原生A-B身份标签、全部候选和全部有效短轨，T_test封存。
- cap128、Q64、真实时间/20秒、既有reference offset与gap600；所有合法query监督保留，远端外推和速度尾部不删。
- 主干d128、6层、8头、FF256、dropout.1；单seed20、FP32、batch64、原scene-pair平衡采样和首尾留出.25、Adam lr1e−4、clip1。
- 物理目标仍为原生四状态：纬经SmoothL1 beta=.001、速度log差平方、航向圆周cos损失，沿现学习权重；不换速度单位、不新增运动学一致性loss。
- 现有全人口覆盖、预测/oracle分组、单源/简单融合、位置/速度/航向和关联指标全保留；完整V只作开发。新增解释性分层不替换既有主表、不重选子集阈值。

## 2. 统一恢复底座：不重新从全局均值生成全部位置

只从当前可见native token反解原生物理值；不读取query四状态、reference、身份真值或指定源优先级。锚轨沿现分组规则确定。每个query时刻的位置基础值：

1. 位于锚轨可见首尾内：用左右真实可见点线性插值，longitude沿最短圆弧。
2. 位于可见范围外：保持最近端点作为基础值，网络残差仍可随时间预测变化。不是把持有值当真值，也没有限制最终外推为常值。
3. 人工端点留出后重新计算基础值和支撑特征，不能复用遮挡前位置。原生至少两点的合同保持。

`position_out = visible_position_base + train_position_scale * learned_residual`。

纬经残差行零初始化，使初始位置从观测底座出发；纬度合法范围、经度wrap沿现输出语义。此设计可降低重新学习绝对位置的负担，但会继承传感器偏差；不能承诺必然好于简单融合，也不能硬选9001制造零误差。

输入四状态改成五维：标准化lat/lon、训练侧标准化log1p(native speed)、sin(course)、cos(course)。速度统计只由T_core已保留可见原生点拟合一次，与物理normalizer分别保存；不使用V/ref拟合，不改变输入点或标签。航向输出仍原wrap与cos损失，不引入atan2零向量问题。

速度输出改为`softplus(raw_speed + b_v)`，取消原生速度std≈415.27直接乘到输出的路径；b_v由训练侧log1p速度均值对应的典型正速度作稳定初始化，具体数值公式见[表示规格](representation.md)。原生速度和速度loss保持，不能把这项数值适配叫创新。新结构使用独立版本，旧018 checkpoint与默认legacy路径仍可加载复算。

## 3. 三阶段改动固定为一套小结构

| 阶段 | 本版具体内容 | 新增参数设计算术 |
|---|---|---:|
| 补全 | query加入左/右最近可见点距离、两侧存在标记、inside-span，5→128无bias投影；距离按T_core既有固定时间尺度归一，留出后重算 | 640 |
| 关联 | 原全局s0加局部对称残差头；每方向最多4对真实token摘要后对称平均，重叠用共同时间范围内最近时刻对，非重叠用早段尾/晚段首；缺证据回退全局分支 | 16,769 |
| 融合 | 同一self decoder隐藏状态，加每属性辅助摘要残差；一份共享128×128无bias Wc，不建四个decoder | 16,384 |

局部头固定260→64→1：对选中配对token的`abs(e_i-e_j)`及`e_i*e_j`分别均值（256维），再加平均绝对时间差/S、是否时间区间重叠、有效配对数/4、非重叠端点间隔/S四维。GELU，最后层零初始化；无配对直接加0。重叠时从双方都位于交叠区间的真实点构造候选：分别以i→j和j→i顺序，按绝对时间差、当前方向的首时刻、次时刻及槽位序排列各取至多4对；分别得到260维特征，再平均两个方向特征后送入同一残差头。这样即使第4名有时间差并列，交换输入也只是交换两个摘要，保持对称；总计至多8对（可重复）。允许某token重复配对，不把它宣称一一匹配；没有交叠内真实点时保留全局分数。所有全局候选都保留，无硬运动门限删边。

融合沿既有提案：`c_k=sum_j w_jk*(r_j-r_i)/(1+sum_j w_jk)`，`h_k=h_self+Wc*c_k`；每属性用同一物理投影的对应行，位置再加本节锚轨观测底座。无helper时c=0。Wc使用Xavier初始化、无bias，四臂相同；不同时把Wc和全部状态投影都清零。query_pool成为物理模块，进入teacher训练，之后冻结；这一点与018不同。

输入投影4→5维另增128参数，物理输出维数仍4。完整实例实计2,190,735参数（旧2,156,814＋33,921），约增加1.57%，约为完整参考8,720,132的25.12%；已完成CPU实例化独立计数。未使用头仍保留在实例数中，不将其都算作推理必经路径。

## 4. 一次训练里的阶段与对照

更新对象、对照定义与选模见[训练规格](training.md)，实际配置和预算见[执行安排](../repair_before_retraining/README.md)：

| 阶段 | 更新 | 主要更新对象 |
|---|---:|---|
| 共享teacher | 1000 | encoder、支撑query、query_pool、decoder、物理输出、Wc、基础/局部身份头；保留self/正对cross物理监督与所有对BCE |
| 效用学习 | 200 | 只更新utility_head；完整teacher及query_pool冻结，按新teacher生成新的self/cross增益目标 |
| 四臂适配 | 各300 | 同一初始decoder、物理输出和Wc；learned臂另更新同容量普通门控。身份/encoder/query_pool/u冻结，共用一个身份分数和阈值 |

四臂为plain、scalar-u、attribute-u、learned-attribute。后三臂共享身份概率因子和正数floor；scalar与attribute只差是否平均四属性效用，learned使用同输入同容量普通属性门控。plain是普通融合。全部在同一新残差结构上训练，不拿旧018分数当同期对照。

teacher终点选定，不看V挑更新数；一次完整V选身份阈值，四臂各在预定300步终点评价并导出预测。保留原合法reference mask与全部输出范围。先用这一张完整表判断整体与属性路由增量，更多baseline、多种子、去各模块的组合消融后置。

这同时改变了结构、表示与训练量，因此新旧018整体差不能分配给某一项；四臂内部才用于比较路由。1000步不是身份已学好保证；若仍弱，应据实际结果决定，不能自动延长。效用路线有额外200步预训练，须披露，不声称与普通门控总优化预算相等。

## 5. 实现与验证

实现位于`association/models/cooperative.py`、`association/data/cooperative.py`、`association/training/cooperative.py`和`association/evaluation/cooperative.py`，入口为`tools/experiments/cooperative_fit.py`。

已核验遮挡后无泄漏、短轨/无helper/局部证据缺失的回退、交换对称、圆周边界、速度映射、阶段更新与冻结、四臂共享条件、checkpoint及合成原生ZIP端到端。完整2,190,735参数结构CPU前反向有限；本地无CUDA硬件，不根据CPU秒数承诺真实效果或服务器成本。

共享拆分后306项保留CPU检查通过；dropout开启的full与shared＋四独立臂输出/优化器/采样逐值一致，见[当前验证](../repair_before_retraining/verification.json)。旧测试数量和实施流水账已归Git历史，必要检查通过即推进已批准实验。

## 6. 机制来源与海上任务适配

这是跨领域机制的有动机组合与海上适配，不要求每个模块原创；是否有效仍由完整实验与同结构四臂回答。来源定点核验来自已完成research-workflow，不因文档清理重新泛搜。

| 已读来源 | 借鉴与本版具体改动 |
| --- | --- |
| [GRU-D作者稿](https://arxiv.org/pdf/1606.01865)，§2—2.1 | 使用缺测标记/距观测的时间。这里改为离线查询左右真实点距离与存在标记，条件化Transformer查询；不搬医疗变量向总体均值衰减的假设。 |
| [CroCo](https://arxiv.org/html/2210.10716v1)，§3 Eq1—2、附录C.2 | 可见自身＋辅助视图恢复被遮内容，改为真实时间、首尾连续留出和四物理属性；不将生成内容伪装为真实量测。 |
| [SUSHI](https://arxiv.org/html/2212.03038)，§4.2/附录A.1 | 全局表征之外加入最近端点、时间差与运动线索。这里用海上跨源重叠/断段接续的真实局部token，保留全部候选；不搬视觉外观、层级图或硬排除不重叠边。 |
| [Gekko](https://arxiv.org/html/2609.01530v1)，§3—4 Eq3/8 | self/cross恢复误差差值和停止梯度监督，改成有方向、逐query、四属性物理效用；推理不读reference，不用效用代替身份。 |
| [MMoE作者页](https://research.google/pubs/modeling-task-relationships-in-multi-task-learning-with-multi-gate-mixture-of-experts/)（官方摘要）、[MTAN作者稿](https://arxiv.org/pdf/1803.10704) §3.1—3.2 | 共享表示、任务专属选择。这里将任务变为物理属性，信息源变为可变数量辅助片段，以身份和效用路由残差；没有复制固定专家库或多层图像mask，不称精确复现。 |

海上近邻[Traj-PathFormer作者仓库](https://github.com/YongshuoLiu/Traj-PathFormer)已有跨源补全、时间查询、局部门控与物理预测，[PTFM](https://doi.org/10.3390/electronics14173461)已有已关联输入的门控/残差融合。候选差异在异步缺段下身份、实际时刻支撑与属性恢复效用的分工，不能称注意力、门控或残差在海上首次。来源借鉴不是迁移有效性的证据。

新self＋摘要残差结构本身可能有收益，因此四臂全部采用同一新结构；scalar-u与attribute-u回答属性路由，learned-attribute用于排除普通门控/容量解释。新teacher生成自己的效用标签，不复用018旧标签冒充适配。没有加入零floor拒绝、额外风险头或额外完整self/cross双解码；更多模块归因等整体信号后考虑。
