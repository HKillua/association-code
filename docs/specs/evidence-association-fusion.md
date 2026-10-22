# 当前关联与融合实现契约

本页只定义仍在使用的代码行为。Level1最新结果及062决策见[研究状态](../research_status.md)，旧044—051开发结论见[历史摘要](../history/README.md)。配置／可运行入口存在不等于新任务已获批。

## 共同物理输入

候选与其普通控制显式使用`motion_observation_policy=canonical_all`，从全部合法原生观测构造Q64紧凑证据；参考Transformer仍cap128。`selected_cap128`仅作旧行为兼容，权重不能跨政策混用。全部节点、i<j候选、A-B标签、完整目标和逐属性mask遵守[数据约定](../data_contract.md)，无共同支撑或短片段不按难度删除。

[motion_evidence.py](../../association/data/motion_evidence.py)的OLS是普通最小二乘：以真实秒拟合纬度及展开经度的全片段斜率。位置支撑内用相邻观测插值，支撑外从真实首／尾端点沿该斜率延伸，不用拟合截距替换观测；前后共用全片段斜率。速度线性、航向短弧插值，外部保持端点。该规则可以使用查询之后的观测，是离线控制。

观测证据包括支撑、最近时距、精确重合标志、点数／跨度、局部北东米制OLS残差和OLS／首尾运动速度。残差不等于校准协方差，支撑不等于准确观测。source只进入同源关系及评价，不输入9001/9002编号或预定哪源正确。预测接口不接身份、reference和目标mask；这些仅供训练标签／组构造及事后评价。改变reference不能改变模型预测。

按场景流式生成证据，训练缓存紧凑特征而非全部原生数组。标准化只用T_core，验证冻结；Level1纯OLS工作点来自当前固定V，记录为`ols-working-point.json`，不能沿用旧Level2的14.2km或把学习残差分数阈值当米距离。

## 学习关联与终点绑定

[EvidenceAssociation](../../association/models/evidence_association.py)使用
`score = -log1p(RMS_distance_m / 1000) + MLP(features)`。
网络为两层64宽SiLU、无dropout、末层零初始化；base FP64，FP32残差转FP64相加，step0保持OLS排序。分数不是校准概率。

特征交换pair顺序不变，包含几何统计、去均值运动、共同支撑距离与存在标志、运动／速度／圆周航向差、双方支撑／点数／跨度／残差及同源标志；精确列序由模块常量及checkpoint保存。固定变换后T内Welford均值／标准差；不输入绝对经纬度。`geometry_only`保留同形网络但屏蔽非几何列，是特征控制，不宣称有效函数空间完全等同。

056为seed20261021、batch256、正负各半、Adam lr=.001、clip1、1000更新，060只将同seed更新增到3000。scene→pair抽样、成功更新的实际场景／正负／唯一pair／重复曝光需记账，1000更新不等于遍历2100场。固定终点是主结果；selected只作辅助，不用V最好步替换终点。

OLS与学习按相同规则选择阈值：完整V的场景Macro-F1、相同分数整体加入、平局优先更高阈值；允许不同数值。系统入口的`baseline/selected/final`显式绑定对应分数、selection、checkpoint实际步数／配置／normalizer、COMPLETED run和全部V人口。加载权重重算分数核对，FP32跨设备容差不能改变阈值接受边。核实际内容，不生成hash门槛。

固定`anchor_direct`／输入顺序：依次取首未分配节点为anchor，只收纳与它直接过阈值的未分配节点，`>=`含等号，不做传递闭包或Hungarian。所有节点恰出现一次，每组首节点输出；完整目标对应`anchor_identity_first`，重复／未覆盖照报。真值只在外部评价对齐，不参与预测分组。

关联报告至少分开边F1/P/R/FP/FN/AUC、实际分组共属指标，以及目标覆盖／重复／最终误差。固定普通三臂Anchor、Mean supported、Hybrid，与OLS／学习两分组形成六臂；Hybrid复制Anchor位置及Mean速度／航向。默认旧入口只跑Mean的兼容行为不作为新的实验协议。

## 059配对困难负采样

保留原`BalancedScenePairSampler`不变。A/B均先生成同seed20261022的原batch；B再用独立RNG改前64个负槽的抽样来源，不消费原RNG。选场、全部正槽和剩余普通负槽完全相同。

所选T场景N个负对按OLS分数降序、原pair索引升序取前`K=(N+9)//10`；K<64有放回，否则无放回。允许原值重合、与普通槽重复，不强制去重／重抽。必须区分每成功更新64个hard来源槽、实际改变槽数0—64、最终128负槽中的hard-pool命中数（可>64）。训练失败不计曝光；池仅来自T，不按V错误或成绩改比例。主效应是同seed B−A，不能用B−056作采样因果效应。

059已完成而未晋升默认；保留实现便于现结果复核，不授权追加种子或改采样方案。

## 属性效用融合

[EvidenceFusion](../../association/models/evidence_fusion.py)从27维可观测证据给每个helper/query预测位置、速度、航向三个utility。两层64宽SiLU，纬经共享一个位置权重。`p=relu(u)`，helper权重`p/(1+sum(p))`，anchor权重`1/(1+sum(p))`；全部非正时精确回退anchor。经度／航向沿anchor最短弧增量混合，速度凸组合。默认不强锁准确观测，不能把区间内插值当准确量测。

直接损失为最终混合输出对anchor真值的`log1p(error/scale)`，scale为1000m／10原生速度／30°；属性各按有效点平均再平均，batch按anchor等权。不是原始全人口MAE/RMSE优化目标。无效真值不生成标签，不能以NaN乘0掩盖。原utility辅助目标为anchor与helper单独误差之对数差，错误helper也对anchor真值评价。

057三臂网络、连续权重、direct loss、seed、抽样与1000更新一致，最后一层weight零、bias共同为.01，避免Direct-only从全零ReLU起步；该初始化相对旧046是共同改动。

| `utility_objective` | 辅助监督 | 保持项 |
| --- | --- | --- |
| `helper_quality` | anchor质量减helper单独质量的log误差收益 | utility回归＋direct；符号BCE关闭 |
| `direct_only` | 不使用utility辅助项；gain只作诊断 | 仅direct，网络／推理不变 |
| `mixed_action` | helper以固定α=.5实际混合后，相对anchor的log误差收益 | 同样回归＋direct；不搜α、不换离散动作头 |

三个`evidence-fusion-level1*.json`与checkpoint必须按上述枚举绑定，不能只凭目录名称解释错臂。固定单helper动作标签不等于实际多helper联合收益或最优权重；监督改变引起的幅度／保守程度也需解释。

训练每步均匀选场／anchor，batch8，轮换身份构造的`clean`、`clean_plus_wrong`、`wrong_only`。错误helper按冻结OLS距离选，同分按输入序；身份只决定训练正误，不按真值误差挑例。**预测分组只用于最终系统评价，不是三臂训练组来源。** 每100步完整V仅监测，固定final1000，不以最好V步选模；训练路由计数标记为每次更新前、可重复的T样本，不能当终点V路由。

## 普通控制、冻结组合与科研解释

[ordinary_fusion](../../association/evaluation/fusion_controls.py)的Mean supported只平均查询时刻有支撑成员，全部无支撑时回anchor；圆周抵消按既定回退。属性拼接是开发控制，不声称动力学一致或新算法。

062的单一[支撑控制](../../association/evaluation/support_control.py)仅在anchor无支撑且预测组helper有支撑时复制Mean位置，其余保留Anchor，速度／航向取Mean；先对全部预测组输出，再做真值对应。冻结056＋057组合的精确来源／容差／输出见[062交接](../../HandOver/L2S-20261022-062.md)。它不重新训练、打分、分组或搜索阈值。

终点helper启用率分别按实际首代表组、唯一目标使用组统计；所有片段轮流当anchor的内部路由不是同一分母。正权重不保证状态变化，输出相同也不保证全部权重为0。位置／速度／航向、平均／尾部、覆盖／重复分开判断，不事后拼冠军属性。

这不是首次引入MLP关联、门控、误差差或多任务注意力：成熟近邻包括[SUSHI](https://arxiv.org/abs/2212.03038)、[MTAN](https://arxiv.org/abs/1803.10704)和海事[PTFM](https://doi.org/10.3390/electronics14173461)。当前可检验贡献是任务约束下的适配与增量；仅有可运行网络或Oracle空间不足以证明。未来若冻结组合有效，再做与具体主张对应的同容量／特征／固定属性权重控制及稳定性验证，不能用大网络代替归因。
