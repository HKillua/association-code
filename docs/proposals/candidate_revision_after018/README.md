# 候选补全与早期v2：观测底座、局部关联与属性效用融合

本文记录cooperative v2的补全及早期身份/融合四臂的数学、表示与训练边界。043仍沿用这一补全模型；**044/045证据关联与046效用融合是另行实现的独立先导，不使用本页的身份支路或融合四臂**，方法见[证据关联与效用融合规格](../../specs/evidence-association-fusion.md)。

2026-10-21更新：043已完成，学习补全尚未通过；045/046冻结普通OLS，未接入043学习输出，二者的冻结连接已由[048评价](../repair_before_retraining/048-analysis.md)。本页旧四臂定义不表示022—025恢复执行，也不代表三项贡献已经接通。当前结果见[研究说明](../../research_brief.md)，执行与预算仅看[交接索引](../../../HandOver/Handover.md)。

## 表示版本与当前选项

`model_variant=support_residual_v2`使用观测位置底座、五维输入、支撑查询与局部身份支路。legacy保留旧四维标准化输入及全状态生成映射，供旧权重复算；不能把legacy权重解释成v2。

| 版本或实验 | 速度输出 | 航向输出 | 结构/目标差别 |
| --- | --- | --- | --- |
| 021、031及v2缺省 | `physical_softplus` | `learned_absolute` | 原六层decoder、teacher身份BCE系数1 |
| 032 | `scaled_log` | `learned_absolute` | 031→032只换速度输出坐标 |
| 034 | `scaled_log` | `observation_residual` | 032→034保留观测航向并预测圆周残差 |
| 039 | 同034 | 同034 | 仅`decoder_norm_first=true`，decoder pre-norm |
| 040 | 同034 | 同034 | 仅`decoder_num_layers=2`，encoder仍六层 |
| 041 | 同034 | 同034 | 仅`teacher_identity_weight=0`，物理self/cross监督保留 |
| 043 | 同034 | 同034 | 相对034仅`position_base_policy=native_ols`，未叠加039—041 |

这些是可区分的实验设置，不能统称021已经使用scaled_log或航向底座。非默认decoder语义、速度与航向输出模式随checkpoint保存并核对；同形权重也不能静默换公式。默认旧配置兼容不等于不同模式可以热切换。

## 输入、观测底座与物理输出

共同条件为T_core512/V64、共享A身份隔离、cap128/Q64、真实时间/20秒、显式offset/gap600逐属性mask、全部有效短轨及候选、T_test封存。主干d128、8头、FF256、dropout.1，原encoder/decoder各六层，040只减decoder。数据规则以[数据契约](../../data_contract.md)为准；底座不对输入点间隔套用reference的600秒监督资格。

v2输入投影前为`[z_lat,z_lon,z_logspeed,sin(theta),cos(theta)]`。仅用T_core cap后可见原生速度拟合`mu_log=mean(log1p(v))`与总体标准差`sigma_log`；标准差小于1e-6时取1并记录常量特征。由原装配数组以float64计算log1p后再转FP32，不从旧FP32速度z反解，不截尾或更换单位。

外部packed张量仍为`[N,M+Q,5]`（四状态槽加tau）：第三状态槽为z_logspeed，第四槽仍为航向z；encode用保存的原生航向mean/std还原角度再转sin/cos。只读当前observed native token及query时间；query四状态、padding、reference与身份标签不进入运动编码。V不重新拟合统计。

锚片段沿既有组内首成员规则确定，不按来源或reference误差挑选。默认`hold`位置底座（上表043之前的候选配置）在可见点内按纬度线性插值、经度最短弧插值；位置恰差180°时取−180°方向，支撑外保持最近端点。endpoint留出后重新计算底座与支撑，不能使用被移除的点。

043的`native_ols`改为从当前可见原生点拟合运动斜率，支撑外从真实端点延伸，支撑内保留插值；点数不足时保留确定回退。新策略的经度半周边界按原差符号选弧，旧`hold`不变。它只替换位置底座，速度、航向及下述训练目标不变；模型自身FP32底座与外部FP64普通控制分开比较。接入与数值边界见[本地实施分析](../repair_before_retraining/local-readiness-analysis.md)，学习效果见[043分析](../repair_before_retraining/043-analysis.md)。

位置输出为`lat_hat=clamp(base_lat+state_std_lat*r_lat,-90,90)`、`lon_hat=wrap180(base_lon+state_std_lon*r_lon)`。底座来自FP32 token反归一化，有舍入误差，不保证原CSV逐位复制。纬经投影行零初始化，使初始输出等于此底座；保留std作为残差尺度。

速度两种输出共享`v_typ=expm1(mu_log)`（仅恰为0时初始化取1e-6）及稳定inverse-softplus常数`b_v=v_typ+log(-expm1(-v_typ))`。该回退不改观测或标签；`b_v`与log统计为冻结buffer，速度投影行/可学习bias零初始化。

- `physical_softplus`：`v_hat=softplus(r_speed+b_v)`，直接输出原生单位，取消旧原生速度std乘数。这是021/031的映射。
- `scaled_log`：`v_ref=softplus(b_v)`按原FP32保存，`v_hat=v_ref*exp(sigma_log*r_speed)`，零残差初值与上一模式一致。这是032/034的映射；不设速度上限，非有限输出显式失败。

`learned_absolute`航向为`wrap360(state_mean_heading+state_std_heading*r_heading)`，航向行沿原随机初始化。`observation_residual`改为`wrap360(heading_base+state_std_heading*r_heading)`，该行及bias清零。底座仅用anchor当前可见航向：内插取最短弧、外侧保持端点；半周邻域按原始差符号选弧，容差为`8*finfo(dtype).eps*360°`，FP32约.000343°。详见[航向规格](../../specs/cooperative-heading-residual.md)。

物理损失保持纬经SmoothL1 beta=.001、速度`[log(v_hat+1e-6)-log(v+1e-6)]²`、航向`1-cos((theta_hat-theta)*pi/180)`及既有学习权重；经度周期损失由实际配置决定。log输入统计不替代utility的原生四维normalizer误差尺度。

零物理投影行使第一步对应物理loss只能先更新head，不能经零行更新encoder/decoder；后续head非零时链路恢复。034四行均零，而021/031/032航向行非零；BCE可直接更新共享encoder。这个启动边界不等于已经证明真实失败原因。

## 支撑查询与局部身份分数

query额外使用左/右最近可见点距离、左右存在标记、inside-span五维特征，经无bias的5→128投影加入查询。人工端点留出后重算。手工时间尺度`S=max(1,max(abs(tau)))`仅从T_core有效native及全部query的打包FP32时间拟合一次，各阶段冻结；它不改实际时间、20秒编码或物理支撑。

身份分数为全局`s0`加局部对称残差。局部头260→64→1、GELU、末层零初始化；无可用配对则残差0，仍保留该全局候选。

- 重叠轨只在双方交叠区间的真实token中选近邻：分别按i→j、j→i顺序，以绝对时间差、两侧时间及槽位序排序，各取至多4对。两个方向分别构造特征后平均，交换输入保持对称；最多8对，可重复，不是一对一匹配。
- 非重叠轨使用早段尾/晚段首局部证据；没有交叠内真实点时回退全局分数。全局候选不以硬运动门限删边。
- 每方向特征为配对编码的`mean(abs(e_i-e_j))`与`mean(e_i*e_j)`共256维，另加平均绝对时间差/S、区间重叠标记、有效配对数/4、非重叠端点间隔/S。

身份支路只读native编码和局部证据，不读取物理decoder预测；补全误差对它主要通过共享encoder及混合训练间接影响，不能把baseline的补全→TAM误差传递直接套用。

## 属性辅助残差与效用监督

令`r_i(t)`为query_pool从锚轨编码取得的摘要，`r_j(t)`为辅助摘要，`h_self(t)`为原self decoder隐藏状态。每属性k构造`c_k=sum_j w_jk*(r_j-r_i)/(1+sum_j w_jk)`，再用`h_k=h_self+Wc*c_k`对应的物理投影行输出。位置/航向底座始终属于anchor；helper只沿残差改变输出。无helper时`c_k=0`。

共享`Wc`为128×128无bias、Xavier初始化；不另建四个decoder，也不同时清零Wc与全部物理头。query_pool在v2成为物理模块，进入teacher训练，随后固定。

| 路由臂 | 辅助权重`w_jk` |
| --- | --- |
| plain | 1 |
| scalar-u | `eps+(1-eps)*sigmoid(s')*clip(mean_k(u_jk),0,1)` |
| attribute-u | `eps+(1-eps)*sigmoid(s')*clip(u_jk,0,1)` |
| learned-attribute | `eps+(1-eps)*sigmoid(s')*sigmoid(g_jk)` |

后三臂共享身份概率和正floor，表中`eps=gate_floor=.001`；g与u同输入、同隐藏宽度、同四输出，g由物理任务直接训练。scalar与attribute只差是否分属性；plain→scalar同时改变身份和效用，不能独占归因u。

效用标签来自同一冻结teacher在相同可见输入、query上的self/cross四属性误差，每方向用anchor自己的reference/mask，负辅助不伪造共同目标。误差为物理绝对残差除以T_core原生normalizer对应std，经度/航向取最短弧。

令停止梯度的误差为e0/e1，监督残差`R=(e0-e1)-max(e0,epsilon)*u`，使用`SmoothL1(R,0)`，beta=1、epsilon=.001；每属性按有效监督数归一后四属性平均。负增益参与监督，只在推理路由clip效用；不对小分母直接做比值回归。新teacher生成自己的标签，不复用旧018标签。

## 阶段训练、冻结与比较边界

下表定义已实现训练语义；021原设计为teacher1000、utility200、四臂各300，后续teacher独立实验只运行第一阶段。表格不构成四臂恢复或新增训练预算。

| 阶段 | 更新对象与信号 | 固定对象 |
| --- | --- | --- |
| teacher | input projection、encoder、支撑查询、query_pool、decoder、物理head、Wc、实际使用role embedding、基础/局部身份头及物理loss参数；所有采样pair双self恢复、正pair双向cross物理恢复、所有pair BCE | utility与普通门控 |
| utility | 只训练utility_head；新teacher的self/cross增益监督 | 完整teacher、query_pool、物理loss参数、身份支路 |
| 各臂adapt | 独立decoder、物理head、Wc、实际role embedding和物理loss参数；learned臂另训g；正pair双向物理恢复 | encoder、输入投影、支撑查询、query_pool、身份、utility及独立teacher |

teacher的BCE系数缺省1；041设0仍计算并记录`identity_bce_raw`，实际贡献记于`loss_components.identity_bce`。self/cross、采样、物理分母与全局clip均保留，不能称self-only或身份关联臂。取消BCE同时影响共享梯度及全局clip，见[消融规格](../../specs/cooperative-decoder-ablations.md)。

基础训练为Adam lr1e−4、clip1、batch64、FP32、单seed20，原scene-pair平衡采样与端点留出.25；阶段/各臂独立优化器，teacher阶段内不周期重置。后续utility/adapt沿原seed21/22；四臂共享初态、统计、采样序列及身份分数/阈值，不能串行继承前一臂。

teacher固定端点，不按V选择更新数；原共享流程在teacher之后一次完整V按macro-F1选身份阈值，utility/adapt不再改变身份分支。独立teacher模式只做完整V补全，不进入身份选阈值。各臂固定终点评价，保留全人口、缺监督、共同有效点及来源/支撑/距离分层，预测分组与oracle分组分报。

utility额外预训练与普通门控的学习机会不同，须披露成本，不称严格等总优化预算。旧018→v2同时改变结构、表示与训练量，不能把差值全归某模块；参数容量相同也不自动排除优化差异。

默认六层v2实例曾实计2,190,735参数，比legacy 2,156,814增加33,921（约1.57%）；039不增参数、040裁层后另计，不能沿用此总数。实际实现为`association/{data,models,training,evaluation}/cooperative.py`，入口为`tools/experiments/cooperative_fit.py`。

## 机制来源与证据限制

以下继承已完成文献核读，不因整理重新检索。来源提供候选机制，尚不证明海上迁移有效或首次；当前贡献由具体任务适配与实证决定。

| 来源与已读范围 | 借鉴及适配 |
| --- | --- |
| [GRU-D](https://arxiv.org/pdf/1606.01865)，§2—2.1 | 缺测/距观测时间，改为左右支撑条件化查询，不搬总体均值衰减假设 |
| [CroCo](https://arxiv.org/html/2210.10716v1)，§3 Eq1—2、附录C.2 | self/cross恢复，改为真实时间、连续端点留出和四物理属性 |
| [SUSHI](https://arxiv.org/html/2212.03038)，§4.2/附录A.1 | 全局表征加局部端点/时间证据，不搬视觉外观、层级图或硬排除边 |
| [Gekko](https://arxiv.org/html/2609.01530v1)，§3—4 Eq3/8 | 停止梯度的self/cross误差差值，改为有方向、逐query/属性效用，推理不读reference |
| [MMoE](https://research.google/pubs/modeling-task-relationships-in-multi-task-learning-with-multi-gate-mixture-of-experts/)（官方摘要）、[MTAN](https://arxiv.org/pdf/1803.10704) §3.1—3.2 | 共享表示与任务专属选择，改为可变辅助片段的物理属性路由，不复制固定专家库 |

近邻[Traj-PathFormer](https://github.com/YongshuoLiu/Traj-PathFormer)已有跨源补全、时间查询与局部门控，[PTFM](https://doi.org/10.3390/electronics14173461)已有门控/残差融合，不能将这些算子称为海上首次。已有034/038证据仍显示学习动态不足；合成正确性检查不等于补全、关联或融合贡献成立。历史负结果见[摘要](../../history/README.md)。
