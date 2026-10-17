# TMTAFF 真实时间条件适配：行为与数学规格

2026-10-18。用户已确认保留所有有效片段的真实观测输入，并批准本页的真实时间输入与 next-query TCM 适配。新增 GPU 实验预算尚未批准；本页授权范围是实现、CPU 验证与后续配置/成本准备。

## 定位与不变项

名称为 **time-adapted approximate TMTAFF（真实时间条件的 TMTAFF 近似复现）**。这不是作者原实现，也不能标为原论文精确复现。新增时间条件改变函数与外推行为，即使没有新增可训练参数，也不等于没有计算增量、没有方法变化或已证明创新。

保留 `AssociationModel` 的输入线性层 `W: R^4 → R^d`、TCM encoder/decoder、TAM、CE/CL 头的全部可训练参数形状与既定损失定义。正式默认架构仍为 6 层、d=128、8 头、FFN=256，查询输出 L=64；小型 CPU 测试可以显式使用小配置，不能据此改正式模型规模。TFM 接收共同查询时刻上的 d 维输出；不新增独立 raw encoder、时间 MLP 或可训练 query embedding。

候选对、A-B 身份标签、划分、主关联 macro-F1 选模/选阈值口径不变。TSA 保留自己的原生观测 adapter，不强制它使用 TMTAFF 的查询网格；机制对照则必须固定 adapter、输入和训练量。

## 输入契约与物理时间

一个场景有共同的原始观测时钟。每个有效片段 i 提供经既有清洗后的实际观测 `(t_ij, x_ij)`，`x=(lat,lon,vel,cou)`，至少两个严格递增时刻。长轨可按显式 cap 选择真实点，保首尾并报告省略量；“保留所有有效片段”不表示每条长轨所有原始采样点都送入网络。原生入口提供 canonical 全点供选点和诊断，训练装配后只保留所选点及诊断，避免在内存重复保存原始数据。新入口不能把短轨先投到 64 格再删成零输入。

物理查询时刻 `Q=(q_1<...<q_L)` 在场景内共同且显式记录。当前 adapter 对非空场景固定使用所有有效 canonical 片段最早/最晚观测之间的 L 点 linspace，场景原点 `o` 固定为最早观测；两者均不从参考轨迹或其有效性推导。空场景须调用方提供独立已声明的查询时刻，不能凭空推断范围。时间尺度 `u=time_unit_seconds>0` 必须为显式有限配置：

`tau(t) = (t-o)/u`。

原生、查询、参考 query_times 使用同一物理时钟。改变 u 会改变模型函数，必须记录，不能在每条轨迹内部按时长缩放或为了词表/长度适配暗改时钟。共同平移全部量测、查询和原点应保持 tau 与输出不变；单独改变 gap/query 时刻应允许改变输出。

float32 打包须保持原生和 query 各自严格增序，也须保持两者之间的支持内/外判定与物理秒数完全一致。若极近的原生端点与 query 因舍入重合并改变支持，明确报 precision-boundary 错误；不能静默把原本未知的 query 标成已知，也不能自行变更精度或时间单位。

打包模型输入为 `[N, M+L, 5]`：前 M 槽为选出的真实观测（四维训练归一化状态加 tau），后 L 槽为共同查询（四维支持内已知量测插值/支持外占位值，加 query tau）。`observed[N,M+L]` 仅真实原生前缀为 true，每条轨迹有效数 `m_i>=2`；原生 padding 与全部 query mask 均为 false。各轨迹真实前缀长度可以不同。

这里 query mask=false 的含义是“不是新增原生观测，不参与 raw-transition teacher loss”，不表示 query 四维一律不可用。模型根据 query 时间是否位于该片段原生首尾支持内决定 known/unknown：支持内的四维是 adapter 从同一 selected 原生观测做物理插值后归一化的已知值，直接进入 embedding；支持外的四维由 pack 写 NaN，模型完全忽略，只使用查询时间外推。不得用同一个 false mask 错误地清零支持内插值。反之，支持外 query 四维占位数值与所有 padding 值不得影响输出。两类 query 值均不能来自 reference。

模型输出固定 `[N,L,d]`，与 Q 一一对应，不返回原生 padding。空场景返回相容空形状；非空有效片段若少于两点、时间非递增、query 非递增、有效值非有限或 mask 不满足 prefix/query 契约，应明确失败，不静默修补。

归一化仅由训练量测拟合，验证冻结。参考真实状态不能参与输入归一化、点选择、查询网格、原点或插值。

## 连续时间编码

令 `PE_k(a)` 为确定性连续 sinusoidal 编码，偶/奇维采用：

`PE_k(a)_{2r}=sin(a * 10000^(-2r/k))`，
`PE_k(a)_{2r+1}=cos(a * 10000^(-2r/k))`。

原生观测 embedding 使用原 W/ReLU：

`e_j = ReLU(W(normalize(x_j) + PE_4(tau(t_j))))`。

TCM decoder 每个 history 槽的输入另外加 `PE_d(tau(next_time_j))`。这是告诉模型当前历史槽所预测的下一时刻，不是把时间作为第五个通道送进原四通道 W，不替换参数形状，也不使用目标状态值。

TAM 输入按共同 Q 排列的 TCM 特征；TFM 对同一 Q 上的两个特征序列融合。输出 latent 不是可直接当作物理位置的四维状态；物理状态仍由 TFM 输出，singleton 使用明确的既定观测透传/缺失策略。

## TCM teacher forcing 与因果性

正向原生序列 `E=(e_1,...,e_m)`：history 为 `H=(e_1,...,e_(m-1))`，next-times 为 `(t_2,...,t_m)`，监督为 `stopgrad(e_2,...,e_m)`。逆向用 `reverse(E)` 和 `reverse(t)` 做同构监督。总 TCM MSE 按两方向有效 transition 的 embedding 元素总数归一化，不对 query、插值点或参考轨迹施加 TCM teacher loss。

teacher encoder 对 H 使用严格的序列因果可见性；decoder 自注意力也因果；decoder history 槽 r 的 cross-attention 只能访问 memory 槽 `j<=r`。只遮 decoder 自注意力不足以避免通过 encoder/cross-attention 读到下一原生目标。

已知 next-time 是允许的外生时间协变量，目标 embedding 的状态值不是。对槽 r，更晚的原生状态不能改变该槽预测。逆向的“因果”是逆向处理顺序；它使用更晚物理时刻的观测向前补全，属于完整场景离线任务，不是实时在线预测。

embedding 监督 detach 保持原约定；TCM-only 阶段的 embedding 冻结方式沿用 trainer 显式设置。不能因为加入时间编码就暗换 loss、margin、精度或学习目标。

## 查询网格内外的补全规则

对于片段选中观测的支持 `[t_1,t_m]`，三部分互斥并覆盖全部 Q：

- `q<t_1`：逆向 TCM endpoint rollout；
- `t_1<=q<=t_m`：仅由同一片段选中的真实观测做物理插值，然后按上面的 W/ReLU/PE_4 在 q 编码；
- `q>t_m`：正向 TCM endpoint rollout。

支持内插值的经纬必须保持成对；lat/vel 线性，lon/cou 沿最短角弧。`|delta|=180°` 时按原始 delta 的符号确定方向，并计数披露，不由参考误差挑方向。经度输出规范化到 `[-180,180)`，course 到 `[0,360)`；规范化前后允许 -180/180 或 0/360 的等价表示。恰等于实际点时恢复该点物理含义；归一化仅在物理插值后应用。支持外不对物理坐标做线性外推，也不向下一片段或 reference 借点。

即使该片段在 Q 上没有任何支持内点，TCM 仍得到其至少两个真实原生观测，并可在两端生成 Q 所需特征；不能因查询网格 miss 而丢片段或将其变成零信息输入。

## 多步 rollout 的 next-time 对应

正向待生成查询为 `a_1<...<a_K` 且全部大于 `t_m`。先对完整原生 `E` 做一次因果 encoder，固定为 memory。第 k 步 history 为：

`H_k=[e_1,...,e_m,z(a_1),...,z(a_(k-1))]`。

decoder history 槽的 next-times 必须恰为：

`[t_2,...,t_m,a_1,...,a_k]`。

decoder 使用 `H_k + PE_d(tau(next-times))`，返回最后一个槽作为 `z(a_k)`。此前 history 槽的查询时间不会因为追加新 query 而重写。所有 decoder 层维持 self-causal mask；矩形 cross mask 仍为 memory index `j<=history index r`，因此生成槽可读全部原生 memory，早期原生槽不可偷看更晚原生状态。generated latent 反馈到 history，但不重新编码或扩充固定原生 memory。

逆向将原生序列和时刻反转，待补的 `q<t_1` 也按降序生成；相同 next-time 规则应用于这个逆序。最终将逆向生成结果还原为共同 Q 的升序输出。不能只 flip latent 而忘记 flip 实际时刻，不能把第一步的 query 重复用于后续所有步。

例：原生时刻 `[2,5,9]`，正向 query `[12,20]`。第一步 history 的 next-times 为 `[5,9,12]`，第二步为 `[5,9,12,20]`。向前 query `[0,-4]` 按生成序排列时，逆向原生 `[9,5,2]` 的第一步 next-times 为 `[5,2,0]`，第二步 `[5,2,0,-4]`；输出仍按 `[-4,0,...]` 排列。

## Trainer / evaluation 接入

原生适配场景需显式标识输入布局；trainer/evaluator 根据声明选择新模型路径，不能凭最后一维是 5 猜测或静默把旧 scene64 当原生。训练 checkpoint/config 必须记录适配版本、time unit、原点/查询规则、选点 cap 和插值规则。可复用参数形状不代表旧 checkpoint 自动具有新语义或已训练过时间条件。

TCM loss 与 association encode/complete 必须用同一套时间输入。TAM/TFM 接收相同 Q 的输出；pair label 与候选始终来自原 frozen population。参考对齐显式使用已声明的 reference offset 和最大插值 gap；每目标的 query_times 必须等于场景 Q，缺参考保留目标行和无效 mask。reference 只进入 TFM 监督与开发评价，不能进入 `complete/encode_tracks`。

完整 V 的阈值仍统一选一次并传入预测分组，保留所有候选与目标。观测/缺失区定义为原始片段支持并集在 Q 上的覆盖，不把新生成点改称真实观测；报告插值支持与真实采样点的区别。单源/简单融合是明确的 oracle 身份诊断，保持同一 Q 与目标总体；它们的观察支持不因模型预测是否存在改变。

## 验收到证据

| 要求 | 可执行验证 |
| --- | --- |
| 无新增 trainable params，核心 shapes 不变 | 对照原 AssociationModel 的 named parameter shapes/count；实际输出为 `[N,L,d]` |
| 短轨全部保留 | 构造在粗 Q 上零支持点的两点短轨；完成/关联仍得到每个 node 的输出与全部候选 |
| 无效 query 状态与 reference 不进预测 | 毒化支持外 query 四维占位、padding、reference 物理值；输出或预测 scores 不变；支持内量测插值变化应改变其 embedding |
| 时间改变可影响外推 | 固定观测值/时刻，仅改外部 query 或 gap；捕获 decoder next-time 或比较实际模型输出 |
| 物理时钟语义正确 | 同时平移 raw/query/origin 保持输出；不规则间隔与不同 origin 明确覆盖 |
| teacher 双向因果 | 改更晚处理顺序的状态，不改更早预测；encoder/self/cross mask 独立验证 |
| 多步 next-query 对应 | 对上述 `[2,5,9]` 示例捕获每步 decoder 条件，检查前后两方向和第二步不覆盖历史条件 |
| 固定 raw memory | 多步 rollout encoder 只处理原生 history；生成 token 仅增长 decoder history |
| 插值角度与边界 | 359/1、179/-179、正负 180 tie、端点、支持外及两点短轨；不由 reference 决定 |
| TCM loss 范围 | 只对 raw transitions；改 query 与 reference 不改 TCM loss，mask/count 不计插值点 |
| 跨层路由正确 | tiny CPU 实际 TCM→TAM→TFM；native trainer/eval 选同模型路径，错时/query/旧布局明确拒绝 |
| 不伪称训练结果 | 只报告 CPU 合成/真实输入诊断；未跑 GPU、未打开 T_test、未声称效果提升 |

## 设计预审与剩余不确定性

按本页可见性与 next-time 规则，数学因果性可闭合；“目标时间已知”与“目标状态已知”必须分别处理。最容易出错的是 decoder 第二步重写前面槽位的 next-time、逆向只翻值不翻时刻、支持内在归一化 heading 上直线插值，以及 trainer 用新输入而 evaluator 仍走旧归一化路径。

原生 teacher 只监督已有片段内部的相邻转移，不能保证对远超训练间隔的 query 外推可靠。确定性 sinusoidal 时间编码也不保证消除长 gap 外推失败。原生序列最多 M 与 rollout 增长 history 会增加 attention/decoder 计算；旧 64 格步时不能直接作为成本。以上是需要后续配置与有界成本测量回答的问题，不通过扩大预算、缩模型或删长 gap 来掩盖。

本适配解决“真实短轨如何进入模型且共同输出具有明确时刻”的输入语义缺口。是否优于原先近似实现、是否优于强 baseline、能否支持课题创新，仍需同协议真实实验；本页不预设必须涨分。

## 本轮独立验证记录

`tests/test_timed_pipeline.py` 的 6 项跨层检查已通过：含真实小型 CPU 四阶段参数更新、最终完整评价、原生零 query-support 短轨、整体时间平移、reference 隔离、双向多步 next-time 独立公式核对与 float32 支持碰撞拒绝。连同 timed model/input 和完整 trainer/evaluator 的针对性回归，共 57 项通过；这不是 GPU 成本或 MTAD 性能结果。

独立预审发现的 float32 raw/query 相对次序缺口已由数据 adapter 增加支持一致性检查，并由跨层反例验证。当前未发现剩余阻断的数学因果性或 trainer/eval dispatch 缺陷；真实数据效果与计算成本仍未知。
