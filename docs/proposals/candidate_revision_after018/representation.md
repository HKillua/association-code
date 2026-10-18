# 候选修正版：可见观测底座与输入／输出表示规格

2026-10-19。状态：**用户已批准并完成本地实现；未进行真实训练，无新增GPU授权。** 本文件只固定表示和物理输出，与主方案的支撑查询、局部身份残差、属性路由融合合并。既有017与历史018均不修改。依据是当前 `association/cooperative.py`、`cooperative_training.py`、`cooperative_evaluation.py`、`fusion.py` 与 `docs/results/dataset_followup018/README.md`；不新增文献检索或原ZIP扫描。

## 1. 一套最小而完整的推荐

1. **位置：当前锚片段可见点形成插值／端点保持底座，网络只预测度单位残差。**
2. **速度：输入改为 log1p 后训练侧标准化；输出改为直接原生单位 softplus，去掉415.27等原生速度std乘数。** 保留原物理log-speed损失，输出并不冒称“log输出”。
3. **航向：仅输入改为sin/cos；输出仍沿用角度映射、wrap和原cos损失。** 不增加二维输出、atan2奇点或新的方向损失。

以上是共同恢复底座，不宣称每项都是创新，也不把018低分已归因为其中某一项。三阶段新机制的比较臂必须共享这套底座；不能以新全套相对旧018的变化独占归因效用路由。若实现必须拆小提交，先交付位置底座，再接速度／航向输入输出；用户批准的是合并后的具体方案，本地提交顺序不等于新增实验轮次。

## 2. 固定不变的研究契约

原native值、身份标签、T_core/V/T_test边界、cap128、全部有效短轨、完整Q64时刻、20秒时间编码、offset、reference gap600及逐属性监督mask均保持。全部远端query仍预测、仍按原mask监督和评价；不裁时域、不删极值、不把数据改成恒速、不更换速度单位。

`FusionLoss`保持：位置SmoothL1 beta=.001（单位度），速度 `[log(v_hat+1e-6)-log(v+1e-6)]²`，航向 `1-cos((theta_hat-theta)*pi/180)`；原任务权重、有效batch分母、regularizer及teacher/utility定义不在本文修改。特别是utility误差仍用原四维normalizer std归一，新增log输入统计量不能悄悄替代utility的误差尺度。新teacher应从新结构产生其self/cross效用标签，不能复用018标签当已匹配。

## 3. 可见位置底座：完全不读取reference、源身份或遮挡答案

接口保持 `encode(values, observed)`；新增返回当前可见native位置（或足以恢复其位置的两维标准化值）。只取前M个且observed=true的槽；query四状态、padding、被endpoint_view移除的点均不可访问。endpoint_view之后重新计算底座和观测支撑，不缓存遮挡前结果。

对确定的锚片段i（沿现有组内锚选择，不按reference误差或source9001挑锚），可见时间严格递增，至少两个点。每个原query q：

- q恰在可见点上：取该点位置。
- 位于首尾之间：只取包围q的最近两个**仍可见**点。纬度线性插值；经度沿最短圆周差插值后wrap到[-180,180)。相邻经度正好差180°时固定取-180°方向，保证确定性。
- q早于首点或晚于末点：持有相应最近端点位置，不做硬恒速外推。距离和左右存在标志由支撑query模块显式告诉网络；残差可以随q改变，因此不是强制输出静止航迹。
- 输入两点之间无论间隔多长仍按上述底座计算；这是模型底座，**不套reference的600秒监督资格规则**。内部大间隔应由nearest-left/right距离显露，不另丢query或更改标签。

位置输出：

`lat_hat = existing_latitude_bound(base_lat + state_std_lat * r_lat)`

`lon_hat = wrap180(base_lon + state_std_lon * r_lon)`

保留原位置std作为残差单位，取消原位置mean的生成式加法；没有新米／度转换。纬度沿现实现clamp到[-90,90]，需在规格中明示是既有合法范围映射；本轮不新设较窄范围或残差clip。长期外推能力、纬度极限附近梯度、现有经度SmoothL1跨日期线的限制均不靠本改动解决。

`state_projection`纬经两行weight与bias在通用初始化结束后置0。因此初始输出就是观测底座；辅助残差经过同样的状态投影行时也不在初始化破坏底座。无helper只回到同一模型的self路径。加入helper后其信息仍必须通过可学习残差改变输出，不直接按source号复制目标。

如果从现有FP32 normalized token恢复物理坐标，会有FP32舍入；这不等于原始CSV逐位复制，不能沿用先前float64往返的4.61毫米作为模型输出保证。测试对恢复后的可见token检查严格底座等式，对原始float64位置按FP32表示误差检查，不声称全V零误差。

## 4. 速度：只压缩输入动态范围，输出不再乘原生std

### 4.1 训练侧统计与打包

仅对当前正常装配、已在内存的T_core cap后可见native速度拟合：

`ell = log1p(v_native)`；`mu_log = mean(ell)`；`sigma_log = population_std(ell)`。

沿用原normalizer的常量特征规则：sigma_log<1e-6时取1并记录constant标志；这只避免除0，不裁剪输入数值。统计不读reference或V，不按人工endpoint遮挡重新fit。使用已加载数组，不再次打开ZIP或新增全库审计。

候选专用打包分支先从**原始已装配native数组**以float64计算log1p，再标准化并转FP32。不允许先从旧FP32速度z逆变换后log1p：原生v=0可能因逆变换舍入成微小负数，随后clamp会成为隐性数据修改。原生负速度或非有限速度仍显式报错，绝不转绝对值／截断修复。

输入投影为五维：`[z_lat, z_lon, z_logspeed, sin(theta), cos(theta)]`。外部packed接口仍为`[N,M+Q,5]`（四状态槽加tau）：候选helper仅将第三槽换成从原native计算的z_logspeed，第四槽仍为旧航向z；encode用保存的航向mean/std还原theta，再在投影前扩展成sin/cos。原数组／标签不改；query状态仍不参与输入。共享的TMTAFF `pack_timed_scene`不改；配置明确新旧速度通道含义，以免旧checkpoint误读，也不得将第三槽按原生速度mean/std再反变换。

### 4.2 输出及初始化

`v_hat = softplus(r_speed + b_v)`，直接保留原生速度单位；不再计算 `std_speed*r+mean_speed`，也不使用exp(log-v)、速度上限、负值裁剪或按参考分位数截尾。

初始化常数：

- `v_typ = expm1(mu_log)`，只用于构造初值；若恰为0，则初始化用1e-6。此下界仅作用于初始bias，不改任何观测、标签或后续输出。
- `b_v = v_typ + log(-expm1(-v_typ))`，即稳定的inverse-softplus；先在float64算，再明确检查可表示且有限。
- speed投影行weight和其learnable bias均置0，故初始预测为训练侧典型速度。`b_v`是冻结buffer；learnable行bias此后正常训练，名字和角色不能混淆。

`softplus`使用稳定实现，不增加人为速度上限；极端有限raw按softplus的稳定线性支路输出。若raw/预测出现非有限值，显式失败并保留原件，不能nan_to_num或静默截断。原生标签零值由既有loss的1e-6处理；softplus在理论上为正、数值下溢可为0，均兼容该loss。

去掉std乘数限制了该输出层相对r的导数到[0,1]，**不保证整网络梯度有界，也不保证无需训练即可覆盖极大速度尾部**。当前log损失仍会重视相对误差；重尾、任务权重和utility尺度限制保留。

## 5. 航向：圆周输入，保留原输出任务

原生theta按现合法输入范围读取，仅做弧度换算并组成sin/cos；不利用参考航向、速度导数或后一个隐藏点生成输入。padding先清零／mask，不能先对无效值做三角函数再希望mask补救。

输出仍为 `theta_hat = remainder(state_std_heading*r_heading + state_mean_heading,360)`，航向行沿原Xavier初始化，cos loss和角误差指标保持。它不需要增加输出维度或atan2归一化分支；359°与1°的输入相邻，而输出周期本来就由wrap/cos处理。数据cou与前向区间差分的一致性只影响含义解释，本版不将其错改为瞬时物理方向或加运动学loss。

## 6. 第一更新的梯度边界与三阶段接合

纬经、速度投影行零初始化后，第一步对应loss可更新这些投影行，但通过这些零行回传到decoder／encoder的梯度为0；**不应写测试断言第一步每个模块都必须收到每项loss梯度**。航向原随机行与teacher身份BCE仍能向共享表征传梯度。头行更新后，后续步的相应链恢复；不能据此声称第一步物理训练完全停止或编码器冻结。

与已讨论三阶段方案兼容：

- 支撑query仅从当前可见点生成，5→D投影进入self decoder query；插值底座不替代该模块。
- 局部身份残差继续只读真实native token／时间及可得性，不读底座预测或reference。
- 属性辅助路由改变隐藏残差，四属性继续分别进入原四行state_projection；所有臂使用同一位置底座、速度映射、圆周输入与初始化。
- 新增支撑投影、融合投影及局部身份头进入teacher白名单；按本版训练表，adapt只继续更新融合投影，支撑投影与身份头固定。原`stage_parameters`按名字选择，漏加teacher训练或错误解冻会成为确定实现bug。冻结教师及utility的边界不得由新增模块偶然破坏。

## 7. 文件、开销与可执行CPU验收

预期最小变更：

| 文件 | 所需变化 |
|---|---|
| `association/cooperative.py` | representation版本字段；候选打包helper；五维输入；可见位置底座；新速度buffer／输出；按行初始化；checkpoint旧新分支 |
| `association/cooperative_training.py` | 已加载训练native拟合log统计；候选打包；保存统计／版本；新结构参数白名单；teacher与四臂保持相同初始化与统计 |
| `association/cooperative_evaluation.py` | 调用同一候选打包helper；检查保存的表示统计，V不refit；物理输出、mask及指标不变 |
| 现有cooperative模型／训练／评价／CLI测试与规格 | 补以下行为测试；保存旧legacy模式与checkpoint读取测试 |

不需要修改 `FusionLoss`、原生ZIP装配、reference插值、共享TMTAFF模型或017配置。候选表示新增可学习参数只有input_projection从4→5维的**D=128个**；状态输出维仍4，底座及统计无可学习参数。018的2,156,814基准对应2,156,942（仅此表示改动的算术估计）。三阶段支撑query另+640，融合W_c另+16,384；局部身份支路由主方案单独计数，不混称已实测总参数／GPU耗时。

必要CPU验收（已实现测试，结果见implementation.md）：

1. 两点／不规则多点精确命中、支撑内插值、两侧端点保持；经度跨±180°短弧；零残差符合基底，非零残差能修正。
2. endpoint遮挡后删掉的点、query状态和padding即使放入极端值或NaN，也不影响可见输入和底座；未遮挡真实token变化能改变结果。mask和Q人口完全保留。
3. v=0、小正数、常见速度和大有限值的log打包／softplus输出／现FusionLoss有限，无silent clipping；统计仅T_core，V取值变化不改变统计；恒定速度统计分支明确。
4. θ=359/1和θ+360的圆周输入一致性，既有物理输出仍[0,360)且loss等价；未用atan2；非法输入继续报错。
5. 初始纬经输出=底座、速度=典型速度（按FP32容差）；首步head行梯度存在且参数白名单准确，零行首步截断只作预期行为；非零头的后续链可回传。无需真实数据训练。
6. 新checkpoint加载前后输出一致；log统计/b_v/表示版本缺失或不一致显式报错；旧018按legacy接口加载，不能拿旧4维projection直接套新5维表示。
7. 复用合成原生ZIP CLI：完整Q／有效mask／teacher-utility-四臂／固定身份与物理导出流程跑通，所有臂共享训练统计。它只证明实现连通，不等于MTAD效果或新GPU许可。

**本文件与三阶段结构的本地实现已获用户明确批准；训练更新数有完整草案，正式GPU运行与预算另行批准。** 本方案不以“先单独花GPU验证每个表示修补”为前置；CPU检查后可在同一完整候选比较中使用共同底座，待真实信号出现后再追加细分消融。
