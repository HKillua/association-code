# 圆周航向TCM：单一表示变体与同预算关联检验

2026-10-20。用户在036分析后明确“好的，立即修正，下一步怎么做”，本项实施已讨论的圆周航向变体。改动不等于效果已验证；原033/035保留为近似参考。新训练预算在代码、配置和命令具体化后确认，不从在途任务借用额度。

## 行为与范围

`ModelConfig.heading_representation`缺省`linear_zscore`，旧配置/权重保持原行为；新选项`circular_sincos`仅允许native-time模型。外部输入仍是四物理状态的原T_core zscore加tau，数据装配、插值、normalizer拟合、mask、短轨、Q64/cap128/unit20、损失、采样和TCM/TAM结构不变。

圆周分支先用保存的训练均值/标准差还原航向：

`theta = radians(z_heading * std_heading + mean_heading)`

内部状态为`[z_lat,z_lon,z_speed,sin(theta),cos(theta)]`。不重新拟合sin/cos统计，不用reference或验证集确定尺度。该变更包含表示宽度/数值尺度/周期性，不称为仅隔离角度跳变。

时间机制固定为`concat(PE4(tau), 0)`，再与上述状态相加并经输入投影/ReLU；不能换成PE5，否则会同时改变时间频率。decoder的PE128保持原样。新增cos维不接时间是控制旧时间投影的设计，不作为新的时间机制。

内部投影4→5列，但先按旧4列完整初始化模型，再在保存/恢复CPU RNG的局部上下文中扩展输入层：前4列逐值复制，仅cos列在旧Xavier界`±sqrt(6/(d_model+4))`内初始化。因此TCM/TAM和之后构建模块的初始参数与随机流保持原seed；实际GPU训练不承诺跨设备逐位相同。新增128个投影参数在TCM及独立TAM阶段均冻结，可训练TCM/TAM参数量不变。

## 一致性与兼容

- 模型统一提供`embed_states(states,times)`供原输入、native目标、reference目标、保持控制和036反馈诊断调用；不保留评价中的硬编码4列变换。
- 圆周模式显式接收原normalizer，并将航向mean/std保存为仅新模式具有的固定buffer。构造、恢复/转移后和评分/诊断前核对该buffer与外部normalizer，禁止静默重解释权重。旧模式不增加state_dict键。
- 旧checkpoint缺新默认字段时，用`ModelConfig(**saved_config)`补默认后比较配置；新旧显式模式不同仍拒绝resume/TCM转移。
- TCM→TAM仍只复制并冻结补齐模块、输入投影及其固定buffer，TAM/optimizer/RNG/预算重新开始；新增航向buffer随源端点保留并验证。旧033可继续按旧模式加载。
- 非native注册模型不接受圆周选项；不悄悄忽略配置。

## 验收与解释

| 要求 | 检查 |
| --- | --- |
| 旧行为不变、角度周期性正确 | 默认旧输出/state/RNG；非零mean/std下0/360等价、359/1相近 |
| 其他初始化与时间不变 | 同seed TCM/TAM、后续随机数与原分支一致；保留PE4及原投影列 |
| 评价同空间、无泄漏 | completion/feedback的目标和hold均走同一模型入口；reference不进预测 |
| 权重和阶段语义正确 | tiny native TCM→独立TAM→snapshot恢复；旧缺字段可用、跨模式或normalizer错配拒绝 |

用户最新要求先只看TCM，037已缩为既有CLI的**新圆周TCM4000及完整V特征诊断**，约2—3h；一次单卡，4h仅提醒，失败不重试。其它设置与033相同，无TAM/TFM/JOINT/T_test或物理融合，取消原串行TAM命令和专用配置。预算仍待确认。旧1000→4000共同四场有真实收益，选择4000用于匹配033训练量，不能推定新版同样受益或4000最优。

037先检查覆盖、训练稳定性及本空间相对hold的一步/自然query/支撑外表现。新旧latent MSE不能直接相减判改进，相对hold比率也不能完全消除几何差异。后续要证明下游收益，仍需相同V人口、相同TAM预算的关联macro-F1/P/R及ITSA/MSTSA；该训练另议，不自动进入。关联改善也不等于物理补齐合格。

若TCM诊断失败，保留负结果，不追加步数、扫尺度或重写指标。下一因素另立单独对照，不把位置坐标、时间编码、free-running loss或短轨权重一起加入本项。

## 本地验证

已用现有CPU环境验证：非零训练均值/尺度下的航向周期性、PE4与其它参数/RNG保持、默认线性模式、无效token隔离；原生ZIP微型TCM→独立TAM→选中快照恢复，冻结补齐权重及原分数一致；旧缺字段checkpoint恢复/转移可用，跨模式与normalizer错配拒绝。评价中的native/reference目标和hold共用模型变换，完整训练及旧CLI相关回归通过。测试中比较快照原分数时使用保存的同一评价batch，避免批次导致的约1e−7浮点差异。

TCM配置经入口配置校验，并与033逐项比较，科学设置只增加航向表示字段；显式advisory与旧任务实际CLI覆盖一致。原串行命令经过bash语法和临时桩核验；当前按用户要求缩为直接托管单个TCM CLI，已检查无TAM调用。尚未执行真实MTAD圆周训练，效果未知。
