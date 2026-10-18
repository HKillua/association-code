# 完整流程重构：本轮行为规格

2026-10-18，用户“可以的，写代码！”后实施。目标为补全、关联、融合共同训练与评价；本轮代码开发和CPU测试不等于新增GPU实验授权。

## 本轮必须实现

- 原生数据入口复用既有划分和清洗规则，保留每个有效片段的真实时间、观测及身份。长轨抽点显式配置、保留首尾；不先压进全场景64格，不默认删短轨。禁止读取T_test。
- TCM teacher/rollout使用相同的因果encoder与decoder memory可见性，包括增长history的矩形mask。结构、参数量、原损失及旧checkpoint格式兼容；修复后的输出不能冒充历史实验。
- 完整TCM→冻结TCM训练TAM→冻结关联模型训练TFM→联合训练驱动。配置显式指定每阶段更新数、采样、微批和累计预算；保留既有成对采样分布，报告真实场景/轨迹/配对覆盖，不能把短训标成论文完整epoch。
- checkpoint包含三个模块、loss权重、阶段optimizer、采样器/RNG/游标、normalizer和累计成本。恢复使用新目录、接续预算；失败/中断不能标完成。
- 原生参考只供训练监督或开发评价，不作为模型输入。新完整trainer拒绝实际观测时刻与共同网格错位、零原生输入片段、train/V场景或组件交叉。共同查询网格上零支撑的短轨可以保留，必须仍有至少两个真实原生点。
- 完整开发评价保留全目标、单例、漏目标、错误混组、重复输出；同时给预测分组与oracle身份分组、单源观测参考、观测区/缺失区误差和覆盖。不得按真值误差挑最好预测，不改原macro-F1阈值选择。

## 必须公开的边界

用户在解释后已明确批准[真实时间条件适配](tmtaff-time-adaptation.md)。原生时间与下一预测时刻进入TCM，支持内使用同一片段选中观测的物理插值，支持外双向递推，输出共同查询网格；这是公开的方法适配，不冒称作者精确实现。现已连通原生装配、四阶段训练和全目标评价，不把合成端到端通过称为真实训练成功。

TFM FC宽度、位置loss解释、CL margin、更新数都必须显式给出。新的GPU成本尚未测得，不能把旧CE步时当完整TFM成本；没有正式READY训练交接。

## 验证

使用带真实物理时标的微型CPU合成fixture检查四阶段、冻结/解冻、失效输入、累计预算和精确断点恢复；独立参考计算验证因果性与全目标分母。进行相关既有测试回归及独立代码审查。完整结果记录在本页后续交付记录中。

## 本轮交付记录

- 全仓 `python -m unittest discover -s tests`：**304项通过**；包含真实小型CPU三模块推理、四阶段训练及7处中断的精确续训。独立复核发现并修复了target query时标未绑定、完整候选漏检、CUDA别名、None全拒绝和空场景边界。
- 原生入口与TSA适配经过交叉审查；[真实输入检查](../results/refactor_native_input_check_20261018.json)：53个既有开发锚点，2887有效片段/240749候选/5739正对全部保留，零空输入，全部真实首尾保留。cap128明确省去20697/114183点，不能写成保留全部原始点。公共TSA入口另用9个已核T_core锚点验证，其他44个不凭空归入训练划分。
- full_scene_evaluation保留所有目标、全部预测组和重复/漏目标；complete_evaluation串联TCM/TAM/TFM并给预测/oracle/单源/等来源简单融合诊断。先按source取固定首有效片段，再做均值；经度/航向用圆均值，抵消处保留无效，不按reference挑结果。
- 完整trainer采用显式更新制，保留原关联scene→pair平衡采样；TCM使用具备时间转移的轨迹，TFM均匀正场景→正配对。统计实际覆盖，未改成完整论文80 epoch。TAM/joint按原macro-F1选完整快照，TFM保留末步；保存原始末步用于继续训练，不能把“关联最优”称为“融合最优”。
- checkpoint按验证间隔、阶段末和显式中断保存，恢复累计已用预算，FAILED不自动重试。核对直接可解释配置/场景元数据/normalizer，不引入SHA/hash准入、不复制整套原始数据到每份checkpoint。精确续训以相同冻结输入和环境为前提，兼容GPU换卡允许但未实际测试CUDA换卡。
- `build_scene_fusion_targets`按显式目标总体构造参考行，缺参考仍保留无效行；每行query_times必须等于输入物理网格。不会自动猜reference offset或最大gap。

上述304项是第一阶段提交时的测试记录，第二阶段时间适配的最终验证另记如下。两阶段均没有GPU训练、真实原生模型推理、新数据划分或新训练READY；目前没有完整模型性能结果。

## 已批准的时间适配与完整入口

- 最终全仓 `python -m unittest discover -s tests`：**347项通过（19.566秒）**。包含合成原生格式ZIP→显式参考对齐→四阶段真实CPU更新→完整V评价，以及含dropout的原生中断恢复、选定快照重载、输入隔离和预算边界。日志中的两条预期FAILED来自拒绝缺少设备/覆盖已有目录的测试，整套测试退出0。
- `timed_inputs.py`保留有效片段原生输入，物理状态先插值后归一化；`timed_models.py`使用确定性连续时间编码和明确next-time条件，原有可训练参数数量/形状不变。teacher与双向多步rollout因果一致；不能从相同参数量推断相同算力。
- `timed_dataset.py`复用冻结manifest，按原标签保留全部A-B目标（含无有效节点/无参考目标）。每场景offset和max_gap必须显式给出，参考逐场景读取并仅保留监督；未用误差挑时间对齐。`tmtaff_complete_fit.py`提供独立inspect-only与fit，准备时间计入累计预算，无默认设备/配置或新预算。
- [真实时间输入检查](../results/timed_native_input_check_20261018.json)：原53锚点2887轨全部保留，25条零查询支撑、121条单查询支撑仍有原生输入；FP32未改变支撑归属或合并时刻。9个已知T_core锚点通过公共输入API，未为其余44个伪造角色。正式cap/time unit仍须随配置公开。
- checkpoint记录adapter、time unit/cap、原点/网格、参考对齐与采样状态；`load_complete_snapshot`按原adapter恢复选定快照。中断续训接续累计耗时，不把旧参数形状兼容当作旧权重已学会时间条件。
- 验证时临时统一association参数的requires_grad=False，退出（含异常）恢复原阶段标志。本地PyTorch2.10的matmul `should_fold`可按权重requires_grad选择mm或bmm，即使已进入inference_mode；相同权重曾出现2.38e-7阈值漂移。统一计算状态后冻结/解冻验证逐位一致，没有放宽断言、改阈值或改精度；历史回执不重写。

## 成本记录补充：行为与验收

只补充`complete_training.py`已有运行记录，不改变配置、采样、选模、预算、RNG或精度，不启动GPU/真实数据实验。`history[*].seconds`继续只统计训练更新；TAM/joint每次完整验证另记`validation_seconds`，范围含评分、阈值选择、必要的selected快照CPU复制及其指标计算，不含checkpoint写盘。`run.json`保留本次调用的逐次验证计时，失败验证也记录截至异常的耗时；恢复后的历史更新保留原计时，但本次调用记录不拼接旧调用。

最终评价另记总秒数，含selected状态恢复、完整关联/融合评价与阈值一致性检查，不含summary写盘；未开始为null，失败保留已用秒数。成功计时在CUDA工作边界同步，异常路径不为补计时再次同步或覆盖原异常，因此失败秒数可能不含尚未完成的CUDA工作。

`run.json`的CUDA峰值只来自本进程在选定设备的PyTorch allocator：建模前同步并reset一次，记录allocated/reserved字节；包含reset时仍存活的分配/缓存，不是模型独占增量、整卡用量或跨resume累计峰值。CPU及尚未成功reset时为null。成功、显式中断、失败均尽力采集；单项读取失败保留先前可读值并记录错误，不覆盖训练异常或触发重试。

验收仅运行相关CPU测试：用可控时钟区分训练/验证/快照/最终评价范围；检查CPU null、异常计时与错误保留；mock CUDA计数器验证一次reset、设备归属、部分读取失败和恢复调用不继承旧峰值。既有精确续训/选定状态测试继续约束行为不变；不把mock计数器当作GPU实测。

本次相关CPU验证：`test_complete_training.py` 19项、`test_complete_snapshot.py` 3项、`test_timed_pipeline.py` 6项全部通过；`git diff --check`通过。未运行GPU或真实原生数据模型推理。

## 已批准的一次训练侧CL margin校准

用户现已批准固定规则的代码实现，未批准新增GPU实验。`training.margin_policy`缺省或`fixed`保留原数值；显式`train_negative_median_v1`仅用于CL、正项一次距离和仅T_core的训练池。它在TCM结束、TAM首次更新之前执行一次：独立BalancedScenePairSampler seed21，固定64批×64对，保留2048正、2048负呈现及重复，不去重、补抽或读V/T_test。批大小/种子不是新增调参接口。

TCM冻结、eval并复用当前TAM缓存；TAM使用train/dropout，端点去重及分块与TAM_FROZEN相同，整个校准no_grad。保存恢复CPU和所选CUDA的Torch RNG，不触碰其他GPU或正式track/pair/TFM采样器；退出包含异常时恢复原模块模式。负距离用原flatten欧氏距离，对2048值取NumPy中位数（偶数中央两值均值）；非有限/负距离、类别或数量不符、m非正/非有限均失败，不回退m=1。m确定后在TAM及joint保持固定。

记录每批场景/原候选索引、标签及距离呈现、全局和逐批重复、正负分位、严格d<m及ties、分项loss和对distance解析导数。解析导数注明单例loss及64批均值口径，不称为参数梯度；不额外反传。64批前向、首次缓存填充与统计全部计入累计预算，缓存前后计数及本次耗时保留，后续TAM已预热。该样本并非全部训练池，ties不保证50%激活，joint分布也可能变化。

数学中位数以float64保存，原训练仍为FP32。额外记录训练dtype中的margin表示；实际hinge/ties、loss与distance导数按训练dtype的减法计算，数学中位数的严格激活/ties并列记录。相邻FP32值的中央均值可能舍入到端点，不能将数学比较的50%冒称实际激活率。该诊断不改变median规则或训练精度，诊断汇总使用float64，不保证与训练批次浮点归约逐位相同。

请求的ModelConfig与有效margin分开保存。校准完成后、TAM首个更新前写完整安全checkpoint；checkpoint、selected快照与快照加载器均保存/恢复有效m和校准来源。INTERRUPTED恢复若已完成校准则不重做；TCM安全边界尚未校准则接续一次。校准失败/预算到期不保存半完成结果作可恢复成功状态，保留此前完整checkpoint并标FAILED。旧fixed配置和旧checkpoint缺新增字段仍兼容；数据/模型语义不一致不能冒充旧checkpoint续训。

CPU验收覆盖真实tiny模型中位数与训练前向、RNG/模式/参数隔离、非法距离/数量/预算异常、一次校准、阶段边界中断续训、selected有效m重载及旧fixed兼容。合成结果不表示真实数据距离尺度或GPU成本。

本次已运行`tests.test_margin_calibration` 13项，加既有完整训练/快照/时间适配集成28项，共**41项CPU通过（8.339秒）**；`git diff --check`通过。其中native_time_v1使用真实tiny四阶段更新、dropout=.1和完整开发评价，TAM末中断后各模块/optimizer/RNG/loss与连续运行逐值一致。所选CUDA RNG用mock检查设备归属，不是CUDA运行；没有真实原生数据模型推理、GPU训练或新增READY。

## 2026-10-18本地收尾验证

用户已确认训练侧一次CL校准和cap128/time_unit20/已核显式offset/gap600。最终全仓 `python -m unittest discover -s tests -q` **371项CPU测试通过，16.242秒，退出0**；覆盖已校准原生四阶段流程、选定/末状态有效margin、含dropout精确恢复、旧fixed checkpoint兼容、预算/RNG/模式隔离以及连续位置loss。CUDA设备RNG/计数器仅mock，不是GPU实测。CLI无效训练参数及校准p=1/T_core跨配置约束在ZIP读取前拒绝。

普通参考装配效率修复减少按身份重复全scene扫描，数值、mask、报告和异常语义与公开旧入口一致；见融合规格及5项新增测试。root和独立agent复核完成；371项包含13项margin、5项分组装配和1项CLI（含5个错误配置子例）新增测试。最后一轮还覆盖数学中位数与FP32实际margin表示的相邻浮点反例，不修改中位数规则或训练精度。

静态检查实际CLI配置、576个显式对齐条目与既有offset完全一致，41个相关文档链接有效。原生真实输入审计和CPU采样模拟另见[证据](../results/frozen_input_audit_20261018/README.md)。原manifest尚未复制回本地，故没有声称真实完整CLI装配/训练通过；服务器原件路径已知。具体成本测量配置已备好，但GPU测量及正式训练预算均未批准。
