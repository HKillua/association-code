# 038后的三组单因素消融

2026-10-20。用户允许多组消融定位问题；复用[038证据](../proposals/repair_before_retraining/038-analysis.md)和034完整结果。实现、配置与命令已完成；用户2026-10-20明确回复“批准039—041三组”，按本规格原范围发布READY，待035/037释放正常名额。

## 问题与对照

034的学习残差近常量且未胜过观测控制；038在原四场确认86/86轨的六层decoder每层都压低query变化。具体归一化/支路/训练原因未确定。

| 实验 | 相对034唯一改变 | 检验的问题 |
| --- | --- | --- |
| 039 | `model.decoder_norm_first=true`，仅decoder使用pre-norm | 改变主残差路径的归一化顺序是否改善动态与误差 |
| 040 | `model.decoder_num_layers=2`，encoder仍6层、decoder原post-norm | 减少反复变换是否改善；同时降低decoder容量和计算，不能解释为纯深度数学效应 |
| 041 | `training.teacher_identity_weight=0`，原结构/物理目标不变 | 移除身份目标是否改善补全；共享encoder与全局clip两条路径均可能受影响 |

三组均从seed20重新训练teacher1000并评价完整V64、保存全部预测；不接utility/adapt、关联选阈值、融合或T_test。保持T_core512、cap128/Q64/unit20、原mask、25%端点裁剪、self及正对cross物理监督、动态loss权重、Adam/lr/batch/FP32/clip1。沿用034位置/航向观测底座、scaled_log速度和全零物理头初始化。

各组正常完成后，允许一次原四场86片段CPU零更新的`--candidate-dynamics`检查，以和038比较层间变化；使用已完成的新teacher，不另训、不扫描更多场景。每组只读诊断预计数秒，15分钟仅提醒；失败只报告，不自动重试。它是同一实验的辅助读数，不决定是否保留或回传完整V负结果。

不把已训练034权重直接换归一化或截层当作新架构成绩；无额外随机种子、组合臂或自动扩大训练。

## 实现与兼容

- `CooperativeConfig.decoder_norm_first`缺省false，严格布尔；只允许v2启用。只传入decoder layer，不改变encoder、不新增终层归一化或参数。
- `decoder_num_layers`缺省null，即沿用`num_layers`；新实验只允许1至`num_layers`的整数。v2可显式选前若干层，encoder深度不变。
- 为让040不因构造顺序改变其他参数初始化，先按原`num_layers`完成全部模块构造和初始化，再保留decoder前N层。相同seed时保留层、encoder、head及其他模块的初始张量和后续RNG状态与034一致；训练时dropout消耗会随深度变化，不声称逐位同训练流。
- 默认旧配置不新增持久state键；非默认归一化/深度需要保存常量语义标记并在加载时核对，防止同形权重被静默解释为另一forward。snapshot通过原config序列化；旧checkpoint缺新config字段仍可加载。
- `teacher_identity_weight`缺省1，仅teacher物理+BCE组合处乘权重。041仍计算同一BCE，`identity_bce_raw`记录原值，`loss_components.identity_bce`记录实际加权贡献，顶层同时记录系数；默认1时原components语义不变。保留参数集合、正负pair采样、物理分母、loss正则及全局clip；权重0不代表self-only，也不改变后续identity阶段。
- 不新增训练入口、框架或自动调参；使用原`cooperative_fit.py fit --phase teacher --save-predictions`。

## 必要验证

1. 默认/显式原设置与旧权重加载兼容；pre-norm只改变decoder。非零头前向能体现顺序差异，实际训练更新与保存/加载后推理一致。
2. 同seed初始化：039与034参数张量一致；040保留层和其他参数一致，encoder仍6层且优化器不含已移除decoder层；RNG状态一致。语义不匹配的checkpoint拒绝加载。
3. BCE默认行为不变；权重0的teacher仍完成物理self/cross更新、保留原采样/分母；身份专用头不再由BCE更新，原始BCE可读。无隐式改clip或方差权重。
4. 微型原生ZIP端到端检查配置读取、实际更新、teacher端点评价/全部预测和重载；只验证程序行为，不代替MTAD效果。独立复核代码、配置差异及命令。

已完成模型32项、训练20项及CLI9项相关检查；包含三臂真实微型原生ZIP各2步、checkpoint/完整预测重载、旧默认字段省略兼容，以及pre-norm/两层decoder的动态读取精确重构。训练检查还确认第一步物理梯度受零头阻断、第二步可回传，不是永久断梯度。真实MTAD效果尚未运行。

独立复核核准三配置各只变034一个字段、六段命令语法及含空格路径的实参展开；无真实服务或训练启动。另直接加载修改前训练代码与新默认代码，在相同fixture/seed做一次teacher更新，原报告字段、模型和目标参数逐位一致。默认行为未被消融入口改变；三份交接已按用户批准改为READY，命令和实验设置未改变。

## 预算与判读

各一次单卡1000步，034实耗约2.5小时，因此三组合计粗估7—8个单卡小时（040可能更快，不预先折算节省）。各4小时仅提醒，不硬停、不自动重试；包括失败在内各一次尝试。035/037仍占正常名额，待其释放后排队，最多两个单卡，不修改在途目录。

按同人口完整V比较各属性MAE及来源/支撑内外、逐轨/逐场和观测控制；零交集为null、全部难例保留。RMS增大只说明幅度恢复，不能作为成功指标；不由V事后挑最有利的属性宣布整体成功。单seed是开发筛查，最终贡献须进一步验证。

034初始四物理head行全零，第一步物理梯度不能经head回到encoder/decoder，而BCE可更新共享encoder，这是代码可推导的启动效应，尚非训练故障确诊。041同时改变共享梯度及统一clip系数；若有效，只能先归因为“移除身份目标的整体作用”，不能直接宣称梯度冲突已证实。新结果返回后再选择必要组合或进一步机制检查，不预先启动额外轮次。
