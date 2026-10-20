# 035模式核查与原生运动控制

落实[研究路线R1/R2](../research_brief.md#接下来怎么推进)，两项独立，均不训练、不替换默认方法。当前缺少同输入TAM模式比较和基于全部原生点的普通运动控制；新增两个窄入口，复用既有checkpoint、原生时间与物理评价实现。

| 项目 | 输入、行为与输出 | 可观察验收 |
| --- | --- | --- |
| R1：`tools/experiments/tam_mode_check.py` | 035 checkpoint的selected快照、原manifest/ZIP；T_core/V按原顺序各前两场有正负pair的场景，各首4正/4负，不足则保留实际数。TCM eval一次完成并缓存；同一缓存比较TAM eval、train、train全部dropout为0，各8次、seed2035。独立编码相同轨迹的双副本。保存原始距离、正负分项原CL、平衡CL、同轨噪声、跨轨变化与来源快照信息 | selected与末端权重有意不同的tiny checkpoint端到端检查，确认读selected；有dropout时独立副本非零，全关闭含MHA后确定；参数/buffer/模式/dropout/RNG恢复，异常也恢复；不读reference或T_test，optimizer_updates=0 |
| R2：`tools/analysis/native_motion_control.py` | ZIP、原manifest、逐属性时间对齐；只完整V，原Q64/cap128/unit20。三臂native_hold、native_endpoint、native_ols共用支撑内原生短弧插值；支撑外分别保持、首尾速度、全部选中原生点OLS速度，从真实端点外推。纯预测函数只接量测/时间；预测后才用reference评价。保存逐场原生输入/预测/target/mask及整体、来源、支撑、原距离分箱、逐场/轨结果 | 不规则时间、两点短轨、Q无支撑点、日期变更线及纬度边界检查；支撑内三臂一致；tiny ZIP端到端保留空场和全部有效片段；真实V人口/query/target/mask与034核对，hold浮点精度差单独记录 |

R1三模式统一关闭MHA fastpath，统一inference_mode路径；只切TAM encoder，TCM保持eval。同缓存另一次默认fastpath确定性eval参照，报告数值差，不算第四MC臂。保留原CL正对一次幂、原margin，不重校准/选阈值。小批平衡loss不是035全训练loss的同人口复现；三个模式的距离分布与同轨随机性才是该诊断的直接证据。真实selected必须为TAM_FROZEN/600（CLI传`--expected-selected-update 600`），tiny可用其它更新数。无需reference/alignment输入。

R2经度unwrap/wrap使用signed shortest arc，latitude超界裁到合法范围并报告；所有合法短轨保留，不能因Q内不足两点而退回hold。全片段拟合属于离线控制，不能声称在线。无速度/航向预测，不加鲁棒门控或按误差挑窗口。参考身份仅用于评价组织，不能进入纯预测函数。

## R2衔接：运动是否改善关联

用户已批准本地CPU零训练比较。直接复用R2三臂完整V预测；不重做补全、不改变候选方法或服务器预算。主要问题是相同评分规则下，运动带来的位置收益是否变成身份关联收益。

- 三臂均对全部`i<j`候选计算64个查询时刻的球面距离RMS，再取负值为分数；半径6371000米。无支撑交集的pair也保留，既不门控也不按target mask评分。
- 只读取缓存中预测、节点/身份/来源、query及支撑描述；不读取`target`或`target_valid`。评分纯函数只接位置和pair，身份在评分后生成评价标签。
- 主要比较采用既有完整V场景macro-F1选阈值程序，各臂单独选；同时把保持臂的一个数值阈值用于全部三臂，区分排序/表征变化和工作点变化。报告两种阈值下F1、P/R及逐边纠错，辅以pooled/逐场ROC AUC。
- 同源、跨源及共同Q支撑/无共同Q支撑只作描述分层，沿用整体阈值，不另选阈值。保留逐场变化与全部负例，不能以总体位置MAE推断关联必然改善。
- 完整V名单、组件、节点、pair和标签与原位置关联控制核对，旧保持与新保持的浮点差单独披露。64场属于17个相关组件，且V已反复用于开发；不作独立测试或显著性主张。

交付一个窄分析入口、必要正确性检查、可复算原分数及简短结论。若运动不改善关联，也如实保留；这一比较决定R3应采用什么普通几何对照，不自动批准匿名校准、融合或训练。

根目录遵循当前AGENTS：不引入依赖、通用框架、hash门槛、重试或额外训练。源权重和服务器S2L不改写；新输出路径已存在则失败并保留。R1真实权重仅服务器有，使用单独042交接；R2可本地CPU完成，无需等待GPU。完成后做聚焦检查、独立复核及真实R2分析，代码和服务器交接分开提交。
