# 离线论文对照资格与原协议适配

2026-11-10 的离线证据摘录。公开聚合核对不涉及重新训练、推理或封存测试读取。当前目标与执行由本路线研究状态和交接维护。 [冻结审查全文](https://github.com/HKillua/association-code/blob/e10a176f2f1a81486e37464feae7ace6d8ffe949/docs/reports/2026-11-10-offline-online-route-review.md)

## 现有离线数值：有优势，也有实质短板

**DE—既有聚合初核，非152完整审结。** 2026-11-10北京时间17:27:42，通过SSH MCP读取原147与152A/B的公开run.json/summary.json。准确来源、字段、单位、预算及未做事项见[152聚合初核记录](/Users/a1/Projects/biye-ronghe/offline_observation/docs/evidence/20261110_route_audit/152-aggregate-observation.json)。

三份summary均为同WS全V300/38组件，场景顺序及partition_run路径一致；152引入的147位置字段与原147逐项一致。位置有效点441144/454952（96.964955%）；9779输出组、9100目标、1068重复组、389漏目标的聚合人口一致。相同计数和路径不替代原数组与真实成员逐项审查。

|现有组合|位置MAE m|RMSE m|P95 m|P99 m|速度MAE m/s|航向MAE °|
|---|---:|---:|---:|---:|---:|---:|
|原147 Qhard|590.760|3303.235|2606.479|8426.138|25.066|75.652|
|152A PTFM＋Mean扩域|738.682|2866.096|2890.872|8375.947|11.872|60.649|
|152B DeepAF任务适配|1361.695|8317.899|2922.260|9143.200|25.066|75.652|

- 原147的位置MAE/P95优于这两个适配；PTFM的RMSE/P99/速航更好。当前证据表明多指标取舍，既不能概括为所有指标都差，也没有证明全面超过可信同行。
- PTFM核心域仅265135位置点、58.277577%覆盖，MAE561.948m不能与全人口590.760m直接排名。上表使用统一覆盖的PTFM＋Mean；完整审查还需核共同核心域、扩域、异质性和原始数组。
- 原147和DeepAF的输出速航沿用Mean；相同数值不是各自网络在这些属性上获得了同等学习效果。单位按152的统一m/s、degree接口解释。
- PTFM完成150epoch/427800更新，DeepAF完成20epoch/80800更新，均单seed20和固定末轮。预算与原论文的差异已在[152合同](../../HandOver/L2S-20261030-152.md)声明，固定末轮不证明充分收敛。
- 这张表使用原147作对照，不能把V32的178/191分数或后续157拼进去。V被反复开发使用，组件重采样也是描述性不确定性；不是封存测试的打榜成绩。

**UN：** 152规定的完整命令/日志、权重、逐场/保存数组、全差异/尾部及Record→Squeeze→Mode C尚未闭环。它们仍是共同WS前端下的项目融合器适配，不是作者完整系统精确复现。

## 离线对照应覆盖哪些层次

|层次/候选|原文任务与可比较部分|当前缺口或应保留条件|
|---|---|---|
|完整系统：TMTAFF，Applied Sciences 2026|原刊明确离线，包含补全、关联和融合；最贴近完整链的任务参照。[原刊摘要/讨论](https://www.mdpi.com/2076-3417/16/14/7251)|本项目适配曾出现常值/尺度失效，不能把失效实现当被击败的强baseline；作者场景列表、处理细节和实现仍有未知。[已有全文复核](/Users/a1/Projects/biye-ronghe/docs/surveys/2026-10-26_mtad_actual_use/tmtaff_reaudit_20261027.md)|
|关联：HTG-TA、Temporal-MPRF|异步轨迹关联强候选，比较关联指标；不把关联论文的“fusion”误读为状态融合。[HTG-TA原刊](https://ieeexplore.ieee.org/document/10557703/)、[Temporal-MPRF DOI](https://doi.org/10.1049/rsn2.70230)|需统一场景、时间窗、属性、校准和标签权限。Temporal-MPRF原主基准有标签辅助去偏，不能无声进入部署预处理；本轮其DOI在线读取失败，具体条件复用[已存正式全文分析](/Users/a1/Projects/biye-ronghe/docs/surveys/2026-10-26_mtad_actual_use/temporal_mprf_reaudit_20261027.md)|
|融合：PTFM，Electronics 2025|原文为已关联双源轨迹融合；§3.1.1使用超过18000组轨迹，不能直接用论文数值对本项目全场Q64排名。[原刊](https://www.mdpi.com/2079-9292/14/17/3461)|152A提供共同任务适配的初步底账；原核心/扩域和训练差异须分别报告。共同前端可用于归因融合增量，整链胜负仍需完整前端对照|
|点迹关联/滤波：DeepAF，Aerospace 2025|历史序列＋当前量测差分集合估计P/V，本身包含递推跟踪机制。[原刊§5](https://www.mdpi.com/2226-4310/12/3/194)|原任务是杂波中的点迹级多目标跟踪，152B是轨迹组恢复适配；不能由其当前慢/差否定原DeepAF在线能力，也不能把它当唯一强对手|

常规运动/滤波、关联分配及融合控制仍应保留作为可解释下界；是否成为强对照取决于合理条件/调参后的真实效果。作者实现优先，缺作者代码不自动排除方法，但应明确复实现有证据与适配差异。已有[23篇MTAD应用清单](/Users/a1/Projects/biye-ronghe/docs/surveys/2026-10-26_mtad_actual_use/README.md)可供分层筛选，篇数不是可直接放入同一排行榜的数量。

## 同日补核：能否接近原文，以及把我们的方法放入作者任务

本节为原文/源码及既有聚合补核，不是新实验、已验证改进或152完整审结。结论是：**可以建立作者任务下的比较；目前尚不能证明已达到或接近原文，也不能保证改协议后会达到。** 训练完成、量纲一致或数字看起来相近，都不等于复现成功。

### 已有同域配对结果，不用不同覆盖的分数相减

通过SSH MCP补读152A原summary的 `all_components.pairwise.WS_PTFM_CORE_vs_WS_SHARED_Q_HARD.common_valid_position`，得到以下既有结果。两臂同时以265135个共同有效位置点评价，占全目标有效点58.277577%；这是保存的直接配对结果，超过“两个计数恰好相同”的证据强度，但未重新打开私有数组独立复算。

|同域方法|位置MAE m|RMSE m|P95 m|P99 m|
|---|---:|---:|---:|---:|
|原147 Qhard|315.829|2434.336|498.559|3119.914|
|152A PTFM core|561.948|1199.516|1832.936|3889.653|

两臂保相同目标/真值/真值mask，再取有效预测交集，源码见[common_target_metrics](../../tools/offline/system_evaluate.py#L243)、[直接配对入口](../../tools/offline/peer_fit.py#L157)及[compare_pair](../../tools/analysis/static_partition_position_control.py:116)。重构前的补读核查确认当时7份相关源码与152A冻结执行版本一致。原字段、补读边界见[初核记录的common_core_followup](/Users/a1/Projects/biye-ronghe/offline_observation/docs/evidence/20261110_route_audit/152-aggregate-observation.json)。补读未记秒级时间，不能沿用初核17:27:42。

该表支持“原147在项目共同支持点的MAE/P95/P99更低，PTFM core的RMSE更低”。**不能据315.829低于原文354.2就宣称超越PTFM论文**：共同评价点不等于作者的场景/已知关联/时间窗、训练人口或上下文条件；当前K2的共同域也不能直接推广到K≥3。

### 原文数值与可消除的协议差异

PTFM原Table1位置MAE为0.3542km、RMSE为2.6180km；原文训练为最多150轮，验证loss连续10轮不改善时降LR，20轮不改善早停并恢复最佳权重。[PTFM原刊§3.1.2与Table1](https://www.mdpi.com/2079-9292/14/17/3461)。152则固定150轮、固定LR和末轮模型，合同已明确属于任务适配；不能把“跑满150轮”解释成与作者训练一致。

|需要对齐的条件|对我们方法的可落地适配|尚未完成或限制|
|---|---|---|
|PTFM：已正确关联的双源轨迹|给双方同一双源配对清单，暂不让关联错误混入融合比较；完整预测关联系统另表|真实分组可作为该融合任务给双方的已知输入，不进入部署关联输入|
|PTFM：共同覆盖内的时间对齐与窗口|按作者描述在A时刻邻域平均B点，reference插值到同一查询时刻；构造双方共用窗口；Table4的50点条件可明确单列|不能把当前整场Q64的连续共同支持段直接叫作者窗口；Table1的完整窗口生成细节仍需保留未知|
|PTFM：训练与选模|复现原文已明确的降LR/早停/val-best；我们在相同训练人口重新拟合，各自训练配方与监督披露|152旧权重不能倒称按新协议训练；禁止按评价结果选择样本/seed去凑论文数字|
|双方：四属性与误差|同窗口、同有效点报告位置/速度/航向及覆盖；我们现有四属性接口可复用|当前147位置头的速航继承Mean；要学习速航须另设目标和拟合，不能冒称现有学习增量|
|TMTAFF：关联和融合分别比较|已有ITSA/MSTSA四项关联指标及四属性有符号ME入口；保留米制MAE/RMSE以解释实际误差|ME不等MAE；标准残差SD不冒称作者ESD。TMTAFF当前失效实现不能提供可信胜负|

可复用核心而非重写整套方法：[source_risk](../../association/evaluation/source_risk.py#L258)支持任意正长度Q和外部分组，但[旧147评价入口](../../tools/analysis/source_risk_full_benchmark.py:94)固定Q64，因此需要独立协议入口。[双源段入口](../../association/models/fusion_peer_inputs.py:25)明确当前同源多片段池化/连续段适配；[152训练](../../tools/offline/peer_fit.py#L192)及[事前差异声明](../proposals/fusion_peer_full_training_20261030.md:17)可定位训练偏离。TMTAFF已有[关联指标](../../association/evaluation/association.py:68)和[论文融合指标](../../association/evaluation/paper_fusion_metrics.py:37)，问题不全是缺指标。

**未知仍须保留：** PTFM准确样本/窗口列表与3:2划分；TMTAFF准确3000场景ID/版本、配准/查询细节、阶段训练分配和ESD代码。依据见[PTFM全文复核](/Users/a1/Projects/biye-ronghe/docs/surveys/2026-10-26_mtad_actual_use/ptfm_reaudit_20261027.md)、[TMTAFF全文复核](/Users/a1/Projects/biye-ronghe/docs/surveys/2026-10-26_mtad_actual_use/tmtaff_reaudit_20261027.md)。原DeepAF的杂波点迹任务与本项目轨迹融合不同，不能靠换单位成为同协议比较。

比较表应区分“原文报告值—本地按公开协议复现—我们同协议结果”。拿不到作者确切划分时，原文值列作参考，两种方法在同一明确重建协议下的配对才支持内部排名，不能标成精确原测试集打榜。“接近”的主要指标/允许差距须在新实验前定义，并结合重复运行波动与其他重要指标判断；不能事后只挑一个恰好相近的数。完全匹配协议但复现明显偏弱时先排查，仍弱则如实披露，不作为充分的超越证据。
