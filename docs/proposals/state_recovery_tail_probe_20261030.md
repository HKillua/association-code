# 141：零支持恢复高损害点的传播与映射核查

2026-10-30；IDEA_PACKAGE / research-pipeline-contract v1.0；idea_id `IDEA-STATE-RECOVERY-TAIL-PROBE-20261030`、plan_version1.0。类型computational diagnostic，证据状态partial；不预先选择修复。本轮用户明确授权分别推进选源、贡献及状态高损害核查，旧批次三项上限已经结案，不提供旧命令续跑许可。

## 依据、问题与竞争解释

消费[Mode C1.57](/Users/a1/Projects/biye-ronghe/docs/surveys/2026-10-26_mtad_actual_use/reassessment_20261027.md#mode-c-157)的OD-P2/P3、F238、M107（AI/UN）、E136、C55，以及[137 Record/Squeeze](../reports/2026-10-30-observation-scale-review.md)。直接证据：观测尺度RTS的完整V风险恶化，预测组零支持2770点承担99.69%SSE，4393的52点尤其集中；附加层此前尚未独立重建，不能直接称外推根因。代码`native_cv_state_control.py::cv_native_positions`在原生跨度外用端点状态线性传播；`observation_scale_pilot.py::recover_equal`和旧Mean在零支持时使用当前状态首节点回退。代码事实不等逐点验证已完成。

主模式为现象驱动/系统审查：高损害究竟绑定了哪些真实输入、预测组与恢复传播？竞争解释为端点状态/传播速度、时距、组成员/对应选择、球面表示数值及其它输入/参考问题；不预先宣布某一物理根因。

可检验预测：①零支持可评点的实际输出应逐点等于其所选预测组首节点的已存CV位置；否则定位恢复/映射差异，不能继续按传播解释。②对这些query涉及的唯一anchor，以原两rho独立复放固定KF/RTS，保留端点位置/速度，完整Q64输出必须绑定旧137保存位置；不能只从wrapped经度反解真实速度而忽略绕圈别名。③两rho间的高损害差异若与仿射传播有关，应能同时看到本次复放得到并与旧位置绑定的传播速率、时距与空间位移的变化；共变仅为过程定位，不证明真实噪声、机动或reference的因果来源。

## 固定输入、人口和分析

- 只读已完成137原run及127原生事件/物理/评价缓存、原manifest；不读原ZIP/CSV、模型/权重或T_test，只对零支持点涉及的唯一anchor复放原固定两rho的KF/RTS以核中间量，不选rho、不新增方法或改变已存物理输出。
- 原T32/12组件与V32/9组件，全部固定目标/首代表/参考mask保留；对每个可评位置点独立重建预测组有支持的不同源数。仅以物理support=0定义本次核查人口，预期T1487/V2770，实际不符须记录并停止，不删除难例。
- 按真实identity下各输入片段[min,max]跨度的逐点并集重建fixed支持层，区别于预测组support。身份/参考只在诊断与评价使用，不进入部署规则。
- 对全部零支持点记录：首代表组/anchor、before/after、距原生端点秒数、组内最小时距、片段/事件数及时间间隔、输入OLS/两端差分速率、保存CV两rho/FWD/RTS对应位置误差、复放端点速度及位置相对原生端点的位移、传播后位置绑定残差。原reported speed只保留native单位，不把其差分语义当独立量测。
- 复放核只接受原生位置/时间、原Q64和已冻结rho，独立实现原CV递推并保存FWD/RTS首末状态；同一node/rho只做一次，逐臂绑定原保存Q64位置。分别记录核调用、事件与转移/后向步；不是新target-reference拟合。经度展开及球面位置输出沿原算法，既保留未wrap局部状态，也核wrapped保存位置，避免反解别名。
- 所有目标完整人口/各方法全量误差、零支持人口和SSE份额、逐scene、before/after、identity-span内外分层均报。4393按已公开现象单独保留，完整V另列误差最高的20个匿名case以供复核；这属于已暴露V的事后机理分析，不作V调参或独立确认检验。完整case映射、坐标和身份只存服务器private。
- 本轮不额外扫描阈值或作显著性筛选，不把阈值分层变部署策略；不改变source优先级、状态、分组或覆盖，不拼139/140成绩。

主要估计对象为原零支持人口映射绑定、传播关系的重建偏差及完整损害作用域；不是新方法MAE胜负。单位分别为m、m²、s、m/s和原native_speed。所有null/空层、缺输入、绕圈歧义保留；检查失败为一次失败终态，不自动换子集或放宽容差。

## 实现规格与核验

新入口`tools/analysis/state_recovery_tail_probe.py`；使用独立NumPy读取原137/127保存数组，从原artifact重建对应，不调用生产CV/恢复核，按已读原方程独立数值复放选定anchor；球面位置误差可复用既有独立测距函数。文件已有则拒绝覆盖。公开run/summary/scene-summary/匿名top-case不含原坐标、source名字、原身份或权重；原case表留private。

| 验收 | 核验 |
|---|---|
| 重建完整T/V人口与零支持/identity跨度口径 | 原64场artifact、native输入及manifest逐项绑定；与137原汇总分母核对 |
| 原输出确属当前anchor与query | 每个零支持点的四臂位置及原生OLS输出绑定 |
| 复放只读物理数组、固定rho | 含跨经度边界/跨度内外的解析及原核等价例，保留端点状态；reference不进入复放 |
| 无旧输出变化、无新方法/选参或训练 | 入口只读白名单及复放成本计数；对所有trace节点Q64核与原保存位置绑定 |
| 一次真实固定终点与保全 | 新checkout/run/launcher，正常退出及真实service双核 |

CPU2，RSS2GiB/AS4GiB，1800s advisory，0GPU/神经训练/teacher/target-reference拟合。原64场压缩缓存读取另计；至多2×(825+1388)＝4426个固定anchor/rho复放核，实际去重计数，不把非选定节点重复复算。预算由仅仿射草案的600s在发布前修订，原因是wrapped保存值的反解不能认证真实速度；未启动科学。保留完整耗时/事件/核调用与未知比例。成熟后由HandOver单独发布READY，当前本文件不构成已启动事实。

## 决策和质量检查

如映射或传播重建不一致，先保全具体链路，另立修复合同；如一致且风险集中与传播参数共存，先归纳可用输入及支持权限，再立单变量的恢复对照，不能直接删点或按V调rho。若不能解释风险，报告被排除/未排除的解释，不追加无界核查。

本核查是已有CV传播与本流水线的过程审查，不宣新算法或绝对原创。价值是把公共支持层定位落实到真实私有中间量，为具体恢复方案提供可反驳依据；数据/分组/旧参考局限保留。下一skill为record-experiment→squeeze-experiment→Mode C，状态修复需要新的具体Idea与真实验证，而非仅停留在本报告。
