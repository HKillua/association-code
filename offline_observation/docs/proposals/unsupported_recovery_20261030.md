# 145：支持域外的端点与传播速度分离

执行记录：原145真实FAILED，见[失败Record/Squeeze](../reports/2026-10-30-unsupported-recovery-entry-failure.md)及[S2L145](../../HandOver/S2L-20261030-145.md)；另立146接口修复成功，科学规格沿本文，原145不重启。

2026-10-30；IDEA_PACKAGE / research-pipeline-contract v1.0；idea_id `IDEA-UNSUPPORTED-RECOVERY-20261030`，plan_version1.0，证据状态partial，pilot。消费Mode C1.59的C55/P04/P13/P14/F242/M107/E141；来源为142正式Record/Squeeze、137原状态选择、127原CV/OLS实现，以及已完整分析的MeanTraj TSP2026关于存在域与输出损失对象的边界。这里复用来源，不新增精读、不把MeanTraj说成本文传播算法。

## 问题与假设

142已核一个CV工作末端×长时距放大路径，但没有证明替代规则有效，也没有查明原native异常的物理来源。主模式为系统审查：原来支持内平滑与支持外线性传播共用工作状态，是否可通过分离端点位置和传播速度，保留支持内输出而降低完整风险？全片段OLS速度与CV速度都来自相同合法原生位置/时间，是不同估计，不冒称新增健康信息或真实船速。

最小可检验预测：将支持外CV速度换为原全片段OLS速度，可消除已核分支的超大速度传播，但可能损害机动/短时外推；完整MAE、RMSE、P95/P99及before/after、组件反例决定实际价值。CV端点误差可能仍使结果变差，故位置与速度必须分离。静止端点是强简单控制，不以“运动模型更复杂”为成功。

## 固定设计与实现规格

- 固定原T32/12组件、V32/9组件、Q64、WS组、目标对应、完整人口。只读127/125/114/133/136/137原run与原manifest；不读取142私有reference选择个案来决定预测；不读ZIP/CSV、teacher/权重、T_test。
- 复放原137冻结的observation-rho，一次/必要node，所有需要的节点由**所有预测组的零支持query**决定，完全不读身份/reference/valid。逐节点Q64与原137绑定，保存端点和成本。不重选rho、不重训模型。
- 支持内原OBS-RTS group输出精确不变。支持外沿原anchor做位置×速度2×2：CV位置/CV速度（原臂）、CV位置/OLS速度、原native端点/CV速度、原native端点/OLS速度（原OLS回退）。另设CV端点静止hold。这里原OLS是以原native端点锚定，并非拟合截距；同米制chart，[north,east]。
- 新五臂：`WS_OBS_ZERO_CVP_OLSV`、`WS_OBS_ZERO_NATIVEP_CVV`、`WS_OBS_ZERO_NATIVEP_OLSV`、`WS_OBS_ZERO_HOLD`、`WS_Q_ZERO_CVP_OLSV`。最后一臂在旧Qhard支持内保持不变，只替换零支持，是相对强贡献控制的独立操作，不拼冠军宣联合训练。
- 新第六臂`WS_Q_ZERO_ALLSOURCE_CVP_OLSV`：零支持时对组内全部片段的CV端点+OLS速度预测，按每源等总质量/源内片段均分做原圆周恢复；支持内仍旧Qhard。来源只定义相等关系，不用编号、名称优先；它与anchor臂比较组候选效果，不宣等权等于可靠性最优。
- 现有四CV、源等权、nearest、Qsoft/Qhard及原V LOW/FixedLOW/Qgate等控制完整保留。所有新预测及私有数组先存，再打开评价reference；速度/航向仍原Mean。非法纬度、非有限数、原数组绑定失败即一次失败保全，不clip补救。
- 复用`observation_common`完整评价/分层/场景/组件/去最大；新增每个主要比较实际误差差的正害/负益、strict>1000m/max。主要比较：CVP_OLSV−原OBS_RTS、nativeP_OLSV−OBS_RTS、HOLD−OBS_RTS、Q_ZERO_CVP_OLSV−旧Qhard、ALLSOURCE−Q_ZERO_CVP_OLSV；T/V各2000次seed20组件描述bootstrap。全臂两两比较保留，支持域只是分层不删点。

## 验收与资源

新模块`association/evaluation/unsupported_recovery.py`只接受物理数组/组/source等价关系，无reference或身份参数。新入口`tools/analysis/unsupported_recovery_pilot.py`，单元核常速解析、经度跨界、因子交叉、支持内不变、换源名与换序；原真实缓存schema在发布前只读预检。保存真实命令/输入/退出/CPU/RSS/复放事件与阶段耗时，科学新checkout/run一次。CPU4/RSS6GiB/AS12GiB、1800s advisory，最多原节点总数个单rho核，0GPU/新拟合/新关联。共享评价不是独立审计；完成后用另一次只读数组复算主位置/人口/差值，范围单列。

## 三层检查、去留与边界

新颖性：这是成熟传播/OLS/域分离在当前可变源接口的有界适配，已有类似状态分段处理不影响价值；没有全球首次主张。新增价值必须是完整损害降低且强控/尾部保住，不能仅消除一个已看过的V极端例。可归因：T/V均固定、各操作分开、支持内exact、不按V选速度上限/时距阈值。T/V已开发且共享组件，结果只作内部pilot，不称独立泛化；真实K2与泛K软件能力分开。

若只修极端RMSE却MAE/P95或多数组件明显恶化，记录取舍不晋级；若仅OBS弱控制改善但不及旧reference-RTS/Qhard，记局部修正；若强控完整风险有增量且反例可解释，再另立扩大合同。本轮固定终点后不自动加seed/rho/阈值。下一skill为record-experiment→squeeze-experiment→Mode C。
