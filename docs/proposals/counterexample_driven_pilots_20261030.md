# 154–156：反例驱动的分离验证

2026-10-30，IDEA_PACKAGE/research-pipeline-contract v1.0，来源Mode C1.65（F248–250/E150–153、SR-C1/C55、LF-B-C2），机制均AI/UN。沿项目discover-innovation问题/系统审查与transfer-ml-methods条件表；各项独立，不拼149/150冠军，不触碰152活动checkout/预算或107原训练。

## 154：常值产生在哪段

idea_id `IDEA-TFM-CONSTANT-PATH-20261030`、plan1.0。冻结107末端checkpoint，仅原127首T2、各首2真实身份跨源正对，共4对/Q64；0训练/V/test/CSV。按原153重建输入，记录TCM/左右encoder、FC memory、decoder输出、raw state projection的跨样本/时间变异与有限性，参数行范数及LayerNorm尺度；实际数组/坐标/权重留private，只回脱敏统计。

干预仅同权重把fc2 memory置零；比较raw投影与物理位置变化。再对保存decoder latent用原投影/affine权重以float64重放，区分浮点量化与原normalized读出变化。预测/干预固定后才读T参考，原TFM、零memory、double投影及普通Mean同点位置/尾部报告。中间统计/操作不证明训练优化根因。软件已知零投影可造成常值、随机投影保留输入响应；hook退出必须移除、不修改权重。

单次新run，GPU1兼容余量≥6GiB、CPU0–3×4/RSS8GiB、1800s advisory；失败保全不重训/自动重试。定位到操作路径后，先Record/Squeeze/Mode C再决定训练修正。

## 155：覆盖不足与同信息模型边界分离

idea_id `IDEA-SOURCE-T-COVERAGE-20261030`、plan1.0。只扩大训练资料为manifest首128个T（原32＋预定96），不按误差/反转挑样本。原32 pack/成员沿133/125，新增96只用151单位一致motion/evaluation；冻结056模型CPU产生原全部i<j分数，按同114 WS构组。151 mps状态只在冻结旧056输入视图换回kn，源评分位置/OLS速度语义与旧10特征一致；不能直接把mps喂旧teacher。

原10维/66系数QUADRATIC、log误差标签/每query每源权/λ1/clip±8不变；新增128训练的12个原T32组件LOCO头＋full共13fit。原T32仍在自己的同组件留出头下评价，V32仅full头，0 V选N/λ/头。原32合法特征/标签逐项绑定，原Qhard输出与133一致；对照原Qhard、T128新hard、sourceequal。新增96的WS分区为训练输入，不改变原T32或V32分区。

评价原T32/V32完整37635/71071预期位置点，全部目标/漏点/重复/0、1、多支持、场/组件/去最大/MAE/RMSE/P95/P99与新增>1km/max；选源错误/救回/新增/excess和匿名slot0非最佳层只作评价。记录新增训练组件/合法监督/罕见角色数量及真实teacher调用、13fit/成本。若V好而T或尾部不利，不升级；更多训练资料不是独立创新或外部泛化证据。0神经更新/raw CSV/test，CPU8–11×4/RSS8/AS14GiB、3600s advisory，各一次，失败保全。

## 156：同来源历史预测选择传播模型

idea_id `IDEA-WITHIN-SOURCE-FORECAST-20261030`、plan1.0。固定原T32/V32、127全部事件/状态、125/114组及133Qhard；只改零源支持的位置。A原Qhard，A′149匿名最近源/片段候选（平局分层均分），A′+B在相同选定候选中改传播，不把源内自验证误差用于跨源排序。

三专家：端点HOLD、端点＋全片段OLS速度、端点＋最近min(8,n)点局部OLS速度。局部窗口8沿既有150物理描述，不依据V调选。对每个片段/外推方向，在25%、50%、75%位置截断历史（至少2个训练点、至少1个严格留出点）；按目标外推时距选择每截断后时间最接近的留出观测，三个专家预测**相同**点，各cut等权平方局部米误差。最小损失专家平局预测等权；n=2无合法历史时回全局OLS，未知计数保留。before用反向历史，整体仍离线，不能称严格在线。

固定源偏移在同来源验证误差中可抵消，但随机噪声相关/非平稳机动/未来超过可验证时距会破坏迁移；记录时距不匹配与无历史，不能称无偏真实风险。参考不进入预测/选模，0 reference系数fit/teacher/神经；原生OLS/候选验证计算不是零成本。

六臂：原Qhard、匿名NEAREST-GLOBAL（绑定149）、NEAREST-HOLD、NEAREST-LOCAL8、NEAREST-CV、NEAREST三专家等权。只动零支持位置，支持内及速度/航向exact不变；纬度饱和/经界沿原native协议并记次数，源改名/成员换序不设固定优先。主要为CV−原Qhard及CV−最近GLOBAL完整MAE，另报全部两两/零域及before-after/完整覆盖/尾部/组件/增害、专家动作与验证域外。seed20组件bootstrap2000次为开发域描述；不按V选窗/阈值或自动扩量。

新CPU0–3×4/RSS6/AS12GiB、1800s advisory一次（154结束后独立显式提交）；原科学输出不动。软件检查常速/跨经界、n2退化、时间平移/单位、源改名/换序、参考不入API和不变域。若只局部收益仍保完整反例；Record→Squeeze→Mode C后决定，不直接叫新机制。

## 验收与结论边界

所有run先公开具体L2S/READY及命令，再在独立冻结目录各一次；原数组/预测先保存后评价。成熟方法适配允许作为候选创新，但本轮155主要是学习条件分离，154为失效定位，156是否有新增价值由实际收益决定。科学后只读输出复算、真实launcher/服务/退出/人口/成本与计划偏差须分别记录；不为了通过改指标或覆盖失败。训练活动152不改，不访问封存测试。
