# 195：固定风险头下的候选—动作条件验证

## IDEA_PACKAGE / plan1.0

2026-11-02；research-pipeline-contract v1.0，workflow v2.0；IDEA-ACTION-CONDITIONED-RISK-195；来源MC-MTAD-20261102-v2.08/C69/P12–14/F304/M123(AI)/E207、SQ194/189/191、AP-GMM-MFKL-20261029。主模式：系统接口假设挑战；evidence_status=partial（代码条件DE，目标因果/风险可迁移UN）。

领域/任务：统一经纬度、速度、航向、时间与来源归属的多源轨迹恢复。问题：178的OLS状态评分是否能通过相同几何语义重算，合理选择node及MC传播状态，减少高害同时保住O平均收益？不预设增加网络或源优先。

### 来源、假设与近邻

[Mode C2.08](/Users/a1/Projects/biye-ronghe/docs/surveys/2026-10-26_mtad_actual_use/reassessment_20261027.md#mode-c-208)保留新静态核查及替代解释。TAES2024多模型GMM研究已有模式保留，不宣此思想首创，也不搬用不存在的Cov/模式概率。本次近邻检索命中2025 TimeMCL/ICRA aWTA仅作待读线索，不使用摘要充精读/不作为方法依据。新价值是目标接口中“选择风险与被选状态”的可检验相容性，不把代码条件称已证致错机制。

H1：相同合法候选下，MC动作改变位置和速度相关风险特征，冻结risk可保留部分区分能力。H2：联合选择node×动作优于先固定node；单node域只有动作选择作用。反例：MC输入偏离训练域、共同源偏差、未来机动不可见、reference错配，或候选仍无更优状态。

### 固定干预与对照

原178 T32/12组件，复用对应leave-component风险头及189全部28臂（HARD/STRUCT×原front/O/PH/PC/PD/PG/MC×MEAN/NN），0新训练/teacher/似然/参考拟合，0V/test。原前端见过T，后端fold不等独立泛化。每组只在count0且query位于全组所有原生片段之前/之后改变位置；内部间隙、单/多源支持、速航、分组/覆盖/阈值不变。

MC候选与189完全同一报告均值/原生测量端点/球面推进，缺报告时回原O。对每node仅更新原12维中的第6/7/8（0-based）列：新状态到**冻结原O组中心/同源中心**的距离、新动作速度到**冻结原O组速度**的距离；其余列与normalizer/权重固定，避免同时重构群组上下文。非MC合法域特征原样。score为经验误差评分，不是概率。

|新增臂|改变|判别作用|
|---|---|---|
|CF|固定原178选中node集合，仅在其O/MC动作间按条件分数选|动作评分能否成立，单node也可变|
|RM|所有node仅MC状态，用MC条件分数重新选node|对照MC检验重选，而非新传播公式|
|JM|全部node×{O,MC}同评分竞争|对照CF分离候选放宽，对照RM分离O保留|

并列最低score按source→node→mode分层等权，源改名/换序不变，任意K接口；这是平局消解，不是固定双源贡献。全40臂，禁止拼前端/阈值冠军。参考只在预测冻结后用于评价/上限：分别报告最佳O、最佳MC、联合最佳、原选中node的最佳动作，说明oracle非部署。

### 终点、统计与解释

主比较JM−O、JM−MC、JM−CF、JM−RM、RM−MC、CF−MC，两前端分别完整位置MAE/RMSE/P95/P99、2000次seed20组件描述CI、去最大组件；全32场不删难例。次要：>1km新增害/救回、旧178对front害与MC对O害的保留/解决/新增、纯/混组、0/1/多支持、eligible、单node与多node；coverage/重复/raw与decoded跨源mPCA/mPFA及四属性同评价器核算；全部控制/全部结果留存。新特征超原normalizer±8范围及实际改node/mode计数单列，不把clipping当风险保证。

若候选上限有空间但部署排序无增量，结案本冻结迁移配方，后续才研究动作条件训练/可观测性，不自动加网络。若平均或尾部变差，不替换178、不凭局部赢家晋级。只有同时保平均/覆盖并降低目标高害，且两前端/组件反例可解释，才另立V验证；本轮不自动V/全300/新seed。

### 实施与验收（spec-driven-coding）

新增独立`action_conditioned_risk`纯函数＋195 CPU入口＋独审，不修改178/189代码或活动冻结目录。输入只已有T motion/events/score/model；输出独立run的40臂raw/状态/分区/策略/逐场与summary、公开聚合；坐标/身份/权重/大预测留server。原run全部COMPLETED且人口绑定；无文件覆盖、非有限或资源错误立刻失败并保全，不自动重试。CPU4线程/RSS8GiB，science1800s与audit1800s advisory，0GPU，原152资源不改；时间含重建/评分/评价/写盘，0训练不等0推理成本。

验收映射：原O/MC重建、保持域及所有旧28臂exact→数组核；源换序/K3/并列→软件测试（不认证真实K3）；score→独立NumPy读取原权重复核；MC→独立速度kn→m/s与球面推进；所有主指标/尾部→保存数组独核；四属性→既有同评价器记账（非全部双实现）。真实list/事件偏移/缺报告输入必须包含，合成仅接口排错、不作科学成绩。

status=pilot；source_transfer_id=none（本领域既有多模型原则＋项目接口干预，不宣复现KL/TimeMCL）；privacy=仅脱敏公共聚合；next_skill=record-experiment→squeeze-experiment→Mode C。全部计划在首次执行前固定，不因结果后改强控或范围。
