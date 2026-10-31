# 170：同T误报预算后保留平均信号，覆盖目标仍未达成

## Record：REC-ASSOCIATION-OPERATING-POINT-20261031-170

COMPLETED/plan1.0，Idea `IDEA-ASSOCIATION-OPERATING-POINT-20261031`，源Mode C1.75/C56、F268/E171，[事前设计](../proposals/association_operating_point_20261031.md)、[L2S170](../../HandOver/L2S-20261031-170.md)。固定168原score/056与159权重，**0训练/teacher/新CSV**；非168失败重试或换网络。T32同raw跨源宏mPFA预算、三scorer同候选规则，0 V/test选择；可行后T32/V32八臂全完成。

冻结a946321，独立checkout/run170 attempt1；科学unit `mtad-1b4ebc8f28c8.service`，2026-10-31北京时间12:38:25–12:40:54，science148.943s/原launcher153.596s，exit0/inactive-dead/success、Restart=no/Linger=yes。CPU0–3×4、GPU0指定UUID仅159评分，提交前可用17519MiB；RSS896,262,144B/CUDA22,912,512B。完整实际argv/输入/退出/成本原件留存，未碰152/107/132及168/169冻结目录。

### 训练侧规则及选择事实

原056在T32 raw跨源宏mPFA=.180990764192559作为共同预算。每个scorer全部T32跨源分数257等分位＋原θ，去重各258候选；每场正类/接受数非零，最大raw宏mPCA、tie先较低mPFA再较大θ。WS/OWN/CTX的T拟合θ分别1.7320103112387182/2.308038208316758/2.0085254541993995，T raw mPCA90.340194/90.022737/92.218552%，mPFA18.099076/17.517081/17.279613%。全774候选/约束/所选值在V读取前保存；无放宽预算/加网格。

有效score=原score＋旧θ−T拟合θ，沿同WS signed合并。WSCAL原阈值未变，保存物理/关联全部等于WS，不能将重复控制当新成绩。T32已被原前端训练，不是独立校准集；这是透明T侧开发规则，不认证外域无监督自适应。部署不读取identity/reference/source优先级。

### V32全人口结果

72196可评价位置点/1427目标；以下mPCA/PFA均跨源decoded场宏百分数，各自覆盖位置人口：

| scorer＋159 | mPCA↑ | mPFA↓ | 点覆盖率↑ | MAE m↓ | RMSE m↓ | P99 m↓ |
|---|---:|---:|---:|---:|---:|---:|
| WS原θ／WSCAL | 78.989324 | 15.550185 | 98.441742 | 527.427043 | 1813.298607 | 7260.393881 |
| OWNCAL | 78.099788 | 15.552328 | 98.508228 | 526.261569 | 1852.977916 | 7423.143410 |
| CTXCAL | 80.683890 | 15.368849 | 97.693778 | 432.800651 | 1477.092737 | 4933.120844 |

raw跨源：原mPCA89.709923/mPFA27.055822；OWNCAL89.109071/26.739648；CTXCAL90.325786/26.530929。不能混用raw/decoded或把PFA当FP/TN。CTXCAL decoded TP1384/FP1119/FN1194，原1309/1178/1269，即净TP+75、FP−59。全all/ITSA/逐场/组件/计数原件保留。

覆盖：CTXCAL70531/72196点，1517groups/136duplicates，1381目标有anchor/漏46；原71071点、1605/217、1388/漏39。CTXCAL对原**新增538/丢1078点、救17/丢24目标**，不能以净−540代替完整获失。OWNCAL71119点/漏35，救6/丢2目标、新143/丢95点；覆盖略升却mPCA−.889536pp、RMSE/P99更差，也不是联合升级。

共同域CTXCAL−WS −87.356404m，组件95%描述区间[−174.523042,−78.105429]；CTXCAL−OWNCAL −87.445919m [−174.841316,−78.483730]。去最大组件分别−121.478853/−119.222645m，平均信号保留。OWNCAL−WS共同差+.377843m [−3.861019,+.941418]，各自MAE的小幅下降不能代替共同人口核验。

T32完整MAE：WS597.178960、OWNCAL597.172695、CTXCAL521.951033m；点覆盖98.459083/98.833194/98.398912%。CTXCAL救17丢13目标但新394丢417点，说明“覆盖多少目标”与“覆盖多少参考时刻”也不同，须同时报告。其共同差−69.920520m/仍377个新>1km害，不以训练侧合法选择称风险已解。

V CTXCAL新增1001个>1km增害，最大64.640km，虽MAE/P99平均下降仍需保全。完整普通Mean强控及四属性都已补齐到各自population；速度/航向仍沿对应组Mean，无新属性监督学习，ME≠MAE/作者ESD未知。原300场157/114基准不被这32场结果替换，也没有可信论文排名。

### 核验与证据

新audit170 unit `mtad-8372a8e0239c.service`，原launcher29.214s/exit0，先独立保存主位置/全人口/覆盖/尾部/组件/主要CI及raw/decoded计数/mPCA/PFA，再完整saved属性。303695exact/52202numeric/336null通过，audit本体16.689s；不重训/重做head前向、不认证truth。后验另用sorted-prefix累计计数独立重算全部774个T候选与所选最优，最大差3.3306690738754696e−16；没有V重选标定。

102份公共原件tar1,535,316B、完整逐场/网格/四属性/argv/服务/退出见[证据索引](/Users/a1/Projects/biye-ronghe/docs/evidence/20261031_association_context_support/170/public-evidence-index.json)。原server `/home/wangguangjie/djs/association-code/runs/LEVEL1-ASSOCIATION-OPERATING-POINT-20261031-170-attempt-1`；private分数/身份/坐标/完整预测原位。本轮无科学失败/重试/扩大网格/追加训练。原代码内部重复检查选优不是独立核验，独立sorted-prefix补项在科学后执行、原argv/成本另记；不倒填为事前新机制证明。

## Squeeze：SQ-ASSOCIATION-OPERATING-POINT-20261031-170

消费上述Record及完整原件。[后验原代码](/Users/a1/Projects/biye-ronghe/docs/evidence/20261031_association_context_support/170/squeeze_source.py)读数7.244s/RSS228,855,808B，外部命令7.534s；0新模型/预测/标定/V选择。

1. **工作点影响部分取舍，未解释全部覆盖损失。** 同一CTX权重由旧θ到同T规则，V位置覆盖97.172974→97.693778%，mPCA82.183293→80.683890%、mPFA15.552221→15.368849%。仍低于原覆盖98.441742%，不是仅靠保守标定便获得三目标同时提升。跨实验只是同冻结网络/数据干预的透明比较，不能把它解释为一般根因或V优化的最佳阈值。
2. **同规则强控后仍有条件平均价值。** CTXCAL共同域明显优于OWNCAL/原，same-anchor共同差−84.073844m，不能仅以删除难点/更换参考解释；但跨源mPCA变化+1.694566pp区间[−1.226893,+11.800341]、mPFA−.181337pp区间[−1.733005,+2.535613]均跨0。9组件、重复开发不认证稳定联合泛化。
3. **风险出口需要分开。** 新1001大害中same-anchor848、0/1/多源177/699/125、pure→pure199；1→2支持的scene3835/query1仍增害64.640km。单纯增支持数、保输出代表或校准阈值均未充分解决相容状态/高害问题；不能把所有问题继续归于零支持。
4. **覆盖与逐边目标存在接口缺口。** 168/170在T/V均可提高部分成对召回/平均状态，却减少完整输出目标/参考时刻；现训练主要成对BCE、构组signed J，不直接承担错并一个目标后的全部输出代价。这是可核操作和可重复现象，**“目标不匹配导致损失”仍为AI机制假设**；竞争解释包括信息/状态质量、对象对应和训练分布。下一需按T错并/误拆对象及受影响query明确代价，并与同信息、同容量、同工作点的成对训练控制分开验证，而非继续V调阈值、永久固定anchor或追加同loss训练。

决策：保留CTX前端条件平均信号，170当前标定配方不作为覆盖＋跨源mPCA/PFA联合升级、不自动全300/新seed或新训练。至此168–170这一轮具体验证收尾；后继从Mode C形成新可分离Idea后再发布，不自动叠补丁。真实K≥3/新域、因果在线与全T新骨架仍未验证。

```yaml
package_type: EXPERIMENT_RECORD_and_SQUEEZE_REPORT
experiment_id: LEVEL1-ASSOCIATION-OPERATING-POINT-20261031-170-attempt-1
source_idea_id: IDEA-ASSOCIATION-OPERATING-POINT-20261031
source_matrix_id: MC-MTAD-20261031-v1.75
plan_version: '1.0'
execution_status: completed_once_with_independent_outputs_T_grid_and_saved_attributes
optimizer_updates: 0
decision: conditional_context_signal_remains_joint_coverage_goal_unmet
missing_evidence: [coverage_preserving_context_gain, stable_cross_metric_gain, group_task_loss_attribution, real_external_K3plus]
next_skill: deep-read_Mode_C_feedback
```
