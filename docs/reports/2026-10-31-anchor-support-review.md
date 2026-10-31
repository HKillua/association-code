# 169：保代表不等保风险，逐query硬保支持限制了收益

## Record：REC-ANCHOR-QUERY-SUPPORT-20261031-169

COMPLETED/plan1.0，Idea `IDEA-ANCHOR-QUERY-SUPPORT-20261031`、Mode C1.74/C53×P13，[事前设计](../proposals/association_context_support_20261031.md)、[L2S169](../../HandOver/L2S-20261031-169.md)。冻结0ca218e，2026-10-31北京时间12:21:42–12:24:18，独立run/checkout attempt1；科学unit `mtad-c677e56f58c3.service`成功exit0/inactive-dead、Restart=no/Linger=yes。原launcher158.849s、science155.414s，CPU8–11×4/GPU1指定UUID仅冻结评分，峰RSS876,843,008B/CUDA22,817,792B；0训练/teacher/CSV/test，不与168叠加。

原T32/V32，WS/REF/ANCHOR/SUPPORT×Mean/冻结159八臂；原056 scores、θ1.7320103112387182、159组件头不变。WS绑159、REF绑165全部数组；ANCHOR固定原输出代表，只动非anchor到原组；SUPPORT再要求每次操作每query支持来源数不减。只读物理/score/source相等关系，无身份/参考、特定源角色/编号权重；全node恰好一组、原query不筛、预测存盘后评价。完整实际范围符合事前计划。

V32同72196位置点/1427目标，跨源decoded场宏均值百分数：

| 分组＋159 | mPCA↑ | mPFA↓ | 点覆盖率↑ | MAE m↓ | P99 m↓ | 对原新增>1km增害 |
|---|---:|---:|---:|---:|---:|---:|
| WS | 78.989324 | 15.550185 | 98.441742 | 527.427043 | 7260.393881 | — |
| REF（原165控制） | 79.916089 | 14.756457 | 98.530389 | 517.947017 | 7166.747110 | 140 |
| ANCHOR | 79.655779 | 14.925592 | 98.441742 | 528.804615 | 7328.237103 | 50 |
| SUPPORT | 78.980969 | 15.548133 | 98.441742 | 527.400156 | 7260.393881 | 0 |

raw全部不变；V REF/ANCHOR/SUPPORT分别8场65moves、8场32moves、1场2moves，J增107.902191/54.367970/7.224814，均未耗尽4N。ANCHOR、SUPPORT原代表/1605groups/217duplicates/漏39目标及71071有效点全部保持，新/失覆盖均0；REF救5丢2目标/新163丢99点，不能只报净收益。

T32 WS/REF/ANCHOR/SUPPORT完整MAE597.178960/597.588040/597.788926/597.178960m；ANCHOR14moves，SUPPORT0moves。T所有八臂及完整输出均保留，不能把V单场2moves宣泛化稳健提升。

共同域ANCHOR−WS +1.377571m，组件95%描述区间[−1.759788,+1.863188]；SUPPORT−WS −.026887m [−.031747,0]，去最大组件0；SUPPORT−ANCHOR −1.404459m [−1.893871,+1.759788]。SUPPORT的微小均值来自一个组件，mPCA还低.008356pp，不能选为联合冠军。ANCHOR跨源F1虽81.350586→82.000397%，位置更差；F1/纯度/代表保持不是物理保证。

普通Mean V WS/ANCHOR/SUPPORT 1276.328389/1281.940969/1276.438924m；当前NN读出仍重要，支持硬限制没带来Mean额外收益。完整四属性经原保存mps数组补齐，组变导致速度/圆周航向随Mean变化，未新学属性，ME/MAE/ESD边界不变。

科学后一次限定audit＋四属性记账原launcher36.112s/exit0、真实服务成功。独立305001exact/52846numeric/336null，主位置/人口/覆盖/尾部/组件/CI、全raw/decoded计数/mPCA/PFA及每次move的J/anchor/所有query来源支持数通过；audit本体20.205s/RSS589,164,544B。不重做159前向/训练或认证truth。完整99份公共原件tar1,183,061B见[索引](/Users/a1/Projects/biye-ronghe/docs/evidence/20261031_association_context_support/169/public-evidence-index.json)，后追加2份只读支持/关联区间记账单列在同目录science；私有坐标/身份/预测/成员轨迹留原server run `/home/wangguangjie/djs/association-code/runs/LEVEL1-ANCHOR-SUPPORT-20261031-169-attempt-1`。

无科学失败/重试/追加预算。后验记账先于最终报告持久化，属科学后的事实采集，解释在本Record后展开，不倒填机制为事前已证；补充代码/原argv/exit0/成本单列。

## Squeeze：SQ-ANCHOR-QUERY-SUPPORT-20261031-169

消费上述Record及全保存原数组；[后验原代码](/Users/a1/Projects/biye-ronghe/docs/evidence/20261031_association_context_support/squeeze_source.py)0新训练/预测/调参，7.599s/RSS228,728,832B，外部命令7.874s。

1. **固定代表隔离了人口变化，但不能保状态。** V ANCHOR新害50全same-anchor、新0/1域16/34、纯→纯16；最坏1支持→0、增10.543km。代表/覆盖保持后还有可重复风险，不能只改evaluation对应或把所有错误称reference变化。
2. **支持约束在此次几乎冻结搜索。** T0move、V仅2；V原ANCHOR搜索224次正增益候选因anchor条件拒绝，SUPPORT搜索另80次因支持条件拒绝（迭代求解访问计数，非80个独立候选）。该“保每一query”条件过强的操作限制可核，但不能从当前0害推导支持数对任意状态的数学安全性。
3. **召回/误报并不一同改善。** V SUPPORT真边救0/丢1、假边去7/加4；只减少部分假边却牺牲一条真边。ANCHOR真边救23/丢6、假边去58/加33；其mPCA+.666455pp、mPFA−.624593pp，但物理平均/尾部变差。REF原例的140害已逐项复现，不新增算法成绩。
4. 单来源支持计数同样不代表状态相容或独立传感器信息；168的1→2支持大害为另一独立反例。真实K2、全窗离线、V重复开发与T/V共享原身份限制不变。

决策：**当前ANCHOR/SUPPORT硬限制配方不扩大、不接最终方法。** 保留“输出对象人口”和“query观测支持/状态质量”是不同接口问题的直接证据；后续若保多对象/候选状态，应在不固定源角色的前提下允许合法纠错，并同时核覆盖与错误贡献。不能为了保覆盖永久冻结原错组，也不能把支持数增加当状态可靠。与168信号分开保留，后继须另有具体可分离Idea。

```yaml
package_type: EXPERIMENT_RECORD_and_SQUEEZE_REPORT
experiment_id: LEVEL1-ANCHOR-SUPPORT-20261031-169-attempt-1
source_idea_id: IDEA-ANCHOR-QUERY-SUPPORT-20261031
source_matrix_id: MC-MTAD-20261031-v1.74
execution_status: completed_once_with_independent_audit_and_attributes
optimizer_updates: 0
decision: close_current_hard_constraint_recipe
missing_evidence: [nontrivial_support_safe_group_gain, stable_joint_coverage_association_physics, external_multisensor_validity]
next_skill: deep-read_Mode_C_feedback
```
