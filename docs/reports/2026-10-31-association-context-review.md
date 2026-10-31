# 168：前端上下文保留条件收益，尚未同时提升覆盖与误报

## Record：REC-ASSOCIATION-EDGE-CONTEXT-20261031-168

COMPLETED，plan1.0，源Idea `IDEA-ASSOCIATION-EDGE-CONTEXT-20261031`，源Mode C1.74/C56，[完整事前设计](../proposals/association_context_support_20261031.md)、[L2S168](../../HandOver/L2S-20261031-168.md)。2026-10-31北京时间12:21:42–12:27:04，冻结0ca218e，独立run/checkout attempt1。科学unit `mtad-2e982292721f.service`，原launcher326.852s、science321.572s，真实exit0/inactive-dead/success，Restart=no/Linger=yes；CPU0–3×4、GPU0指定UUID共卡未碰他人。峰RSS1,540,681,728B、CUDA31,441,920B；包括准备/训练/评价，不是纯网络延迟或在线能力。

问题：原056独立pair评分后由WS构组；新增邻边信息是否优于同容量额外训练，并改善跨源mPCA/PFA、覆盖与物理风险？实际三个前端A冻结056、A′OWN、B CTX；所有后端分别普通Mean与冻结159NN，六臂完整评价。056隐藏/normalizer/原θ不更新，新增6209参数头各1000GPU更新，共2000更新；每头同一256000pair呈现（128000正/128000负）、66213unique/775705eligible，128场全部访问，两个头共享这些呈现而非512000不同pair。base056原参数5697，单部署前端含新头11906参数；不含159后端/物理预处理，未完整重训全T新骨架。

T128固定前128，新增96场151明确mps→旧teacher kn一次，OLS/endpoint mps不转；前32原127。T128特征冻结后读T标签、T-only normalizer；两个头同初始化/batch/seed20/BCE/Adam.001/clip1，固定1000末步，无V选模型/阈值。T32是前端训练诊断，不是整链LOCO；后端159仍原组件留出/full。每场所有预测/分区先落盘后评价，原WS重算与159绑定；teacher T32差0、V最大3.814697265625e−6，小于事前1e−4。全部原pair，不删难例，T_test未开。

V32同72196个可评价位置点/1427目标；以下是各自可覆盖人口，位置MAE不能脱离覆盖比较。跨源为MSTSA decoded、场宏均值百分数；mPFA=FP/(TP+FP)，不是FP/TN。

| 前端＋159后端 | 跨源mPCA↑ | 跨源mPFA↓ | 位置点覆盖率↑ | MAE m↓ | RMSE m↓ | P99 m↓ |
|---|---:|---:|---:|---:|---:|---:|
| 原056＋WS | 78.989324 | 15.550185 | 98.441742 | 527.427043 | 1813.298607 | 7260.393881 |
| OWN＋WS | 82.249836 | 15.984275 | 97.263006 | 497.034528 | 1744.724041 | 6342.130070 |
| CTX＋WS | 82.183293 | 15.552221 | 97.172974 | 436.130771 | 1473.238836 | 4765.538122 |

raw跨源mPCA/PFA：原89.709923/27.055822、OWN94.328553/29.146933、CTX92.435369/27.822184。all decoded F1、ITSA、所有TP/FP/FN/TN及逐场均在原件，不以decoded改进说raw误报也下降。CTX vs原跨源mPCA+3.193969pp，mPFA反而+.002035pp，不能称双指标均提升；相对OWN mPCA略低、mPFA较好。

覆盖：WS/OWN/CTX分别71071/70220/70155点，1388/1366/1367目标有anchor，漏39/61/60；groups1605/1508/1472，重复217/142/105。CTX对WS**新417/丢1333点、救11/丢32目标**，不能用净−916掩盖获失。对OWN新1007/丢1072点、救25/丢24目标。OWN对原新117/丢968点、救3/丢25目标。

共同位置MAE差：OWN−WS −27.181158m，组件95%描述区间[−74.133641,−21.106131]；CTX−WS −88.592968m [−173.860390,−81.772645]；CTX−OWN −56.406119m [−110.478732,−42.625637]。因此当前均值信号不只是删去难点，但丢覆盖仍是实质失败。去最大组件CTX−WS −124.901848m，CTX−OWN −64.600504m，非仅最大组件平均效应。V仅9组件、重复开发/T-V共享原A身份，区间不认证独立泛化。

T32完整WS/OWN/CTX MAE597.178960/535.866371/499.020931m；覆盖98.459083/97.478024/97.470176%。CTX对WS救11/丢20目标，新增375个>1km增害，最大25.258km；共同差−87.067593m，CTX−OWN区间[−159.219726,+65.740895]跨0。平均有益同时丢人口在T已经存在，不能只归为V域偏移。

物理强控：V普通Mean的原/OWN/CTX为1276.328389/1313.097619/1290.298635m，CTX Mean仍差原Mean；效果依赖当前159读出及新分组的交互，不是图本身保证更好状态。速度/航向仍随对应组普通Mean输出，完整经纬度/位置、mps速度、圆周航向及论文ME等指标已从保存数组补齐，不能宣这些属性新增学习改善，ME不等MAE/作者ESD仍未知。

### 核验、原件与偏差

科学后一次独立audit＋完整四属性记账，原audit launcher21.733s/exit0、unit终态成功。独立检查198112exact/32373numeric/252null，全部保存主位置/人口/获失/尾部/组件/主要CI，以及纯NumPy raw/decoded全pair计数/mPCA/PFA通过；未独立训练或前向重做新head，不认证truth/作者方法胜负。audit本体11.123s/RSS544,841,728B；四属性是同评价入口原数组补项，不冒称独立属性算法审计。

完整104份公共原件（tar1,247,967B）与逐场/曲线/曝光/argv/退出/服务在[资料库](/Users/a1/Projects/biye-ronghe/docs/evidence/20261031_association_context_support/168/public-evidence-index.json)。服务器原run `/home/wangguangjie/djs/association-code/runs/LEVEL1-ASSOCIATION-CONTEXT-20261031-168-attempt-1`；private坐标/身份/分数/权重保留，不复制到Git。本轮无重试/加步/换配方。READY前本地sampler方法名/fixture次序修正见Idea；组间T诊断非LOCO是事前已声明限制。后验支持/纯度与关联区间脚本在最终报告落盘前作为补充记账执行，真实代码/argv/时间保留，不倒填事前机制检验。

## Squeeze：SQ-ASSOCIATION-EDGE-CONTEXT-20261031-168

消费上述Record及全部原数组/补充记账。后验代码[完整原件](/Users/a1/Projects/biye-ronghe/docs/evidence/20261031_association_context_support/squeeze_source.py)，0新训练/预测/调参，读数5.221s/RSS202,178,560B（外部命令5.535s），不是第二次科学验证。

- **同容量条件信号**：CTX−OWN共同域均值改善及V较低mPFA，支持继续研究关系条件；T该增量区间跨0、平均改善依赖NN后端，未认证唯一消息机制。CTX首/末采样batch BCE .110768/.098656，OWN末.129632；批次不同不能将两点当完整收敛或机制证明，11点完整曲线保留。
- **正确边与错误边一起新增**：V CTX相对原decoded真边救235/丢113，假边去406/加466，净TP+122/FP+60；mPCA升不等误关联解决。mPCA变化组件区间[+1.294027,+12.756081]pp，mPFA变化[−2.238032,+1.721963]pp跨0；都为9组件描述区间。CTX比OWN更少假边，但不能以此略过丢目标。
- **平均与局部大错分开**：V CTX新增1027个>1km增害，0/1/多源域143/708/176，其中same-anchor878、pure→pure201。OWN新增535。最坏共享例scene3835/query1，same-anchor、1支持→2支持，却7645.021→72284.858m，增加64.640km。故“支持源数增加便可靠”与“这些全是零支持回退”均不成立；无真值条件下识别相容状态仍未解决。
- **同anchor亦有真均值信号**：CTX same-anchor共同差−90.778607m，排除“所有改善只来自更换reference anchor”的简单解释；这不认证官方reference真实物理质量。真实K仍主要2、全窗离线，11906参数不等在线端到端或整体更快。

决策：保留前端关系条件的平均/召回信号，**当前固定旧θ的CTX不能作为覆盖＋mPCA＋mPFA联合升级**；不自动全300/加步/叠169。后继应先把“新score的决策工作点”和“对象/输出支持约束”分开，使用T侧定规则及同score强控制，不能在V上挑阈值后包装消息收益。纯多源高害与0/1回退各保出口。回流Mode C时只报告有界条件，不写第三节点上下文已证明是根因。

```yaml
package_type: EXPERIMENT_RECORD_and_SQUEEZE_REPORT
experiment_id: LEVEL1-ASSOCIATION-CONTEXT-20261031-168-attempt-1
source_idea_id: IDEA-ASSOCIATION-EDGE-CONTEXT-20261031
source_matrix_id: MC-MTAD-20261031-v1.74
plan_version: '1.0'
execution_status: completed_once_with_audit_and_saved_attributes
optimizer_updates: 2000
test_opened: false
decision: retain_conditional_context_signal_not_joint_system_upgrade
missing_evidence: [coverage_preserving_gain, stable_lower_cross_mPFA, full_V300_new_head, independent_domain_K3plus]
next_skill: deep-read_Mode_C_feedback
```
