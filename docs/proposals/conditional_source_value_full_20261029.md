# 131：冻结125收益head的更广开发扩展

2026-10-29，Idea/实现规格1.0。125已完成限定独立主位置/人口核验、Record→Squeeze→[Mode C1.46](/Users/a1/Projects/biye-ronghe/docs/surveys/2026-10-26_mtad_actual_use/reassessment_20261027.md#mode-c-146)回流（F224/M95/E122）；本Idea已独立review与软件核验，实际READY/运行只由主交接发布。125正向信号允许准备扩展，不允许据V重新挑head、特征或配方。本文消费原CC-C1及该回流，不另创未经筛选的新方向。

## 1. 问题与来源边界

123/124表明固定借源的平均误差、收益点数与场景/组件表现可相互冲突；同一常量混合的损害幅度可能压过更多的小收益。125的具体问题是合法输入能否预测`y=e_FixedLOW−e_soft75`，随后按预测收益正负选择两个固定位置候选。二次ridge是125**事前主探针**，不能因V成绩再指定主头；常量、线性、置乱同样完整报告。

131只检验125冻结head在其余268场、原V32未见组件的适用范围。125的T head-LOCO及V32初步正向事实，须以其正式Record/Squeeze和独立数组核验为证据入口；本提案不提前认证数值或作用机制。新增价值若成立，是已有物理候选之间可转移的条件贡献决策；不把ridge、二次项或门控名称当原创，不以类似方法存在否定适配价值。

条件均值收益的符号对应两个固定候选的预期MAE决策，这是条件代数，不是新定理。有限样本估计、分布转移、错组/机动/共同偏差/reference关系仍可解释差异。计数和拟合残差是合法输入代理，不称有效独立样本数、协方差、绝对物理源质量；本项没有识别这些机制的干预。

## 2. 不变项与输入依赖

- 同056 canonical_all/Q64、同114 WS、全节点/全目标、首anchor对应、FULL valid和速度/航向沿原Mean。T_test封存。完整V300/38组件均为反复开发材料，原32与新增268分开；不是独立测试或可信同行比较。
- 直接读取125 `private/full-T-heads.npz`的四头`mean/std/beta/permutation`原数组。125未另存mode字符串，固定按TMEAN/LINEAR/QUADRATIC/SHUFFLED名字还原；严格核16个数组键、FP64/有限数、15维共同normalizer、1/16/136/136个系数和固定T置乱。校验同056/114/123来源、125原32/T32、15特征顺序、原θ与冻结先于V标记；不重新T读取、训练、teacher前向、归一化、中心拟合、λ、alpha或头选择。
- 原32全部11臂、全部公开method/pair和完整artifact直接导入125。禁止调用候选构造、head预测或输出重建；只从已保存数组提取聚合误差，并校验原55对及原32聚合methods/pairs与125相同。原V32特征文件只取保存的both mask形成一致评价分层，不重新构造物理特征。
- 新268的六强控Mean/FixedMean/LOW/FixedLOW/Anchor/NEAR及三fold calibration直接读取123各场保存数组。原123还报告low-only/high-only/no-support层，131沿125一致的固定region＋both层；六强控完整artifact逐字段exact，公共人口/全位置/坐标统计及共有分层与123绑定，不重算其预测或中心。
- 056缓存没有完整native质量，故新268按正式三分卷`数据集level1.zip.001.zip`、`.002.zip`、`.003.zip`逐场读取原factory：`MTADArchive.load_scene→_record→prepare_timed_scene(DATA)→scene_motion_evidence(canonical_all)`。不用Q64反拟合质量，不读reference成员；node/source/query/states/support/nearest/exact逐项与056 exact，pairs与056评分缓存对齐。只用056原scores，不读取其label给gate。
- 原生loader会读取身份/标签metadata并生成pair label；**不能声称整个进程在预测后才读取V身份**。新增候选/特征/动作API严格allowlist：states、support、nearest、sources、native_count、fit_rmse、OLS速度，以及冻结groups/calibration/scores/Mean/hard；不接identity、reference或目标support。source order仅原经验排序。无track时仅沿原factory的label区间构造原Q64，并与056 exact；不发明新时间范围。

## 3. 唯一新增计算与冻结决策

新268每场复用125 `make_actions_features`，按真实native_count/fit_rmse_m/ols_velocity_mps等产生同15特征和唯一必要候选soft75，hard直接用123 FixedLOW；中心/source_order/group_fold均冻结。hard和soft只差原两源支持均值混合，非both域exact相同；速度/航向原Mean不变。

四头均使用125同`z=clip((x−mean)/std,−8,8)`；线性`[1,z]`，二次/置乱`[1,z,z_i*z_j(i≤j)]`，TMEAN为常量。所有头固定`both & (predicted_gain>0)`才选soft75，否则原hard。不存在131重新拟合或置乱、搜索、V看结果调阈值/alpha/特征/归一化的路径。新features、native质量、predicted_gain/gate落盘后，才显式读取评价reference并构造新五臂的评价artifact；原生metadata读取事实仍按上节披露。

原32×11=352及新268×6=1608，共**1960个导入场景臂**；新增268×(soft75＋四head)=**1340个场景臂**。总3300臂、11臂55配对；原32不重复推断，不隐藏旧6臂共享准备成本。保存所有300场完整11臂artifact，便于独立数组核验。0新神经模型/神经前向/优化更新/解析head fit；恢复四个已有解析头并实际执行268×4次解析head应用，按真实组query数计操作量。

## 4. 估计量、分层与解释出口

主估计量为**新268固定二次head相对FixedLOW的同点径向MAE差**；同时对TMEAN、LINEAR、SHUFFLED、soft75、FixedMean等全部强控给出同人口差。全11臂55对、完整MAE/RMSE/P95/P99、论文坐标统计、覆盖/缺失/重复/无reference、所有场景和组件、动作率及成本共同报告；不取单模块最优拼装。预测组both与真目标support的评价层严格分开。

预定报告全300、原32、新268、38组件、场景数最大组件去除；新增268按原V32组件seen/unseen分层，并附全V及新增268按T32 head训练组件seen/unseen分层。T32组件集来自固定manifest前32个T_core，原V32集来自125；不是看结果划组。teacher曾全T训练、θ历史V开发，**这些unseen只描述该声明集合的组件成员关系，不是全链未见组件/身份或独立泛化**。不把query当IID作显著性，不删难例或仅报有利组件。

- 新268收益延续且相对简单/置乱头有新增价值：支持当前固定候选和开发协议下的条件贡献，继续保留尾部/异质性限制；不自动认证物理机制或扩大实验。
- 只胜hard但不胜常量/线性/soft或置乱：不能将收益归因二次条件信息，保留更简单强控。
- 收益局限原32/seen或被尾部/多数场景损失抵消：收窄125适用范围，保留分布、上游训练/阈值、reference与候选不足等解释；不从V调参、重选head或自动重跑。
- 无论结果，固定终点后先Record→Squeeze→Mode C；新训练、候选替换及后续范围均另定具体合同。

## 5. 实现、资源与验收

独立`tools/analysis/conditional_source_value_full.py`，CLI：

```text
python tools/analysis/conditional_source_value_full.py \
  --archive <001.zip> <002.zip> <003.zip> \
  --manifest configs/data/mtad-level1-scene3000.json \
  --source-system <056>/system --partition-run <114> \
  --source-control-run <123> --pilot-run <125> \
  --output <全新131目录> --cpu-threads 4 --time-policy advisory
```

CPU4、RSS6GiB/AS12GiB、5400s advisory，一次attempt-1，逐场原生读取不保留常驻SSH/守护；墙时不自动中断或加量。记录完整命令/输入/退出、真实原生事件数、268场factory读取、1960导入/1340新增、头系数与解析调用数、CPU/墙钟/RSS和计划偏差。错误、非有限、绑定失配或OOM保全原件，不自动重试/换配方；output拒绝覆盖。坐标、身份、中心、系数/normalizer、大数组留服务器，仅必要脱敏证据回传。

| 验收项 | 针对性软件证据 |
|---|---|
| 原模型还原且0fit/同预测 | 原FP64参数round-trip预测逐元素相同；错shape/不同normalizer拒绝 |
| 原32直接导入 | 禁用候选/head/输出评价调用时仍导入全部11臂和55对；公开值及artifact保留，错pair拒绝 |
| 新场只五候选、合法输入 | 六强控及soft与原构造exact；零标签固定头回退hard；identity字段被allowlist拒绝 |
| native质量来自真实事件链 | 软件原生131点超cap128仍计131，真实残差非零；056状态一项不一致即失败 |

软件fixture仅验证实现，不认证MTAD收益、作者复现或域机制。服务器真实运行/完整数组独立核验尚待主交接发布后执行。

```yaml
package_type: IDEA_PACKAGE
package_version: research-pipeline-contract v1.0
idea_id: IDEA-CONDITIONAL-SOURCE-VALUE-FULL-20261029
plan_version: '1.0'
source_matrix_id: MC-RESTART-20261027@1.46
selected_C_id: CC-C1
source_P_ids: [P11, P14]
source_F_ids: [F221, F222, CC-F3, CC-F4, F224]
source_M_ids: [M92, M93, CC-M1]
source_E_ids: [E119, E120, E122]
research_question: 原T32冻结条件收益head在其余268开发场景和预定组件分层是否仍有新增价值
hypothesis_and_mechanism: 固定候选的参考风险差条件信息可转移仍AI；不识别物理源质量
testable_predictions: 事前二次头新268相对hard的MAE差及常量/线性/置乱/soft强控完整尾部异质性
design_and_controls: 同125四head/15特征/参数/soft75与6旧控，11臂55对；原32直接导入
sample_materials_or_experimental_units: 反复开发V300/38组件，原32与新268分开；T/test不新读
primary_outcome_and_estimand: 新268二次头相对FixedLOW同点径向MAE差
secondary_outcomes: 全部强控/配对、完整人口尾部、场景组件/seen-unseen/去最大、动作和代价
analysis_plan_and_uncertainty: 预定组件集合，非query-IID；unseen仅指定集合且teacher曾全T训练
resources_time_cost_and_risk: CPU4/RSS6GiB/AS12GiB/5400s advisory/一次；1960导入/1340新增/0refit/0新NN前向
ethics_safety_privacy_or_compliance: predictor输入无ID/ref；loader身份metadata如实披露；private留server，test封存
stopping_and_interpretation_rules: 固定V300终点；失败保全不重试，完成先回流，不自动后继
status: diagnostic-preparation-not-READY
evidence_status: partial
missing_evidence: [ModeC146_record_handoff, remaining268_effect, physical_quality_identification, credible_peer]
next_skill: record-experiment_after_actual_execution
```
