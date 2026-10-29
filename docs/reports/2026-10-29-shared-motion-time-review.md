# 122：一阶速度相关共享项没有稳定留组增益

2026-10-29。先记录真实运行，再做Squeeze，最后提供Mode C输入。本项唯一原V32/9组件运行正常退出0，四臂/六配对/1605组完整保留。**1024个可定义留出组中，JOINT相对STATIC的输入相对残差均值1160.363→1163.698m、RMSE1684.297→1684.910m，主要方向不利；去最大组件后均值差进一步为+49.974m。** 本轮不支持直接应用当前时间补偿配方，也不把整个时序研究判为无价值。

这里的量是`||组均值跨源位置差−b+vτ||`，**不是对reference的物理恢复误差、不是MTAD位置MAE，也没有改变任何预测**。参数带秒单位不证明发现了真实时钟。原生状态—观测分离、正确动态模型或不同合法时间权限是独立研究问题。

## 1. EXPERIMENT_RECORD：实际操作、血缘与执行

```yaml
package_type: EXPERIMENT_RECORD
package_version: research-pipeline-contract v1.0
experiment_id: EXP-SHARED-MOTION-TIME-RESIDUAL-122-20261029
package_id: EXP-SHARED-MOTION-TIME-RESIDUAL-122-20261029
created_at: '2026-10-29'
source_idea_id: IDEA-SHARED-MOTION-TIME-RESIDUAL-20261029
plan_version: '1.0'
execution_status: completed / diagnostic
research_question: 常量共享相对差之外是否存在能在留出预测组上重现的速度相关剩余
hypothesis: 跨组运动差异使b与速度列可分；联合模型应在相同留出组超越静态并通过置换比较
planned_vs_actual_design_and_controls: 原V32/WS/三fold/四臂六配对/一次seed20训练和留出独立置换全部执行；预测不变
actual_configuration: 本节真实命令、参数、原run和launcher原件
samples_materials_or_data_provenance: 原056物理缓存→原114 WS→原118 membership/order/fold绑定→输入相对差/差分速度→留组拟合及残差
results_with_uncertainty: 四臂表/六配对表/完整人口/组件及拟合稳定性见§2–5；单次开发材料，无IID或显著性主张
all_runs_or_repetitions: 一个attempt-1；零神经训练/前向；独立保存数组核验是同一结果的数值审核，不是重复研究
observations_and_deviations: 执行前收窄数组读取flags及metadata措辞，不改配方；无已发现执行中改参/扩量/重试
evidence_locations: 资料库122目录、主仓Idea/实现/原L2S；实际组向量和带符号参数仅在原run private
known_limits_and_missing_evidence: Q64重建导数/工作chart/预测分组/单次置换/重复V及共享身份；独立审核不覆盖原支撑差分/参数拟合/组件重聚合或真实clock
next_routes: squeeze-experiment then deep-read Mode C
next_skill: squeeze-experiment
```

上游为[事前Idea1.0](../proposals/shared_motion_time_residual_20261029.md)、[输入/时序矩阵IT-C3](/Users/a1/Projects/biye-ronghe/docs/surveys/2026-10-27_expanded_search/input_time_action_matrix_20261029.md#7-后续增量先核非平凡作用再分离共享修正的空间与时间解释)及[原交接](../../HandOver/L2S-20261029-122.md)。实现为[shared_motion_time_residual.py](../../tools/analysis/shared_motion_time_residual.py)。独立checkout `/home/wangguangjie/djs/association-code-run122`冻结`6612de9`，事后原件核tracked工作区干净；该版本用于历史定位，不作科学准入。

原run：`/home/wangguangjie/djs/association-code/runs/LEVEL1-SHARED-MOTION-TIME-RESIDUAL-20261029-122-attempt-1`，launcher为相同路径加`-launch`。必要脱敏原件位于[资料库122目录](/Users/a1/Projects/biye-ronghe/docs/evidence/20261029_small_validation/122/)：`run.json`、`summary.json`、全部32场的`scene-summary.json.gz`、request/launch/status/exit_code/stdout/stderr/worker.stderr及后补独立审计JSON和完整执行代码。真实chart、带符号b/τ、组差/速度、置换行号和残差数组保存在原run的`private/residuals-<scene>.npz`，本报告不复制。

实际payload按`launcher-request.json`原样记录：

```bash
/usr/bin/taskset --cpu-list 8-11 /home/wangguangjie/djs/mtad-field-20261016/bin/python \
  tools/analysis/shared_motion_time_residual.py \
  --source-system /home/wangguangjie/djs/association-code/runs/LEVEL1-ASSOCIATION-20261022-056-attempt-1/system \
  --partition-run /home/wangguangjie/djs/association-code/runs/LEVEL1-STATIC-PARTITION-PILOT-20261029-114-attempt-1 \
  --shared-run /home/wangguangjie/djs/association-code/runs/LEVEL1-SHARED-COMPONENT-20261029-118-attempt-1 \
  --manifest configs/data/mtad-level1-scene3000.json \
  --output /home/wangguangjie/djs/association-code/runs/LEVEL1-SHARED-MOTION-TIME-RESIDUAL-20261029-122-attempt-1 \
  --cpu-threads 4
```

cwd为上述冻结checkout，`CUDA_VISIBLE_DEVICES=''`、OMP/MKL/OpenBLAS线程各4。systemd托管`mtad-f7b6f584cdf4.service`，Linger=yes、Restart=no，600s为advisory。主run、launcher returncode及exit_code.txt均为0，stdout有COMPLETED，stderr/worker.stderr均为空，未触发advisory或超时。后补`launcher-posthoc-service-terminal.txt`及`terminal-synthesis.json`在**UTC12:54:50.137810**核同一服务`inactive/dead`、Result=success、ExecMainStatus=0，且原run/launcher/exit一致；这些实际终态证据已经读取，不由SUBMITTED或进程消失推断。

程序UTC **2026-10-29T12:51:46.884944→12:52:01.049806**，最终单调计时 **14.164893s**；summary写入前为14.160225s，不替代最终总成本。launcher记录总计 **21.160754s**。RSS **2007040000B**，CPU亲和8–11、AS8GiB/RSS4GiB预算。解析参数拟合计数366，另有数据提取/线性代数成本；**0神经训练、0模型/前向/optimizer、0GPU**，不是“零计算”。未按V目标调参、重选组/阈值/normalizer或原118中心，未访问ZIP/CSV/T_test。硬件型号及完整运行包版本未在这些原件单列，不据此前环境推填；上述整次成本不等在线逐事件时延。

**计划偏差/措辞订正：** 执行前独立复核发现，复用的metadata绑定链会读含既有评价汇总的056等JSON，因此将“未读reference/identity”的过宽flags改为`reference_array_fields_loaded=false`、`identity_array_fields_loaded=false`、`association_label_array_fields_loaded=false`，并显式记录`metadata_binding_uses_existing_evaluation_summaries=true`。它们是来源/人口绑定，不参与拟合或择参。这是执行前证据措辞修正，未换输入白名单、拟合、置换、终点或配方；不能在报告重新扩大成“进程完全未读任何含评价信息的文件”。实际参数函数只接受物理字段/冻结分组，不接受reference、identity或标签数组。

## 2. 完整输入人口与四臂主结果

原32场/9共享身份组件、3851节点、1605组及原顺序全部保留。共同支撑跨源pair-query为43004；两成员在`q−1/q/q+1`均支撑、可求真实秒中心差分的pair-query为39304。每组先平均这些d/v，再在训练fold组等权拟合，因此39304不是独立样本量。

- 1024组有合法导数且有可用训练chart，四臂均可评价，占全部组 **63.8006%**。
- 565组没有跨源共同支撑，另16组虽有共同支撑但无合格三点导数；合计581组/36.1994%留在无定义分母，未填零残差或删组抬效果。
- 32场均至少有可定义组；四臂使用完全相同1024留出组。没有physical输出、reference误差、覆盖/重复或关联分数的增量。

`STATIC_LS`拟合b；`TIME_LS`在b=0约束下仅拟合τ；`JOINT_LS`拟合b和τ；置换臂在同折集合内固定一次seed20打乱速度对应。RCOND=1e−8仅用于数值秩。符号为`d=b−vτ`，残差`d−b+vτ`。下表m均指输入相对残差范数，不能简称“恢复MAE”。

| 臂 | 组数 | 组等权均值(m) | RMSE(m) | 中位(m) | P95(m) | 场景等权均值(m) |
|---|---:|---:|---:|---:|---:|---:|
| STATIC_LS | 1024 | 1160.363 | 1684.297 | 677.791 | 3645.462 | 469.704 |
| TIME_LS | 1024 | 2991.580 | 3192.025 | 3269.543 | 4443.506 | 2685.613 |
| JOINT_LS | 1024 | 1163.698 | 1684.910 | 717.769 | 3630.456 | 496.433 |
| JOINT_PERMUTED_LS | 1024 | 1172.236 | 1692.148 | 728.775 | 3655.770 | 502.764 |

主要JOINT−STATIC：均值 **+3.335018m**、RMSE **+.613038m**、场景等权均值 **+26.728535m**，中位+39.977681m；P95反而−15.005944m。不能只挑P95小幅改善或518个组变好宣净胜。只有一次固定开发诊断/一份置换，未构建CI、p值或等效性检验；组、场景和共享组件也不具独立同分布保证。

## 3. 六个配对全部保留

下表均为candidate−baseline；好/差/同按残差均值或逐组范数差、容差1e−9m。场景和组件均值在各自原合格组上比较，不把query数当重复次数。

| candidate vs baseline | Δ均值(m) | ΔRMSE(m) | 组好/差/同 | 场景好/差/同 | 组件好/差/同 |
|---|---:|---:|---:|---:|---:|
| TIME vs STATIC | +1831.217314 | +1507.728357 | 160/856/8 | 0/30/2 | 0/8/1 |
| JOINT vs STATIC | +3.335018 | +.613038 | 518/498/8 | 8/22/2 | 2/6/1 |
| PERM vs STATIC | +11.872877 | +7.850988 | 465/551/8 | 9/21/2 | 3/5/1 |
| JOINT vs TIME | −1827.882296 | −1507.115319 | 859/157/8 | 30/0/2 | 8/0/1 |
| PERM vs TIME | −1819.344437 | −1499.877370 | 858/158/8 | 30/0/2 | 8/0/1 |
| PERM vs JOINT | +8.537859 | +7.237950 | 477/535/12 | 14/15/3 | 5/3/1 |

带b的模型远优于“仅时间、b=0”说明该限制在此工作模型/人口不合适，不证明所有差都是真实空间bias。JOINT整体比置换低8.538m，但未越过STATIC，且场景/组件方向不一致；一次阴性差值不能单独认证时间中介。未保存训练残差曲线/训练目标对比，且嵌套模型训练误差本来不会增；不补造收敛或训练收益图。

## 4. 组件、集中度及去最大组件

下表为原件九组件全部组等权均值；C编号仅便于阅读，完整component ID来自manifest血缘，不是目标身份。

| C/组件 | 场数 | 全部组/可评组 | STATIC | TIME | JOINT | PERM | JOINT−STATIC(m) |
|---|---:|---:|---:|---:|---:|---:|---:|
| C01 `component-0b9d7d23108cbf0a74ed5828` | 1 | 4/4 | 48.308 | 1655.290 | 70.457 | 63.934 | +22.149 |
| C02 `component-23608a7a22c5322f7254ffc1` | 19 | 1322/838 | 1283.912 | 2965.014 | 1276.895 | 1289.234 | −7.017 |
| C03 `component-75391b641c3cd7287e382451` | 1 | 53/29 | 1571.983 | 3839.124 | 1617.318 | 1544.348 | +45.335 |
| C04 `component-9ea441151d6468bd21826aea` | 1 | 6/3 | 95.207 | 1567.908 | 98.956 | 102.619 | +3.749 |
| C05 `component-a077aa3eb774a8b670334e7c` | 1 | 15/10 | 44.677 | 2550.787 | 53.003 | 43.389 | +8.326 |
| C06 `component-c260c3bbf884e315f715f00b` | 1 | 6/6 | 91.962 | 2281.654 | 82.089 | 74.459 | −9.873 |
| C07 `component-c7082a697d934f0004db48b9` | 6 | 145/100 | 202.124 | 3098.121 | 236.695 | 252.649 | +34.571 |
| C08 `component-d091361fb95b75d6d2a1a9c5` | 1 | 49/32 | 1399.576 | 3337.408 | 1537.058 | 1508.327 | +137.482 |
| C09 `component-fdbc675a9f9f6a7e1c67e3bb` | 1 | 5/2 | 114.751 | 114.751 | 114.751 | 114.751 | 0 |

最大C02占838/1024＝**81.8359%**可评组；其中JOINT略有利，不能凭总体接近就称跨组件稳健。按事前方案去最大组件的敏感性保留13场/8组件、644节点、283组/186可评组/97无定义组，仍只分析本次结果、不重新拟合或从主要人口删C02：

| 臂 | 剩余186组均值(m) | RMSE(m) | P95(m) | 场景等权均值(m) |
|---|---:|---:|---:|---:|
| STATIC | 603.727 | 1196.137 | 2738.903 | 389.993 |
| TIME | 3111.269 | 3376.778 | 4432.686 | 2990.145 |
| JOINT | 653.701 | 1254.604 | 2944.324 | 457.542 |
| PERM | 645.114 | 1223.627 | 3024.657 | 465.249 |

JOINT−STATIC **+49.973967m/+58.467101m RMSE**；PERM−JOINT反为 **−8.587034m/−30.977398m RMSE**。正确速度对应相对置换的总体优势没有在这个边界延续。C06的小组件正向、C09回退同值均保留，不把小正例拼成成功路线。主要场景差已完整保存于scene-summary及六pair的scene_mean_changes；最不利如1877 +425.355m、864 +243.521m、4473 +137.482m，与700 −44.070m、3835 −23.421m、1830 −16.045m等有利场景一并存在。

本节组件与去最大组件数字来自生产summary；本地另从全部公开逐场n×mean/n×RMSE²重构全体四臂，均值/RMSE与summary最大差约6.82e−13m。它只是公开摘要的一致性重聚合，**不等于独立原数组的组件审核**；中位/P95不能由逐场分位数平均得到。

## 5. 识别条件、回退与参数稳定性

96个scene-fold中所有chart可定义，STATIC 96个均拟合。TIME/JOINT/PERM各90个数值满秩、6个因训练合格组少于2回退STATIC；每时间臂的回退涉及8个实际留出组，其τ为null，不能将回退解释为估得零时差。**本次没有数值秩不足导致的额外回退**，不能将不利结果解释为普遍缺少速度变化或程序完全没启用时间项。相关小场景：692、1293、2038、4272；692和4272的所有实际留出组都回退，故主要差为0。

- 训练折`D_v=mean(||v−vbar||²)`全96折min/median/max＝0/10.057537/64.158962 m²s⁻²；90个联合拟合折为.073561/12.359272/64.158962。
- 联合拟合90折最小设计奇异值min/median/max＝.334561/2.866791/13.477011。它们仅证明此工作矩阵通过固定数值秩条件；不同参数列有既定单位，不能转写成真实clock的可观性保证或标定置信区间。
- 四臂解析拟合总数366＝96个STATIC＋90×3个时间模型；18个模型-fold回退不算成功新时间拟合。当前fold全体组及其chart均排除在本fold拟合之外是实现/软件证据；实际保存原数组审计没有重拟合核此关系，证据层级分开。

| 时间臂 | 90拟合折的τ绝对值 min/median/max(s) | 逐场跨fold τ标准差 min/median/max(s) | 正/负拟合折 |
|---|---:|---:|---:|
| TIME | .410/98.467/4522.251 | 0/63.251/1872.444 | 32/58 |
| JOINT | .104/7.887/492.067 | 0/4.303/208.132 | 46/44 |
| PERM | .036/6.850/492.067 | 0/4.955/257.520 | 44/46 |

上表秒数是**工作回归系数**，不是已核传感器时延。JOINT在11场有已拟合fold之间的符号变化，TIME为16场、PERM为23场；这是描述性稳定性警示，不是时钟必然不存在的证明。STATIC空间系数范数min/median/max为1302.000/2689.881/4548.407m；JOINT为1402.988/2702.497/4547.780m，仍是相对工作参数，不是独立真实bias。

一次预定置换实际改变训练group出现1946/2048、留出group 932/1024。2048训练出现是1024组在其余两fold的重复，不是额外独立样本。96折留出组数为0/1/≥2的分别4/13/79；30折留出速度对应没有改变，其中包括4个空集合、13个单组及13个≥2组的随机恒等置换。训练侧12折没有改变，其中6折仅1训练组、6折为≥2组的恒等置换。按事前合同没有拒绝这些置换或重抽，负控在小集合的区分力有限；只有一次置换不构成零分布检验。

## 6. 独立保存残差核验：通过的准确范围

后补原件为[审计JSON](/Users/a1/Projects/biye-ronghe/docs/evidence/20261029_small_validation/122/independent-saved-residual-audit.json)及[完整执行代码](/Users/a1/Projects/biye-ronghe/docs/evidence/20261029_small_validation/122/independent-residual-audit-command.txt)。本报告已实际读取两者。一次审核COMPLETED/exit0，32场/4臂/1024可评组，1126项检查、其中870个数值比较，最大**统计量**绝对差0；成本.559269s、RSS36642816B，另列于主运行14.164893s之外。

审核独立用保存的组d/v、fold chart、b/τ和留出置换行号重建各组残差，核有效mask exact及数组allclose（rtol1e−9/atol1e−7）；随后重算逐场/全体group count、均值、RMSE、中位和P95，核六个配对的均值、好/差组计数及逐场均值差。逐组数组按容差通过，JSON最大差字段只汇总所比较统计量，不能扩写成每个原数组逐bit完全一致。

**未审核：** 原始support与三点差分提取、训练fold参数拟合、物理clock、组件重聚合、绝对reference误差或预测质量；六配对RMSE差/同组计数等未在该脚本逐字段单列独立比较，不称所有派生字段都已全面独立认证。它是本次结果的有限数值核验，不是第二次实验、作者复现或机制验证。运行前五项合成数学/缓存检查也只属软件证据，不算MTAD结果。

## 7. SQUEEZE_REPORT：结果解释与研究出口

| 层级 | 当前可支持内容 | 不能推出的结论 |
|---|---|---|
| DE | 四臂同1024组，联合总体均值/RMSE不如静态，8场好/22场差/2同；去最大组件后仍不利 | 不等时序、未知时间错位或连续时间状态模型普遍无效 |
| DE | 大多数fold数值满秩并实际拟合，非全部回退；时间系数跨fold有明显变化，时间only明显更差 | 数值满秩不等物理clock可辨，不能按秒系数改时间戳或借reference挑τ |
| DE | 正确速度比一次置换总体略好，但在去最大组件后反向，且仍没有超越STATIC | 不支持稳定速度中介，也不能把置换一份差值当显著性证据 |
| AI | 当前Q64组均值/一阶共享时间项增加的自由度没有带来稳定留组净收益；静态项已解释大部分当前可用共享结构 | 不能唯一归因为没有真实时差；插值/局部线性近似、组错配、运动相关空间误差及代表性均可影响 |
| UN | native时间与正确观测模型下的动态分离、不同时间权限、物理状态精度，以及原reference生成方式 | 本项没有训练完整网络、应用时间补偿或测量这些能力 |

**当前处置：** 保留本次负结果和数值条件，结束这一次一阶时间系数配方，不自动扩量、调rcond、重新置换、应用τ或叠加状态补丁。118等共享修正的实际正向证据仍按其自身报告成立；本项STATIC是不同组人口/LS目标，不能把1160m与118另一分母的相对残差或绝对位置误差直接排名。输入表征的真实保信息增量、原生状态—观测分离及其强滤波/平滑对照仍是独立候选，应由各自矩阵的证据和具体条件推进，不需要靠本项为它们背书。

```yaml
package_type: SQUEEZE_REPORT
package_version: research-pipeline-contract v1.0
package_id: SQ-SHARED-MOTION-TIME-RESIDUAL-122-20261029
squeeze_id: SQ-SHARED-MOTION-TIME-RESIDUAL-122-20261029
created_at: '2026-10-29'
input_experiment_record_ids: [EXP-SHARED-MOTION-TIME-RESIDUAL-122-20261029]
claim_scope: 原V32/9组件/固定WS/Q64工作导数的1024组相对残差；不是物理恢复或clock
data_quality_and_provenance_gaps: 581无定义组保留；metadata读含旧评价的摘要仅绑定；support差分/拟合/组件原数组审核范围未覆盖
primary_result_with_uncertainty: JOINT减STATIC均值+3.335018m/RMSE+.613038m；8场/2组件好、22场/6组件差、2场/1组件同；无独立测试或CI
process_dynamics: 一个原32终点attempt正常退出0；366解析拟合、0神经训练，14.164893s/launcher21.160754s/RSS2007040000B；有限审计另.559269s
mechanism_and_intermediate_findings: 90联合折满秩、6训练不足回退；时间系数变动，正确速度相对置换的微优势不跨组件稳定
errors_heterogeneity_and_boundary_conditions: 最大组件占81.8359%可评组；去除后JOINT均值+49.973967/RMSE+58.467101；P95总体小好而中位差
robustness_sensitivity_and_alternative_explanations: 同人口四臂/六配对/一次固定置换/组件敏感性；机动与插值、错误关联、工作模型及样本限制仍竞争解释
new_P: none；细化IT-P3，不扩大成时钟不存在的新命题
new_F: IT-F14；当前一阶速度共享项未稳定超越静态，数值满秩与目标净增量分离
new_M: IT-M7/AI；组留出下额外速度列未形成稳定增益，其数值可拟合性不等物理成因可辨
evidence_index: 本文§1原run/launcher、§2–5 summary/scene-summary、§6有限独立残差审计及代码
open_questions: 原生时间与观测角色分离的独立效果；一阶近似/错误分组对残差作用；真实clock和物理恢复未测
next_routes: deep-read Mode C；不自动后继或应用时间补偿
next_skill: deep-read Mode C
evidence_status: partial
mode_C_input_bundle:
  source_matrix_id: MC-INPUT-TIME-ACTION-20261029@followup
  source_idea_id: IDEA-SHARED-MOTION-TIME-RESIDUAL-20261029
  selected_C_id: IT-C3
  P_updates:
    - id: IT-P3
      change: 保留共享差成因待辨问题；此次具体Q64一阶时间项未提供稳定留组增量
  F_updates:
    - id: IT-F14
      evidence_level: DE
      phenomenon: 同1024留出组JOINT主要均值/RMSE较STATIC差；去最大组件不利更大；数值拟合条件多数成立仍无稳定净收益
      objects_and_conditions: 原V32/9组件/WS/三fold组等权/离线Q64三点支撑差分/单次seed20置换
      evidence_locations: 本文§2–6及122原summary/scene-summary/有限独立残差审核
      counterexamples: 总体P95小幅改善，部分场景/组件有利；正确速度总体比一次置换略好
  M_updates:
    - id: IT-M7
      evidence_level: AI
      chain: 增加一阶速度列可数值拟合且训练目标嵌套，但留出风险与跨组件稳定性未改善；不将统计拟合升级为物理时差识别
      alternatives: Q64插值导数、有限局部线性工作模型、预测组错配、运动相关空间项、重复开发与集中组件
      status: 当前操作链可记录，唯一成因UN
  E_updates:
    - id: IT-E14
      source: EXP-SHARED-MOTION-TIME-RESIDUAL-122-20261029 / SQ-SHARED-MOTION-TIME-RESIDUAL-122-20261029
      scope: 一次真实四臂中间量诊断及有限保存残差核验；0物理输出改变
  C_updates:
    - id: IT-C3
      status: 当前Q64一阶时间配方不推进物理补偿；有新增具体条件证据再另形成Idea
    - id: IT-C2
      status: 原生状态—观测分离与密度校正为独立条件路线，未由122验证或否定
  cross_matrix_targets: [IT-P3×IT-F14, IT-P2×IT-F14, SC-P1×IT-F14]
  unsupported_links: 不支持真实clock发现、绝对融合精度增量、作者TSP复现、在线端到端或可信同行胜出
  missing_evidence: [native_clock_identifiability, raw_support_derivative_and_fit_independent_audit, physical_state_increment, independent_generalization]
  next_skill: deep-read Mode C
```

以上IT编号为局部交接建议，未在本报告修改全局Mode C、研究状态或队列。实际当前任务仍只读主仓研究状态和HandOver。
