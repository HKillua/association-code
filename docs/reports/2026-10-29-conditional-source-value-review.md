# 125：T条件收益预测与冻结V32十一臂验证的真实记录与深挖

2026-10-29。EXPERIMENT_RECORD `EXP-125-20261029`＋SQUEEZE_REPORT `SQ-CONDITIONAL-SOURCE-VALUE-125-20261029`，research-pipeline-contract v1.0。来源为[执行前Idea1.0](../proposals/conditional_source_value_20261029.md)、[独立合同](../../HandOver/L2S-20261029-125.md)与Mode C1.45/CC-C1。科学运行一次attempt-1，completed/exit0；**首次通用独立审计KeyError失败的原件保留；另发的125专用保存数组审计已成功终态，限定范围见§7。** 以下分清生产结果、公开统计独立复核和尚缺的私有数组证据。

当前观察是：在固定两个恢复候选和当前reference协议内，新二次收益head在T的head-LOCO及冻结V32均有净MAE收益，超过常量、线性和置乱强控；V相对FixedLOW由447.060488降至435.061115m，降低11.999373m（2.684%）。但P95变差、5779个共同点恶化、覆盖与217个重复组均未改善，不能称整条系统已胜同行、已识别物理源质量或具备在线端到端资格。

## 1. Record：问题、计划与实际操作

124中固定借高源25%的soft相对FixedLOW有多数场景/点改善，却总体MAE更差；本项问的是：**合法预测输入能否在T估计借高源的有符号误差变化，并冻结规则迁移到V？** 主探针预先固定为二次收益回归，不按V分数择头。理论上的条件均值决策只是已有风险代数，不是本项目新定理，也不保证有限样本可估准。

- 原056教师`association/final.pt`及原normalizer冻结，FP64几何base＋FP32 residual；原θ=1.7320103112387182。T按114的signed-margin sum WS构组，身份不参与分组。教师曾在全T2100训练，θ曾在V开发。
- 127不可变`native-cache.json=CACHE_COMPLETED`提供T32/V32真实事件派生motion；物理输入与身份/reference文件分开。原生事件不是从Q64反插值伪造。本项仍沿固定离线Q64输出评价，不能据此宣因果在线。
- T为manifest前32场/12组件，V为118/121/124原32场/9组件；T_test封存。逐场先保存输入/预测再读该场T标签；全T四head冻结后逐场V动作保存，再读该场V身份/reference。T监督不是预测权限。
- hard为原完整三fold heldout共享中心下的FixedLOW，soft为**同一完整中心**下低源权重0.75/高源0.25的支持均值。低/高按源字符串排序，只是经验gauge，不是物理可靠性标签。单源/无支持域保持hard exact。
- 监督`y=e_hard−e_soft`，单位米、正数代表soft有益，不截断y。只给首anchor固定对应的首目标代表、both支持且reference位置有效query赋标签；重复组、未知/缺参考/非both不进拟合，但保留其组与输出。不能借身份拆纯组或删除难例。
- 15个合法输入：校正后/前两源均值距离、OLS速度差、两源拟合残差、nearest、native_count、支持成员数、冻结中心norm、其它fold eligible组数、组内平均signed margin及负margin比例。前13列按规格log1p及单位处理；无绝对坐标、身份、reference、真实目标support或V误差。拟合残差不是协方差，成员数不是独立有效样本数。
- 四head：T收益均值常量；线性ridge（16系数）；二次ridge（136系数）；同二次但各fit训练y以seed20一次置乱的负控。仅训练组件计算15列mean/std，z裁[-8,8]，二次展开后不再次标准化；λ=1、截距不惩罚。无标签时固定零收益→hard，本次未触发。动作均固定`predicted_gain>0 AND both`才选soft。
- T按整个组件LOCO，12fold×4，再全T拟合四head并冻结，共52次解析fit；**LOCO隔离新head及其normalizer，不隔离上游教师/历史阈值，不是全链未见组件或独立身份泛化实验。**
- V六旧强控（Mean/FixedMean/LOW/FixedLOW/Anchor/NEAR，经123原32来源121）及124固定soft75完整artifact导入，另四gate，共11臂/55配对；没有V重拟中心、normalizer、λ、动作阈值或alpha搜索，没有新分组/关联分数。

实际设计保持上述T32/V32终点，未自动扩268/全300，没有增加seed/优化步或打开封存测试。最初尚未发布的V300设想因056缓存缺native质量字段，在执行前缩为127真实字段完备的T32/V32；这是已在Idea中披露的范围选择，不能写成已验证300场。

## 2. T侧：head-LOCO是否识别了条件收益

T共有860个目标、825个首anchor代表、35个无group-anchor目标；911组×64=58304个组query。重复anchor组86个不赋训练标签；eligible训练/OOF共**22627点**，其它组query35677个仍保留输出。所有首代表matched且reference有效的共同点共**37635点**。后者仍不是所有目标reference点：无匹配目标不进入误差交集，不能把它叫无缺失全目标MAE。

eligible中y>0有15562点（68.7762%）；`|y|≤1e−6m`为0，正负分类未被这一级近零gain主导。各head使用完全相同的标签人口。以下收益均以hard作基线，**全共同点ΔMAE=−总实现收益/37635**；eligible条件均值不可冒充全共同点收益。

| T OOF head | AUROC | 选soft点/22627 | eligible实现收益均值m | 实现收益总和m | 全共同点ΔMAE m | eligible oracle regret m | gain预测MSE m² |
|---|---:|---:|---:|---:|---:|---:|---:|
| 常量TMEAN | 0.485642 | 0（0%） | 0 | 0 | 0 | 40.166724 | 80899.247242 |
| LINEAR | 0.606684 | 8043（35.5460%） | 5.473239 | 123842.976268 | -3.290633 | 34.693485 | 42721.184986 |
| QUADRATIC（预定主探针） | 0.719278 | 17323（76.5590%） | 27.083529 | 612819.000719 | -16.283220 | 13.083195 | 23496.433083 |
| SHUFFLED | 0.504996 | 183（0.8088%） | -0.012360 | -279.677900 | +0.007431 | 40.179084 | 80285.336975 |

- 常量各fold内恒定、没有选soft；合并fold的AUROC0.485642来自不同fold常数之间的排序，不能当组件内部识别能力。每个组件内部常量AUROC为0.5。正gain占多数仍可能使无条件平均gain为负，故常量不动作与正例比例不矛盾。
- LINEAR和QUADRATIC均32/32场、12/12组件净ΔMAE<0；SHUFFLED为9好/4差/19同场、1好/3差/8同组件；TMEAN全部相同。一次置乱仅是固定负控，不是精确随机化检验或多seed不确定性。
- 二次head相对hard的eligible两候选oracle regret由40.166724降至13.083195m，回收约67.43%的该oracle收益；全共同点regret由24.149129降至7.865908m。oracle需要真误差，只作离线诊断，不是可部署方法或第三个比较臂。

完整12组件保留，组件ID只为绑定公开证据；Δ均为二次head相对hard：

| T组件 | eligible点 | 全共同点 | AUROC | 全共同点ΔMAE m |
|---|---:|---:|---:|---:|
| 13bf8d8c05af98b86d432a08 | 1461 | 2812 | 0.833518 | -11.950239 |
| 1b730762e8dcc866d831d9a3 | 250 | 558 | 0.781307 | -14.223773 |
| 23608a7a22c5322f7254ffc1 | 14493 | 23340 | 0.701058 | -16.138320 |
| 2631a2fab1075a4e376f5869 | 90 | 324 | 0.595622 | -7.535583 |
| 4eb65272866a82b14a0055d6 | 124 | 389 | 0.610163 | -11.286342 |
| 5ee24adf22c874beaf8d076d | 122 | 318 | 0.723214 | -0.942002 |
| 75391b641c3cd7287e382451 | 121 | 151 | 0.679757 | -27.373389 |
| 805a0d28ca2328b9fac2c2f2 | 222 | 370 | 0.586580 | -21.731256 |
| 9ea441151d6468bd21826aea | 195 | 219 | 0.628710 | -33.433824 |
| c7082a697d934f0004db48b9 | 4825 | 7714 | 0.730212 | -18.919031 |
| d784c25f6ac4208b0a38e796 | 477 | 1108 | 0.629123 | -11.768939 |
| e062a18f3a426eba715c16d2 | 247 | 332 | 0.639705 | -27.111050 |

最大T组件占14493/22627 eligible（64.05%）；去掉后8134 eligible/14295全共同点，二次/线性/常量/置乱ΔMAE仍为−16.519806/−4.796509/0/−0.023544m。收益不是只由该组件贡献，但最小组件仅90个eligible且同轨迹query相关，不能把22627点当独立样本做显著性。T绝对hard/soft全人口MAE没有从这份OOF差值凭空补出。

## 3. V完整十一臂：相同71071位置交集，单位米

| 方法 | MAE | RMSE | P95 | P99 | 场景等权MAE |
|---|---:|---:|---:|---:|---:|
| WS_MEAN | 1276.328389 | 2168.265557 | 3103.794978 | 7194.824823 | 1249.718245 |
| WS_FIXED_MEAN | 656.041340 | 1864.654434 | 2444.380637 | 7541.587652 | 484.962267 |
| WS_LOW_SOURCE_SUPPORTED | 527.446344 | 1813.303585 | 2942.040400 | 7260.393881 | 478.072440 |
| WS_FIXED_LOW_SOURCE_SUPPORTED（hard） | 447.060488 | 1740.813819 | 1012.562755 | 7526.008427 | 382.696475 |
| WS_ANCHOR | 844.759832 | 2457.606705 | 4385.888699 | 10460.984233 | 986.493191 |
| WS_NEAREST_SUPPORTED | 927.434649 | 2173.150207 | 3873.502039 | 7301.504313 | 886.352046 |
| WS_FIXED_SOFT75_SOURCE_SUPPORTED | 531.011512 | 1769.733042 | 1507.629030 | 7539.335041 | 419.261139 |
| WS_TMEAN_VALUE_GATE | 447.060488 | 1740.813819 | 1012.562755 | 7526.008427 | 382.696475 |
| WS_LINEAR_VALUE_GATE | 444.456112 | 1740.552940 | 1011.561465 | 7526.008427 | 378.347535 |
| WS_QUADRATIC_VALUE_GATE | 435.061115 | 1739.320069 | 1014.478232 | 7526.008427 | 363.909268 |
| WS_SHUFFLED_VALUE_GATE | 448.041326 | 1741.146439 | 1029.875664 | 7526.008427 | 384.566997 |

以下差为QUADRATIC−基线，负数为误差下降。场景/组件好差使用1e−9m容差，舍入不作收益；完整55配对均在原summary及逐场原件，以下列主探针全部10对（hard与TMEAN相同合一行）：

| 基线 | ΔMAE | ΔRMSE | ΔP95 | ΔP99 | 场景MAE好/差/同 |
|---|---:|---:|---:|---:|---:|
| hard／TMEAN | -11.999373 | -1.493750 | **+1.915477** | 0 | 30/1/1 |
| LINEAR | -9.394997 | -1.232870 | **+2.916767** | 0 | 30/1/1 |
| SHUFFLED | -12.980211 | -1.826370 | -15.397432 | 0 | 32/0/0 |
| 固定soft75 | -95.950397 | -30.412973 | -493.150798 | -13.326614 | 19/13/0 |
| FixedMean | -220.980225 | -125.334365 | -1429.902405 | -15.579224 | 16/16/0 |
| LOW | -92.385229 | -73.983516 | -1927.562168 | **+265.614547** | 31/0/1 |
| Mean | -841.267274 | -428.945487 | -2089.316746 | **+331.183604** | 32/0/0 |
| Anchor | -409.698717 | -718.286636 | -3371.410467 | -2934.975806 | 32/0/0 |
| NEAREST_SUPPORTED | -492.373534 | -433.830137 | -2859.023807 | **+224.504115** | 32/0/0 |

相对hard与linear各8好/0差/1同组件，相对shuffle9/0/0，相对soft和FixedMean各6/3/0。唯一对hard/linear恶化场3835分别+0.221798/+0.218822m，4272相同。相对FixedMean仍有16场更差、相对soft仍13场更差；不能把整体平均下降写成全部场景/全部尾部胜出。

## 4. 动作、损害幅度与候选解释

全部1605组×64=102720个group-query，其中both支持41518；共同reference有效且both的首代表点39847。group动作数与首代表被评价点不是同一分母：重复、无参考和非首代表仍有合法推断。

| head | 选soft group-query | 占全部组query | 占both组query | 共同位置点好/差/同（相对hard） | 获益总和m | 损害总和m |
|---|---:|---:|---:|---:|---:|---:|
| TMEAN | 0 | 0% | 0% | 0/0/71071 | 0 | 0 |
| LINEAR | 12291 | 11.9655% | 29.6040% | 8577/3504/58990 | 250604.060279 | 65508.447361 |
| QUADRATIC | 26072 | 25.3816% | 62.7969% | 19625/5779/45667 | 1034390.256432 | 181582.795612 |
| SHUFFLED | 242 | 0.2356% | 0.5829% | 31/151/70889 | 4989.102793 | 74698.224060 |

- 二次head在共同评价点实际改变25404/39847（63.7539%）；不能将其与26072个group动作混同。相对hard净误差和−852807.460821m，除71071为−11.999373m。19625个改善点平均获益52.707784m，**5779个恶化点平均损害31.421145m**；一次正净收益并不等零风险。
- 固定always-soft的获益总和1267081.684855m、损害7233564.906877m；二次gate保留约81.64%的获益量、仅2.51%的损害量。这是对两个固定动作的事后DE误差账本，不证明源噪声/协方差/共同偏差根因，也不是所有可能融合动作的最优风险。
- V固定soft的正gain点21094/39847（52.9375%），T为68.7762%；两集合条件分布有所不同。标签群体、场景、组、中心及reference关系都可能参与，不能由此唯一归因为分布漂移或物理质量变化。
- 置乱head在T近随机、V净MAE恶化0.980838m，151个恶化点平均损害494.690226m；说明“加一个同容量head就自然变好”的解释不符本次控制。一次置乱、非独立上游和同一开发数据仍限制因果强度。

## 5. 组件、去最大组件与失败边界

九个V组件完整呈现（点数为共同位置交集）：

| V组件 | 场数 | 点数 | Q−hard ΔMAE m | Q−linear ΔMAE m | Q−soft ΔMAE m |
|---|---:|---:|---:|---:|---:|
| 0b9d7d23108cbf0a74ed5828 | 1 | 248 | -24.210822 | -19.481295 | +0.255865 |
| 23608a7a22c5322f7254ffc1 | 19 | 58707 | -10.969249 | -8.612731 | -108.863961 |
| 75391b641c3cd7287e382451 | 1 | 2406 | -10.024021 | -7.480448 | -73.436308 |
| 9ea441151d6468bd21826aea | 1 | 204 | -9.053533 | -4.222469 | -309.364129 |
| a077aa3eb774a8b670334e7c | 1 | 713 | -19.414382 | -15.981160 | +0.242103 |
| c260c3bbf884e315f715f00b | 1 | 186 | -20.950222 | -17.588213 | +0.194993 |
| c7082a697d934f0004db48b9 | 6 | 6463 | -20.165839 | -15.484508 | -7.440015 |
| d091361fb95b75d6d2a1a9c5 | 1 | 2020 | -14.255322 | -11.708514 | -49.829492 |
| fdbc675a9f9f6a7e1c67e3bb | 1 | 124 | 0 | 0 | -322.161696 |

最大组件19场/58707点（82.603%）；去掉后13场/12364点：

| 方法 | MAE m | RMSE m | P95 m | P99 m |
|---|---:|---:|---:|---:|
| hard／TMEAN | 564.276714 | 2519.443973 | 1175.447806 | 10327.828005 |
| 固定soft75 | 582.020031 | 2527.335702 | 1533.814331 | 10327.828005 |
| LINEAR | 560.495453 | 2519.192833 | 1175.447806 | 10327.828005 |
| QUADRATIC | 547.386083 | 2517.747238 | 1182.174234 | 10327.828005 |
| SHUFFLED | 565.200549 | 2519.683318 | 1185.931549 | 10327.828005 |

Q相对hard/linear的ΔMAE为−16.890631/−13.109370m，均12好/0差/1同场，7好/0差/1同组件；P95均**+6.726428m**，P99相同。相对softΔMAE−34.633949m，9好/4差场；相对FixedMean−74.622393m，7好/6差场。相对LOW/Mean/NEAR的P99仍+3.683937m。组件剔除是预定敏感性描述，不把余下组件当独立全链泛化队列。

真目标support内70027点：Q相对hardΔMAE−12.176451m，19621好/5777差/44629同。support外1044点：hard/Q/Anchor MAE分别1412.801254/1412.679482/**860.341976m**；Q只4点改善/2点恶化/1038同，ΔMAE−0.121772m、RMSE3829.273131m，P95/P99=4529.700343/12883.033335m未变。当前gate几乎没有处理这类边界，且仍明显差于Anchor。真实target-support只作事后评价，不能用于gate。far>1800s/no_input层点数0、误差为null，保留未定义，不称该类困难已解决。

## 6. 完整人口与论文坐标指标

全11臂相同1427目标/1605组/**217重复组**，1388个anchor匹配目标、39无预测、0 reference不可用目标。reference有效72196点，交集71071、缺预测1125，点coverage98.4417419%，轨迹coverage97.2669937%。55对在全局、各9组件及去最大聚合的gained/lost/rescued/lost-target均0；没有关联F1、覆盖或duplicate增益。全输出available_prediction_count102720，配对target-aligned available_prediction_count88832=1388×64，两种计数不可混用。

论文坐标诊断为scene-equal target-mean，单位1e−4 degree；ME是有符号均值，不能当MAE或径向米误差。以下完整保留11臂，不与TMTAFF论文数字直接判胜：

| 方法 | 纬度ME | 纬度MAE | 经度ME | 经度MAE |
|---|---:|---:|---:|---:|
| Mean | -29.868418 | 77.134087 | -25.865700 | 81.390119 |
| FixedMean | 3.359785 | 25.931731 | 7.246279 | 31.802728 |
| LOW | -2.614399 | 26.195071 | 0.594569 | 31.228885 |
| hard | 0.755010 | 19.710151 | 3.931008 | 25.014271 |
| Anchor | -2.246031 | 46.839205 | -0.397063 | 57.830447 |
| NEAR | -16.117392 | 52.300791 | -12.423073 | 56.953095 |
| 固定soft75 | 2.044533 | 22.016478 | 5.573597 | 27.506272 |
| TMEAN | 0.755010 | 19.710151 | 3.931008 | 25.014271 |
| LINEAR | 0.787446 | 19.470778 | 3.935449 | 24.776536 |
| QUADRATIC | 0.748832 | 18.641403 | 3.918746 | 23.913368 |
| SHUFFLED | 0.861137 | 19.910901 | 4.037577 | 25.162260 |

完整坐标绝对ME/残差SD/printed Eq29诊断与缺失分母保存在原JSON。本次是内部固定人口恢复对照，尚无可信同行复现和同协议全数据胜负证据。

## 7. 真实执行、血缘与核验边界

- 冻结代码2d6cd7b，服务器独立checkout`/home/wangguangjie/djs/association-code-run125`；脚本`tools/analysis/conditional_source_value.py`。manifest=`configs/data/mtad-level1-scene3000.json`；原run=`runs/LEVEL1-CONDITIONAL-SOURCE-VALUE-20261029-125-attempt-1`，同名`-launch`保存真实argv、环境、日志与退出。输入完整路径逐项在request/run，复现命令见独立合同，不用本报告重新生成可领取任务。
- 真实数据血缘：056冻结teacher＋114 WS＋123原32控制＋124同中心soft＋127冻结native-motion/evaluation → T合法特征/hard-soft保存 → T标签与12fold head → 全T head冻结 → V七旧artifact＋四gate保存 → V评价/55配对。坐标、身份、中心、teacher/head权重与大预测留服务器，不复制到本报告。
- 服务器22:55:05.639484至23:01:36.257227（UTC+8）实际科学运行，run/launcher/exit_code均0；服务`mtad-70b051b1ddef.service`终态inactive/dead、Result=success、ExecMainStatus=0、Restart=no。不是只有SUBMITTED或push。
- CPU0–3/4线程，GPU禁用；RSS6GiB/AS12GiB、1800s advisory未超。最终程序**390.617800518s**，launcher**396.620508115s**，进程报告peak RSS **2000748544B**（约1.863GiB）。summary写盘前389.742584225s不是最终全程；不将程序RSS当所有并行服务/主机联合峰值。
- 构造1个冻结056模型，真实teacher forward92次；52个解析head fit=39个ridge解＋13个常量均值，0无标签回退，0新增神经优化更新。不得写0模型/0前向或把缓存和拟合准备成本隐去。125成本含自身cache读/评分/特征/拟合/评价/写盘，**不含127原生读取及旧056训练等共享上游成本**，也不代表单次部署推理时延、总模型内存或端到端参数成本。
- 生产导入支路核七旧臂完整artifact exact。独立公开JSON对原121六臂及124固定soft，在全局与9组件共同字段逐叶**9450项exact、0数值差**；125省去旧低单源/高单源/无支持3个附加层的210个嵌套key，因此不能声称整个JSON schema相同。公开逐场统计重聚11臂MAE/RMSE，最大差4.55e−13m；OOF逐场32场唯一、counts与总收益重聚一致，`−gain/37635`与full ΔMAE一致。
- 上述独立工作只核**公开统计自洽**；AUROC/分位数、私有y/head参数/动作生成、特征与reference隔离并未由读JSON独立认证。第一次冻结`independent-saved-array-audit.json`实际`FAILED/exit1/KeyError`：5项exact、**0项numeric**，0.524386s、RSS93306880B；其scope仍带旧120/121/123/124描述，不能借此认证125。根因是旧auditor读取不存在的completed_scenes，125正式字段为completed_V_scenes；失败记录不改写。新审计另按123声明的原32谱系读取121保存控制。原科学run不重跑。
- **新专用独立审计已完成**：冻结46902b3，服务mtad-78122f4f4933.service真实success/exit0、inactive/dead、Restart=no；输出independent-value-array-audit.json，32场/11臂55对/9组件及去最大共11聚合，88781 numeric、381477 exact、41404 null通过，七旧控224场景臂artifact exact。逐组/首代表/重复/完整reference registry/validity及同点误差、覆盖0变化独立重算；MAE/RMSE最大差1.916e-9m，P95 2.556e-9m，P99 2.094e-9m，MSE4.530e-6m²。程序17.626684s、launcher18.429599s、RSS150552576B另计。未核T拟合/teacher/head特征或动作合法性、support语义、论文属性/speed/heading、逐点利害和reference物理正确性；不能把主位置审计扩大到所有科学机制。原科学及审计冻结worktree均clean。

**后到的T侧独立补证（UTC15:32:25–15:32:59）：** 冻结814affd，新输出independent-T-array-audit.json及单独T-audit-launcher，原科学125不重跑。3598exact/1808numeric通过，独立核32场/12组件、首anchor重复与eligible22627/全matched-refvalid37635两分母、球面hard−soft gain、48fold＋4fullT保存头及T-only normalizer/λ1参数；39个ridge与13个常量仅为审计参数重解，不保存新模型或择参。独立gain最大差4.906e-9m、符号不一致0，mean/std/beta和OOF全部收益指标差0；0teacher前向/0V payload。程序33.270113s、launcher33.724041s、RSS140509184B，服务终态success/exit0。未核teacher生成、特征构造/合法性、WS/中心、文件打开时序或reference物理语义；LOCO依然只隔离新head。该补证晚于131发布，不倒写为131启动前已完成。

| 计划/核验项 | 实际事实 | 对解释的影响 |
|---|---|---|
| T32/12组件、V32/9组件、四head/11臂55对 | 按一次合同完成 | 未覆盖V余268/全300或新增组件分布 |
| 训练冻结与0神经更新 | 1teacher、92前向、52解析fit，参数/阈值未搜索 | 有监督探针，不能称完全无训练/无需标签 |
| 不改预测人口、缺失不删 | 同1427/1605/217及1125缺失点 | 位置收益不等系统覆盖/重复改进 |
| 完整both/单源/no-support层 | 本125直接导出both，旧3层额外key未带入 | 保留旧121/124层证据；新gate域外exact约束不能冒充新独立分层审核 |
| 独立保存数组审核 | 首次KeyError保留；另发专用主位置/人口审计通过 | 限定32场11臂55对及旧控绑定，随后T参数/OOF补证通过；teacher/特征/动作路径未完整认证 |
| 重复/不确定性 | 单次固定seed20置乱，没有多seed/外部队列或置信区间 | 场景/组件与去最大是描述稳健性，不是显著性或总体保证 |

## 8. Squeeze：支持什么、没有回答什么

**DE（生产结果＋公开统计复核）：** 当前有监督的条件贡献探针具有实际可用信号。二次head在新head-LOCO优于常量/线性/置乱，固定到V后净MAE仍更低，改善分布到8/9组件，去最大组件仍保留；不是只把124中“多数点改善”换一个说法。它回答的是两个已固定恢复动作之间、当前reference风险差的可预测性。

**负面及限制：** 对hard的P95+1.915477m、去最大P95+6.726428m；5779点恶化；对Mean/LOW/NEAR的P99更差；13场不及固定soft、16场不及FixedMean；support外基本无改善且不如Anchor；39无预测目标/1125缺失点/217重复组均未改变。当前125只有位置通道恢复变化，不能称条件B或关联/输出增量。

**AI（可发展的解释）：** 借用收益可能依赖两源不一致、运动拟合状态和组内证据的交互；同容量置乱不利、二次胜线性与此相容。尚不能从相关特征归因哪一列、何种交互或物理根因。组错误、共享偏差、机动、reference生成关系、缺标签选择、上游teacher/阈值历史均是未排除解释。

**UN：** 更完整V及真正隔离上游的泛化，论文协议下可信同行收益，物理源方差/协方差或可靠性识别，轻量化总链成本，严格在线/端到端训练与推理，均没有在本项证明。136系数不等整条模型只有136参数；不因已有回归/门控先例否定方向，新增价值应通过明确适配和进一步可归因收益成立。

主位置/人口核验通过后，将窄结论回流Mode C；下一步对更广人口、未改善尾部/边界与组件隔离确定独立Idea/预算，不从V反调λ、alpha、动作阈值或重选特征。若核验不符，保全原件、定位差异并下调相应结论，不能自动重跑科学配方。本报告不发布新实验。

## 9. 原件与标准交接

[必要脱敏原件](/Users/a1/Projects/biye-ronghe/docs/evidence/20261029_small_validation/125/)包含run、summary、scene-summary.json.gz、T-oof、T-fold-index、同launcher请求/日志/退出、真实服务终态及第一次audit失败、新独立audit与其launcher。私有原件留服务器原125；不把公开JSON检查说成私有数组认证。

```yaml
package_type: EXPERIMENT_RECORD
package_version: research-pipeline-contract v1.0
experiment_id: EXP-125-20261029
source_idea_id: IDEA-CONDITIONAL-SOURCE-VALUE-20261029
plan_version: '1.0'
execution_status: completed_pilot_exit0
research_question: T合法输入是否能预测固定hard/soft的有符号收益并冻结到V
hypothesis: 条件收益可学；物理质量或根因仍AI/UN
planned_vs_actual_design_and_controls: T32/12fold_new_head_LOCO+fullT_four_heads,V32/11arms/55pairs_once
actual_configuration: frozen2d6cd7b,CPU4,RSS6GiB,AS12GiB,1800s_advisory,seed20_shuffle,lambda1
samples_materials_or_data_provenance: frozen056+114+123_original32_from121+124_soft75+127_CACHE_COMPLETED
results_with_uncertainty: T_full_common_delta_Q=-16.283220m;V_MAE447.060488->435.061115m;single_development_run_no_IID_CI
all_runs_or_repetitions: one_scientific_attempt_exit0;first_independent_audit_FAILED_KeyError_numeric0
observations_and_deviations: V_P95_worse,harm5779,coverage_and_duplicates_unchanged,old3_extra_strata_omitted
actual_cost: program390.617800518s,launcher396.620508115s,peakRSS2000748544B,teacher1_forward92,analytic52_neural_updates0
known_limits_and_missing_evidence: main_position_population_and_T_parameter_OOF_audits_passed_feature_teacher_unverified;LOCO_new_head_only;V32_not300;physical_quality_and_credible_peer_unverified
next_routes: squeeze_and_ModeC_then_frozen_broader_Idea
next_skill: squeeze-experiment
```

```yaml
package_type: SQUEEZE_REPORT
package_version: research-pipeline-contract v1.0
squeeze_id: SQ-CONDITIONAL-SOURCE-VALUE-125-20261029
input_experiment_record_ids: [EXP-125-20261029]
evidence_status: partial_main_position_population_and_T_parameter_OOF_audits_passed_feature_teacher_unverified
claim_scope: frozen_two_action_reference_risk_predictability_on_T_head_LOCO_and_original_V32
primary_result_with_uncertainty: Q_vs_hard_V_deltaMAE=-11.999373m,deltaRMSE=-1.493750m,deltaP95=+1.915477m,deltaP99=0;30good1bad1tie_scenes,8good0bad1tie_components
process_dynamics: 390.6178s_program,52_analytic_fits,92_teacher_forwards,0_neural_updates,advisory_not_exceeded
mechanism_and_intermediate_findings: nonlinear_conditional_signal_compatible_AI;no_feature_or_physical_root_cause_identified
errors_heterogeneity_and_boundary_conditions: harm5779;outside_support1044_points_near_unchanged;217duplicates_and1125missing_points_unchanged;P99_worse_than_Mean_LOW_NEAR
robustness_sensitivity_and_alternative_explanations: T12_component_LOCO_new_head_only,V9_components_and_without_largest,constant_linear_shuffle_controls;shared_upstream_and_reference_relationship_remain
new_P: 如何将可预测的两动作贡献收益扩展到完整人口并控制尾部且独立于上游开发历史
new_F: 当前合法条件收益head在T和冻结V有净MAE增量，但不改善覆盖重复且部分尾部恶化
new_M: AI条件交互可能有用；UN具体特征因果/物理可靠性
open_questions: feature_teacher_and_action_generation;broader_and_upstream_isolated_transfer;tail_control;online_total_cost
next_routes: preserve_failed_audit_and_passed_separate_audit;deep-read_ModeC_before_broader_Idea
next_skill: deep-read Mode C
```
