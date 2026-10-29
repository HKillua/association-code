# 121：源优先对共享校正收益的实际归因

2026-10-29。原V32的十臂、45配对一次完成退出0。固定同71071点，未校正的低source支持优先（LOW）MAE527.446m，已经低于原118固定校正均值（FixedMean）的656.041m；低source优先后复用同一校正（FixedLOW）进一步达到447.060m。因此**本次不支持“校正后的双源均值比低源优先有额外精度价值”这一主预测**，但支持共享校正在低源不覆盖等条件下仍有独立用途。该经验源优先规则尚无全V结果、物理无偏源识别或可信同行胜出证据。

先按record-experiment记录真实输入、运行与全部结果，再按squeeze-experiment分析来源、作用域、尾部与异质性。公开汇总独立重聚合与主要保存数组独立审计均通过各自声明范围，source路由/支持语义及物理解释仍有未审条件。本报告不启动后继或更改121原配方。

## 1. EXPERIMENT_RECORD：设计、执行和原件

| 字段 | 实际事实 |
|---|---|
| ID／上游 | `LEVEL1-SOURCE-PREFERENCE-20261029-121-attempt-1`；`IDEA-SOURCE-PREFERENCE-CONTROL-20261029@1.0`；融合局部矩阵`MC-FUSION-SOURCE-PREFERENCE-20261029 / SC-C3` |
| 研究问题／主估计量 | 固定同一校正后，保留高source的均值是否优于低source支持优先；主估计量为**FixedMean−FixedLOW**的径向MAE差，越负才支持这一主预测 |
| 实际设计 | 118原V32/9共享身份组件，原八臂保存预测直接导入；仅新算LOW、FixedLOW两臂，十臂全部45个无序配对。原32终点，无扩量、训练或重拟合 |
| 血缘 | 056system原Q64物理状态/support与缓存reference→114保存WS membership→118保存八臂完整artifact及校正中心/source顺序/fold→121仅计算两个支持优先控制。原八臂逐字段exact重建绑定，中心/Huber均不重算 |
| 新控制 | 低source在query有支持则仅均值该源；否则均值高source支持；全无支持回退同组首节点。FixedLOW先按118保存fold中心只校正高source，再走同一规则；速度/航向继承原WS_MEAN |
| 信息与人口 | 预测函数只接合法物理输入与保存中心，不接身份/reference/真实target支持；冻结十臂后绑定评价。1427 target、91328 target-query、72196参考有效位置点；71071共同可评点、1125缺预测点、19132缺参考query；全部target均有部分参考 |
| 分组／覆盖 | WS1605组、217重复组、1388首代表匹配target、39未匹配target；全部十臂不变。位置点coverage98.441742%、目标coverage97.266994%；group-query102720，不与target可评点混用 |
| 命令／冻结代码 | cwd=`/home/wangguangjie/djs/association-code-run121`，启动冻结`1322dc9`（主任务启动核验；run.json本身不含Git字段）；实际argv为`/usr/bin/taskset --cpu-list 4-7 /home/wangguangjie/djs/mtad-field-20261016/bin/python tools/analysis/source_preference_control.py`及原056/114/118、manifest、新输出、cpu_threads=4，完整见request原件 |
| 程序终态／时间 | run COMPLETED、exit_code0；UTC12:41:28.480516→12:43:06.111135，北京时间20:41:28→20:43:06；程序97.630659s。summary在finalizing前取97.100015s，两截点保留不混用 |
| launcher／服务 | launcher COMPLETED、returncode0、exit_code.txt=0；UTC12:41:21.702308→12:43:07.020079，elapsed105.318386s。启动`mtad-3a6295de9482.service`；本地服务终态原件明确inactive/dead、Result=success、ExecMainStatus0、Restart=no、Linger=yes |
| 资源 | CPU4/affinity4–7；RSS2007138304B（1914.156MiB）、AS8589934592B；600s advisory未超。CUDA_VISIBLE_DEVICES为空，0GPU/模型/前向/优化/新score/分组/normalizer/阈值/ZIP/CSV/测试读取；cached_reference_read=true，仅绑定评价 |
| 次数／偏差 | 唯一attempt-1，256个原场景臂导入、64个新场景臂，32场无排除或替换。全无support回退先于实施已明确为相应输入上的同一首节点，FixedLOW跟随校正后输入；这是实施前澄清，未在看到结果后换配方。stdout仅完成路径，stderr与worker.stderr均0B |
| 未记录项 | run未保存Python/NumPy/Torch确切版本，不以本机版本补填。source排序的物理可靠性、reference生成和绝对真值语义未知；没有训练收敛曲线或重复seed可分析 |

事前[Idea](../proposals/source_preference_control_20261029.md)、[实际实现](../../tools/analysis/source_preference_control.py)、[融合上游矩阵](/Users/a1/Projects/biye-ronghe/docs/surveys/2026-10-27_expanded_search/fusion_action_matrix_20261029.md#6-118回流后的下一层问题相对校正绝对锚定与真正多源增量)。原件：[run](/Users/a1/Projects/biye-ronghe/docs/evidence/20261029_small_validation/121/run.json)、[summary](/Users/a1/Projects/biye-ronghe/docs/evidence/20261029_small_validation/121/summary.json)、[全部32场](/Users/a1/Projects/biye-ronghe/docs/evidence/20261029_small_validation/121/scene-summary.json.gz)、[命令](/Users/a1/Projects/biye-ronghe/docs/evidence/20261029_small_validation/121/launcher-request.json)、[服务提交](/Users/a1/Projects/biye-ronghe/docs/evidence/20261029_small_validation/121/launcher-launch.json)、[launcher终态](/Users/a1/Projects/biye-ronghe/docs/evidence/20261029_small_validation/121/launcher-status.json)、[真实服务终态](/Users/a1/Projects/biye-ronghe/docs/evidence/20261029_small_validation/121/launcher-posthoc-service-terminal.txt)、[退出码](/Users/a1/Projects/biye-ronghe/docs/evidence/20261029_small_validation/121/launcher-exit_code.txt)。私有源顺序、中心、身份、坐标和全部预测仍留服务器原run/private。

### 1.1 全部十臂主结果

简称仅用于本报告：M=MEAN，H=HUBER，FM/FH=FIXED_MEAN/HUBER，SM/SH=SYMMETRIC_MEAN/HUBER，A=ANCHOR，N=NEAREST_SUPPORTED，L=LOW_SOURCE_SUPPORTED，FL=FIXED_LOW_SOURCE_SUPPORTED；实际方法名均有`WS_`前缀。所有行同一71071点，单位米。

| 方法 | MAE | RMSE | P95 | P99 |
|---|---:|---:|---:|---:|
| M | 1276.328 | 2168.266 | 3103.795 | 7194.825 |
| H | 1275.620 | 2168.558 | 3114.053 | 7194.825 |
| FM | 656.041 | 1864.654 | 2444.381 | 7541.588 |
| FH | 656.374 | 1866.778 | 2455.881 | 7541.588 |
| SM | 1716.225 | 2300.417 | 2390.411 | 7008.621 |
| SH | 1717.793 | 2301.627 | 2395.856 | 7008.621 |
| A | 844.760 | 2457.607 | 4385.889 | 10460.984 |
| N | 927.435 | 2173.150 | 3873.502 | 7301.504 |
| L | 527.446 | 1813.304 | 2942.040 | 7260.394 |
| FL | 447.060 | 1740.814 | 1012.563 | 7526.008 |

原八臂没有新增预测或再拟合；其值与118完整公开主结果一致。L较M的MAE−748.882m、RMSE−354.962m；FL较L再−80.386m、−72.490m。FM−FL的事前主MAE差为**+208.980852m**，方向与主预测相反。程序键`WS_FIXED_LOW_SOURCE_SUPPORTED_vs_WS_FIXED_MEAN`保存的是FL−FM=−208.980852m，不能漏掉反号。

### 1.2 全部45配对与不利方向

以下一律candidate−baseline，MAE/RMSE/P95/P99下降有利；场景、组件为MAE好/差/同，容差1e−9m。各配对均71071共同点，gained/lost/rescued/lost-target全部0。P95/P99为两误差向量分位数之差，不是逐点差的分位数。表不把同次45配对当45次独立实验。

| baseline → candidate | ΔMAE | ΔRMSE | ΔP95 | ΔP99 | 场景好/差/同 | 组件好/差/同 |
|---|---:|---:|---:|---:|---:|---:|
| M -> H | -0.709 | 0.292 | 10.258 | 0.000 | 11/21/0 | 3/6/0 |
| M -> FM | -620.287 | -303.611 | -659.414 | 346.763 | 31/0/1 | 8/0/1 |
| M -> FH | -619.955 | -301.488 | -647.914 | 346.763 | 32/0/0 | 9/0/0 |
| M -> SM | 439.897 | 132.151 | -713.384 | -186.204 | 2/30/0 | 0/9/0 |
| M -> SH | 441.465 | 133.361 | -707.939 | -186.204 | 2/30/0 | 0/9/0 |
| M -> A | -431.569 | 289.341 | 1282.094 | 3266.159 | 26/6/0 | 5/4/0 |
| M -> N | -348.894 | 4.885 | 769.707 | 106.679 | 32/0/0 | 9/0/0 |
| M -> L | -748.882 | -354.962 | -161.755 | 65.569 | 32/0/0 | 9/0/0 |
| M -> FL | -829.268 | -427.452 | -2091.232 | 331.184 | 32/0/0 | 9/0/0 |
| H -> FM | -619.578 | -303.903 | -669.672 | 346.763 | 31/1/0 | 8/1/0 |
| H -> FH | -619.246 | -301.780 | -658.172 | 346.763 | 31/0/1 | 8/0/1 |
| H -> SM | 440.605 | 131.859 | -723.642 | -186.204 | 2/30/0 | 0/9/0 |
| H -> SH | 442.173 | 133.069 | -718.197 | -186.204 | 2/30/0 | 0/9/0 |
| H -> A | -430.860 | 289.049 | 1271.836 | 3266.159 | 26/6/0 | 5/4/0 |
| H -> N | -348.185 | 4.592 | 759.449 | 106.679 | 32/0/0 | 9/0/0 |
| H -> L | -748.173 | -355.254 | -172.013 | 65.569 | 32/0/0 | 9/0/0 |
| H -> FL | -828.559 | -427.744 | -2101.490 | 331.184 | 32/0/0 | 9/0/0 |
| FM -> FH | 0.332 | 2.123 | 11.500 | 0.000 | 11/7/14 | 4/2/3 |
| FM -> SM | 1060.184 | 435.762 | -53.970 | -532.967 | 0/32/0 | 0/9/0 |
| FM -> SH | 1061.752 | 436.972 | -48.524 | -532.967 | 0/32/0 | 0/9/0 |
| FM -> A | 188.718 | 592.952 | 1941.508 | 2919.397 | 4/28/0 | 1/8/0 |
| FM -> N | 271.393 | 308.496 | 1429.121 | -240.083 | 4/28/0 | 2/7/0 |
| FM -> L | -128.595 | -51.351 | 497.660 | -281.194 | 9/23/0 | 4/5/0 |
| FM -> FL | -208.981 | -123.841 | -1431.818 | -15.579 | 12/20/0 | 5/4/0 |
| FH -> SM | 1059.851 | 433.639 | -65.470 | -532.967 | 0/32/0 | 0/9/0 |
| FH -> SH | 1061.419 | 434.849 | -60.025 | -532.967 | 0/32/0 | 0/9/0 |
| FH -> A | 188.386 | 590.829 | 1930.008 | 2919.397 | 4/28/0 | 1/8/0 |
| FH -> N | 271.061 | 306.373 | 1417.621 | -240.083 | 4/28/0 | 2/7/0 |
| FH -> L | -128.927 | -53.474 | 486.159 | -281.194 | 9/23/0 | 4/5/0 |
| FH -> FL | -209.313 | -125.964 | -1443.318 | -15.579 | 12/20/0 | 5/4/0 |
| SM -> SH | 1.568 | 1.210 | 5.445 | 0.000 | 9/9/14 | 3/3/3 |
| SM -> A | -871.465 | 157.190 | 1995.478 | 3452.364 | 28/4/0 | 6/3/0 |
| SM -> N | -788.790 | -127.267 | 1483.091 | 292.884 | 32/0/0 | 9/0/0 |
| SM -> L | -1188.779 | -487.113 | 551.630 | 251.773 | 32/0/0 | 9/0/0 |
| SM -> FL | -1269.164 | -559.603 | -1377.848 | 517.388 | 32/0/0 | 9/0/0 |
| SH -> A | -873.033 | 155.980 | 1990.033 | 3452.364 | 28/4/0 | 6/3/0 |
| SH -> N | -790.358 | -128.477 | 1477.646 | 292.884 | 32/0/0 | 9/0/0 |
| SH -> L | -1190.347 | -488.323 | 546.184 | 251.773 | 32/0/0 | 9/0/0 |
| SH -> FL | -1270.733 | -560.813 | -1383.293 | 517.388 | 32/0/0 | 9/0/0 |
| A -> N | 82.675 | -284.456 | -512.387 | -3159.480 | 14/18/0 | 5/4/0 |
| A -> L | -317.313 | -644.303 | -1443.848 | -3200.590 | 30/2/0 | 8/1/0 |
| A -> FL | -397.699 | -716.793 | -3373.326 | -2934.976 | 32/0/0 | 9/0/0 |
| N -> L | -399.988 | -359.847 | -931.462 | -41.110 | 32/0/0 | 9/0/0 |
| N -> FL | -480.374 | -432.336 | -2860.939 | 224.504 | 32/0/0 | 9/0/0 |
| L -> FL | -80.386 | -72.490 | -1929.478 | 265.615 | 29/0/3 | 7/0/2 |

FL虽然整体优于FM，**仅12场好、20场差；5组件好、4组件差**。因此“更低全点加权MAE”不意味着逐场或逐组件占优。相反FL对A/N的场景MAE在本32均好，也不能据此给未见场景保证。

## 2. SQUEEZE_REPORT：校正、源优先和作用域

### 2.1 主2×2说明什么

固定原M1276.328、FM656.041、L527.446、FL447.060m，得到：

| 干预或交互 | 同人口径向MAE差/m | 有边界的解释 |
|---|---:|---|
| FM−M | −620.287049 | 118固定校正本身仍是真实内部增量，但基线未含源优先 |
| L−M | −748.882045 | 不做校正，只按输入source支持优先已解释很大改善 |
| FL−L | −80.385856 | 同源优先下校正仍有附加作用；不能说校正完全无效 |
| FL−FM | −208.980852 | 校正后的均值未超过更强的源优先控制；事前反方向结果 |
| `(FL−L)−(FM−M)` | +539.901193 | 校正效应依贡献结构改变，不能把两种主效应简单相加 |

沿“M→L→FL”这条明确顺序，L解释了总MAE下降的90.3064%，其余为同顺序再加校正的差；这是代数分解，不是可识别的物理因果贡献比例，也不证明该源无偏。换路径分解不同，交互本身表明不能从一个2×2给各模块普遍固定贡献。

从实现可见：低源有support时FL不给高源权重，所以“FL更好”并不支持双源都有报告时保留高源的额外价值。它可能形成一个有效的源条件融合解法：有低源支撑用低源，高源在低源缺席时经过同一共享校正后回退。是否能学到不依赖数字源序的质量/锚定条件、是否适用于其他源排序仍UN。

### 2.2 预测组支持类型与完整目标支撑分开

输入作用率按全部1605预测组×64计：双方有support41518、仅低26562、仅高11335、全无23305，总102720。L→FL位置发生变化的group-query14327；这些含重复组及无有效reference时刻，不能作为可评target点数。按固定首代表映射且reference可评后，四类型点数为39847/25194/3260/2770，总71071。

下表四列分别M/FM/L/FL的MAE与RMSE，米；预测组类型来自合法输入，不由真实target支撑门控。

| 预测组support类型 | 可评target点 | M MAE/RMSE | FM MAE/RMSE | L MAE/RMSE | FL MAE/RMSE |
|---|---:|---:|---:|---:|---:|
| both_sources | 39847 | 1560.647/1662.255 | 597.680/1019.169 | 224.943/492.263 | 224.943/492.263 |
| low_only | 25194 | 213.940/515.119 | 213.940/515.119 | 213.940/515.119 | 213.940/515.119 |
| high_only | 3260 | 3512.820/3966.701 | 1775.593/3194.026 | 3512.820/3966.701 | 1775.593/3194.026 |
| no_support | 2770 | 4216.985/7742.476 | 4199.027/7736.103 | 4216.985/7742.476 | 4199.027/7736.103 |

双方有support时L与FL相同到此汇总精度，FM的MAE597.680m反而高于二者224.943m；这是主反方向效果的主要发生域。仅高源支持时校正把3512.820→1775.593m，L与M、FL与FM在汇总精度下分别相同，支持高源回退中的附加价值。全无support仍约4200m，校正只微小改善；不能用整体均值掩盖域外高风险。

target完整身份支撑是另一个事后评价层：支撑内70027点、支撑外1044点。FL支撑内MAE432.663/RMSE1690.267m，支撑外1412.801/3829.284m；支撑外虽优于M1627.407/3898.870、N1626.465/3901.317，仍**差于A860.342/3460.636m**。far_gt_1800s与no_input均0共同可评点，不能说这些层有效。预测组无support2770点与真实target支撑外1044点不是同一集合，错组可使两种支持语义不同。

### 2.3 尾部、点级恶化与场景完整性

- L对M：32/32场、9/9组件MAE好，仍有702个共同点误差更大，P99+65.569m。
- FL对M：32/32场、9/9组件MAE好，仍有1317点更大，P99+331.184m。
- FL对L：29好/0差/3同场，7好/0差/2同组件；2640点更好、624点更差、67807同。P95−1929.478m，但P99**+265.615m**，不能以较低RMSE解释为全部尾部风险已解。
- FL对N：总体MAE/RMSE/P95较低，但P99+224.504m。FL对FM的P99低15.579m，不代表对所有强控制尾部更好。
- L对A恶化场为2576(+26.931m)、3156(+6.792m)；FL对A在32场MAE均好。L对FM有23场MAE差，全部幅度见公开派生JSON，不能把较低全点均值说成普遍占优。

FL对FM的**全部20个MAE恶化场**（米，FL−FM）为：692(+38.325)、1187(+1.066)、1582(+33.170)、1840(+3.175)、1877(+14.315)、2153(+27.550)、2262(+34.885)、2352(+45.506)、2416(+35.010)、2479(+45.956)、2576(+31.926)、3274(+27.911)、3619(+39.467)、3908(+43.399)、3916(+25.042)、3956(+46.888)、3981(+26.571)、4316(+22.803)、4393(+33.087)、4968(+3.153)。反面场全部保留，不按reference拼接两臂冠军。

### 2.4 九组件与最大组件外的事后敏感性

下表各值为MAE，米；全部九组件列明，完整RMSE/尾部/45比较在原summary中。

| 组件前缀 | 场景 | 点数 | M | FM | L | FL |
|---|---:|---:|---:|---:|---:|---:|
| 0b9d7d23 | 1 | 248 | 821.917 | 155.301 | 287.685 | 190.310 |
| 23608a7a | 19 | 58707 | 1241.122 | 663.209 | 494.452 | 422.374 |
| 75391b64 | 1 | 2406 | 1446.795 | 627.947 | 717.377 | 473.457 |
| 9ea44115 | 1 | 204 | 2782.578 | 2222.062 | 1560.540 | 1540.993 |
| a077aa3e | 1 | 713 | 997.787 | 345.056 | 433.472 | 372.967 |
| c260c3bb | 1 | 186 | 922.333 | 435.164 | 460.206 | 460.206 |
| c7082a69 | 6 | 6463 | 1498.914 | 605.342 | 712.518 | 612.613 |
| d091361f | 1 | 2020 | 1424.401 | 644.255 | 636.087 | 548.073 |
| fdbc675a | 1 | 124 | 1186.850 | 1186.850 | 468.324 | 468.324 |

FL对FM恶化的四组件是0b9d7d23(+35.010)、a077aa3e(+27.911)、c260c3bb(+25.042)、c7082a69(+7.271m)。最大组件23608a7a有19/32场、58707/71071点（82.603%），因此点加权风险与“好场数量”可以反向。

事后仅读公开逐场`N×MAE/MSE`，保留主要人口另列“最大组件外”13场/12364点：M1443.496/2861.164，FM622.008/2555.913，L684.111/2609.221，FL564.277/2519.444m（MAE/RMSE）。FL的总体主要方向并非只来自最大组件；但**L对FM在这一子集反而更差**，所以“未校正源优先已经足够”也不成立。此敏感性不是新运行、事前主检验或独立泛化；不由逐场分位数拼造子集P95/P99。

### 2.5 论文坐标统计完整保留，不能用有符号ME宣布胜出

以下经/纬均为10⁻⁴度，按目标时间→场景内目标等权→场景等权。SD为本项目标准残差样本SD，**不是已核作者mESD**；印刷Eq29诊断和absolute_ME等完整字段在原summary保留，不用较好者替换MAE。每臂1427 declared targets，1388有ME/MAE、1385有至少两点可算SD，各32场有定义。

| 方法 | 经度ME | 经度MAE | 经度SD | 纬度ME | 纬度MAE | 纬度SD |
|---|---:|---:|---:|---:|---:|---:|
| M | -25.866 | 81.390 | 46.241 | -29.868 | 77.134 | 39.234 |
| H | -25.826 | 81.350 | 46.272 | -29.809 | 77.113 | 39.256 |
| FM | 7.246 | 31.803 | 33.793 | 3.360 | 25.932 | 25.297 |
| FH | 7.243 | 31.788 | 33.817 | 3.385 | 25.941 | 25.327 |
| SM | -33.345 | 104.450 | 33.793 | -37.363 | 100.494 | 25.297 |
| SH | -33.348 | 104.464 | 33.816 | -37.337 | 100.537 | 25.327 |
| A | -0.397 | 57.830 | 71.328 | -2.246 | 46.839 | 57.507 |
| N | -12.423 | 56.953 | 79.187 | -16.117 | 52.301 | 73.261 |
| L | 0.595 | 31.229 | 46.832 | -2.614 | 26.195 | 40.176 |
| FL | 3.931 | 25.014 | 34.726 | 0.755 | 19.710 | 26.988 |

本次仅位置新控制；速度/航向继承WS_MEAN，不能作为新学习四属性成绩，也没有新增训练、关联raw/decoded改善或TMTAFF完整复现。坐标有符号误差靠近0不代替米制精度、尾部、覆盖和可信同行共同任务比较。

## 3. 核验范围与尚未核实的条件

1. **运行内绑定：** 32/32原118八臂全部artifact字段exact；十臂全部target/reference/组/首代表/valid/coverage/duplicate不变，45配对全部覆盖转移0。保存中心和source顺序/fold复用，不重新拟合。真实预测构造入口没有身份/reference参数。
2. **独立公开重聚合：** [只读派生脚本](/Users/a1/Projects/biye-ronghe/docs/evidence/20261029_small_validation/121/derive-public-review.py)、[派生结果](/Users/a1/Projects/biye-ronghe/docs/evidence/20261029_small_validation/121/public-review-derived.json)核6147个数值/离散项：10方法×全32及9组件的主要人口、N加权MAE/MSE/RMSE和固定分层、8旧臂与118公开position/population/paper、45配对逐场差/benefit−harm/计数及支持作用率。最大绝对算术差3.7253e−8（不同量纲中最大，主要为累加和），容差rtol1e−9/atol1e−7；不把这说成原数组独立审计，也不独立重算P95/P99。
3. **独立保存数组审计：通过声明范围。** [实际审计原件](/Users/a1/Projects/biye-ronghe/docs/evidence/20261029_small_validation/121/independent-saved-array-audit.json)一次exit0，32场/9组件/10臂/每场45配对、全32及9组件共10汇总人口；194480 exact、49320 numeric、140 null检查。独立重算主要位置/人口/共同点评价，最大MAE/RMSE差2.153e−9m、P95差2.315e−9m、P99差2.153e−9m、MSE差6.855e−6m²、mean_error_change差2.457e−10m；coverage差0，容差rtol1e−9、米atol1e−5/MSE atol.001/rate atol1e−12。耗时9.395888s、RSS146599936B，另属事后审计成本，不混主程序成本。审计直接读取121保存的全部artifact，原件的`imported118_scene_count=0`是审计读取路径口径，不改变本研究原八臂256个场景臂从118导入的血缘。

   **审计准备偏差另记：** 据主任务现场记录，最初通过MCP upload投递主仓审计脚本被allowed-local-path仅含资料库的限制拒绝；随后一次启动壳因脚本不存在退出，未进入audit程序、未读取数组或创建审计JSON，不能当研究重跑或第二个真实审计结果。之后正常经主仓commit `ae7ba4f`→GitHub→服务器独立`association-code-audit120121` checkout同步，首个真正读取121数组的审计即上述9.395888s、exit0的一次。实际入口为`/usr/bin/taskset --cpu-list 4-7 /home/wangguangjie/djs/mtad-field-20261016/bin/python /home/wangguangjie/djs/association-code-audit120121/tools/analysis/audit_shared_control_arrays.py --run /home/wangguangjie/djs/association-code/runs/LEVEL1-SOURCE-PREFERENCE-20261029-121-attempt-1 --output /home/wangguangjie/djs/association-code/runs/LEVEL1-SOURCE-PREFERENCE-20261029-121-attempt-1/independent-saved-array-audit.json`，execute_command返回0。此准备事件不改变研究121原run或配方；原审计JSON未收录这些前置壳事件，证据来源明确为主任务操作记录。
4. **未独立核范围：** 中心拟合/fold/Huber机制、manifest/真实组件身份、reference物理正确性、输入support/时间/observed语义、支持条件分层与source-priority路由、paper/速度/航向、逐点benefit/harm整数等均在审计`not_audited`中明确排除。公开汇总及主要数组一致不能将这些自动升级为已验证；米量级作用域由运算和公开分层支持，未宣非both层所有浮点状态逐值exact。
5. **计划偏差与边界：** 事前回退澄清不是运行后改参；没有V择源、重新选fold或删场。当前只固定原32重复开发，场间共享身份且V已反复查看；无新seed/训练/收敛数据，不做点IID CI或显著性。source排序是经验锚定，不能称匿名源置换不变或识别物理健康源。
6. **未获结论：** reference生成/唯一真实时钟与绝对无偏锚点、source语义跨数据迁移、真正观测噪声/相关矩、全V新源优先效果、端到端可训练或严格在线均未知。0参数/0GPU不代表整套离线输入和聚类搜索已满足部署延迟。

## 4. 由事实到后续价值：保留简单有效点，收窄理论主张

118的共享项收益是真实的，但121表明普通Anchor/NEAR还不构成充分的源质量控制。把低源按query支持优先纳入后，校正双源均值的额外优势没有出现；更合理的本次解释是：经验源偏好改善双方可见域，共享校正在低源缺席的回退域另有用途。该分工有直接算子和分层依据，仍不是物理bias已被唯一识别。

值得继续的是“在合法输入下，何时保留哪类来源、何时利用校正后的其他来源，以及支撑外怎样避免尾部”这一具体问题。可发展不依赖数字源序的质量/锚定信息、状态—观测分工或更强可信baseline；已有成熟源选择与配准并不否定适配价值。不能先加大网络包装成新理论，也不能因源优先很简单就丢掉其真实效果。新方案须先回矩阵、形成独立Idea；本次原32到终点结束，不自动扩量或训练。

## 5. 标准Record→Squeeze与Mode C输入

```yaml
package_type: EXPERIMENT_RECORD
package_version: research-pipeline-contract v1.0
package_id: EXP-SOURCE-PREFERENCE-121-20261029
experiment_id: LEVEL1-SOURCE-PREFERENCE-20261029-121-attempt-1
created_at: '2026-10-29'
source_ids: [IDEA-SOURCE-PREFERENCE-CONTROL-20261029]
source_idea_id: IDEA-SOURCE-PREFERENCE-CONTROL-20261029
plan_version: '1.0'
execution_status: completed-exploratory
research_question_and_hypothesis: 同冻结校正下双源均值是否优于低source支持优先；主FixedMean减FixedLOW
planned_vs_actual_design_and_controls: 原32十臂45配对；旧八臂256场景臂导入，仅64新场景臂；中心不重拟合；实施前回退澄清见正文
actual_configuration: 冻结1322dc9，CPU4/affinity4-7/RSS2007138304B/AS8GiB，程序97.630659s/launcher105.318386s，exit0，0训练
samples_materials_or_data_provenance: 056Q64物理输入与缓存reference、114WS、118保存artifact/中心/源顺序/fold；全部32/9组件/1427target
all_runs_or_repetitions: 一个attempt-1；无排除/重跑/扩量/V择参；全部十臂非十次独立重复
results_with_uncertainty: 主FixedMean减FixedLOW为+208.980852m，未支持主预测；LOW527.446/1813.304、FixedLOW447.060/1740.814m；全部负场/组件/P99保留
observations_and_deviations: 源优先解释大部顺序增益，校正对LOW仍有80.386m；FL对FM20场差，P99较LOW高265.615m；不是物理无偏源识别
evidence_locations: 本报告§1全部run/launcher/summary/32场原件；§3公开派生脚本与核验
known_limits_and_missing_evidence: 主要数组审计通过声明范围，source路由/支持/物理机制等未审；环境精确版本/物理reference/全V源优先/可信同行/独立泛化未知
evidence_status: partial
missing_evidence: [independent_source_routing_audit, lawful_absolute_anchor, matched_credible_peer, fullV_source_preference_effect, independent_generalization]
next_routes: [squeeze-experiment]
next_skill: squeeze-experiment
```

```yaml
package_type: SQUEEZE_REPORT
package_version: research-pipeline-contract v1.0
package_id: SQ-SOURCE-PREFERENCE-121-20261029
squeeze_id: SQ-SOURCE-PREFERENCE-121-20261029
created_at: '2026-10-29'
source_ids: [EXP-SOURCE-PREFERENCE-121-20261029]
input_experiment_record_ids: [EXP-SOURCE-PREFERENCE-121-20261029]
claim_scope: 原V32同WS/71071点的共享校正与query源优先内部归因，不是学习/同行/全V新源优先结果
data_quality_and_provenance_gaps: 旧八臂真实exact绑定，公开6147项重聚合与主要保存数组49320数值检查通过，source路由/支持语义及物理条件等未审
primary_result_with_uncertainty: 主FixedMean减FixedLOW为+208.980852m；不支持双源均值额外价值，LOW较Mean降748.882m、FixedLOW较LOW再降80.386m；无IID显著性
process_dynamics: 一个固定32终点、0训练，程序97.630659s/launcher105.318386s，RSS1914.156MiB；无收敛或seed轨迹
mechanism_and_intermediate_findings: 双方支持域的源优先解释主要差异；仅高源回退中校正仍有作用；交互+539.901m使两主效应不可简单相加
errors_heterogeneity_and_boundary_conditions: FL对FM12好20差场/5好4差组件；FL对LOW624点恶化/P99+265.615m；真实target支撑外仍输Anchor；全部负面见§2
robustness_sensitivity_and_alternative_explanations: 最大组件外13场仍FL优于FM/LOW，但LOW不再优FM；事后敏感性非独立泛化；source顺序/reference生成解释未排除
new_P: none；细化既有共享分量、锚定与来源贡献粒度问题
new_F: SP-F1：同固定校正下源优先超过双源均值且跨场有反向异质性；SP-F2：校正对源优先仍有回退域增量但P99损失
new_M: SP-M1：源支持条件决定校正作用域，算子DE与分层支持；物理噪声/无偏源因果解释UN
evidence_index: 本报告§1–3、121真实原件、公开派生结果和独立主要数组审计；未审范围明列
open_questions: 合法绝对锚定和非数字源序质量条件；支撑外/尾部；强可信同行；全V源优先及独立泛化
mode_C_input_bundle: 本报告下一表SP-P/F/M/E；由根矩阵统一编号，不把重复118结果再计新证据
evidence_status: partial
missing_evidence: [independent_source_routing_audit, lawful_absolute_anchor, matched_credible_peer, fullV_source_preference_effect, independent_generalization]
next_routes: [deep-read Mode C增量]
next_skill: deep-read Mode C
```

| 局部项 | 回流内容 | 证据与边界 |
|---|---|---|
| SP-P | 不增全局P；细化已有“经验锚定、跨源共享项与局部贡献”的可辨性 | 不将一个内部控制新包装为无人识别的领域问题 |
| SP-F1 | LOW已胜FixedMean全点MAE/RMSE，FixedLOW再降低；主FixedMean−FixedLOW=+208.981m，反驳本轮双源均值额外价值预测 | 同71071点，十臂45配对；原32开发，不代表全V或同行 |
| SP-F2 | FL对LOW MAE−80.386但P99+265.615m；FL对FM20场/4组件差，最大组件外主要均值仍好；支撑外输Anchor | 正反与全部人口/组件同时保留 |
| SP-M1 | 源支持条件决定是否引入高源，校正与来源贡献结构交互；双方支持时低源优先更好，仅高源时校正有用 | 算子和分层DE；物理bias/源健康/absolute gauge解释AI或UN |
| SP-E1 | 121一次真实COMPLETED/exit0、输入/预测/人口绑定、全部10×32场/45配对/9组件与实际成本；公开6147项独立重聚合及主要数组49320数值/194480 exact核验通过 | 数组审计限明示范围；source路由/支持/物理机制等仍未独立核，不以数值一致替代物理真实性 |
