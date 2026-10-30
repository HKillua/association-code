# 140：平方风险评分与集合几何读出 — Record → Squeeze

记录日期2026-10-30。**科学与独立数组审计均真实COMPLETED/exit0；13个解析拟合、T32/V32固定终点完成，四主比较实际损害补项已从保存数组独立核出。** 新平方风险hard未超过旧Qhard；同一新评分下，geometry也未超过hard或inverse。几何代理求解准确不等于实际物理误差改善，不能只报相对源等权的收益。

## Record：问题、实际设计与执行

- package_type：EXPERIMENT_RECORD；package_version：research-pipeline-contract v1.0。
- experiment_id / package_id：LEVEL1-SOURCE-CONTRIBUTION-READOUT-20261030-140-attempt-1。
- source_idea_id：IDEA-SOURCE-CONTRIBUTION-READOUT-20261030；plan_version1.0；[事前Idea/spec](../proposals/source_contribution_readout_20261030.md)、[L2S140](../../HandOver/L2S-20261030-140.md)。
- source_matrix_id：MC-SELECTION-READOUT-20261030-v1.0；selected_C_id：SR-C2，见[定向24格矩阵](/Users/a1/Projects/biye-ronghe/docs/surveys/2026-10-27_expanded_search/selection_readout_matrix_20261030.md)。
- 研究类型：computational pilot；执行状态completed，科学解释exploratory。一次科学attempt，未重训teacher或端到端网络。

研究问题分两层：A→A′，把旧log径向误差评分换成直接弦长平方风险评分并共同clip，是否改善完整误差；A′→A′+B，同一评分下利用候选间几何的集合读出，是否超越hard与普通inverse。不能把A′的新监督拟合隐去，写成“只改变融合权重”。139的米制目标是另一独立实验，本报告不拼其赢家。

固定操作为原10维源特征、共享66系数二次ridge、训练权1/K、折内weighted mean/std、zclip8、非截距λ1；12组件留一＋fullT共13fit。响应为`[2R sin(e/(2R))/1000]^2`（km²），来自原T真实径向标签，**不是旧Q逆变换平方**。这是风险值回归，不是直接训练最终位置MSE。三个新臂共同使用`max(raw,0)`；CHARD取最小风险、exact平局均分；CINV倒数权重floor=1e-6km²；CGEO最小化`r·w−0.5wᵀDw`。D是3D球面嵌入端点弦长平方，真实K≤2闭式；泛K算法仅软件验证。

最终仍将源权重分给本源支持原节点，用原纬度/圆周经度恢复；速度航向保持Mean，零支持回退原Mean。3D代理的未归一化向量与此物理恢复不同。旧T的源等权/Qsoft/Qhard为本次从原OOF评分**新恢复**，不是导入历史T artifact；V保留十旧臂exact对照。

### 输入、命令、终点及成本

| 项目 | 实际记录 |
|---|---|
| 冻结代码与目录 | `65366cb`；`/home/wangguangjie/djs/association-code-run140`。Git只定位历史，不替代数值验收 |
| 数据血缘 | 原127 native-motion/evaluation → 原125 T组、114 V WS → 原133特征/标签/OOF与V十臂 → 13新risk头 → 同头三种贡献 → 原节点恢复 → 固定人口评价 |
| 原输入 | native127、value125、partition114、source-risk133的原run绝对路径逐项在run/request保存；manifest固定原run136的`configs/data/mtad-level1-scene3000.json` |
| 科学run / launcher | `LEVEL1-SOURCE-CONTRIBUTION-READOUT-20261030-140-attempt-1` / 同父`…-140-launch-attempt-1`，原件不覆盖 |
| 资源 | CPU4/taskset12–15；OMP/OpenBLAS/MKL/NumExpr4；RSS6GiB/AS12GiB；1800s advisory；CUDA_VISIBLE_DEVICES空，0GPU |
| 科学时间 | UTC03:04:46.439330 → 03:08:18.986509，即北京时间11:04:46 → 11:08:18 |
| 实测 | 程序212.547231942s；launcher232.535009129s；峰值RSS1183793152B；未超时/建议、无清理杀进程 |
| 分阶段 | T缓存绑定3.124s、13fit25.942s、T预测评价40.811s、V预测评价141.401s；含缓存/恢复/评价/IO，不是单query在线时延 |
| 真实服务 | `mtad-b3551f823123.service`；观察UTC03:10:53，inactive/dead、Result=success、ExecMainStatus=0、MainPID=0、Restart=no |
| 完成范围 | T训练缓存32、T评价32、V评价32；13拟合/13线性系统solve、0无标签回退；T6臂15对、V13臂78对 |
| 未发生的操作 | 0teacher/神经模型/optimizer/native-events/grid；未读ZIP/CSV/封存test，未新分组/时序重建/配准；fullT模型先冻结再V |

实际payload（完整env/cwd见request）为：

```bash
/usr/bin/taskset --cpu-list 12-15 /home/wangguangjie/djs/mtad-field-20261016/bin/python tools/analysis/source_contribution_readout_pilot.py --native-run /home/wangguangjie/djs/association-code/runs/LEVEL1-NATIVE-CV-STATE-20261029-127-attempt-1 --value-run /home/wangguangjie/djs/association-code/runs/LEVEL1-CONDITIONAL-SOURCE-VALUE-20261029-125-attempt-1 --partition-run /home/wangguangjie/djs/association-code/runs/LEVEL1-STATIC-PARTITION-PILOT-20261029-114-attempt-1 --source-risk-run /home/wangguangjie/djs/association-code/runs/LEVEL1-SOURCE-RISK-20261030-133-attempt-1 --manifest /home/wangguangjie/djs/association-code-run136/configs/data/mtad-level1-scene3000.json --output /home/wangguangjie/djs/association-code/runs/LEVEL1-SOURCE-CONTRIBUTION-READOUT-20261030-140-attempt-1 --time-policy advisory --cpu-threads 4
```

[run原件](/Users/a1/Projects/biye-ronghe/docs/evidence/20261030_source_contribution/140/run.json)、[request](/Users/a1/Projects/biye-ronghe/docs/evidence/20261030_source_contribution/140/launcher/request.json)、[launcher退出](/Users/a1/Projects/biye-ronghe/docs/evidence/20261030_source_contribution/140/launcher/status.json)、[服务终态](/Users/a1/Projects/biye-ronghe/docs/evidence/20261030_source_contribution/140/observed-service-terminal.json)、[13折清单](/Users/a1/Projects/biye-ronghe/docs/evidence/20261030_source_contribution/140/T-fold-index.json)。原身份、坐标、β、权重与预测只留服务器private。

### 人口与不确定性

| 固定人口 | T | V |
|---|---:|---:|
| 场景/组件 | 32/12 | 32/9 |
| 目标 / 预测组 / 重复组 / 首代表匹配目标 | 860 / 911 / 86 / 825 | 1427 / 1605 / 217 / 1388 |
| 参考可评位置 / 完整共同可评位置 / 缺输出 | 38224 / 37635 / 589 | 72196 / 71071 / 1125 |
| 未匹配目标 | 35 | 39 |
| 点覆盖 / 目标覆盖 | 98.4591% / 95.9302% | 98.4417% / 97.2670% |
| 完整组query中0/1/2+可用源 | 17771 / 16930 / 23603 | 23305 / 37897 / 41518 |
| 首代表、reference可评的多源query | 22627 | 39847 |
| 其中源级标签 / 权重和 | 45254 / 22627 | 79694 / 39847（仅预测后评价，非训练） |

完整组query（T58304/V102720）、共同位置点（T37635/V71071）与多源标签query是不同分母。原首代表、无支持、重复/未匹配均保留，各臂人口及覆盖固定。原V十臂为已有内部强控，LOW/FixedLOW不恢复成最终固定源策略。T与V共享4个组件，覆盖V的67780/71071＝95.37%可评点；上游teacher已用T且V多轮开发，LOCO仅隔离本轮critic，不能称独立泛化或可信同行比较。

主要估计量是全共同人口位置径向MAE，单位m；RMSE/P95/P99、scene macro、逐场/组件及去最大一起报告。四主差分别在T/V以2000次seed20组件bootstrap给描述性95%区间，未校正跨比较选择，不解释成独立测试显著性。所有方法仅一次既定解析拟合流程，没有新seed重复。

## Record：完整结果

表中CHARD/CINV/CGEO为同一个新弦长平方risk critic的hard/inverse/geometry；均非论文作者方法。以下T为组件留出预测，V为fullT模型预测。

### T固定37635点

| 方法 | MAE m | RMSE m | P95 m | P99 m | 场宏MAE m |
|---|---:|---:|---:|---:|---:|
| 源等权 | 1311.223 | 2878.589 | 2544.797 | 10480.821 | 1371.921 |
| 旧 Qsoft | 1175.045 | 2810.466 | 2434.711 | 10406.995 | 1227.567 |
| 旧 Qhard | 619.476 | 2666.191 | 2436.283 | 10500.827 | 643.082 |
| CHARD | 831.652 | 2773.122 | 3388.605 | 10416.891 | 983.984 |
| CINV | 1123.908 | 2813.357 | 2891.423 | 10372.524 | 1176.791 |
| CGEO | 1131.533 | 2817.394 | 2826.235 | 10406.995 | 1217.067 |

### V固定71071点：全部十三臂

| 方法 | MAE m | RMSE m | P95 m | P99 m | 场宏MAE m |
|---|---:|---:|---:|---:|---:|
| 原 Mean | 1276.328 | 2168.266 | 3103.795 | 7194.825 | 1249.718 |
| nearest-node | 927.435 | 2173.150 | 3873.502 | 7301.504 | 886.352 |
| 历史 LOW | 527.446 | 1813.304 | 2942.040 | 7260.394 | 478.072 |
| 历史 FixedLOW | 447.060 | 1740.814 | 1012.563 | 7526.008 | 382.696 |
| 历史 Qgate | 435.061 | 1739.320 | 1014.478 | 7526.008 | 363.909 |
| 源等权 | 1276.441 | 2168.137 | 3097.435 | 7194.825 | 1248.763 |
| 旧 Qsoft | 1128.755 | 2060.605 | 2990.606 | 7194.825 | 1106.865 |
| 旧 Qhard | 547.721 | 1829.356 | 3125.980 | 7260.394 | 505.468 |
| 旧 linear-soft | 1162.880 | 2085.495 | 3011.683 | 7194.825 | 1132.549 |
| 旧 shuffled-soft | 1276.493 | 2168.187 | 3101.807 | 7194.825 | 1248.680 |
| CHARD | 973.953 | 2145.131 | 3716.808 | 7261.257 | 1031.585 |
| CINV | 1068.388 | 2109.304 | 3500.770 | 7261.257 | 1098.538 |
| CGEO | 1151.619 | 2118.915 | 3313.962 | 7194.825 | 1163.918 |

CHARD相对旧Qhard，V MAE恶化426.232m、RMSE恶化315.775m、P95恶化590.827m，P99也略差。CGEO相对CHARD，MAE恶化177.666m，但RMSE降低26.216m、P95降低402.845m、P99降低66.432m；相对CINV，CGEO MAE与RMSE均差、P95/P99较低。这是明确的目标/尾部取舍，不能用P99局部改善声称整体净胜。三个新臂MAE均低于源等权，但都未超过旧Qhard和历史LOW/FixedLOW/Qgate；CGEO连旧Qsoft的1128.755m也未超过。

### 四项预定主要比较：差为candidate−baseline，负值才是MAE改善

| 分区 | 比较 | 差 m | 组件bootstrap95% m | 改善/恶化/持平场 | 改善/恶化/持平组件 | 去最大组件差 m |
|---|---|---:|---|---|---|---:|
| T | CHARD−旧 Qhard | 212.177 | [69.924, 516.645] | 1/30/1 | 0/12/0 | 492.637 |
| T | CINV−CHARD | 292.256 | [-5.569, 435.571] | 7/25/0 | 3/9/0 | 6.159 |
| T | CGEO−CHARD | 299.881 | [53.277, 408.460] | 2/30/0 | 0/12/0 | 81.692 |
| T | CGEO−CINV | 7.624 | [-28.037, 97.631] | 12/20/0 | 2/10/0 | 75.533 |
| V | CHARD−旧 Qhard | 426.232 | [406.370, 562.480] | 0/32/0 | 0/9/0 | 517.619 |
| V | CINV−CHARD | 94.435 | [33.635, 101.730] | 8/24/0 | 3/6/0 | 55.279 |
| V | CGEO−CHARD | 177.666 | [110.641, 186.556] | 6/26/0 | 2/7/0 | 138.222 |
| V | CGEO−CINV | 83.230 | [70.076, 101.577] | 7/25/0 | 2/7/0 | 82.943 |

最大组件以场数定义，为两分区共有的`component-23608…`。去除此组件，T保留14295点、V12364点；四主差的点估计仍都为正。V新hard相对旧hard在32场/9组件全部恶化；geometry相对同risk hard在26场/7组件恶化、相对inverse在25场/7组件恶化。T inverse−hard与geometry−inverse区间跨0，不能将V方向硬套成每个T组件一致的结论。

### 实际物理误差损害补证

事前要求实际物理输出的正害/负益及>1km、最大损害；生产公共pairwise只存前者，>1km字段原来仅在端点选择regret。按明确的只读审计补充合同，已在`additional_actual_harm`补齐：独立大圆误差的`candidate−baseline`，完整共同target/query人口，strict `>1000m`（恰等1000属于`(0,1000]`）。正负号按float64实际差值，不设epsilon死区；0差仍记平局。**这里的损害是对表内基线增加的实际误差，端点regret则相对reference条件最佳单源，二者基线和输出对象都不同。**

| 分区/完整点数 | 比较 | 正害 / 负益 / 平局点 | 正害和 / 负益和（10⁶m） | 实际损害>1km点数 | >1km损害和（10⁶m） | 最大实际损害 m | 去最大组件后>1km点数 |
|---|---|---|---:|---:|---:|---:|---:|
| T / 37635 | CHARD−旧Qhard | 4902 / 304 / 32429 | 8.517536 / 0.532271 | 3540 | 7.533372 | 9885.517 | 3002 |
| T / 37635 | CINV−CHARD | 16159 / 4176 / 17300 | 13.237766 / 2.238704 | 4644 | 6.025432 | 3379.054 | 666 |
| T / 37635 | CGEO−CHARD | 15728 / 2503 / 19404 | 13.223031 / 1.937028 | 5079 | 6.967956 | 3929.475 | 1136 |
| T / 37635 | CGEO−CINV | 8820 / 11515 / 17300 | 2.625144 / 2.338202 | 673 | 0.975171 | 2775.228 | 673 |
| V / 71071 | CHARD−旧Qhard | 15375 / 649 / 55047 | 30.754491 / 0.461776 | 12415 | 28.972282 | 9685.410 | 2881 |
| V / 71071 | CINV−CHARD | 17904 / 14045 / 39122 | 12.560202 / 5.848599 | 6095 | 8.334060 | 3642.196 | 927 |
| V / 71071 | CGEO−CHARD | 20232 / 7256 / 43583 | 20.033987 / 7.407122 | 10037 | 13.990219 | 3655.874 | 1481 |
| V / 71071 | CGEO−CINV | 19168 / 12781 / 39122 | 10.749223 / 4.833961 | 4137 | 5.777978 | 3615.130 | 765 |

V的CGEO−CHARD中，最大组件承担8556/10037个>1km损害点；去除此组件仍有1481个，故不能将损害简单归结为一个大组件。V的CHARD−旧Qhard在去最大后仍有2881个>1km损害点。T的CGEO−CINV共有673个>1km损害点，全部在最大组件之外；这与其去最大MAE差由+7.624扩大为+75.533m相容，但不是物理原因已查明。

完整逐场/组件与去最大统计全部在[独立审计原件](/Users/a1/Projects/biye-ronghe/docs/evidence/20261030_source_contribution/140/independent-audit.json)的`additional_actual_harm`，没有只存最坏/最好几场。统计仅消费原保存预测与reference，不生成新状态、不拟合模型、不改阈值；它完成原计划漏写的公共诊断，不是第二次科学实验。其成本计入下述独审总成本，未单独测时，不宣额外成本为零。

### 评分、clip、floor与集合代理

以下数值按各场实际分母加权汇总公共诊断，未把场均值直接平均；分位数没有可加性，因此不拼接源槽位分位数。所有单位km²/km⁴如实保留。

| 诊断 | T（OOF） | V（fullT预测） |
|---|---:|---:|
| 可用source-query分母 | 64136 | 120933 |
| raw负值＝共同clip至0数 | 10503（16.38%） | 60756（50.24%） |
| 负风险均值 / 最小值 km² | −9.247 / −58.142 | −12.768 / −64.730 |
| 多源reference可评源级分母 | 45254 | 79694 |
| 此域raw负值 / inverse floor命中 | 6990 / 6990 | 28483 / 28483 |
| weighted raw风险回归平方残差 km⁴ | 46753.044 | 196.256 |
| weighted clipped风险回归平方残差 km⁴ | 46731.540 | 169.804 |
| 有支持组query分母 | 40533 | 79415 |
| CHARD / CINV / CGEO的F<0组query数 | 2487 / 4706 / 4773 | 8238 / 22046 / 22611 |
| CGEO最小F km² | −33.625 | −20.946 |
| CGEO最大duality gap km² | 2.8422e−14 | 2.8422e−14 |
| CGEO求解未收敛数 / 最大迭代 | 0 / 0 | 0 / 0 |

全部实际K≤2使用闭式，因此0迭代不是未求解。hard/inverse的F gap是集合代理的次优性上界证书，并非精确的目标值差，**不是其算法收敛失败**。CINV实际floor命中与全部raw负数相同，表明本批floor并非“没起作用”；没有扫描floor或与其他floor比较，不能推出不敏感。raw共同clip会创造风险平局；新hard已经包含平局均分的物理混合，不能将所有hard结果解释为纯单源结果。

**AI代数边界：** 在单纯形`sum(w)=1`上，对读出使用的同一风险向量加共同常数，有`F(w;r+c·1)=F(w;r)+c`，故geometry和hard的最优动作不变，inverse一般会改变。因此负F说明其数值不能直接解释为已校准的非负期望平方误差，**负F本身不能证明动作为什么变差**；geometry决策所需的正确条件风险差允许一个共同偏移，不以绝对风险逐值准确作为必要门槛。此不变性针对进入读出的r；若在`max(raw,0)`之前移动raw，截断关系可能改变，不代表整条流水线不变。本轮仍未分离相对风险误差、clip、条件信息及实际圆周恢复的作用，实际负结果不因此改变。

| 同critic贡献（有支持组query加权） | T平均最大源权重 | V平均最大源权重 | T熵 | V熵 | T距等权L1 | V距等权L1 |
|---|---:|---:|---:|---:|---:|---:|
| CHARD | .969321 | .948133 | .042530 | .071903 | .520958 | .419064 |
| CINV | .799388 | .862774 | .336498 | .222127 | .181092 | .248345 |
| CGEO | .813282 | .826558 | .325039 | .289000 | .208879 | .175914 |

V的CGEO相对CINV更靠近源等权，T的关系相反；这只是实际权重分布，不能直接宣布“混得更多导致变差”。风险回归残差是km⁴，与最终位置RMSE并非同指标；T/V标签构成不同，不能据46753对196反推训练发散、过拟合或V泛化更好。

### 端点排序诊断：不能代替实际融合误差

| 分区及源端点规则 | 多源query数 | 期望端点误差 m | 端点选择regret m | 最优端点credit | regret>1km点数 | >1km regret和 m |
|---|---:|---:|---:|---:|---:|---:|
| T旧Qhard | 22627 | 317.099 | 47.264 | .973395 | 471 | 1002904.527 |
| T新chord hard | 22627 | 679.748 | 409.913 | .816502 | 3944 | 8259798.985 |
| V旧Qhard | 39847 | 261.105 | 42.774 | .971742 | 665 | 1496185.620 |
| V新chord hard | 39847 | 1038.174 | 819.843 | .686375 | 13245 | 30827310.346 |

exact平局下这里是按权重的**端点误差期望/credit**，实际输出却是先混合状态再算误差；两者不相等。上述计数不是strict误选总数，credit包含平局分数，也不能将`1-credit`直接换成严格错误次数。新评分在T/V端点损害均显著更大，已能说明“先有合法平方风险标签”不保证形成可用排序，但目标、clip、特征表达和有限ridge误差尚未分离归因。

### 支持域与已有状态问题分开

| 可评位置分层 | T点数 | CHARD / CINV / CGEO MAE m | V点数 | CHARD / CINV / CGEO MAE m |
|---|---:|---|---:|---|
| 预测组零可用源 | 1487 | 6197.185 / 6197.185 / 6197.185 | 2770 | 4216.985 / 4216.985 / 4216.985 |
| 预测组单可用源 | 13521 | 512.073 / 512.073 / 512.073 | 28454 | 591.896 / 591.896 / 591.896 |
| 预测组多可用源 | 22627 | 670.008 / 1156.112 / 1168.793 | 39847 | 1021.331 / 1189.765 / 1338.215 |

此表为生产公开固定分层的加总；独审主范围核完整/observed/missing及所有保存源支持、权重和物理输出，**不将全部附加分层汇总冒称逐项独审**。0/1源处三个新臂保持同输出，差异集中在有选择/混合机会的多源域；140没有改变137的CV状态或本轮142的传播机制。真实identity各片段span并集和预测组支持仍为不同口径，不能把二者混为原始事件命中。

完整数值来源：[T汇总](/Users/a1/Projects/biye-ronghe/docs/evidence/20261030_source_contribution/140/T-summary.json)、[V汇总](/Users/a1/Projects/biye-ronghe/docs/evidence/20261030_source_contribution/140/summary.json)、[T逐场](/Users/a1/Projects/biye-ronghe/docs/evidence/20261030_source_contribution/140/T-scene-summary.json.gz)、[V逐场](/Users/a1/Projects/biye-ronghe/docs/evidence/20261030_source_contribution/140/scene-summary.json.gz)。

### 独立数组核验与只读补充的实际成本

原科学冻结`65366cb`保持不变；在独立audit checkout使用`347c118`（后继只读统计版本），按[L2S140审计补充合同](../../HandOver/L2S-20261030-140.md)一次运行。既有软件scope之外明确增加四主差actual_harm，未改变原方法或源产物。科学结果先完成，真实审计再从源private保存数组独立重建，而不是将人工fixture验收充当真实核验。

| 核验项 | 实际结果与范围 |
|---|---|
| 终态 | COMPLETED/exit0；428212 exact / 113722 numeric / 511 null检查通过 |
| 标签与拟合 | 原源分组/支持/10维特征、45254个T源标签、13个独立normalizer/66系数ridge及LOCO/full评分；原折名单和隔离一致 |
| 贡献及恢复 | raw/common clip、真实K≤2的hard/inverse/geometry权重、F与gap；原node纬度/圆周经度恢复、speed/heading exact、T旧三控制新恢复与V十旧臂exact |
| 人口与比较 | 完整T32/12组件/6臂15对、V32/9组件/13臂78对；全人口/主指标/组件/去最大；T/V各4个主要bootstrap |
| 实际损害 | 四个预定主比较T/V各完整共同人口、正负差与strict>1km/max、32场/各组件/去最大；与源端点regret分开 |
| 时间 | UTC03:13:45.789239 → 03:15:06.653752；程序80.865101557s，launcher81.747019317s；新增统计包含在总审计时间内 |
| 资源 | CPU4/taskset4–7，峰值RSS286355456B，0GPU；1800s advisory未超，原6GiB/12GiB预算未改变 |
| 真实服务 | `mtad-fdaf324af845.service`；UTC03:23:24观察inactive/dead、success、ExecMainStatus=0、MainPID=0、Restart=no |

审计请求指向原140/127/133及同一个原run136绝对manifest，输出为原140下首次`independent-audit.json`；没有修改原science checkout。参见[审计请求](/Users/a1/Projects/biye-ronghe/docs/evidence/20261030_source_contribution/140/audit-launcher/request.json)、[审计launcher退出](/Users/a1/Projects/biye-ronghe/docs/evidence/20261030_source_contribution/140/audit-launcher/status.json)、[审计服务终态](/Users/a1/Projects/biye-ronghe/docs/evidence/20261030_source_contribution/140/observed-audit-terminal.json)。

### 偏差、独立核验与限制

- 计划的13fit、原T/V32、T6/V13、15/78比较、8主bootstrap均已真实完成并通过上述限定独立数组核验；零额外操作由元数据和代码路径支持，审计不认证所有运行文件访问顺序。未用软件fixture或提交回执替代真实科学结果。
- 实际混合>1km与最大害未由冻结生产写入公共表，按原计划在独立审计中补算并单独留痕，不覆盖源结果。不把补证当新方法或第2次科学attempt。
- 本轮A→A′同时含目标从log径向到chord²及共用非负clip，不是只改变一个相同量纲的loss。新头的有限投影、负值、clip和几何条件信息不足需分开讨论。
- 实际只有K≤2，泛K置换/求解仅软件检查；未验证真实K≥3、换传感器、在线或全链联合训练。位置外的speed/heading保持原Mean，不能称四状态都获得改善。
- 独立审计不认证reference物理正确、来源真实health/Cov、上游teacher、原127 OLS正确性、运行文件访问顺序或可信同行净胜；这些保持原证据边界。

## Squeeze：两层失败分别保留，不把所有问题归为一个原因

- package_type：SQUEEZE_REPORT；package_version：research-pipeline-contract v1.0。
- package_id：SQ-SOURCE-CONTRIBUTION-READOUT-20261030-140；input_record为上文140实际run；source_idea_id同上。
- 分析范围：完整生产公开结果的四主比较、全部强控、端点/贡献/求解诊断与异质性，并消费已完成的限定独立核验和实际损害。无新服务器科学运行、重拟合或搜索。

| 层级 | 直接观察DE | 可以支持什么 | 竞争解释／未知UN |
|---|---|---|---|
| 风险评分A′ | CHARD较旧Qhard在V全部32场/9组件更差；T也30场/12组件更差，端点credit下降、regret增大 | 当前直接chord²回归＋clip未建立优于原log排序的A′；不能把集合失败全归后端读出 | 极端风险响应支配、同10维投影不足、clip平局及参考/错组构成尚未分离；没有消融即不认定某一个根因 |
| 同评分读出B | CGEO相对hard/inverse的V四指标存在取舍，完整MAE两者都未超；去最大仍差 | 在本次明确A′上，集合几何项没有满足事前增量条件 | 风险与几何条件不相容、代理目标/真实圆周恢复失配、实际混合空间小仍竞争；不能单凭一次失败否定成熟集合优化 |
| 代理求解 | CGEO gap约2.84e−14、无未收敛，但F大量负值 | 数值终点可精确而物理风险仍差；负F不能读成已校准期望平方误差 | 不能据gap证明r是真条件矩，也不能仅据负F唯一指认相关噪声/偏差；失配可来自风险估计、clip或条件信息 |
| 异质性 | V geometry优于hard的6场/2组件保留，T geometry−inverse区间跨0；V geometry−hard有10037个实际增害>1km点，去最大仍1481个 | 不删局部收益，也不据局部选择配方；总体和高害问题不只由最大组件产生 | 未完成真实逐点几何代理与物理恢复差的联合诊断，不能将某子群命名为已知机制 |

### 与原矩阵的具体关系及下一个可区分问题

**SR-C2的A′和B都没有达到本配方的事前晋级条件。** 这一结论是本次critic/clip/读出的有限结论；保留成熟集合目标的合理迁移价值。数学恒等式对同一3D对象成立，实验证据否定的是“当前估计risk与实际恢复组合可以获得净增量”的具体预期，不是否定恒等式本身。

现在最小可发展的分离问题有三项，均需另立具体Idea才能执行，不能在本报告自动调参：

1. **评分目标与clip分离：** 先在T已有缓存区分raw排序、clip造成的平局和高代价错误各占什么作用域；与139米制目标证据独立对照后，才决定是否需要改表达、约束或损失。不能见负值就自动改floor/加网络，也不能把原速度/航向当独立健康证据。
2. **真实几何条件与风险相容：** 利用已保存r、D与真实endpoint误差识别哪些T点的F失配和高害共存，区分“单源风险估计已经错”与“同评分下集合项增加损害”。这是关联诊断，若要机制归因还需独立干预；不预定相关误差根因。
3. **代理输出与实际输出的差别：** 在保持人口与评分不变时，可设计小范围A/A′/A′+B隔离3D代理目标与原节点圆周恢复的影响；现有报告尚无此比较，不能把它写成已证实的损害来源。

本配方不自动扩268、增加seed、重fit或扫描floor/FW步数。139选源目标与142状态传播的结果仍按各自Record/Squeeze消费，不把它们与140模块拼成未经联合评价的新成绩。对可信同行、实际K≥3、独立传感器部署与在线效果仍无新增证据。

### Mode C交接包（编号由联合回流统一分配）

```yaml
package_type: SQUEEZE_REPORT
package_version: research-pipeline-contract v1.0
package_id: SQ-SOURCE-CONTRIBUTION-READOUT-20261030-140
input_record: LEVEL1-SOURCE-CONTRIBUTION-READOUT-20261030-140-attempt-1
source_idea_id: IDEA-SOURCE-CONTRIBUTION-READOUT-20261030
source_matrix_id: MC-SELECTION-READOUT-20261030-v1.0
selected_C_id: SR-C2
execution_status: COMPLETED_exit0
verification_status: completed_limited_independent_array_audit_exit0
new_P: none_required
new_F_candidate: same_critic_chord_geometry_numerically_solved_but_not_actual_MAE_increment
new_M_candidate: risk_projection_clip_geometry_condition_and_physical_readout_mismatch_AI_UN
new_E_candidate: real_140_fixed64scenes_13fits_limited_independent_audit_and_actual_harm
negative_relation: SR_C2_current_Aprime_and_B_not_promoted
qualified_deep_sources_added: 0
no_automatic_followup: true
next_skill: deep-read_Mode_C_joint_feedback_after_three_final_records
```

**回流已完成：** [Mode C1.59](/Users/a1/Projects/biye-ronghe/docs/surveys/2026-10-26_mtad_actual_use/reassessment_20261027.md#mode-c-159)登记F241/M109（AI/UN）/E140；[SR v1.1](/Users/a1/Projects/biye-ronghe/docs/surveys/2026-10-27_expanded_search/selection_readout_matrix_20261030.md#selection-readout-feedback)更新SR-C2，分别保留A′评分、B读出、实际尾部及共同偏移代数边界。上面的交接包保留输出时的下游位置，现该步骤已完成；没有新论文或自动后继。执行回执见[S2L140](../../HandOver/S2L-20261030-140.md)。
