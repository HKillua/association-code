# 124：软混合与中心收缩分离验证的真实记录与深挖

2026-10-29。EXPERIMENT_RECORD `EXP-124-20261029`＋SQUEEZE_REPORT `SQ-SOURCE-MIX-SHRINK-124-20261029`，research-pipeline-contract v1.0；一次attempt-1，completed/exit0。[执行前Idea1.0](../proposals/source_mix_shrink_control_20261029.md)、[独立交接](../../HandOver/L2S-20261029-124.md)，来源MX-C1/Mode C1.42及123完成回流的1.43。

## 1. 问题、固定设计与实际配置

- 检验固定soft贡献与中心幅度两个独立操作是否超过FixedLOW，不能把两源差视作已识别精度、把组数视作meff或称已拟合EB。原V32/9组件/WS/FULL Q64/首代表/人口冻结。
- 原121十臂320场景臂直接导入，新增5臂160场景臂，共15臂105配对；新5为无中心soft0.75、完整中心soft0.75、收缩hard、收缩soft0.75、完整中心源等权0.5。各源内支持成员等权，单支持源总质量1，全无support回相应输入首anchor；双源场景的高源singleton仍校正。速度/航向复制原Mean。
- 每fold原中心乘m/(m+4)，m为其他fold原eligible组数，不重新拟合。0.75/0.5/m0=4已在3de43f4、查看123结果前冻结；123 Record→Squeeze→Mode C1.43完成后才发布和执行本项，没有看新分数选参数。
- 预测函数不接identity/reference/target support；全部预测锁定后评价。0神经训练/模型/前向/GPU/新scores/分组/阈值/normalizer/中心拟合/Huber/ZIP/CSV/T_test。工作输入是离线Q64，不宣在线或同行复现。
- 原十方法/45pair及9组件对应旧子树独立公开JSON逐叶153930项exact、0差；不等私有数组审核。独立按逐场公开统计汇总15臂MAE/RMSE最大差4.55e−13m；完整真实artifact由生产导入支路逐字段exact，另有下面独立保存数组核验。

## 2. 全15臂结果（同71071点，单位米）

| 方法 | MAE | RMSE | P95 | P99 | 场景等权MAE |
|---|---:|---:|---:|---:|---:|
| MEAN | 1276.328389 | 2168.265557 | 3103.794978 | 7194.824823 | 1249.718245 |
| HUBER | 1275.619770 | 2168.557775 | 3114.052946 | 7194.824823 | 1249.739278 |
| FIXED_MEAN | 656.041340 | 1864.654434 | 2444.380637 | 7541.587652 | 484.962267 |
| FIXED_HUBER | 656.373673 | 1866.777656 | 2455.880959 | 7541.587652 | 485.063925 |
| SYMMETRIC_MEAN | 1716.224977 | 2300.416823 | 2390.410708 | 7008.620566 | 1595.335609 |
| SYMMETRIC_HUBER | 1717.793043 | 2301.626833 | 2395.856192 | 7008.620566 | 1595.790864 |
| ANCHOR | 844.759832 | 2457.606705 | 4385.888699 | 10460.984233 | 986.493191 |
| NEAREST_SUPPORTED | 927.434649 | 2173.150207 | 3873.502039 | 7301.504313 | 886.352046 |
| LOW_SOURCE_SUPPORTED | 527.446344 | 1813.303585 | 2942.040400 | 7260.393881 | 478.072440 |
| FIXED_LOW_SOURCE_SUPPORTED | 447.060488 | 1740.813819 | 1012.562755 | 7526.008427 | 382.696475 |
| SOFT75_SOURCE_SUPPORTED | 855.331611 | 1904.871849 | 2945.547795 | 7198.912945 | 810.581293 |
| FIXED_SOFT75_SOURCE_SUPPORTED | 531.011512 | 1769.733042 | 1507.629030 | 7539.335041 | 419.261139 |
| SHRINK_LOW_SOURCE_SUPPORTED | 455.287299 | 1741.314948 | 1163.727309 | 7482.533035 | 412.296502 |
| SHRINK_SOFT75_SOURCE_SUPPORTED | 555.445817 | 1771.047397 | 1501.215843 | 7488.627530 | 515.883648 |
| FIXED_EQUAL_SOURCE_SUPPORTED | 655.615588 | 1863.516170 | 2420.771831 | 7541.587652 | 484.782926 |

## 3. 两主效应、交互与强控

所有差值candidate−baseline，正数为误差更高；场景/组件好差同统一使用1e−9m容差，不将约1e−12m的舍入差算效果。下面是确定性评价指标差，非统计显著性或逐点线性因果分解；RMSE和分位数尤其不能视作线性误差组成。

| 比较 | MAE差 | RMSE差 | P95差 | P99差 |
|---|---:|---:|---:|---:|
| FS−FH（完整中心下soft主比较） | +83.951024 | +28.919224 | +495.066275 | +13.326614 |
| SH−FH（hard下收缩主比较） | +8.226811 | +0.501129 | +151.164553 | -43.475393 |
| SS−SH（收缩下soft） | +100.158518 | +29.732449 | +337.488534 | +6.094495 |
| SS−FS（soft下收缩） | +24.434305 | +1.314355 | -6.413187 | -50.707511 |
| SS−FH（组合，不能归给单项） | +108.385329 | +30.233578 | +488.653087 | -37.380898 |
| 四格交互 SS−SH−FS+FH | +16.207494 | +0.813225 | -157.577741 | -7.232119 |

| 比较 | 场景MAE好/差/同 | 组件MAE好/差/同 |
|---|---:|---:|
| WS_FIXED_SOFT75_SOURCE_SUPPORTED−WS_FIXED_LOW_SOURCE_SUPPORTED | 20/12/0 | 4/5/0 |
| WS_SHRINK_LOW_SOURCE_SUPPORTED−WS_FIXED_LOW_SOURCE_SUPPORTED | 2/27/3 | 0/7/2 |
| WS_SHRINK_SOFT75_SOURCE_SUPPORTED−WS_FIXED_LOW_SOURCE_SUPPORTED | 0/32/0 | 0/9/0 |
| WS_SHRINK_SOFT75_SOURCE_SUPPORTED−WS_FIXED_SOFT75_SOURCE_SUPPORTED | 2/29/1 | 0/8/1 |
| WS_SHRINK_SOFT75_SOURCE_SUPPORTED−WS_SHRINK_LOW_SOURCE_SUPPORTED | 1/31/0 | 0/9/0 |
| WS_SOFT75_SOURCE_SUPPORTED−WS_LOW_SOURCE_SUPPORTED | 0/32/0 | 0/9/0 |
| WS_FIXED_EQUAL_SOURCE_SUPPORTED−WS_FIXED_MEAN | 5/5/22 | 2/2/5 |

- 无中心soft相对LOW MAE+327.885267m，32场/9组件全差；不能把无中心的软贡献称有效。完整中心soft相对无中心soft MAE−324.320099m，说明原共享校正在该控制下仍有增量，但P99+340.422096m。
- 完整中心源等权0.5相对FixedMean MAE仅−0.425752m/RMSE−1.138264m，5场好/5差/22同、组件2好/2差/5同，去最大组件MAE反+0.115572m。源等权与按fragment等权不同，但未获稳定独立增量。
- FS相对FH有20场改善，却整体MAE/RMSE/P95/P99全差、4组件好/5差。SH相对FH只有P99较好、27场/7组件较差；组合比FH全部32场/9组件MAE差。不能挑P99或多数场之一宣整体成功。

## 4. 最大组件、作用域与完整人口

最大组件19场/58707点（82.603%），去掉后余13场/12364点：

| 臂 | MAE | RMSE | P95 | P99 |
|---|---:|---:|---:|---:|
| 完整中心/硬优先 FH | 564.276714 | 2519.443973 | 1175.447806 | 10327.828005 |
| 完整中心/soft0.75 FS | 582.020031 | 2527.335702 | 1533.814331 | 10327.828005 |
| 收缩中心/硬优先 SH | 587.514494 | 2523.189812 | 1537.394582 | 10299.384128 |
| 收缩中心/soft0.75 SS | 646.066076 | 2533.372051 | 1636.838778 | 10299.384128 |

去最大组件后FS/SH/SS对FH的MAE差仍为+17.743318/+23.237780/+81.789363m，当前负结果不是只由最大组件解释。

| 预测组support层 | 点数 | FH MAE | FS MAE | SH MAE | SS MAE |
|---|---:|---:|---:|---:|---:|
| predicted_group_both_sources | 39847 | 224.942581 | 374.677398 | 224.942581 | 403.585040 |
| predicted_group_low_only | 25194 | 213.940252 | 213.940252 | 213.940252 | 213.940252 |
| predicted_group_high_only | 3260 | 1775.592948 | 1775.592948 | 1954.408305 | 1954.408305 |
| predicted_group_no_support | 2770 | 4199.027143 | 4199.027143 | 4199.658773 | 4199.658773 |

- 固定soft只改both：21094点好/18753差，但收益总1267081.685m、损害总7233564.907m，故更多改善点也不保证更低整体MAE。其余三域与同中心hard exact一致。
- 收缩hard只在high_only和可能的高源首anchor no_support改变输出：前者1212点好/2021差/27同、MAE+178.815356m；后者15好/16差/2739同、MAE+.631630m。不能遗漏校正anchor回退；both/low_only不变。
- 全部组query人口102720，其中both41518/low26562/high11335/none23305；可参考有效点对应39847/25194/3260/2770＝71071。新增臂相对FH的exact位置变化组query数：无中心soft55845、fixedSoft41518、shrinkHard14327、shrinkSoft55845、equal41518；此分母不是71071，也不等有意义米级效果点数。
- 96个scene-fold的m最小1/中位9/最大189，27折m≤4；lambda最小.2/中位.692307692/最大.979274611。原中心norm中位2798.281m→收缩后1930.937m，两者各4个零中心；m<2的中心本已零。96fold有关联，不是96独立重复或meff。
- 真目标支撑外1044点FH/FS/SH/SS MAE1412.801254/1415.149530/1409.135048/1410.988247m，虽部分收缩在此略好，不能用于部署门控；它不等2770个预测组no_support点。
- 全15臂相同1427目标/1605组/217重复，1388目标命中/39无预测；reference有效72196、共同71071、缺1125，coverage98.4417419%。105配对gained/lost/rescued/lost-target全0；完整旧十臂、逐场/9组件/全部分层和配对留原JSON，无删难例/组/点。

## 5. 执行、核验与成本

- 冻结1a2967f，独立服务器`association-code-run124`；原服务mtad-25a33ce6acd6.service。程序最终run耗时167.049659s（summary写盘前165.758344s不是最终全程），launcher176.664933s/RSS2010722304B。CPU0–3/4线程，RSS4GiB/AS8GiB/600s advisory，未超建议；完整命令/输入/环境/起止在request/run。
- 真实原run/launcher/exit_code均0；服务inactive/dead、success、ExecMainStatus0、Restart=no、Linger=yes，冻结树干净。一次attempt，无失败/重试/改参/新增训练或人口偏离。
- 独立NumPy保存数组主位置/人口审核32场/15臂/105对、9组件及去最大共11聚合：111780numeric/401183exact/210null通过，最大MAE/RMSE差2.153e−9m、MSE6.855e−6m²。程序17.358434s/launcher17.922584s/RSS238694400B，成本另记；审计服务mtad-a0ba4fc47e98同样成功终态。
- 未独立审核真实source/support语义、中心拟合/先验/物理bias、reference正确性、支持分层/逐点benefit-harm/论文坐标指标/速度航向；这些公开分层为生产评价输出，不能借主数组审计扩大认证。审计字段imported_original32_scene_count=0表示未从外部pilot路径取缺省private，不表示本次没有导入旧320臂；预测导入量以原run/全旧artifact绑定为准。
- 合成软件测试仅证明本次会改变的导入/质量/回退/日期线/失败保全链路；真实科学结果是本节服务器运行，不将测试数/CPU试例充作模型效果。

## 6. Squeeze：当前处置与未证机制

**支持到这里：** 已有固定共享中心与合法源优先效果保留；把当前经验soft=.75与m0=4加入，并未形成更好的总体恢复，收缩P99的微小改善伴随整体MAE和多数场景损失。软贡献的场景多数与点总体仍反向，需以合法条件风险研究，不直接确定一个全局混合值。

**不能支持：** 低源物理更干净、中心小样本方差为根因、已识别两源精度/EB、所有软混合或收缩无效、可信国际同行胜出。欠校正/错组/运动/共同偏差/reference关系都是竞争解释，不由这次分数唯一决定。

**处置：** 本次固定soft=.75/m0=4配方到终点关闭，不自动扩268、调alpha/m0、加seed或训练。下一可区分问题是合法输入是否能在T侧预测借高源的条件性误差变化，再冻结到V；这仍须自己的矩阵/Idea和有监督强控，不从V误差做oracle。输入时间形状、native状态及同信息关联/对象假设独立继续，不以前述失败当整类禁令。

## 7. 原件与标准交接

[必要脱敏原件](/Users/a1/Projects/biye-ronghe/docs/evidence/20261029_small_validation/124/)保留run/summary/完整逐场、原launcher七文件/终态、audit及其launcher/终态；坐标、身份、中心、权重与大数组留原服务器run。

```yaml
package_type: EXPERIMENT_RECORD
package_version: research-pipeline-contract v1.0
experiment_id: EXP-124-20261029
source_idea_id: IDEA-SOURCE-MIX-SHRINK-CONTROL-20261029
plan_version: "1.0"
execution_status: completed
research_question: 固定软贡献与中心幅度独立主效应及交互
hypothesis: 两操作可降低当前恢复风险，尚未识别其物理机制
planned_vs_actual_design_and_controls: §1/执行前Idea，原十臂导入新增五臂，15臂105配对；无偏离
actual_configuration: §1/5，冻结1a2967f，CPU4/RSS4GiB/AS8GiB/600s advisory
samples_materials_or_data_provenance: 原121/056/114/118与manifest，原V32/9组件
results_with_uncertainty: §2至4，单次确定性配对/完整场景组件，无query-IID推断
all_runs_or_repetitions: 一次attempt-1/exit0，一次独立数组审计另计
observations_and_deviations: 无选参排除重试，旧公开子树和完整artifact均绑定
evidence_locations: §7及服务器原run
known_limits_and_missing_evidence: §5/6物理机制与未审范围
next_routes: squeeze-experiment
next_skill: squeeze-experiment
```

```yaml
package_type: SQUEEZE_REPORT
package_version: research-pipeline-contract v1.0
squeeze_id: SQ-SOURCE-MIX-SHRINK-124-20261029
input_experiment_record_ids: [EXP-124-20261029]
claim_scope: 原开发V32/固定WS/同Q64的经验混合和收缩，非同行或概率保证
data_quality_and_provenance_gaps: §5独立主位置审计有限，真实噪声/参考语义未识别
primary_result_with_uncertainty: fixedSoft-FixedLOW MAE+83.951024m，shrinkHard-FixedLOW+8.226811m；组合全32场/9组件差
process_dynamics: 一次固定32终点/0神经训练，成本见§5
mechanism_and_intermediate_findings: 支持域的实际操作可分离，真实中心采样方差和源精度UN
errors_heterogeneity_and_boundary_conditions: 多数场和点加权反向，收缩P99改善伴随MAE及多数场恶化
robustness_sensitivity_and_alternative_explanations: 去最大组件仍不利，保留全部15臂/105配对；当前配方而非整类否定
new_P: none，沿用P02/P11/P14
new_F: 固定soft和收缩的有限负结果及非线性交互/风险取舍
new_M: 操作作用域DE，物理解释AI/UN
evidence_index: §7全部原件及主仓交接
open_questions: T侧合法条件收益、独立native状态/输入/关联/对象路线
next_routes: deep-read Mode C；停止本配方自动后继
next_skill: deep-read Mode C
```
