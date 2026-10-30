# 142：高损害恢复点的独立复放与计算链

2026-10-30。**此前只在公共汇总中看到的风险集中，现已核到真实端点状态和输出传播路径。** 142一次COMPLETED/exit0，完整64场、4257个零支持可评点，独立复放624个唯一anchor的原两rho共1248核。所有被复放节点的Q64位置与原137绑定，全部零支持输出映射到当前组anchor；四种CV输出的逐case复放位置差为0m。这里确认计算如何产生保存结果，不认证物理噪声真值或原始传感器故障来源。

最严重的4393有7个anchor、52个零支持点，其中**一个anchor的27个跨度后点承担全V观测RTS SSE的99.1792%**。其工作模型末端速率775.164m/s；最大传播3018.192s，局部传播位移2339593.867m，最终径向误差2389909.389m。该路径不是日期线反解猜测、重新分组或RTS单独造成；新复放保留真实未wrap局部端点状态与chart。当前仍没有新的恢复方法成绩。

## Record：计划、输入与实际执行

- package_type：EXPERIMENT_RECORD；package_version：research-pipeline-contract v1.0；experiment_id：LEVEL1-STATE-RECOVERY-TAIL-CONTINUATION-20261030-142-attempt-1。
- source_idea_id：IDEA-STATE-RECOVERY-TAIL-CONTINUATION-20261030；plan_version1.0；[142 Idea/spec](../proposals/state_recovery_tail_continuation_20261030.md)、[固定L2S142](../../HandOver/L2S-20261030-142.md)。科学设计继承141事前C55/F238/M107/E136；[Mode C1.58技术E138](/Users/a1/Projects/biye-ronghe/docs/surveys/2026-10-26_mtad_actual_use/reassessment_20261027.md#mode-c-158)仅记录141入口失败。
- 研究类型：computational diagnostic，暴露T/V的事后过程核查；不是新方法比较、神经训练、噪声辨识或独立测试。主要估计对象：原零支持人口、aligned→group→anchor绑定、原核位置/支持层复算及端点传播中间量。
- 冻结代码ddc35de，checkout `/home/wangguangjie/djs/association-code-run142`；[原141 FAILED现场](2026-10-30-state-recovery-tail-entry-failure.md)保持，不覆盖、重启或倒填成功。142只修真实`position.all`读取路径、fixture布局与新任务元数据，数学核/两rho/人口/容差/预算不变。
- 原输入：137 `LEVEL1-OBSERVATION-SCALE-20261030-137-attempt-1`的物理/完整artifact/逐场；127 `LEVEL1-NATIVE-CV-STATE-20261029-127-attempt-1`的native-events/native-motion/native-evaluation；原137绝对manifest。真实命令及环境见[request](/Users/a1/Projects/biye-ronghe/docs/evidence/20261030_state_recovery_tail/142/launcher/request.json)。
- 实际run `/home/wangguangjie/djs/association-code/runs/LEVEL1-STATE-RECOVERY-TAIL-CONTINUATION-20261030-142-attempt-1`；launcher同父`LEVEL1-STATE-RECOVERY-TAIL-CONTINUATION-20261030-142-launch-attempt-1`；unit `mtad-6f62d97c6e72.service`，Linger=yes、Restart=no。

| 项目 | 计划 | 实际 |
|---|---|---|
| 分区/人口 | 原T32/12组件、V32/9组件，不删难例 | 全64场；T37635/V71071可评点；零支持T1487/V2770全保留 |
| 对应/支持 | 独立native span及source等价类，真identity span另列 | 全部node的native offsets、canonical行、时间与span绑定；首代表/重复及aligned输出/valid逐组绑定 |
| 固定原核复放 | 只零支持涉及唯一anchor×原两rho，最多4426核 | T225/V399唯一anchor；总624×2=1248核，367326次原生事件处理，398036转移与398036RTS后向步 |
| 结果绑定 | 每trace完整Q64及全部零支持输出、原指标 | 原5臂full及5支持层mae/mse/rmse/P95/P99均核；四CV逐case保存位置与复放误差全0m |
| 工作参数 | 原reference-rho1e-8、observation-rho1e-4 | 未重选rho、未拟新head、无新方法预测/teacher/神经更新/新组/封存测试 |
| 资源 | CPU2/taskset16–17、RSS2GiB/AS4GiB、1800s advisory、GPU0 | 程序117.324207682s，launcher118.237015305s；峰值290975744B，未超建议 |
| 实际时间/终态 | 完整终点正常退出 | UTC 2026-10-30 03:04:31.192948→03:06:28.517116（北京时间11:04:31→11:06:28）；run/launcher exit0；inactive/dead、success、MainPID0 |

0新方法不等0计算，1248核/事件/转移/后向步和I/O全部记账。这不是整系统在线时延。原算法是完整native窗chart/RTS，FWD也不能因命名就称严格在线。

## 人口、原误差与支持层独立复核

以下是**原137保存输出的独立评分**，不是142新预测成绩，单位m。完整原5臂见公共summary，本表压缩重复控制以突出所核对象。

| 分区/原方法 | 点数 | MAE | RMSE | P95 | P99 |
|---|---:|---:|---:|---:|---:|
| T 原生OLS源等权 | 37635 | 1311.223 | 2878.589 | 2544.797 | 10480.821 |
| T reference-RTS源等权 | 37635 | 1244.531 | 2761.939 | 2452.560 | 9453.277 |
| T observation-RTS源等权 | 37635 | 1267.062 | 2709.036 | 2657.598 | 9383.673 |
| V 原生OLS源等权 | 71071 | 1276.441 | 2168.137 | 3097.435 | 7194.825 |
| V reference-RTS源等权 | 71071 | 1198.685 | 2136.307 | 2913.120 | 6224.031 |
| V observation-RTS源等权 | 71071 | 1784.722 | 27531.051 | 3332.561 | 9514.676 |

T原生span内/外37048/587，V70027/1044；预测组0/1/多源分别T1487/13521/22627，V2770/28454/39847。两套mask均从真实native与组成员独立重建；先前137审计未独立重建附加层的缺口，本次补到这些字段，但没有补成所有物理机制或上游算法都已认证。

T零支持点中938点在真实identity各输入片段span并集内，V为2032点；真实identity仅用于诊断，不能在部署时据其借用别组片段。span是每片段[min,max]闭区间，不是exact事件集合，也不是从身份最早至最晚填满所有间隙。预测组0支持不等于该目标所有输入都没有覆盖当前时刻。

### 零支持before/after的异质性

| 分区/相对anchor原生跨度 | 点数 | 原OLS MAE/RMSE | reference-RTS MAE/RMSE | observation-RTS MAE/RMSE |
|---|---:|---:|---:|---:|
| T before | 728 | 7560.658/11942.064 | 7169.523/12610.121 | 5612.037/8544.646 |
| T after | 759 | 4889.400/8947.905 | 4627.843/10513.974 | 6197.953/10107.053 |
| V before | 1655 | 4551.600/7969.453 | 3713.888/6534.456 | 6103.858/9949.141 |
| V after | 1115 | 3720.313/7392.734 | 3018.898/8783.522 | 36004.473/219127.888 |

V零支持仅3.8975%可评点，承担观测RTS SSE的99.6917%；T对应47.3158%，不能把V的集中程度推广到T。T before甚至有明显收益，V before仍比旧控制差；因此“全部跨度外都无价值”或“只处理4393即可”均超出证据。完整各场/各层与原FWD控制仍保留，未删点改变总体。

## Squeeze：从风险集中落实到计算路径

- package_type：SQUEEZE_REPORT；package_version：research-pipeline-contract v1.0；squeeze_id：SQ-STATE-RECOVERY-TAIL-CONTINUATION-20261030-142。
- 输入是本Record和原142完整输出，evidence_status为限定数组/原核数值复放verified，物理生成与真实噪声机制unknown。没有追加新方法、参数搜索或V选型。

### 1. 4393不是52个相同点：7个anchor中一个27点分支主导

原4393零支持52点来自7个anchor，before9/after43。按新完整case表的事后描述，一个纯组anchor的27个after点占全V观测RTS总SSE **99.179239%**，而52点合计约99.1795%。其余6个anchor的25点并没有775m/s的同一状态，不能把52点全部写成一条相同轨迹。

该27点分支：418个原生事件，原生跨度4175.808s，最大原生间隔10.014s；所有点pure，相对真实identity的span并集其中17点仍在内。工作模型末端速率与整段输入OLS不同：

| 同anchor的过程量 | 实测值/意义 |
|---|---|
| 原生末两点差分速率 | 620.978m/s，是带噪位置的差分，不是真实船速 |
| 原输入整片段OLS速率 | 9.930m/s，不能将其当物理真值，但反映不同时间范围的估计差异 |
| reference-rho末端工作状态速率 | 35.537m/s，仍非校准真实速度 |
| observation-rho末端工作状态速率 | 775.164m/s，所有27点共用同一原生末端状态 |
| 最大after传播间隔 | 3018.192s；该例组内最近原生时距也相同，并非改选更近组员便必然修复 |
| 最大局部速度传播位移 | 2339593.867m，等于该间隔×末端工作速率 |
| 当前CV末端位置距原生末点 | 2314.036m；不是相对真实端点的误差，也不能加到径向误差当线性恒等式 |
| 原query实际径向误差 | 最大2389909.389m、27点均值1222553.040m；端点/传播在局部平面，最终评价在球面，二者不是同一个标量 |

**可支持的计算链DE：** 固定原输入/工作尺度 → 复放得到上述末端状态 → 原无支持回退选当前anchor → 原代码按时距延长该状态 → 保存query位置及其参考误差逐点绑定。FWD和RTS在末端状态按算法相同，因此这27点两者产生相同保存位置；该相等是既有递推结构，不另算新机制证据。数值复放已排除“这里只是wrapped经度反解错了”与“误差来自另一个输出组”的解释。

**仍未核实的物理解释UN：** 原native末端快速变化来自何种噪声、异常量测、轨迹/时间生成或其它过程；工作R=I与真实噪声是否匹配；原reference是否物理准确。不能把620.978m/s差分直接当真实机动，也不能用本例宣整个数据集的传感器坏了。需要修复的是当前计算在合法观测支撑之外如何恢复，不是先给传感器身份定优劣。

### 2. 另一个结构边界：当前anchor不一定是组内时间最近成员

对完整case表事后只读记账：T225个唯一anchor/1487点，anchor距端点绝对时距均值1288.646s、中位802.667s；V399个唯一anchor/2770点分别1366.894s、855.111s。T549点、V1127点的anchor绝对时距比组内最小nearest大超过1e-6s；这1e-6仅区分浮点等值的记账容差，不构成部署门限。平均多出的时距分别198.769s/256.626s。

这定位了“组有多个候选却沿首节点传播”的潜在输出选择边界，但尚未验证最近成员更准确。尤其上述27个主导坏点的组内最小时距与anchor相等；单纯换nearest不一定解决该分支。也不能借真实identity从其它组取帮助来做非法oracle修复。

### 3. 计划偏差、附加分析及证据边界

科学142按修复合同一次到完整终点，未放宽schema/容差、改变原两rho/人口或改组；原141失败独立保留。0新方法/0新拟合不是0核计算。所有被选择trace的原生事件、全部span、4个CV方向/尺度位置及zero输出对应已核；其它节点没有额外重跑KF，更没有认证真实噪声协方差。

上述绝对时距、唯一anchor和4393七个分支的细分是完成后从**全部64个原case文件**一次只读衍生的事后Squeeze，不冒称原预注册主量。独立成本0.425041575s、峰值53768192B，写新`additional-tail-accounting.json`，没有覆盖科学summary/private/输入或重新预测。该统计只输出匿名分支和聚合，原身份/坐标仍在服务器。

本次没有新方法差值，故不造bootstrap/新seed、同行胜负或跨域/K≥3成绩。接口可变K与支持处理的存在，不等于真实多传感器有效性已验证。独立源码转录与原核软件等价/原保存值一致证明计算链，不能替代传感器真实性认证。

### 4. 去留与可检验的后继

M107从“只有公共层集中、传播中间量未知”推进到本例明确的算法放大路径；观测选择目标与恢复权限是否普遍匹配、异常输入为何出现，仍未完全分离。当前全局内点选rho配方不晋级、不按V改rho/删坏点，也不退固定源优先。

可发展的下一方案应分离**支持域外如何传播**与**组内合法候选如何选取**：保持支持内状态、候选和全人口固定，比较现有CV延伸与稳定的原生运动/端点恢复，再在同一恢复基础上检验source无关的可观察候选规则；必须保留T before收益、V before不足和27点组内最近也无助的反例。此处没有验证哪个替代方案有效，不把“已查清计算链”包装成“已修复精度”。后继须新的具体Idea/固定合同，不能直接从诊断调一个速度上限。

## 原件与Mode C输入

[run](/Users/a1/Projects/biye-ronghe/docs/evidence/20261030_state_recovery_tail/142/run.json)、[summary/全部支持层](/Users/a1/Projects/biye-ronghe/docs/evidence/20261030_state_recovery_tail/142/summary.json)、[64场](/Users/a1/Projects/biye-ronghe/docs/evidence/20261030_state_recovery_tail/142/scene-summary.json.gz)、[匿名V top20](/Users/a1/Projects/biye-ronghe/docs/evidence/20261030_state_recovery_tail/142/top-cases-V_select.json)、[额外只读记账](/Users/a1/Projects/biye-ronghe/docs/evidence/20261030_state_recovery_tail/142/additional-tail-accounting.json)、[命令](/Users/a1/Projects/biye-ronghe/docs/evidence/20261030_state_recovery_tail/142/launcher/request.json)、[退出](/Users/a1/Projects/biye-ronghe/docs/evidence/20261030_state_recovery_tail/142/launcher/status.json)、[真实服务终态](/Users/a1/Projects/biye-ronghe/docs/evidence/20261030_state_recovery_tail/142/observed-service-terminal.json)均完整保留。T top20及736块schema预检同目录。原private `trace-*.npz`/`cases-*.json.gz`、全部原native/坐标/reference/身份留原server runs，不下载。

Mode C handoff：消费本Record/SQ，新增应为明确的“原核高速末端×无支持长传播的数值路径及组候选时距边界”现象与证据，物理生成原因保持UN；P沿原状态/完整风险问题，M107可限定增强，不把数学结构恒等式或原137误差重复计成独立实证。下一skill为deep-read增量回流，再从具体未决点构造后继Idea。

**回流已完成：** [Mode C1.59](/Users/a1/Projects/biye-ronghe/docs/surveys/2026-10-26_mtad_actual_use/reassessment_20261027.md#mode-c-159)登记F242/E141，限定增强M107；[OD v1.5](/Users/a1/Projects/biye-ronghe/docs/surveys/2026-10-27_expanded_search/observation_driven_matrix_20261030.md#state-recovery-142-feedback)定向更新6旧格并增3格。物理来源仍UN，原141技术E138保留；没有新恢复方法成绩或运行任务。执行回执见[S2L142](../../HandOver/S2L-20261030-142.md)。
