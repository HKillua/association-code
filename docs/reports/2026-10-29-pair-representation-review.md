# 108：候选B的内容是否进入表示——实际记录与深挖

2026-10-29。**固定T32中，402个具有真实正负eligible候选的anchor，在明确的literal整窗均值＋逐列minmax接口下全部发生容差内输入碰撞。** stats、pointwise和rawB均在这402个anchor上保留至少一项跨标签内容差异；这不等于所有碰撞消失、可正确分类、作者实际接口错误或EQ-Reg有效。stats可逆的是已经形成的`d=A−mean(B)`，不是此前被均值删去的B时间形状。

按 `EXPERIMENT_RECORD → SQUEEZE_REPORT → Mode C输入`记录，消费[事前Idea1.1](../proposals/pair_representation_dependency_20261028.md)及[L2S-108](../../HandOver/L2S-20261028-108.md)。这是本轮真实一次计算诊断，没有训练/模型成绩。当前执行/决定不在本文维护。

## 1. EXPERIMENT_RECORD

| 字段 | 实际事实 |
|---|---|
| experiment_id / upstream | `LEVEL1-PAIR-REPRESENTATION-DEPENDENCY-20261028-108-attempt-1`；`IDEA-PAIR-REP-DEPENDENCY-20261028@1.1`；来源C45/P09，假设挑战 |
| 问题/主要估计对象 | 固定A及q/validity，替换同场eligible异源B时，四个明确接口是否保留B内容；混合identity anchor的literal完整输入实际正负碰撞 |
| 样本/抽样 | Level1固定T_core2100按数字排序，以`random.Random(20261028)`抽32场再按数字排序；与实际32场名单逐项一致。每场每source最多8 anchor，一个连续16原生点窗，抽样/选窗不看身份或模型成绩 |
| 对齐/人口 | 当前MTADArchive合法节点清洗；B原生首尾覆盖整个q才eligible，同一线性插值，航向unwrap插值后映回[0,360)。不外推；不满足覆盖记outside-support/unknown，没有视为负例或无效后抹去 |
| 计算条件 | float64逐列minmax；零range研究分支u=0。literal/stats/pointwise/rawB同A/q/validity，不含B ID/真实identity或reference；native速度不换算m/s，标量航向仅印刷公式控制 |
| 精度合同 | normalized分量1e−8；stats m/r及rawB按属性采用`64*eps64*max(1,abs(x),abs(y))`。这只区分浮点误差，不是物理有用效果门槛；没看结果改阈值 |
| 实际读取/大小 | 只32个label＋32个measurement，64次读取尝试、64已读、64已解析；reference/V/test payload各0。声明解压总量label71,525B、measurement31,122,124B；最大成员5,559,082B，全部≤64MiB |
| 合法数据审计 | 456,793量测行、1,642标签/候选/合法节点，隔离0、精确重复删除0、区间外量测诊断0、隔离损失正边0；本批没有隔离，不证明整个数据集无隔离/质量问题 |
| 实际时间/终态 | 程序UTC2026-10-28 20:51:10.332959–20:51:25.673301（北京时间10月29日04:51:10–04:51:25），15.340373s；launcher15.815680s，二者COMPLETED、实际退出0。32个progress事件顺序完整；600s advisory未触发 |
| 资源/权限 | CPU4；RSS150,088KiB＝153,690,112B≈146.57MiB；AS8GiB、RAM预算4GiB。0GPU/模型构造/前向/优化/checkpoint读取/新normalizer，0reference/V/test/未用CSV payload |
| 代码/环境 | [实际脚本](../../tools/analysis/diagnose_pair_representation_dependency.py)，冻结worktree`association-code-run108`；完整Python路径、三ZIP分卷顺序及argv/environment见launch-request。具体Python/NumPy版本未写回传run，不拿本地版本补填 |
| 全部提交/偏差 | 科学job实际提交1次、attempt-1，没有失败科学run/重试/换包/加场/调整阈值。启动前操作者读错路径定位文件，过程及纠正如下，不能隐藏也不能误写成实验失败 |

**启动前路径定位纠正：** 原L2S指`090C/execution.json['arguments']['archive']`；根线程实际误读了`090C/preparation.json`的`arguments`，该字段不存在，尚未提交job。随后从真实056 `system/run.json inputs.archive`解析原三分卷，并与107已录TCM argv逐项匹配同一包，再启动108。此为启动前操作定位纠正，`scientific_inputs_changed=false`、`actual_job_submissions=1`；没有改分卷顺序/科学输入或把一次失败计算包装成成功。原件见[定位纠正记录](/Users/a1/Projects/biye-ronghe/docs/evidence/20261029_small_validation/108/archive-path-resolution.json)。

原件链：

- [真实run](/Users/a1/Projects/biye-ronghe/docs/evidence/20261029_small_validation/108/run.json)、[全summary](/Users/a1/Projects/biye-ronghe/docs/evidence/20261029_small_validation/108/summary.json)、[全部32场/匿名方向](/Users/a1/Projects/biye-ronghe/docs/evidence/20261029_small_validation/108/scenes.json)、[32事件progress](/Users/a1/Projects/biye-ronghe/docs/evidence/20261029_small_validation/108/progress.jsonl)。
- [64成员读取状态](/Users/a1/Projects/biye-ronghe/docs/evidence/20261029_small_validation/108/read-members.json)、[预检声明大小](/Users/a1/Projects/biye-ronghe/docs/evidence/20261029_small_validation/108/member-inventory.json)、[实际命令](/Users/a1/Projects/biye-ronghe/docs/evidence/20261029_small_validation/108/launch-request.json)、[launcher终态](/Users/a1/Projects/biye-ronghe/docs/evidence/20261029_small_validation/108/launch-status.json)、[退出0](/Users/a1/Projects/biye-ronghe/docs/evidence/20261029_small_validation/108/exit_code.txt)。

血缘为：固定manifest抽T32 → 声明成员大小预检/只读64白名单CSV → 当前reader合法节点 → 固定A/q和全部异源B的support资格 → 同一个插值器/四表示 → labels后置正负比较 → 方向/场景/全局计数与误差。原CSV、A/B窗口向量/身份/坐标及private仍留服务器，本地本轮只读回传脱敏结果和代码，不重读CSV、不跑模型。

**独立核验范围：** 本地从冻结manifest按原seed重建名单，与run/scenes/progress完全一致；64读取记录与inventory逐项相符、无额外成员、全部READ/≤限额；方向→场景→汇总的全部计数、四版本计数、内容差异maxima、loader审计、literal零/近常数计数及逐场范围均重聚合一致，全部计数恒等式通过。全局逐pair误差样本没有在白名单中，因此summary的误差P50/P95/P99只能核样本数/有限性/单调性并引用实际输出，**不能声称本地重新算过这些分位数或重新对原向量逐一比碰撞**。该层来自实际冻结脚本的直接向量比较与原运行，缺口保持显式。

### 1.1 全部分母

| 层级 | 分子/分母及解释 |
|---|---|
| 全部合法anchor节点 | 1,642；其中未抽1,149、抽中493。每场合法节点11–253，抽中11–16，非所有节点穷举 |
| 窗口 | 抽中493＝不足16点2＋有窗491。没有换窗重抽把不足补齐 |
| 有窗anchor的候选类型 | 491＝无eligible2＋仅正5＋仅负82＋正负混合402；有eligible489。混合为402/493（81.5416%抽中）或402/489（82.2086%有候选），分母不同 |
| 实际异源候选 | 12,918＝eligible6,560（50.7819%）＋outside-support6,358（49.2181%）。这些是选中有窗anchor→B的有向比较，不是全场全部无向pair人口 |
| eligible标签 | 6,560＝同A-B identity407＋不同identity6,153。正例占6.2043%；unknown未补成负例，未声称完整召回 |
| 直接literal碰撞 | 402/402混合anchor存在至少一个正B/负B完整输入在1e−8内碰撞；逐场混合8–16，每场均有实证支持，不只是一个场的例子 |

从407正eligible对＝402混合anchor＋5仅正anchor还可严格推出：本次每个有正候选的anchor恰有一个正eligible B。它只是本固定抽样/支撑规则下的组合约束，不是整个MTAD“每源每身份单轨”的证明，也不能据此驳回其他场景的fragment容量问题。

### 1.2 四个明确版本的全部结果

完整输入均附同一A_q/q/validity，下面仅列改变的分支。rawB保留的是16个对齐query上的B，非全部原生轨迹。

| 版本 | B相关分支 | 有eligible anchor | 全候选在容差内不变 | mixed anchor | mixed至少一项exact变化 | mixed至少一项超数值容差变化 |
|---|---|---:|---:|---:|---:|---:|
| literal | `N(A−mean(B))` | 489 | 489 | 402 | 397 | 0 |
| stats | `(u,min(d),range(d))` | 489 | 8 | 402 | 402 | 402 |
| pointwise | `N(A_t−B_t)` | 489 | 8 | 402 | 402 | 402 |
| rawB | `(u,B_q)` | 489 | 8 | 402 | 402 | 402 |

397/402 mixed literal存在**精确float64不等**，但最大跨标签差仅4.218847494e−15，全部小于预定1e−8；故应说“容差内不变/碰撞”，不能说bitwise全相同，亦不能把消减产生的低位差当稳定可辨信息。其余5个无exact变化anchor分布在313场2个、345场1个、4766场2个。

stats/pointwise/rawB的402/402仅表示**至少一个正负候选组合有差异**，不是“每个正负组合均不碰撞”；脚本只对literal显式逐正/负向量查存在碰撞，控制版本的any-change由分量包络max判断。全部候选不变的8个control anchor均在非混合人口，但公开输出不保留各anchor候选数，不能断言它们全是单候选、不能把未变化自动算作方法失败或成功。

### 1.3 normalized/native差异与浮点边界

literal全部6,560 pair对`N(A)`的max绝对误差：min0，P50=1.110223025e−16，P95=P99=2.220446049e−16，max4.218847494e−15；超1e−8数0/6,560。跨候选/跨标签的literal normalized最大差同为4.218847494e−15。pointwise normalized内容最大差为1.0，说明它确实改变依赖，未测关联准确率。

stats和rawB保留literal的normalized分支，其最大差仍仅4.218847494e−15；有效内容变化来自新增native分量，不能把原单位阈值混成1e−8。逐属性最大差如下，均是数值内容范围，不是误差、物理偏差或效果门槛：

| 版本/比较 | lat（度） | lon（度） | vel（原生数值） | cou（标量度） |
|---|---:|---:|---:|---:|
| stats / 全部候选 | 3.6202576771 | 5.50556807803 | 1186.13622872 | 211.913158609 |
| stats / 跨标签 | 3.12884838526 | 5.1537776033 | 1186.13622872 | 211.913158609 |
| rawB / 全部候选 | 3.63592388844 | 5.51688223538 | 5681.51725744 | 359.802197391 |
| rawB / 跨标签 | 3.14149081546 | 5.16200814303 | 5668.95784154 | 359.356586711 |

rawB中航向差可近360°，这是标量公式控制，不能当圆周角误差；velocity值未换算m/s，不能据这些差解释物理异常。没有把native量不同单位拼成统一“信息效果大小”。

stats以`d_hat=u*r+m`重构d，误差分布全部各n=6,560。relative分母是`max(1,abs(d))`，不是在d≈0处直接除d的相对误差；candidate_mean通过`mean(A−d_hat)`恢复原对齐B均值。每个分布min均0，完整P50/P95/P99/max如下：

| 属性 | 项 | P50 | P95 | P99 | max |
|---|---|---:|---:|---:|---:|
| lat | absolute | 0 | 0 | 4.33680868994e-19 | 5.55111512313e-17 |
| lat | relative | 0 | 0 | 4.33680868994e-19 | 5.55111512313e-17 |
| lat | candidate_mean_absolute | 3.5527136788e-15 | 7.1054273576e-15 | 7.1054273576e-15 | 7.1054273576e-15 |
| lon | absolute | 0 | 0 | 4.33680868994e-19 | 1.38777878078e-17 |
| lon | relative | 0 | 0 | 4.33680868994e-19 | 1.38777878078e-17 |
| lon | candidate_mean_absolute | 2.84217094304e-14 | 5.68434188608e-14 | 5.68434188608e-14 | 5.68434188608e-14 |
| vel | absolute | 3.5527136788e-15 | 1.42108547152e-14 | 2.13162820728e-14 | 4.54747350886e-13 |
| vel | relative | 1.73295010063e-16 | 1.11488883939e-15 | 3.5527136788e-15 | 7.1054273576e-15 |
| vel | candidate_mean_absolute | 7.1054273576e-15 | 2.13162820728e-14 | 2.84217094304e-14 | 3.41060513165e-13 |
| cou | absolute | 2.84217094304e-14 | 2.84217094304e-14 | 4.26325641456e-14 | 5.68434188608e-14 |
| cou | relative | 4.3919935488e-16 | 5.72714149879e-15 | 2.44903754878e-14 | 2.84217094304e-14 |
| cou | candidate_mean_absolute | 2.84217094304e-14 | 5.68434188608e-14 | 8.52651282912e-14 | 1.13686837722e-13 |

重构d最大绝对误差分别5.5511e−17度、1.3878e−17度、4.5475e−13原速度、5.6843e−14标量度；重构B均值最大误差7.1054e−15/5.6843e−14/3.4106e−13/1.1369e−13（按对应原单位）。它支持此次计算数值上可逆d及均值，没有测试/证明恢复B逐时形状。

**零/近常数的范围必须准确：** 回传`zero_range_pair_columns`和`near_constant_pair_columns`四属性均0/6,560，它们由literal `d=A−mean(B)`的span产生。因此本批没有触发literal的u=0常数分支，未观察到该分支/极小range的真实放大案例。pointwise `A−B`也调用相同normalize，但没有单独回传它的zero/near计数，不能把上述全0推广为四个版本所有列都无常数情形。这是实际记录的粒度缺口，不据此补跑/换阈值或声称覆盖所有数值病态。

### 1.4 全32场与匿名source方向

表内eligible/正/负及outside全保留；每场literal跨标签碰撞数与mixed数完全一致，三个control的超容差any-change也均等于mixed数，未单独重复32次相同列。少窗和无eligible没有从选择分母消失。

| 场景 | 合法节点 | 抽anchor | 不足窗 | 无eligible | 仅正/仅负/混合 | eligible正/负 | outside |
|---|---:|---:|---:|---:|---:|---:|---:|
| 11 | 27 | 16 | 0 | 0 | 0/6/10 | 10/94 | 112 |
| 66 | 121 | 16 | 0 | 0 | 0/4/12 | 12/446 | 510 |
| 313 | 13 | 13 | 0 | 0 | 1/1/11 | 12/19 | 53 |
| 345 | 20 | 16 | 0 | 0 | 0/0/16 | 16/72 | 72 |
| 347 | 75 | 16 | 0 | 0 | 0/4/12 | 12/291 | 297 |
| 408 | 34 | 16 | 0 | 0 | 0/0/16 | 16/176 | 80 |
| 529 | 37 | 16 | 0 | 0 | 0/5/11 | 11/126 | 159 |
| 775 | 16 | 15 | 1 | 0 | 0/2/12 | 12/42 | 56 |
| 797 | 253 | 16 | 0 | 0 | 0/4/12 | 12/1127 | 885 |
| 1319 | 46 | 16 | 0 | 0 | 0/4/12 | 12/176 | 180 |
| 1323 | 114 | 16 | 0 | 0 | 0/2/14 | 14/465 | 433 |
| 1448 | 16 | 14 | 0 | 0 | 0/5/9 | 9/48 | 51 |
| 1450 | 23 | 16 | 0 | 0 | 0/1/15 | 15/86 | 83 |
| 1643 | 38 | 16 | 0 | 0 | 0/3/13 | 13/123 | 168 |
| 1829 | 31 | 16 | 0 | 0 | 0/1/15 | 15/77 | 156 |
| 1909 | 16 | 16 | 0 | 0 | 0/4/12 | 12/56 | 60 |
| 1982 | 77 | 16 | 0 | 0 | 0/1/15 | 15/348 | 253 |
| 1998 | 58 | 16 | 0 | 0 | 0/4/12 | 12/222 | 230 |
| 2005 | 47 | 16 | 0 | 0 | 0/2/14 | 14/144 | 218 |
| 2018 | 93 | 16 | 0 | 0 | 0/1/15 | 15/341 | 388 |
| 2029 | 22 | 16 | 0 | 0 | 1/3/12 | 13/57 | 106 |
| 2211 | 129 | 16 | 1 | 0 | 0/2/13 | 13/416 | 535 |
| 2503 | 21 | 16 | 0 | 0 | 0/1/15 | 15/96 | 57 |
| 3354 | 55 | 16 | 0 | 0 | 0/4/12 | 12/233 | 195 |
| 3442 | 11 | 11 | 0 | 0 | 1/2/8 | 9/22 | 29 |
| 3863 | 16 | 14 | 0 | 1 | 1/2/10 | 11/36 | 61 |
| 4003 | 87 | 16 | 0 | 0 | 0/2/14 | 14/322 | 360 |
| 4524 | 43 | 16 | 0 | 0 | 0/5/11 | 11/196 | 137 |
| 4526 | 15 | 14 | 0 | 0 | 0/3/11 | 11/61 | 30 |
| 4561 | 32 | 16 | 0 | 0 | 0/1/15 | 15/98 | 143 |
| 4766 | 12 | 12 | 0 | 1 | 1/0/10 | 11/26 | 33 |
| 4866 | 44 | 16 | 0 | 0 | 0/3/13 | 13/111 | 228 |

全部32场各2个匿名source方向。方向0：251 anchor全部有窗、1无eligible、5仅正/68仅负/177混合；2,940 eligible（182正/2,758负）、2,943 outside。方向1：242 anchor，2不足窗、1无eligible、0仅正/14仅负/225混合；3,620 eligible（225正/3,395负）、3,415 outside。literal碰撞分别177/177和225/225，exact微差175和222；三个control超容差any-change分别177和225。ordinal仅是每场source排序后的匿名序，不据此赋予质量含义或声称方向0/1的因果差异。

## 2. SQUEEZE_REPORT：机制支持、竞争解释与发展价值

| 层级 | 具体支持及限制 |
|---|---|
| DE：实测现象 | 固定A/q/validity后，literal在6,560 pair与489候选anchor上均容差内不依赖B，402真实正负anchor直接碰撞。与“只有摘要或代数猜测”不同，本次有明确T侧真实材料及全部分母；仍仅针对本接口。 |
| AI＋实测支持：机制 | 逐列常数`c=mean(B)`被minmax平移不变性删除：非零range下`N(A−c)=N(A)`。浮点实现有低位差，数值计数支持容差性质；本批没有验证零range/病态range经验边界。 |
| DE：接口反证控制 | pointwise/rawB及stats均402/402具有跨标签内容变化，因此不能把literal限制推广为所有配对接口、所有归一化或任何有B旁路的模型失败。 |
| 结构边界 | stats保留m/r后可逆d，但d只依赖B均值。任何同均值不同时间形状的B会仍共享该理论接口；本次没有统计真实同均值异时形B碰撞发生率。已有软件用例不能代替该真实发生率。 |
| 未确认的学习机制 | any-change不等可识别identity；没有分类器/训练、样本独立性检验、可分margin、source-bias稳健性或真实关联/融合损失。pointwise归一化也会删除自身逐属性尺度/平移，不能称其保全所有候选信息；rawB只保留已插值的16点。 |
| 作者/领域边界 | CA Algorithm1 literal、正文逐时邻域均值及C=4实际装配仍须区分。此次没有使用作者代码/权重，不认证作者网络真实输入恰为literal，也不证明同行都无B通路。固定T32结果不能认证“领域尚未解决”或EQ-Reg有效。 |
| 目标可用性边界 | 49.2181%选定有向候选不覆盖q而为unknown；结果没有消除不齐时/非重叠fragment困难，不能从eligible子集推全任务召回。 |

**实际新增价值在分清“已删信息”和“没用好信息”，并把适配分工拆开。** 若可信baseline的实际接口确会删除关键候选内容，可先建立保信息A′：显式保留B/q及合法支撑、或按任务选保留均值/相对运动/逐时结构，比较同任务表现与成本。stats、pointwise、rawB代表不同保留条件，不能互相代替，也不能把额外通道的任何收益归于正则。若可信完整接口早已有足够B通路，则C45的literal诊断只作接口/复现警示；仍可围绕已测残余source-bias与身份区分取舍发展方法，但不能继续用本碰撞作为那个残余的实验证明。

EQ-Reg若后续值得发展，须在**同一保信息A′、相同paired增强/容量/预算**上比较无正则与有正则，先分清共同世界变换、source共享偏差和逐轨独立扰动的任务合法性，航向采用合法圆周处理，速度按新单位合同；不旋转时间轴，不假定后端正则能恢复已删除的信息。当前未测试这些迁移B的条件/增量，不因零训练诊断通过而自动开训。

过程证据显示32场依次完成，累计首场2.586s、末场15.269s，终态15.340s；首段包含初始化/读取，不能当纯场景计算耗时或训练收敛轨迹。无停止/超时/重试及样本排除；每场均发生literal混合碰撞提供异质人口支持，但固定32场/493抽anchor不是全T发生率、IID身份样本或独立性能CI。保持全部不利/未知结果，不重抽增加有利样本。

## 3. Mode C增量输入（候选ID，主矩阵统一编号）

| 类别 | 可回流内容 | 属性与来源 |
|---|---|---|
| P | `none`；不把minmax平移不变性包装新领域问题。细化C45输入依赖资格，问题未很好解决程度仍靠可信完整接口和残余证据。 | 未证明新增研究问题/真实方法瓶颈。 |
| F108-a | T32真实eligible人口下，literal 402/402混合anchor存在直接正负碰撞，489/489候选anchor容差内不变；397/402仍有机器低位exact差异。 | DE；只在本四接口/抽样/支撑合同下。 |
| F108-b | stats/pointwise/rawB均402/402有超预定分量容差的跨标签any-change；stats d及B均值可逆至本次误差表。 | DE；any-change非全部无碰撞/非可分类/非EQ-Reg效果。 |
| F108-c | 493抽anchor、12,918有向候选只6,560 eligible，6,358支撑unknown；两方向及32场分母完整。 | DE；说明结论条件范围，非全任务错失率。 |
| M108-a | 整窗B均值在literal minmax下被消去；m/r只逆d，不逆均值之前的B时形。 | 代码结构DE、条件代数AI及实际容差支持；作者实际装配UN。 |
| E108-a | 完整run/launcher/exit0、64白名单读取、T32与逐方向/逐场/汇总复核及浮点范围。 | 实际执行；未下载pair向量，分位数直接来源summary而非本地重算。 |

```yaml
package_type: EXPERIMENT_RECORD
package_version: research-pipeline-contract v1.0
experiment_id: LEVEL1-PAIR-REPRESENTATION-DEPENDENCY-20261028-108-attempt-1
source_idea_id: IDEA-PAIR-REP-DEPENDENCY-20261028
plan_version: '1.1'
execution_status: COMPLETED
research_question_and_hypothesis: '固定A/q/validity后四个明确表示是否保留B内容；literal实际正负是否碰撞'
planned_vs_actual_design_and_controls: '固定T32/每source至多8anchor/16点窗/全部异源候选，四版本均完成；不换窗/调阈/补抽/重试'
actual_configuration: 'seed20261028,float64,normalized1e-8/native64eps；CPU4/RSS150088KiB/15.340373s；0模型/前向/训练/reference/V/test'
samples_materials_or_data_provenance: '同一Level1原三分卷；T32 label/measurement64成员→合法reader→固定窗/支撑资格→四接口→后置label统计'
all_runs_or_repetitions: '科学job提交1次；32场探索不是32独立训练seed；启动前定位纠正非科学run失败'
results_with_uncertainty: '402/402混合literal碰撞、489/489候选anchor容差不变；三个control402/402 any-change；完整分母/逐场/误差如正文；无性能CI'
observations_and_deviations: '397 literal mixed有1e-15级exact微差；unknown49.2181%；启动前误读preparation纠正为同包056+107来源，科学输入不变'
evidence_locations: '本报告§1全部原件和脚本；archive-path-resolution保留定位过程'
known_limits_and_missing_evidence: '作者真实装配/可信baseline残余/EQ-Reg效果未知；pair误差向量未回传、quantile未本地重算；pointwise常数计数未回传'
next_routes: [squeeze-experiment]
next_skill: squeeze-experiment
```

```yaml
package_type: SQUEEZE_REPORT
package_version: research-pipeline-contract v1.0
squeeze_id: SQ-PAIR-REP-DEPENDENCY-20261029-108
input_experiment_record_ids: [LEVEL1-PAIR-REPRESENTATION-DEPENDENCY-20261028-108-attempt-1]
claim_scope: '本次T32/eligible规则/四个明确接口的候选内容依赖；非作者实现或关联融合效果'
data_quality_and_provenance_gaps: '64成员/32事件/全分母及嵌套统计复核通过；pair向量不在白名单，global误差quantile引用实际summary'
primary_result_with_uncertainty: '402混合literal全碰撞，低位exact不等保留；三个control全有any-change，无性能CI/全T外推'
process_dynamics: '15.340373s/RSS150088KiB/32场完整，0训练轨迹'
mechanism_and_intermediate_findings: '条件minmax删除B均值与实测一致；stats逆d及均值，不逆B时形'
errors_heterogeneity_and_boundary_conditions: '2短窗/2无eligible/5仅正/82仅负/402混合，unknown6358；literal无zero/near列，本批未覆盖pointwise常数情况'
robustness_sensitivity_and_alternative_explanations: '32场两方向均支持；rawB/pointwise限制强命题；不能由literal旁路缺失推作者或领域失败'
new_P: none
new_F: [F108-a, F108-b, F108-c]
new_M: ['M108-a: conditional algebra AI plus code/actual tolerance DE; author assembly UN']
evidence_index: 'E108-a及§1原件/全部表；实际提交1次的路径纠正记录独立保存'
open_questions: '作者完整输入与可信baseline残余、保信息/稳健性的代价、同A′下迁移B实际增量'
next_routes: ['deep-read Mode C增量', '核可信实际接口/残余后才另定适配Idea；无自动训练']
mode_C_input_bundle: '本报告§3 P/F/M/E表，主矩阵统一编号'
next_skill: deep-read Mode C
```
