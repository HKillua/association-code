# 当前服务器任务

**当前仅042为新READY：035 selected600同输入、零更新模式核查。** 未收到领取/在途回执，发布不等于运行。035、037、039—041均已COMPLETED并复核；历史L2S的READY不构成待办，022—025继续HOLD。 **（服务器补注：042已于`17:18:34Z`领取并于`17:19:03Z`COMPLETED／退出0，见[S2L-042](S2L-20261020-042.md)；**当前无可领取READY**，043—046仍为DRAFT未领。四张卡已无我方进程。）**

| 当前可领取任务 | 范围与预算 | 输出与依据 |
| --- | --- | --- |
| [042 TAM模式核查](L2S-20261020-042.md) | 原035 selected600，固定T_core/V小批，三模式MC8及一次默认fastpath参照；一次单卡FP32/CPU4、0更新，30分钟仅提醒，失败不重试 **（服务器补注 `17:19:03Z`：COMPLETED／退出0，墙钟27.726秒、payload 26.686秒，advisory未触发；块逐字30行1,559字节 SHA256 `cbe5b55f…99e4d7`、零替换；树@3999957、索引0独占。`optimizer_updates=0`、`reference_read=false`、`test_opened=false`、`threshold_reselected/margin_recalibrated/normalizer_refit`全false、repeats 8、seed 2035；selected原样读出update600、margin 75.2613410949707、threshold −2.635511918924749e−05、macro_f1 .5128651743707023；四样本＝T_core 2606/1622＋V_select 1671/2056各首4正4负、shortfall 0；**状态恢复八项全true**。读数：eval 正1.9640e−05／负5.6690e−05；train 正**76.7573**／负**76.7046**（比值1.0007，几乎不可分）；train全关dropout 与 eval **逐位相同**，四样本 `max_absolute_pair_distance_difference_eval_vs_train_zero` **全为0.0**；默认fastpath参照差仅1.3e−6~2.5e−6。⇒ train/eval 差被机械隔离到 dropout 一项，是否即为035精度与loss平台成因、要不要动dropout，按草案归本地，我未改模型或加训练。原件`run/preparation/summary.json.gz`均cmp自证＋`distances.npz`（48数组）已入库，源权重未重传。）**| 新`TAM-MODE-CHECK-20261020-042-attempt-1`及同名`-launch`，完整命令见独立L2S；无reference、校准/阈值选择、下游或T_test |

R2原生运动及关联衔接已经在本地CPU完整V完成，**无需服务器重跑**。OLS位置MAE8849.53→4737.59m；新增同评分全29008对关联F1 .61563→.72717，共用保持阈值也改善，但14场F1退化。见[原生运动](../docs/proposals/repair_before_retraining/native-motion-analysis.md)和[关联衔接](../docs/proposals/repair_before_retraining/motion-association-analysis.md)；普通离线控制，不是新训练、学习贡献或融合成绩。

本地随后完成[补全锚点/局部趋势消融](../docs/proposals/repair_before_retraining/completion-motion-ablations-analysis.md)，零训练/完整V/退出0；两种修改均未改善补全，不替换原底座，不需服务器重跑。此结果不新增READY，042及原额度不变，补全下一训练尚未发布。

2026-10-20—21本地已完成[OLS模型接入、R3匿名校准、R4固定组融合](../docs/proposals/repair_before_retraining/local-readiness-analysis.md)。R3/R4均为完整V、零更新、已独立复核的CPU控制，**不要在服务器重跑**。匿名校准尚无可靠分组收益；正确helper有明显价值，但预测组污染及可靠观测破坏仍待改进。候选OLS底座已可训练且通过CPU检查，未新增真实训练结果。

[043仅改OLS底座](L2S-20261020-043.md)的配置、命令、一次teacher1000步与完整V评价方案已备齐，**DRAFT，不在可领取名单，不增加已批预算**。用户本轮要求完成本地工作，不等于已批准该GPU训练。旧权重不得热切换；后续仍由本地决策/写代码，服务器执行/回传。

## 已备齐、待用户指定执行的方案

本地按用户“继续推进，本地改完再让服务器做实验”完成[证据关联/效用融合](../docs/specs/evidence-association-fusion.md)代码。下面均为 **DRAFT**，不计作已批准GPU预算。用户可以直接在服务器端明确批准所选编号；服务器记录该授权原话、更新本索引当前可领取名单后执行，不需把同一授权转回本地重复确认。未收到明确批准时不自动启动。

| 编号 | 固定实验 | 次数与范围 |
| --- | --- | --- |
| [043](L2S-20261020-043.md) | 候选补全仅改OLS底座，学习修正是否胜过自身底座 | teacher1000步一次；原4小时提醒方案 |
| [044](L2S-20261021-044.md) | 几何特征关联控制 | 1000步一次；1小时提醒 |
| [045](L2S-20261021-045.md) | 同模型增加运动/支撑证据关联 | 1000步一次；1小时提醒；与044比较 |
| [046](L2S-20261021-046.md) | 固定原OLS分组的属性效用融合，允许拒绝helper | 1000步一次；1小时提醒；固定最终端点 |

042先完成既有baseline诊断；若用户批准候选，建议044/045成对进行，随后046；043可按名额独立推进。它们互不依赖新权重，不启动任何后续联合训练。全部获批也仍最多两个单卡并行，每项一次、失败不重试、时长只提醒。044/045读取全部T_core 18,249,384对并分块预计算；046使用同一固定运动与原分组，无需等待045输出。新头的真实GPU耗时尚未测量，记录准备、前50步和完整V耗时，不套用Transformer小时估计。推送不等于服务器收到启动指令。

## 本轮结果

| 任务 | 已核终态与研究含义 |
| --- | --- |
| [035 baseline关联](S2L-20261020-035.md) | TAM1000、退出0、5.34小时；固定033 TCM4000，V最佳在TAM600（macro-F1约.5129），最终1000步约.4916。仍低于同V位置控制.6156；无TFM/JOINT或T_test。[分析](../docs/proposals/repair_before_retraining/035-analysis.md)。 |
| [037圆周TCM](S2L-20261020-037.md) | TCM4000、退出0、2.58小时；本空间自然query特征误差比hold低23.78%。新旧特征空间不能直接按MSE或相对hold比率排名；没有新TAM或物理结果。[分析](../docs/proposals/repair_before_retraining/037-analysis.md)。 |
| [039 pre-norm](S2L-20261020-039.md) | teacher1000及一次CPU动态读取完成；时间变化幅度恢复，位置MAE却由034的8924米增至9075米。机制读数改善不等于补全改善。 |
| [040两层decoder](S2L-20261020-040.md) | teacher1000及CPU读取完成；参数1395599，比034少36.3%；位置8915米，仅小幅改善且仍差于底座8849.75米。 |
| [041去身份BCE](S2L-20261020-041.md) | teacher1000及CPU读取完成，BCE加权贡献逐步为0；速度MAE19.727→18.414，但时间输出仍近常量，不称已训练关联臂。 |

三臂详细结论及完整人口、观测控制、消动态、来源与成本见[039—041分析](../docs/proposals/repair_before_retraining/039-041-analysis.md)。不自动合并消融改动、不把单属性小幅收益宣布为完整方法成功。

## 下一步与执行边界

当前[研究路线](../docs/research_brief.md#接下来怎么推进)：R1已发布042；R2/R3/R4本地控制已完成；043候选底座训练为待授权草案。044—046已经实现对应可靠性与可拒绝属性效用的最小实验，不直接采用无收益的迭代校准；待用户安排真实训练。这里只登记明确发布的执行任务，研究代号和配置存在均不构成READY。

- baseline优先问题是035同输入的TAM train/eval距离差异：V距离缩到约1e-5，训练loss约19、margin75.26；现有不同人口记录不能确诊dropout或loss错误。042现已实现并发布，先读结果再决定是否固定037按035方案新训TAM；新TAM未授权，不追加035或进入TFM/JOINT。 **（服务器观察，仅事实与缺口，不代填方案：042 已COMPLETED／退出0、27.726秒，同一份035 selected600权重、同一批缓存输入、每模式8次重复。实测三件事——① **把train模式的dropout概率全部归零后，逐对距离与eval模式完全相同：四个样本的 `max_absolute_pair_distance_difference_eval_vs_train_zero` 全部为 `0.0`（逐位，不是近似）**；② dropout 一生效，正对均值从 1.9640e−05 抬到 **76.7573**、负对从 5.6690e−05 抬到 **76.7046**，正负比值 **1.0007**（几乎不可分）；③ 默认fastpath与受控eval的距离差仅 1.3e−6～2.5e−6，比dropout效应小约七个量级。`state_restoration` 八项（参数与buffer／梯度／requires_grad／模块模式／dropout概率／fastpath／CPU RNG／CUDA RNG）全为 true，`optimizer_updates=0`、`reference_read=false`、`test_opened=false`、阈值与margin与normalizer均未重估（selected600 原样读出 margin 75.2613410949707、threshold −2.635511918924749e−05、macro-F1 .5128651743707023）。⇒ 本会话此前记录的“V距离缩到约1e-5”这一现象，与train／eval落差同源：train／eval 差在机械上等于dropout项，评估侧的绝对尺度则是所有pair距离挤在零附近。**缺口**：043／044／045／046 四条草案都不覆盖 train／eval 一致性（dropout 概率或与之匹配的归一化尺度），因此这项修复目前无草案可领；按分工我不写配置、不自行加实验，也不据本轮自动去dropout重训——若用户决定推进，需本地出具体的单因素方案与预算。另提示两条已回传证据可并列参考：039 pre-norm 把末层时间幅度提升约1,138倍但位置MAE反劣151米，041 关BCE末层幅度更低（0.0002248），说明“幅度”与“精度”在候选线里并不同步。详见[S2L-042](S2L-20261020-042.md)。）**
- 候选三组没有建立整体补全优势，先据同人口与动态净收益选择下一研究改动，不因hidden RMS变大就替换默认模型或加训。关联、融合设计不要求补全极致，但新的核心方法/真实执行仍需具体方案。
- 本轮S2L转述用户在服务器端允许四卡调度，实际035/037/039/040/041共五个已批作业使用四卡，039/041共卡；记录实际资源使用，不从回执扩展新实验或重试授权。不同共卡程度不作严格速度基准。
- 服务器监督脚本漏`-attempt-1`导致回报延迟，任务本身正常结束。之后直接使用实际提交记录中的完整输出/record-dir，首次状态缺失先核路径，不空循环等待；不为此再建监控框架或重跑任务。

原L2S/S2L、失败原件和服务器权重保留；本地只更新索引与分析，不改服务器原回执。GPU允许兼容共卡，不触他人进程；时长仅建议，超建议报告并询问，等待期间继续原固定工作，不自动加步或重试。共享A隔离与T_test封存。push不唤醒服务器，本地不SSH或创建服务器守护。历史与结果入口见[分析索引](../docs/proposals/repair_before_retraining/README.md)、[历史摘要](../docs/history/README.md)。
