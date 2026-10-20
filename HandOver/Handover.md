# 当前服务器任务

**当前仅042为新READY：035 selected600同输入、零更新模式核查。** 未收到领取/在途回执，发布不等于运行。035、037、039—041均已COMPLETED并复核；历史L2S的READY不构成待办，022—025继续HOLD。

| 当前可领取任务 | 范围与预算 | 输出与依据 |
| --- | --- | --- |
| [042 TAM模式核查](L2S-20261020-042.md) | 原035 selected600，固定T_core/V小批，三模式MC8及一次默认fastpath参照；一次单卡FP32/CPU4、0更新，30分钟仅提醒，失败不重试 | 新`TAM-MODE-CHECK-20261020-042-attempt-1`及同名`-launch`，完整命令见独立L2S；无reference、校准/阈值选择、下游或T_test |

R2原生运动及关联衔接已经在本地CPU完整V完成，**无需服务器重跑**。OLS位置MAE8849.53→4737.59m；新增同评分全29008对关联F1 .61563→.72717，共用保持阈值也改善，但14场F1退化。见[原生运动](../docs/proposals/repair_before_retraining/native-motion-analysis.md)和[关联衔接](../docs/proposals/repair_before_retraining/motion-association-analysis.md)；普通离线控制，不是新训练、学习贡献或融合成绩。

本地随后完成[补全锚点/局部趋势消融](../docs/proposals/repair_before_retraining/completion-motion-ablations-analysis.md)，零训练/完整V/退出0；两种修改均未改善补全，不替换原底座，不需服务器重跑。此结果不新增READY，042及原额度不变，补全下一训练尚未发布。

2026-10-20—21本地已完成[OLS模型接入、R3匿名校准、R4固定组融合](../docs/proposals/repair_before_retraining/local-readiness-analysis.md)。R3/R4均为完整V、零更新、已独立复核的CPU控制，**不要在服务器重跑**。匿名校准尚无可靠分组收益；正确helper有明显价值，但预测组污染及可靠观测破坏仍待改进。候选OLS底座已可训练且通过CPU检查，未新增真实训练结果。

[043仅改OLS底座](L2S-20261020-043.md)的配置、命令、一次teacher1000步与完整V评价方案已备齐，**DRAFT，不在可领取名单，不增加已批预算**。用户本轮要求完成本地工作，不等于已批准该GPU训练。旧权重不得热切换；后续仍由本地决策/写代码，服务器执行/回传。

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

当前[研究路线](../docs/research_brief.md#接下来怎么推进)：R1已发布042；R2/R3/R4本地控制已完成；043候选底座训练为待授权草案。下一学习型关联/融合设计须处理R3/R4揭示的对应可靠性与观测保护问题，不直接采用无收益的迭代校准。这里只登记明确发布的执行任务，研究代号和配置存在均不构成READY。

- baseline优先问题是035同输入的TAM train/eval距离差异：V距离缩到约1e-5，训练loss约19、margin75.26；现有不同人口记录不能确诊dropout或loss错误。042现已实现并发布，先读结果再决定是否固定037按035方案新训TAM；新TAM未授权，不追加035或进入TFM/JOINT。
- 候选三组没有建立整体补全优势，先据同人口与动态净收益选择下一研究改动，不因hidden RMS变大就替换默认模型或加训。关联、融合设计不要求补全极致，但新的核心方法/真实执行仍需具体方案。
- 本轮S2L转述用户在服务器端允许四卡调度，实际035/037/039/040/041共五个已批作业使用四卡，039/041共卡；记录实际资源使用，不从回执扩展新实验或重试授权。不同共卡程度不作严格速度基准。
- 服务器监督脚本漏`-attempt-1`导致回报延迟，任务本身正常结束。之后直接使用实际提交记录中的完整输出/record-dir，首次状态缺失先核路径，不空循环等待；不为此再建监控框架或重跑任务。

原L2S/S2L、失败原件和服务器权重保留；本地只更新索引与分析，不改服务器原回执。GPU允许兼容共卡，不触他人进程；时长仅建议，超建议报告并询问，等待期间继续原固定工作，不自动加步或重试。共享A隔离与T_test封存。push不唤醒服务器，本地不SSH或创建服务器守护。历史与结果入口见[分析索引](../docs/proposals/repair_before_retraining/README.md)、[历史摘要](../docs/history/README.md)。
