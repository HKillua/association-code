# 099 同行机会判断：实现与本地验收

**后续方法审计更正：** 本文23项软件验收及简单CV可学性结果仍然成立，但不能据此认证DeepAF原方法已可靠复现。用户要求先核实方法后，099已改为HOLD_METHOD_QUALIFICATION；新增审计发现初始化坐标问题以及训练语义／原生行为验证缺口，详见[方法审计](2026-10-25-deepaf-method-audit.md)。旧READY状态不再有效。

## 决定及范围

用户要求两到三周内取得稳定的同行优势，但首先尽快判断是否有机会。当前可信证据只支持启动一轮有限的直接对照，不能先断言大概率超过同行。首个对手选择与098机制最接近、且为用户提供TMTAFF论文所列同行的DeepAF；保留P/V双网核心，公开fixed056位置任务适配差异。原文依据为[DeepAF作者全文](https://mdpi-res.com/d_attachment/aerospace/aerospace-12-00194/article_deploy/aerospace-12-00194.pdf)，Eq.(9)–(14)、§3.1–3.3及§4.1。

科学决定见[099合同](../specs/peer-feasibility-sprint-099.md)：72小时给机会四态、第7天决定当前配方是否继续，第14–21天只为显示优势的冻结方法确认第二同行、第三seed及封存测试。没有可信对手时记未解决；不能拿失败的同行适配记获胜。098原合同不改。

## 当前证据基点

截至本轮核查，远端main为`d05c6037bec06853c3b74ba9695c536c8ef708d9`，服务器最后回执时间2026-10-25 08:16:53Z。098A/B已提交运行，D/C等合法空槽，098尚无终值；不是实时GPU监控。090C／097A全人口位置MAE为564.834006／583.975593m，来自各自原生final-summary，并不能直接与不同指标、人口和协议的论文表格排榜。

- [090C原生终值](../../HandOver/results/LEVEL1-FROZEN-ADAPTER-20261024-090C/final-summary.json.gz)
- [097A原生终值](../../HandOver/results/LEVEL1-SEQUENCE-BUDGET-20261025-097A/final-summary.json.gz)
- [最新执行索引](../../HandOver/Handover.md)

## 实现与独立审阅

新增DeepAF模型、薄trainer、完整人口evaluator、实验入口、缓存比较／学习率选择器及四份预声明配置。P/V各6层encoder＋6层decoder、d32、4head、FF64，总计265732参数；直接linear输出，不继承我方权重。Cross-attention按原文Eq12负平方距离，self-attention保留scaled dot product。

固定组／Q64、球面局部米制、可见OLS启动、history5、scale10、dropout0、闭环TBPTT1、P desired-increment及GT位置差分V监督均公开为适配选择。GT只进入训练loss与事后评价，预测函数不接受reference或身份。速度／航向维持普通Mean；内部V参与运动递推，不把该分支称作已学会完整多属性融合。

A/B同seed、同初始化、同成功采样，两学习率各8000成功联合更新；源4000 batch重放两次，第二轮重置采样RNG而连续Adam。模式按source_step而非global step选择，避免4000非3整除的周期错误。固定T诊断仅用源第一个batch，在0/4000/8000记录loss与位置误差，不计训练曝光。CPU评价按八组批处理，所有原组、重复组和参考mask保留。

缓存比较逐值检查输入、组、truth、mask和普通控制，并从实际输出重算Haversine误差、四加权、全点分位、逐场／组件及最大组件外结果。A/B按DeepAF自身final8000全人口MAE选择更强者，C只复验该配方。对C如实记录V学习率选择；所有任务都不按V挑中途checkpoint。球面局部轨迹越过πR会被独立重算并标为几何质量未解决，不允许绕行折回掩盖失效。

独立审阅发现并已修复两处集成／解释问题：endpoint summary补齐评价scope；099C区分V选学习率与固定checkpoint，避免`validation_selection=False`误导。原论文仅明确Adam，学习率和8000预算不能冒称作者原值；同曝光不等于等FLOPs或充分收敛。

## 软件验收

统一执行：

```sh
python -m unittest tests.test_deepaf_peer_model tests.test_deepaf_training \
  tests.test_deepaf_peer_pipeline tests.test_deepaf_peer_comparison
```

**23项全部通过，35.907秒，退出码0。** 实际使用本地既有Torch2.5.1 CPU环境。

- 模型／单位：Eq12、双网结构、dateline／近极点往返、对跖拒绝、真实dt、prev-V和自身历史、padding／support、无GT输入、cache与replay一致、分块梯度归一、双网真实更新。
- 训练：真实source4→peer8，两轮batch与mode、持续Adam、0/4/8检查点与固定T probe、失败曝光、损坏checkpoint／replay拒绝、不覆写完成结果。
- 完整贯通：真实source4→强预算对照4→DeepAF8，三个完整CPU端点；每端点4个原始组与127个有效位置点，包含重复／未选组。真实peer_metadata、ours_metadata和cached comparison能读取实际产物，MAE与原缓存直接重算一致。fixture只替代生产规模／archive来源，不替代模型、训练、检查点、mask匹配或误差计算。
- 缓存对比：已知物理误差、输入／GT／mask／组篡改拒绝、强peer选择及固定平手规则、几何绕行检查、输出不覆盖。

交接中的2个Bash块与嵌入Python已做语法检查，没有执行任何服务器命令。

### 非生产的可学性检查

完整P/V、seed20261023、Adam .001、clip1、B2×Q64、真实20s间隔，反复训练两个固定合成CV组32次。已固化为`test_closed_loop_learns_two_simple_cv_groups_in_32_updates`，只断言有限且明显改善，不锁精确成绩。独立观察：总P/V目标7848.097527→0.098146，局部位置MAE15509.322m→43.272m，中途有振荡，没有选checkpoint或据此改配方。

另一本地CPU单线程B8×Q64、至多3成员形状的一次完整更新实测0.9296秒、进程RSS约321MiB。这些只验证可学性与实现成本量级，**不是Level1成绩、生产收敛、服务器GPU ETA或同行胜负证据**。正式吞吐由服务器最初100步回传，不以advisory作超时截断。

## 执行边界

正式Level1 archive、GPU训练和V300实际分数尚未在此环境执行。三臂最多24000更新、各一次、失败保全；无自动改lr、加步或补seed。099等待098四项领取后在同一提交锁内争取合法空槽，全局最多四个单卡任务，CPU阶段仍保留原领取。具体实施冻结、启动块及发布状态见[099交接](../../HandOver/L2S-20261025-099.md)。只有服务器服务／输出／S2L回执才能写已领取或运行。
