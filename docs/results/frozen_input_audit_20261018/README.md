# 冻结训练/开发池：数据审计最终摘要

2026-10-18的原生时间、监督和采样CPU审计已完成；2026-10-19合并重复过程文字。只读冻结T_core512/V_select64，没有模型前向、训练、GPU或T_test payload读取。007原manifest已由012取回并核对，真实CLI inspect-only装配亦完成；不存在“等manifest才能继续”的未结事项。当前生效规则见[数据契约](../../data_contract.md)。

## 人口、选点与时间

| 统计 | T_core | V_select |
| --- | ---: | ---: |
| 场景 / 有效轨 / 隔离轨 | 512 / 54,792 / 1,873 | 64 / 1,635 / 51 |
| 原生点 | 2,938,785 | 47,303 |
| cap128保点率 | 78.62% | 97.35% |
| Q64零/单支撑轨 | 567 / 1,798 | 8 / 62 |
| 正候选 / 负候选 | 110,733 / 18,138,651 | 3,297 / 25,711 |

所有有效片段至少两真实时刻，cap128保真实首尾、零空输入；零共同查询支撑不代表原生输入丢失。训练cap64/256分别保60.76%/90.89%原生点，full需最大2177点；这些对照不是变更授权。cap128省去21.38%训练点，不能拿53锚点18.13%的局部损失代替。最大保留点间隔909秒，抽点会加大内部间隔。

原生相邻间隔中位数训练9001/9002为30/15.722秒，V为67/34.003秒；当前共同Q64间隔中位约114秒。20秒仅是时间编码单位。FP32转换未破坏native/query各自严格递增或支撑归属，最大舍入0.000304699秒；训练cap128仍有1个不同内部native/query时刻转FP32后重合，不能称全部时间无碰撞。

## 时钟、reference复用与监督

576场景偏移按已固定规则取标签涉及身份的最早reference时间，独立用同身份唯一四状态匹配反推核验；不是按模型误差搜索。73场景偏移非零，最大2453秒。有效9001节点的1,094,244训练点＋18,093开发点，共1,112,337点，反推偏移均一致且对齐后四维原值与reference完全相同。结论不涵盖被隔离节点、其他传感器或全库，不能作为网络效果；限制观测区降噪主张。

| 监督人口 | T_core | V_select |
| --- | ---: | ---: |
| 标签目标 / 目标×Q64 | 12,699 / 812,736 | 377 / 24,128 |
| gap600有效lat/lon/vel查询 | 623,895（76.7648%） | 18,364（76.1107%） |
| gap180有效lat/lon/vel查询 | 470,395（57.8780%） | 12,516（51.8733%） |
| gap180全无监督训练身份 | 57 | 0 |

gap600无监督查询中，训练99.4657%、V99.8439%来自query超出该身份实际reference首末支撑；其余为属性无效/同刻冲突，非缺整份参考或此次时钟失败。训练cou有效数比前三维多2，其余mask相同；gap600每标签身份每属性至少一个合法query。逐身份/属性原因分类与生产对齐mask逐位一致，互斥计数加回全部人口。

gap180的57个零监督身份中53个仍有正候选，213轨、366正对。既有seed模拟未遇整批全零，不能说改180必失败；也不能为回避无监督删身份/重抽。用户已确认cap128/time_unit20/offset/gap600，600秒插值可能平滑真实转弯的风险仍须披露。

## 抽样覆盖不等于充分训练

原scene→pair平衡分布的正负eligible场景均512。两步成本测量实际TCM仅见127轨；TFM仅见2场景、67轨、96独立pair。按99%期望场景覆盖的算术阶梯模拟，TFM仍只见36,972/54,120条可正配对轨（68.31%）；TAM只触及约0.50%全pair。场景抽到不等于学完，覆盖阶梯不是收敛要求或预算授权，不要求穷举全部配对。

## 原始证据与失败保留

- [原生输入/覆盖汇总](native_sampling/attempt2/results.json)、[逐轨端点元数据](native_sampling/attempt2/scene_metadata.jsonl.gz)、[gap180采样联结](native_sampling/attempt2/gap180_sampling_risk.json)。全部576场景成功分析40.5432秒，退出0；首attempt遗漏intervals在首场景失败，原[source](native_sampling/audit_native_sampling_failed_attempt1.py)、[结果](native_sampling/results.json)、[stderr](native_sampling/stderr.log)、[退出码](native_sampling/exit_code.txt)保留，修复后独立attempt2，不覆盖。
- [时钟/参考核验](../frozen_clock_audit_20261018/frozen_clock_supervision_audit_20261018.json)、[逐场景明细](../frozen_clock_audit_20261018/frozen_clock_supervision_scenes_20261018.jsonl)、[冻结名单来源](../frozen_clock_audit_20261018/frozen_scene_lists_20261018.json)。原审计64.1865秒，退出0。
- [监督互斥分类](supervision_gap_causes_20261018.json)、[逐场景明细](supervision_gap_causes_scenes_20261018.jsonl.gz)、[零监督身份补核](supervision_gap_zero_targets_20261018.json)。分类47.2703秒、补核9.8892秒，均退出0；只补读36个已定位场景，没有重复扫描其余540场景。
- [原manifest实际CPU装配汇总](../evidence012_local_review/native_inspect_summary.json)与[012核验](../evidence012_local_review/README.md)：512/64场景、54,792/1,635轨一致，无模型/GPU。
- [当时CPU回归日志](local_cpu_regression.log)与[命令范围](local_cpu_regression.json)留作软件历史，不当作模型效果。

root曾独立核验576条明细、每feature分类总数、各cap保留＋省去点数、正负候选和轨数一致，未重读CSV。原分析脚本/命令/退出码全部原路径保留，部分依赖其记录的研究目录路径；这是证据档案，不是服务器待执行任务，也无需为了文档整理再次扫描。
