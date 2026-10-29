# 128：同信息关联诊断的实际中断记录与深挖

2026-10-29。EXPERIMENT_RECORD `EXP-128-20261029`，来源 `IDEA-ASSOCIATION-GEOMETRY-SIGNAL-20261029`/1.0；SQUEEZE_REPORT `SQ-ASSOCIATION-GEOMETRY-FAILURE-128-20261029`。记录状态 **interrupted / data-incomplete**；科学主要终点未达到，不能写COMPLETED、无差异或方法失败。[事前Idea](../proposals/association_geometry_signal_20261029.md)、[实际合同](../../HandOver/L2S-20261029-128.md)。

## Record：计划与实际

原计划固定V32/9组件、同056全部pair/114 WS候选、十个冻结scorer及共同可定义人口、两成员球面恒等与指定多成员欧氏重构；0模型/前向/训练/新分区/位置输出/reference数组/T_test。CPU8–11/4线程、RSS4GiB/AS8GiB、600s advisory，一次attempt-1。

实际冻结834c5aa、独立`association-code-run128`、systemd `mtad-6b9ec1dd0a50.service`，Linger=yes/Restart=no。原run为`/home/wangguangjie/djs/association-code/runs/LEVEL1-ASSOCIATION-GEOMETRY-SIGNAL-20261029-128-attempt-1`，原launcher同名`-launch`。真实request保存完整命令/输入/环境。

- 仅前6场 `692,700,864,1187,1293,1582` 完成逐场feature/evaluation/公开row；下一场1830有551节点/151525 pair，未完成公开row。不是完整V32，也不是可任意当代表性成功子集。
- 用户服务日志明确：北京时间22:32:56（UTC14:32:56）`systemd-oomd killed 3 process(es) in this unit`，主进程code=killed/status9，Result=signal、ActiveState=failed/SubState=failed。worker和payload PID均已不存在。确认是oomd终止，不从其推断物理模型或数据质量。
- launcher/status与run残留RUNNING，没有exit_code.txt和summary。这些是失去终态写盘的陈旧文件，**不能据此继续认定运行或补写退出0**；payload独立退出码未知，服务主进程的signal9是实际可核证据。
- 服务起止时间22:32:03→22:32:56，约53秒；服务累计CPU51.911492秒。run最后checkpoint为19.151389秒、当时peak_rss2010796032B；两者不是完整程序墙时或最终RSS峰值。MemoryMax=infinity，未取得完整峰值；不能拿最后采样否认后续瞬时内存，也不能断言触发了本脚本4GiB检查。
- 原frozen checkout干净。没有重启/重跑128、放宽容差、删场或覆盖原文件；后验另存terminal-synthesis、服务原日志与6场partial-summary，明确其后验来源。private坐标/身份/完整几何仍在服务器。

## Squeeze：可提取和不可提取的信息

**DE/技术观察：** 服务状态与静态run文件可以分离；本次失败需要服务日志而非RUNNING字样判定。完整pair距离旧实现一次广播多个`P×64×2`中间数组，理论上会显著增加瞬时工作集；1830最终距离矩阵本身约77.6MB，但多个临时数组和原框架驻留量不能由它代表。

**AI/UN：** 日志识别到oomd终止；没有完整瞬时峰值、oomd具体选择理由/压力时间序列，因此不把广播分配宣称唯一根因。数据规模是可修复的工程条件，不是本领域尚未解决的科学机制。

**主要研究问题仍UN：** 全V32候选几何可辨性、跨组件稳定性、缺失人口及LOO/静态控制比较均未得到完整结果。不能按前6场排名决定新头/新机制，不能把中断写为LOO不work或关联失败。

**下一项具体修复：** 原128保持失败；在独立130 Idea/新目录下保留全部公式/候选/阈值/容差/数据人口，仅将pair几何的临时广播改为4096边分块，直接导入原6场完整feature/evaluation/row且exact核对，只计算剩余26场。完整结果再汇总32场。它是记录回流后的独立修复合同，不是原attempt自动重试；不改变oomd配置、不触他人进程、不保证分块必定解决所有资源问题。

## 原件与交接

[必要脱敏证据](/Users/a1/Projects/biye-ronghe/docs/evidence/20261029_small_validation/128/)包含原run、原launcher request/launch/status及空日志、posthoc服务状态/原journal、terminal-synthesis和6场partial-scene-summary。**不存在的summary/exit_code不补造。** 真实全部feature/identity/几何只留服务器。

```yaml
package_type: EXPERIMENT_RECORD
package_version: research-pipeline-contract v1.0
experiment_id: EXP-128-20261029
source_idea_id: IDEA-ASSOCIATION-GEOMETRY-SIGNAL-20261029
plan_version: '1.0'
execution_status: interrupted
research_question: 固定候选上的同信息pair/LOO可辨性与完整缺失人口
hypothesis: 可能存在表示价值，条件动作增量尚未验证
planned_vs_actual_design_and_controls: 原32只完成前6，科学终点未达；未改配方
actual_configuration: 上文真实request/冻结834c5aa/CPU4/4GiB-RSS检查及8GiB-AS/600s advisory
samples_materials_or_data_provenance: 原056/114/manifest V32，6场完成不是全体
results_with_uncertainty: 无完整科学主结果；仅技术中断和6场中间证据
all_runs_or_repetitions: 一次attempt-1，服务被oomd SIGKILL，不自动重试
observations_and_deviations: 原run与launcher陈旧RUNNING，独立payload退出码未知
known_limits_and_missing_evidence: 最终峰值/具体oomd选择理由/完整V32比较均缺
next_skill: squeeze-experiment
```

```yaml
package_type: SQUEEZE_REPORT
squeeze_id: SQ-ASSOCIATION-GEOMETRY-FAILURE-128-20261029
input_experiment_record_ids: [EXP-128-20261029]
claim_scope: 运行/资源技术条件；不支持科学正负结论
primary_result_with_uncertainty: systemd-oomd终止，完成6/32，科学主结果缺失
new_P: none
new_F: 技术中断及RUNNING元数据陈旧，非领域新现象
new_M: 广播工作集可能贡献资源风险AI，完整因果未知
evidence_index: 原journal/服务状态/原run及partial文件
open_questions: 有界内存实现后完整同信息信号
next_routes: deep-read Mode C后独立修复Idea
next_skill: deep-read Mode C
```
