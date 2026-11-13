# 193阶段信息任务：GPU导入副作用导致启动失败

## EXPERIMENT_RECORD：EXP-TEMPORAL-GATE-193

2026-11-01，research-pipeline-contract v1.0；来源 IDEA-TEMPORAL-GATE-193/plan1.0，[原计划](../proposals/temporal_gate_20261101.md)、[完整原命令](../../HandOver/L2S-20261101-193.md)，computational pilot，执行状态 interrupted/technical-failure。目的仍是固定178选择/O和MC两估计器后，比较同容量MEAN/BAG/ORDER的任务风险；不是固定两传感器贡献配方。原计划T128/39fit/4992GPU更新、T32/44臂/0V、固定末步与原28强控，失败后不重投193。

### 实际输入、操作和终态

- 冻结checkout `/home/wangguangjie/djs/association-code-run193`，代码8f92108；输入原178/189/183，T名单、标签和公开配置未修改。Python3.11/TorchCUDA12.1，3090 GPU0按UUID，CPU8–11×4/RSS8GiB/CUDA1GiB。
- 原run `runs/LEVEL1-TEMPORAL-GATE-20261101-193-attempt-1`、原launcher同前缀 `-launch-attempt-1`，均在服务器主仓；unit `mtad-f17ac121a4fe.service`，真实 FAILED/exit1/failed/failed/Result=exit-code/Restart=no，未运行规定成功后审计。原件及checkout未改。
- payload UTC 2026-10-31 22:09:51.509723至22:10:00.793806（北京11月1日06:09:51–06:10:00），launcher9.422683402895927s；science4.556723836809397s、峰RSS490758144B、CUDA已分配峰0。
- 实际0新fit/0更新/0完整T或V评价/0成功冻结风险前向。**已完成首场157节点/43989事件的输入阶段描述及157次native均值计算和临时MC候选，未保存/尚未checkpoint**；run.json的native_mean_computations=0不能解释为实际零计算。没有新MAE/尾部/覆盖比较或CI，所有三组效果均未知，不将技术失败列作方法阴性。
- 错误位于第一次 `predict_residual(...).to('cuda:0')`，`RuntimeError: No CUDA GPUs are available`，尚无optimizer调用。9项本地软件检查虽通过，但未检查实际训练入口的设备环境副作用；未将其冒充GPU验收。

### 原件与血缘

[脱敏原件及传输清单](/Users/a1/Projects/biye-ronghe/offline_observation/docs/evidence/20261101_temporal_gate/193)含13文件/11185原始字节，tar4015B；逐文件数量/长度及压缩包长度核齐。science/run.json、完整launcher/request/launch/status/exit/stdout/stderr、cuda-import-evidence.json均保留；原坐标/身份/权重/大数组未外传。真实命令以request和L2S为准，Git仅定位。额外只读设备能力与导入探针不计入launcher，单项墙时未单独捕获，0训练，成本缺口如实保留。

## SQUEEZE_REPORT：SQ-TEMPORAL-GATE-FAILURE-193

消费上述EXP及完整stderr/配置/有界导入探针，证据 verified-technical，科学效应 none。DE：同一Python新进程在unset/数字0/实际GPU0 UUID下都可见CUDA；同样taskset与UUID下导入训练入口后环境从UUID变为空字符串，最小一元素CUDA分配复现错误。调用链为 temporal_gate_pilot → drift_memory_pilot（只想复用ARMS常量）→ history_tail_qualification → decoded_coverage_closeout → audit_shared_control_arrays 第26行清空CUDA_VISIBLE_DEVICES。由代码副作用及干预前后直接定位技术原因，非服务器坏或几何假设被反驳。

|检查维度|发现/边界|
|---|---|
|过程|首场输入描述后第一次模型搬GPU失败；无训练收敛/组别效应可分析。|
|机制|导入审计模块导致训练设备被隐藏；去除训练入口对该常量的间接审计依赖是最小修复，不修改审计原有CPU行为。|
|错误/稳健性|父进程测试已导入CPU审计，无法观察导入前后变化；需全新子进程给定CUDA环境的回归检查，再在新冻结checkout做实际GPU分配。|
|科学解释|阶段信息/顺序是否有用仍UN；193既不支持也不排除C68，原178/MC条件效果不变。|
|偏差|实际未达39fit/4992更新/T32/44臂；无删点/改配方/换CPU继续，原失败完整保全。|

下一先回Mode C2.06/F302/E205（无新M/C/论文计数），再由C68形成**独立194技术修复合同**，只改入口依赖/前置设备可见性与失败血缘，保持科学配置、种子、拟合/更新总量、T/V范围、指标与停止规则。新任务一次尝试、失败仍保全；不自动进入V/全300。记录不是对193自动重试的授权，具体194须新READY。

```yaml
package_type: SQUEEZE_REPORT
package_version: research-pipeline-contract v1.0
package_id: SQ-TEMPORAL-GATE-FAILURE-193
input_experiment_record_ids: [EXP-TEMPORAL-GATE-193]
source_idea_id: IDEA-TEMPORAL-GATE-193
plan_version: '1.0'
claim_scope: import_side_effect_technical_failure_only
primary_result_with_uncertainty: no_scientific_result_0fits_0updates_0complete_T_0V
process_dynamics: first_157_native_means_uncheckpointed_before_CUDA_transfer_failure
new_P: none
new_F: F302
new_M: none
new_E: E205
open_questions: [phase_information_effect, order_increment, complete_risk_tail]
next_routes: deep-read_Mode_C_then_separate_repair_Idea
next_skill: deep-read_Mode_C
```
