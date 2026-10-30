# 141：状态恢复高损害核查的指标入口失败 — Record → Squeeze

记录日期2026-10-30。**141真实执行FAILED/exit1，未完成任何场景，也未调用CV复放核。** 失败发生在首个T场的旧指标读取：代码把真实`methods[name].position.all`误读成平铺`position`，触发`KeyError: 'p95'`。这是可定位的技术失败，不是状态传播假设的负结果；137原结果和失败现场均保留。

## Record：执行事实

- package_type：EXPERIMENT_RECORD；package_version：research-pipeline-contract v1.0。
- experiment_id / package_id：EXP-STATE-RECOVERY-TAIL-PROBE-20261030-141。
- source_idea_id：IDEA-STATE-RECOVERY-TAIL-PROBE-20261030；plan_version1.0；[事前Idea/spec](../proposals/state_recovery_tail_probe_20261030.md)、[唯一执行合同](../../HandOver/L2S-20261030-141.md)。
- source_matrix：Mode C1.57，OD-P2/P3、F238、M107（AI/UN）、E136、C55；直接上游为[137正式Record/Squeeze](2026-10-30-observation-scale-review.md)。
- 研究类型：computational diagnostic，已保存输出的来源/传播核查；不是新模型训练或新方法比较。
- execution_status：FAILED/exit1；completed_scenes=0；replay_calls=0。记录完成不等科学任务完成。

原问题是：137观测尺度RTS的零支持高损害，是否确实由所选预测组anchor的原CV状态及其传播得到；哪些端点状态、时间距离、成员/目标对应或表示因素与损害共存？事前竞争解释包括传播速度/时距、首代表映射、成员选择、球面表示数值及输入/参考问题，未预先认定外推或观测尺度是因果根源。

### 输入、条件与实际执行

| 字段 | 固定计划与实际证据 |
|---|---|
| 冻结历史/checkout | dd75fd7；`/home/wangguangjie/djs/association-code-run141`。按原L2S发布，冻结源码的异常行与真实stderr一致；版本仅供历史定位 |
| 原状态输入 | `/home/wangguangjie/djs/association-code/runs/LEVEL1-OBSERVATION-SCALE-20261030-137-attempt-1` |
| 原生输入 | `/home/wangguangjie/djs/association-code/runs/LEVEL1-NATIVE-CV-STATE-20261029-127-attempt-1` |
| manifest | `/home/wangguangjie/djs/association-code-run137/configs/data/mtad-level1-scene3000.json`，与137记录的绝对路径绑定 |
| 新输出 | `/home/wangguangjie/djs/association-code/runs/LEVEL1-STATE-RECOVERY-TAIL-PROBE-20261030-141-attempt-1` |
| 唯一launcher | `/home/wangguangjie/djs/association-code/runs/LEVEL1-STATE-RECOVERY-TAIL-PROBE-20261030-141-launch-attempt-1` |
| 实际服务 | `mtad-0f7fadcfb4ab.service`；systemd、Linger=yes、Restart=no |
| 预算 | CPU2/taskset16–17、RSS2GiB/AS4GiB、1800s advisory；GPU禁用；OMP/OpenBLAS/MKL/NumExpr均2线程 |
| 程序时间 | 2026-10-30T02:08:07.205439+00:00 → 02:08:07.658377+00:00；北京时间10:08:07开始并在同秒结束 |
| 程序/launcher成本 | 0.454232821s / 1.231083094s；峰值61890560B。程序单调计时与时间戳差保留各自原口径，不混合扣算 |
| 退出 | run.exit_code=1、launcher.returncode=1；未超时/未超建议、无group cleanup；真实服务failed/failed、Result=exit-code、ExecMainStatus=1、MainPID=0 |
| 产物 | 终态观察时private_files=[]、summary_exists=false；没有逐场结果、case表、CV端点trace或完整指标汇总 |

实际payload如下；环境、cwd、launcher参数由[request原件](/Users/a1/Projects/biye-ronghe/docs/evidence/20261030_state_recovery_tail/141/launcher/request.json)保存，不用提交返回代替执行结果：

```bash
/usr/bin/taskset --cpu-list 16-17 /home/wangguangjie/djs/mtad-field-20261016/bin/python tools/analysis/state_recovery_tail_probe.py --state-run /home/wangguangjie/djs/association-code/runs/LEVEL1-OBSERVATION-SCALE-20261030-137-attempt-1 --native-run /home/wangguangjie/djs/association-code/runs/LEVEL1-NATIVE-CV-STATE-20261029-127-attempt-1 --manifest /home/wangguangjie/djs/association-code-run137/configs/data/mtad-level1-scene3000.json --output /home/wangguangjie/djs/association-code/runs/LEVEL1-STATE-RECOVERY-TAIL-PROBE-20261030-141-attempt-1 --time-policy advisory
```

### 人口与操作：计划值不能写成已完成值

| 项目 | 事前计划 | 实际 |
|---|---|---|
| 场景 | T32/12组件＋V32/9组件，共64场 | 首个T_core场景0的`inspect_scene`内失败；0完成场，V场循环尚未进入 |
| 完整有效位置点 | 原137 T37635、V71071保持原对应/有效mask | 没有141完整人口重建结果；上述只是旧137既有分母 |
| 零支持核查 | 预期T1487/V2770；完整保留、不删难例 | 未完成核对，不能将预期人口写为141实测通过 |
| 支持口径 | 预测组可用源数与真实identity各输入片段时间跨度并集分别重建 | 首场内部执行至指标绑定前；无持久化逐场核验，完整支持层仍未认证 |
| 两rho CV复放 | 仅零支持点涉及的唯一anchor，原两rho，最多4426核；保存端点与Q64绑定 | 0复放调用。源码异常发生在`traces`/复放循环之前，run零计数与空private相互一致 |
| 新训练/输出/搜索 | 0teacher、0神经更新、0target-reference拟合、0新方法预测、不选rho | run均为0，代码未进入相应新操作；没有修复或替代预测 |
| 终点 | 原人口/传播绑定、全部逐场/汇总/匿名case、Record→Squeeze | 只得到技术失败日志和终态；科学终点未达成 |

0完成场不等于0输入读取。首场已读取必要原缓存并执行部分对应/误差计算后才遇到异常，但未保存可复核的场级结果。不能据此声称首场所有前置数组已独立通过，更不能把失败前内存里的局部计算补为完整成绩。

### 异常证据及原因定位

[stderr原件](/Users/a1/Projects/biye-ronghe/docs/evidence/20261030_state_recovery_tail/141/launcher/stderr.log)明确指向冻结源码`inspect_scene`第222行：

```python
expected['methods'][name]['position'][key]
KeyError: 'p95'
```

本地只读复核原137 [T逐场原件](/Users/a1/Projects/biye-ronghe/docs/evidence/20261030_observation_scale/137/T-scene-summary.json.gz)：首场ID=`0`，各方法的`position`键为`all/observed/missing`，其中`position.all`才包含mae/mse/rmse/p95/p99等。因此不是原137缺少P95，也不是数值比较失败，而是141读取层级错误。`error_stats`先枚举p95，故该缺键首先暴露；不能据缺键发生顺序认为其他主指标已经逐一核过。

再读冻结测试`tests/test_state_recovery_tail_probe.py`第81行：人工fixture构造的是`position=probe.error_stats(...)`平铺字典，正好与错误消费者吻合，未覆盖真实137的嵌套布局。这解释了为何发布前3项软件检查通过仍没有发现入口不兼容。软件核原CV数学等价、映射及篡改检查的范围保留，但其通过不能认证真实缓存适配或科学完成。

日志里的两个相同异常栈分别来自内部记录与外层异常输出；同一launcher/payload一次退出，不代表两次科学尝试。没有证据支持OOM、网络故障、物理模型发散或超时导致本次终止。

### 偏差和证据边界

| 偏差 | 可直接支持的影响 | 不能推出 |
|---|---|---|
| 指标汇总读取层级错误 | 首T场入口终止，完整诊断不可用 | 观测RTS已被新诊断否定，或CV传播机制已查明 |
| 人工fixture与真实137schema不同 | 发布前覆盖遗漏；需真实schema消费检查 | 所有旧137指标或独立审计不可信 |
| 0完成场/0复放 | 科学主要估计对象未观测；没有新效应或区间 | 效应为0、无差异、全部输入均未读取 |
| 本项失败后保全 | 原run/launcher未覆盖，141不重新启动 | 后继142已经成功，或可以跳过新合同自动重试 |

原件完整入口：[run](/Users/a1/Projects/biye-ronghe/docs/evidence/20261030_state_recovery_tail/141/run.json)、[真实终态观察](/Users/a1/Projects/biye-ronghe/docs/evidence/20261030_state_recovery_tail/141/observed-service-terminal.json)、[launcher status](/Users/a1/Projects/biye-ronghe/docs/evidence/20261030_state_recovery_tail/141/launcher/status.json)、[launch提交记录](/Users/a1/Projects/biye-ronghe/docs/evidence/20261030_state_recovery_tail/141/launcher/launch.json)。终态观察时间2026-10-30T02:11:22.469877+00:00（北京时间10:11:22）。没有141独立数组审计“通过”结果；本次核验是原件/冻结代码/真实schema一致性复核。

## Squeeze：技术失败能提供什么，不能提供什么

- package_type：SQUEEZE_REPORT；package_version：research-pipeline-contract v1.0。
- package_id：SQ-STATE-RECOVERY-TAIL-ENTRY-FAILURE-20261030-141。
- input_record：EXP-STATE-RECOVERY-TAIL-PROBE-20261030-141；source_idea_id同上；范围仅本次失败事实与最小技术后继，不追加实验/服务器访问。

**直接观察DE：** 实际消费者要求平铺指标，而生产137提供分层指标；首个T入口因此失败。手造fixture复制了消费者假设，没有用真实生产schema约束它。这是一项技术接口事实，足以要求修正访问路径与测试覆盖。

**解释与归因：** 缺键异常由错误层级直接解释，可与“原137没有P95”区分；并非只凭猜测归因。科学层面的端点速度、时间距离、错误组映射和球面传播仍没有得到141逐点证据。此前137的风险集中事实保持原限定范围；其候选机制M107仍是AI/UN，不能因本失败升格或否定。

**本次不可做的深挖：** 没有新完整误差/trace，不能分析场景/组件/去最大、零支持SSE、before/after、identity跨度、top-case、rho差异、置信区间、收敛或物理机制。这里均是“未完成”，不是“无显著差异”。不把137旧表复制成141结果，不补造样本统计以填满流程。

**最小具体后继条件：** 另立142独立合同、新代码历史/checkout/run/launcher，仅修正指标消费路径，并用真实137公开嵌套schema验证消费者；保留数学复放、原两rho、完整T32/V32/零支持人口、全部比较/容差/预算和解释边界。真实schema的缺层/缺key应在明确入口拒绝，不能使用默认0、跳过字段或放宽容差掩盖不兼容。原141及失败原件不动，不能在其目录“接着跑”；142的真实结果需自行记录，不能倒填为141成功。本记录不声明142已提交或完成。

**Mode C回流建议：** 只追加“真实缓存消费者schema失配使141未执行科学检验”的技术事实/证据及已限定后继；保留C55/F238/M107/E136的科学状态，不新增物理机制或方法效果，不增加论文/精读数量。Record→Squeeze先完成，交独立矩阵代理消费，不能把完成两份记录当作科研假设得到支持。
