# 107现场核验：训练完成，固定组评价预检失败

2026-10-29由本地研究代理通过已授权ssh-campus现场读取；不是新训练、服务器补造回执或同任务重试。输入是原107运行和launcher原件，读取时任务已结束。当前安排只看[交接](../../HandOver/Handover.md)。

原运行：`/home/wangguangjie/djs/association-code/runs/LEVEL1-TMTAFF-UNIT-BASELINE-A-20261027-107-attempt-1`；冻结code为`association-code-run107`，未修改。原件下载在资料库的[107终态证据](/Users/a1/Projects/biye-ronghe/docs/evidence/20261029_small_validation/107/)，仅终态证据，不在资料库复制当前队列。

| 阶段 | 原执行事实 |
|---|---|
| TCM | COMPLETED，2424步、300场验证，退出0；2026-10-27 14:25:02–18:41:03 UTC，15,361.029s。 |
| TFM | COMPLETED，1000步、300场验证，退出0；2026-10-27 18:41:36–10月28日18:30:28 UTC，85,731.836s。初始化只转入冻结TCM，TAM unused。 |
| fixed056 | FAILED，退出1，validation_status NOT_STARTED；2026-10-28 18:31:43–18:56:44 UTC，1,501.280s。模型评价前`bind_scenes`读取090C旧NPZ的`sources`触发KeyError。 |
| launcher | FAILED/returncode1，非timed_out；总elapsed102,762.430s，UTC10月28日18:56:45结束（北京时间10月29日02:56:45）；Restart未重试。 |

可直接核原件：[pipeline-run](/Users/a1/Projects/biye-ronghe/docs/evidence/20261029_small_validation/107/pipeline-run.json)、[launch-status](/Users/a1/Projects/biye-ronghe/docs/evidence/20261029_small_validation/107/launch-status.json)、[退出](/Users/a1/Projects/biye-ronghe/docs/evidence/20261029_small_validation/107/exit_code.txt)、[fixed056-run](/Users/a1/Projects/biye-ronghe/docs/evidence/20261029_small_validation/107/fixed056-run.json)、[真实traceback](/Users/a1/Projects/biye-ronghe/docs/evidence/20261029_small_validation/107/fixed056.stderr.log)。巨大训练summary/history、checkpoint与完整数据留原服务器，不将本次文件存在/COMPLETED写成已认证同行性能。

现场只读NPZ字段目录确认：056 fragment有`sources`和`learned_assoc_membership`；090C final fragment没有这两字段，但有node_ids/identities、group offsets/node indices、native_states/support/nearest_seconds/exact_observation/query_seconds及group_anchor_node_indices。两者各300个文件。090C根run明确COMPLETED且source_system指向同056。故这是**旧缓存schema与新预检读法不兼容**，不是已观察到数值不一致、训练崩溃或数据坏。

修复必须保留真实绑定：056 sources对当前scene和原scores；090C元数据对同056、相同节点序及真实已有输入/分组；旧membership从完整offset/flat严格重建。缺字段不能填假来源/跳过人口核查。软件修复后只能独立新任务使用既有107固定checkpoint评价，0新增训练，保留本次失败原件；不自动重跑107整条管线。具体修复/核验另记，不在本报告倒填已完成。
