# 2026-11-10 仓库瘦身与入口整理

用户要求先整体瘦身，再重新梳理。此次仅整理存储、退役隔离工具和纠正导航；研究结论、模型计算、数据规则及实验预算不变。整理前主仓为 `a63f12a5f56e34ff3b0e68a4182a78312ef50f5e`，本机及服务器 main 均干净并与 origin 一致。

## 范围与核验

|操作|完成条件|核验方式|
|---|---|---|
|清理明确生成缓存、196重复临时回传|正式原件保留，临时引用改指正式位置|删除前逐文件比对；gzip比较解压正文|
|本机实验JSON透明压缩|原路径、完整字节、权限和修改时间保留，普通JSON读取不变|APFS/ditto先小样验证，再逐文件比对内容；仅在分配空间减少且源文件未变时原子替换|
|退役结案隔离工具及专属测试|剩余代码无导入/动态调用缺口，历史科学材料仍可定位|全仓引用/AST依赖核验、受影响测试及历史源码链接|
|整理方法和工具入口|MTAD量测、Q64、WS分组及恢复链路准确；旧组件明确范围|对照实际入口，保留原科学正文，检查相对链接|

环境、原始数据、权重、预测、服务器runs、冻结checkout、未审152结果、132失败现场、矩阵、Idea及Record/Squeeze全部保留。服务器只通过SSH MCP核对现场、清理主仓代码缓存和同步干净main，不改冻结运行目录。本轮没有新科学实验或历史重跑。

## 完成记录

已完成以下处理；字节数按实际文件/分配空间统计，GB为十进制。

|处理|实际结果|
|---|---|
|本机资料库原件透明压缩|仅`docs/evidence/**/*.json`中≥256 KiB、非软链接/硬链接、尚未压缩的113文件。逻辑正文1,302,172,095 B不变；磁盘分配1,302,413,312→103,161,856 B，释放**1,199,251,456 B（约1.20 GB）**。每文件复制后与替换后均校验完整字节，并保留权限/owner/mtime；路径和JSON格式不变。逐文件记录见[存储核验清单](/Users/a1/Projects/biye-ronghe/docs/evidence/storage-maintenance-20261110.jsonl)。这是本机APFS存储收益，不是Git克隆体积或服务器runs减少。|
|196重复临时回传|删除4文件/5,827,755 B。正式原件留在[196/run](/Users/a1/Projects/biye-ronghe/offline_observation/docs/evidence/20261103_action_risk_validation/196/run)；3文件同字节，summary.gz解压正文相同、压缩头不同。报告改指正式来源，原回传SHA语义保留。|
|可再生Python缓存|本机代码与资料库evidence共114文件/2,220,625 B（磁盘2,465,792 B）；服务器main代码树132文件/3,298,596 B（磁盘3,563,520 B）。未清作者下载包、环境、runs或冻结checkout。|
|隔离工具退役|096、122、131、138及153/154/161/164/167诊断链，共12脚本＋12专属测试，308,358 B。组外代码/配置引用为零；所有原合同、报告、失败记录和数值证据保留。|
|入口纠正|既有方法说明按measurement→航迹段→Q64/运动→评分→WS构组→恢复→JSON→评价导航，旧anchor和072网络明确为对应组件；补195/196及两种计时边界，历史工具/99条交接记录折叠，原记录行逐字保留。|

主仓约1.1 GiB主要由Git历史及环境构成；服务器main约122 GiB主要为原runs。两者没有通过改写Git历史或删除实验产物压缩，不能将本机资料库释放量混报为服务器收益。原论文、矩阵及未筛方案不删除。

## 退役依据与按需恢复

- 096：原结案比较；122：共享运动/时间残差科学完成；131：完整验证及独审完成；138：观测冲突科学及独审完成。完整判断分别仍在对应报告和S2L。
- 153/154/161/164/167：冻结TFM诊断链整体退役。153首次审计失败和修复后的数值复核继续按原报告区分；不把失败改写成首次成功。模型、训练、checkpoint加载器均保留。
- 跨实验共享的`compare_association_duration`、`compare_position_harm`、条件价值基础实现和仍有引用的配置保留。126/132相关失败链不纳入。

以下文件在删除前版本[a63f12a](https://github.com/HKillua/association-code/tree/a63f12a5f56e34ff3b0e68a4182a78312ef50f5e)完整可读；单文件恢复不恢复旧任务授权。122报告源码链接已改指该历史版本。

```text
tools/analysis/compare_completion_systems.py
tests/evaluation/test_compare_completion_systems.py
tools/analysis/shared_motion_time_residual.py
tests/integration/test_shared_motion_time_residual.py
tools/analysis/observation_conflict_probe.py
tools/analysis/audit_observation_conflict.py
tests/integration/test_observation_conflict_probe.py
tests/integration/test_audit_observation_conflict.py
tools/analysis/conditional_source_value_full.py
tools/analysis/audit_conditional_full_arrays.py
tests/integration/test_conditional_source_value_full.py
tests/integration/test_audit_conditional_full_arrays.py
tools/analysis/tmtaff_training_physical_probe.py
tools/analysis/audit_tmtaff_training_probe.py
tools/analysis/tmtaff_constant_path.py
tools/analysis/tmtaff_decoder_layers.py
tools/analysis/tmtaff_key_value_probe.py
tools/analysis/tmtaff_gradient_flow_probe.py
tests/integration/test_tmtaff_training_physical_probe.py
tests/integration/test_tmtaff_probe_numeric_audit.py
tests/integration/test_tmtaff_constant_path.py
tests/integration/test_tmtaff_decoder_layers.py
tests/integration/test_tmtaff_key_value_probe.py
tests/integration/test_tmtaff_gradient_probe.py
```

按需查看单文件，例如：

```sh
git show a63f12a:tools/analysis/shared_motion_time_residual.py
```

若确需恢复该源码：

```sh
git restore --source=a63f12a -- tools/analysis/shared_motion_time_residual.py
```

## 验证与同步

- 剩余450个Python文件AST解析通过；剩余代码中无退役模块名引用，独立复核另覆盖配置/脚本。模型、基线、配置、项目skills、提案、结果原件无Git内容改动；保留源码/测试没有数值修改。
- unittest发现844项测试，全部测试模块导入成功；这是导入检查，不是844项全部执行。实际运行条件价值保存数组、TMTAFF下游和history候选的10项相关检查，12.772s全部通过；不读取真实数据或启动科学训练。
- 414份Markdown中的2,555处本地路径有效，分析工具索引覆盖全部保留模块，折叠标签配对，`git diff --check`通过。原文带`:行号`链接按文件路径核验；未把行号误当文件名。
- 独立审查确认24份退役文件均可完整恢复、99条原交接行一致，未发现代码/导航缺陷。存储一致性依据实际逐文件原始记录，不把事后审查冒称独立重建了压缩前状态。
- 本机提交后同步origin及服务器干净main；冻结checkout和runs原位保留。具体提交见本文件所在Git提交。
