# 当前交接

更新：2026-11-10。这里只维护执行、待核验和保全事项；研究目标/结论看[研究状态](../docs/research_status.md)。历次完整合同和回执从[交接记录索引](README.md)查找，整理前正文已[完整保留](../docs/history/2026-11-02-research-process.md#原交接正文)。

## 当前执行

离线精度与在线实时持续并行推进，具体工作只看[研究状态](../docs/research_status.md#下一步)。仓库编排分为[在线回放](../tools/online/README.md)和[离线评价](../tools/offline/README.md)，原合同使用的命令和输出格式继续兼容。当前可领范围以本交接为准，本次入口迁移未发布新实验READY。

**195/196科学与规定独审均已完成；152A/B科学已完成，完整结果与规定收尾待核。** [原合同](L2S-20261030-152.md)、[不可变启动回执](S2L-20261030-152-start.md)。全T2100/46126组，fresh seed20各一次，冻结bda61e5，比较对象仍原147。GPU2/3按合同UUID；不得加157/178、改预算或活动checkout。

|任务|固定额度|北京时间2026-11-05 14:12:15有界终态核验|原服务|
|---|---|---|---|
|152A PTFM|150epoch/427800更新|COMPLETED/exit0，150epoch/427800更新、300/300V|`mtad-e358bc6e1a62.service`，inactive/dead、ExecMainStatus0、Restart=no|
|152B DeepAF|20epoch/80800更新|COMPLETED/exit0，20epoch/80800更新、300/300V|`mtad-ea2cca7761dd.service`，inactive/dead、ExecMainStatus0、Restart=no|

152A北京时间11月4日03:59:15、152B 11月3日22:34:06真实结束，原launcher returncode0，科学运行内部368244.844486/348728.843220s；均已完成300场V。11月10日通过SSH MCP完成现有公开run/summary的有界聚合初核，见[路线审查与来源字段](../docs/reports/2026-11-10-offline-online-route-review.md)；未读取私有数组、重做推理或启动独审。完整输出/规定审查仍未完成，不预宣科学胜负。按原合同取必要公共原件，核完整输入/命令/退出、人口、指标、尾部和成本，完成规定收尾与Record→Squeeze→Mode C。保持同一run与launcher，不重投或续旧148预算。

### 196已收尾：固定动作评分的原V32验证

[196独立合同](L2S-20261103-196.md)、[原启动](S2L-20261103-196-start.md)、[终态回执](S2L-20261103-196.md)。科学mtad-aea314e70600.service与规定独审mtad-5e4919de16da.service均COMPLETED/真实exit0/inactive/dead/Restart=no，11月3日北京13:19:01/13:25:57结束。原V32/9组件/40臂/T128 full头、0新fit/teacher/T复评/test；CPU科学8–11、审计16–19各4线程无GPU，一次attempt1无重试。68公共原件、全32场/完整指标/保持域/三套害救回与动作/组件独核已齐，Record→Squeeze→Mode C2.10完成。本196无待领/待审，不扩V300/训练/默认模型。

### 完整链路与1Hz工程计时已完成

[原固定规格](L2S-20261105-pipeline-timing.md)、[接口修正与剩余项目合同](L2S-20261105-pipeline-timing-repair.md)、[完整报告与公共原件](../docs/reports/2026-11-05-pipeline-online-timing.md)。原attempt1分组数组/列表接口失败，unit mtad-cf08277092dc.service真实exit1/failed、4条已核记录保留；新独立repair checkout94710b7只执行余68条计时/34预热，成功导入原4条，合72条无重复。repair unit mtad-6f2766c68408.service于11月5日北京15:06:52真实exit0/inactive/dead/Restart=no；Linger=yes、CPU16–19×4、无GPU、SSH cgroup分离已核，RSS563150848B，launcher822.312491s。原/修正目录均冻结。

三场×全窗口/连续两个1秒prefix×四完整应用组合，原始measurement bytes→清理/OLS→完整关联/分组→融合恢复→JSON输出；没有GT/reference/test/新训练。未来行0、12套全窗口分区exact，原生输入/159+178完整保持已核，CPU与原保存端点的数值差逐项披露。完整72条分阶段计时及失败/终态原件已回传；资料库包清单34文件，private坐标与原权重原位保留，无待运行/守护或自动追加测量。

### 11月5日部署工程计时已完成

[事前有界范围](L2S-20261105-timing.md)、[结果与14公共原件](../docs/reports/2026-11-05-deployment-timing.md)。一次常驻模型计时，CPU16–19×4/GPU0兼容共卡、原权重只读、0fit/update/reference/test；service mtad-58b32b103f4a.service真实exit0/inactive/dead/Restart=no。固定两组Q64/2warmup＋5计时的CPU/GPU后端延迟及排除范围已记录，不提供新精度结论、完整关联前端或流式延迟认证，无后续自动测量。

### 195已收尾

[195原合同](L2S-20261102-195.md)、[不可变启动回执](S2L-20261102-195-start.md)、[终态回执](S2L-20261102-195.md)：原T32/40臂、0新训练/0V，科学10:26:57/独审13:11:31均成功exit0，服务inactive/dead、Restart=no。原28控制/完整指标/覆盖/尾部/四属性/条件选择均核，61公共原件及后验只读聚合完整回传，Record→Squeeze→Mode C2.09已闭环。无本任务待领/待审；后续V验证仅按上面的独立196合同，不重投195、不改原run/活动152。

## 本轮收尾与保全

- 194科学与规定独审均已成功，66公共原件及完整记录已核齐，无待审/待领；193技术失败独立保全。[194终态回执](S2L-20261101-194.md)、[193失败回执](S2L-20261101-193.md)。原science/audit和其他已结案目录冻结，该194合同不提供追加运行许可。
- 132：用户暂拿不到系统OOMD日志、要求保留现场；1000更新权重、计划、normalizer、缓存及失败产物不改，不重训/重启。[回执](S2L-20261029-132.md)。原107/113及143/144/146也不重跑。
- 其他100–194历史任务与失败/修复记录从[记录索引](README.md)、[完整报告](../docs/reports/README.md)按需读取，历史READY不提供当前执行许可。135仍为保留草案。

## 执行与回传边界

按[服务器规则](../docs/server_autonomy.md)一次任务、独立冻结目录、最多四单卡/FIFO；失败保全，不自动重试或替换配方，不触碰他人进程。墙时advisory、不加训练额度，不建SSH值守/守护或跨聊天消息。

数据、权重、原身份/坐标、大预测及服务器runs原位保留；公共小型原件和完整回执按原合同回主仓/资料库。195/196及规定独审已完成，部署有界计时已完成；152科学完成待原合同完整收尾，无自动扩大全300或训练额度。Git主仓可同步，活动checkout不得pull。
