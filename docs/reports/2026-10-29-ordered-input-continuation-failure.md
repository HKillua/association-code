# 132：首臂1000更新后被oomd终止，完整输入比较尚未完成

2026-10-29。EXPERIMENT_RECORD `EXP-132-FAILURE-20261029`→SQUEEZE_REPORT `SQ-ORDERED-INPUT-CONTINUATION-FAILURE-132-20261029`。消费[132执行前Idea1.0](../proposals/ordered_shape_continuation_20261029.md)、原126科学合同及[126真实FD失败](2026-10-29-ordered-input-failure.md)。原失败不覆盖，本次失败也不改写原run/launcher；[脱敏原件](/Users/a1/Projects/biye-ronghe/docs/evidence/20261029_small_validation/132/)完整保留。没有自动新run或重训。

## Record：真实服务胜于陈旧RUNNING

`LEVEL1-ORDERED-SHAPE-CONTINUATION-20261029-132-attempt-1`，冻结c166ccb、独立association-code-run132。服务mtad-792d19ae7466.service于北京时间23:42:11启动、23:44:50退出；journal明确 **systemd-oomd killed 9 process(es) in this unit**，ExecMainCode2/ExecMainStatus9、Result=signal、ActiveState=failed/SubState=failed、Restart=no。worker/payload原PID均不存在。非7200s建议墙时终止，未触碰oomd设置或他人进程。

原run及launcher仍`RUNNING`，缺exit_code.txt和完整summary；stderr为空。单独terminal-synthesis.json记录effective_status=FAILED/cause=systemd_oomd，保留原件陈旧状态，**payload完整退出码未知**，不补造0或137。run最后checkpoint的112.997130s/RSS1455505408B仅最后记录，不是完整墙时或真实峰值。服务生命期约159s、CPU95.205853s；完整程序/launcher墙时和真实峰值未知。126已消耗的2949.129498s准备成本另计，不能只报本次成功训练部分。

CPU12–15/4线程、共享RTX3090原UUID逻辑cuda:0、RSS8GiB程序检查、CUDA无AS限。原T2100/V300五类缓存2400场均完成头/文件长度/metadata核验；V原节点/来源/身份/query/support小字段绑定，T采样pair/label合法性检查完成。只读软链原缓存、四normalizer值与126原件exact，没有CSV/ZIP读取、normalizer重算或reference/T_test读取。大46列feature payload未重新扫描或重生成，不能冒称独立认证全部输入数值。

FD修复已实际越过旧失败点：全2100场采样输入普通ndarray加载，1000×256×2 int64共享plan及shared-initial.pt已保存，没有再次报Errno24。这证明本次载入路径可执行，不是输入形状科学效果。

| 原科学训练/评价项 | 本次实际事实 |
|---|---|
| SUMMARY优化 | history恰1000行/update1–1000，loss与gradient全部有限；final.pt已保存 |
| 保存权重核验 | 只用torch.load(weights_only=True,CPU)反序列化，6209参数全部有限，checkpoint.update=1000，六个Adam state.step均1000 |
| Idea谱系 | checkpoint保留原IDEA-ORDERED-SHAPE-SIGNAL-20261029；新132run为技术继续Idea，原先已披露，不冒充原126成功 |
| ORDERED/DISTRIBUTION/SHUFFLED | 各0更新，无训练目录/权重 |
| validation | 最后phase=final_validation/SUMMARY，尚无完整任一arm-result、保存scores或summary；未独立核出raw/论文资格成绩 |
| completed_arms=[] | 代码在训练及完整评价/保存后才登记完成；不与“SUMMARY已1000更新”矛盾 |
| 实际四臂比较 | 未完成，不能评价有序输入优于/劣于分布或shuffle，不能称四臂训练完成 |

限制性posthoc-checkpoint-inventory.json只核上述metadata、history、plan形状及继承norm一致性；没有构造模型/前向，没有下载权重或调用完整四臂科学审计。私有cache/初值/plan/SUMMARY权重留服务器，未来无需重训这1000步。

## Squeeze：已知原因与未知触发条件分开

DE：实际终止机制是oomd，原活动记录陈旧；旧FD路径已修复、真实GPU优化已经发生。它既不是有序信息无效，也不是学习方法有效的结果。上一失败在采样前零更新，本次在首臂1000步后，阶段和证据不能混为一谈。

UN：最后phase包含末步评分、全V阈值选择、分层评价和保存，死亡前没有逐子阶段trace，**不能断定具体死在阈值排序或某个算子，也不能把最后1.455GB RSS当杀死时峰值**。代码中这些步骤会持有全V分数/多份排序与评价数组，是可检查的内存候选；系统/祖先cgroup压力和OOMD选取条件没有被识别。

定向只读代码分析：V实际4439092个pair，单个float64/int64同长向量约33.868MiB；精确阈值搜索同时保留约20–30条此量级的排序/累计/差分数组，静态额外量级约0.66–0.99GiB，尚不含sort/unique/repeat工作区。这是可测AI候选，不是当时峰值测量；后续可按最后使用释放、原位累计或精确外存事件合并，并以原全部候选、>=语义、并列规则及指标exact为约束。四臂records累计不是此次杀死的原因，因为仅首臂开始；未来仍应共享pairs/labels只分存scores。不得为省内存改float32近似、阈值网格、删pair或缩V。

事后资源快照：MemAvailable97637128kB、SwapTotal2097148kB/SwapFree0，app.slice MemoryCurrent7641686016B、MemoryHigh/Max=infinity。快照晚于终止，不代替当时状态，也不支持“物理RAM耗尽”的唯一解释。当前SSH账号查询systemd-oomd系统日志返回不可见，只有本服务journal可读；管理员日志是实际证据缺口。未执行sudo、调整swap、禁用oomd或改变其它任务。

下一讨论应先核系统级触发条件，并准备保持相同科学人口/阈值语义的分阶段末步评价、明确峰值而非仅总时间；保留全体pair，不缩验证集/删难例/改比较指标来躲问题。若另订继续合同，应先用现有SUMMARY/final.pt完成其固定评价，不重训该1000步，再决定其余三个1000步臂；共享初值/plan/norm均已保存。任何内存实现调整需同小输入与原评分/阈值/统计exact核对，不能直接在失败checkout上改代码继续。

依据项目AGENTS“活动checkout/config/runs冻结”“失败保全不自动重试”，本次先保全现场、记录回流并提出具体下一讨论项，没有发第三次自动尝试。等待管理员证据期间，本轮其它四线已完成的独立结果与反例照常成立，不因输入执行失败抹去或扩大其结论。
