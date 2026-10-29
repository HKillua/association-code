# 126：完整输入缓存后、采样计划构造前的文件句柄失败

2026-10-29。EXPERIMENT_RECORD `EXP-126-FAILURE-20261029`→SQUEEZE_REPORT `SQ-ORDERED-INPUT-FAILURE-126-20261029`。上游[输入Idea1.1](../proposals/ordered_shape_signal_20261029.md)、[原合同](../../HandOver/L2S-20261029-126.md)。这是可复核的资源/代码失败，不是输入有序信息的科学负结果。[原件](/Users/a1/Projects/biye-ronghe/docs/evidence/20261029_small_validation/126/)含真实run、同launcher/退出/日志、服务终态和事后文件清单；原缓存/坐标/身份留服务器，未重启原126。

## Record

实验`LEVEL1-ORDERED-SHAPE-SIGNAL-20261029-126-attempt-1`，冻结603d244，独立association-code-run126。原run FAILED/exit1，同launcher returncode1；服务mtad-6deeed0fedd1.service failed/Result=exit-code/ExecMainStatus1、Restart=no，非超时/oomd。UTC14:39:33.342727→15:28:42.472225，程序2949.129498s，launcher2953.056775s；RSS1171898368B。实际0 optimizer update、四臂均0、0完成臂，无history/权重/共享初值/批次计划/完整评价summary。

原计划是T2100/V300，SUMMARY/ORDERED/DISTRIBUTION/SHUFFLED四臂、同6209参数/共享初值/真实1000×256批次/臂，完整raw与正式论文资格评价。四个T-only normalizer已经写成normalizers.json；归一化本身已完成。run.phase仍`normalizer`是进入build_plan前未更新phase的陈旧阶段名，不能据此归因归一化泄漏。

stderr真实调用链：run:404→build_plan:220→sampler_input:215–216→np.load(...mmap_mode='r')→mmap.mmap，`OSError [Errno24] Too many open files`。sampler_input用列表推导先同时打开T2100的pairs及labels，共需4200个memmap；整个列表构建成功后才进入采样器构造及finally关闭。当前unit的文件句柄软限见事后原件；异常发生在列表构造阶段，原finally尚未进入。fit_normalizers已有逐场try/finally关闭，并非这条错误链。

事后仅文件名/数量/尺寸核对：T_core与V_select分别有2100/300份features/base/pairs/labels/metadata，五类全部齐。T pairs442390144B、labels27901384B、features5084664256B；V pairs71063872B、labels4477492B、features816831328B。准备计数末值299是枚举下标，不能据它说缺最后一场；真实最后4968存在。**这还不是2400份缓存内容/输入一致性的独立认证。** 无data-audit/sampling-plan-audit，因为异常早于两者写出；不得补造为原126已完成的记录。

计划偏差：完整缓存及normalizer准备完成，但未进入四臂实际优化和科学评价。约49分钟准备成本已经消耗，不能隐去或将新任务只报训练成本。CUDA设备曾按原合同绑定，未执行成功优化，不能写完整GPU训练过或有新F1。T_test/reference原件未读的原run标记保留；冻结worktree实际clean。

## Squeeze

DE支持一个具体可修复问题：全量mmap列表使文件句柄需求与场景数线性增长；小测试未暴露真实2100场规模。0优化更新故不能评价ordered、无序分布和shuffle的科学差别。缓存齐全给出了避免重复原始CSV准备的条件，但需后继核头/形状/scene绑定，不能只信有文件。

最低变更是新独立132：逐场普通ndarray读取pairs/labels，使np.load的文件描述符及时关闭，沿原BalancedScenePairSampler、seed20和1000×256规则产生同一科学采样计划；T两类原数组约470MB，成本相对8GiB预算可明确计入。已有normalizer和2400场缓存只读复用，不重开ZIP、不改特征/损失/网络/初值/预算/终点。以同一小输入比较修前/修后实际计划及曝光，用低FD软限的有界软件反例验证不再依赖4200个同时打开的map；不靠修改宿主限制或吞掉场景规避。

该后继须先完成本Record→Squeeze→Mode C技术回流，再有自己的Idea/READY/新run与一次终点，原126保持失败。若内容绑定不成立，保留失败而不自行再生成缓存、替换配方或重试。此处只给技术恢复连接，未发布执行，未产生任何新的领域机制或创新效果结论。
