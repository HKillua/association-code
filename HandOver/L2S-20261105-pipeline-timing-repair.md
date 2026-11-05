# 完整链路工程计时：分组接口修正后的剩余项目

2026-11-05 READY，继承[原固定规格](L2S-20261105-pipeline-timing.md)的原始量测输入、三场/三个截止/四组合与1Hz解释范围；原attempt1技术失败保全，不恢复/覆盖或重投旧任务。

原science unit mtad-cf08277092dc.service真实exit1/failed，27.897811s，已完成692全窗口PTFM/DeepAF各2条，共4条计时，后在HARD/O178的ordinary_fusion边界失败。根因：分组生成NumPy数组，旧ordinary_fusion要求普通列表，其比较表达式触发ambiguous truth value。该失败没有方法效果含义；原4条已通过其全窗口motion绑定，继续保留来源，不重复计时。

修正仅将group成员列表按原接口传递，single-O内部显式转索引数组。新增真实模块集成探针覆盖：原始量测→056关联→合并为多节点组→159恢复→178/O→序列化，HARD/STRUCT均通过；原4项检查、编译/CLI接口通过，不训练、不认证精度。

新任务独立checkout `association-code-timing-e2e-repair-20261105`；新run `DEPLOYMENT-PIPELINE-TIMING-REPAIR-20261105-attempt-1`、同前缀launch-attempt-1。显式`--completed-run /home/wangguangjie/djs/association-code/runs/DEPLOYMENT-PIPELINE-TIMING-20261105-attempt-1`仅读取原4条已核公共记录，严格绑定scene/case/method/repeat及软件绑定标记，不读旧private预测作推理输入。只执行余68条计时/34次预热；合并结果仍72条，原4条注明source_run。完整窗口原输入/分区绑定、无未来prefix、全部预处理/关联/159+178恢复和输出语义不变。

CPU16–19×4、无GPU/RSS4GiB/1800s advisory，systemd/Linger=yes/Restart=no，固定剩余项目后停止；技术失败保全且不自动再试。0新增模型训练/normalizer/reference fit/optimizer，原生OLS重算计入。原152/178/前次单组计时和本次失败目录不动。公共原件/真实退出与完整/局部/模拟socket缺口按原规格回传。

命令沿原launcher，将cwd/record-dir/output改为上面repair新目录，并在payload入口增加上述`--completed-run`；实际argv/env由request.json完整保存。这里发布的是具体接口修正及剩余项目的新合同，不追加场景/重复/优化/训练或旧失败额度。
