# 固定056比较的090C旧缓存格式修复依据

2026-10-29。只修评价前绑定，不训练、不运行模型、不改107冻结目录/预算。原107失败保留；根线程统一发布后续独立零训练评价，不重启原链。

## 失败与根因

根线程[现场读取107真实状态及原始证据](2026-10-29-107-failure-discovery.md)：TCM2424与TFM1000均COMPLETED，固定056子进程在`bind_scenes`预检读取090C `final/predictions/states-1.npz`时失败：`KeyError: sources is not a file in the archive`，尚无该比较的TFM前向。现场headers又确认090C没有`learned_assoc_membership`，不是只缺一个字段。

本地生产writer `association/evaluation/sequence_position.py`的payload明确仅复制`node_ids/identities/target_identity_ids/native_states/support/nearest_seconds/exact_observation/query_seconds/learned_assoc_group_offsets/learned_assoc_group_node_indices`并保存`group_anchor_node_indices`，没有sources或membership。056缓存含这两字段。因而是新wrapper误假定两代缓存同格式，**不是训练权重缺失，也不是来源/分组实测不一致**。

## 最小行为修复

改动仅在`tools/experiments/tmtaff_fixed056_evaluate.py`的前置绑定：

1. 完成的090C根run必须明确指向本次同一056 `source_system`；主入口原有090C配置/最终端点/全V校验继续执行。
2. 056 `node_ids/sources`必须与已通过`_records_for`核对的原`final-scores.npz`记录一致；090C完整node顺序必须与056相同。旧sources缺失时，经同源metadata＋相同node顺序传递绑定，**不从node字符串猜来源，不修改旧NPZ或补造字段**；如090C实际含sources，仍逐值比较。
3. 两边实际offset/flat数组继续逐值比较，并各自验证整数类型、首尾、严格递增、完整唯一覆盖。由其实际顺序导出membership，与056真实cached membership比较；旧缓存若显式有membership也必须一致。旧已有anchors必须等于每组首节点。
4. node、identity、target registry、query、native states、support、nearest等原核验全部保留，并新增exact_observation核验。prepared新单位输入到056原始量测的生产`verify_native_source`、原分组重放、reference/valid、全人口及关联各scope/场计数核验没有放松。
5. 公有`source_verification`报告有多少场采用传递source绑定、多少场从实际完整partition导出membership，避免把派生信息冒称NPZ已保存。

原始cache不变；不存在跳过来源/分组检查的fallback。没有改变分数、阈值、分区、输入/单位、模型或评价公式。

## 验证及限制

`tests/evaluation/test_fixed056_controls.py`新增真实legacy字段集成功用例；来源破坏、node/identity/target/query/native/support/nearest/exact破坏、offset/flat重复或逆序、membership/anchor错误都拒绝。含实际legacy NPZ通过`bind_scenes`入口的回归，并拒绝错误source_system或未完成090C metadata。该模块新增4项，目前13测试；与107原链6测试合计**19测试通过**：

```bash
.venv/bin/python -m unittest tests.evaluation.test_fixed056_controls tests.integration.test_tmtaff_unit_baseline_A
```

这是CPU合成/小型文件的软件回归，不是服务器真实300场验收或新模型成绩。没有读取服务器权重、运行GPU、重训或修补数据；真实旧NPZ路径/headers由根线程现场读取，后续仍须独立执行完整绑定与固定107 checkpoint评价，失败保全且不自动重试。原失败不能被此软件检查覆盖成成功。
