# 013 selected checkpoint 的独立评价入口

实现入口：[tmtaff_complete_evaluate.py](../../../tools/experiments/tmtaff_complete_evaluate.py)。这是本地实现与合成 CPU 验证，**没有执行服务器或真实 GPU 评价，也没有授予新预算或创建 READY**。

013 的训练更新已完成，但终评被中断。新入口只读 `checkpoint.pt` 中的 `selected` 模型、融合模块、normalizer 和原阈值；不会恢复 FAILED 训练、创建 optimizer、重新拟合 normalizer、重做 margin calibration 或重新选择阈值。所选模型的关联来源与融合来源保留在 `source_selection`，不会混称为最后 JOINT 状态。合法的 `threshold=None` 仍表示全拒绝。

## 使用命令

以下是待另行批准预算后使用的命令模板，不是下发请求。`APPROVED_EVALUATION_SECONDS` 必须由新一轮授权给定；输出目录须不存在，GPU 的选择沿现有动态调度规则，`cuda:0` 是包装器暴露后的逻辑设备。

```bash
python tools/experiments/tmtaff_complete_evaluate.py \
  --checkpoint runs/TMTAFF-TIME-COST-20261018-013-attempt-1/checkpoint.pt \
  --archive '/home/wangguangjie/djs/data/mtad/数据集level2.zip' \
  --manifest runs/PREP-NATIVE-20261017-007-attempt-1/manifest.json \
  --alignment configs/tmtaff-time-reference-alignment.json \
  --output runs/TMTAFF-TIME-EVALUATION-NEW-attempt-1 \
  --device cuda:0 --cpu-threads 4 \
  --max-seconds "${APPROVED_EVALUATION_SECONDS:?set the separately approved evaluation budget}"
```

不接收新模型/数据配置、stage、resume、latest 或阈值覆盖参数。cap、time unit、共同查询长度、验证 microbatch 和 singleton 策略来自原 checkpoint。013 对应 cap 128、time unit 20 秒、64 个查询；这些必须由真实 checkpoint 内容确认，不能靠命令文案替代。

## 行为与验收

| 要求 | 可检查产物/行为 |
|---|---|
| 所有配置训练更新已经完成 | 校验保存 history/cursor；接受最后 JOINT 更新已落盘、stage 尚未递增的正常 checkpoint。未完成则拒绝，不补更新。 |
| 原划分和对齐策略 | 首先直接核对 manifest 中场景顺序、角色、component 和 per-scene offset/gap；装配后与保存的完整 `data_contract` 比较，包括 grid、node/identity、时间、支撑和目标人口。无 SHA/hash 门槛。 |
| 原完整 V 人口 | 使用原 `V_select` 全部场景、全候选、所有 label 目标；保留无有效节点场景。不读取 `T_test` CSV。 |
| 原 selected 与阈值 | `load_complete_snapshot(..., selected=True)`；`evaluate_complete(..., select_on_validation=False, threshold=saved_threshold)`。报告保留原选择信息，新的评价指标以本次输出为准。 |
| 中断可定位 | `evaluation-progress.json` 记录关联评分/物理融合的最后场景边界与累计融合 forward；失败写 `run.json`。 |
| 不覆盖与不伪报成功 | 新目录；成功才有 `run.json: COMPLETED`。错误为 FAILED/退出 1；预算耗尽为 INTERRUPTED/退出 124；SIGTERM 或键盘中断为 INTERRUPTED/退出 130。末尾落盘超时会将 summary 移到 `summary.incomplete.json`，不作为完成结果。 |

现有 `load_timed_dataset` 没有 V-only 接口，因此仍装配原训练池与 V；训练侧仅用于原 `data_contract` 核对，随后释放，模型在准备完成后才放到指定设备。准备会按既有 loader 读取训练侧 reference，但不使用它进行优化、校准、归一化拟合或选模。`preparation.json` 记录实际访问范围与耗时。未另造数据框架。

`--max-seconds` 从参数解析后开始，包含导入、CPU checkpoint 加载、装配、契约核对、模型恢复、完整评价和结果落盘；不继承 013 的旧训练预算。检查发生在装配/推理边界及落盘前后，无法保证在单个不可中断的底层运算中精确停止；正式执行仍需外层有界 TERM/清理包装器。SIGKILL 或机器掉电不能保证补写终态；仅有 RUNNING 或 progress 中 completed 都不能当成功，必须核对 run 终态和完整 summary。

保存契约是实际元数据直接比较，并不完整存储所有原始状态数值；它不是 ZIP 内容任意改动的检测器。应继续使用原路径数据与 alignment。新设备上的浮点差异可能影响原阈值下的评分，本入口会报告实际结果，不再拟合阈值来追平旧指标。

## 本地验证

```bash
/Users/a1/Projects/association-code/.venv/bin/python -m unittest tests.test_complete_evaluate_cli -v
```

8 项测试通过（2026-10-18，CPU，最终一次 0.549 秒）。真实走合成 ZIP → checkpoint 恢复 → 完整 V 关联及融合评价；测试 fixture 的 checkpoint 是合成模型状态与元数据，不冒充真实训练。覆盖原 selected、有限/全拒绝阈值、零 optimizer/训练/校准/重选、封存测试 CSV、无节点 V 人口、契约变更拒绝、准备预算、SIGTERM 进度留痕、末尾 I/O 超时及目录不覆盖。未执行真实 013 checkpoint 或真实数据评价。
