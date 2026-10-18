# 原007 manifest的真实原生CLI装配检查

2026-10-18，012回传原件后首次运行既有 `--inspect-only`，**退出码0、状态INSPECTED**。这关闭此前“本地只有冻结名单、尚未用完整原manifest跑真实CLI装配”的缺口。没有构造模型、前向、optimizer、校准或GPU训练，不重复先前时钟推断，也不重建划分。

实际命令（工作目录 `/Users/a1/Projects/association-code-v2`）：

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
 /Users/a1/Projects/association-code/.venv/bin/python tools/experiments/tmtaff_complete_fit.py \
 --archive /Users/a1/Downloads/数据集level2.zip \
 --manifest HandOver/results/EVIDENCE-20261018-012/prepared007/manifest.json \
 --alignment configs/tmtaff-time-reference-alignment.json \
 --config configs/tmtaff-time-cost-probe.json \
 --output runs/LOCAL-NATIVE-INSPECT-20261018-012-attempt-1 \
 --inspect-only --cpu-threads 4
```

输出为 `INSPECTED native-time inputs: /Users/a1/Projects/association-code-v2/runs/LOCAL-NATIVE-INSPECT-20261018-012-attempt-1`。原 `preparation.json` 122,492,141字节保存在该本地runs目录（Git忽略），[小型摘要](native_inspect_summary.json)与[提取脚本](summarize_native_inspect.py)入库。报告记录准备耗时47.357秒，计时在结果写盘前，**不是包含IO的整个CLI耗时**，也不是服务器性能估计。

| 人口 | T_core | V_select |
| --- | ---: | ---: |
| 场景 | 512 | 64 |
| 有效片段 | 54,792 | 1,635 |
| cap后真实点 | 2,310,515 | 46,051 |
| 全候选对 / 正对 | 18,249,384 / 110,733 | 29,008 / 3,297 |
| 标签目标 | 12,699 | 377 |
| 无有效输入目标（仍保留） | 49 | 0 |
| 无reference行目标 | 0 | 0 |

装配沿已批准cap128、time_unit20、64查询、逐场景offset/gap600。实际访问记录为1152个训练/开发标签与量测成员、576个reference成员，场景集合与原manifest T_core/V完全一致，均无T_test场景；report的test_opened和legacy_scene64均false。参考有效query数按属性保存，非所有目标全部时刻都有监督，“无reference行目标0”不等于监督覆盖100%。

此检查不拟合normalizer，不加载013 checkpoint，不产生模型指标；它只证明当前真实输入与参考装配入口能够按原manifest完成。
