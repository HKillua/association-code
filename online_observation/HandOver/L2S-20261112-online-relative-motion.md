# 在线相对运动与连续报文诊断

2026-11-12，READY。只用原 28 场和 C-ON37 冻结风险窗，固定四边原图、发布与时钟。一次本地 CPU，零训练、选型及新场读取，先全部冻结无标签量，再读 A-B。结果按两源分差、同窗端点和相对位移、同刻单报和连续报文、正确持续及稀疏长错间隙统计。 [固定设计](../docs/proposals/online_relative_motion_20261112.md)

```sh
.venv/bin/python -m online_observation.relative_motion_gap \
  --archive PART1 PART2 PART3 --manifest configs/data/mtad-level1-scene3000.json \
  --graph-run PRIVATE_C28_OUTPUT --gap-run PRIVATE_C37_CAPTURE --output-dir NEW_PRIVATE_DIRECTORY
```

核对候选快照、窗口、原生时间、C-ON37 首个实报／屏蔽读出和无报负控。出现技术异常则保全。只有达到事前真边净收益、组件及稀疏可达性条件，才另立新留场验证。
