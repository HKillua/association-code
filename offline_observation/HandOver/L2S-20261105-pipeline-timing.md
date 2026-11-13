# 传感器输入到输出及1Hz回放：有界工程计时

2026-11-05 READY。用户明确要求完整应用链路，并指定每秒更新一次；上一轮单组后端数字不作端到端结论。本合同只做工程计时，不读取真实关联标签/reference/test，不训练/调参或宣精度。

## 行为与固定范围

- t0为应用已接收原始传感器量测CSV字节缓冲，t1为所有组的经纬度/速度/航向、成员与query时间已序列化为可发送JSON bytes。完整包含CSV解码/属性检查/排序去重/节点组装、Q64网格与原生OLS/支持证据、全部候选对特征与056网络、必要CTX/head、WS构组、所有恢复与最终序列化；不导入预计算motion/score/group作为预测输入。
- 四条实际组合：056+WS+PTFM152A；056+WS+DeepAF152B；056+HARD175+159完整多源恢复+178/O；056+STRUCT173+159+178/O。PTFM/DeepAF在本项目是融合器，配套共同关联前端计入总时长；不冒充其论文原生完整系统或TMTAFF端到端速度。
- 固定原V32场692/3981/1187（原节点11/49/144，对应小/中/较大规模，不代表最大999节点场）。每场三种输入：全窗口所有量测已收到；原始时间范围中点的received prefix；中点+1秒的received prefix。逐点检查time≤arrival cutoff，禁止未来观测进入计算。prefix是最朴素历史buffer全量重算，不是已实现增量服务；输出仍64个历史query，最新query不晚于cutoff。
- 每条件/方法1次预热绑定＋2次计时，共72条计时记录、36次预热。三场/三个到数截止/四组合全部完成后停止，不扩大规模/重复/优化配方。每次方法从原始字节重新独立计时，不用共享预处理摊薄总成本。
- 原始归档只读measurement member；legacy loader必需的片段元数据由source/batch和已收到的时间范围生成匿名互异ID，不读GT身份或真实标签。完整窗口必须绑定原127节点/原生状态/支持/运动证据，HARD/STRUCT分区绑定原178。178单O读出、常驻risk对原API逐值绑定。若绑定失败，attempt保全且不自动重试；不能删场放宽容差继续。

## 资源与执行

新冻结checkout `association-code-timing-e2e-20261105`，科学入口`tools/analysis/deployment_pipeline_timing.py`；新run `/home/wangguangjie/djs/association-code/runs/DEPLOYMENT-PIPELINE-TIMING-20261105-attempt-1`、同前缀launch-attempt-1。CPU16–19/4线程，无GPU，RSS4GiB，1800s advisory，systemd/Linger=yes/Restart=no；一次attempt1，失败保全。原模型、原实验、旧计时和152产物只读。0新增模型/reference/normalizer fit或optimizer；输入原生OLS须重新计算且计入成本。

现场命令以launch request为完整原件，入口：

```bash
/home/wangguangjie/djs/mtad-field-20261016/bin/python tools/experiments/launch_detached.py --cwd /home/wangguangjie/djs/association-code-timing-e2e-20261105 --record-dir /home/wangguangjie/djs/association-code/runs/DEPLOYMENT-PIPELINE-TIMING-20261105-launch-attempt-1 --max-seconds 1800 --time-policy advisory --env CUDA_VISIBLE_DEVICES= --env OMP_NUM_THREADS=4 --env OPENBLAS_NUM_THREADS=4 --env MKL_NUM_THREADS=4 -- /usr/bin/taskset --cpu-list 16-19 /home/wangguangjie/djs/mtad-field-20261016/bin/python tools/analysis/deployment_pipeline_timing.py --root /home/wangguangjie/djs/association-code/runs --output /home/wangguangjie/djs/association-code/runs/DEPLOYMENT-PIPELINE-TIMING-20261105-attempt-1 --time-policy advisory
```

## 验收与解释

4项本地针对性检查已过：原始byte→motion、未来行拒绝、single-O动作/平局与原函数exact、匿名source/batch分组；CLI help/编译通过。合成仅验软件。真实输入/分区绑定与完整计时由本次执行取得。

公共run/全部72条分阶段计时/真实launcher/终态回传，坐标输出与成员索引等private原位保留。报告每场input字节/行/节点/对/组/输出数、完整延迟与阶段，按1Hz说明服务时间/工作量能否跟上。2个计时样本不冒充P95或统计CI，不从64点总耗时除64冒充增量步耗时。

物理通信/传感器传输、真实socket队列、乱序/迟到/丢包处理及实际设备排队未模拟，必须说明。应用收到一个历史buffer到结果可发送的计时不等传感器物理采集到显示的全部延迟；观测积累/窗口等待另列。1Hz是输出频率，不自动等1秒硬截止；同时报告数据新鲜度和吞吐，不能把2s计算直接判为非在线。prefix计时可证明当次无未来输入，不证明在线精度与全窗口原结果相同。
