# 首轮工具

本地 AI 编写和测试；服务器 AI 只按当前 [Handover](../../HandOver/Handover.md) 执行，并负责运行监控与回传。以下是工具说明，不是自动执行全部工具的指令。

## 环境检查

在准备使用的已有 Python 环境中：

```bash
python tools/pilot/server_check.py --workspace . --output /本轮新输出目录/environment.json
```

报告记录 Python、PyTorch/CUDA、可见 GPU、磁盘空间。不安装依赖、不下载数据、不训练；输出文件已存在时拒绝覆盖。环境未配置也能出报告。

**脚本整体退出 0 不代表环境就绪。** 服务器回执要同时报告 JSON 内 `nvidia_smi.exit_code`、`pytorch.exit_code`、`pytorch.details.cuda_available`、可见设备数和错误信息。GPU 占用只是快照，不能代替用户分配的可用设备/时段；四张卡显存不能当作一张卡合计。

## Level2 审计

```bash
python tools/pilot/audit_mtad_level2.py --archive /实际数据路径/数据集level2.zip --output /本轮全新审计目录
```

使用标准库直接读取 ZIP，不解压。核对整包哈希/目录，默认抽样场景 0、1、2、100、4999；结果包括报告和原始目录索引。官方 Level2 包 MD5：`8ddcea161a49b47ecdac72ae45804438`。

退出 1 表示发现内容/验收问题，报告仍被保留；不能擅自删行、修改数据或通过重复运行掩盖。原本地已完成整包和抽样审计，首轮服务器交接**不要求重跑该审计**。关联基线可以使用有效位置，即使 vel/cou 非有限；这与审计器严格检查所有字段是两个不同规格，修改前由本地固定协议。

## 本地测试

```bash
python -m unittest discover -s tools/pilot -p 'test_audit_mtad_level2.py'
```

15 项测试使用合成小 ZIP；不会读取真实数据，也不是模型性能实验。
