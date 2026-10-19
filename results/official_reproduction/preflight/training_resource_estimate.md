# 训练资源评估

## 官方训练负载

官方 YAML 配置：1,024 个模型输入 Gaussians；64 groups × 32 neighbors；4 个 target block，大小 `[11, 9, 7, 5]`；总 batch 256；`step_per_update=1`；`max_epoch=300`。优化器是 AdamW，lr `1e-3`、weight decay `0.05`（bias、token 和一维参数不施加 decay）；调度为 cosine，300 epoch、10 epoch warmup、最小 lr `1e-6`。EMA teacher momentum 从 `0.996` 线性增至 `0.9999`。默认 seed 为 0，分布式时按 rank 加 seed offset。

模型读取每个 PLY 后先采样 8,192 条 Gaussian 特征，runner 再随机选 1,024 条输入。官方 split 有 51,934 个 train 条目。默认非分布式 DataLoader 不 drop last，因此每 epoch 203 个 batch，其中最后一个 batch 为 222 条。代码循环边界实际跑 301 个 epoch 标签（0..300），约 61,103 次 optimizer update；配置名义上的 300 轮若按 203 steps 计算是 60,900 次。此为官方源码细节，不是本审计修改。

## 当前容量与实测

| 资源 | 当前状态 / 实测 |
|---|---|
| GPU | 2 × RTX A6000，48 GiB/卡 |
| 官方 launcher | `none`，runner 使用 DataParallel；batch 256 可拆为约 128/卡 |
| 合成单卡 batch 128 | smoke test peak allocated 10.19 GiB |
| 合成单卡 batch 256 | smoke test peak allocated 19.98 GiB |
| 模型参数 | 46,269,376 总参数；24,442,944 可训练参数 |
| 参数 + FP32 AdamW m/v 粗略静态量 | 约 0.35 GiB；不含激活、CUDA context、临时 tensor 和 DataParallel 开销 |
| RAM / swap | 134.7 GB / 8.59 GB；可用 RAM 66.9 GB，swap 已用 8.52 GB |
| 磁盘 | 501.1 GB 可用；完整压缩数据包约 305.7 GB，解压后大小未知 |

显存判断：现有合成 batch 256 在单张 48 GiB A6000 上通过，支持“显存大概率不是首要容量阻塞”的判断；真实双卡 DataParallel、真实 PLY batch 和 300/301 轮持续负载尚未基准测试。没有理由现在改 batch 或引入梯度累积。batch 或累积步数变化会改变每次 optimizer update 的样本数、噪声/梯度尺度、AdamW 与 EMA 更新频率；不能把这类运行称为无差异的官方复现。

可用卡状态：GPU 0 当时近乎空闲；GPU 1 有其他作业且约 94% 利用率。官方双卡命令会与该作业竞争，不应在该状态下开始长跑。获取两卡独占窗口后再做真实模型/数据的首轮计时。

## 主要资源风险

1. **数据存储是当前明确风险。** 仅完整压缩归档就约 284.7 GiB。当前 466.8 GiB 可用，若压缩包与展开目录同时保留，仅剩约 182.1 GiB（十进制约 195.5 GB），尚未计 checkpoint、临时文件及日志。解压后的实际字节数受权限限制未测，故当前不能批准全量落盘方案。
2. **长时间 GPU 占用与共享负载。** 实际约 61.1k 个 optimizer update，外加每轮数据读取；没有实际 PLY 的 step time，不能可靠报小时数。应以获批样本 loader 检查和后续一个完整 epoch 的实测耗时外推。
3. **CPU/I/O 与 swap。** 默认 `num_workers=16`；每个 batch 的数据在主机侧采样、再送入 GPU。当前 swap 几乎全满，启动前应检查是否有可回收的系统压力，并关注磁盘吞吐和 DataLoader 等待时间。
4. **协议边界。** 官方“300 epoch”配置对应代码 301 次循环；train/test split 也有 520/520 路径重叠。执行时应原样保留并在复现说明中注明，不应私自修补后仍称 bit-for-bit 官方协议。

## 判断

显存容量候选：**可行，尚未用真实数据验证**。当前整机状态：**不具备直接开始完整复现的条件**，原因是数据访问未授权、磁盘展开预算未知、GPU 1 被高负载任务占用、无公开 checkpoint。初次加载测试应先从获批的单类别 ZIP 抽取少量 PLY，不改官方训练 YAML。完整数据只有在实际解压体量与 checkpoint 余量确认后才可准备。

来源：

- 官方配置：<https://github.com/Amazingren/Gaussian-JEPA/blob/e73f9b66a2cc50e0a3090f920cab8baef0571d60/cfgs/pretrain/gaussian_jepa.yaml>
- 官方 runner：<https://github.com/Amazingren/Gaussian-JEPA/blob/e73f9b66a2cc50e0a3090f920cab8baef0571d60/tools/runner_pretrain.py>
- 官方使用指南：<https://github.com/Amazingren/Gaussian-JEPA/blob/e73f9b66a2cc50e0a3090f920cab8baef0571d60/docs/getting_started.md>
