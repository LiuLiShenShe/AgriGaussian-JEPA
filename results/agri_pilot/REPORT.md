# Stage 3 — Small-Scale Plant Gaussian-JEPA Pilot

<!-- Historical output: interpreter paths below are the ones actually used at the time. The project now uses uv (env/bootstrap.sh); re-run with $AGRI_PY_GS / $AGRI_PY_JEPA. -->


生成时间：2026-10-15（Asia/Shanghai）  
实验分支：`agri/plant-gs-experiments`  
数据范围：`/fj/AgriGaussian-JEPA/data/plant_gs/F2DMAS_fg_20261014/manifest.json`  
本报告只覆盖现有 19 株植物，没有下载或加入外部数据集，也没有修改官方 Gaussian-JEPA Encoder、Predictor 或 Loss。

## 结论

这次实验确认了真实植物 PLY 可以通过农业数据加载器进入官方 Gaussian-JEPA，并完成从随机初始化开始的训练、EMA Teacher 更新、检查点保存/恢复和固定测试集评估。训练集上的总 Loss 从第 1 个 Epoch 的 `1.0800` 降到第 20 个 Epoch 的 `0.2535`。

表征结果是混合的：连续空间缺失下的 Partial-to-Complete R@1 从随机初始化的 `0.417/0.333/0.333`（缺失 30%/55%/70%）变为训练后的 `1.000/1.000/1.000`，但训练后不同测试植物的平均 Cosine Similarity 变为 `0.999695`，触发了跨植物相似度过高的坍缩警告；Embedding 方差从 `8.319e-06` 降到 `1.818e-07`。因此，本次结果证明“流程能跑通”，但还不能证明模型已经学到可靠的植物差异表征或农业泛化能力。

在当前证据下，不建议直接把这个 20 Epoch 结果当作大规模预训练有效性的证明。后续扩充数据前，应先确认这种跨植物相似度升高是否由训练目标、全局 mean/max pooling、样本规模或背景/花盆共因造成；本阶段按要求到此停止，等待下一阶段确认。

## 1. 数据审计与固定划分

- Manifest 中有 `19` 份可用 PLY：18 份 `complete` 和 1 份 `reused`。
- `BaiZhang` 未纳入，因为 manifest 状态为 `missing_sam`，原因是没有对应 SAM mask。
- 每份 PLY 都存在，manifest Gaussian 数量与 PLY 顶点数一致，解码后为 14 维属性 `[xyz, opacity, scale(3), rotation(4), f_dc(3)]`。
- 所有解码属性均为有限值，解码 scale 均为正；Gaussian 数量范围为 `265,263`–`309,692`。
- 固定划分种子为 `20261015`，单位是整株植物 PLY，训练/测试无交集，禁止同一植株采样结果跨集合。

训练集（15 株）：

`CaoMei1, CaoMei2, ChangShouHua1, ChangShouHua2, ChangShouHua3, DouBanLv1, DouBanLv3, HongZhang, KongQueZhuYu, WanNianQing1, WanNianQing2, WangWenCao1, WangWenCao2, XianKeLai3, XiangPiShu1`

保留测试集（4 株）：

`DouBanLv2, XianKeLai1, XianKeLai2, XiangPiShu2`

每株审计摘要如下；完整字段保存在 [`data_audit.json`](../../experiments/agri_pilot/data_audit.json)。

| 植株 ID | Gaussians | Silhouette IoU | Background alpha | 审计警告 |
|---|---:|---:|---:|---|
| CaoMei1 | 277,541 | 0.9530 | 0.0055 | — |
| CaoMei2 | 309,692 | 0.9651 | 0.0050 | — |
| ChangShouHua1 | 284,359 | 0.9715 | 0.0090 | — |
| ChangShouHua2 | 273,584 | 0.9682 | 0.0098 | — |
| ChangShouHua3 | 274,469 | 0.9739 | 0.0102 | — |
| DouBanLv1 | 273,378 | 0.9617 | 0.0058 | — |
| DouBanLv2 | 275,735 | 0.9695 | 0.0081 | — |
| DouBanLv3 | 291,908 | 0.9690 | 0.0084 | — |
| HongZhang | 280,739 | 0.9617 | 0.0194 | — |
| KongQueZhuYu | 295,872 | 0.9759 | 0.0091 | — |
| WanNianQing1 | 287,325 | 0.9631 | 0.0117 | — |
| WanNianQing2 | 287,587 | 0.9667 | 0.0096 | — |
| WangWenCao1 | 285,611 | 0.9249 | 0.0081 | — |
| WangWenCao2 | 276,717 | 0.9665 | 0.0032 | — |
| XianKeLai1 | 287,079 | 0.9708 | 0.0083 | — |
| XianKeLai2 | 305,367 | 0.9667 | 0.0077 | — |
| XianKeLai3 | 286,024 | 0.9581 | 0.0111 | — |
| XiangPiShu1 | 265,263 | 0.8582 | 0.0400 | silhouette_iou, background_alpha |
| XiangPiShu2 | 292,511 | 0.9126 | 0.0363 | background_alpha |

有 manifest 质量警告的植株为 `XiangPiShu1, XiangPiShu2`。当前数据定义保留 SAM 前景中的植物、花盆和 marker，因此花盆/marker 没有被语义上排除；残余背景 alpha 也可能被模型当作输入信号。

## 2. 训练配置与可复现性

- 随机种子：`20261015`。
- 训练：15 株、batch size `2`、20 Epoch、每个样本采样 `1024` 个 Gaussian。
- 归一化：`pretrain`，先对完整 PLY 归一化，再按固定 Epoch 种子采样。
- 优化器：AdamW，learning rate `0.0003`，weight decay `0.05`；Cosine scheduler，EMA momentum `0.996`→`0.9999`。
- 官方模型配置：`G=64`、`S=32`、`M=4`、target sizes `[11, 9, 7, 5]`、Transformer dimension `384`、Encoder depth `12`、Predictor depth `4`、`lambda_ground=0.1`。
- 参数量：总参数 `46,269,376`，可训练参数 `24,442,944`，Student Encoder `21,826,432`。
- 环境：`NVIDIA RTX A6000`，PyTorch `2.0.1`，NumPy `1.26.4`。
- 官方模型源文件 SHA256：`7d5ad18656a6a53f404f2963ae86ebc04c1f25f060decbd73439566ad377e4a3`。
- 采样索引：训练集每个 Epoch 保存在 [`experiments/agri_pilot/pilot/sampling_indices/`](../../experiments/agri_pilot/pilot/sampling_indices/)；测试集固定采样协议保存在 [`fixed_test_sampling_indices.npz`](fixed_test_sampling_indices.npz)，测试采样种子为 `4101, 4102, 4103`。

实际训练命令：

```bash
/home/test/biosoft/enter/envs/gaussian_jepa/bin/python -u agri/plant_gs/jepa/pilot_train.py --stage pilot --epochs 20 --initial-checkpoint experiments/agri_pilot/initial_random.pt --output experiments/agri_pilot/pilot --device cuda:0
```

Sanity Check 使用 3 株训练植物完成 2 Epoch，并从第 1 个 Epoch 的 checkpoint 恢复到第 2 个 Epoch。训练日志显示所有 batch 的 Student Encoder 梯度有限且非零，EMA Teacher 每个 batch 都发生更新。

## 3. 训练日志

Loss 曲线：[`loss_curves.png`](loss_curves.png) / [`loss_curves.pdf`](loss_curves.pdf)。逐 batch 日志：[`train_metrics.csv`](../../experiments/agri_pilot/pilot/train_metrics.csv)。

| 项目 | Epoch 1 | 最低/代表值 | Epoch 20 |
|---|---:|---:|---:|
| geo | 0.443969 | 0.020305 | 0.021877 |
| app | 0.439749 | 0.019606 | 0.019606 |
| ground | 0.196278 | 0.196278 | 0.212008 |
| total | 1.079996 | 0.253172 | 0.253492 |

训练统计：平均每 Epoch `1.75` 秒，峰值 GPU allocated `0.645` GiB；160/160 个 batch 的 Student 梯度有限，160/160 个 batch 的 EMA Teacher 更新标志为真。

训练后 checkpoint：[`checkpoint_last.pt`](../../experiments/agri_pilot/pilot/checkpoint_last.pt)。随机初始化 checkpoint：[`initial_random.pt`](../../experiments/agri_pilot/initial_random.pt)。

## 4. 固定测试集前后对照

评估命令：

```bash
/home/test/biosoft/enter/envs/gaussian_jepa/bin/python -u agri/plant_gs/jepa/pilot_eval.py --baseline experiments/agri_pilot/initial_random.pt --trained experiments/agri_pilot/pilot/checkpoint_last.pt --output results/agri_pilot --device cuda:0
```

四株测试植物的 Gaussian 索引、采样种子和模型标签全部固定。重采样指标中的不同采样是同一植株的重复测量，不能当作相互独立的生物学样本。

| 指标 | 随机初始化 | Pilot 训练后 | 训练后−随机 |
|---|---:|---:|---:|
| Resampling cosine to reference | 0.997368 | 0.999816 | +0.002448 |
| Resampling L2 to reference | 0.070199 | 0.019164 | -0.051035 |
| Same-plant cosine | 0.996800 | 0.999822 | +0.003022 |
| Different-plant cosine | 0.989348 | 0.999695 | +0.010347 |
| Embedding variance | 0.000008 | 0.000000 | -0.000008 |
| Effective rank | 1.820354 | 2.052473 | +0.232120 |

连续空间缺失使用固定 XYZ 方向上的 projection cap，先移除投影最高的一侧，保证缺失区域空间连续；每个测试植物使用 3 个方向。

| 连续缺失比例 | 随机 partial→same cosine | 随机 R@1 | 训练后 partial→same cosine | 训练后 R@1 |
|---:|---:|---:|---:|---:|
| 30% | 0.945923 ± 0.035154 | 0.417 | 0.999875 ± 0.000038 | 1.000 |
| 55% | 0.858414 ± 0.100902 | 0.333 | 0.999791 ± 0.000081 | 1.000 |
| 70% | 0.803261 ± 0.123399 | 0.333 | 0.999713 ± 0.000093 | 1.000 |

结果 JSON、NPZ embedding 和 CSV 对照表：[`evaluation_comparison.json`](evaluation_comparison.json)、[`random_init_metrics.json`](random_init_metrics.json)、[`pilot_trained_metrics.json`](pilot_trained_metrics.json)、[`metrics_comparison.csv`](metrics_comparison.csv)。

## 5. 坍缩检查

随机初始化模型的描述性指标为：跨植物平均 Cosine `0.989348`、Embedding 方差 `8.319e-06`、有效秩 `1.8204`。训练后分别为 `0.999695`、`1.818e-07`、`2.0525`。

训练后触发的描述性警告：`different_plant_cosine_above_0.99=True`；方差阈值和有效秩阈值在本次四株测试中未触发。这个判断只能作为小样本诊断，不能作为正式的群体级坍缩证明。R@1 全部为 1.0 与跨植物 Cosine 过高同时出现，说明当前检索结果可能主要反映表征对遮挡的共同稳定性，不能单独解释为学到了植物身份差异。

## 6. 背景、花盆和数据限制

- SAM 前景的目标定义包含保留下来的花盆和 marker；这些因素与植物 ID、采集设置可能相关，形成混杂信号。
- 审计中的最大 background alpha 为 `0.0400`；质量警告集中在 `XiangPiShu1, XiangPiShu2`，其中 `XiangPiShu1` 的 silhouette IoU 约为 `0.8582`。
- 只有 19 株植物，测试集只有 4 株；没有多随机种子重复，也没有外部植物数据或语义标签。
- 训练采用 batch size 2、20 Epoch，是连通性和初步行为验证，不是正式超参数搜索或大规模预训练。
- 评估使用 mean/max pooling 的固定 768 维 embedding；这种 pooling 可能弱化局部植物结构差异。
- 本报告不把训练 Loss 下降、R@1 提升或单次有效秩结果外推为农业泛化能力。

## 7. 阶段结论

Stage 3 的工程验证通过：真实植物 PLY、农业 Dataset、官方 Gaussian-JEPA Encoder/Predictor/Loss、优化器、EMA、checkpoint/resume 和固定评估协议均已跑通。科学结论暂定为“训练产生了明显的训练损失下降和遮挡稳定性，但同时出现跨植物表征过于相似的信号，尚不足以证明植物可区分表征已经学好”。

本阶段没有启动 PAA、TomatoMAP 等外部数据集，也没有开始正式大规模预训练。是否进入数据扩充阶段，等待后续确认。
