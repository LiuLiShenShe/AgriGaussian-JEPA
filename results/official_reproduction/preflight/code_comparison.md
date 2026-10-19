# Gaussian-JEPA 代码对比

## 提交与文件对比

| 对象 | SHA / 数量 |
|---|---|
| 官方 `Amazingren/Gaussian-JEPA` 当前公开提交 | `e73f9b66a2cc50e0a3090f920cab8baef0571d60`（2026-08-24） |
| 本地分支 tip | `af2f510ee1e7f284ffb3f5a2ec6c4605b55bbcc3` |
| 本地相对官方基线新增的提交 | 6 个农业实验提交；官方基线是当前 tip 的祖先 |
| 官方跟踪文件 | 106 个；逐文件与官方提交比较均完全一致 |
| 本地额外跟踪文件 | 120 个，全部位于 `agri/` 和 `results/` |

核心模型 `models/Gaussian_JEPA_ExpMultiScale.py`、`models/neural_nn.py`、`models/transformer.py`、模型 registry、ShapeNet Dataset、PLY IO、预训练 runner、优化器构建器、预训练 YAML、ShapeNet55-GS Dataset YAML、官方 split 和 `environment.yml` 均与官方提交字节一致。原版 Dataset、split 和配置没有被农业实验修改。Stage 3 / 3.5 的代码与结果是本地新增内容；没有覆盖或回退它们。

关键文件 SHA-256（本地值；同官方文件）：

| 文件 | SHA-256 |
|---|---|
| `models/Gaussian_JEPA_ExpMultiScale.py` | `7d5ad18656a6a53f404f2963ae86ebc04c1f25f060decbd73439566ad377e4a3` |
| `models/neural_nn.py` | `f327106a676b3187a7c140cf0af320ae081f11a37c3d9f8417a67a16e97ac1c3` |
| `models/transformer.py` | `ab01e9526ceac9aff8359fc56ef0c31300b13b555abd5b5c46e6e13677875eba` |
| `datasets/ShapeNet55Gaussian.py` | `fb34c666509819678c4620c28e492920bd23d36c799a4c8977a069f726941be8` |
| `cfgs/pretrain/gaussian_jepa.yaml` | `f5d142513ebc57969946442ce41269054c5dd8eb3c7d9313aaa2fd63a4116400` |
| `datasets/shapenet_split/train.txt` | `d6b2ff4d11f3517542f9fdc8b518fe08466e37bd7bc71e6ef38613a048efd61e` |
| `datasets/shapenet_split/test.txt` | `4e655bd75ae72319817f33ff2a34a8d1d8db7e61b00619f05d82749b643f7c28` |

## CUDA 扩展

官方 `environment.yml` 固定 Python 3.9、PyTorch 2.0.1、PyTorch CUDA 11.8，并从源码安装 `pointnet2_ops` 和 `knn_cuda`。本机 `gaussian_jepa` 环境版本吻合，两个已安装扩展均可导入；最小 FPS/gather/KNN 调用以及官方合成数据 smoke test 通过。环境内 CUDA_HOME/nvcc 为 11.8。系统 PATH 还暴露 CUDA 12.6，因此未来编译时应显式固定 `CUDA_HOME="$CONDA_PREFIX"` 和 PATH 顺序。

## 官方模型与训练入口

官方配置位于 `cfgs/pretrain/gaussian_jepa.yaml`：1,024 输入 Gaussians、64 groups、每组 32 个邻居、总 batch 256、`step_per_update=1`、`max_epoch=300`、AdamW（lr `0.001`、weight decay `0.05`）、Cosine LR（300 epoch、10 epoch warmup、最小 lr `1e-6`）、EMA momentum `0.996 -> 0.9999`、seed `0`。训练入口是 `main.py`，官方指南命令为：

```bash
python main.py \
  --config cfgs/pretrain/gaussian_jepa.yaml \
  --exp_name gaussian_jepa \
  --launcher none \
  --seed 0
```

`--launcher none` 在 runner 中使用 `torch.nn.DataParallel`；可见两张 GPU 时，batch 256 会分到两卡。Dataset 每个对象先采样 8,192 个 PLY Gaussians，再由 runner 随机选 1,024 个作为模型输入。

**原代码轮数边界需记录。** 配置和文档称 300 epochs，但 `tools/runner_pretrain.py` 使用 `range(start_epoch, config.max_epoch + 1)` 且从 0 开始，实际 epoch 标签为 0..300，共 301 次数据遍历。非分布式 DataLoader 不丢弃最后一个 batch：51,934 条训练记录对应 203 steps/epoch，因此该命令按当前代码会执行约 61,103 次 optimizer update，而非 60,900 次。EMA 分母仍使用 `n_batches * max_epoch`。本审计不改官方实现；报告结果时应披露该实现细节。

## Checkpoint 状态

官方 `CHECKPOINTS.md` 明确写明公开 release 不包含预训练权重。文档列出参考文件名、380,976,155 bytes 和 SHA-256，但没有下载 URL；仓库也无 `.pth` / `.pt` / `.ckpt` 文件。本地 `checkpoints/` 目录不存在。`docs/getting_started.md` 却要求从 `CHECKPOINTS.md` 链接文件下载，和 checkpoint 页面自身说明不一致。当前可验证状态是：无可获取的官方公开权重，必须从随机初始化开始训练；不得把 smoke test 随机初始化结果称为预训练 checkpoint。

来源：

- 官方仓库提交：<https://github.com/Amazingren/Gaussian-JEPA/tree/e73f9b66a2cc50e0a3090f920cab8baef0571d60>
- 官方训练指南：<https://github.com/Amazingren/Gaussian-JEPA/blob/e73f9b66a2cc50e0a3090f920cab8baef0571d60/docs/getting_started.md>
- Checkpoint 清单：<https://github.com/Amazingren/Gaussian-JEPA/blob/e73f9b66a2cc50e0a3090f920cab8baef0571d60/CHECKPOINTS.md>
