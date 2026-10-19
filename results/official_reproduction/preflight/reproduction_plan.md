# 原版 Gaussian-JEPA 复现准备计划

## 就绪判定

| 项目 | 状态 | 说明 |
|---|---|---|
| 原版核心代码 | PASS | 本地 106 个官方跟踪文件与 commit `e73f9b66a2cc50e0a3090f920cab8baef0571d60` 完全一致 |
| 当前环境 / CUDA 扩展 | PASS（smoke 范围） | PyTorch 2.0.1 + CUDA 11.8；FPS/gather/KNN、合成模型 batch 128/256 均通过 |
| ShapeSplatsV1 权限 | BLOCKED | 当前已登录账号不在 gated 授权列表 |
| ShapeSplatsV1 本地数据 | BLOCKED | 标准项目路径和对应 Hub cache 均无数据 |
| PLY 属性实测 | BLOCKED | 无法读取受限 DATA.md 或 PLY 样本 |
| 全量数据磁盘预算 | BLOCKED | 压缩包约 305.7 GB；解压体量未知，当前可用空间 501.1 GB |
| 官方 checkpoint | NOT AVAILABLE | CHECKPOINTS.md 明确说公开 release 不包含权重；需从随机初始化训练 |
| 完整训练即刻启动 | NO-GO | 同时受数据权限、磁盘确认和 GPU 1 当前高负载约束 |

## 下一阶段准备顺序

1. 在数据集官方页面以真实身份申请访问并接受条款；等待获批。本阶段不绕过许可。
2. 获批后固定数据 revision `1dabf50223bf360307afa4e78539fbd1aa73d65e`。先只下载最小候选 `02954340.zip`（元数据大小 241,517,282 bytes），仅提取一条 train 和一条 test PLY 到隔离 smoke 目录；确认 ZIP 成员路径、PLY header、所有预期字段和官方 loader 输出 shape/finite 值。
3. 用全量文件的 Hub 元数据评估总压缩体积，并先从获批的小样本测 ZIP 展开率。明确数据、输出、checkpoint 同盘预算及清理/保留策略后，才准备完整数据。当前可用 501.1 GB，不能假定足够。
4. 预约 GPU 0 和 GPU 1 空闲窗口。合成 batch 256 已通过单卡 19.98 GiB smoke；下一步是用真实 PLY 做 loader 检查，再在正式配置下计时一个 epoch/确认双卡峰值。不要先改 YAML。
5. 按原配置从零启动。记录代码 SHA、数据 revision、split SHA、环境版本、seed、GPU、实际 epoch 标签、optimizer update 数、日志和 checkpoint SHA。由于公开权重不存在，随机初始化只能是训练起点，不能标为预训练模型。

若进行单类别/少量对象的短训练，它只能叫 pipeline smoke / 小样本试跑，不能称为 ShapeNet55-GS 官方结果。需将子集 split 和临时配置隔离保存；不得改写跟踪的官方 YAML，也不得把非官方 batch/epoch 结果与正式复现混称。

## 授权后少量样本命令

先在仓库根目录、已激活环境中运行。以下命令只下载一个类别归档，不下载完整数据集：

```bash
conda activate gaussian_jepa
export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CONDA_PREFIX/bin:$PATH"
nvcc --version

hf download ShapeNet/ShapeSplatsV1 02954340.zip \
  --repo-type dataset \
  --revision 1dabf50223bf360307afa4e78539fbd1aa73d65e \
  --local-dir data/shapesplat_raw
```

ZIP 内部结构尚未实证。授权后以 Python `zipfile` 查找下面两个 basename，要求各唯一匹配，再只写出这两个文件；若缺失或匹配不唯一则停止检查，不猜路径：

```text
02954340-6c607b6b3d6fc1cdf0d11ae402ef940e.ply  # train
02954340-13295f09e0e9e5d4a6e67635b9d1cee5.ply  # test
```

建立隔离 split：

```bash
mkdir -p data/shapenet_smoke_split data/shapesplat_ply
printf '%s\n' '02954340-6c607b6b3d6fc1cdf0d11ae402ef940e.ply' \
  > data/shapenet_smoke_split/train.txt
printf '%s\n' '02954340-13295f09e0e9e5d4a6e67635b9d1cee5.ply' \
  > data/shapenet_smoke_split/test.txt
export SHAPENET55GS_PLY_ROOT="$PWD/data/shapesplat_ply"
export SHAPENET55GS_SPLIT_ROOT="$PWD/data/shapenet_smoke_split"
```

之后直接构造官方 train Dataset 并读取一个样本（这是数据检查，不是训练）：

```bash
conda run -n gaussian_jepa python -c '
import torch
from utils.config import cfg_from_yaml_file
from datasets.build import build_dataset_from_cfg
c = cfg_from_yaml_file("cfgs/pretrain/gaussian_jepa.yaml")
d = build_dataset_from_cfg(c.dataset.train._base_, c.dataset.train.others)
sample = d[0]
x = sample[2]
assert x.shape == (8192, 14), x.shape
assert torch.isfinite(x).all()
print("loader PASS", sample[0], sample[1], tuple(x.shape))
'
```

若 loader smoke 未通过，先核查授权数据的 PLY 字段/目录映射，不修改 Dataset 来掩盖差异。全量训练 YAML 的 `N_POINTS=8192`、模型 `npoints=1024`、batch 和 epoch 都保持官方值。

## 官方完整训练命令

**只在访问获批、全量数据准备完成、磁盘余量核实、两张 GPU 空闲后执行。此处列出命令，不在本阶段运行。**

```bash
cd /fj/AgriGaussian-JEPA/Gaussian-JEPA
conda activate gaussian_jepa
export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CONDA_PREFIX/bin:$PATH"
export CUDA_VISIBLE_DEVICES=0,1
export SHAPENET55GS_PLY_ROOT="$PWD/data/shapesplat_ply"
export SHAPENET55GS_SPLIT_ROOT="$PWD/datasets/shapenet_split"

python main.py \
  --config cfgs/pretrain/gaussian_jepa.yaml \
  --exp_name gaussian_jepa \
  --launcher none \
  --seed 0
```

该命令保持官方配置；默认输出到 `experiments/gaussian_jepa/pretrain/gaussian_jepa/`，与现有 `experiments/agri_pilot` 分离。不要添加 `--total_bs`、不要改 YAML、不要添加梯度累积。如果显存/设备状况不支持原配置，应先停止并记录差异，再单独审批非官方变体。注意该公开 runner 按 epoch 0..300 实际执行 301 轮；若任务要求严格 300 次数据遍历，必须先向项目负责人确认如何处理，不能悄悄改代码后声称完全符合公开协议。

## 五个问题的直接回答

1. **能否按公开协议在当前服务器复现？** 代码和当前模型显存 smoke 条件满足；当前整体 NO-GO。访问批准、数据与磁盘确认、GPU 独占窗口缺一不可。公开代码自身还存在 300 配置对应 301 次循环的协议细节。
2. **当前缺少什么？** ShapeSplatsV1 授权与数据、实测 PLY 兼容性、解压后磁盘预算、可公开取得的 checkpoint（公开版没有）、GPU 1 空闲窗口。若没有 checkpoint，完整流程必须从零训练。
3. **首次少量数据测试要准备什么？** 获批后只下载 `02954340.zip`，从中仅提取一条 train/test PLY；用独立 smoke split 和官方 Dataset 类验证属性字段、路径、8192×14 输出及有限值。此后才考虑单独标记的非官方小样本短训；不能把它当正式复现。
4. **300 epoch 的主要资源风险？** 约 305.7 GB 压缩数据、未知的展开体积、当前 501.1 GB 可用空间；约 61.1k 次实际 optimizer update 的长时间 GPU/数据 I/O 占用；GPU 1 目前 94% 利用率；RAM swap 近满。显存 smoke 显示 48 GB/卡有余量，但未测真实数据双卡长跑。
5. **下一阶段准确命令？** 先完成官方授权，然后使用本报告“授权后少量样本命令”中的 pinned-revision 单 ZIP 下载和 loader 检查；完整从零训练只在全部 gate 解除后，执行上面的官方 `python main.py ... --seed 0` 命令。当前不能直接运行全量下载或训练。

## 来源

- 官方仓库与训练指南：<https://github.com/Amazingren/Gaussian-JEPA>
- 官方 checkpoint 声明：<https://github.com/Amazingren/Gaussian-JEPA/blob/e73f9b66a2cc50e0a3090f920cab8baef0571d60/CHECKPOINTS.md>
- gated 数据集与申请入口：<https://huggingface.co/datasets/ShapeNet/ShapeSplatsV1>
- Context7 官方 Hub 文档：<https://huggingface.co/docs/huggingface_hub/guides/download>
