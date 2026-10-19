# ShapeSplatsV1 数据要求与状态

## 访问状态

数据源：<https://huggingface.co/datasets/ShapeNet/ShapeSplatsV1>  
审计时 revision：`1dabf50223bf360307afa4e78539fbd1aa73d65e`  
数据卡最后更新时间：2026-09-08。仓库标记为 `gated: manual`。

服务器已有 Hugging Face 登录凭证，但当前账号**未获数据许可**：官方 API 对受限文件返回 `403 GatedRepo`，错误明确表示账号不在授权列表中。README 可以读取，数据内容和 `DATA.md` 受限。没有尝试绕过许可，没有下载数据，也没有读取或记录凭证值。需要通过数据页正式申请并接受其数据条款后再访问。

项目 `data/` 只有 `rgb`、`plant_gs`、`sam`；预期的 `data/shapesplat_ply`、仓库内 `data/shapesplat_ply`、`/data/shapesplat_ply` 和对应 Hugging Face cache 均不存在。搜索到的其他 PLY 是农业/其他项目文件，不能替代 ShapeSplatsV1。

## 官方 split 与磁盘

官方仓库跟踪的 `datasets/shapenet_split/train.txt` 和 `test.txt` 与官方 commit 完全一致：

| 列表 | 行数 | taxonomy 数 |
|---|---:|---:|
| train | 51,934 | 55 |
| test / 当前 pretrain val 配置使用的列表 | 520 | 43 |

本地列表中的 520 个 test 路径全部也出现在 train 列表中。原版预训练 runner 构造了 test loader，但没有调用文件中定义的 `validate()`，所以当前预训练入口没有执行该 SVM 验证流程；若将这些 split 用于独立泛化评估，不能把 test 当作与 train 不相交的留出集。严格复现时保留官方列表，并在报告中说明重叠，不自行重划分。

Hugging Face 元数据列出 57 个 ZIP、55 个 taxonomy ID，总压缩文件大小 `305,670,868,287` bytes（约 305.7 GB / 284.7 GiB）。这仅是下载压缩包大小；由于权限未获批，无法核实 ZIP 展开后的 PLY 总量、训练输出和临时空间需求。当前文件系统可用 `501,143,281,664` bytes（约 501.1 GB）；若保留压缩包，理论剩余约 195.5 GB，不能据此断定解压空间足够。因此不应开始全量下载；完整数据磁盘预算目前未通过审计。

最小单类归档候选 `02954340.zip` 为 241,517,282 bytes（约 230.3 MiB）。官方 split 中有该类的 train 样本 `02954340-6c607b6b3d6fc1cdf0d11ae402ef940e.ply` 与 test 样本 `02954340-13295f09e0e9e5d4a6e67635b9d1cee5.ply`。需获批后确认 ZIP 内部成员路径，再只提取这两个 PLY。该方案下载一个类别 ZIP，而不是整个数据集。

## PLY 与 loader 契约

原版 loader 使用 `plyfile.PlyData.read`，并要求顶点字段至少包含：

- `x`, `y`, `z`
- `opacity`
- `scale_0` 至 `scale_2`
- `rot_0` 至 `rot_3`
- `f_dc_0` 至 `f_dc_2`

loader 将 opacity logits 做 sigmoid、scale 字段做 exp、归一化 quaternion，并取 3 个 DC SH 通道，组合为 14 维 Gaussian 特征。数据卡说明 PLY 使用自定义 vertex 属性，但受限的 `DATA.md` 和任何实际 PLY header 都无法读取。因此字段名/类型与 loader 的真实兼容性**尚未实证**，必须由授权后的小样本读取测试确认。

官方配置默认 `GS_PATH: data/shapesplat_ply`，支持 `SHAPENET55GS_PLY_ROOT` 与 `SHAPENET55GS_SPLIT_ROOT` 环境变量；根目录下文件名必须与 split 中的 `taxonomy-model.ply` 完全一致。

## 授权后的最小加载测试

以下操作仅在获批后执行。它下载一个最小类别归档，并从中只写出两个 split 所需 PLY：

```bash
hf download ShapeNet/ShapeSplatsV1 02954340.zip \
  --repo-type dataset \
  --revision 1dabf50223bf360307afa4e78539fbd1aa73d65e \
  --local-dir data/shapesplat_raw

mkdir -p data/shapenet_smoke_split data/shapesplat_ply
printf '%s\n' '02954340-6c607b6b3d6fc1cdf0d11ae402ef940e.ply' \
  > data/shapenet_smoke_split/train.txt
printf '%s\n' '02954340-13295f09e0e9e5d4a6e67635b9d1cee5.ply' \
  > data/shapenet_smoke_split/test.txt
```

ZIP 内部目录尚未验证；授权后用 `zipfile` 按 basename 查找并要求每个 basename 恰好匹配一个成员，再将这两个成员写到 `data/shapesplat_ply/`。随后以两个环境变量指向这两个 smoke 根目录，分别构造官方 train/val Dataset 并调用 `dataset[0]`，核验返回 14 维数据、finite 数值、采样数量及 PLY 字段。该步骤只验证加载，不训练，也不代表全量数据审计完成。

来源：

- 数据集卡与许可说明：<https://huggingface.co/datasets/ShapeNet/ShapeSplatsV1>
- 锁定 revision 的 README：<https://huggingface.co/datasets/ShapeNet/ShapeSplatsV1/blob/1dabf50223bf360307afa4e78539fbd1aa73d65e/README.md>
- Hub CLI 下载参数文档：<https://huggingface.co/docs/huggingface_hub/package_reference/cli#hf-download>
- Hub 文件下载指南：<https://huggingface.co/docs/huggingface_hub/guides/download>
