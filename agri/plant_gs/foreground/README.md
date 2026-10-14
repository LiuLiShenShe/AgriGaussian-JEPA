# XianKeLai1：SAM 前景专用 gsplat 重建

这个目录是独立、可复现的前景重建实现，不改动上层旧脚本，也不会写入原始 RGB/SAM 目录。
目标是 **现有 SAM 掩膜中的全部物体**：植株、花盆和蓝色小块；不是“只保留植株”。

## 本次数据核查

- 原始 RGB 250 张；247 张可正常解码。`0111.jpg`、`0130.jpg`、`0134.jpg` 文件损坏，排除并记录。
- SAM 有 208 张非空掩膜，但 **SAM 编号与 RGB 编号不一致**。
- 利用 `crop_XXXX.png` 前景与源 RGB 的像素对应关系恢复配对，而不是按文件名配对。
  抽样匹配后，再用全分辨率验证，前景逐像素精确一致比例至少 99.9%；歧义匹配拒绝。
- 205 组高置信度配对。SAM `0024` 的 crop 不可读；SAM `0100`、`0118` 无可靠源 RGB 匹配，未强行加入。
- 247 张可读 RGB 全部通过 COLMAP 定位，205 个已定位视角有可靠掩膜。
- 平均稀疏点重投影误差 0.7193 像素（1080×1920 SfM 工作分辨率）。
- 71,074 个 SfM 点中，46,060 个通过前景筛选，用于高斯初始化。

以上是这次运行的实际核查结果，不是通用数据集的预期数值。

## 为什么这不是“黑背景图片训练”

1. 全 RGB 只用于 SfM 相机定位，丰富背景特征能辅助定位。
2. 初始化只使用满足前景轨迹与多视角掩膜支持的 SfM 点。
3. RGB 损失只在 SAM 前景内计算，SSIM 使用前景支持权重；不以室内背景 RGB 为训练目标。
4. 单独约束前景 alpha 接近 1、背景 alpha 接近 0。因此黑色背景高斯仍会受到惩罚。
5. 对训练出的三维高斯再次做多视角轮廓支持、低不透明度、无效参数和远端漂浮点过滤。
6. 验证时渲染 **不乘 SAM 掩膜**，报告未遮罩渲染的背景 alpha 与轮廓 IoU。

多视角轮廓测试不是深度遮挡测试。遮挡、掩膜误差、细叶和高斯有限渲染尾部意味着不能承诺
所有视角每个背景像素的 alpha 数学上严格等于零；最终结果必须查看 `final/report.json`
及 `final/preview.jpg`，不能只检查套了掩膜的漂亮图片。

## 环境

本次使用 Python 3.13.12、PyTorch 2.9.1+cu128、gsplat 1.6.0、COLMAP CLI 3.9.1
（CPU 特征提取和匹配）、OpenCV、Pillow、NumPy、SciPy、plyfile。
训练/导出使用 GPU 1（RTX A6000），不占用桌面 GPU 0。gsplat CUDA 后端已经通过渲染与反向传播检查。

## 运行

从 `/fj/AgriGaussian-JEPA` 执行。输出目录必须与原始输入目录不同；新实验用新的目录名。

```bash
CODE=Gaussian-JEPA/agri/plant_gs/foreground
WS=/fj/AgriGaussian-JEPA/data/plant_gs/XianKeLai1_fg_20261014

# 1. 配对、审计、生成 SfM 工作图片和同名掩膜。
# --max-unmatched 是显式排除预算，不会降低像素验证阈值。
python "$CODE/prepare.py" \
  --rgb /fj/AgriGaussian-JEPA/data/rgb/XianKeLai1 \
  --sam /fj/AgriGaussian-JEPA/data/sam/XianKeLai1 \
  --workspace "$WS" --max-unmatched 3
# 同一输入重试配对用 --resume。已经完成时无需重跑。

# 2. 全 RGB 相机定位。done 标记允许跳过已完成的步骤。
bash "$CODE/run_sfm.sh" "$WS"

# 3. RGB/掩膜共用去畸变映射，更新相机内参，筛选前景初始化点。
python "$CODE/build_dataset.py" --workspace "$WS"

# 4. 实际训练分辨率 720×1280，179 个训练视角 + 26 个留出视角。
CUDA_VISIBLE_DEVICES=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=8 \
python -u "$CODE/train.py" --data "$WS/dataset" \
  --result "$WS/results/train_30k" --steps 30000

# 5. 清理最佳 checkpoint；同一掩膜用于所有高斯的多视角检查。
CUDA_VISIBLE_DEVICES=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=8 \
python -u "$CODE/export.py" \
  --checkpoint "$WS/results/train_30k/splats_best.pt" --output "$WS/final"

# 6. 检查 Gaussian-JEPA 属性读取兼容性（不是运行 JEPA 训练）。
python Gaussian-JEPA/agri/plant_gs/verify_gsjepa_compatible.py \
  "$WS/final/point_cloud.ply" --json "$WS/final/gsjepa_compatibility.json"
```

训练的高斯数量阈值是约 30 万的软上限（一次 densify 可以略微越过）。
现有完整训练目录不会被重跑覆盖。checkpoint 是推理/导出参数，**不是优化器断点续训文件**。
`history.jsonl`、`status.json` 和重定向到 `logs/` 的日志可查看进度。
代码现已避免在周期性的 opacity reset 后立刻测评：推迟一个训练视角遍历周期再评估。
本次 30k 运行的早期 3000/6000/9000/12000 测评发生在 reset 后，因此暂时透明，
不能代表恢复后的质量；15000 步起不再 reset，最终选择正常恢复后最佳留出得分的 checkpoint。
该修复只改变以后运行的测评时间，不改变学习更新。

## 输出格式与坐标

- `final/point_cloud.ply`：**可渲染的标准 3DGS PLY**，包含位置、SH、opacity、scale、quaternion。
  scale 是对数、opacity 是 logit，不是已经激活的值。四元数已归一化。
- `final/gaussian_centers.ply`：普通 XYZ/RGB 高斯中心点云，适合点云查看器；**不是完整高斯模型**。
- `final/splats_foreground.pt`：清理后的 gsplat 参数。
- `final/preview.jpg`：留出视角原图前景 / **未经掩膜的渲染** / 渲染 alpha 三列对照。
- `final/renders/*_transparent.png`：正确去预乘的 RGBA 透明图，不是再次套上 SAM 后的抠图。
- `final/report.json`：清理数量、未遮罩渲染指标、PLY 格式校验、归一化信息。
- `final/cleanup_scores.npz`：每个原始高斯的掩膜支持比例、被检查视角数、不透明度和保留决定。
- `pairing.csv`、`preparation.json`：源文件哈希、真实配对与所有排除原因。
- `dataset/meta.json`：实际 SfM、前景初始化数据。
- `results/train_30k/`：未清理的最佳/最终参数及阶段备份。

PLY 位于以主体为中心的归一化坐标系，不表示米制尺寸。
COLMAP 世界坐标转换：`world_xyz = normalized_xyz / normalization.scale + normalization.center`。
将 PLY 与相机参数一起使用时也要应用同一归一化转换；完整信息保存在 checkpoint/report 中。
不使用额外人工尺寸信息时，单目 SfM 的尺度本身是任意的。

清理默认：3 像素掩膜边界容差、至少 10 个视角检验、至少 80% 轮廓支持、opacity ≥0.01。
提高 `--mask-ratio` 会减少残留，也可能删除细叶/花茎；不能盲目把阈值设成 1。

## 检查

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q \
  Gaussian-JEPA/agri/plant_gs/foreground/tests
```

测试覆盖错位编号配对、拒绝歧义/不匹配/空掩膜、三维点投影范围、
“黑色背景高斯也应罚 alpha”、标准 PLY 原始参数格式、opacity reset 后的评估时机。
另外已执行实际 CUDA 反向传播、200 步训练与清理冒烟验证。

## 官方资料

- https://docs.gsplat.studio/main/apis/rasterization.html
- https://docs.gsplat.studio/main/apis/strategy.html
- https://github.com/nerfstudio-project/gsplat
- https://colmap.github.io/cli.html

已使用 Context7 获取 gsplat/COLMAP 文档，并与已安装版本源码/CLI 帮助对照。
用户指定博查搜索此次返回 HTTP 403；未将搜索失败伪装成搜索成功。
