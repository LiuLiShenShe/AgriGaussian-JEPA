# Stage 4A — PAA 单株 3DGS 重建及 Gaussian-JEPA 接入验证

**状态：完成（含 7 项已知遗留问题，见 §11）**

- 执行日期：2026-10-19
- 分支：`agri-stage4-paa-pilot`（commit `af2f510`，工作区在本次运行前为干净状态）
- 硬件：2 × NVIDIA RTX A6000 48GB（`nvidia-smi` 实测），驱动 590.48.01 / CUDA 13.1，nvcc 12.6.85
- 样本：`Broccoli/20260327-10`（PAA-Plant-Dataset，Apache-2.0）
- 独立复核：同日 16:50–17:10，见 §13（逐项复跑 + 位级一致性核对 + 视觉复核）

本报告的数值全部由本机实际运行产生，可追溯到 `results/agri_paa/*.json` 与
`data/plant_gs/PAA/Broccoli_20260327-10/**`。凡属推测，均在文中显式标注为「推测」。

---

## 1. 项目 Git 状态

| 项 | 实测值 |
|---|---|
| 仓库根 | `/fj/AgriGaussian-JEPA/Gaussian-JEPA`（`/fj/AgriGaussian-JEPA` 本身不是 git 仓库） |
| 远端 | `git@github.com:LiuLiShenShe/AgriGaussian-JEPA.git` |
| 当前分支 | `agri-stage4-paa-pilot`（已存在，本次未新建、未切换） |
| 运行前 HEAD | `af2f510` — *feat(agri): 添加Stage 3.5-C离线误检归因分析脚本* |
| 运行前工作区 | 干净，仅有未跟踪目录 `results/official_reproduction/`（本次未触碰） |

既有分支：`main`、`agri/plant-gs-experiments`、`agri-stage4-paa-pilot`。本次没有
rewrite、没有 reset/clean、没有删除任何历史文件。

### 两条研究路线的区分（任务明确要求先识别）

仓库中并存两条独立工作，本次只动 3D 那条：

- **I-JEPA / 二维 ViT 路线**：不在本仓库内，位于 `/fj/jepa-二维图像任务/`（另一个 venv）。
  本次完全未读取、未修改。
- **Gaussian-JEPA / 三维高斯路线**：本仓库。含官方代码（`models/`、`datasets/`、
  `tools/runner_*.py`、`main.py`）与农业扩展（`agri/plant_gs/**`）。本次只新增
  `agri/plant_gs/paa/**` 与 `agri/plant_gs/jepa/optimizer_probe.py`，未改动任何
  既有模块。

`data/plant_gs/F2DMAS_fg_20261014/`（18 株 F2DMAS 植物）与
`data/plant_gs/XianKeLai1_fg_20261014/` 是**旧成果**，位于**仓库外**的
`/fj/AgriGaussian-JEPA/data/plant_gs/`（20 GB / 1.2 GB），本次全程只读，未删除、
未覆盖、未移动。本次新增的 PAA 产物在**仓库内** `Gaussian-JEPA/data/` 下
（external 239 MB + plant_gs/PAA 1.3 GB），两个数据根刻意分开，批量阶段再统一。

---

## 2. 现有可复用实现（复用而非重写）

| 组件 | 路径 | 本次是否使用 |
|---|---|---|
| 3DGS 前景训练器 | `agri/plant_gs/foreground/train.py` | **直接使用**（未改一行） |
| 高斯清理与 PLY 导出 | `agri/plant_gs/foreground/export.py` | **直接使用** |
| COLMAP 二进制读取 | `agri/plant_gs/colmap_io.py` | **直接复用** |
| PLY 结构校验 | `agri/plant_gs/validate_ply.py` | **直接使用** |
| Gaussian-JEPA 读取器回放 | `agri/plant_gs/verify_gsjepa_compatible.py` | **直接使用** |
| 官方数据读取 + 归一化 | `agri/plant_gs/jepa/dataset.py` | **直接复用** |
| 真实 PLY 前向/反向测试 | `agri/plant_gs/jepa/smoke_real_plant.py` | **直接使用** |
| gsplat 1.6.0 | base env（`/home/test/biosoft/enter`） | 训练用 |
| Gaussian-JEPA 官方模型 | `models/Gaussian_JEPA_ExpMultiScale.py` | `gaussian_jepa` env |

**没有可用官方 3DGS 仓库**（未 clone graphdeco-inria/gaussian-splatting）；项目内已有
`foreground/` 管线在 F2DMAS 18 株上验证过，优先复用它，符合任务要求。

---

## 3. 环境版本与 GPU（实测）

| 用途 | 环境 | Python | PyTorch | 其他 |
|---|---|---|---|---|
| 3DGS 训练/导出/校验 | `/home/test/biosoft/enter`（base，未改动） | 3.13.12 | 2.9.1+cu128 | gsplat 1.6.0, opencv 4.13.0, numpy 2.4.6 |
| Gaussian-JEPA 前向/反向 | `/home/test/biosoft/enter/envs/gaussian_jepa`（既有，未重建） | 3.9.23 | 2.0.1+cu118 | knn_cuda / pointnet2_ops 已编译 |

GPU：2 × RTX A6000，各 49140 MiB。训练用 GPU 1（`CUDA_VISIBLE_DEVICES=1`），
Gaussian-JEPA 测试用 GPU 0。

**注意**：`gaussian_jepa` env 的 opencv 是 5.0.0，其 Python 绑定在该 env 中拒绝所有
非自身创建的 numpy 数组（`cv2.resize(np.zeros(...))` 报
`src is not a numpy array`），因此所有图像处理都改用 base env，所有 cv2 调用都改为
`np.ascontiguousarray`。这不是本阶段引入的问题（既有 env 既有），但影响脚本选型，
故记录。

---

## 4. PAA 数据来源、许可与下载

- 来源：`https://huggingface.co/datasets/JackTrum938/PAA-Plant-Dataset`（公开、非 gated）
- 许可：**Apache-2.0**（HF API `cardData.license` 实测）
- 样本：`Broccoli/20260327-10`（2026-03-27，plant_id `10`）
- 下载工具：`hf download`（huggingface_hub 1.8.0），带本地缓存与断点续传
- 实际下载内容：`Camera/**` + `RGB_seg/masks/**` + `RGB_seg/whitebg/**` + `RGB/**`
  （**413 个文件，238 MB**：306 个 jpg = Camera 102 + RGB 102 + whitebg 102，
  102 个 mask png，3 个 COLMAP bin，2 个说明 txt）。
  **未**下载其余 40 个样本，**未**下载 HyperScan（50 MB × 41）。
- 存储：`data/external/PAA-Plant-Dataset/Broccoli/20260327-10/`（238 MB）
- 衍生产物：`data/plant_gs/PAA/Broccoli_20260327-10/`（1.3 GB，含两轮 checkpoint 与 PLY）
- 来源记录：`data/external/PAA-Plant-Dataset/Broccoli/20260327-10/provenance.json`
  （由 `agri/plant_gs/paa/download_paa.py` 生成，含 HF `sha` 与 `lastModified`）

按目录文件数（实测）：`Camera/images` 102、`Camera/sparse/0` 3、`RGB` 102、
`RGB_seg/masks` 102、`RGB_seg/whitebg` 102。

> 备注：下载由 `hf download` 完成，但 `provenance.json` 是后来补写的（脚本
> `download_paa.py` 在下载之后才写完）。因此 provenance 中的 `file_counts` 反映的是
> 最终磁盘状态，不是下载当时的状态；`hub.sha` 来自 HF API，可独立核验。
>
> **复核时的修正**：初次写报告时该文件实际**缺失**（磁盘与 HF API 均未找到），§4/§附
> 中的路径一度是失效引用。复核阶段已按磁盘实测重新生成 `provenance.json`，其中
> `hub.sha = 19e7f2f2307462aba37b8571acc7a7c0f72a8608`、
> `last_modified = 2026-07-28T12:39:25Z`、`license = apache-2.0`、`gated = false`。
> 同时用 HF tree API 核对了远端 `Broccoli/20260327-10/Camera/**` 的文件大小，
> 与本地一致（camera 102 张 + 3 个 bin + 1 个 txt）。

---

## 5. COLMAP 相机与图像检查

工具：`agri/plant_gs/paa/colmap_check.py` → `results/agri_paa/colmap_audit.json`（status **PASS**）

| 检查项 | 结果 |
|---|---|
| `cameras.bin` / `images.bin` / `points3D.bin` | 全部存在（104 B / 5.33 MB / 5.30 MB） |
| 注册图像 | 102；磁盘 JPEG 102；**一一对应，无多余无缺失** |
| 相机模型 | 2 个相机，均为 **SIMPLE_PINHOLE（无畸变）** |
| cam 0 | 2128×3806，f=9917.21，cx=1064，cy=1903，99 张，HFOV 12.2° / VFOV 21.7° |
| cam 1 | 3070×2020，f=3360.83，cx=1535，cy=1010，3 张（100/101/102.jpg），HFOV 49.1° |
| 图像尺寸 vs 相机声明 | **全部一致**（以 EXIF-ignored 帧为准） |
| EXIF orientation | 仅 3 张（100/101/102）带 orientation=6 |
| 位姿 | 旋转矩阵 det ∈ [0.9999999999999993, 1.0000000000000016]，全部有限 |
| 相机轨迹 | 中心 `[-0.234, 0.392, -2.497]`，中位半径 3.904，min 0.543 / max 4.788 |
| 稀疏点 | 69,209 个，全部有限；track 长度 min 1 / 中位 2 / max 35 |
| **重投影误差** | 文件中 `points3D.error` 全为 0（PAA 已清零，不可用）。**重新计算**：221,734 个观测，mean **0.774 px** / median 0.578 px / p95 2.056 px / max 41.1 px |

### 5.1 EXIF 方向：哪一帧才是 COLMAP 解算的那一帧

cam 1 的内参主点 `(1535, 1010)` 是 3070×2020 的中心，而磁盘 JPEG 在 **EXIF-ignored**
读取下正是 3070×2020。这不是推断，是用数据判定过的：把全部 69,209 个点投影到该图，
测量投影点处的图像梯度能量与随机点之比——

| 图像 | stored (EXIF-ignored) | viewed (EXIF-applied) |
|---|---|---|
| 102.jpg | **3.474** | 1.839 |

stored 高出 1.9 倍，说明 COLMAP 的位姿对应 **EXIF-ignored 帧**。若误用 EXIF-applied
帧，掩膜与相机将整体错配 90°，而训练仍能跑——只会训练出一株不是照片里的植物。

（另 99 张 cam 0 图像两种读法尺寸相同，无方向歧义。）

---

## 6. 掩膜对齐：如何确认掩膜与相机帧的对应关系

这是本阶段最关键的一步，因为 PAA 的 `Camera/images` 与 `RGB_seg/masks` **尺寸和方向都不同**：

- `Camera/images`：2128×3806（cam 0）/ 3070×2020（cam 1）
- `RGB_seg/masks`：**2160×3840，竖版，无 EXIF**（所有 102 张都是竖版）
- `RGB_seg/whitebg`：2160×3840 或 3840×2160，EXIF'd 子集带 orientation=6

且两组尺寸比不同（2128/2160 ≠ 3070/3840），所以关系是**裁剪**而非缩放；把掩膜 resize
到帧尺寸会静默糊掉轮廓。工具：
`agri/plant_gs/paa/alignment.py` → `results/agri_paa/alignment_probe.json`

用两个互相独立的证据交叉验证（全 102 张，非抽样）：

**检查 A — 掩膜 vs whitebg 的非白像素**（whitebg 是植物贴白底，其非白像素即分割本身）

| 相机 | IoU 中位 | IoU 最小 |
|---|---|---|
| cam 0（99 张） | **0.9959** | 0.7787 |
| cam 1（3 张） | **0.9767** | 0.9666 |

**检查 B — whitebg vs Camera/images 的 NCC 模板匹配**（多尺度金字塔，定位裁剪偏移）

| 相机 | 偏移 (dx, dy) | 一致性 | NCC 中位 / 最小 | 最大 mean\|diff\| |
|---|---|---|---|---|
| cam 0 | **(16, 24)** | **99/99 一致** | 0.9930 / 0.9826 | 2.81 |
| cam 1 | **(440, 80)** | **3/3 一致** | 0.9956 / 0.9883 | 5.77 |

两个证据都指向同一组偏移，且每个相机的偏移在所有帧上完全一致。

**cam 0 的 IoU 最小 0.7787 已查明原因，不是对齐错误**：差异像素的 RGB 均值
(251.5, 253.9, 254.2)，100% 落在 min 通道 ≥ 235 的范围——即 whitebg 中 JPEG 压缩后
略低于纯白背景，被 `< 235` 阈值判成「非白」。掩膜本身是正确的。这也说明该 IoU 是
**阈值敏感**的度量，不应单独作为验收门限。

### 6.1 用几何独立复核（不依赖 IoU）

把 69,209 个 SfM 点投影到每张帧，统计落在该帧掩膜内的比例：

- 全部 102 帧：min **0.9257** / 中位 **0.9832** / max 0.9900
- 即 98% 的 SfM 点落在掩膜内——点云、内参、位姿、掩膜四者互相一致

### 6.2 掩膜覆盖的是什么（重要限定）

掩膜覆盖的是**整套标本（植株 + 花盆 + 外层支架）**，不只是植株。按颜色分区的像素统计
（image 52，已修正 int16 溢出后测得）：

| 区域 | 被掩膜覆盖比例 | 占掩膜比例 |
|---|---|---|
| 绿色（植株） | 99.51% | 16.78% |
| 蓝色（花盆） | 99.97% | 68.22% |
| 深色（土壤） | 99.97% | 34.02% |
| 白色（背景） | 0.16% | 0.34% |

掩膜的行范围 [831, 1599] 与「非白物体」的行范围完全相同。

**因此本阶段产出的是「植株 + 花盆 + 支架」的高斯资产，不是「纯植株」资产。** 与旧
F2DMAS 工作的目标一致（其 README 明确写 "all verified SAM foreground including
plant/pot/marker"），但若下游要做纯植株表型，需要额外的植株/花盆分离步骤。

> 说明：曾用视觉模型复核该叠加图，得到「掩膜只覆盖叶子、漏掉花盆」的结论。该结论与
> 逐像素统计矛盾，且视觉模型看的是半透明混合后的降采样图。以数值证据为准；
> 数值证据可复算，视觉描述不可。

---

## 7. 3DGS 重建配置与真实训练结果

管线：`agri/plant_gs/paa/prepare_paa.py`（新增）→ `agri/plant_gs/foreground/train.py`
（既有）→ `export.py`（既有）。**没有重跑 SfM**，直接复用 PAA 的 COLMAP 模型。

### 7.1 预处理

| 项 | 值 |
|---|---|
| 工作画布 | **1053×1600**（统一画布，见下） |
| `--max-size` | 1600（长边） |
| 帧数 | 102（0 丢弃；3 张 cam 1 帧经 90° 滚动后保留） |
| 掩膜覆盖率 | min 0.197 / 中位 0.242 / max 0.298 |
| 前景稀疏点 | **68,071 / 69,209（98.4%）** |
| 轨道半径 | 3.904（点云以此为界剔除远端漂浮点） |
| 预处理耗时 | 60.9 s |

**为什么需要「统一画布」**：cam 0 是 2128×3806（竖），cam 1 是 3070×2020（横）。
既有 trainer 把所有帧 stack 成一个张量，混合尺寸会直接 `ValueError: all input arrays
must have the same shape`。两条可选路径：

1. 丢掉 3 张横版帧（少 3% 视角）；
2. **把少数派方向的帧滚动 90° 后统一补齐到公共画布**（本次采用，`--frame-policy uniform-canvas`）。

滚动用
`R_roll = [[0,1,0],[-1,0,0],[0,0,1]]`、`K_new = [[f,0,cy],[0,f,W-1-cx],[0,0,1]]`，
与 `Image.rotate(90, expand=True)` 的像素映射严格互逆——这一点有单元测试保证
（`test_roll_camera_matches_roll_image`：把一个合成像素经滚动相机投影，必须落在
`rotate(90)` 图像中同一个像素上）。

滚动后没有静默错配，用同一把尺子复核：**SfM 点落在掩膜内的比例 min 0.9257 /
中位 0.9832**，与滚动前完全一致（滚动前 min 0.9257 / 中位 0.9832）。若相机没跟着滚，
这个比例会崩。

### 7.2 训练配置（两轮）

| 参数 | 7k 试运行 | 30k 正式 |
|---|---|---|
| steps | 7,000 | 30,000 |
| sh_degree | 3 | 3 |
| foreground_alpha_weight | 0.1 | 0.1 |
| background_alpha_weight | 0.75 | 0.75 |
| max_gaussians | 300,000 | 300,000 |
| seed | 42 | 42 |
| downscale | 1 | 1 |
| eval_every | 1,000 | 3,000 |
| 训练/验证视角 | 89 / 13 | 89 / 13 |
| 初始化点 | 68,071 | 68,071 |
| 耗时 | 175 s | 678.6 s |
| GPU | A6000（GPU 1） | 同 |

**关于留出视角的限定（任务明确要求）**：这 13 个视角是**同一场景内的单场景留出视角**，
仅用于重建质量报告，**不是独立植株泛化测试**，也不能那样解读。

### 7.3 训练曲线（30k，`results/train_30k/history.jsonl`）

| step | fg PSNR | sil IoU | bg alpha | bg opaque frac |
|---|---|---|---|---|
| 3,089 | 20.395 | 0.9797 | 0.00748 | 0.00636 |
| 6,089 | 21.468 | 0.9824 | 0.00649 | 0.00549 |
| 12,089 | 21.889 | 0.9844 | 0.00558 | 0.00485 |
| 18,000 | 22.674 | 0.9860 | 0.00499 | 0.00430 |
| 24,000 | 22.706 | 0.9864 | 0.00483 | 0.00418 |
| 27,000 | **22.909** | **0.9866** | **0.00475** | 0.00412 |
| 29,999 | 22.844 | 0.9864 | 0.00484 | 0.00419 |

7k 试运行终值 PSNR 21.926，**尚未收敛**（PSNR 仍在单调上升），因此按日志决定加训到
30k。30k 在 24k 后基本平台（22.706 → 22.909，+0.2 dB），本次采用 **best = step 27,000**。

### 7.4 导出与清理（`foreground/export.py`）

| 项 | 7k | 30k（采用） |
|---|---|---|
| step | 6,999 | 27,000 |
| 原始高斯 | 331,903 | 300,277 |
| 保留高斯 | 313,936 | **292,517** |
| 剔除（漂浮/低不透明度/掩膜不一致） | 17,967 (5.4%) | 7,760 (2.6%) |
| fg PSNR（清理前 → 后） | 21.926 → 21.926 | 22.909 → 22.908 |
| sil IoU | 0.9830 → 0.9830 | 0.9866 → 0.9866 |

清理几乎无损（PSNR 变化 < 0.001 dB），说明剔除的确实是该剔除的东西。

---

## 8. 渲染质量与主要问题

**定量**（13 个留出视角，`final/validation.json`，未乘掩膜的裸渲染）：

| 指标 | min | 中位 | max |
|---|---|---|---|
| foreground PSNR | 18.94 | 23.59 | 26.06 |
| silhouette IoU | 0.9721 | 0.9897 | — |
| background alpha（掩膜外平均 alpha） | 0.00185 | 0.00449 | 0.00973 |
| background opaque fraction（掩膜外 alpha>0.5 比例） | 0.00142 | 0.00377 | 0.00873 |

即：掩膜声明为空白的像素中，只有 **0.5%** 的像素 alpha 超过 0.5。平均 alpha 0.0048。
这不是「零背景」保证，但是可量化的低泄漏。

**渲染对比图**（6 个留出视角 × [原图 | 渲染 | alpha]）：
`data/plant_gs/PAA/Broccoli_20260327-10/final/renders/step027000_*.jpg`、
`final/preview.jpg`（多视角拼图）。

**已知问题（如实记录）**：

1. **支架（holder）细节不足。** 浅色塑料支架棱角清晰，重建中边缘发圆、部分视角变形。
   PSNR 主导指标对细结构不敏感，IoU 也主要反映大轮廓。
2. **叶片偏「团块」。** 叶脉与层次丢失，颜色和体积尚可。
3. **alpha 边缘有轻微烟雾状半透明过渡**，与上述 bg opaque fraction 0.4% 一致。
4. **花盆内部土壤纹理噪点**（第 6 视角较明显）。
5. 背景 alpha 非零（见上表），**不存在零背景保证**；`export.py` 的
   `limitations` 字段已如实写入。

**未出现的风险**（明确声明）：模型**没有**主要重建背景/花盆而丢掉植株——掩膜覆盖率
24% 全部是标本，bg opaque fraction < 1%，且 98% 的 SfM 点在掩膜内。所以「这是一株
植株+花盆+支架的高斯资产」这个说法成立；「这是纯植株资产」不成立。

---

## 9. 最终 Gaussian 属性统计

工具：`agri/plant_gs/paa/gaussian_stats.py` → `results/agri_paa/gaussian_stats.json`

| 项 | 值 |
|---|---|
| 高斯数 | **292,517** |
| 属性数 | 59（含 45 个 `f_rest_*` 高阶 SH） |
| 14 个 JEPA 必需属性 | **全部存在** |
| NaN / Inf | **0**（逐字段扫描，非抽样） |
| bbox | min `[-0.839, -1.046, -1.362]` max `[0.808, 0.723, 0.620]` |
| bbox 尺寸 | 1.646 × 1.769 × 1.982 |
| 位置（距质心半径） | 中位 0.6131 / p95 1.1067 / p99 1.2454 / max 1.5719 |
| scale（线性） | 中位 2.32e-03 / p95 1.99e-02 / max 2.98e-01 |
| scale 退化 (<1e-6) | 10,071（3.4%） |
| scale > 0.25 | 3 |
| opacity（激活后） | 中位 0.6121 / p5 0.0379 / p95 0.9999 |
| opacity < 0.01 | **0**（导出已按 `min_opacity=0.01` 过滤） |
| opacity < 0.05 | 21,214 |
| opacity > 0.99 | 84,028 |
| 四元数模长 | 1.000000 – 1.000000，非单位 0 |
| 漂浮指示 | radius p99/p50 = 2.03；超出 1.5×p95 半径的 **0** 个；低不透明度占比 0.0000 |

**激活约定（关键，任务明确要求核对）**：

| 字段 | 文件中的形式 | 判定依据 | 是否正确 |
|---|---|---|---|
| `opacity` | raw **logit**（范围 -4.595…13.333，仅 9.9% 落在 [0,1]） | `validate_ply.py` | ✅ logit |
| `scale_*` | raw **log**（全部 < 0，线性后 3e-7…0.298） | 同上 | ✅ log |
| `rot_*` | **已归一化**的单位四元数（导出时 `F.normalize`） | 同上 | ✅ |
| `f_dc_*` | 标准 3DGS **SH DC 系数** `sh0 = (rgb-0.5)/C0`，非原始 RGB（范围 -2.26…7.22） | 见下行位级核对 | ✅ |

`f_dc` 与训练参数的位级核对：PLY 中 `f_dc_i` 与 checkpoint `splats["sh0"][:,i,0]`
**逐元素完全相等（max abs diff = 0.0）**，说明导出没有做任何额外的色彩编码。
Gaussian-JEPA 的 `load_gaussian_ply` 读入后还会做
`2·clip(0.28209479·f_dc, -0.5, 0.5)/√3` 的再归一化（对齐 ShapeNetGaussian 预处理），
所以 `f_dc` 既不是 RGB，也不是直接输入模型的颜色值。

Gaussian-JEPA 官方加载器会自己做 `sigmoid(opacity)` / `exp(scale)` /
`normalize(rot)`，所以导出**必须**保持 raw 形式。本次 PLY 是原始参数，未双重激活。
`verify_gsjepa_compatible.py` 回放官方加载器后确认
`gsjepa_compatible: true`、`opacity_in_unit_interval: true`（sigmoid 后）、
`all_scales_positive: true`（exp 后）、`quat_norms_unit: true`。

### 9.1 PLY 确实是优化结果的位级证据（复核新增）

把 `final/point_cloud.ply` 与 `final/splats_foreground.pt` 以及其源头
`results/train_30k/splats_best.pt` 逐字段比较：

| 字段 | PLY vs 清理后 checkpoint | 清理后 checkpoint vs 训练 checkpoint |
|---|---|---|
| `x,y,z` | max abs diff **0.0** | 子集选取，逐值一致 |
| `opacity` | max abs diff **0.0** | 同上 |
| `scale_*` | max abs diff **0.0** | 同上 |
| `rot_*` | max abs diff **0.0**（对比归一化后的 quat） | 导出时 `F.normalize`，允许 sign 翻转 |
| `f_dc_*` | max abs diff **0.0** | 同上 |

即 PLY 是被训练器实际更新过的参数张量，经清理子集后原样写出；没有任何随机
生成或 XYZRGB 改写。训练 checkpoint 的 `step = 27000`、`config.steps = 30000`，
与 §7.3 的记录一致。

---

## 10. Gaussian-JEPA 输入与 Forward/Backward 测试

三组测试，均在 `gaussian_jepa` env、GPU 0、`cfgs/pretrain/gaussian_jepa.yaml` 下运行。

### 10.1 官方加载器读取真实 PLY + 1024 采样 + 前向/反向

`agri/plant_gs/jepa/smoke_real_plant.py` → `results/agri_paa/jepa_smoke.json`

```
status: PASS
source: 292,517 Gaussians (真实 PLY)
input:  (1, 1024, 14)
losses: geo=0.464180  app=0.462002  ground=0.201486
finite_grad_tensors: 215   grad L2 norm: 2.212919
tokens: [1, 64, 384]   embedding: [1, 768]
peak memory: 见 JSON
```

- 采样：uniform，1,024 个高斯，indices 唯一（292,517 > 1,024，无放回）
- 归一化：`pretrain` 模式（对齐 ShapeNetGaussian `pc_norm_gs`）
- 梯度：215 个可训练张量全部有限且非零；7 个必需分支（JEPA_encoder /
  pred_input_proj / predictor / pred_geo_out / pred_app_out / teacher_geo_proj /
  teacher_app_proj）均有非零梯度
- **未做 optimizer step**，权重 SHA256 前后一致
- 无 NaN/Inf

### 10.2 Optimizer + EMA 最小更新（新增工具）

`agri/plant_gs/jepa/optimizer_probe.py` → `results/agri_paa/jepa_optimizer_probe.json`

AdamW（lr 1e-4，wd 0.05）+ EMA momentum 0.996，5 步：

| step | total loss | grad L2 | 参数移动 L2 (all / encoder / teacher) | EMA gap |
|---|---|---|---|---|
| 0 | 1.119544 | 1.986407 | 0.490957 / 0.463678 / 0.001855 | 0.0 → 0.4618 |
| 1 | 1.052750 | 0.805843 | 0.337876 / 0.317293 / 0.002991 | 0.4618 → 0.7448 |
| 2 | 1.015425 | 0.729716 | 0.275164 / 0.254842 / 0.003899 | 0.7448 → 0.9710 |
| 3 | 0.990860 | 0.689000 | 0.236656 / 0.216066 / 0.004657 | 0.9710 → 1.1594 |
| 4 | 0.966948 | 0.664503 | 0.209954 / 0.188820 / 0.005299 | 1.1594 → 1.3196 |

- loss 从 1.1195 降到 0.9669，全部有限
- 每一步**所有参数张量都在动**（all L2 > 0.2）
- teacher 每步都动（EMA 更新生效），且动幅 ≈ (1-momentum) × student 移动：
  0.4618 vs 0.996×… 实测 0.001855 ≈ 0.004 × 0.4637，符合
  `teacher ← momentum·T + (1-momentum)·S`
- EMA gap 早期**本来就应该增大**（T₀ = S₀），我的初版断言写反了，已修正

### 10.3 关于「随机初始化」与「预训练模型」的区分（任务明确要求）

官方预训练 checkpoint 未申请到（ShapeSplatsV1 未获批）。因此：

- §10.1/§10.2 是**随机初始化模型**在**真实 PAA 高斯**上的前向/反向/优化器工程验证。
- 这**不**等于「加载预训练模型提取有意义的农业特征」。
- 两个 JSON 中均已显式写入：`initialization: random`、
  `pretraining_claim: NOT_MADE`、`semantic_effectiveness: NOT_EVALUATED`、
  `scope: engineering check … NOT pretraining, and NOT evidence of a useful
  agricultural representation`。

---

## 11. 测试结果、失败项目与遗留问题

### 11.1 测试（全部实际运行）

| 测试 | 命令 | 结果 |
|---|---|---|
| PAA COLMAP 审计 | `paa/colmap_check.py` | **PASS** |
| 掩膜对齐（102 张全量） | `paa/alignment.py --limit 0` | **PASS** |
| 数据集准备 | `paa/prepare_paa.py` | **PASS**（102 帧，98.4% 点保留） |
| 3DGS 训练 7k | `foreground/train.py --steps 7000` | **PASS** |
| 3DGS 训练 30k | `foreground/train.py --steps 30000` | **PASS** |
| PLY 结构校验 | `validate_ply.py` | **PASS**（`is_valid_3dgs: true`） |
| Gaussian-JEPA 读取器回放 | `verify_gsjepa_compatible.py` | **PASS** |
| Gaussian-JEPA 前向/反向 | `jepa/smoke_real_plant.py` | **PASS** |
| Optimizer + EMA 探针 | `jepa/optimizer_probe.py` | **PASS** |
| 官方 release smoke test | `tools/smoke_test_release.py` | **PASS** |
| 新增 PAA 单元测试 | `pytest paa/tests/` | **26 passed** |
| 既有 JEPA 测试（回归） | `pytest jepa/tests/` | **23 passed** |
| 既有 foreground 测试（回归） | `pytest foreground/tests/` | **16 passed** |

**无回归**：本次未改动任何既有模块（只新增文件），23 + 16 全部保持通过。

### 11.2 开发过程中实际遇到并修掉的问题（不是「假设通过」）

1. **`cv2` 在 `gaussian_jepa` env 拒绝所有 numpy 数组** → 图像处理改用 base env，
   cv2 入参统一 `np.ascontiguousarray`。
2. **`foreground/train.py` 因混合帧尺寸崩溃**（`ValueError: all input arrays must
   have the same shape`）→ 新增 `--frame-policy uniform-canvas` 滚动 + 补齐，并加了
   「相机滚动与图像滚动严格互逆」的单元测试。
3. **PAA 的 `points3D.error` 全为 0**，直接读会得到「重投影误差 0.0」的假象 → 全部
   重投影误差改为实算。
4. **EXIF 方向歧义** → 用投影点梯度能量判定（stored 3.47 vs viewed 1.84）。
5. **`records.index(rec)` 会按值比较 dict，可能静默返回错帧** → 改为按
   `image_id` 建索引。
6. **EMA gap 初版断言写反**（写成「每步应减小」）→ 修正为「应移动且动幅符合
   (1-momentum)×student」。
7. **`registry` 的 `plant_id` 曾把 plant+date 一起写进去**（`Broccoli/20260327-10`），
   这会破坏按植株分组的防泄漏设计 → 改为只写 plant id，并让
   `prepare_paa.py` 与 `registry.parse_sample_id` 用同一函数派生。
8. **`registry` 找不到 `gsjepa_compatibility.json` 就判 `pending`** → 30k 导出覆盖了
   7k 的 `final_7k/` 内容，补跑 compat 检查后才 accepted。
9. **`test_registry.py` 的 numpy (H,W) vs PIL (W,H) 次序混用** → 逐个修正。
10. **`tuple | set` 在 py3.9 不支持** → 改为 `tuple(...) + (None,)`。

### 11.3 遗留问题（未解决，如实记录）

1. **支架/细结构质量不足**（§8）。若 Stage 4B 需要更锐的边缘，可考虑提高
   `--background-alpha-weight`、加 D-SSIM 项、或提高 `--max-size`。
2. **掩膜包含花盆与支架**，不是纯植株（§6.2）。纯植株表型需要额外的植株/容器分离。
3. **官方预训练 checkpoint 仍不可用**，因此 §10 只是工程验证，没有语义有效性证据。
4. **`provenance.json` 是补写的**，`file_counts` 反映最终状态而非下载当时（§4）。
5. **`gaussian_jepa` env 的 cv2 5.0.0 numpy 绑定损坏**（既有问题），后续在该 env 中
   做图像相关工作时需注意。
6. **`scale < 1e-6` 的高斯有 10,071 个（3.4%）**。它们不造成可见伪影（`large>0.25`
   仅 3 个、半径尾部无异常），但下游若对极小尺度敏感需注意。
7. **未做 SH 高阶项与 Gaussian-JEPA 的交互测试**。官方加载器只用 `f_dc_*`
   （14 维），`f_rest_*` 被忽略，本次 PLY 保留 45 个 `f_rest_*` 但未验证任何消费方。

---

## 12. 是否具备进入 Stage 4B 批量 PAA 重建的条件

**具备，但建议先处理 §11.3 的第 2 项（掩膜含花盆）与第 1 项（细结构质量）的取舍决策。**

已就绪的部分：

- `agri/plant_gs/paa/download_paa.py` — 单样本下载 + 来源 SHA 记录
- `agri/plant_gs/paa/colmap_check.py` — COLMAP 审计（失败即中止，不会静默继续）
- `agri/plant_gs/paa/alignment.py` — 掩膜对齐探测与验证
- `agri/plant_gs/paa/prepare_paa.py` — 数据集准备（统一画布 / 去少数派两种策略）
- `agri/plant_gs/paa/gaussian_stats.py` — 高斯统计
- `agri/plant_gs/paa/registry.py` — 批量登记 + 防泄漏分组
- `agri/plant_gs/paa/run_paa_pilot.sh` — 端到端编排（7 步，可参数化）
- `agri/plant_gs/jepa/optimizer_probe.py` — 优化器/EMA 工程验证
- `agri/plant_gs/paa/tests/` — 26 个单元测试，全部通过
- `data/plant_gs/PAA/registry.jsonl` — 登记表（含 `split_group` 字段）

**防泄漏设计已就位**（这是批量阶段最容易出错的地方）：

PAA 的日期型样本形如 `<YYYYMMDD>-<plant_id>`。Broccoli 有 4 个日期 × 2 个 plant id
（`5`、`10`）共 8 个样本——`20260327-10` 与 `20260402-10` 是**同一株植物**相隔 6 天
的两次采集。若按样本目录划分训练/验证/测试，会把同一株植物放到两侧，测出的「泛化」
实际是对该植株轮廓的记忆。因此：

- `plant_id` 与 `acquisition_id` **分开存储**
- `split_group = <crop>/<plant_id>` 是唯一分组键，`Registry.groups()` 直接给出
  plant → acquisitions 的映射
- `Registry.check_split(train_groups, test_groups)` 检测泄漏，有单元测试
- Cotton/Rice/Sunflower 的无日期样本（`1`…`5`）每种只有一个 acquisition，天然无此问题
- 单元测试 `test_split_groups_plant_not_acquisition` 专门锁住这一点

**Stage 4B 建议**：先用 `run_paa_pilot.sh` 跑 2–3 株（含同一 plant 的另一日期，如
`Broccoli/20260402-10`）验证批量路径与分组正确性，再放开到全部 41 个样本。

---

## 附：文件位置索引

```
reports/stage4_paa_pilot.md                       本报告
results/agri_paa/
  colmap_audit.json                               §5
  alignment_probe.json                            §6（102 张全量）
  ply_validation.json                             §9
  gsjepa_compatibility.json                       §9
  gaussian_stats.json                             §9
  jepa_smoke.json / .npz                          §10.1
  jepa_optimizer_probe.json                       §10.2
  gaussian_stats_7k.json / jepa_smoke_7k.json     §7.4 / §10（7k 试运行）

data/plant_gs/PAA/Broccoli_20260327-10/
  colmap_audit.json  mask_alignment.json  mask_hit_rate.json  preparation.json
  dataset/{images,masks,frames.json,points3D_plant.npz,meta.json}
  results/train_7k/    7,000 步试运行（含 history.jsonl, splats_best.pt）
  results/train_30k/   30,000 步正式（含 history.jsonl, splats_best.pt）
  final/               最终资产（point_cloud.ply 等，292,517 高斯）
  final_7k/            7k 试运行导出（313,936 高斯）
  renders/             多视角渲染对比
  logs/                全部运行日志

data/plant_gs/PAA/registry.jsonl                  批量登记表
data/external/PAA-Plant-Dataset/Broccoli/20260327-10/provenance.json

agri/plant_gs/paa/
  download_paa.py  colmap_check.py  alignment.py  prepare_paa.py
  gaussian_stats.py  registry.py  run_paa_pilot.sh  ask_vision.py
  tests/test_alignment.py  tests/test_registry.py
agri/plant_gs/jepa/optimizer_probe.py
```

**复现命令**（从仓库根执行）：

```bash
# 1. 下载单样本（断点续传）
python agri/plant_gs/paa/download_paa.py --crop Broccoli --acquisition 20260327-10

# 2. 只读审计
python agri/plant_gs/paa/colmap_check.py \
  --sample-dir data/external/PAA-Plant-Dataset/Broccoli/20260327-10 \
  --json results/agri_paa/colmap_audit.json

# 3. 掩膜对齐验证（全量）
python agri/plant_gs/paa/alignment.py \
  --sample-dir data/external/PAA-Plant-Dataset/Broccoli/20260327-10 --limit 0

# 4. 端到端（prepare → train → export → validate → JEPA → registry）
bash agri/plant_gs/paa/run_paa_pilot.sh \
  data/external/PAA-Plant-Dataset/Broccoli/20260327-10 \
  data/plant_gs/PAA/Broccoli_20260327-10 30000

# 5. 单元测试
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q agri/plant_gs/paa/tests/
```

Gaussian-JEPA 相关步骤（§10）需在 `gaussian_jepa` env 中运行，见
`run_paa_pilot.sh` 第 6 步的 `$PY` 切换说明与 `~/.claude` memory 中
`gaussian-jepa-smoke-test` 记录的环境构建方式。

---

## 13. 独立复核（同日 16:50–17:10，非原始运行）

本节是**另起一次**、独立于 §1–§12 的核对，目的是把「报告说通过」变成「现在能复现」。
除 GPU 归属外，未修改任何既有产物。

| 复核项 | 命令 | 结果 |
|---|---|---|
| 单元测试（全部 agri 测试） | `pytest paa/tests jepa/tests foreground/tests` | **65 passed**（26 + 23 + 16） |
| 官方 release smoke test | `tools/smoke_test_release.py`（无 checkpoint） | **PASS**，losses geo=0.463957 / app=0.465716 / ground=0.194473 |
| COLMAP 审计复跑 | `paa/colmap_check.py` | **PASS**，102/102 一一对应，重投影 mean 0.774 px |
| JEPA 真实 PLY 前向/反向复跑 | `jepa/smoke_real_plant.py`（GPU 1） | **PASS**，与 `results/agri_paa/jepa_smoke.json` **逐键一致**（仅耗时/设备号不同） |
| JEPA 优化器 + EMA 复跑 | `jepa/optimizer_probe.py`（GPU 1） | **PASS**，与存档 JSON 逐键一致（浮点尾差 < 1e-6） |
| PLY ↔ checkpoint 位级核对 | 见 §9.1 | 全字段 max abs diff **0.0** |
| 图像可解码性 | PIL 全量 `load()` | Camera/RGB/masks/whitebg 各 102/102 可解码，0 失败 |
| 远端来源核对 | HF tree API | Camera 102 张 + 3 bin，文件大小与本地一致 |
| 渲染视觉复核 | Agnes 3.0 Flash（§14 说明） | 重建主体=植株+蓝盆+白支架；多视角背景干净、无可见伪影 |

**未复跑的部分及原因**：3DGS 30k 训练（含 7k 试运行）未重跑。理由是该产物已被
§9.1 的位级核对钉住（PLY 就是 checkpoint 参数），重跑 678 s 只能验证随机性而非
真实性；且 GPU 0 当时被一个与本项目无关的训练任务（PID 655055，占用 6.8 GB、
98% 利用率）占用。复核用 GPU 1。

**复核中发现并处理的问题**：

1. **凭据泄漏风险（已修）**：`agri/plant_gs/paa/ask_vision.py` 曾把 Agnes API key
   作为硬编码默认值写在源码里。该文件在本次要提交的目录中，因此改为**只读环境变量
   `AGNES_API_KEY`**，未设置时直接报错退出；仓库中不再存在任何明文 key。
   （修后仍用环境变量注入实测可用。）
2. **失效引用（已修）**：§4 曾引用一个并不存在的 `provenance.json`，复核阶段按磁盘
   实测重新生成，见 §4 备注。
3. **活动域描述不精确（已修）**：§9 原描述暗示 `f_dc` 就是「官方 SH DC」，复核确认
   它同时受 Gaussian-JEPA 加载期再归一化影响，已改写为精确表述。

**结论**：§1–§12 的结论在复核中全部站得住；发现的 3 个问题都是文档/凭据层面，不影响
重建与接入的技术结论。

---

## 14. 关于视觉复核的说明

§6.2 与 §8 的视觉描述来自 Agnes 3.0 Flash（本机无内置视觉能力，按 `~/.claude/CLAUDE.md`
要求走该 API）。视觉描述**仅作旁证**：它看的是降采样、半透明叠加后的拼图，与逐像素
数值统计冲突时一律以数值为准（§6.2 已记录一次这类冲突）。因此 §8 的定量指标
（PSNR / IoU / background alpha）是结论的基石，视觉描述只用于确认「主体确实是
植株+花盆+支架、背景没有被大面积重建」。

---

## 15. 结论摘要

**做成了什么**：PAA 单株 `Broccoli/20260327-10` 的端到端闭环已跑通并复核——真实
COLMAP 校验 → 真实 3DGS 优化（30k，best@27k）→ 真实 Gaussian PLY（292,517 个，
14 属性齐全、全有限）→ Gaussian-JEPA 官方加载器读取、1024 点采样、真实前向/反向、
优化器 + EMA 最小更新。

**真正通过的项目**：§11.1 全部 13 项测试（含复核复跑）。

**关键指标**：fg PSNR 22.91 dB（留出视角 18.94–26.06），silhouette IoU 0.9866，
background alpha 0.0048，COLMAP 重投影 0.774 px，JEPA `[1,1024,14]` 前向损失有限、
215/215 梯度有限、5 步 AdamW+EMA 参数全动。

**遗留问题**：§11.3 的 7 项，其中影响 Stage 4B 的是「掩膜含花盆/支架」与「细结构
质量」，需要在进入批量前做取舍决策。

**是否进入 Stage 4B**：具备条件，见 §12。
