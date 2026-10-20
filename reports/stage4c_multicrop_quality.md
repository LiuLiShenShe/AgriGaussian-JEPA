# Stage 4C — 跨作物重建验证与植物语义质量基准

**状态：部分完成。四种新作物全部完成独立重建并通过质量门控；六种作物的语义分离全部执行并有可检查结果（其中 Rice 失败并保留）；人工标注任务已准备，零人工确认。**

- 执行日期：2026-10-20
- 分支：`agri-stage4c-multicrop`，基于 `agri-stage4b-fix-repro`（`5e03401`）
- 硬件：2 × NVIDIA RTX A6000 48GB，同时宿主有其他租户（实测期间 load average 34–139）
- 数据：PAA-Plant-Dataset（`JackTrum938/PAA-Plant-Dataset`，Apache-2.0，Hub revision
  `19e7f2f2307462aba37b8571acc7a7c0f72a8608`），新增 Cabbage/Tobacco/Rice/Sunflower 各 1 组
- 环境：3DGS = base env（torch 2.9.1+cu128 / gsplat 1.6.0）；
  Gaussian-JEPA = `envs/gaussian_jepa`（torch 2.0.1，`knn_cuda` / `pointnet2_ops` 已编译）

本报告的所有数值来自本轮实际运行。凡属推断、未测量或不可验证者，均显式标注。
**本阶段不声称完成 JEPA 农业语义预训练。**

---

## 0. 结论速览

| 问题 | 结论 |
|---|---|
| 四种新作物重建是否成功 | **是，4/4 通过门控并注册为 accepted** |
| 是否全部走 Stage 4B-Fix Runner（锁 + 两阶段 + 溯源） | **是**，8 个 registry 行的 provenance 均为 `verified` |
| 六种作物是否都有可检查的语义分离结果 | **是，7 个有效 carve + 1 个失败（Rice 全空），全部保留** |
| 语义分离是否"稳定" | **不稳定**。72.6 万→3.9 万高斯的植物占比跨度 0.129–0.628，两后端 plant IoU 中位 0.29–0.68 |
| 是否获得人工分割准确率 | **否**。48 个任务已就绪，`tasks_human_confirmed = 0`，工具拒绝给出任何准确率 |
| 是否可以用于预训练 | **full-specimen 8 个资产 `train_eligible`；plant-only 0 个**（无人工真值支撑） |
| 是否允许全量扩展 PAA 全部 41 组 | **有条件允许**：只扩 full-specimen 侧，且必须先解决 Rice 类失败与白花/棉铃误判 |
| TomatoMAP | **无法评估**。镜像端点能读出公开数据集但不返回 TomatoMAP；官方端点本机无路由。属访问结论，非数据质量结论 |

---

## 1. 开始前状态检查

`main` = `062dcbe` 不含 Stage 4B-Fix，因此从 `agri-stage4b-fix-repro`
（HEAD `5e03401`）建立分支 `agri-stage4c-multicrop`：

```bash
git checkout -b agri-stage4c-multicrop          # from agri-stage4b-fix-repro
git log --oneline -1                            # 5e03401
```

已核对的既有文件：`reports/stage4b_fix_repro.md`、
`agri/plant_gs/foreground/{runlock,provenance,verify_provenance}.py`、
`agri/plant_gs/paa/{run_paa_pilot,stages}.sh`，以及
`agri/plant_gs/paa/{semantic,merge_labels,plant_only,plant_reference_panel,
plant_dataset,sampling_analysis,identity_manifest,registry}.py`。

**未删除或修改任何旧资产**：污染目录 `data/plant_gs/PAA/Cotton_1/`
仍为只读，本轮通过库构建将其标记为 `sealed=true`（见 §6.2），
它的事故地位不变，但被显式排除在资格判定之外。

---

## 2. Task 1：跨作物数据检查与重建

### 2.1 样本选择（先查真实清单，不猜路径）

用 Hub tree API 枚举 `JackTrum938/PAA-Plant-Dataset`（revision `19e7f2f2…`）
全部 6 类作物 41 组采集。每组的 `Camera/images`、`RGB_seg/masks`、`RGB_seg/whitebg`
计数都确认存在后才下载：

| 作物 | 采集组 | 本次选样 | 选择理由 |
|---|---:|---|---|
| Cabbage | 8 | `20260328-9` | 102/102 帧完整 |
| Tobacco | 8 | `20260401-23` | 102/102 帧完整 |
| Rice | 5 | `1` | 102/102 帧完整（Rice/4 仅 99 帧、Rice/5 仅 100 帧） |
| Sunflower | 7 | `1` | 102/102 帧完整 |

下载只用 `Camera/**` + `RGB_seg/masks/**` + `RGB_seg/whitebg/**`
（`download_paa.py` 默认），每例约 2–4 分钟，
并在每个样本目录写 `provenance.json`（Hub SHA、文件计数、下载时间）。

### 2.2 逐样本结果（均为本轮实测）

| 资产 | COLMAP 相机 | 重投影 mean/p95 (px) | 注册/用帧 | 掩膜命中中位 | 7k 门控 | 30k 步 | 留出 PSNR (dB) | IoU | bg α | 高斯数 |
|---|---|---|---:|---:|---|---:|---:|---:|---:|---:|
| Cabbage/20260328-9 | FULL_OPENCV ×2 | 0.835 / 2.358 | 102 / 102 | 0.9729 | PASS (21.311) | 27,000 | **27.233** | 0.9838 | 0.01032 | 299,839 |
| Tobacco/20260401-23 | FULL_OPENCV ×2 | 0.747 / 2.031 | 102 / 102 | 0.9902 | PASS (27.845) | 27,000 | **29.138** | 0.9937 | 0.00319 | 305,998 |
| Rice/1 | FULL_OPENCV ×2 | 0.673 / 1.925 | 102 / 102 | 0.9765 | PASS (25.077) | 29,999 | **25.077** | 0.9825 | 0.00465 | 292,746 |
| Sunflower/1 | SIMPLE_PINHOLE ×2 | 0.610 / — | 102 / 102 | 0.9526 | PASS (21.311) | 27,000 | **22.601** | 0.9704 | 0.01181 | 304,134 |

四个新样本全部 7k 门控一次通过，**没有发生为了让训练继续而放宽阈值**。
留出视角指标均为**单场景内**指标，不是独立植株泛化。

### 2.3 溯源链与 JEPA 接入（四例均相同流程）

```
results/train_30k/splats_best.pt
   SHA256 + 内部 step 从文件读出
        │  export（前后哈希一致，否则拒绝）
        ▼
final/point_cloud.ply        → provenance_verification.json : pass
        │  verify_provenance.py（独立重算）
        ▼
jepa_smoke.json / jepa_optimizer_probe.json : PASS / PASS
        │  gsplat 侧校验
        ▼
registry upsert（accepted 要求 provenance == verified）
```

实测：四例 `provenance_verification.status` 全为 `pass`，`library.entries[].gaussian_ply_path`
的 SHA256 与 provenance 记录**逐例重新哈希一致**（8/8，`governance.ply_hash_recheck`）。
JEPA 侧四例均为 `gsjepa_compatible: true`，1024 高斯前向/反向与 EMA 探针全 PASS，
初始化 `random`，`semantic_effectiveness = NOT_EVALUATED`。

### 2.4 Camera 覆盖（Stage 4C 前置门限）

库内现有 2 种 COLMAP 相机模型：

| 模型 | 样本 | 是否需要整流 |
|---|---|---|
| SIMPLE_PINHOLE | Broccoli×2、Cotton/1、Sunflower/1、Cotton/1@sealed | 否 |
| FULL_OPENCV | Broccoli/20260402-10、Cabbage、Rice、Tobacco | **是**（已按相机整流图像与掩膜） |

Stage 4B-Fix 提出的"覆盖 ≥ 3 种相机模型"门限**未达成**（现有 2 种）。
PAA 只含这两种模型，因此这不是本阶段能补的，需在 §7 的扩展决策里说明。

### 2.5 运行资源（单样本，含语义）

| 阶段 | 单样本耗时（实测） |
|---|---|
| 下载 | 2–4 min（镜像，8 workers） |
| COLMAP 审计 | ~30 s |
| 配准 + 准备 | 75–112 s |
| 3DGS 7k | 245–327 s |
| 3DGS 30k | 520–1,120 s |
| 导出 + 溯源 + PLY 校验 + JEPA | ~3 min |
| 语义标注（102 帧 × 2 后端） | 17–20 min |
| 三维投票 + 纯植物导出 | ~2 min |
| 细结构与残差评测 | ~2.5 min |
| **合计** | **约 50–65 min / 样本**（宿主 load average 34–139，非独占卡） |

---

## 3. Task 2：跨作物植物语义分离

### 3.1 流程

每个作物一个代表性采集，跑 `agri/plant_gs/paa/stage4c/run_semantics.sh`：
ADE20K SegFormer + SAM/CLIP 两后端逐帧标注 → 保守合并（plant 取交集）→
12 视角人工参考面板 → 可见性加权三维语义投票 → 从 `final/point_cloud.ply`
（而非 checkpoint）carve → 新增的 carve 来源验证 → 细结构质量评测。

**Full-specimen PLY 全部保留**，未修改、未删除。

### 3.2 结果总表（同一套参数：min_plant_fraction 0.5 / min_visible 3 / depth_tol 0.1）

| 资产 | 源高斯 | 判为植物 | 纯植物保留 | 无证据 | carve 来源验证 | 两后端 plant IoU 中位 |
|---|---:|---:|---:|---:|---|---:|
| Broccoli/20260327-10 | 292,517 | 85,099 | **83,810** (0.287) | 2,154 | pass | 0.534 |
| Broccoli/20260327-5 | 302,152 | 117,997 | **117,026** (0.387) | 1,957 | pass | 0.620 |
| Broccoli/20260402-10 | 307,159 | 98,600 | **97,469** (0.317) | 2,345 | pass | 0.469 |
| Cabbage/20260328-9 | 299,839 | 39,492 | **38,737** (0.129) | 756 | pass | 0.379 |
| Cotton/1 | 319,969 | 133,569 | **132,685** (0.415) | 973 | pass | 0.675 |
| Rice/1 | 292,746 | **1** | **0** (0.000) | 6 | pass（空集） | **0.000** |
| Sunflower/1 | 304,134 | 190,971 | **189,648** (0.624) | 1,632 | pass | 0.685 |
| Tobacco/20260401-23 | 305,998 | 62,306 | **62,306** (0.204) | 210 | pass | 0.294 |

"判为植物"是多数投票类，"纯植物保留"再叠加 plant_fraction≥0.5 与 opacity≥0.01，
所以两个数字会差一点。二者差得小说明阈值不是主要损失来源，**语义标签本身**才是。

### 3.3 Rice/1 —— 失败案例（按要求保留，不遮掩）

Rice 的 carve 保留了 **0 个高斯**。原因可从产物里定位：

- ADE20K SegFormer 在 Rice 帧内的前几大类是 `wall`（1,220,507 像素）、`pot`、`armchair`、`floor`，
  `plant` 只有 8,402 像素——它基本没把稻穗识别为植物；
- SAM+CLIP 反过来把 53.8% 的标本区域标为 `support`、18.5% 标为 `soil`；
- 两后端 plant IoU 中位 **0.0**，合并后 plant 交集只剩 410,496 像素（占两者并集 28.6%）。

交集规则在这里**放大**了两个后端的共同盲区：稻穗细密、垂直、形态近似支撑结构，
两套零样本/域外模型都失手，取交集后几乎没有像素存活。

工具侧还暴露了两个真实缺陷，都已修：

1. `plant_only.py` 在 carve 为空时调用 gsplat 导出器，抛
   `RuntimeError: cannot reshape tensor of 0 elements` 并 **SIGFPE 崩溃**，
   连带丢失整份报告。现在空 carve 写成 0 顶点 PLY 并记录原因，报告落盘。
2. 首次运行 `run_semantics.sh` 还撞上 `merge_labels.py` 的
   `NameError: name 'per_frame' is not defined`（在每个 PNG 都写完之后才抛），
   导致"labels_merged 看起来完整但没有 summary"。修的是那个未定义名，
   并补了一条回归测试。

### 3.4 细结构、白组织、残差（对每个 held-out 视角）

细结构的定义是**形态学的**，不是器官标注——按 keeping 前的说明，这里没有器官真值。
`thin` = plant 掩膜经 7×7 盒式开运算后被移除的像素（细茎、叶柄、叶缘、叶脉）；
`thick` = 开运算后仍存活的像素（叶盘、花冠）。

| 资产 | thick 覆盖 | thin 覆盖 | **细结构落差** | 白组织占标本 | 白标本被标 support | 非植物门上不透明残留 |
|---|---:|---:|---:|---:|---:|---:|
| Broccoli/20260327-10 | 0.909 | 0.489 | **0.420** | 0.951 | 0.298 | 0.0028 |
| Broccoli/20260327-5 | 0.922 | 0.673 | **0.248** | 0.910 | 0.315 | 0.0024 |
| Broccoli/20260402-10 | 0.921 | 0.652 | **0.269** | 0.859 | 0.331 | 0.0015 |
| Cabbage/20260328-9 | 0.710 | 0.570 | **0.140** | 0.846 | 0.279 | 0.0002 |
| Cotton/1 | 0.994 | 0.894 | **0.100** | 0.946 | **0.540** | 0.0016 |
| Sunflower/1 | 0.954 | 0.729 | **0.225** | 0.644 | 0.232 | 0.0275 |
| Tobacco/20260401-23 | 0.979 | 0.717 | **0.261** | 0.881 | 0.104 | 0.0003 |

三条如实解读：

1. **Broccoli/20260327-10 的细结构落点最大（0.420）。** 这不是同一个 Broccoli 之间
   的随机波动：该样本的 segmentation 一致性也最低（0.534），说明标签分歧直接变成了
   边缘高斯被投成非植物。
2. **Cotton 的白组织风险确认且严重**：标本内 94.6% 的像素是"亮且不饱和"，
   其中 **54.0% 被候选标签标为 support**。Stage 4B 已知白色棉花会让零样本模型失效，
   本轮有数字了：棉铃与白花正在被当支架删掉。这是本阶段最重要的语义风险。
3. **Sunflower 的非植物残留最高（0.0275）**，其余在 0.0002–0.0028 量级。
   Sunflower 同时是保留比例最高的（0.624），两者一致——最宽松的样本残差也最大。

### 3.5 渲染质量代价（对候选标签的植物区域评）

| 资产 | PSNR full-in-plant → plant-only-in-plant | IoU full → plant-only | 植物覆盖率 | 非植物区 α 残留 |
|---|---|---:|---:|---:|
| Broccoli/20260327-10 | 21.580 → 17.828 | 0.102 → 0.468 | 0.903 | 0.058 |
| Broccoli/20260327-5 | 25.191 → 20.498 | — | 0.919 | 0.079 |
| Broccoli/20260402-10 | 25.253 → 17.651 | — | 0.919 | 0.123 |
| Cabbage/20260328-9 | 27.460 → 12.206 | — | 0.707 | 0.053 |
| Cotton/1 | 23.469 → 21.436 | — | 0.991 | 0.140 |
| Rice/1 | 无（空 carve） | — | — | — |
| Sunflower/1 | 24.264 → 19.673 | — | 0.950 | 0.280 |
| Tobacco/20260401-23 | 28.444 → 24.355 | — | 0.976 | 0.033 |

**Carve 是有代价的，且代价是实的**：植物区域内 PSNR 掉 3.8–15.3 dB。
Cabbage 掉得最多（15.3 dB）而 plant IoU 一致性最低（0.379），两者是同一枚硬币。

### 3.6 交集不等价于更准

按要求报告：**交集压低误报的同时必然损失真植物召回**。可量化的两条：

- 合并不变量：`plant_both` 只占两者植物并集的 **42%–63%**（随作物变化），
  即交集直接丢掉 37%–58% 的植物像素，其中大部分是标签分歧而非真背景。
- 在 3D 侧的表现是：Cabbage 判为植物的比例只有 0.129，Cotton 是 0.415。
  同一套阈值下 3.2 倍的跨度，主要不是植物学差异，而是标签一致性差异
  （两后端 IoU 0.379 vs 0.675）。

---

## 4. Task 3：人工标注基准（成功状态 = 工具与待标注集就绪）

### 4.1 生成了什么

| 产物 | 位置 | 数量 |
|---|---|---|
| 任务清单 | `results/agri_paa/stage4c/annotation/manifest.json` | **48 个任务**，6 作物 × 8 视角 |
| 待标注图像 | `annotation/images/<task>.jpg` | 48 |
| 模型预标注 | `annotation/preannotations/<task>/{labels_segformer,labels_sam_clip,labels_merged}.png` | 3 × 48 |
| 人工标签目录 | `annotation/labels/` | **0 个文件**（无人工输入） |
| 确认记录 | `annotation/confirmations/` | 0 |
| 标注说明 | `annotation/ANNOTATION_GUIDE.md` | 1 |
| 审阅界面 | `annotation/review.html` + `review_server.py` | 1（已实测 HTTP 200） |
| 评价结果 | `annotation/evaluation.json` | `status: annotation_pending` |

### 4.2 类别定义（与 semantic.py 的类 id 独立）

`0 Background | 1 Plant | 2 Pot/Soil | 3 Support | 4 Uncertain`
——刻意不复用语义后端的整数 id（它有独立 pot/soil、没有 Uncertain），
避免把 pot/soil 混淆误判为正确。

### 4.3 视角选择：可复现的"覆盖率"而非手工挑

`select_views` 把每帧变成特征向量（方位角 sin/cos、俯仰、植物/非植物覆盖、
两后端分歧、标本面积），再做 farthest-point 采样，并在 manifest 记录每视角
`selected_because`（它在哪些轴上处于 5%/95% 分位）。态度是描述性的：
"这个视角是分歧最大的"，不是声称"该轴已覆盖"。

### 4.4 三条让"伪标签冒充人工"不可能的结构

1. `labels/` **只由 `annotation.py import` 写入**，import 要求 annotator 姓名、
   校验类 id 合法、写 SHA256 确认记录。
2. 手工塞进 `labels/` 的 PNG 没有确认记录 → `load_confirmed` 记为
   `unconfirmed_ignored`，**不计入任何指标**。
3. 确认后标签被改动 → SHA256 不匹配 → 忽略。

### 4.5 评测口径

`Plant IoU / Precision / Recall / False Plant Rate`（False Plant Rate =
真非植物像素中被误判为植物的比例），分作物、分后端报告；
**Uncertain 像素从计算中排除，单独报告比例**。

### 4.6 当前状态（如实）

```
status              = annotation_pending
tasks_total         = 48
tasks_human_confirmed = 0
per_backend         = {}
```

**没有任何分割准确率被计算或声称。** 伪标签如果只和自己比对，得出的 1.0 不是
测量结果；本报告不呈现这种数字。工具侧已用测试固化这一点
（`test_unconfirmed_label_png_is_ignored_not_scored`）。

复现命令（给标注者）：

```bash
python -m agri.plant_gs.paa.stage4c.annotation serve \
    --dir results/agri_paa/stage4c/annotation --port 8931
# 打开 http://127.0.0.1:8931/review.html ，填名字、逐任务涂改、Save
python -m agri.plant_gs.paa.stage4c.annotation import --dir <dir> --task <id> \
    --labels <png> --annotator <name>
python -m agri.plant_gs.paa.stage4c.annotation evaluate --dir <dir>
```

---

## 5. Task 4：Gaussian-JEPA 输入质量对比

### 5.1 三条硬约束

1. **采样器不重写**：直接 import `agri.plant_gs.jepa.dataset.sample_gaussians`
   ——冒烟测试用的那个均匀采样器。
2. **同一预算**：三个臂都是 `RandomState(seed).choice(..., 1024)`，
   population 只有"全资产 / 植物子集"之别。归一化统计量**始终取全资产**，
   否则会同时改变坐标框架，比较就混了两件事。
3. **同 seed 配对 + 8 个 seed 重复**：报告均值与方差（`seeds: 0..7`，
   各臂 `distinct_indices_across_seeds = 8` 证明 seed 真的生效）。

植物感知采样器的 pool = `classes == PLANT`（**排除 -1 无证据高斯**，
不把"没有证据"当"植物的证据"）。

### 5.2 结果（8 seed 均值）

| 资产 | 总体植物占比 | uniform 植物占比 | aware | uniform 体素覆盖 | aware | plant-only 资产 |
|---|---:|---:|---:|---:|---:|---:|
| Broccoli/20260327-10 | 0.291 | 0.285 | 1.00 | 0.563 | 0.665 | 0.660 |
| Broccoli/20260327-5 | 0.391 | 0.388 | 1.00 | 0.583 | 0.670 | 0.675 |
| Broccoli/20260402-10 | 0.321 | 0.312 | 1.00 | 0.537 | 0.652 | 0.653 |
| Cabbage/20260328-9 | 0.129 | 0.126 | 1.00 | 0.634 | 0.719 | 0.707 |
| Cotton/1 | 0.417 | 0.416 | 1.00 | 0.622 | 0.747 | 0.740 |
| Rice/1 | 0.000 | 0.000 | 1.00 | 1.000 | 1.000 | 无（空资产） |
| Sunflower/1 | 0.628 | 0.625 | 1.00 | 0.562 | 0.583 | 0.594 |
| Tobacco/20260401-23 | 0.204 | 0.209 | 1.00 | 0.632 | 0.800 | 0.789 |

细结构保留比（draw 中最小十分位植物高斯的占比 ÷ 该十分位在植物总体中的占比，
1.0 = 无偏）：

| 资产 | uniform | aware | plant-only 资产 |
|---|---:|---:|---:|
| Broccoli/20260327-10 | **0.839** | 1.022 | 1.030 |
| Broccoli/20260327-5 | 0.960 | 1.040 | 1.020 |
| Broccoli/20260402-10 | 0.971 | **0.958** | 0.989 |
| Cabbage/20260328-9 | 0.977 | 1.044 | 0.958 |
| Cotton/1 | 0.989 | 1.016 | 1.002 |
| Rice/1 | 0.000 | 1.000 | — |
| Sunflower/1 | 1.035 | 1.005 | 0.991 |
| Tobacco/20260401-23 | **0.881** | 1.036 | 1.022 |

Tobacco 单 seed 示例（可复算）：uniform 样里 29.3% 是 background、40.7% 是 pot、
21.4% 才是 plant；体素覆盖 0.620，其中 **452 个体素落在植物之外**。
植物感知 sample 无体外素、覆盖 0.807。

### 5.3 三条结论

1. **uniform 采样器把大量预算花在非植物上**：最低的 Cabbage 只有 12.6% 是植物，
   Tobacco 也只有约 21%。对"农业植物表征"这个目标，原版采样器在 PAA 上是
   **天然浪费**，且浪费比例随作物变化 5 倍。
2. **植物感知采样用同一预算把植物体素覆盖从 0.56–0.63 提到 0.58–0.80**，
   细结构保留比从 0.84–1.04 变成 0.96–1.04（largely 更接近 1.0）。
   代价是它依赖**语义标签**——而标签是本阶段最不可靠的一环（Rice 全错、
   Cotton 白组织误判），所以这个提升还不能兑现。
3. **Broccoli/20260402-10 是唯一 aware 不如 uniform 的细结构样本（0.958 vs 0.971）**。
   它的两后端一致性也最低（0.469）。这说明植物感知采样的提升**不是免费的**，
   与标签质量耦合。

### 5.4 本模块**没有**做的事

没有比较 JEPA loss，没有比较表征质量。随机初始化网络上单次前向的 loss
差异不能说明哪种输入更好。原始均匀采样保留为基线，JEPA 模型结构与损失未改。

---

## 6. Task 5：数据治理与预训练准备

### 6.1 统一资产登记

`results/agri_paa/stage4c/assets.json`（9 行，含封存目录）+ `governance.json`。
每条记录 40+ 字段，带 `evidence` 指向来源文件。字段覆盖 brief 要求的全部项：
数据集/作物/植株 ID/采集 ID/`source_revision`(Hub SHA)/相机模型/COLMAP 质量/
重建质量/checkpoint SHA256/Gaussian PLY SHA256/导出来源验证状态/
Plant-only 可用性/语义标注质量及真值来源/是否可预训练/不通过原因。

### 6.2 四个状态的判定结果（`governance.counts`）

| 状态 | 数量 | 说明 |
|---|---:|---|
| `reconstruction_verified` | **8 / 9** | 唯一例外是封存的 Cotton/1 事故目录 |
| `semantic_verified` | **0** | 没有任何人工确认标注（§4.6） |
| `plant_only_verified` | **0** | carve 存在但真值来源是模型，不构成证据 |
| `train_eligible` | **full-specimen 8；plant-only 0** | 见 §6.4 |

封存目录的处理：`data/plant_gs/PAA/Cotton_1/` 与干净复现工作区共享
acquisition id `Cotton/1`。库构建把前者改名为 `Cotton/1@sealed`
（原始字节未动），guard 新增 `sealed` 判定：**目录不可写即封存证据**，
`asset_sealed` 是不可被其他证据覆盖的首要拒绝理由。

### 6.3 身份与 hash 审计（`governance.audit`）

- `clean: true`，`error_count: 0`，`warning_count: 0`
- `split_clean: true`，`split_leakage: []`
- 检查项：重复 asset_id、PLY 哈希冲突、checkpoint 哈希复用、
  缺失 provenance、provenance 缺哈希/缺 step、缺失源 revision、
  缺失 plant_id、split_group 与 crop/plant 不一致、一 split_group 对应多 plant。
- 局限（如 brief 要求，如实说明）：**本审计证明的是库的内部一致性**，
  不能证明某个 plant_id 真的对应那株物理植物——那是数据采集方的责任。

### 6.4 划分（`governance.split`）与它的脆弱性

7 个可用 split group（Broccoli/10、Broccoli/5、Cabbage/9、Cotton/1、Rice/1、
Sunflower/1、Tobacco/23），按 group 分：

```
train: Broccoli/10  Broccoli/5  Rice/1  Sunflower/1      (5 assets)
val  : Cabbage/9                                       (1 asset)
test : Cotton/1     Tobacco/23                          (2 assets)
```

`leakage: true`（无同一 group 跨划分），`unassigned_assets: []`。
**但必须写明**：7 个植株做三划分不足以支撑泛化结论——`split.caveat` 也这么记。
val 只有 1 株 1 个采集，任何"val 提升了 X"都只是这一株的读数。
要谈未见植株，应该做 leave-one-crop-out 并报告逐作物分布，而不是单个数。

### 6.5 明确不宣称的事

- 不宣称任何分割准确率（无人工真值）。
- 不宣称任何表征质量（全部 JEPA 数字来自随机初始化模型）。
- 不宣称任何植株泛化（重建指标都是单场景留出视角）。
- 不因"JEPA 能加载"就把资产标为适合正式预训练。

---

## 7. Task 6：下一阶段的数据规模决策

### 7.1 四种新作物的重建是否成功？

**成功。** 4/4 独立完成 9 步流水线（COLMAP 审计 → 配准 → 7k 门控 → 30k →
PLY 导出 → 溯源 → JEPA 前后向 → 统计 → 注册），门控全部一次通过，
provenance 全部 `verified`，PLY 哈希重算 8/8 一致。

### 7.2 六种作物的 Plant-only 分离是否稳定？

**不稳定。** 三个证据：

- 植物占比 0.129–0.628（4.9 倍跨度），不能只用植物学差异解释；
- 两后端 plant IoU 0.294–0.685，Rice 为 0.0；
- 代价同样不稳定：植物区域 PSNR 掉 3.8–15.3 dB。

### 7.3 哪些作物存在严重细结构损失？

按形态学 thin 覆盖排序，**Broccoli/20260327-10 最差（0.489，落差 0.420）**，
其次 Broccoli/20260402-10（0.652）与 Broccoli/20260327-5（0.673）。
Cabbage 的绝对覆盖最低（0.570）但 thick 也低（0.710），落差 0.140 反而是第二小。
Cotton 最好（thin 0.894）。

### 7.4 Full-specimen 还是 Plant-only？是否仍需并行保留？

**建议：两种都保留并行，但只有 full-specimen 现在有资格进入预训练。**

理由：
- full-specimen 有 8 个 `reconstruction_verified` 资产，来源链完整；
- plant-only 的 `train_eligible = 0`，因为它的真值来源是模型候选标签。
  把它当预训练输入，等于把 Stage 4B 的"加载成功 ≠ 可训"问题换成
  "候选标签 ≠ 植物语义"的问题再犯一次。
- 而 §5 已量化 uniform 采样在 PAA 上的浪费，说明**未来一定要有某种植物感知的输入**。
  所以全标本资产现在入库，同时保留纯植物资产作为"等人工真值"的候选。

### 7.5 现有可靠资产有多少组采集、多少株独立植物？

- **8 组采集 / 7 株独立植物 / 6 种作物**（均 `reconstruction_verified`）。
- PAA 全库是 **41 组采集 / 23 株独立植物 / 6 种作物**（Hub 实测，
  `identity_manifest.json`）。
- 所以本轮覆盖了 41 组中的 8 组（19.5%）、23 株中的 7 株（30.4%）。

### 7.6 是否可以扩展至 PAA 全部 41 组？

**有条件允许，且条件明确。**

可扩的部分：**full-specimen 侧**。理由是这条链已经跑通 8 次、无人干预、
每次都有可复算来源链；每例约 50–65 分钟，双卡并行铺开 33 组约 10–15 小时。

**不**在满足这两条之前铺开：
1. Rice 类失败必须先定性——它到底是"标签后端不适配该作物形态"还是
   "本数据集里 Rice 的 acquisition 有问题"。33 组里 Rice 还有 4 组，
   若未定性就铺开，会以同样方式失败 4 次。
2. 需为每类作物保留 12 视角参考面板并做人工抽检。Cabbage 的 plant IoU 一致性
   0.379 说明它不是偶发。

若铺开：按作物逐个进行（先 Broccoli 剩 5 组 / Cotton 4 组 / Tobacco 7 组 /
Sunflower 6 组，最后 Rice 与 Cabbage 各 4 组），每个作物一旦出现
空 carve 或一致性 < 0.35 即停止并上报。

### 7.7 是否需要优先引入 TomatoMAP？

**不能回答，因为拿不到。** 详见 §8。

---

## 8. TomatoMAP 预检

`tomatomap_preflight.py` 只发 API 元数据请求（HEAD/GET 元信息，**不下数据**）：

| 端点 | 已知公开数据集（对照） | TomatoMAP/TomatoMAP |
|---|---|---|
| `https://huggingface.co` | 本机不可达 | 不可达 |
| `https://hf-mirror.com` | **200** | **401 "Invalid username or password"** |

口径校准：镜像端点对**不存在的** dataset id 也返回同一个 401。
所以 401 说明"gated / private / id 不对"，三者无法区分；
它**不**能作为"该数据集存在且需要申请"的证据。

结论已写入 `results/agri_paa/stage4c/tomatomap_preflight.json`：

> 访问层面：拿不到。数据层面（相机内外参、旋转台姿态、单个采集单元
> 可重建性）：**未评估**。本轮未下载任何番茄数据，未启动任何番茄重建。

要推进需要：HF token / 已批准的访问申请 / 或准确的 repo id。

对 Task 6.7 的回答：**本阶段无法判断是否需要优先引入 TomatoMAP**。
以 PAA 侧证据能说的是——即便拿到 TomatoMAP，`plant-only` 侧的缺口
（无人工真值）不会因为它而自动补齐；而 full-specimen 侧的 7 株/6 作物
才是更直接限制泛化实验的瓶颈。

---

## 9. 测试

| 套件 | 数量 | 结果 |
|---|---:|---|
| `agri/plant_gs/foreground/tests` | 74 | passed |
| `agri/plant_gs/paa/tests` | 106 | passed |
| `agri/plant_gs/jepa/tests` | 23 | passed |
| **合计** | **203** | **203 passed / 0 failed** |

`agri/plant_gs/paa/tests/test_stage4c.py`（新增 34 项）覆盖：

- guard：加载成功不构成证据、门限/缺失指标都拦截、伪标签不验证语义、
  空 carve 需要语义验证、封存目录不可被证据推翻、hash 重算检测改动；
- 库审计：哈希冲突、缺失 revision、split 泄漏、未知/重复 group；
- annotation：导入拒绝非法类 id、未确认 PNG 被忽略且不出指标、
  Uncertain 排除与报告、确认流程产出真实指标；
- plant-only provenance：真子集通过、**行数相同但数值不同必须失败**；
- input_quality：预算一致、undecided 不被当植物、细结构以植物为主体；
- `merge_labels` 回归：PNG 写完后 NameError 那个缺陷；
- `plant_only` 空 carve：写出 0 顶点 PLY 而非崩溃。

回归命令：

```bash
/home/test/biosoft/enter/bin/python -m pytest \
  agri/plant_gs/foreground/tests agri/plant_gs/paa/tests agri/plant_gs/jepa/tests -q
```

---

## 10. 产物位置

| 内容 | 路径 |
|---|---|
| 主报告 | `reports/stage4c_multicrop_quality.md` |
| 机器可读结果 | `results/agri_paa/stage4c/` |
| 资产库 | `results/agri_paa/stage4c/assets.json` |
| 治理与划分 | `results/agri_paa/stage4c/governance.json` |
| 相机汇总 | `results/agri_paa/stage4c/camera_summary.json` |
| 全库身份清单 | `results/agri_paa/stage4c/identity_manifest.json` |
| 输入质量对比 | `results/agri_paa/stage4c/input_quality/input_quality.json` |
| 标注任务与评测 | `results/agri_paa/stage4c/annotation/` |
| TomatoMAP 预检 | `results/agri_paa/stage4c/tomatomap_preflight.json` |
| 新工作区 | `data/plant_gs/PAA/{Cabbage,Tobacco,Rice,Sunflower}_*/` |
| 逻辑资产前的实际文件在 gitignored `data/` 下，未提交 |

新增/修改代码：

| 文件 | 作用 |
|---|---|
| `paa/stage4c/guard.py` | 四状态判定、库审计、划分、加载闸门、hash 重算 |
| `paa/stage4c/assets.py` | 资产装配 + 治理 + 划分 + PLY 哈希重算 |
| `paa/stage4c/annotation.py` | 标注任务导出/导入/评测/审阅界面 |
| `paa/stage4c/input_quality.py` | Full / aware / plant-only 三臂输入质量 |
| `paa/stage4c/plant_only_provenance.py` | carve → 源行 的可验证链 |
| `paa/stage4c/semantic_quality.py` | 细结构/白组织/残差 分区评测 |
| `paa/stage4c/camera_summary.py` | 跨作物相机表 |
| `paa/stage4c/tomatomap_preflight.py` | TomatoMAP 访问预检 |
| `paa/stage4c/run_semantics.sh` | 单样本语义流水线 |
| `paa/stage4c/run_stage4c.sh` | 总入口 |
| `paa/tests/test_stage4c.py` | 34 项新测试 |
| `paa/plant_only.py` (改) | carve 空集不再崩溃 + 记录 source_ply 哈希 |
| `paa/merge_labels.py` (改) | 修 `per_frame` NameError |

**未提交**：原始图像、PLY、checkpoint、GPU 缓存、大文件（`data/` 全目录 gitignored）。

```bash
git push -u origin agri-stage4c-multicrop
```

---

## 11. 本轮的真实边界

1. **所有重建指标都是单场景留出视角**，不是植株泛化。
2. **没有任何人工分割真值**，所有语义数字都只是"忠实于候选标签"。
3. **所有 JEPA 数字来自随机初始化模型**，官方预训练 checkpoint 仍不可用。
4. **学位划分只有 7 株**，泛化结论需 leave-one-crop-out + 逐作物分布。
5. **宿主非独占**（load 34–139），墙钟时间不做性能对比。
6. **相机模型覆盖 2 种**，Stage 4B-Fix 提出的 ≥3 种门限未达成——PAA 本身只有 2 种。
7. **Rice/1 的语义失败原因只定位到标签层**（两后端共同盲区），未做根因级验证。
