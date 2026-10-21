# Stage 4D — 语义可靠性修复、人工评价与 TomatoMAP 单样本预检

**状态：工程修复与预检完成；人工真值为 0；TomatoMAP 48 视角恢复成功，但位姿条件未成立，因此未重建。**

- 执行日期：2026-10-21
- 分支：`agri-stage4d-semantic-tomato`，基于包含 `a00d650` 的 `9f2a6ee`（Stage 4C 之后的环境迁移提交）
- 基准：Stage 4C `reports/stage4c_multicrop_quality.md`、Stage 4B-Fix `reports/stage4b_fix_repro.md`
- 本阶段**没有**启动正式 JEPA 预训练，**没有**批量下载 TomatoMAP，**没有**修改任何已完成的 Full-specimen Gaussian 资产或旧版 Plant-only PLY

所有数值来自本轮实际运行。凡属推断、未测量或不可执行者，均显式标注。

---

## 0. 结论速览

| # | 问题 | 结论 |
|---:|---|---|
| 1 | 资产资格判断的两个问题修复了吗 | **是，两个都修了并有集成测试**。Plant-only 现在区分 5 种状态；人工确认标签能经 `semantic_gate` 影响资产资格 |
| 2 | 有多少张真正人工确认的标签 | **0 张**。48 个任务、12 个优先任务就绪，但没有人类标注者；所有作物仍 `annotation_pending` |
| 3 | Rice 是否恢复了非空且合理的植物分离 | **否**。没有人工真值，按规则不重新 carve；诊断显示两个后端在 102 帧里 54/102 帧几乎找不到植物 |
| 4 | Cotton 白组织误判是否有独立人工证据证明改善 | **否**。没有人工证据；候选标签仍把 62.3% 的白色标本像素标为 support |
| 5 | TomatoMAP 正确数据源是否可访问 | **是**。`Voxel51/tomato-map` 公开、未 gated、CC BY 4.0；官方端点不可达，镜像可达；e!DAL DOI 429（站点维护） |
| 6 | 是否恢复出完整 48 视角 | **是**。plant 1 / 2023-08-16 的 48/48 张图像全部下载并解码，4 相机 × 12 转台位 |
| 7 | 是否实际完成番茄单样本 3DGS 重建 | **否**。位姿条件不成立：无外参，且 `pi2` 标定不是可用的针孔模型；背景是静态温室，直接 SfM 会重建背景 |
| 8 | PAA 是否具备批量扩充条件 | **否，条件性**。33 个剩余采集已列出；被 Rice 根因、人工语义检查、Cabbage 一致性三项阻塞 |
| 9 | Stage 5 可以开展哪些有意义的 JEPA 对比 | 随机编码器 / Gaussian-JEPA / Gaussian-MAE 的**库内排名**与**重采样稳定性**；不可声称未见植株泛化或语义有效性 |
| 10 | Git 分支、commit 与结果路径 | 分支 `agri-stage4d-semantic-tomato`；结果在 `results/agri_paa/stage4d/`；commit 见 §9 |

---

## 1. Task 0：审计与保护

### 1.1 分支与基线

```bash
git checkout -b agri-stage4d-semantic-tomato   # from 9f2a6ee, which contains a00d650
git log --oneline -2
# 9f2a6ee build(env): 用 uv 取代 conda，环境可无 root、无 conda 重建
# a00d650 feat(stage4c): 跨作物重建、语义质量基准与数据治理
```

### 1.2 保护措施

| 对象 | 处理 |
|---|---|
| `data/plant_gs/PAA/Cotton_1/`（事故封存目录） | 未触碰；仍为只读；库构建继续标记 `sealed=true` 并改名为 `Cotton/1@sealed` |
| 旧 checkpoint / Full-specimen PLY | 未修改。8 个 PLY 的 SHA256 重新计算，8/8 与 provenance 记录一致 |
| 旧 Plant-only PLY | 未覆盖。本轮**没有**重新执行 carve |
| `results/agri_paa/stage4c/` | 保留原样；Stage 4D 只新增 `stage4d/`，并更新标注 manifest 的 priority 字段 |
| 训练锁、两阶段 Runner、provenance 机制 | 未改动；新增模块只读这些产物 |

### 1.3 审计发现的一个环境事实

`src/models/pi{1-4}.npz`（TomatoMAP 标定）**不在** Hugging Face 数据集仓库里（4 个文件均 404），而在代码仓库 `0YJ/TomatoMAP` 的 `src/models/` 下（4 个文件均 200）。数据集卡片的描述会让读者以为它在数据集内。详见 §5.4。

---

## 2. Task 1：资产治理接口修复

### 2.1 问题 A —— Plant-only 数据结构不一致

**修复前**：`build_entry()` 写

```python
entry["plant_only"] = {"plant_only": {...}, "plant_only_segformer": {...}}
```

而 `plant_only_checks()` 读 `entry["plant_only"]["available"]`。键从未相遇，于是**所有**有真实 carve 的资产都被报成 `plant_only_missing`。Stage 4C 的 `governance.json` 里 9/9 资产都是 `"no plant-only asset"`，包括 7 个 provenance 已验证的 carve 和 Rice 的空 carve。

**修复**：`assets.summarise_plant_only()` 成为唯一契约，产出 `state` 与原始 `candidates`；`guard.plant_only_checks()` 只读 `state`，并保留对两种历史形状的兼容。五种状态：

| state | 含义 | 本轮数量 |
|---|---|---:|
| `absent` | 没有候选目录 | 1（Cotton/1@sealed） |
| `failed_empty` | carve 失败，或保留 0 个高斯 | 1（Rice/1） |
| `candidate_source_unverified` | carve 存在，但 provenance 未通过 | 0 |
| `source_verified_semantics_unverified` | carve 可溯源，但标签没有人工证据 | **7** |
| `verified` | carve 可溯源且标签通过人工门控 | 0 |

**修复后的实际判定**（`results/agri_paa/stage4d/governance.json`）：

- `plant_only_states.counts = {absent: 1, failed_empty: 1, candidate_source_unverified: 0, source_verified_semantics_unverified: 7, verified: 0}`
- `plant_only_pretrain` 拒绝理由从“9 × no plant-only asset”变为：
  - 7 个资产：`semantic separation has no human-confirmed evidence`
  - Rice/1：`the plant-only carve failed or retained no Gaussians`
  - Cotton/1@sealed：`no plant-only asset`（封存目录确实没有）

这满足 brief 的要求：**存在且来源合法的候选，不再被报告为 `plant_only_missing`**。

### 2.2 问题 B —— 人工标注评价与资格判定脱节

**修复前**：`read_semantics()` 固定返回

```python
"benchmark": {"status": "annotation_pending", "truth_source": "model_candidate"}
```

人工确认的标签即使存在，也没有路径进入 `governance.json`。

**修复**：新增 `agri/plant_gs/paa/stage4c/semantic_gate.py`，形成完整链路：

```
labels/<task>.png + confirmations/<task>.json (SHA256)
        │  annotation.load_confirmed()
        ▼
逐任务打分：Plant IoU / Precision / Recall / False Plant Rate
        + 细结构召回 / 白色组织召回（区域由“人工标签本身”定义）
        │
        ▼
逐资产门控（只对 carve 实际使用的 labels_merged 评分）
        │
        ▼
逐作物汇总（描述性，certifies_assets = false）
        │
        ▼
assets.read_semantics() → guard.semantic_checks() → semantic_verified
```

**门控参数**（`semantic_gate.py`）：

| 项 | 值 |
|---|---:|
| 最少确认视角 / 资产 | 6 |
| 最低平均 Plant IoU | 0.70 |
| 最低平均 Plant Recall | 0.80 |
| 最高 False Plant Rate | 0.10 |
| 单视角 IoU 下限 | 0.50 |
| 达到下限的视角比例 | ≥ 80% |

**三条防止过度认证的规则**：

1. **按资产，不按作物。** Broccoli 有 3 个资产，只有 1 个采集有标注任务；另外两个保持 `annotation_pending`。作物汇总显式写 `certifies_assets: false`。
2. **单张图不能认证。** 1 个确认视角 → `insufficient_coverage`；6 个确认视角但有一个视角 IoU 0.2 → 失败（`worst view IoU < 0.50`）。
3. **只有 carve 真正使用的标签参与认证。** `labels_merged` 是验证后端；SegFormer 与 SAM+CLIP 只用于诊断。

**集成测试**（`agri/plant_gs/paa/tests/test_stage4d.py`，16 项）实际覆盖了状态翻转：

- `test_one_confirmed_view_never_certifies_an_asset` —— 1 张确认 → `insufficient_coverage`
- `test_enough_confirmed_views_certify_the_asset` —— 6 张 → `verified`
- `test_a_crop_average_certifies_no_sibling_asset` —— Broccoli/5 通过，Broccoli/10 仍 pending
- `test_a_failing_label_set_stays_unverified` —— 6 张确认但候选全背景 → `failed`
- `test_rebuilding_the_library_flips_states_after_confirmation` —— 重建库后 `semantic_verified` 与 `plant_only_verified` 同时为真
- `test_empty_carve_stays_blocked_even_with_confirmed_semantics` —— 即使语义通过，0 高斯 carve 仍被 `plant_only_failed_empty` 阻塞

**真实运行结果**：由于没有人工标签，`results/agri_paa/stage4d/semantic_gate.json` 为

```json
{"tasks_total": 48, "tasks_human_confirmed": 0,
 "assets_semantic_verified": [],
 "per_crop": {"Broccoli": "annotation_pending", "Cabbage": "annotation_pending",
              "Cotton": "annotation_pending", "Rice": "annotation_pending",
              "Sunflower": "annotation_pending", "Tobacco": "annotation_pending"}}
```

这是正确输出，不是失败：工具能把人工标签接进来，但没有人类，就不该有认证。

---

## 3. Task 2：人工评价集与标注辅助

### 3.1 现有 48 个任务的一致性检查

命令：`python -m agri.plant_gs.paa.stage4c.annotation check --dir results/agri_paa/stage4c/annotation`

| 项 | 数量 | 说明 |
|---|---:|---|
| manifest 任务 | 48 | 6 作物 × 8 视角 |
| 图像 | 73 | **25 个历史孤儿**（来自一次 16 视角导出） |
| 预标注目录 | 73 | 同上 |
| 人工标签 PNG | 0 | — |
| 确认记录 | 0 | — |
| 缺图像/预标注的任务 | 0 | 所有任务输入齐全 |
| `consistent` | `true` | 无阻塞问题 |

孤儿文件被列出而不是删除：UI 与 `evaluate` 都只读 `manifest.json`，孤儿不会进入任何指标；删除已跟踪文件反而会让“任务集被重新导出”这件事不可见。检查结果写入 `annotation/check.json`。

### 3.2 12 张问题驱动优先任务

`prioritise` 用可复现的评分选视角，不是手工挑：

```
score = 3.0 × plant_absence          # 候选几乎找不到植物（Rice 型全漏）
      + 2.0 × white_support_fraction # 亮且不饱和的植物组织被标 support（Cotton 型）
      + 2.0 × disagreement_fraction  # 两后端分歧最大
      + 1.0 × thin_fraction          # 植物标签里细结构占比
      + 0.3 × nonplant_fraction
```

每作物取前 2，共 12 张；48 张全部保留。

| 排名 | 任务 | 得分 | 植物占标本 | 白组织被判 support | 两后端分歧 | 主要理由 |
|---:|---|---:|---:|---:|---:|---|
| 1 | `Rice_1_40` | 5.36 | 0.002 | 0.84 | 0.004 | 几乎找不到植物；白/亮组织被判 support |
| 2 | `Rice_1_52` | 5.24 | 0.005 | 0.98 | 0.002 | 同上 |
| 3 | `Cotton_1_87` | 5.13 | 0.014 | 0.98 | 0.075 | 几乎找不到植物；白组织被判 support |
| 4 | `Tobacco_20260401-23_16` | 3.45 | 0.018 | 0.16 | 0.022 | 几乎找不到植物 |
| 5 | `Tobacco_20260401-23_20` | 3.21 | 0.007 | 0.00 | 0.043 | 几乎找不到植物；分歧最大 |
| 6 | `Cabbage_20260328-9_97` | 2.39 | 0.073 | 0.98 | 0.180 | 白组织被判 support；分歧最大 |
| 7 | `Cabbage_20260328-9_69` | 2.37 | 0.108 | 0.99 | 0.162 | 同上 |
| 8 | `Sunflower_1_10` | 2.23 | 0.085 | 0.97 | 0.124 | 白组织被判 support；分歧 |
| 9 | `Sunflower_1_13` | 2.15 | 0.102 | 0.96 | 0.094 | 同上 |
| 10 | `Cotton_1_82` | 1.96 | 0.032 | 0.89 | 0.039 | 白组织；细结构占比高 |
| 11 | `Broccoli_20260327-10_100` | 1.92 | 0.148 | 0.90 | 0.051 | 白组织被判 support；分歧 |
| 12 | `Broccoli_20260327-10_79` | 1.92 | 0.028 | 0.85 | 0.069 | 同上 |

这 12 张覆盖了 Stage 4C 的三个失败模式：Rice 的全漏、Cotton/Cabbage 的白组织误判、Broccoli 的细结构/分歧。写入 `annotation/priority_tasks.json`，并回填到 `manifest.json` 的 `priority`、`priority_rank`、`hardness` 字段。

### 3.3 标注工具改进

| 改进 | 实现 |
|---|---|
| 原图 / 候选标签 / 已确认标签对比 | 三层 canvas：原图、`labels_merged` 候选叠加（可开关、可调透明度）、人工笔刷层；已有 `labels/<task>.png` 自动载入继续编辑 |
| 快速修细茎、叶柄、白色组织 | 类按钮 + 键盘 `1/2/3/4/0`；`Plant` 一键切换，配合 3–61 px 笔刷；候选叠加用来定位“白组织被判 support”的区域，但**不写入**标签 |
| Uncertain 类 | `4` 类；评测时从 IoU/Precision/Recall 中排除并单独报告比例 |
| 保存标注者、时间、SHA256 | `/api/save` 先写 `labels/<task>.png`，再写 `saves/<task>.json`（annotator、`saved_utc`、`label_sha256`、`status: saved_not_confirmed`）；`import` 才是确认，确认记录包含 `label_sha256` 与 `save_record_matches` |
| 按作物/视角查看进度 | `annotation status --dir ...` 输出每作物 `views_total/confirmed/pending`、优先任务确认数；UI 侧栏按作物分组并显示 `P` 优先标记 |
| 查看/检查 | `annotation check` 对账 manifest 与磁盘；`annotation prioritise` 可重算优先级 |

**保存 ≠ 确认** 的结构没有改变：`labels/` 里的 PNG 若没有确认记录，评测记为 `unconfirmed_ignored`，不进入任何指标。

### 3.4 人工确认的真实状态

```
tasks_total            = 48
tasks_human_confirmed  = 0
tasks_unconfirmed_ignored = 0
annotators             = []
```

**本轮没有获得任何真实人工确认。** 因此：

- 没有计算 Plant IoU / Precision / Recall / False Plant Rate
- 没有计算细结构/白色组织召回
- 没有把 Agent、分割模型或视觉模型的判断当作人类真值

这些指标的计算路径已在 `semantic_gate.score_task` 中实现并有测试；一旦有人确认，`annotation evaluate` 与 `semantic_gate` 会给出分作物、分后端、分资产的结果。

---

## 4. Task 3：Rice 与 Cotton（及 Cabbage）的语义失败诊断

### 4.1 五个候选臂的实际可用性

| 臂 | 状态 | 说明 |
|---|---|---|
| A. SegFormer | **已测量** | `nvidia/segformer-b5-finetuned-ade-640-640`，本地权重可用 |
| B. SAM + CLIP | **已测量** | 复用 Stage 4C 已生成的 102 帧标签；权重不在本地也可直接读标签 |
| C. 两后端交集 | **已测量** | `labels_merged` |
| D. SAM 2 / Grounded SAM 2 | **未运行** | `import sam2` 失败；本地无 SAM 2/Grounded SAM 权重；Stage 4C 引用的 `/data/models/image/sam-vit/...` 已不存在。无任何准确性暗示 |
| E. 人工提示 + 多视角几何约束 | **未运行** | 没有人类放提示；用模型伪造提示正是本项目禁止的替换。轨道稳定性作为**代理**单独报告，不冒充 E |

### 4.2 测量口径（没有人工真值，因此不报准确率）

`diagnose_semantics.py` 对 102 帧逐帧测：

- 植物占标本掩膜的比例（及其跨轨道的变异系数 CV）
- 近空帧数（植物 < 1% 标本）
- 两后端 Plant IoU、交集占并集比例
- 白色组织（亮且不饱和）中未被判为植物的比例、被判为 support 的比例
- 自身植物掩膜中细结构（7×7 开运算移除）的比例

这些是**候选方法差异与稳定性**，不是精度。

### 4.3 结果（102 帧实测）

| 资产 | A 植物占标本 | B 植物占标本 | C 交集 | 近空帧 A/B/C | A–B IoU 均值 | 白组织未被判植物 A/B/C | B 把白组织判 support |
|---|---:|---:|---:|---|---:|---:|---:|
| Rice/1 | **0.045** | **0.019** | **0.014** | 1 / **54** / **55** | **0.296** | 87.2% / 91.6% / 93.7% | **65.4%** |
| Cotton/1 | 0.468 | 0.282 | 0.277 | 0 / 0 / 0 | 0.626 | 83.9% / 94.1% / 94.6% | **62.3%** |
| Cabbage/20260328-9 | 0.443 | 0.172 | 0.171 | 0 / 2 / 2 | **0.382** | 93.1% / 98.1% / 98.1% | **64.6%** |

轨道稳定性（植物占标本的 CV，越低越稳定）：Rice A 0.735 / B 1.077 / C 1.127；Cotton 0.333 / 0.304 / 0.310；Cabbage 0.395 / 0.697 / 0.701。

### 4.4 三条如实解读

1. **Rice 不是“阈值太严”，是标签全错。** A 平均只把 4.5% 的标本判为植物，B 是 1.9%，C 是 1.4%；B/C 在 54–55 帧里几乎为空。两后端 IoU 0.296，交集只占并集 29.6%——交集规则在这里放大共同盲区，与 Stage 4C 的判断一致。**没有人工标签，不能重新 carve；即使 carve 也会是第二个候选，不是“恢复”。**
2. **Cotton 的白组织问题在候选层面依旧严重。** 白色标本像素中，B 把 62.3% 判为 support，C 把 94.6% 判为非植物。这是“白色棉铃被当支架删掉”的直接量化。**但这仍不是独立人工证据**；只能说候选标签没有改善。
3. **Cabbage 的瓶颈是 B，不是 A。** A 保留 44.3%，B 只有 17.2%，C 被 B 拉低到 17.1%；A–B IoU 0.382。Stage 4C 观察到的“Cabbage plant-only 渲染损失大”，在这个诊断里对应的是交集把 A 的大量植物像素删掉。

### 4.5 本轮**没有**做的事

- 没有用任何候选 2D 标签重新执行三维投票或 Plant-only 导出。
- 没有比较 JEPA loss 或表征质量。
- 没有宣称任何分割准确率提升。
- 没有把颜色阈值当成生产方法；诊断里的白/细结构测量只用于定位失败，不用于删支架。

---

## 5. Task 4：TomatoMAP 正确数据源预检与单样本恢复

### 5.1 访问矩阵（实测）

| 源 | HTTP | 分类 | 结论 |
|---|---:|---|---|
| `https://huggingface.co/api/datasets/Voxel51/tomato-map` | 无 | `unreachable` | 本机无路由（Errno 101） |
| `https://hf-mirror.com/api/datasets/Voxel51/tomato-map` | **200** | `present` | **公开、未 gated、CC BY 4.0** |
| `https://doi.org/10.5447/ipk/2025/14` | **429** | `rate_limited` | 跳转后返回 IPK “Wartungsarbeiten”（维护）页面 |
| `https://edal.ipk-gatersleben.de/` | 200 | `present` | 仓库根可达，DOI 落地页被限流 |
| `https://github.com/0YJ/TomatoMAP` | 200 | `present` | 代码仓库可达 |
| `https://doi.org/10.1038/s41597-026-06926-9` | 200 | `present` | 论文 DOI 可达 |

关键区分：

- **官方 HF 端点不可达**是网络路由问题，不是数据集不存在。
- **镜像 200 且 `gated=false`** 证明不需要申请或凭证。
- **e!DAL DOI 429** 是仓库维护/限流，不是数据集被撤下。
- 数据集真实元数据：`sha=c978c56cfc6d1bd35d0b34709f9b14cfc85b165f`，`lastModified=2026-07-10`，`usedStorage=48,942,497,361` B（45.6 GiB），`downloads=7150`，68,075 个文件，68,069 个 FiftyOne 样本。

### 5.2 真实字段（不是猜仓库 ID）

`metadata.json`（26 KB，已下载）与 `samples.json`（218 MB，已下载）实测：

| 字段 | 值 |
|---|---|
| `plant_id`, `piid`, `pose_id`, `pose_degrees` | 每样本字段，pose 1–12，0–330°，30° 步长 |
| `camera_label`, `camera_angle_deg`, `camera_position_desc` | 4 相机，45/90/135/180° |
| `group.name` | `pos_1`..`pos_4`（相机高差） |
| `image_set_id` | 同一转台位的 4 个同步曝光 |
| `capture_datetime` | 采集时间 |
| `bbch_stage`, `classification`, `ground_truth` | 生长阶段、7 类 ROI 检测框 |
| `camera_intrinsics`, `static_transforms` | **两个都是空对象 `{}`** |

规模：64,464 张 rig 图像 + 3,605 张微距图 = 68,069；`(plant_id, date)` 共 **1,343 个完整采集单元，每个恰好 48 张**，覆盖 101 株。

**FiftyOne 分组 ≠ COLMAP 位姿**：

- `group` 只表示相机高差，`image_set_id` 只表示“同一转台位的 4 个同步曝光”。
- 没有每张图的外参矩阵，也没有 `camera_intrinsics` / `static_transforms`。
- 微距花果分割集是另一套相机、另一时间窗，**没有 plant_id/pose/session 字段**，不能按 plant id 与 rig 图像合并。`metadata.json` 的 `seg_source` 字段明确写了这一点。

### 5.3 48 视角恢复（真实下载）

选择 plant 1 / 2023-08-16，按 `image_set_id`、`pose_id`、`camera_label`、`capture_datetime` 从 `samples.json` 恢复：

| 检查 | 结果 |
|---|---|
| 图像数 | **48 / 48** |
| 相机 × 位 | 4 × 12，无缺格、无重复 |
| 解码 | 48/48 可解码 |
| 尺寸 | 1080 × 1440，全部一致 |
| 唯一 SHA256 | 48 / 48 |
| 总大小 | 26,179,410 B（26 MB） |
| 采集时间戳 | 4 个（每相机一个，相差 ≤ 34 s，同一采集单元） |
| 许可证 | CC BY 4.0 |

图像内容抽检（Agnes 3.0 Flash 识图，记录在 `tomatomap_unit/vision_background_check.json`）：植株直立、盆沿在画面底部、茎旁有浅色支架与黑色夹子、背景是**温室玻璃与结构梁**，不是纯白幕布。

### 5.4 标定文件：位置与可用性

| 路径 | HTTP | 结论 |
|---|---:|---|
| `Voxel51/tomato-map` 的 `data/calibration/pi{1-4}.npz` | **404** | 不在 HF 数据集仓库 |
| `0YJ/TomatoMAP` 的 `src/models/pi{1-4}.npz` | **200** | 在代码仓库，4 个文件均 656 B |

下载后实测：

| 相机 | fx | fy | cx | cy | 判定 |
|---|---:|---:|---:|---:|---|
| pi1 | 1293.2 | 1298.4 | 785.8 | 513.1 | 对一个 1440×1080 画幅合理 |
| pi2 | **0.1456** | **0.0792** | 645.7 | 789.2 | **不可用的针孔模型** |
| pi3 | 1303.6 | 1303.6 | 714.5 | 527.2 | 对 1440×1080 合理 |
| pi4 | 1294.3 | 1296.5 | 775.3 | 567.7 | 对 1440×1080 合理 |

关键不一致：**发布图像是 1080×1440 纵向，而 pi1/pi3/pi4 的主点位于 1440×1080 横向画幅的中心附近**（cx≈0.55×1440，cy≈0.48×1080）。图像没有 EXIF Orientation（48/48 均为 None），所以不能靠 EXIF 自动纠正。要么需要把图像旋转 90° 后再用这组内参（未验证的假设），要么标定与发布图像的画幅对应关系本身有问题。**本轮没有猜一个缩放/旋转因子硬套。**

### 5.5 位姿条件与重建决定

| 条件 | 状态 |
|---|---|
| 48 视角完整 | ✅ |
| 内参存在 | ✅（代码仓库） |
| 内参可用 | ❌ pi2 不可用；pi1/3/4 与发布画幅不一致 |
| 外参 / COLMAP 位姿 | ❌ 数据卡明确“no extrinsic rig calibration” |
| 可用的静止背景掩膜 | ❌ rig 图像不带掩膜；背景是温室结构 |
| 位姿条件成立 | **❌** |

**因此没有执行 3DGS 重建。** 原因按层记录：

1. 没有外参；四台相机的相对几何未知。
2. `pi2` 不是可用的针孔模型，`pi1/3/4` 与发布图像画幅不一致。
3. SfM 会锁定静态温室背景——这正是 brief 要求排除的干扰，而 rig 图像没有植物掩膜可直接用于遮挡。

这不是“数据不可重建”，而是**位姿/标定信息不足**。48 张图像本身完整、可解码、有唯一哈希。下一步的具体获取路径：

- 向作者索取每张图的外参，或确认标定对应的画幅/旋转约定；
- 或者：为 48 张图生成植物/支架掩膜后做受约束 SfM（相机约束在已知转台轴上），用重投影误差与轮廓 IoU 验证位姿，再重建。

### 5.6 本轮**没有**下载/提交的东西

- 没有下载 45.6 GiB 全量数据。
- 没有下载 68,069 个样本的媒体文件；只下载了 `metadata.json`（26 KB）、`fiftyone.yml`、`README.md`、`samples.json`（218 MB 元数据）和 **1 个采集单元的 48 张图（26 MB）**，以及 4 个标定 npz（2.6 KB）。
- 48 张图与 npz 放在 gitignored 的 `data/external/TomatoMAP/plant1_20230816/`，不提交；提交的是 SHA256 清单与报告。

---

## 6. Task 5：PAA 批量扩充决策

`results/agri_paa/stage4d/paa_batch_plan.json`，由 `identity_manifest.json` + Stage 4D `assets.json` + 实测磁盘/耗时生成。

| 项 | 值 |
|---|---:|
| PAA 全库采集 / 独立植株 / 作物 | 41 / 23 / 6 |
| 当前可信资产 / 独立植株 | **8 / 7** |
| 剩余采集 | **33** |
| 其中：新独立植株 | **22** |
| 其中：已知植株的额外日期 | **11** |
| 单工作区磁盘（实测均值 / 最大） | 2.50 GiB / 3.23 GiB |
| 33 组预估存储 | **82.5 GiB** |
| 单卡预估 / 双卡预估 | **30.9 h / 15.5 h** |

耗时口径：8 个现有工作区的 `7k+30k` 训练时间逐工作区求和后取均值（1,694.7 s），加准备（86.3 s），再加 Stage 4C §2.5 实测的下载/语义/导出/carve 时间（1,590 s），得单单位约 3,371 s；33 组 = 111,247 s ≈ **30.9 单卡小时**，双卡 ≈ 15.5 h。宿主 load 30–50，非独占，墙钟不作性能结论。

**决策：`conditional`，不授权批量扩充。** 三个阻塞项：

| 阻塞 | 现状 | 满足条件 |
|---|---|---|
| `rice_root_cause` | 未定因；剩余集合里还有 4 组 Rice | 一个在轨道上稳定的稻穗候选标签，并在 Rice 优先视角上验证 |
| `human_semantic_check` | 0 张人工确认 | 12 张优先任务确认，再跑语义门控 |
| `cabbage_consistency` | A–B IoU 0.382，B 把 65% 白组织判 support | 确认 Cabbage 优先视角后重新评分 |

扩充顺序已写成清单：**先新独立植株（22），再额外日期（11）**；每个资产仍必须通过独立的语义门控，不能因为 PLY 可加载就跳过。

---

## 7. Task 6：Stage 5 评价协议

`results/agri_paa/stage4d/stage5_protocol.json` + `stage5_configs/`。没有修改 Gaussian-JEPA 的模型结构或损失。

### 7.1 按独立植株的划分

```
train: Broccoli/10  Broccoli/5  Rice/1  Sunflower/1          (5 assets)
val  : Cabbage/9                                            (1 asset)
test : Cotton/1     Tobacco/23                              (2 assets)
```

与 `governance.json` 的划分逐组一致（同 seed 20261020）。

- `leakage: false`，没有 split_group 跨划分。
- Broccoli/10 有两个日期（2026-03-27、2026-04-02），同属一个 group；按日期划分会泄漏，按 group 划分不会。
- 7 株不足以支撑泛化结论；协议文件把这句话与数字写在一起。

### 7.2 跨日期同株泄漏检查

`plants_with_multiple_acquisitions = {"Broccoli/10": ["2026-03-27", "2026-04-02"]}`。检查的是“同一植株的所有采集是否在同一侧”，不是“同一作物是否在不同侧”。后者允许，并且正是 leave-one-crop-out 与随机划分的区别。

### 7.3 重复高斯采样稳定性

同一资产、同一预算（1,024）、8 个 seed，相对 16³ 体素格（每资产自己的包围盒归一化）：

| 资产 | 参考体素数 | 体素覆盖 均值±std | 最小–最大 |
|---|---:|---:|---:|
| Broccoli/20260327-10 | 1,806 | 0.294 ± 0.008 | 0.284–0.307 |
| Broccoli/20260327-5 | 1,572 | 0.326 ± 0.007 | 0.318–0.337 |
| Broccoli/20260402-10 | 1,556 | 0.333 ± 0.006 | 0.326–0.341 |
| Cabbage/20260328-9 | 1,221 | 0.374 ± 0.007 | 0.360–0.385 |
| Cotton/1 | 791 | 0.486 ± 0.012 | 0.463–0.506 |
| Rice/1 | 1,075 | 0.349 ± 0.006 | 0.341–0.359 |
| Sunflower/1 | 962 | 0.459 ± 0.010 | 0.440–0.473 |
| Tobacco/20260401-23 | 1,569 | 0.345 ± 0.005 | 0.337–0.353 |

用途：**任何小于 seed 间波动的 JEPA 差异都不是差异。** 这是 Stage 5 比较的噪声地板。

### 7.4 部分遮挡表征稳定性

`gaussian_classes.npz` 的 `seen`（该高斯在多少视角可见）按四分位分层。示例：

- **Rice/1**：四个可见性层植物占比全为 **0.00**——与“全漏”一致。
- **Cotton/1**：低可见性层植物 0.46，高可见性层 0.22——低可见性层反而更多被判植物，这与白组织判 support 的方向一致，值得在 Stage 5 单独报告。
- **Broccoli/20260327-5**：最低层植物 0.00，最高层 0.95——可见性分层非常强。

协议要求：低可见性层与高可见性层分别报告；只在多数视角可见的高斯上有效的模型，没有学到部分遮挡。

### 7.5 Leave-One-Crop-Out

6 个 fold，每折留出一个作物。示例 Rice fold：训练 6 株 / 测试 1 株。协议明确：报告逐折分布，不报告单一均值。

### 7.6 三个配置模板

| 文件 | 作用 |
|---|---|
| `stage5_configs/gaussian_jepa.yaml` | 官方 `GaussianJEPA` 模型块**逐字复制**：`num_targets [11,9,7,5]`、`group_size 32`、`num_group 64`、`lambda_ground 0.1`、`num_swd_projections 256`、`trans_dim 384`、`depth 12` |
| `stage5_configs/random_encoder.yaml` | 同一编码器，冻结、不更新；线性探针；是“学到东西”的地板 |
| `stage5_configs/gaussian_mae.yaml` | 同编码器，属性空间 mask-and-reconstruct，无 EMA target、无 JEPA predictor；隔离目标函数 |

三者的 `evaluation.claims_forbidden` 都写明：不得声称未见植株泛化、不得声称分割准确率、不得用训练 loss 下降推断语义有效性。正式预训练可以从随机初始化开始；本阶段只准备数据与评价条件。

---

## 8. 测试

新增 `agri/plant_gs/paa/tests/test_stage4d.py`（16 项），覆盖：

- Plant-only 五种状态、历史字典形状不再被报 `plant_only_missing`、空 carve → `failed_empty`
- 1 张确认不认证；6 张确认才认证；作物汇总不认证兄弟资产；候选全背景 → `failed`
- 重建资产库后 `semantic_verified` / `plant_only_verified` 正确翻转
- 空 carve 即使语义通过仍被阻塞
- 优先任务选择命中白组织与全漏；`check` 把孤儿列为警告而非错误；`status` 按作物统计

回归运行（真实输出，按套件选择能提供其依赖的解释器）：

```bash
envs/agri-gs/bin/python   -m pytest agri/plant_gs/paa/tests        -q   # 134 passed
envs/agri-gs/bin/python   -m pytest agri/plant_gs/foreground/tests -q   #  74 passed
envs/agri-jepa/bin/python -m pytest agri/plant_gs/jepa/tests       -q   #  23 passed
```

```
agri/plant_gs/paa/tests         134 passed, 0 failed
agri/plant_gs/foreground/tests   74 passed, 0 failed
agri/plant_gs/jepa/tests         23 passed, 0 failed
------------------------------------------------
合计                            231 passed, 0 failed
```

**一个必须如实报告的环境事实**：uv 迁移后，没有任何一个 venv 同时具备 `gsplat` 与 `easydict`。
`agri-gs` 有 gsplat，但 `jepa/tests` 在它里面 collect 失败（`No module named 'easydict'`）；
`agri-jepa` 有 easydict，但没有 gsplat，`paa/tests` 里 10 项、`foreground/tests` 4 项无法导入。
Stage 4C 报告的“203 项在一个解释器里全过”在当前双 venv 布局下**不可复现**，除非把两个依赖装进同一个 env。
本轮没有为了凑数字去改环境，而是按套件使用正确解释器，并顺手修了测试里一个真实的迁移缺陷：
`test_runner_stages.py` 把 `PY_GS` 硬编码为 `python`，而 uv venv 不提供 `python` 这个名字，
现在改为 `sys.executable`（3 处），修改后 28 项相关测试全过。

---

## 9. 产物与提交

| 内容 | 路径 |
|---|---|
| 主报告 | `reports/stage4d_semantics_tomatomap.md` |
| 资产库（修复后） | `results/agri_paa/stage4d/assets.json` |
| 治理与状态计数 | `results/agri_paa/stage4d/governance.json` |
| 语义门控 | `results/agri_paa/stage4d/semantic_gate.json` |
| 标注一致性/优先级 | `results/agri_paa/stage4c/annotation/{check.json,priority_tasks.json,manifest.json}` |
| Rice/Cotton/Cabbage 诊断 | `results/agri_paa/stage4d/diagnosis/*.json` |
| TomatoMAP 预检 | `results/agri_paa/stage4d/tomatomap_preflight.json` |
| TomatoMAP 单元验证 | `results/agri_paa/stage4d/tomatomap_unit/{unit_report.json,image_sha256.json,vision_background_check.json}` |
| PAA 批量计划 | `results/agri_paa/stage4d/paa_batch_plan.json` |
| Stage 5 协议与配置 | `results/agri_paa/stage4d/stage5_protocol.json`、`stage5_configs/*.yaml` |

新增/修改代码：

| 文件 | 作用 |
|---|---|
| `paa/stage4c/assets.py` | Plant-only 契约 `summarise_plant_only`；`read_semantics` 读取门控报告；schema 升为 `paa-stage4d-1` |
| `paa/stage4c/guard.py` | 五种 Plant-only 状态；`semantic_checks` 读取 `verified`；兼容历史形状 |
| `paa/stage4c/semantic_gate.py` | 人工标签 → 指标 → 逐资产门控 → 逐作物汇总 |
| `paa/stage4c/annotation.py` | 优先级选择、`check`/`status`/`prioritise`、保存记录（SHA/时间）、新审阅 UI |
| `paa/stage4c/diagnose_semantics.py` | A/B/C 候选差异与轨道稳定性；D/E 未运行的明确记录 |
| `paa/stage4c/tomatomap_preflight.py` | 正确 repo、分类不可达/不存在/受限/下载失败、标定位置 |
| `paa/stage4d/tomatomap_unit.py` | 48 视角恢复验证、标定可用性、位姿条件判定 |
| `paa/stage4d/protocols.py` | Stage 5 划分/泄漏/采样稳定性/遮挡分层/配置模板 |
| `paa/stage4d/batch_plan.py` | 33 组批量清单、存储/算力、阻塞条件 |
| `paa/tests/test_stage4d.py` | 16 项集成测试 |

**不提交**：PLY、checkpoint、原始图像、TomatoMAP 48 张图、`samples.json`、私密凭据。`data/` 全部 gitignored。

Git：

```bash
git push -u origin agri-stage4d-semantic-tomato
# commit: see the commit on this branch; the stage adds only code, small JSON reports,
# the annotation manifest update, and the new report.
```

---

## 10. 真实边界

1. **人工真值仍是 0。** 没有 Plant IoU/Precision/Recall，没有细结构或白组织的真实召回；任何“分割更准”的说法都未发生。
2. **Rice 没有被“修好”。** 诊断定位到标签层（A/B 共同盲区），但没有人工证据，按规则不重新 carve。
3. **Cotton 白组织没有独立人工证据。** 候选标签仍把 62.3% 的白色标本像素判为 support。
4. **TomatoMAP 没有重建。** 位姿条件不成立；这是信息不足，不是数据不可重构。48 视角、解码、哈希、许可证均已验证。
5. **e!DAL 的 DOI 拿到的是 429/维护页，不是数据集内容。** 未据此推断数据缺失。
6. **PAA 批量扩充是 `conditional`，不是 `go`。** 33 组、82.5 GiB、双卡 15.5 h 是计划，不是授权。
7. **Stage 5 协议只做工程验证。** 7 株、8 个资产不足以支撑泛化结论；重采样稳定性是噪声地板，不是模型好坏。
