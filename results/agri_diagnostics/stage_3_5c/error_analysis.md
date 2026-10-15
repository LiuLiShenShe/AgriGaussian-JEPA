# Stage 3.5-C：误检对象离线诊断

本阶段只读取 Stage 3.5-A 已保存的检索记录、特征和原语索引；没有模型前向、训练、重采样或新增候选规模实验。官方 Mean+Max 读出保持不变。

## 范围与复现

- 分支：`agri/plant-gs-experiments`；筛选：`pilot_trained` / `mean_max` / `independent_full_cap_1024` / `mixed19_auxiliary`。
- Query：DouBanLv2、XianKeLai1、XianKeLai2、XiangPiShu2；19 个 Gallery 身份及集合见 `summary.json`。
- 4 株独立保留植物，每株 3 个缺失比例 × XYZ 三个方向，共 36 个 Query；重复 Query 不是新增独立生物学样本。
- Gallery 与 Query 均为原来的 1024 Gaussian；从保存的索引复核 36 个正确身份交集全部为 0。
- 原表 `candidate_seen_train=15` 表示整个 Gallery 中的训练植物数量，并非被预测候选的布尔标签；本报告用 `sampling_plan.json` 的 `gallery_split` 对照固定植株划分确定每个候选的训练可见性。
- 特征为保存的 768 维 Mean+Max 原始向量，仅复用 L2 归一化和 float64 Cosine/排序；没有中心化、白化或特征选择。
- Gallery seed：`[4101]`；Query seeds：`[351000, 351001, 351002, 351100, 351101, 351102, 351200, 351201, 351202]`；没有新采样种子。
- 归一化：Stage 3 full-asset pretrain normalization; never renormalize a partial query；遮挡：remove largest projections; full-cap protocols use full original PLY; exact-count stable ties。XYZ 是各资产重建坐标，不保证跨植物具有相同的植物学方向。
- Checkpoint 仅作来源标识（本阶段不加载）：`experiments/agri_pilot/pilot/checkpoint_last.pt`，SHA256 `2f0c5b372db37d40308640f0c89a1cc9327ca037a4c94e26a21ff55b6f3c065c`。

在仓库 `Gaussian-JEPA` 目录执行：

```bash
/home/test/biosoft/enter/envs/gaussian_jepa/bin/python -m agri.plant_gs.jepa.analyze_stage35c
/home/test/biosoft/enter/envs/gaussian_jepa/bin/python -m agri.plant_gs.jepa.analyze_stage35c --output /tmp/agri-stage35c-reproduction-20261015
/home/test/biosoft/enter/envs/gaussian_jepa/bin/python -m agri.plant_gs.jepa.verify_stage35c --reproduction /tmp/agri-stage35c-reproduction-20261015
```

输出目录必须不存在；复现时使用新的目录，原 Stage 3 / 3.5-A / 3.5-B 产物禁止覆盖。

## 逐株结果

19 候选下命中 **14/36（38.89%）**，误检 22 次，平均 Retrieval Margin 为 -1.124841e-05。Margin 定义为正确身份 Cosine 减去最强错误身份 Cosine。

| Query 植物 | 命中 / 9 | R@1 | 平均 Margin | 中位 Margin | 最低 Margin | 最常见错误候选（次数） |
|---|---:|---:|---:|---:|---:|---|
| DouBanLv2 | 4/9 | 44.44% | +2.078126e-06 | -7.970726e-07 | -2.702347e-05 | WanNianQing1 (2); WanNianQing2 (2) |
| XianKeLai1 | 6/9 | 66.67% | +8.170295e-07 | +7.700622e-06 | -6.581634e-05 | CaoMei1 (2) |
| XianKeLai2 | 3/9 | 33.33% | -2.762263e-05 | -1.164922e-05 | -1.173325e-04 | CaoMei1 (4) |
| XiangPiShu2 | 1/9 | 11.11% | -2.026615e-05 | -2.584823e-05 | -4.111522e-05 | XiangPiShu1 (4) |

证据：`per_plant_errors.csv`、`query_error_details.csv`。各植株只有 9 个固定遮挡条件，比例只描述这批条件。

## 哪些候选造成误检

| 错误候选 | 训练可见性 | 错误 Top-1 次数 | 错误来源 Query（次数） |
|---|---|---:|---|
| CaoMei1 | seen_train | 7 | {"DouBanLv2":1,"XianKeLai1":2,"XianKeLai2":4} |
| XiangPiShu1 | seen_train | 4 | {"XiangPiShu2":4} |
| HongZhang | seen_train | 2 | {"XiangPiShu2":2} |
| WanNianQing1 | seen_train | 2 | {"DouBanLv2":2} |
| WanNianQing2 | seen_train | 2 | {"DouBanLv2":2} |
| WangWenCao2 | seen_train | 2 | {"XianKeLai2":2} |
| DouBanLv1 | seen_train | 1 | {"XiangPiShu2":1} |
| XianKeLai2 | unseen_test | 1 | {"XiangPiShu2":1} |
| XianKeLai3 | seen_train | 1 | {"XianKeLai1":1} |

证据：`candidate_error_summary.csv` 包含全部 19 个候选（包括零误检项）；`confusion_matrix.csv` 是 4×19 的完整预测计数矩阵，包含正确检索。

## 指定干扰候选的 Cosine 与排名

下表每行汇总该 Query 植物的 9 个条件。排名是完整 19 候选中的排名，Top-3 包含错误候选排名 ≤3 的次数；“超过正确”是该候选 Cosine 高于正确身份的次数。

| 候选 | Query 植物 | 平均 Cosine | 平均排名 | 排名范围 | Top-3 / 9 | 超过正确 / 9 | 错误 Top-1 / 9 |
|---|---|---:|---:|---:|---:|---:|---:|
| CaoMei1 | DouBanLv2 | 0.999745993 | 7.222 | 1–13 | 1 | 1 | 1 |
| CaoMei1 | XianKeLai1 | 0.999730883 | 1.889 | 1–3 | 9 | 3 | 2 |
| CaoMei1 | XianKeLai2 | 0.999799910 | 3.000 | 1–7 | 6 | 4 | 4 |
| CaoMei1 | XiangPiShu2 | 0.999703900 | 13.444 | 12–15 | 0 | 0 | 0 |
| XiangPiShu1 | DouBanLv2 | 0.999670159 | 14.556 | 9–17 | 0 | 0 | 0 |
| XiangPiShu1 | XianKeLai1 | 0.999316083 | 15.667 | 14–16 | 0 | 0 | 0 |
| XiangPiShu1 | XianKeLai2 | 0.999623931 | 14.333 | 9–16 | 0 | 0 | 0 |
| XiangPiShu1 | XiangPiShu2 | 0.999800223 | 2.000 | 1–4 | 8 | 7 | 4 |
| WanNianQing1 | DouBanLv2 | 0.999789821 | 2.333 | 1–5 | 8 | 3 | 2 |
| WanNianQing1 | XianKeLai1 | 0.999516274 | 9.667 | 9–12 | 0 | 0 | 0 |
| WanNianQing1 | XianKeLai2 | 0.999706074 | 9.222 | 7–12 | 0 | 0 | 0 |
| WanNianQing1 | XiangPiShu2 | 0.999739024 | 8.667 | 5–11 | 0 | 0 | 0 |
| WanNianQing2 | DouBanLv2 | 0.999781805 | 3.667 | 1–8 | 4 | 3 | 2 |
| WanNianQing2 | XianKeLai1 | 0.999394112 | 14.000 | 13–16 | 0 | 0 | 0 |
| WanNianQing2 | XianKeLai2 | 0.999639945 | 13.889 | 12–17 | 0 | 0 | 0 |
| WanNianQing2 | XiangPiShu2 | 0.999730353 | 9.444 | 5–13 | 0 | 0 | 0 |

CaoMei1 是跨多株的主要干扰项，但并非对全部植物都排在前列；XiangPiShu1 的高排名集中在 XiangPiShu2，WanNianQing1/2 的高排名集中在 DouBanLv2。绝对 Cosine 接近 1 不等于排名相近：必须同时看相对正确身份的差值和完整候选排名。

证据：`hard_negative_ranking.csv` 保存 36×18=648 个错误候选的逐 Query Cosine、排名和相对正确身份差值；`focus_candidate_statistics.csv` 另按缺失比例汇总四个重点候选。

## 命名类别、训练集合与遮挡的描述性关联

- 同一命名类别误检 5/22；不同命名类别 17/22。同类别 5 次具体为 XiangPiShu2→XiangPiShu1 四次、XianKeLai1→XianKeLai3 一次。错误并非主要集中在同类别。
- 类别定义仅为去掉 Plant ID 末尾数字的文件命名组，不能当作经过核实的物种标签。各类别候选数不同，因此上述次数不能解释为控制候选组成后的类别效应。
- 训练见过的候选造成 21/22 次误检，保留测试候选造成 1/22 次。每个 Query 有 15 个训练集错误候选、3 个保留集错误候选，比较机会为 540 对 108。候选不具可交换性，不用这些机会数构造随机错误概率或因果效应。
- **候选规模与训练可见性完全混杂**：从已有 4 候选到已有 19 候选，新增的 15 株全是训练植物。不能独立估计“候选增加”和“候选被训练见过”的作用；21/22 不能证明训练记忆或训练失败。

| 连续区域缺失 | 命中 / 12 | 错误 | R@1 | 平均 Margin | 训练候选误检 | 同类别误检 |
|---|---:|---:|---:|---:|---:|---:|
| 30% | 5/12 | 7 | 41.67% | -7.141227e-06 | 6 | 1 |
| 55% | 4/12 | 8 | 33.33% | -1.659004e-05 | 8 | 3 |
| 70% | 5/12 | 7 | 41.67% | -1.001396e-05 | 7 | 1 |

30%、55%、70% 缺失分别有 7、8、7 次错误，没有随缺失比例单调恶化的证据；只有每个比例 12 个相关条件，不能作人口层面的趋势推断。
已有方向条件的错误计数为 X 5/12、Y 10/12、Z 7/12；只作描述，不把各植物的重建坐标轴视作已对齐的语义方向。
证据：`occlusion_errors.csv`、`axis_errors.csv` 和逐 Query 表。

## 新候选造成排名下降，不等于模型自身退化

同一批已保存特征中，原有 4 候选命中 29/36，原有 19 候选命中 14/36。15 个原本正确的 Query 被新增候选超过；原来错误的 7 个仍然错误，没有候选增加后错误变正确的情形。

这只是核对两份已有评估记录，没有重新执行候选数量扩展实验。对每个 Query 逐项验证：

```text
正确身份的 19 候选排名 = 原 4 候选排名 + 排在正确身份之前的新增候选数
```

36 个 Query 全部满足上式；原 4 个候选的向量和分数没有变化，正确身份 Cosine 在两份旧记录中最大绝对差为 0.000e+00。
因此可以定位为新增干扰候选暴露了现有表征的相对区分不足，不能仅据此推断同一 Encoder 随候选数增加而发生了表征退化，也不能以此判定预训练失败。本阶段没有训练前后对照或新的模型因果干预。

## 花盆、颜色与结构：哪些证据有，哪些没有

- 原数据审计说明 SAM 前景可能保留花盆、标记物；现有预览图可直接看到部分这些对象。这里只能确认它们进入了资产，不能量化模型使用它们的程度。
- `error_cases.png` / `.pdf` 使用既有预览图的第一视角：Query 植物的完整 masked RGB、正确身份完整渲染、错误身份完整渲染。**它们不是实际空间遮挡 Query 的新渲染，也不是 1024 原语 Gallery 的渲染**。不同资产相机不对齐，不能从这张图推断颜色、花盆或结构导致了某次误检。
- 案例固定选择四个指定候选各自 Margin 最低的误检，再补齐未覆盖 Query 植物；全部错误仍保留在数字表里。原图路径、SHA256、裁剪坐标和选择规则见 `error_case_manifest.json`。
- 没有植物/花盆分区标签、颜色或几何的受控干预，也没有三者各自的特征贡献，因此“花盆相近”“颜色相近”“结构相近”均为待验证假设，不能报告为已确定原因。

| 资产 | 原重建 Silhouette IoU | 原 Background Alpha | 原质量警告 |
|---|---:|---:|---|
| CaoMei1 | 0.952982 | 0.005547 | 无 |
| XiangPiShu1 | 0.858197 | 0.039999 | silhouette_iou, background_alpha |
| WanNianQing1 | 0.963077 | 0.011722 | 无 |
| WanNianQing2 | 0.966700 | 0.009600 | 无 |
| XiangPiShu2 | 0.912569 | 0.036341 | background_alpha |

以上只是已有重建质量记录，不能把 Alpha/IoU 当成花盆占比或检索错误的因果解释。CaoMei1 没有原质量警告却产生最多误检；XiangPiShu1/2 都有残留背景相关警告，这提供排查线索，但不证明背景导致二者互混。

## 结论的证据强弱与唯一下一因素

| 判断 | 证据强度与边界 |
|---|---|
| CaoMei1、XiangPiShu1 是主要误检候选，且存在 Query 身份偏向 | 这批固定数据内的直接计数、完整 Cosine 排名证据强；不是跨数据集结论。 |
| 误检多数跨命名类别，且多数指向训练植物 | 直接计数成立；类别组成、训练可见性与新增候选数未控制，因果解释不足。 |
| 19 候选结果下降由新增候选超过正确身份产生 | 同一保存特征、分数与逐 Query 排名恒等式直接支持；不说明 Encoder 动态退化。 |
| 花盆 / 颜色 / 结构主导错误 | 只有共现与图像参考，没有独立受控证据，均为假设。 |
| 训练集身份导致干扰 | 21/22 来自训练候选，但候选数量混杂，原因未确定。 |

**下一步最有必要单独验证：候选的训练可见性。** 后续若获批准，需在固定候选数量、身份/命名类别构成、1024 预算和评估协议的匹配设计中，仅改变候选是否参与预训练；用不参与当前模型选择的确认数据检验这一因素。现有冻结模型和 19 株划分本身无法实现此因果对照。这个建议针对当前最大的解释缺口，不表示已经认定训练可见性是主因；本阶段不执行该实验。

尚不确定：为什么 CaoMei1 特别接近部分仙客来；为什么万年青主要干扰 DouBanLv2；这些相对排序来自植物本体、花盆、外观、结构、重建质量还是训练经历；四株之外是否复现。

## 校验与文件索引

- 已保存 CSV 的命中、Top-1、正确排名全部由特征复算匹配；分数最大误差 0.000e+00，预设容差 1e-12。没有使用新 Token 或新采样。
- 原始模型、checkpoint、PLY、预览图及 Stage 3/A/B 产物前后 SHA256 清单见 `protected_artifacts_before.json` / `protected_artifacts_after.json`。
- `analysis_checks.json`：完整索引零重叠、36 个条件唯一性、原表复算、所有数值有限性与排名恒等式。
- `verification.json`：独立实现重算全 36×19 分数与排名、汇总表一致性、两次独立进程输出复现；`run_completion.json` 给出最终完成状态。
- `summary.json`：所有汇总、集合、来源指纹、解释边界与推荐因素。`candidate_error_summary.csv`、`per_plant_errors.csv`、`occlusion_errors.csv`、`focus_candidate_statistics.csv` 提供可复核明细。

本报告只描述现有 4 株保留 Query 与混合 19 候选，不报告统计显著性，不将 36 个 Query 当成 36 株独立植物，不据测试结果优化、选择或修改模型。本阶段完成后停止。
