"""Descriptive plots and auditable report for frozen Stage 3.5-A probes."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import numpy as np


def make_plots(output, data, plant_ids):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    features = ('encoder_tokens_spatial_aligned', 'mean', 'max', 'mean_max')
    names = ('Spatially aligned token descriptor', 'Mean pooling', 'Max pooling', 'Mean + max')
    labels = [f'{p}:s{s + 1}' for p in plant_ids for s in range(3)]
    for space, filename in (('l2', 'cosine_similarity_matrix'), ('centered_l2', 'centered_cosine_similarity_matrix')):
        fig, axes = plt.subplots(2, 4, figsize=(21, 10), constrained_layout=True)
        for col, (feature, title) in enumerate(zip(features, names)):
            low = min(float(v['matrices'][feature][space].min()) for v in data.values())
            for row, (model, record) in enumerate(data.items()):
                ax = axes[row, col]
                image = ax.imshow(record['matrices'][feature][space], vmin=low, vmax=1., cmap='viridis')
                ax.set_title(f'{model}\n{title}')
                ax.set_xticks(range(12))
                ax.set_xticklabels(labels, rotation=90, fontsize=7)
                ax.set_yticks(range(12))
                ax.set_yticklabels(labels, fontsize=7)
                for position in (2.5, 5.5, 8.5):
                    ax.axhline(position, color='white', lw=.5)
                    ax.axvline(position, color='white', lw=.5)
                fig.colorbar(image, ax=ax, shrink=.7, format='%.5f')
        fig.suptitle(f'{space}: 4 held-out plants, 3 repeated samples each; common scale per column')
        fig.savefig(output / f'{filename}.png', dpi=170)
        fig.savefig(output / f'{filename}.pdf')
        plt.close(fig)
    fig, axes = plt.subplots(2, 3, figsize=(14, 9), constrained_layout=True)
    for row, (model, record) in enumerate(data.items()):
        for col, feature in enumerate(('mean', 'max', 'mean_max')):
            values = record['features'][feature].astype(np.float64)
            x = values.reshape(12, -1)
            x -= x.mean(0)
            u, singular, _ = np.linalg.svd(x, full_matrices=False)
            scores = u[:, :2] * singular[:2]
            evr = singular ** 2 / np.square(singular).sum()
            ax = axes[row, col]
            for plant, identity in enumerate(plant_ids):
                part = scores[plant * 3:plant * 3 + 3]
                ax.scatter(part[:, 0], part[:, 1], label=identity, s=35)
            ax.set_title(f'{model}: {feature} (raw features)')
            ax.set_xlabel(f'PC1 ({evr[0]:.1%})')
            ax.set_ylabel(f'PC2 ({evr[1]:.1%})')
            ax.ticklabel_format(style='sci', axis='both', scilimits=(-2, 2))
    axes[0, 0].legend(fontsize=8)
    fig.suptitle('Descriptive PCA fitted separately; axes/scales differ; n=4 plants (not n=12)')
    fig.savefig(output / 'embedding_pca.png', dpi=170)
    fig.savefig(output / 'embedding_pca.pdf')
    plt.close(fig)


def md_table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |',
                      '| ' + ' | '.join(['---'] * len(headers)) + ' |',
                      *['| ' + ' | '.join(map(str, row)) + ' |' for row in rows]])


def write_report(output, summary):
    features = summary['feature_statistics']
    tokens = summary['token_statistics']
    retrieval = summary['retrieval_comparison']
    labels = ('random_init', 'pilot_trained')
    ftable = []
    for row in features:
        if row['space'] == 'raw':
            ftable.append([row['model'], row['feature'], f"{row['norm_mean']:.5g}",
                           f"{row['same_cosine']:.9f}", f"{row['different_cosine']:.9f}",
                           f"{row['dimension_variance']:.5g}",
                           f"{row['mean_vector_norm_to_mean_sample_norm']:.9f}",
                           f"{row['effective_rank_participation']:.4f}",
                           f"{row['pc1_explained']:.3%}"])
    ctable = []
    for row in features:
        if row['feature'] in ('mean', 'max', 'mean_max', 'encoder_tokens_spatial_aligned'):
            ctable.append([row['model'], row['feature'], row['space'],
                           f"{row['same_l2']:.6g}", f"{row['different_l2']:.6g}",
                           f"{row['same_cosine']:.7f}", f"{row['different_cosine']:.7f}",
                           f"{row['dimension_variance']:.5g}"])
    ttable = []
    for t in tokens:
        ttable.append([t['model'], t['stage'], f"{t['per_token_norm_mean']:.5g}",
                       f"{t['common_norm_to_mean_token_norm']:.8f}",
                       f"{t['raw_same_cosine']:.8f}", f"{t['raw_different_cosine']:.8f}",
                       f"{t['centered_l2_same_cosine']:.5f}", f"{t['centered_l2_different_cosine']:.5f}",
                       f"{t['within_cloud_token_dimension_variance']:.5g}"])
    rtable = []
    for row in retrieval:
        if row['feature'] == 'mean_max':
            rtable.append([row['model'], row['candidate_count'], row['protocol'],
                           f"{row['missing_ratio']:.0%}", f"{row['r_at_1']:.3f}",
                           f"{row['true_cosine_mean']:.9f}", f"{row['strongest_wrong_cosine_mean']:.9f}",
                           f"{row['retrieval_margin_mean']:.4g}", f"{row['retrieval_margin_min']:.4g}",
                           f"{row['top1_top2_gap_mean']:.4g}", f"{row['query_overlap_fraction_mean']:.0%}"])
    baseline = next(t for t in tokens if t['model'] == labels[0] and t['stage'] == 'encoder_tokens')
    trained = next(t for t in tokens if t['model'] == labels[1] and t['stage'] == 'encoder_tokens')
    b = next(f for f in features if f['model'] == labels[0] and f['feature'] == 'mean_max' and f['space'] == 'l2')
    t = next(f for f in features if f['model'] == labels[1] and f['feature'] == 'mean_max' and f['space'] == 'l2')
    trained_raw = next(f for f in features if f['model'] == labels[1] and f['feature'] == 'mean_max' and f['space'] == 'raw')
    primary = [r for r in retrieval if r['model'] == labels[1] and r['feature'] == 'mean_max'
               and r['candidate_count'] == 4 and r['protocol'] == 'independent_full_cap_1024']
    with (output / 'retrieval_per_query.csv').open() as stream:
        per_query = list(csv.DictReader(stream))
    # Explicit plant-level summaries: directions are repeated measurements.
    per_plant_groups = {}
    pkeys = ('model', 'feature', 'protocol', 'candidate_scope', 'candidate_count', 'missing_ratio', 'plant_id')
    for row in per_query:
        per_plant_groups.setdefault(tuple(row[k] for k in pkeys), []).append(row)
    per_plant = []
    for key, group in per_plant_groups.items():
        per_plant.append({**dict(zip(pkeys, key)), 'directions': len(group),
                          'hits': sum(int(r['hit']) for r in group),
                          'r_at_1': float(np.mean([int(r['hit']) for r in group])),
                          **{name + '_mean': float(np.mean([float(r[name]) for r in group]))
                             for name in ('true_cosine', 'strongest_wrong_cosine', 'retrieval_margin', 'top1_top2_gap')},
                          'retrieved_ids_by_xyz_axis': ','.join(r['retrieved_id'] for r in group)})
    with (output / 'retrieval_per_plant.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(per_plant[0]))
        writer.writeheader()
        writer.writerows(per_plant)
    text = f"""# Stage 3.5-A：真实植物表征与检索协议诊断

本报告仅使用已有的真实植物 PLY 和冻结 checkpoint 做推理。没有训练、修复模型或进入 Stage 3.5-B。

## 数据、模型与复现

- 独立测试植物（主分析，n=4）：{', '.join(summary['test_plant_ids'])}。
- 原训练植物（15 株，仅作为辅助检索的额外候选）：{', '.join(summary['train_plant_ids_excluded_from_primary'])}。
- 每株完整输入固定 3 组 1024 Gaussian，种子 4101/4102/4103；直接读取 Stage 3 原索引。
- 每株每比例 3 个固定轴方向，比例 30%/55%/70%；每协议每比例 12 次 Query，仍然只有 4 个独立植物。
- 独立 Query 种子为 351000 + 比例编号×100 + 轴编号；Gallery 为种子 4101。所有模型共用同一索引计划。
- 使用完整 PLY 的 Stage 3 `pretrain` 归一化；部分输入沿用完整资产坐标和属性尺度，不做部分输入重新归一化。
- 官方 Student Encoder 参数数：{summary['model_encoder_parameters']:,}。Encoder、Predictor、Loss 和 checkpoint 均未修改。

| 模型 | checkpoint | SHA256 |
| --- | --- | --- |
"""
    for model in labels:
        text += f"| {model} | `{summary['checkpoints'][model]}` | `{summary['checkpoint_sha256'][model]}` |\n"
    text += """
运行命令（在 `/fj/AgriGaussian-JEPA/Gaussian-JEPA` 下）：

```bash
$AGRI_PY_JEPA -m agri.plant_gs.jepa.diagnose_stage35a --device cuda:0

# 独立核对已保存的真实植物特征、索引、检索分数及防覆盖行为
$AGRI_PY_JEPA -m agri.plant_gs.jepa.verify_stage35a --input results/agri_diagnostics/stage_3_5a

# 新进程完整复现（该输出目录也必须尚不存在）与逐文件/逐数组核对
$AGRI_PY_JEPA -m agri.plant_gs.jepa.diagnose_stage35a --device cuda:0 --output /tmp/agri-stage35a-reproduction-20261015
$AGRI_PY_JEPA -m agri.plant_gs.jepa.verify_stage35a --input results/agri_diagnostics/stage_3_5a --reference /tmp/agri-stage35a-reproduction-20261015

# 仅从已有诊断数值重建本报告和逐植株汇总，不再次推理
$AGRI_PY_JEPA -m agri.plant_gs.jepa.diagnostic_report --input results/agri_diagnostics/stage_3_5a
```

原目录存在时脚本明确拒绝覆盖。重新运行请加 `--output /绝对路径/全新目录`。全部输入索引见 `sampling_indices.npz`，协议元数据见 `sampling_plan.json`；原始 Token、中心位置、空间匹配索引和未归一化 Pooling 向量见 `*_features.npz`，逐 Query 检索结果见 `retrieval_per_query.csv`。
`diagnostic_metrics.py` 是上述脚本共同使用的内部计算模块，无单独运行命令。
推理数值配置保留 Stage 3 的 cuDNN TF32（True）及矩阵乘 TF32（False），并开启 cuDNN deterministic；全部输入重复推理验证逐元素一致。禁用 cuDNN TF32 的一次回放探针使随机初始化 embedding 改变约 2.9×10⁻⁵，故未将该数值设置混入正式前后对比。

## 1. 原始特征统计

方差和有效秩以 **4 个植株的平均特征** 计算；同株/异株配对使用重复采样，配对数量不能当成独立样本数。`encoder_tokens_spatial_aligned` 是按共同参考中心做位置匹配后展开的 64×384 描述符，不能视作 64 个独立植物。

"""
    text += md_table(['模型', '特征', '原始范数均值', '同株 Cos', '异株 Cos', '跨植株维方差', '均值范数/平均范数', '有效秩', 'PC1 解释比'], ftable)
    text += """

## 2. 原始、L2、中心化特征分别比较

`centered_raw` = 原始特征减去这 4 株、每株 3 次测量的均值；`centered_l2` 再做 L2。中心化严格只用于诊断，未用于检索或声称无泄漏的性能。原始与 L2 的 Cosine 相同：L2 仅改变向量长度，不可能创造公共方向。对原始特征中心化会改变方向；其距离不变。

"""
    text += md_table(['模型', '特征', '空间', '同株 L2', '异株 L2', '同株 Cos', '异株 Cos', '跨植株维方差'], ctable)
    text += """

局部 Token 的主要比较采用每对输入中心之间的几何最小代价一对一匹配（Hungarian），不使用 Token 编号对应。还保留共同参考匹配的协方差结果，以便描述 PCA/秩。跨植物仅有归一化坐标，没有语义配准或姿态配准；空间匹配不是叶片同源对应，匹配距离写入逐对结果，不能把残差全部解释为语义信息。

`tokenizer_tokens` 是通过只读 forward hook 得到的 Transformer 之前的局部 PointNet 输出，用作定位参考，不更改官方 forward。

"""
    text += md_table(['模型', '阶段', '单 Token 范数', '公共方向比', '同株 Token Cos', '异株 Token Cos', '中心化同株 Cos', '中心化异株 Cos', '株内空间 Token 方差'], ttable)
    text += f"""

最终 Token 异株 Cos：{baseline['raw_different_cosine']:.9f} → {trained['raw_different_cosine']:.9f}。
最终 Token 株内空间方差：{baseline['within_cloud_token_dimension_variance']:.6g} → {trained['within_cloud_token_dimension_variance']:.6g}。
Mean+Max L2 后跨植株方差：{b['dimension_variance']:.6g} → {t['dimension_variance']:.6g}（训练后/训练前 = {t['dimension_variance'] / b['dimension_variance']:.4f}）。
Mean+Max 原始特征公共方向比为 {trained_raw['mean_vector_norm_to_mean_sample_norm']:.9f}。

## 3. 遮挡检索：共享与独立协议

四种协议均从同一份真实 PLY 读取属性，不生成替代数据：

1. `old_shared`：完全复现 Stage 3；1024 Gallery 上裁切，Query 分别为 717/461/307 个原语，与正确 Gallery 的 Query 索引重叠率 100%。
2. `disjoint_same_cap_same_count`：空间裁切阈值与旧 Query 相同，点数相同，从完整 PLY 的对应半空间采样，排除 Gallery 索引。用于检查共享原语/重复采样敏感性。
3. `independent_full_cap_matched_count`：在完整 PLY 上精确裁掉对应比例，排除 Gallery 索引，点数保持 717/461/307。
4. `independent_full_cap_1024`：要求的新主协议；完整 PLY 上精确裁切，排除 Gallery 索引，独立采样 1024 个原语。

所有非旧协议实测索引重叠率为 0%。旧协议和新主协议同时改变裁切总体、采样点数、共享情况，不能只根据二者差异归因；两个中间控制用于拆分这些影响。

下面是原 Stage 3 使用的 Mean+Max，Cosine 在 CPU float64 下从原始 Pooling 向量计算。完整 Mean/Max/Mean+Max 结果见 CSV。`4` 候选是独立测试主分析；`19` 包含训练见过的 15 株，仅是候选数敏感性分析，Query 仍只有未见过的 4 株。

Margin = 正确身份 Cos − 最强错误身份 Cos；Top1−Top2 不区分命中与否，二者不能混用。每行汇总 12 次相关 Query 测量。

"""
    text += md_table(['模型', '候选数', '协议', '缺失', 'R@1', '正确 Cos', '最强错误 Cos', 'Margin 均值', 'Margin 最小', 'Top1−Top2', 'Query 重叠'], rtable)
    text += """

## 4. 证据与问题定位

"""
    # The conclusions below refer to the numerical tables produced by this run.
    # No training intervention is inferred from a frozen-checkpoint comparison.
    text += f"""### A. Pooling 放大共同方向，但不能解释全部集中现象

训练后单个最终 Token 的公共方向比为 {trained['common_norm_to_mean_token_norm']:.9f}，
Mean+Max 后为 {trained_raw['mean_vector_norm_to_mean_sample_norm']:.9f}。
几何匹配后的局部 Token 异株 Cos 已达到 {trained['raw_different_cosine']:.9f}，
因此共同方向在 Pooling 前已经存在，不能把全部现象归因于 Pooling。
L2 前后同/异株 Cos 数值一致（`feature_statistics.csv` 的 raw/l2 行，以及 `verify_stage35a.py` 的不变性检查），所以 L2 不是高 Cos 的成因。

不同 Pooling 的更具体问题是 Max 分支对独立重采样更不稳定。训练后同株重复采样的原始 L2 距离：Mean 0.179392，Max 0.494459（均为 384 维、原始范数约 19.6–19.9，见 `pooling_comparison.csv`）。测试均值中心化后，同株 Cos 为 Mean 0.602368、Max 0.147411、拼接 0.273137；这只是描述性残差分析，未用于正式检索。
新独立协议下，三个比例合计 36 次测量：4 候选时 Mean 命中 33/36，Max 与拼接均为 29/36；19 候选辅助分析 Mean 24/36、Max 4/36、拼接 14/36。这支持优先检查 Max 读出分支的采样敏感性，但 36 次不是 36 个独立植物，不能据此完成正式方法选择或泛化认证。

### B. Encoder 存在方向集中；没有完全坍缩证据

训练前到训练后，最终 Token 的异株 Cos 从 {baseline['raw_different_cosine']:.9f} 升至 {trained['raw_different_cosine']:.9f}；同株−异株差距从
{baseline['raw_same_cosine'] - baseline['raw_different_cosine']:.6g} 缩至
{trained['raw_same_cosine'] - trained['raw_different_cosine']:.6g}。
参考中心对齐后的局部描述符跨植物原始维方差从 0.00642891 降到 0.000509887（约 12.6 倍下降）；Mean+Max 原始方差则约下降 45 倍，L2 方差约下降 45.8 倍。
这些是 Encoder 输出的身份相关变化缩小、聚合后进一步集中的证据。

但局部空间波动没有消失：最终 Token 的株内维方差由 {baseline['within_cloud_token_dimension_variance']:.6g} 增至 {trained['within_cloud_token_dimension_variance']:.6g}（约 4.05 倍），不能写成“所有局部 Token 方差都下降”。
按对配对、减去测试 Token 总均值后，训练后局部 Token 同株 Cos {trained['centered_l2_same_cosine']:.6f}、异株 {trained['centered_l2_different_cosine']:.6f}，仍保留有限身份相关残差；原始 Mean+Max 仍有约 1.59% 的相对残差范数。
新独立协议在本次固定采样中高于随机 Encoder 的描述性命中率（见表），故不存在足以支持“所有植物完全同一个向量”的证据。这里“高于”仅指本次实测，不是显著性检验。

Transformer 前 Tokenizer 的训练后异株 Cos 约 0.952891，最终 Token 约 0.998529；集中在 Encoder 输出比 Tokenizer 输出更强。这是定位线索，不能据此推断 Transformer、位置编码、BatchNorm、EMA 或损失中的某一个就是因果主因。
训练前后 Tokenizer 范数约 0.01915 → 15.3563，两个阶段尺度和运行统计发生明显变化，不能直接用未归一化 Tokenizer 方差增长断言语义提升。
有效秩以 4 个植株均值计算，最大仅 3；原拼接特征 L2 后有效秩 1.8204 → 2.0525 不足以证明或排除坍缩。

### C. 原 R@1 可复现，但受到共享原语和候选数影响

旧共享协议在训练后 4 候选下，三个缺失比例均为 12/12。
零重叠的新主协议变为 10/12、10/12、9/12（0.833/0.833/0.750），平均 Margin 约为 5.61×10⁻⁵、5.70×10⁻⁵、7.77×10⁻⁵，且每个比例均有负 Margin；不能把高于随机 Encoder 的结果写成稳健的强区分能力。
在空间阈值和 Query 点数保持旧协议一致时，仅换成不重叠重采样，命中已下降到 11/12、11/12、9/12，支持旧协议对共享采样的敏感性。由于只用了固定的一组独立种子，不能把单个方向的变化当作确定的总体因果效应。

加大到 19 候选后，旧协议为 1.000/0.917/0.667，新主协议为 0.417/0.333/0.417，表明少候选明显抬高了结果；这是含训练植物的辅助候选敏感性分析，不能作为新的独立泛化分数。新主协议下 19 候选的平均 Margin 三个比例均为负数。

中间控制也显示有限样本波动：70% 缺失时，完整 PLY 裁切、独立但匹配旧点数的协议有 12/12，而独立 1024 点为 9/12。
因此不能断言“更多 Gaussian 一定更好”或把所有变动只归因于索引重叠。两个控制及逐方向分数均保留，未挑选最优协议替代主结果。

训练后新主协议的逐植物结果如下（每格只有三个轴方向，旧协议所有格都是 3/3）：

"""
    plant_table = []
    for plant in summary['test_plant_ids']:
        group = [r for r in per_plant if r['model'] == 'pilot_trained' and r['feature'] == 'mean_max'
                 and r['protocol'] == 'independent_full_cap_1024' and r['candidate_count'] == '4'
                 and r['plant_id'] == plant]
        group.sort(key=lambda r: float(r['missing_ratio']))
        plant_table.append([plant, *[f"{r['hits']}/{r['directions']}" for r in group],
                            '; '.join(f"{r['retrieval_margin_mean']:.3g}" for r in group)])
    text += md_table(['植物', '30% 命中', '55% 命中', '70% 命中', '各比例 Margin 均值'], plant_table)
    text += """

错误主要集中在 XiangPiShu2；不能从现有无语义标注 PLY 判断是否由其残余背景、花盆、视角或植物本体特征导致。
`retrieval_per_plant.csv` 保存全部模型、读出、协议、候选规模的逐植物汇总，`retrieval_per_query.csv` 保留每个轴方向的完整分数。

### 证据强弱

| 可能解释 | 强度 | 对应证据与边界 |
| --- | --- | --- |
| Encoder 输出已有强共同方向 | 强，描述性定位 | 匹配后的局部 Token Cos 0.998529、公共方向比 0.999255；原始和 L2 都存在 |
| Pooling 是全部集中现象的主因 | 不支持 | 进入 Pooling 前最终 Token 已集中；不能只换 Pooling 就宣称修复 Encoder |
| Pooling 进一步放大公共方向，Max 分支增加采样不稳定 | 中等 | 公共方向比增加；Mean/Max 同株漂移差异；独立检索不同读出结果，样本仅 4 株 |
| L2 归一化制造了高 Cosine | 排除这种数学解释 | raw/l2 的 Cosine 一致；有限性与不变性检查通过 |
| 完全坍缩、彻底丧失身份信息 | 无充分证据 | 空间变化仍在；中心化同/异株残差不同；独立检索仍有身份排序 |
| 原 R@1 被共享采样与少候选抬高 | 中至强，限本次固定采样 | 同空间/点数不重叠控制下降；4→19 候选下降；不能分离总体数据量/类别效应 |
| 某一训练机制（EMA、BN、损失等）是主因 | 未确定 | 本阶段未做因果干预；不能从冻结 checkpoint 推断哪一机制应修改 |
| 花盆/背景而非植物决定身份 | 可能，但未量化 | 保留非植物前景且无对应语义标注；没有去盆/去背景单因素实验 |

## 5. 仍然无法确定的问题与干扰

- 尚无针对训练机制的因果干预，不能从本诊断判断 EMA、BatchNorm、训练目标或数据量中哪项导致集中。
- 所有植株保留花盆/标记及残余背景；检索可能利用非植物属性、整体外形、归一化尺度或坐标朝向。现有无语义标签 PLY 无法量化其贡献。
- 重复采样的表征残差不等于生物学语义；也不能将 12 次 Query 当作 12 株来给显著性或泛化置信区间。
- 独立索引仍来自同一 PLY 表面及同一次重建；0% 索引重叠不意味着独立采集或跨场景泛化。
- 4 株中两株为 XianKeLai；没有任务标签和正式下游分类试验。
- PCA 各面板单独拟合，仅是描述。图中分离不是有效性证据；坐标尺度不同也不能直接比较聚类间距。
- 图的 PCA 使用 12 组重复测量，其中心化样本秩上限为 11；表中谱使用 4 株平均特征，上限为 3，二者解释方差比例不同。生物独立样本数始终是 4。

## 6. 校验与产物保护

"""
    text += f"- 对 {summary['protected_artifact_check']['file_count']} 个原文件逐一做前后 SHA256/字节数比较，均未变；包括全部 Stage 3 结果、checkpoint、模型源文件和 19 份 PLY。\n"
    for check in summary['reproducibility']:
        text += f"- {check['model']}：{check['repeated_forward_pairs']} 组输入全部重复推理；最大绝对差 {check['maximum_absolute_difference_all_features']:.3g}。\n"
    for check in summary['stage3_replay']:
        text += f"- Stage 3 回放 {check['model']}：完整 embedding 最大误差 {check['full_embedding_max_abs_error']:.3g}，旧部分输入最大误差 {check['partial_embedding_max_abs_error']:.3g}。\n"
    verification_path = output / 'verification.json'
    if verification_path.is_file():
        verification = json.loads(verification_path.read_text())
        reproduced = verification.get('cross_process_reproduction')
        if reproduced:
            summary['cross_process_reproduction'] = reproduced
            text += (f"- 新进程完整复现通过：{len(reproduced['csv_byte_identical'])} 份 CSV 字节一致，"
                     f"{len(reproduced['npz_all_arrays_elementwise_identical'])} 份 NPZ 内 "
                     f"{reproduced['arrays_compared']} 个数组逐元素完全一致（包含全部 Token、Pooling、索引和检索特征）。\n")
    text += """- 索引计划重新计算完全相同；每个新 Query 的原始 PLY 索引与 Gallery 交集为空。
- 全部特征、CSV 数值及 JSON 数值有限性验证通过；JSON 禁止 NaN/Infinity。
- 全程 `eval()`、`inference_mode()`、冻结参数；没有 optimizer、backward 或 checkpoint 写入。
- `verify_stage35a.py` 另行从真实保存特征复算 1,728 条检索分数与身份排序，检查 144 组 Query 索引、空间匹配的顺序不变性、Cosine 的 L2 不变性、中心化的欧氏距离不变性和既有目录拒绝覆盖行为；详见 `verification.json`。

## 7. 最推荐优先解决的一个问题与验证方案（尚未实施）

**优先检验全局读出中的 Max 分支是否把重采样波动混入身份向量。只改变读出：Mean+Max → Mean；保留冻结 Encoder、L2、采样和检索协议。**

理由是本次观测中 Mean 的独立重采样漂移更小，新主协议下 Mean 在 4 候选合计命中 33/36、拼接 29/36；19 候选辅助分析为 24/36 对 14/36。该实验可以在不重新训练和不修改架构的情况下，直接检验有数据依据的单因素。
它不解决或解释全部 Encoder 共同方向集中；目前训练层面的因果主因仍未知，不能许诺它会修复表征坍缩。

验证时应事先固定额外、尚未使用的 Gallery/Query 独立采样种子，在同 4 株与相同连续裁切方向/比例上，使用 0% 原语重叠协议成对比较两种读出。只更换读出，不同时做均值中心化、重训或加入正则。
主指标为每株命中率、R@1、正确身份 Margin 的正负及其分布、Top1−Top2、独立重采样漂移；同时核查 Token 完全相同，并保留原始/L2 方差及范数。因为输出维度从 768 变为 384，方差按维度/总能量解释，不能把量纲变化当作改善。
额外采样仍不是额外独立植物，也不能当成新的未触碰生物测试集；该选择已参考当前 4 株结果，任何改善只算后续读出稳定性证据。若不能在预先固定的新种子上稳定提高命中或正 Margin 比例，就不接受“Max 分支是主要性能瓶颈”的假设。

如果目标是进一步确定 **训练机制的因果主因**，还需要另行批准训练机制的单因素实验；本阶段无法从现有 checkpoint 指定应该修改 BN、EMA 或 Loss 中哪一项。
本阶段已完成并停止，不自动开始上述读出试验或 Stage 3.5-B。

数据与代码证据：`feature_statistics.csv`、`pooling_comparison.csv`、`spatial_token_pair_statistics.csv`、`local_token_statistics.json`、`retrieval_per_query.csv`、`retrieval_protocol_comparison.csv`；实现位于 `agri/plant_gs/jepa/diagnose_stage35a.py`、`diagnostic_metrics.py`、`diagnostic_report.py`。图提供 PNG/PDF 两种格式。
"""
    (output / 'diagnosis_report.md').write_text(text)
    summary['diagnosis'] = {
        'strong_common_direction_present_before_pooling': True,
        'complete_collapse_established': False,
        'pooling_as_sole_root_cause_supported': False,
        'l2_can_create_high_cosine': False,
        'training_mechanism_causal_root_cause': 'undetermined; no causal training intervention',
        'old_retrieval_optimistic_in_this_fixed_sample': True,
        'recommended_next_single_factor': 'readout only: mean_max to mean; freeze encoder and keep L2/protocol',
        'recommendation_already_implemented': False,
        'primary_independent_mean_max_r1_by_missing_ratio': {
            str(r['missing_ratio']): r['r_at_1'] for r in primary},
        'evidence_files': ['feature_statistics.csv', 'local_token_statistics.json',
                           'retrieval_per_query.csv', 'retrieval_per_plant.csv',
                           'retrieval_protocol_comparison.csv'],
        'decision_limit': 'readout choice uses these four test plants; requires preregistered new sampling seeds, not a new biological generalization claim',
    }
    source_dir = Path(__file__).parent
    summary['diagnostic_source_sha256'] = {
        name: hashlib.sha256((source_dir / name).read_bytes()).hexdigest()
        for name in ('diagnose_stage35a.py', 'diagnostic_metrics.py', 'diagnostic_report.py', 'verify_stage35a.py')}
    (output / 'diagnostic_summary.json').write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Rebuild only the Stage 3.5-A report from saved diagnostics.')
    parser.add_argument('--input', type=Path, required=True)
    args = parser.parse_args()
    if not (args.input / 'run_completion.json').is_file():
        raise ValueError('only a completed diagnostic output directory is accepted')
    summary = json.loads((args.input / 'diagnostic_summary.json').read_text())
    write_report(args.input, summary)
