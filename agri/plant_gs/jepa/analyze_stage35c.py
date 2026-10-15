#!/usr/bin/env python3
"""Offline error attribution from the *saved* Stage 3.5-A retrieval features.

No model, checkpoint, PLY loader, sampler, or training code is imported. The
existing 4-candidate scores are used only to audit how the saved 19-way ranks
arose. This script refuses an existing output directory.
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / 'results/agri_diagnostics/stage_3_5a'
OUTPUT = ROOT / 'results/agri_diagnostics/stage_3_5c'
BRANCH = 'agri/plant-gs-experiments'
MODEL = 'pilot_trained'
FEATURE = 'mean_max'
PROTOCOL = 'independent_full_cap_1024'
FOCUS = ('CaoMei1', 'XiangPiShu1', 'WanNianQing1', 'WanNianQing2')
SCORE_ATOL = 1e-12  # float32 stored features; float64 norm/dot roundoff only.


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def write_csv(path, rows):
    if not rows:
        raise ValueError(f'empty table: {path}')
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path):
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


def category(plant_id):
    """Filename group only: these names are not validated taxonomic labels."""
    return re.sub(r'\d+$', '', plant_id)


def unit(value):
    value = np.asarray(value, dtype=np.float64)
    if not np.isfinite(value).all():
        raise ValueError('nonfinite saved feature')
    norm = np.linalg.norm(value, axis=-1, keepdims=True)
    if np.any(norm <= 0):
        raise ValueError('zero saved feature')
    return value / norm


def counts(rows, key):
    return dict(sorted(collections.Counter(row[key] for row in rows).items(),
                       key=lambda pair: (-pair[1], str(pair[0]))))


def compact_json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'))


def protected_paths(audit):
    """Hash all prior stage products, core code, original PLYs and previews."""
    paths = set()
    for directory in ('experiments/agri_pilot', 'results/agri_pilot',
                      'results/agri_diagnostics/stage_3_5a',
                      'results/agri_diagnostics/stage_3_5b'):
        paths.update(path for path in (ROOT / directory).rglob('*') if path.is_file())
    for directory in ('models', 'datasets', 'utils', 'agri/plant_gs/jepa'):
        paths.update((ROOT / directory).rglob('*.py'))
    paths.update((ROOT / 'agri/plant_gs/jepa').glob('*.yaml'))
    for record in audit['records']:
        paths.add(Path(record['ply']).resolve())
        if record.get('reconstruction_preview'):
            paths.add(Path(record['reconstruction_preview']).resolve())
    return sorted(paths)


def snapshot(paths):
    return {str(path): {'sha256': sha256(path), 'bytes': path.stat().st_size}
            for path in paths}


def score_summary(scores, gallery_ids, plant_id):
    order = np.argsort(-scores, kind='stable')
    target = gallery_ids.index(plant_id)
    wrong = [index for index in order if index != target]
    return {
        'retrieved_id': gallery_ids[int(order[0])],
        'hit': int(int(order[0]) == target),
        'true_rank': int(np.where(order == target)[0][0] + 1),
        'true_cosine': float(scores[target]),
        'strongest_wrong_cosine': float(scores[wrong[0]]),
        'retrieval_margin': float(scores[target] - scores[wrong[0]]),
        'top1_top2_gap': float(scores[order[0]] - scores[order[1]]),
    }


def audit_old_row(computed, saved):
    errors = []
    for key in ('retrieved_id', 'hit', 'true_rank'):
        expected = saved[key] if key == 'retrieved_id' else int(saved[key])
        if computed[key] != expected:
            raise AssertionError(f"saved rank mismatch {saved['key']} {key}")
    for key in ('true_cosine', 'strongest_wrong_cosine', 'retrieval_margin', 'top1_top2_gap'):
        error = abs(computed[key] - float(saved[key]))
        if error > SCORE_ATOL:
            raise AssertionError(f"saved score mismatch {saved['key']} {key}: {error}")
        errors.append(error)
    return max(errors)


def aggregate_queries(rows):
    margins = np.asarray([row['retrieval_margin'] for row in rows], dtype=np.float64)
    errors = [row for row in rows if not row['hit']]
    return {
        'queries': len(rows), 'hits': sum(row['hit'] for row in rows),
        'errors': len(errors), 'r_at_1': sum(row['hit'] for row in rows) / len(rows),
        'mean_margin': float(margins.mean()), 'median_margin': float(np.median(margins)),
        'min_margin': float(margins.min()), 'max_margin': float(margins.max()),
        'mean_true_rank_19': float(np.mean([row['true_rank_19'] for row in rows])),
        'seen_train_errors': sum(row['retrieved_split'] == 'seen_train' for row in errors),
        'unseen_test_errors': sum(row['retrieved_split'] == 'unseen_test' for row in errors),
        'same_named_category_errors': sum(row['same_named_category'] for row in errors),
        'different_named_category_errors': sum(not row['same_named_category'] for row in errors),
    }


def rank_statistics(rows):
    return {
        'queries': len(rows),
        'mean_cosine': float(np.mean([row['cosine'] for row in rows])),
        'min_cosine': min(row['cosine'] for row in rows),
        'max_cosine': max(row['cosine'] for row in rows),
        'mean_rank_19': float(np.mean([row['rank_19'] for row in rows])),
        'median_rank_19': float(np.median([row['rank_19'] for row in rows])),
        'min_rank_19': min(row['rank_19'] for row in rows),
        'max_rank_19': max(row['rank_19'] for row in rows),
        'top3_count': sum(row['rank_19'] <= 3 for row in rows),
        'above_true_count': sum(row['beats_true'] for row in rows),
        'false_top1_count': sum(row['is_top1_error'] for row in rows),
    }


def select_cases(query_rows):
    """Worst margin for each requested candidate, then cover missing query IDs.

    Selection is descriptive, fixed here before plotting, and does not alter
    evaluation. All 22 errors remain in the tables regardless of selection.
    """
    errors = [row for row in query_rows if not row['hit']]
    cases = []
    for candidate in FOCUS:
        options = [row for row in errors if row['retrieved_id'] == candidate]
        if options:
            cases.append(min(options, key=lambda row: (row['retrieval_margin'], row['query_key'])))
    for plant in sorted({row['query_plant_id'] for row in query_rows}):
        if plant not in {row['query_plant_id'] for row in cases}:
            options = [row for row in errors if row['query_plant_id'] == plant]
            if options:
                cases.append(min(options, key=lambda row: (row['retrieval_margin'], row['query_key'])))
    return cases


def figures(output, query_rows, gallery_ids, plants, audit_by_id):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image

    matrix = np.asarray([[sum(row['query_plant_id'] == plant and row['retrieved_id'] == candidate
                              for row in query_rows) for candidate in gallery_ids] for plant in plants])
    fig, ax = plt.subplots(figsize=(15, 4.5))
    image = ax.imshow(matrix, cmap='Blues', vmin=0, vmax=max(1, int(matrix.max())))
    for row in range(len(plants)):
        for col in range(len(gallery_ids)):
            ax.text(col, row, str(matrix[row, col]), ha='center', va='center', fontsize=9,
                    color='white' if matrix[row, col] >= 4 else 'black')
    ax.set_xticks(np.arange(len(gallery_ids)))
    ax.set_xticklabels([name + (' *' if name not in plants else '') for name in gallery_ids],
                       rotation=55, ha='right')
    ax.set_yticks(np.arange(len(plants)))
    ax.set_yticklabels(plants)
    ax.set_xlabel('Predicted gallery identity (* = seen in training)')
    ax.set_ylabel('Held-out query identity')
    ax.set_title('Saved Stage 3.5-A results: 36 queries, 19 candidates; 4 independent plants')
    fig.colorbar(image, ax=ax, label='Query count', shrink=0.8)
    fig.tight_layout()
    for suffix in ('png', 'pdf'):
        fig.savefig(output / ('confusion_matrix.' + suffix), dpi=180)
    plt.close(fig)

    cases = select_cases(query_rows)
    fig, axes = plt.subplots(len(cases), 3, figsize=(11, 4.5 * len(cases)), squeeze=False)
    manifest = []
    for case_index, case in enumerate(cases):
        plant, false_id = case['query_plant_id'], case['retrieved_id']
        for col, identity in enumerate((plant, plant, false_id)):
            preview = Path(audit_by_id[identity]['reconstruction_preview'])
            with Image.open(preview) as source:
                # Existing reconstruction preview: 6 views x 3 columns
                # (masked RGB, unmasked render, alpha). Fixed first view only.
                if source.size != (1080, 3972):
                    raise ValueError(f'unknown preview layout: {preview}: {source.size}')
                column = 0 if col == 0 else 1
                box = (column * 360, 0, (column + 1) * 360, 662)
                crop = source.crop(box).copy()
            axes[case_index, col].imshow(crop)
            axes[case_index, col].axis('off')
            role = ('Query identity / full RGB reference', 'Correct identity / full render',
                    'Incorrect Top-1 / full render')[col]
            axes[case_index, col].set_title(f'{role}\n{identity}', fontsize=10)
            manifest.append({
                'query_key': case['query_key'], 'query_plant_id': plant,
                'incorrect_candidate': false_id, 'column': col,
                'role': role, 'image_plant_id': identity, 'source_preview': str(preview),
                'source_sha256': sha256(preview), 'crop_box_xyxy': list(box),
                'is_actual_occluded_query_render': False,
            })
        axes[case_index, 0].set_ylabel(
            f"{case['query_key']}: missing {case['missing_ratio']:.0%} {case['axis_name']}\n"
            f"True rank {case['true_rank_19']}; overlap=0", fontsize=10)
        axes[case_index, 1].text(0.5, -0.015, f"Cosine = {case['true_cosine']:.9f}",
                               transform=axes[case_index, 1].transAxes, ha='center', va='top',
                               fontsize=10, bbox={'facecolor': 'white', 'edgecolor': 'none'})
        axes[case_index, 2].text(
            0.5, -0.015,
            f"Cosine = {case['strongest_wrong_cosine']:.9f}\nMargin = {case['retrieval_margin']:+.3e}",
            transform=axes[case_index, 2].transAxes, ha='center', va='top', fontsize=10,
            bbox={'facecolor': 'white', 'edgecolor': 'none'})
        # Axis-off suppresses ylabel, so put the case metadata in a visible title.
        axes[case_index, 0].set_title(
            f"{case['query_key']}: {plant}, missing {case['missing_ratio']:.0%} {case['axis_name']}\n"
            f"Full RGB reference (NOT the occluded query); true rank {case['true_rank_19']}", fontsize=10)
    fig.suptitle('Existing-image references for numerical errors; all views are complete assets\n'
                 'No new rendering; camera views are not aligned between plants; no visual causal attribution',
                 fontsize=12, y=0.998)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    for suffix in ('png', 'pdf'):
        fig.savefig(output / ('error_cases.' + suffix), dpi=150)
    plt.close(fig)
    write_json(output / 'error_case_manifest.json', {
        'selection_rule': 'minimum margin per requested candidate, then per uncovered query plant; ties by query key',
        'all_errors_in_numeric_tables': True,
        'image_caveat': 'Existing first-view complete-asset RGB/render references, NOT partial-query renders or sampled-gallery renders.',
        'cases': manifest,
    })


def report(output, summary, plants, candidates, focus, ratios, axes, audit_by_id):
    overall = summary['overall']
    lines = [
        '# Stage 3.5-C：误检对象离线诊断', '',
        '本阶段只读取 Stage 3.5-A 已保存的检索记录、特征和原语索引；没有模型前向、训练、重采样或新增候选规模实验。官方 Mean+Max 读出保持不变。', '',
        '## 范围与复现', '',
        f"- 分支：`{BRANCH}`；筛选：`{MODEL}` / `{FEATURE}` / `{PROTOCOL}` / `mixed19_auxiliary`。",
        '- Query：DouBanLv2、XianKeLai1、XianKeLai2、XiangPiShu2；19 个 Gallery 身份及集合见 `summary.json`。',
        '- 4 株独立保留植物，每株 3 个缺失比例 × XYZ 三个方向，共 36 个 Query；重复 Query 不是新增独立生物学样本。',
        '- Gallery 与 Query 均为原来的 1024 Gaussian；从保存的索引复核 36 个正确身份交集全部为 0。',
        '- 原表 `candidate_seen_train=15` 表示整个 Gallery 中的训练植物数量，并非被预测候选的布尔标签；本报告用 `sampling_plan.json` 的 `gallery_split` 对照固定植株划分确定每个候选的训练可见性。',
        '- 特征为保存的 768 维 Mean+Max 原始向量，仅复用 L2 归一化和 float64 Cosine/排序；没有中心化、白化或特征选择。',
        f"- Gallery seed：`{summary['seeds']['gallery']}`；Query seeds：`{summary['seeds']['query']}`；没有新采样种子。",
        f"- 归一化：{summary['normalization']}；遮挡：{summary['cap_definition']}。XYZ 是各资产重建坐标，不保证跨植物具有相同的植物学方向。",
        f"- Checkpoint 仅作来源标识（本阶段不加载）：`experiments/agri_pilot/pilot/checkpoint_last.pt`，SHA256 `{summary['checkpoint_sha256']}`。", '',
        '在仓库 `Gaussian-JEPA` 目录执行：', '', '```bash',
        '/home/test/biosoft/enter/envs/gaussian_jepa/bin/python -m agri.plant_gs.jepa.analyze_stage35c',
        '/home/test/biosoft/enter/envs/gaussian_jepa/bin/python -m agri.plant_gs.jepa.analyze_stage35c --output /tmp/agri-stage35c-reproduction-20261015',
        '/home/test/biosoft/enter/envs/gaussian_jepa/bin/python -m agri.plant_gs.jepa.verify_stage35c --reproduction /tmp/agri-stage35c-reproduction-20261015',
        '```', '',
        '输出目录必须不存在；复现时使用新的目录，原 Stage 3 / 3.5-A / 3.5-B 产物禁止覆盖。', '',
        '## 逐株结果', '',
        f"19 候选下命中 **{overall['hits']}/{overall['queries']}（{overall['r_at_1']:.2%}）**，误检 {overall['errors']} 次，平均 Retrieval Margin 为 {overall['mean_margin']:+.6e}。Margin 定义为正确身份 Cosine 减去最强错误身份 Cosine。", '',
        '| Query 植物 | 命中 / 9 | R@1 | 平均 Margin | 中位 Margin | 最低 Margin | 最常见错误候选（次数） |',
        '|---|---:|---:|---:|---:|---:|---|',
    ]
    for row in plants:
        lines.append(f"| {row['query_plant_id']} | {row['hits']}/9 | {row['r_at_1']:.2%} | {row['mean_margin']:+.6e} | {row['median_margin']:+.6e} | {row['min_margin']:+.6e} | {row['most_common_wrong_candidates']} |")
    lines += ['', '证据：`per_plant_errors.csv`、`query_error_details.csv`。各植株只有 9 个固定遮挡条件，比例只描述这批条件。', '',
              '## 哪些候选造成误检', '',
              '| 错误候选 | 训练可见性 | 错误 Top-1 次数 | 错误来源 Query（次数） |',
              '|---|---|---:|---|']
    for row in candidates:
        if row['false_top1_count']:
            lines.append(f"| {row['candidate_plant_id']} | {row['candidate_split']} | {row['false_top1_count']} | {row['false_top1_query_distribution']} |")
    lines += ['', '证据：`candidate_error_summary.csv` 包含全部 19 个候选（包括零误检项）；`confusion_matrix.csv` 是 4×19 的完整预测计数矩阵，包含正确检索。', '',
              '## 指定干扰候选的 Cosine 与排名', '',
              '下表每行汇总该 Query 植物的 9 个条件。排名是完整 19 候选中的排名，Top-3 包含错误候选排名 ≤3 的次数；“超过正确”是该候选 Cosine 高于正确身份的次数。', '',
              '| 候选 | Query 植物 | 平均 Cosine | 平均排名 | 排名范围 | Top-3 / 9 | 超过正确 / 9 | 错误 Top-1 / 9 |',
              '|---|---|---:|---:|---:|---:|---:|---:|']
    for row in focus:
        if row['missing_ratio'] == 'all':
            lines.append(f"| {row['candidate_plant_id']} | {row['query_plant_id']} | {row['mean_cosine']:.9f} | {row['mean_rank_19']:.3f} | {row['min_rank_19']}–{row['max_rank_19']} | {row['top3_count']} | {row['above_true_count']} | {row['false_top1_count']} |")
    lines += ['',
              'CaoMei1 是跨多株的主要干扰项，但并非对全部植物都排在前列；XiangPiShu1 的高排名集中在 XiangPiShu2，WanNianQing1/2 的高排名集中在 DouBanLv2。绝对 Cosine 接近 1 不等于排名相近：必须同时看相对正确身份的差值和完整候选排名。', '',
              '证据：`hard_negative_ranking.csv` 保存 36×18=648 个错误候选的逐 Query Cosine、排名和相对正确身份差值；`focus_candidate_statistics.csv` 另按缺失比例汇总四个重点候选。', '',
              '## 命名类别、训练集合与遮挡的描述性关联', '',
              f"- 同一命名类别误检 {overall['same_named_category_errors']}/{overall['errors']}；不同命名类别 {overall['different_named_category_errors']}/{overall['errors']}。同类别 5 次具体为 XiangPiShu2→XiangPiShu1 四次、XianKeLai1→XianKeLai3 一次。错误并非主要集中在同类别。",
              '- 类别定义仅为去掉 Plant ID 末尾数字的文件命名组，不能当作经过核实的物种标签。各类别候选数不同，因此上述次数不能解释为控制候选组成后的类别效应。',
              f"- 训练见过的候选造成 {overall['seen_train_errors']}/{overall['errors']} 次误检，保留测试候选造成 {overall['unseen_test_errors']}/{overall['errors']} 次。每个 Query 有 15 个训练集错误候选、3 个保留集错误候选，比较机会为 540 对 108。候选不具可交换性，不用这些机会数构造随机错误概率或因果效应。",
              '- **候选规模与训练可见性完全混杂**：从已有 4 候选到已有 19 候选，新增的 15 株全是训练植物。不能独立估计“候选增加”和“候选被训练见过”的作用；21/22 不能证明训练记忆或训练失败。', '',
              '| 连续区域缺失 | 命中 / 12 | 错误 | R@1 | 平均 Margin | 训练候选误检 | 同类别误检 |',
              '|---|---:|---:|---:|---:|---:|---:|']
    for row in ratios:
        lines.append(f"| {row['missing_ratio']:.0%} | {row['hits']}/12 | {row['errors']} | {row['r_at_1']:.2%} | {row['mean_margin']:+.6e} | {row['seen_train_errors']} | {row['same_named_category_errors']} |")
    lines += ['',
              '30%、55%、70% 缺失分别有 7、8、7 次错误，没有随缺失比例单调恶化的证据；只有每个比例 12 个相关条件，不能作人口层面的趋势推断。',
              '已有方向条件的错误计数为 ' + '、'.join(f"{row['axis_name']} {row['errors']}/{row['queries']}" for row in axes) + '；只作描述，不把各植物的重建坐标轴视作已对齐的语义方向。',
              '证据：`occlusion_errors.csv`、`axis_errors.csv` 和逐 Query 表。', '',
              '## 新候选造成排名下降，不等于模型自身退化', '',
              f"同一批已保存特征中，原有 4 候选命中 {summary['saved_scope_rank_audit']['heldout4_hits']}/36，原有 19 候选命中 {overall['hits']}/36。{summary['saved_scope_rank_audit']['correct4_to_wrong19']} 个原本正确的 Query 被新增候选超过；原来错误的 {summary['saved_scope_rank_audit']['wrong_in_both']} 个仍然错误，没有候选增加后错误变正确的情形。", '',
              '这只是核对两份已有评估记录，没有重新执行候选数量扩展实验。对每个 Query 逐项验证：', '',
              '```text',
              '正确身份的 19 候选排名 = 原 4 候选排名 + 排在正确身份之前的新增候选数',
              '```', '',
              f"36 个 Query 全部满足上式；原 4 个候选的向量和分数没有变化，正确身份 Cosine 在两份旧记录中最大绝对差为 {summary['saved_scope_rank_audit']['saved_true_cosine_max_abs_difference']:.3e}。",
              '因此可以定位为新增干扰候选暴露了现有表征的相对区分不足，不能仅据此推断同一 Encoder 随候选数增加而发生了表征退化，也不能以此判定预训练失败。本阶段没有训练前后对照或新的模型因果干预。', '',
              '## 花盆、颜色与结构：哪些证据有，哪些没有', '',
              '- 原数据审计说明 SAM 前景可能保留花盆、标记物；现有预览图可直接看到部分这些对象。这里只能确认它们进入了资产，不能量化模型使用它们的程度。',
              '- `error_cases.png` / `.pdf` 使用既有预览图的第一视角：Query 植物的完整 masked RGB、正确身份完整渲染、错误身份完整渲染。**它们不是实际空间遮挡 Query 的新渲染，也不是 1024 原语 Gallery 的渲染**。不同资产相机不对齐，不能从这张图推断颜色、花盆或结构导致了某次误检。',
              '- 案例固定选择四个指定候选各自 Margin 最低的误检，再补齐未覆盖 Query 植物；全部错误仍保留在数字表里。原图路径、SHA256、裁剪坐标和选择规则见 `error_case_manifest.json`。',
              '- 没有植物/花盆分区标签、颜色或几何的受控干预，也没有三者各自的特征贡献，因此“花盆相近”“颜色相近”“结构相近”均为待验证假设，不能报告为已确定原因。', '',
              '| 资产 | 原重建 Silhouette IoU | 原 Background Alpha | 原质量警告 |',
              '|---|---:|---:|---|']
    for plant in FOCUS + ('XiangPiShu2',):
        row = audit_by_id[plant]
        lines.append(f"| {plant} | {row['silhouette_iou']:.6f} | {row['background_alpha']:.6f} | {', '.join(row['quality_warnings']) or '无'} |")
    lines += ['',
              '以上只是已有重建质量记录，不能把 Alpha/IoU 当成花盆占比或检索错误的因果解释。CaoMei1 没有原质量警告却产生最多误检；XiangPiShu1/2 都有残留背景相关警告，这提供排查线索，但不证明背景导致二者互混。', '',
              '## 结论的证据强弱与唯一下一因素', '',
              '| 判断 | 证据强度与边界 |',
              '|---|---|',
              '| CaoMei1、XiangPiShu1 是主要误检候选，且存在 Query 身份偏向 | 这批固定数据内的直接计数、完整 Cosine 排名证据强；不是跨数据集结论。 |',
              '| 误检多数跨命名类别，且多数指向训练植物 | 直接计数成立；类别组成、训练可见性与新增候选数未控制，因果解释不足。 |',
              '| 19 候选结果下降由新增候选超过正确身份产生 | 同一保存特征、分数与逐 Query 排名恒等式直接支持；不说明 Encoder 动态退化。 |',
              '| 花盆 / 颜色 / 结构主导错误 | 只有共现与图像参考，没有独立受控证据，均为假设。 |',
              '| 训练集身份导致干扰 | 21/22 来自训练候选，但候选数量混杂，原因未确定。 |', '',
              '**下一步最有必要单独验证：候选的训练可见性。** 后续若获批准，需在固定候选数量、身份/命名类别构成、1024 预算和评估协议的匹配设计中，仅改变候选是否参与预训练；用不参与当前模型选择的确认数据检验这一因素。现有冻结模型和 19 株划分本身无法实现此因果对照。这个建议针对当前最大的解释缺口，不表示已经认定训练可见性是主因；本阶段不执行该实验。', '',
              '尚不确定：为什么 CaoMei1 特别接近部分仙客来；为什么万年青主要干扰 DouBanLv2；这些相对排序来自植物本体、花盆、外观、结构、重建质量还是训练经历；四株之外是否复现。', '',
              '## 校验与文件索引', '',
              f"- 已保存 CSV 的命中、Top-1、正确排名全部由特征复算匹配；分数最大误差 {summary['checks']['saved_score_replay_max_abs_error']:.3e}，预设容差 {SCORE_ATOL:.0e}。没有使用新 Token 或新采样。",
              '- 原始模型、checkpoint、PLY、预览图及 Stage 3/A/B 产物前后 SHA256 清单见 `protected_artifacts_before.json` / `protected_artifacts_after.json`。',
              '- `analysis_checks.json`：完整索引零重叠、36 个条件唯一性、原表复算、所有数值有限性与排名恒等式。',
              '- `verification.json`：独立实现重算全 36×19 分数与排名、汇总表一致性、两次独立进程输出复现；`run_completion.json` 给出最终完成状态。',
              '- `summary.json`：所有汇总、集合、来源指纹、解释边界与推荐因素。`candidate_error_summary.csv`、`per_plant_errors.csv`、`occlusion_errors.csv`、`focus_candidate_statistics.csv` 提供可复核明细。', '',
              '本报告只描述现有 4 株保留 Query 与混合 19 候选，不报告统计显著性，不将 36 个 Query 当成 36 株独立植物，不据测试结果优化、选择或修改模型。本阶段完成后停止。',
    ]
    (output / 'error_analysis.md').write_text('\n'.join(lines) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=SOURCE)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    args = parser.parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    branch = subprocess.check_output(['git', 'branch', '--show-current'], cwd=ROOT, text=True).strip()
    if branch != BRANCH:
        raise RuntimeError(f'expected branch {BRANCH}, found {branch}')
    if source != SOURCE.resolve():
        raise ValueError('this diagnosis must use the existing Stage 3.5-A source')
    for protected in (source, ROOT / 'experiments', ROOT / 'results/agri_pilot',
                      ROOT / 'results/agri_diagnostics/stage_3_5b', ROOT / 'models',
                      ROOT / 'datasets', ROOT / 'utils'):
        if output == protected or protected in output.parents:
            raise ValueError(f'output would be inside protected path: {protected}')
    if output.exists():
        raise FileExistsError(f'refusing to overwrite {output}')
    audit = json.loads((ROOT / 'experiments/agri_pilot/data_audit.json').read_text())
    split = json.loads((ROOT / 'experiments/agri_pilot/split.json').read_text())
    old_summary = json.loads((source / 'diagnostic_summary.json').read_text())
    plan = json.loads((source / 'sampling_plan.json').read_text())
    plants = split['test_plant_ids']
    training = split['train_plant_ids']
    if set(plants) & set(training) or len(plants) != 4 or len(training) != 15:
        raise ValueError('unexpected plant-level split')
    expected_splits = {plant: 'seen_train' for plant in training}
    expected_splits.update({plant: 'unseen_test' for plant in plants})
    if (plan['gallery_split'] != expected_splits or plan['test_plant_ids'] != plants
            or old_summary['test_plant_ids'] != plants):
        raise ValueError('saved train/test identity assignments are inconsistent')
    audit_by_id = {row['plant_id']: row for row in audit['records']}
    paths = protected_paths(audit)
    print(f'Hashing {len(paths)} protected prior files (including PLYs and previews)', flush=True)
    before = snapshot(paths)
    checkpoint = str(ROOT / 'experiments/agri_pilot/pilot/checkpoint_last.pt')
    if before[checkpoint]['sha256'] != old_summary['checkpoint_sha256'][MODEL]:
        raise ValueError('checkpoint provenance does not match Stage 3.5-A')
    saved = [row for row in read_csv(source / 'retrieval_per_query.csv')
             if row['model'] == MODEL and row['feature'] == FEATURE and row['protocol'] == PROTOCOL]
    rows19 = {row['key']: row for row in saved if row['candidate_scope'] == 'mixed19_auxiliary'}
    rows4 = {row['key']: row for row in saved if row['candidate_scope'] == 'heldout4_primary'}
    if len(saved) != 72 or len(rows19) != 36 or len(rows4) != 36 or rows19.keys() != rows4.keys():
        raise ValueError('expected exactly the same saved 36 queries in both scopes')
    plan_by_key = {row['key']: row for row in plan['queries']}
    query_rows, rankings, cosine_arrays = [], [], []
    maximum_error, minimum_score_gap = 0.0, 2.0
    with np.load(source / 'pilot_trained_retrieval_features.npz', allow_pickle=False) as features, \
            np.load(source / 'sampling_indices.npz', allow_pickle=False) as indices:
        gallery_ids = features['gallery_ids'].tolist()
        if gallery_ids != plan['auxiliary_gallery_ids'] or set(gallery_ids) != set(training + plants):
            raise ValueError('gallery identities do not match saved split and plan')
        gallery = unit(features['gallery_mean_max'])
        if gallery.shape != (19, 768):
            raise ValueError('expected saved 19 x 768 Mean+Max features')
        heldout_indices = [gallery_ids.index(plant) for plant in plants]
        for identity in gallery_ids:
            if indices['gallery_' + identity].shape != (1024,):
                raise ValueError('gallery budget changed')
        for key, old in sorted(rows19.items()):
            item = plan_by_key[key]
            plant = old['plant_id']
            if any(str(item[field]) != old[field] for field in
                   ('plant_id', 'protocol', 'missing_ratio', 'axis', 'query_seed', 'gallery_seed')):
                raise ValueError(f'query metadata differs from fixed source plan: {key}')
            if int(old['candidate_count']) != 19 or int(old['candidate_seen_train']) != 15:
                raise ValueError('unexpected mixed gallery scope')
            query_indices = indices[key]
            gallery_indices = indices['gallery_' + plant]
            overlap = len(np.intersect1d(query_indices, gallery_indices))
            if (query_indices.shape != (1024,) or len(np.unique(query_indices)) != 1024
                    or len(np.unique(gallery_indices)) != 1024 or overlap
                    or int(old['overlap_count']) or float(old['query_overlap_fraction'])):
                raise ValueError('saved query/gallery budget, uniqueness or overlap violation')
            if min(query_indices.min(), gallery_indices.min()) < 0 or max(
                    query_indices.max(), gallery_indices.max()) >= audit_by_id[plant]['gaussian_count']:
                raise ValueError('saved original PLY row index out of range')
            query = unit(features[key + '_mean_max'])
            if query.shape != (768,):
                raise ValueError('query feature shape changed')
            scores = gallery @ query
            cosine_arrays.append(scores)
            current = score_summary(scores, gallery_ids, plant)
            current4 = score_summary(scores[heldout_indices], plants, plant)
            maximum_error = max(maximum_error, audit_old_row(current, old),
                                audit_old_row(current4, rows4[key]))
            order = np.argsort(-scores, kind='stable')
            rank = np.empty(19, dtype=np.int64)
            rank[order] = np.arange(1, 20)
            minimum_score_gap = min(minimum_score_gap, float(np.diff(np.sort(scores)).min()))
            target = gallery_ids.index(plant)
            added_ahead = sum(rank[gallery_ids.index(pid)] < rank[target] for pid in training)
            if current['true_rank'] != current4['true_rank'] + added_ahead:
                raise AssertionError('rank identity for pre-existing candidate scopes failed')
            query_row = {
                'query_key': key, 'query_plant_id': plant, 'query_named_category': category(plant),
                'query_split': 'unseen_test', 'missing_ratio': float(old['missing_ratio']),
                'axis': int(old['axis']), 'axis_name': 'XYZ'[int(old['axis'])],
                'query_seed': int(old['query_seed']), 'gallery_seed': int(old['gallery_seed']),
                'query_gaussians': 1024, 'gallery_gaussians': 1024,
                'query_correct_gallery_overlap_count': overlap,
                'query_correct_gallery_overlap_fraction': overlap / 1024,
                'retrieved_id': current['retrieved_id'],
                'retrieved_named_category': category(current['retrieved_id']),
                'retrieved_split': plan['gallery_split'][current['retrieved_id']],
                'same_named_category': int(category(plant) == category(current['retrieved_id'])),
                'hit': current['hit'], 'true_cosine': current['true_cosine'],
                'strongest_wrong_cosine': current['strongest_wrong_cosine'],
                'retrieval_margin': current['retrieval_margin'],
                'top1_top2_gap': current['top1_top2_gap'],
                'true_rank_19': current['true_rank'], 'true_rank_4': current4['true_rank'],
                'saved_hit_4': current4['hit'], 'added_seen_candidates_ahead': int(added_ahead),
                'ply': audit_by_id[plant]['ply'],
            }
            query_rows.append(query_row)
            for candidate_index in order:
                candidate = gallery_ids[int(candidate_index)]
                if candidate == plant:
                    continue
                rankings.append({
                    'query_key': key, 'query_plant_id': plant,
                    'query_named_category': category(plant),
                    'missing_ratio': query_row['missing_ratio'], 'axis': query_row['axis'],
                    'axis_name': query_row['axis_name'], 'query_seed': query_row['query_seed'],
                    'gallery_seed': query_row['gallery_seed'], 'candidate_plant_id': candidate,
                    'candidate_named_category': category(candidate),
                    'candidate_split': plan['gallery_split'][candidate],
                    'same_named_category': int(category(candidate) == category(plant)),
                    'cosine': float(scores[candidate_index]), 'rank_19': int(rank[candidate_index]),
                    'true_cosine': current['true_cosine'], 'true_rank_19': current['true_rank'],
                    'candidate_minus_true_cosine': float(scores[candidate_index] - scores[target]),
                    'beats_true': int(scores[candidate_index] > scores[target]),
                    'is_top1_error': int(rank[candidate_index] == 1),
                    'focus_candidate': int(candidate in FOCUS),
                    'query_correct_gallery_overlap_count': overlap,
                })
    conditions = {(r['query_plant_id'], r['missing_ratio'], r['axis']) for r in query_rows}
    if conditions != {(p, r, a) for p in plants for r in (0.3, 0.55, 0.7) for a in range(3)}:
        raise ValueError('missing, duplicated or unexpected fixed query condition')
    if minimum_score_gap <= 0:
        raise ValueError('ties require explicit reconsideration of added-candidate rank audit')
    per_plant = []
    for plant in plants:
        subset = [row for row in query_rows if row['query_plant_id'] == plant]
        errors = [row for row in subset if not row['hit']]
        distribution = counts(errors, 'retrieved_id')
        most = max(distribution.values(), default=0)
        per_plant.append({
            'query_plant_id': plant, 'named_category': category(plant), 'query_split': 'unseen_test',
            **aggregate_queries(subset),
            'most_common_wrong_candidates': '; '.join(f'{name} ({count})' for name, count in distribution.items() if count == most),
            'wrong_candidate_distribution': compact_json(distribution),
            'saved_hits_4': sum(row['saved_hit_4'] for row in subset),
            'correct4_to_wrong19': sum(row['saved_hit_4'] and not row['hit'] for row in subset),
            **{f'errors_missing_{int(ratio * 100)}': sum(not row['hit'] and row['missing_ratio'] == ratio for row in subset)
               for ratio in (0.3, 0.55, 0.7)},
        })
    candidates = []
    for candidate in gallery_ids:
        subset = [row for row in rankings if row['candidate_plant_id'] == candidate]
        errors = [row for row in subset if row['is_top1_error']]
        candidates.append({
            'candidate_plant_id': candidate, 'candidate_named_category': category(candidate),
            'candidate_split': plan['gallery_split'][candidate], **rank_statistics(subset),
            'false_top1_query_distribution': compact_json(counts(errors, 'query_plant_id')),
            'same_named_category_false_top1': sum(row['same_named_category'] for row in errors),
            'different_named_category_false_top1': sum(not row['same_named_category'] for row in errors),
            **{f'false_top1_missing_{int(ratio * 100)}': sum(row['missing_ratio'] == ratio for row in errors)
               for ratio in (0.3, 0.55, 0.7)},
        })
    candidates.sort(key=lambda row: (-row['false_top1_count'], row['candidate_plant_id']))
    focus = []
    for candidate in FOCUS:
        for plant in plants:
            for ratio in ('all', 0.3, 0.55, 0.7):
                subset = [row for row in rankings if row['candidate_plant_id'] == candidate
                          and row['query_plant_id'] == plant
                          and (ratio == 'all' or row['missing_ratio'] == ratio)]
                focus.append({'candidate_plant_id': candidate, 'candidate_split': 'seen_train',
                              'query_plant_id': plant, 'missing_ratio': ratio, **rank_statistics(subset)})
    ratios = [{'missing_ratio': ratio, **aggregate_queries([row for row in query_rows if row['missing_ratio'] == ratio])}
              for ratio in (0.3, 0.55, 0.7)]
    axes = [{'axis': axis, 'axis_name': 'XYZ'[axis],
             **aggregate_queries([row for row in query_rows if row['axis'] == axis])} for axis in range(3)]
    overall = aggregate_queries(query_rows)
    source_names = ('retrieval_per_query.csv', 'diagnostic_summary.json', 'sampling_plan.json',
                    'sampling_indices.npz', 'pilot_trained_retrieval_features.npz')
    checks = {
        'saved_score_replay_max_abs_error': maximum_error, 'score_atol': SCORE_ATOL,
        'saved_rank_hit_and_prediction_replay_exact': True,
        'all_query_correct_gallery_original_index_intersections_zero': True,
        'all_query_and_gallery_budgets_1024': True, 'all_36_conditions_unique_and_present': True,
        'all_36_saved_scope_rank_identities_hold': True,
        'minimum_any_pair_score_gap': minimum_score_gap, 'score_ties': False,
        'all_saved_and_computed_numeric_features_finite': bool(np.isfinite(cosine_arrays).all()),
    }
    summary = {
        'stage': '3.5-C', 'mode': 'offline_saved_features_only', 'branch': branch,
        'filter': {'model': MODEL, 'feature': FEATURE, 'protocol': PROTOCOL, 'candidate_scope': 'mixed19_auxiliary'},
        'independent_plant_count': 4, 'test_plant_ids': plants, 'train_plant_ids': training,
        'gallery_ids': gallery_ids, 'gallery_split': plan['gallery_split'],
        'checkpoint_sha256': old_summary['checkpoint_sha256'][MODEL],
        'seeds': {'gallery': sorted({row['gallery_seed'] for row in query_rows}),
                  'query': sorted({row['query_seed'] for row in query_rows}), 'new_seeds': []},
        'normalization': plan['normalization'], 'cap_definition': plan['cap'],
        'source_sha256': {str(source / name): sha256(source / name) for name in source_names},
        'overall': overall, 'per_plant': per_plant, 'by_missing_ratio': ratios, 'by_axis': axes,
        'error_candidate_counts': {row['candidate_plant_id']: row['false_top1_count'] for row in candidates},
        'focus_candidate_statistics': [row for row in focus if row['missing_ratio'] == 'all'],
        'saved_scope_rank_audit': {
            'new_experiment_performed': False,
            'heldout4_hits': sum(row['saved_hit_4'] for row in query_rows),
            'mixed19_hits': overall['hits'],
            'correct4_to_wrong19': sum(row['saved_hit_4'] and not row['hit'] for row in query_rows),
            'wrong_in_both': sum(not row['saved_hit_4'] and not row['hit'] for row in query_rows),
            'wrong4_to_correct19': sum(not row['saved_hit_4'] and row['hit'] for row in query_rows),
            'saved_true_cosine_max_abs_difference': max(abs(float(rows19[key]['true_cosine']) - float(rows4[key]['true_cosine'])) for key in rows19),
            'rank_identity_verified_queries': 36,
        },
        'training_visibility_confound': {
            'all_15_added_candidates_seen_in_training': True,
            'seen_wrong_candidates_per_query': 15, 'heldout_wrong_candidates_per_query': 3,
            'seen_wrong_comparisons': 36 * 15, 'heldout_wrong_comparisons': 36 * 3,
            'causal_effect_identifiable': False,
        },
        'category_definition': 'Remove trailing digits from plant_id; filename group only, not validated taxonomy.',
        'checks': checks,
        'causal_attributions': {'pot': 'untested hypothesis', 'color': 'untested hypothesis',
                               'structure': 'untested hypothesis', 'training_visibility': 'confounded hypothesis'},
        'recommended_next_single_factor': 'candidate training visibility, with gallery size and composition controlled in a separately authorized confirmatory design',
        'model_selection_or_optimization_performed': False,
        'limitations': ['4 independent query plants only; 36 related conditions, not 36 plants',
                        'Single saved gallery sampling per identity; no new evaluation seeds',
                        'Candidate count and training visibility are confounded',
                        'Unequal named-category composition; labels are filename groups',
                        'Images are full-asset references, not actual occluded-query renders',
                        'No controlled pot, color, structure, training, or reconstruction-quality intervention'],
    }
    # Fail before producing reports if a numerical statistic cannot be serialized.
    json.dumps(summary, allow_nan=False)
    if sum(row['false_top1_count'] for row in candidates) != overall['errors'] or len(rankings) != 648:
        raise AssertionError('candidate and query error totals disagree')
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'protected_artifacts_before.json', before)
    write_csv(output / 'query_error_details.csv', query_rows)
    write_csv(output / 'hard_negative_ranking.csv', rankings)
    write_csv(output / 'per_plant_errors.csv', per_plant)
    write_csv(output / 'candidate_error_summary.csv', candidates)
    write_csv(output / 'focus_candidate_statistics.csv', focus)
    write_csv(output / 'occlusion_errors.csv', ratios)
    write_csv(output / 'axis_errors.csv', axes)
    write_csv(output / 'confusion_matrix.csv', [
        {'query_plant_id': plant, **{candidate: sum(row['query_plant_id'] == plant and row['retrieved_id'] == candidate
                                                  for row in query_rows) for candidate in gallery_ids}}
        for plant in plants])
    np.savez_compressed(output / 'saved_cosine_scores.npz',
                        query_keys=np.asarray([row['query_key'] for row in query_rows]),
                        gallery_ids=np.asarray(gallery_ids), cosine_scores=np.asarray(cosine_arrays))
    write_json(output / 'summary.json', summary)
    figures(output, query_rows, gallery_ids, plants, audit_by_id)
    report(output, summary, per_plant, candidates, focus, ratios, axes, audit_by_id)
    print('Checking that prior products, core model and original assets are unchanged', flush=True)
    after_paths = protected_paths(audit)
    after = snapshot(after_paths)
    write_json(output / 'protected_artifacts_after.json', after)
    if after != before:
        raise AssertionError('a protected input changed during analysis')
    write_json(output / 'analysis_checks.json', {**checks, 'protected_files_unchanged': True,
                                               'protected_file_count': len(before),
                                               'new_model_forward_calls': 0, 'new_sampling_calls': 0})
    write_json(output / 'provenance.json', {
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'python': sys.executable, 'numpy': np.__version__, 'source': str(source),
        'output': str(output), 'git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'analysis_script_sha256': sha256(Path(__file__)),
    })
    print(json.dumps({'output': str(output), 'hits': overall['hits'], 'errors': overall['errors'],
                      'same_named_category_errors': overall['same_named_category_errors'],
                      'seen_train_errors': overall['seen_train_errors'],
                      'saved_score_replay_max_abs_error': maximum_error}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
