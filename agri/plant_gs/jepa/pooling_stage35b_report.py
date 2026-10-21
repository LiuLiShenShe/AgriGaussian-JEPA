"""Paired float64 scoring, descriptive plant-level summaries and Stage 3.5-B report."""
from __future__ import annotations

import csv
import json

import numpy as np

POOLINGS = ('mean_max', 'mean')
TOLERANCE = 1e-12


def score(query, gallery, plant_index, plant_ids):
    # Inputs have already been L2-normalized in float64.
    if query.dtype != np.float64 or gallery.dtype != np.float64:
        raise TypeError('retrieval must use float64')
    values = np.clip(gallery @ query, -1.0, 1.0)
    order = np.argsort(-values, kind='stable')
    correct = float(values[plant_index])
    wrong = float(np.max(np.delete(values, plant_index)))
    return {'correct_cosine': correct, 'strongest_wrong_cosine': wrong,
            'retrieval_margin': correct - wrong,
            'top1_top2_gap': float(values[order[0]] - values[order[1]]),
            'prediction': plant_ids[int(order[0])], 'correct': int(order[0] == plant_index),
            'exact_top1_tie': int(np.sum(values == values[order[0]]) > 1)}


def aggregate(rows, scope, group):
    result = {'scope': scope, 'group': group, 'queries': len(rows),
              'plants': len({r['plant_id'] for r in rows})}
    for pooling in POOLINGS:
        margins = np.asarray([r[pooling + '_retrieval_margin'] for r in rows], dtype=np.float64)
        hits = [r[pooling + '_correct'] for r in rows]
        plant_r1 = [np.mean([r[pooling + '_correct'] for r in rows if r['plant_id'] == p])
                    for p in sorted({r['plant_id'] for r in rows})]
        result.update({pooling + '_hits': int(sum(hits)), pooling + '_r1': float(np.mean(hits)),
                       pooling + '_plant_macro_r1': float(np.mean(plant_r1)),
                       pooling + '_mean_margin': float(margins.mean()),
                       pooling + '_median_margin': float(np.median(margins)),
                       pooling + '_min_margin': float(margins.min()),
                       pooling + '_positive_margin_fraction': float(np.mean(margins > 0))})
    for metric in ('r1', 'mean_margin', 'median_margin', 'min_margin', 'positive_margin_fraction'):
        result['delta_' + metric] = result['mean_' + metric] - result['mean_max_' + metric]
    result.update({
        'mean_hit_wins': sum(r['mean_correct'] > r['mean_max_correct'] for r in rows),
        'mean_hit_losses': sum(r['mean_correct'] < r['mean_max_correct'] for r in rows),
        'hit_ties_both_correct': sum(r['mean_correct'] == r['mean_max_correct'] == 1 for r in rows),
        'hit_ties_both_wrong': sum(r['mean_correct'] == r['mean_max_correct'] == 0 for r in rows),
        'mean_margin_wins': sum(r['delta_margin'] > TOLERANCE for r in rows),
        'mean_margin_losses': sum(r['delta_margin'] < -TOLERANCE for r in rows),
        'margin_ties': sum(abs(r['delta_margin']) <= TOLERANCE for r in rows),
    })
    result['hit_ties'] = result['hit_ties_both_correct'] + result['hit_ties_both_wrong']
    return result


def analyze(protocol, plan, features):
    lookup = {r['key']: i for i, r in enumerate(plan)}
    assets = {r['plant_id']: r for r in protocol['assets']}
    rows, drift = [], []
    for item in plan:
        if item['kind'] == 'gallery':
            continue
        index, gi = lookup[item['key']], lookup[item['gallery_key']]
        base = {k: item[k] for k in ('scheme', 'plant_id', 'gallery_seed', 'seed')}
        base.update({'source_ply': assets[item['plant_id']]['ply'], 'input_key': item['key'],
                     'gallery_key': item['gallery_key'], 'gaussians': 1024,
                     'overlap_count': item['overlap_count'],
                     'overlap_fraction': item['overlap_fraction']})
        if item['kind'] == 'resampling':
            row = dict(base)
            for pooling in POOLINGS:
                q, g = features[pooling][index], features[pooling][gi]
                cosine = float(np.clip(np.dot(q, g), -1.0, 1.0))
                row.update({pooling + '_cosine': cosine,
                            pooling + '_cosine_distance': 1 - cosine,
                            pooling + '_l2_drift': float(np.linalg.norm(q - g))})
            row['delta_l2_drift'] = row['mean_l2_drift'] - row['mean_max_l2_drift']
            row['delta_cosine_distance'] = row['mean_cosine_distance'] - row['mean_max_cosine_distance']
            drift.append(row)
            continue
        row = {**base, 'missing_ratio': item['missing_ratio'], 'direction': item['direction'],
               'actual_missing_fraction': item['actual_missing_fraction']}
        gallery_indices = [lookup[item['scheme'] + '_' + p + '_gallery'] for p in protocol['plant_ids']]
        for pooling in POOLINGS:
            result = score(features[pooling][index], features[pooling][gallery_indices],
                           protocol['plant_ids'].index(item['plant_id']), protocol['plant_ids'])
            row.update({pooling + '_' + k: v for k, v in result.items()})
        row['delta_margin'] = row['mean_retrieval_margin'] - row['mean_max_retrieval_margin']
        row['delta_hit'] = row['mean_correct'] - row['mean_max_correct']
        row['paired_hit_outcome'] = ('mean_win' if row['delta_hit'] > 0 else
                                     'mean_loss' if row['delta_hit'] < 0 else 'tie')
        rows.append(row)
    pooling = [aggregate(rows, 'overall', 'all')]
    for ratio in protocol['missing_ratios']:
        pooling.append(aggregate([r for r in rows if r['missing_ratio'] == ratio], 'ratio', str(ratio)))
    for scheme in protocol['schemes']:
        sid = scheme['id']
        subset = [r for r in rows if r['scheme'] == sid]
        pooling.append(aggregate(subset, 'scheme', sid))
        for ratio in protocol['missing_ratios']:
            pooling.append(aggregate([r for r in subset if r['missing_ratio'] == ratio],
                                     'scheme_ratio', f'{sid}/{ratio}'))
    for direction in protocol['directions']:
        pooling.append(aggregate([r for r in rows if r['direction'] == direction], 'direction', direction))
    plants = []
    for plant in protocol['plant_ids']:
        subset = [r for r in rows if r['plant_id'] == plant]
        plant_rows = [aggregate(subset, 'plant', plant)]
        for ratio in protocol['missing_ratios']:
            plant_rows.append(aggregate([r for r in subset if r['missing_ratio'] == ratio],
                                        'plant_ratio', f'{plant}/{ratio}'))
        for scheme in protocol['schemes']:
            sid = scheme['id']
            scheme_rows = [r for r in subset if r['scheme'] == sid]
            plant_rows.append(aggregate(scheme_rows, 'plant_scheme', f'{plant}/{sid}'))
            for ratio in protocol['missing_ratios']:
                plant_rows.append(aggregate([r for r in scheme_rows if r['missing_ratio'] == ratio],
                                            'plant_scheme_ratio', f'{plant}/{sid}/{ratio}'))
        plants.extend([{'plant_id': plant, **r} for r in plant_rows])
    overall = pooling[0]
    by_scheme = [r for r in pooling if r['scope'] == 'scheme']
    by_ratio = [r for r in pooling if r['scope'] == 'ratio']
    by_plant = [r for r in plants if r['scope'] == 'plant']
    checks = {
        'overall_r1_higher': overall['delta_r1'] > 0,
        'no_scheme_lower': all(r['delta_r1'] >= 0 for r in by_scheme),
        'at_least_two_schemes_higher': sum(r['delta_r1'] > 0 for r in by_scheme) >= 2,
        'at_least_two_plants_higher': sum(r['delta_r1'] > 0 for r in by_plant) >= 2,
        'no_plant_lower': all(r['delta_r1'] >= 0 for r in by_plant),
        'no_ratio_lower': all(r['delta_r1'] >= 0 for r in by_ratio),
    }
    drift_summary = {p + '_mean_l2': float(np.mean([r[p + '_l2_drift'] for r in drift])) for p in POOLINGS}
    drift_summary.update({p + '_mean_cosine_distance': float(np.mean([r[p + '_cosine_distance']
                                                                  for r in drift])) for p in POOLINGS})
    drift_summary['delta_mean_l2'] = drift_summary['mean_mean_l2'] - drift_summary['mean_max_mean_l2']
    stable = all(checks.values())
    summary = {
        'stage': '3.5-B', 'biological_n': 4, 'query_count': len(rows),
        'full_resampling_pairs': len(drift), 'gallery_candidates': 4,
        'overall': overall, 'by_ratio': by_ratio, 'by_scheme': by_scheme,
        'by_direction': [r for r in pooling if r['scope'] == 'direction'],
        'by_plant': by_plant, 'sampling_drift': drift_summary,
        'per_plant_paired_r1_deltas': {r['plant_id']: r['delta_r1'] for r in by_plant},
        'stable_r1_rule_checks': checks, 'stable_r1_gain': stable,
        'margin_improved': overall['delta_mean_margin'] > 0,
        'drift_reduced': drift_summary['delta_mean_l2'] < 0,
        'readout_decision': 'mean_candidate_only' if stable else 'retain_original_mean_max',
        'no_production_readout_changed': True,
        'interpretation_limit': 'new sampling seeds, not new biological objects; '
                                'descriptive plant-level n=4; no encoder-collapse remedy claim',
        'next_single_priority': '检索结论对候选集合规模的敏感性；固定模型、读出和采样预算，'
                                '单独改变候选数量。现有 19 株含 15 株训练见过的植物，'
                                '只能用于标明 seen/unseen 的辅助诊断，不能当独立泛化测试。',
    }
    return summary, {'retrieval_per_query': rows, 'pooling_comparison': pooling,
                     'per_plant_comparison': plants, 'sampling_drift': drift}


def dump_csv(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def md_table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |',
                      '| ' + ' | '.join(['---'] * len(headers)) + ' |'] +
                     ['| ' + ' | '.join(map(str, row)) + ' |' for row in rows])


def counts(row, pooling):
    return f"{row[pooling + '_hits']}/{row['queries']} ({row[pooling + '_r1']:.2%})"


def make_report(protocol, summary, tables):
    overall, drift = summary['overall'], summary['sampling_drift']
    rows = []
    for ratio in summary['by_ratio']:
        rows.append([f"R@1 {float(ratio['group']):.0%}", counts(ratio, 'mean_max'),
                     counts(ratio, 'mean'), f"{ratio['delta_r1']:+.2%}"])
    rows += [['R@1 overall', counts(overall, 'mean_max'), counts(overall, 'mean'), f"{overall['delta_r1']:+.2%}"],
             ['Mean Margin', f"{overall['mean_max_mean_margin']:.9e}", f"{overall['mean_mean_margin']:.9e}",
              f"{overall['delta_mean_margin']:+.9e}"],
             ['Median Margin', f"{overall['mean_max_median_margin']:.9e}", f"{overall['mean_median_margin']:.9e}",
              f"{overall['delta_median_margin']:+.9e}"],
             ['Minimum Margin', f"{overall['mean_max_min_margin']:.9e}", f"{overall['mean_min_margin']:.9e}",
              f"{overall['delta_min_margin']:+.9e}"],
             ['正 Margin 比例', f"{overall['mean_max_positive_margin_fraction']:.2%}",
              f"{overall['mean_positive_margin_fraction']:.2%}", f"{overall['delta_positive_margin_fraction']:+.2%}"],
             ['Resampling Drift (L2)', f"{drift['mean_max_mean_l2']:.9e}", f"{drift['mean_mean_l2']:.9e}",
              f"{drift['delta_mean_l2']:+.9e}"],
             ['Resampling Drift (1-cos)', f"{drift['mean_max_mean_cosine_distance']:.9e}",
              f"{drift['mean_mean_cosine_distance']:.9e}",
              f"{drift['mean_mean_cosine_distance'] - drift['mean_max_mean_cosine_distance']:+.9e}"]]
    lines = ['# Stage 3.5-B：冻结 Encoder 的全局 Pooling 单变量验证', '',
             f"固定 checkpoint：`{protocol['checkpoint']}`；SHA256 `{protocol['checkpoint_sha256']}`。",
             f"协议在推理前固定于 `{protocol['created_at_utc']}`；SHA256 `{summary['protocol_sha256']}`。",
             '', '## 数据支持的结论', '',
             ('按预先写入 protocol.json 的保守描述性规则，Mean 在本轮满足稳定 R@1 改善条件，'
              '只列为读出候选。' if summary['stable_r1_gain'] else
              '按预先写入 protocol.json 的保守描述性规则，Mean 没有满足稳定 R@1 改善的全部条件；保留原始 Mean+Max。'),
             '该规则不是显著性检验，也不证明对新植株泛化；本轮不改变任何生产读出配置。', '',
             md_table(['指标', 'Mean+Max', 'Mean', '差值 (Mean − Mean+Max)'], rows), '',
             f"逐 Query 命中结果：Mean 胜 {overall['mean_hit_wins']}，负 {overall['mean_hit_losses']}，"
             f"平 {overall['hit_ties']}（均正确 {overall['hit_ties_both_correct']}，均错误 {overall['hit_ties_both_wrong']}）。",
             f"Margin 逐 Query 胜 {overall['mean_margin_wins']}，负 {overall['mean_margin_losses']}，"
             f"平 {overall['margin_ties']}，绝对差 ≤ 1e-12 视为 Margin 打平。", '',
             'R@1、Margin 和漂移分别解释；更小的归一化漂移也可能受到强公共方向压缩的影响，'
             '不能单独视作身份信息增加。', '', '## 逐株与遮挡比例证据', '',
             md_table(['植株', 'Mean+Max', 'Mean', 'R@1 差值', 'Mean+Max Mean Margin', 'Mean Mean Margin'],
                      [[r['plant_id'], counts(r, 'mean_max'), counts(r, 'mean'), f"{r['delta_r1']:+.2%}",
                        f"{r['mean_max_mean_margin']:.7e}", f"{r['mean_mean_margin']:.7e}"]
                       for r in summary['by_plant']]), '',
             md_table(['植株 / 缺失比例', 'Mean+Max', 'Mean', 'R@1 差值', 'Mean Margin 差值'],
                      [[r['group'], counts(r, 'mean_max'), counts(r, 'mean'), f"{r['delta_r1']:+.2%}",
                        f"{r['delta_mean_margin']:+.7e}"]
                       for r in tables['per_plant_comparison'] if r['scope'] == 'plant_ratio']), '',
             '## 每组新种子与方向', '',
             md_table(['方案', 'Gallery seed', '完整重采样 seed', 'Query seeds (30%,55%,70%; XYZ)'],
                      [[s['id'], s['gallery_seed'], s['resampling_seed'], str(s['query_seeds'])]
                       for s in protocol['schemes']]), '',
             md_table(['方案 / 比例', 'Mean+Max', 'Mean', 'R@1 差值', 'Mean Margin 差值'],
                      [[r['group'], counts(r, 'mean_max'), counts(r, 'mean'), f"{r['delta_r1']:+.2%}",
                        f"{r['delta_mean_margin']:+.7e}"] for r in tables['pooling_comparison']
                       if r['scope'] in ('scheme', 'scheme_ratio', 'direction')]), '',
             '逐株 × 方案 × 遮挡比例的全部结果见 `per_plant_comparison.csv`；'
             '每个 Query 的预测、两个 Cosine、Margin、Top1−Top2 间隔、种子和重叠率见 `retrieval_per_query.csv`。', '',
             '## 完整植株重采样漂移', '',
             '每方案每株 1 对完整重采样，共 12 对；Probe 从完整资产采样，排除该 Gallery 的原语 ID，'
             '两者均为 1024。原始种子、原始 PLY 路径和索引已保存。', '',
             md_table(['植株 / 方案', 'Mean+Max L2 drift', 'Mean L2 drift', '差值'],
                      [[r['plant_id'] + '/' + r['scheme'], f"{r['mean_max_l2_drift']:.8e}",
                        f"{r['mean_l2_drift']:.8e}", f"{r['delta_l2_drift']:+.8e}"]
                       for r in tables['sampling_drift']]), '',
             '## 单变量控制与审计', '',
             '- 固定 4 株未参与训练的植物：' + '、'.join(protocol['plant_ids']) + '。',
             '- 每方案 4 个 Gallery，36 个遮挡 Query；3 方案合计 108 个 Query。'
             '所有输入均 1024 × 14，所有 Encoder 输出均 64 × 384。',
             '- 全 PLY 先按原农业加载器的 pretrain 模式归一化，之后沿 +X/+Y/+Z '
             '删除全资产最高投影的连续空间区域，按原 Stage 3.5-A 的稳定排序与 round 规则删除原语。'
             '缺失比例指 Gaussian 数量比例，不是体积比例。Query 不重新归一化。',
             '- 各 Query 从保留区域排除正确 Gallery 的原始行 ID 后无放回采样 1024 个。'
             '所有 Query/Gallery 索引交集严格为 0；索引身份相同才算重复，未声称去除了空间邻近。',
             '- 每份输入只前向一次，共 132 次。两种读出同时来自同一个 Token tensor；'
             'float32 Encoder 和原 torch Mean/Max；读出先转换 float64 再进行 L2、Cosine、Margin 与漂移计算。',
             '- 未加载随机模型作为额外变量；仅冻结同一训练后 student Encoder。模型参数、缓冲区及 checkpoint '
             'SHA256 均校验，禁止训练、反向传播、参数更新、中心化、白化或投影。',
             '- 旧种子审计包含 Stage 3/3.5-A 保存的所有 JSON/NPZ seed 字段、源码常量和 '
             'Dataset seed+epoch 推导的全部训练/加载器种子。新种子集合与历史集合无交集。'
             '完整种子审计和对应源文件 SHA256 见 protocol.json。',
             '- 所有采样索引见 sampling_indices.npz，索引对应的 PLY、空间阈值与实际删除比例见 sampling_plan.json。'
             '原始 Token、分组坐标、两种原始池化分量及两种归一化读出见 encoder_features.npz。', '',
             md_table(['预先固定的稳定性规则', '满足'], [[k, str(v)] for k, v in summary['stable_r1_rule_checks'].items()]), '',
             '## 尚不确定的判断与实验限制', '',
             '- 新种子避免复用 Stage 3.5-A 的具体随机采样，却仍使用已经参与假设形成的相同 4 株；'
             '本轮只验证这些植株上的采样稳健性，不能当作全新生物学验证。',
             '- 独立实验单位是植株 n=4。108 个 Query、12 个完整重采样对和 3 个方案均为重复测量。'
             '不把它们用于查询级显著性检验；平衡设计的总 R@1 与植株宏平均一致，逐株成对差值均已列出。',
             '- 候选集合只有 4 株。R@1 不表明在更大或更相似的候选集合仍能成功；'
             '余弦均接近 1 时还需结合绝对 Margin、最小 Margin 和逐株失败查看。',
             '- 本轮不能解释 Encoder 公共方向的成因，也不能区分训练目标、数据规模和重建属性等潜在来源。'
             '读出方式的效果不等于解决 Encoder 表征集中；未重新计算或要求公共方向比值下降。',
             '- 花盆、标签、残留背景沿用原 PLY，未分离语义部件。无法把任何检索收益专门归因于植物器官。',
             '- 零原始索引重叠不能使相邻的高斯或相似的重建纹理成为独立观测；'
             '连续单侧 XYZ 遮挡也不覆盖所有现实遮挡或坐标旋转。', '',
             '## 下一阶段只建议处理一个问题', '', summary['next_single_priority'],
             '预先固定候选身份、采样、种子和评价集合，仅添加干扰候选，'
             '报告每个未见过测试植株的 R@1 和 Margin 随候选数量的变化；'
             '如果该阶段要声称新植株泛化，还需另行获得未用于选择读出方案的独立测试植株。'
             '本轮未执行该实验，也没有扩充数据或改动默认 Pooling。', '',
             '## 运行和复现命令', '',
             '工作目录 `/fj/AgriGaussian-JEPA/Gaussian-JEPA`，分支 `agri/plant-gs-experiments`。', '',
             '```bash',
             '$AGRI_PY_JEPA -m agri.plant_gs.jepa.pooling_stage35b --prepare',
             '$AGRI_PY_JEPA -m agri.plant_gs.jepa.pooling_stage35b --run --device cuda:0',
             '$AGRI_PY_JEPA -m agri.plant_gs.jepa.verify_stage35b --output results/agri_diagnostics/stage_3_5b',
             '$AGRI_PY_JEPA -m agri.plant_gs.jepa.pooling_stage35b --replay-from results/agri_diagnostics/stage_3_5b --output /tmp/agri-stage35b-reproduction-20261015 --device cuda:0',
             '$AGRI_PY_JEPA -m agri.plant_gs.jepa.verify_stage35b --output /tmp/agri-stage35b-reproduction-20261015',
             '$AGRI_PY_JEPA -m agri.plant_gs.jepa.verify_stage35b --output results/agri_diagnostics/stage_3_5b --reference /tmp/agri-stage35b-reproduction-20261015',
             '```', '',
             'prepare 只用于尚不存在的目录；run 只接受仅含固定协议与保护清单的目录，拒绝覆盖已有结果。'
             '复现采用原 protocol.json，复制到新目录后重新加载真实 PLY 和 checkpoint，重新采样、重新推理。'
             'verification.json 记录有限性、索引约束、同 Token 读出、独立分数重算、产物保护和逐数组复现检查。'
             '只有该文件与 run_completion.json 均通过，才把本轮视为完成。', '']
    return '\n'.join(lines)


def write_outputs(output, protocol, summary, tables):
    for name, rows in tables.items():
        dump_csv(output / (name + '.csv'), rows)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharey=True)
    for ax, plant in zip(axes.flat, protocol['plant_ids']):
        rows = [r for r in tables['retrieval_per_query'] if r['plant_id'] == plant]
        x = np.arange(len(rows))
        ax.plot(x, [r['mean_max_retrieval_margin'] for r in rows], 'o-',
                color='#0072B2', markersize=3, linewidth=0.8, label='Mean+Max')
        ax.plot(x, [r['mean_retrieval_margin'] for r in rows], 's-',
                color='#D55E00', markersize=3, linewidth=0.8, label='Mean')
        ax.axhline(0, color='black', linewidth=0.7)
        ax.set_title(plant)
        ax.set_xlabel('Paired queries: scheme, missing ratio, XYZ')
        ax.set_ylabel('True identity cosine - strongest wrong cosine')
        ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    fig.suptitle('Frozen encoder; zero index overlap; 4 plants (108 repeated queries)')
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output / 'retrieval_margin.png', dpi=180)
    fig.savefig(output / 'retrieval_margin.pdf')
    plt.close(fig)
    (output / 'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    (output / 'diagnosis_report.md').write_text(make_report(protocol, summary, tables))
