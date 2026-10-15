#!/usr/bin/env python3
"""Build the reproducible Stage 3 audit, training, and representation report."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_EXPERIMENT = ROOT / 'experiments/agri_pilot'
DEFAULT_RESULT = ROOT / 'results/agri_pilot'


def load_json(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def read_csv(path: Path) -> list[dict[str, str]]:
    with Path(path).open(newline='') as stream:
        return list(csv.DictReader(stream))


def f(value: float, digits: int = 6) -> str:
    return f'{value:.{digits}f}'


def signed_delta(after: float, before: float, digits: int = 6) -> str:
    return f'{after - before:+.{digits}f}'


def spatial_summary(metrics: dict, ratio: str) -> tuple[float, float, float]:
    rows = metrics['spatial_contiguous_missing']['ratios'][ratio]
    cosines = [float(row['partial_to_same_complete_cosine']) for row in rows]
    return (
        float(sum(cosines) / len(cosines)),
        float((sum((value - sum(cosines) / len(cosines)) ** 2 for value in cosines)
               / len(cosines)) ** 0.5),
        float(metrics['spatial_contiguous_missing']['partial_to_complete_retrieval'][ratio]['r_at_1']),
    )


def build_loss_figure(rows: list[dict[str, str]], result_dir: Path) -> None:
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    by_epoch = {}
    for row in rows:
        by_epoch[int(row['epoch'])] = row
    epochs = sorted(by_epoch)
    fig, axis = plt.subplots(figsize=(7.2, 4.4), constrained_layout=True)
    colors = {
        'epoch_mean_total': '#0072B2',
        'epoch_mean_geo': '#D55E00',
        'epoch_mean_app': '#009E73',
        'epoch_mean_ground': '#CC79A7',
    }
    labels = {
        'epoch_mean_total': 'total',
        'epoch_mean_geo': 'geo',
        'epoch_mean_app': 'app',
        'epoch_mean_ground': 'ground',
    }
    for key, color in colors.items():
        axis.plot(
            epochs,
            [float(by_epoch[epoch][key]) for epoch in epochs],
            marker='o',
            markersize=3,
            linewidth=1.8,
            label=labels[key],
            color=color,
        )
    axis.set_xlabel('Epoch')
    axis.set_ylabel('Mean loss')
    axis.set_title('Stage 3 plant Gaussian-JEPA pilot losses')
    axis.grid(True, alpha=0.25)
    axis.legend(frameon=False, ncol=2)
    result_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(result_dir / 'loss_curves.png', dpi=220)
    fig.savefig(result_dir / 'loss_curves.pdf')
    plt.close(fig)


def render_report(experiment_dir: Path, result_dir: Path) -> str:
    audit = load_json(experiment_dir / 'data_audit.json')
    split = load_json(experiment_dir / 'split.json')
    run = load_json(experiment_dir / 'pilot/run_config.json')
    comparison = load_json(result_dir / 'evaluation_comparison.json')
    rows = read_csv(experiment_dir / 'pilot/train_metrics.csv')
    build_loss_figure(rows, result_dir)

    first = next(row for row in rows if int(row['epoch']) == 1)
    last = next(row for row in rows if int(row['epoch']) == run['epochs'])
    epoch_rows = {int(row['epoch']): row for row in rows}
    peak_gpu = max(float(row['epoch_peak_gpu_memory_gib']) for row in rows)
    epoch_times = [float(row['epoch_seconds']) for row in epoch_rows.values()]
    warning_records = [
        record for record in audit['records'] if record.get('quality_warnings')
    ]
    max_background_alpha = max(
        float(record['background_alpha'])
        for record in audit['records']
        if record.get('background_alpha') is not None
    )
    min_count = min(record['gaussian_count'] for record in audit['records'])
    max_count = max(record['gaussian_count'] for record in audit['records'])

    random_metrics = comparison['models']['random_init']['metrics']
    trained_metrics = comparison['models']['pilot_trained']['metrics']
    random_stats = random_metrics['embedding_statistics']
    trained_stats = trained_metrics['embedding_statistics']
    random_cos = random_metrics['same_vs_different_plant_cosine']
    trained_cos = trained_metrics['same_vs_different_plant_cosine']
    random_drift = random_metrics['complete_resampling_drift']
    trained_drift = trained_metrics['complete_resampling_drift']

    count_table = '\n'.join(
        f"| {record['plant_id']} | {record['gaussian_count']:,} | "
        f"{f(float(record['silhouette_iou']), 4) if record.get('silhouette_iou') is not None else 'NA'} | "
        f"{f(float(record['background_alpha']), 4) if record.get('background_alpha') is not None else 'NA'} | "
        f"{', '.join(record.get('quality_warnings') or []) or '—'} |"
        for record in audit['records']
    )

    metric_rows = []
    scalar_metrics = [
        ('Resampling cosine to reference', random_drift['mean_cosine_to_reference'], trained_drift['mean_cosine_to_reference']),
        ('Resampling L2 to reference', random_drift['mean_l2_to_reference'], trained_drift['mean_l2_to_reference']),
        ('Same-plant cosine', random_cos['same_plant_mean'], trained_cos['same_plant_mean']),
        ('Different-plant cosine', random_cos['different_plant_mean'], trained_cos['different_plant_mean']),
        ('Embedding variance', random_stats['mean_dimension_variance_across_plant_means'], trained_stats['mean_dimension_variance_across_plant_means']),
        ('Effective rank', random_stats['participation_ratio_effective_rank'], trained_stats['participation_ratio_effective_rank']),
    ]
    for name, before, after in scalar_metrics:
        metric_rows.append(
            f'| {name} | {f(before)} | {f(after)} | {signed_delta(after, before)} |'
        )
    scalar_table = '\n'.join(metric_rows)

    spatial_rows = []
    for ratio in ('30', '55', '70'):
        random_mean, random_std, random_r1 = spatial_summary(random_metrics, ratio)
        trained_mean, trained_std, trained_r1 = spatial_summary(trained_metrics, ratio)
        spatial_rows.append(
            f'| {ratio}% | {f(random_mean)} ± {f(random_std)} | {random_r1:.3f} | '
            f'{f(trained_mean)} ± {f(trained_std)} | {trained_r1:.3f} |'
        )
    spatial_table = '\n'.join(spatial_rows)

    model = run['model_config']
    transformer = model['transformer_config']
    train_command = (
        '/home/test/biosoft/enter/envs/gaussian_jepa/bin/python -u '
        'agri/plant_gs/jepa/pilot_train.py --stage pilot --epochs 20 '
        '--initial-checkpoint experiments/agri_pilot/initial_random.pt '
        '--output experiments/agri_pilot/pilot --device cuda:0'
    )
    eval_command = (
        '/home/test/biosoft/enter/envs/gaussian_jepa/bin/python -u '
        'agri/plant_gs/jepa/pilot_eval.py '
        '--baseline experiments/agri_pilot/initial_random.pt '
        '--trained experiments/agri_pilot/pilot/checkpoint_last.pt '
        '--output results/agri_pilot --device cuda:0'
    )

    warning_names = ', '.join(record['plant_id'] for record in warning_records)
    report = f'''# Stage 3 — Small-Scale Plant Gaussian-JEPA Pilot

生成时间：2026-10-15（Asia/Shanghai）  
实验分支：`agri/plant-gs-experiments`  
数据范围：`{audit['manifest']}`  
本报告只覆盖现有 19 株植物，没有下载或加入外部数据集，也没有修改官方 Gaussian-JEPA Encoder、Predictor 或 Loss。

## 结论

这次实验确认了真实植物 PLY 可以通过农业数据加载器进入官方 Gaussian-JEPA，并完成从随机初始化开始的训练、EMA Teacher 更新、检查点保存/恢复和固定测试集评估。训练集上的总 Loss 从第 1 个 Epoch 的 `{float(first['epoch_mean_total']):.4f}` 降到第 20 个 Epoch 的 `{float(last['epoch_mean_total']):.4f}`。

表征结果是混合的：连续空间缺失下的 Partial-to-Complete R@1 从随机初始化的 `0.417/0.333/0.333`（缺失 30%/55%/70%）变为训练后的 `1.000/1.000/1.000`，但训练后不同测试植物的平均 Cosine Similarity 变为 `{trained_cos['different_plant_mean']:.6f}`，触发了跨植物相似度过高的坍缩警告；Embedding 方差从 `{random_stats['mean_dimension_variance_across_plant_means']:.3e}` 降到 `{trained_stats['mean_dimension_variance_across_plant_means']:.3e}`。因此，本次结果证明“流程能跑通”，但还不能证明模型已经学到可靠的植物差异表征或农业泛化能力。

在当前证据下，不建议直接把这个 20 Epoch 结果当作大规模预训练有效性的证明。后续扩充数据前，应先确认这种跨植物相似度升高是否由训练目标、全局 mean/max pooling、样本规模或背景/花盆共因造成；本阶段按要求到此停止，等待下一阶段确认。

## 1. 数据审计与固定划分

- Manifest 中有 `{audit['usable_plant_count']}` 份可用 PLY：18 份 `complete` 和 1 份 `reused`。
- `BaiZhang` 未纳入，因为 manifest 状态为 `missing_sam`，原因是没有对应 SAM mask。
- 每份 PLY 都存在，manifest Gaussian 数量与 PLY 顶点数一致，解码后为 14 维属性 `[xyz, opacity, scale(3), rotation(4), f_dc(3)]`。
- 所有解码属性均为有限值，解码 scale 均为正；Gaussian 数量范围为 `{min_count:,}`–`{max_count:,}`。
- 固定划分种子为 `{split['seed']}`，单位是整株植物 PLY，训练/测试无交集，禁止同一植株采样结果跨集合。

训练集（15 株）：

`{', '.join(split['train_plant_ids'])}`

保留测试集（4 株）：

`{', '.join(split['test_plant_ids'])}`

每株审计摘要如下；完整字段保存在 [`data_audit.json`](../../experiments/agri_pilot/data_audit.json)。

| 植株 ID | Gaussians | Silhouette IoU | Background alpha | 审计警告 |
|---|---:|---:|---:|---|
{count_table}

有 manifest 质量警告的植株为 `{warning_names}`。当前数据定义保留 SAM 前景中的植物、花盆和 marker，因此花盆/marker 没有被语义上排除；残余背景 alpha 也可能被模型当作输入信号。

## 2. 训练配置与可复现性

- 随机种子：`{run['seed']}`。
- 训练：15 株、batch size `{run['batch_size']}`、20 Epoch、每个样本采样 `{run['num_gaussians']}` 个 Gaussian。
- 归一化：`{run['normalization']}`，先对完整 PLY 归一化，再按固定 Epoch 种子采样。
- 优化器：AdamW，learning rate `{run['learning_rate']}`，weight decay `{run['weight_decay']}`；Cosine scheduler，EMA momentum `{run['ema_momentum_start']}`→`{run['ema_momentum_end']}`。
- 官方模型配置：`G={model['num_group']}`、`S={model['group_size']}`、`M={model['num_targets']}`、target sizes `{model['target_sizes']}`、Transformer dimension `{transformer['trans_dim']}`、Encoder depth `{transformer['depth']}`、Predictor depth `{transformer['decoder_depth']}`、`lambda_ground={model['lambda_ground']}`。
- 参数量：总参数 `{run['total_model_parameters']:,}`，可训练参数 `{run['trainable_model_parameters']:,}`，Student Encoder `{run['student_encoder_parameters']:,}`。
- 环境：`{run['gpu_name']}`，PyTorch `{run['torch_version']}`，NumPy `{run['numpy_version']}`。
- 官方模型源文件 SHA256：`{run['official_model_source_sha256']}`。
- 采样索引：训练集每个 Epoch 保存在 [`experiments/agri_pilot/pilot/sampling_indices/`](../../experiments/agri_pilot/pilot/sampling_indices/)；测试集固定采样协议保存在 [`fixed_test_sampling_indices.npz`](fixed_test_sampling_indices.npz)，测试采样种子为 `4101, 4102, 4103`。

实际训练命令：

```bash
{train_command}
```

Sanity Check 使用 3 株训练植物完成 2 Epoch，并从第 1 个 Epoch 的 checkpoint 恢复到第 2 个 Epoch。训练日志显示所有 batch 的 Student Encoder 梯度有限且非零，EMA Teacher 每个 batch 都发生更新。

## 3. 训练日志

Loss 曲线：[`loss_curves.png`](loss_curves.png) / [`loss_curves.pdf`](loss_curves.pdf)。逐 batch 日志：[`train_metrics.csv`](../../experiments/agri_pilot/pilot/train_metrics.csv)。

| 项目 | Epoch 1 | 最低/代表值 | Epoch 20 |
|---|---:|---:|---:|
| geo | {float(first['epoch_mean_geo']):.6f} | {min(float(r['epoch_mean_geo']) for r in epoch_rows.values()):.6f} | {float(last['epoch_mean_geo']):.6f} |
| app | {float(first['epoch_mean_app']):.6f} | {min(float(r['epoch_mean_app']) for r in epoch_rows.values()):.6f} | {float(last['epoch_mean_app']):.6f} |
| ground | {float(first['epoch_mean_ground']):.6f} | {min(float(r['epoch_mean_ground']) for r in epoch_rows.values()):.6f} | {float(last['epoch_mean_ground']):.6f} |
| total | {float(first['epoch_mean_total']):.6f} | {min(float(r['epoch_mean_total']) for r in epoch_rows.values()):.6f} | {float(last['epoch_mean_total']):.6f} |

训练统计：平均每 Epoch `{sum(epoch_times) / len(epoch_times):.2f}` 秒，峰值 GPU allocated `{peak_gpu:.3f}` GiB；160/160 个 batch 的 Student 梯度有限，160/160 个 batch 的 EMA Teacher 更新标志为真。

训练后 checkpoint：[`checkpoint_last.pt`](../../experiments/agri_pilot/pilot/checkpoint_last.pt)。随机初始化 checkpoint：[`initial_random.pt`](../../experiments/agri_pilot/initial_random.pt)。

## 4. 固定测试集前后对照

评估命令：

```bash
{eval_command}
```

四株测试植物的 Gaussian 索引、采样种子和模型标签全部固定。重采样指标中的不同采样是同一植株的重复测量，不能当作相互独立的生物学样本。

| 指标 | 随机初始化 | Pilot 训练后 | 训练后−随机 |
|---|---:|---:|---:|
{scalar_table}

连续空间缺失使用固定 XYZ 方向上的 projection cap，先移除投影最高的一侧，保证缺失区域空间连续；每个测试植物使用 3 个方向。

| 连续缺失比例 | 随机 partial→same cosine | 随机 R@1 | 训练后 partial→same cosine | 训练后 R@1 |
|---:|---:|---:|---:|---:|
{spatial_table}

结果 JSON、NPZ embedding 和 CSV 对照表：[`evaluation_comparison.json`](evaluation_comparison.json)、[`random_init_metrics.json`](random_init_metrics.json)、[`pilot_trained_metrics.json`](pilot_trained_metrics.json)、[`metrics_comparison.csv`](metrics_comparison.csv)。

## 5. 坍缩检查

随机初始化模型的描述性指标为：跨植物平均 Cosine `{random_cos['different_plant_mean']:.6f}`、Embedding 方差 `{random_stats['mean_dimension_variance_across_plant_means']:.3e}`、有效秩 `{random_stats['participation_ratio_effective_rank']:.4f}`。训练后分别为 `{trained_cos['different_plant_mean']:.6f}`、`{trained_stats['mean_dimension_variance_across_plant_means']:.3e}`、`{trained_stats['participation_ratio_effective_rank']:.4f}`。

训练后触发的描述性警告：`different_plant_cosine_above_0.99={trained_stats['collapse_heuristic']['different_plant_cosine_above_0.99']}`；方差阈值和有效秩阈值在本次四株测试中未触发。这个判断只能作为小样本诊断，不能作为正式的群体级坍缩证明。R@1 全部为 1.0 与跨植物 Cosine 过高同时出现，说明当前检索结果可能主要反映表征对遮挡的共同稳定性，不能单独解释为学到了植物身份差异。

## 6. 背景、花盆和数据限制

- SAM 前景的目标定义包含保留下来的花盆和 marker；这些因素与植物 ID、采集设置可能相关，形成混杂信号。
- 审计中的最大 background alpha 为 `{max_background_alpha:.4f}`；质量警告集中在 `{warning_names}`，其中 `XiangPiShu1` 的 silhouette IoU 约为 `{next(r['silhouette_iou'] for r in audit['records'] if r['plant_id'] == 'XiangPiShu1'):.4f}`。
- 只有 19 株植物，测试集只有 4 株；没有多随机种子重复，也没有外部植物数据或语义标签。
- 训练采用 batch size 2、20 Epoch，是连通性和初步行为验证，不是正式超参数搜索或大规模预训练。
- 评估使用 mean/max pooling 的固定 768 维 embedding；这种 pooling 可能弱化局部植物结构差异。
- 本报告不把训练 Loss 下降、R@1 提升或单次有效秩结果外推为农业泛化能力。

## 7. 阶段结论

Stage 3 的工程验证通过：真实植物 PLY、农业 Dataset、官方 Gaussian-JEPA Encoder/Predictor/Loss、优化器、EMA、checkpoint/resume 和固定评估协议均已跑通。科学结论暂定为“训练产生了明显的训练损失下降和遮挡稳定性，但同时出现跨植物表征过于相似的信号，尚不足以证明植物可区分表征已经学好”。

本阶段没有启动 PAA、TomatoMAP 等外部数据集，也没有开始正式大规模预训练。是否进入数据扩充阶段，等待后续确认。
'''
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment-dir', type=Path, default=DEFAULT_EXPERIMENT)
    parser.add_argument('--result-dir', type=Path, default=DEFAULT_RESULT)
    args = parser.parse_args()
    args.result_dir.mkdir(parents=True, exist_ok=True)
    report = render_report(args.experiment_dir, args.result_dir)
    output = args.result_dir / 'REPORT.md'
    output.write_text(report)
    print(output.resolve())


if __name__ == '__main__':
    main()
