# Stage 3.5-A：真实植物表征与检索协议诊断

本报告仅使用已有的真实植物 PLY 和冻结 checkpoint 做推理。没有训练、修复模型或进入 Stage 3.5-B。

## 数据、模型与复现

- 独立测试植物（主分析，n=4）：DouBanLv2, XianKeLai1, XianKeLai2, XiangPiShu2。
- 原训练植物（15 株，仅作为辅助检索的额外候选）：CaoMei1, CaoMei2, ChangShouHua1, ChangShouHua2, ChangShouHua3, DouBanLv1, DouBanLv3, HongZhang, KongQueZhuYu, WanNianQing1, WanNianQing2, WangWenCao1, WangWenCao2, XianKeLai3, XiangPiShu1。
- 每株完整输入固定 3 组 1024 Gaussian，种子 4101/4102/4103；直接读取 Stage 3 原索引。
- 每株每比例 3 个固定轴方向，比例 30%/55%/70%；每协议每比例 12 次 Query，仍然只有 4 个独立植物。
- 独立 Query 种子为 351000 + 比例编号×100 + 轴编号；Gallery 为种子 4101。所有模型共用同一索引计划。
- 使用完整 PLY 的 Stage 3 `pretrain` 归一化；部分输入沿用完整资产坐标和属性尺度，不做部分输入重新归一化。
- 官方 Student Encoder 参数数：21,826,432。Encoder、Predictor、Loss 和 checkpoint 均未修改。

| 模型 | checkpoint | SHA256 |
| --- | --- | --- |
| random_init | `/fj/AgriGaussian-JEPA/Gaussian-JEPA/experiments/agri_pilot/initial_random.pt` | `99f8c18fca8dfae92586b30d4fe8c413e9c44a2e2f5033045a5fe48c55064e8e` |
| pilot_trained | `/fj/AgriGaussian-JEPA/Gaussian-JEPA/experiments/agri_pilot/pilot/checkpoint_last.pt` | `2f0c5b372db37d40308640f0c89a1cc9327ca037a4c94e26a21ff55b6f3c065c` |

运行命令（在 `/fj/AgriGaussian-JEPA/Gaussian-JEPA` 下）：

```bash
/home/test/biosoft/enter/envs/gaussian_jepa/bin/python -m agri.plant_gs.jepa.diagnose_stage35a --device cuda:0

# 独立核对已保存的真实植物特征、索引、检索分数及防覆盖行为
/home/test/biosoft/enter/envs/gaussian_jepa/bin/python -m agri.plant_gs.jepa.verify_stage35a --input results/agri_diagnostics/stage_3_5a

# 新进程完整复现（该输出目录也必须尚不存在）与逐文件/逐数组核对
/home/test/biosoft/enter/envs/gaussian_jepa/bin/python -m agri.plant_gs.jepa.diagnose_stage35a --device cuda:0 --output /tmp/agri-stage35a-reproduction-20261015
/home/test/biosoft/enter/envs/gaussian_jepa/bin/python -m agri.plant_gs.jepa.verify_stage35a --input results/agri_diagnostics/stage_3_5a --reference /tmp/agri-stage35a-reproduction-20261015

# 仅从已有诊断数值重建本报告和逐植株汇总，不再次推理
/home/test/biosoft/enter/envs/gaussian_jepa/bin/python -m agri.plant_gs.jepa.diagnostic_report --input results/agri_diagnostics/stage_3_5a
```

原目录存在时脚本明确拒绝覆盖。重新运行请加 `--output /绝对路径/全新目录`。全部输入索引见 `sampling_indices.npz`，协议元数据见 `sampling_plan.json`；原始 Token、中心位置、空间匹配索引和未归一化 Pooling 向量见 `*_features.npz`，逐 Query 检索结果见 `retrieval_per_query.csv`。
`diagnostic_metrics.py` 是上述脚本共同使用的内部计算模块，无单独运行命令。
推理数值配置保留 Stage 3 的 cuDNN TF32（True）及矩阵乘 TF32（False），并开启 cuDNN deterministic；全部输入重复推理验证逐元素一致。禁用 cuDNN TF32 的一次回放探针使随机初始化 embedding 改变约 2.9×10⁻⁵，故未将该数值设置混入正式前后对比。

## 1. 原始特征统计

方差和有效秩以 **4 个植株的平均特征** 计算；同株/异株配对使用重复采样，配对数量不能当成独立样本数。`encoder_tokens_spatial_aligned` 是按共同参考中心做位置匹配后展开的 64×384 描述符，不能视作 64 个独立植物。

| 模型 | 特征 | 原始范数均值 | 同株 Cos | 异株 Cos | 跨植株维方差 | 均值范数/平均范数 | 有效秩 | PC1 解释比 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| random_init | mean | 19.593 | 0.996794066 | 0.989329204 | 0.006398 | 0.995722125 | 1.8176 | 70.486% |
| random_init | max | 19.59 | 0.996806245 | 0.989367420 | 0.0063734 | 0.995737711 | 1.8232 | 70.359% |
| random_init | mean_max | 27.706 | 0.996800150 | 0.989348299 | 0.0063857 | 0.995729913 | 1.8204 | 70.421% |
| random_init | encoder_tokens_spatial_aligned | 156.77 | 0.996767783 | 0.989273185 | 0.0064289 | 0.995698843 | 1.8264 | 70.247% |
| random_init | tokenizer_tokens_spatial_aligned | 0.15328 | 0.971373269 | 0.965306162 | 1.1228e-08 | 0.984487057 | 2.9070 | 40.846% |
| pilot_trained | mean | 19.639 | 0.999957777 | 0.999817722 | 0.0001161 | 0.999928124 | 1.5052 | 80.080% |
| pilot_trained | max | 19.85 | 0.999690297 | 0.999576706 | 0.00016782 | 0.999815430 | 2.4759 | 54.658% |
| pilot_trained | mean_max | 27.923 | 0.999822155 | 0.999695312 | 0.00014196 | 0.999870910 | 2.0564 | 65.019% |
| pilot_trained | encoder_tokens_spatial_aligned | 157.21 | 0.998801271 | 0.998524860 | 0.00050989 | 0.999346715 | 2.7892 | 45.982% |
| pilot_trained | tokenizer_tokens_spatial_aligned | 122.99 | 0.958497393 | 0.950815368 | 0.0099544 | 0.977854125 | 2.8846 | 42.274% |

## 2. 原始、L2、中心化特征分别比较

`centered_raw` = 原始特征减去这 4 株、每株 3 次测量的均值；`centered_l2` 再做 L2。中心化严格只用于诊断，未用于检索或声称无泄漏的性能。原始与 L2 的 Cosine 相同：L2 仅改变向量长度，不可能创造公共方向。对原始特征中心化会改变方向；其距离不变。

| 模型 | 特征 | 空间 | 同株 L2 | 异株 L2 | 同株 Cos | 异株 Cos | 跨植株维方差 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| random_init | mean | raw | 1.48105 | 2.7676 | 0.9967941 | 0.9893292 | 0.006398 |
| random_init | mean | l2 | 0.0755932 | 0.141258 | 0.9967941 | 0.9893292 | 1.6667e-05 |
| random_init | mean | centered_raw | 1.48105 | 2.7676 | 0.5908399 | -0.2201581 | 0.006398 |
| random_init | mean | centered_l2 | 0.853542 | 1.54749 | 0.5908399 | -0.2201581 | 0.0018504 |
| random_init | max | raw | 1.47802 | 2.76296 | 0.9968062 | 0.9893674 | 0.0063734 |
| random_init | max | l2 | 0.0754456 | 0.14104 | 0.9968062 | 0.9893674 | 1.6608e-05 |
| random_init | max | centered_raw | 1.47802 | 2.76296 | 0.5912844 | -0.2203885 | 0.0063734 |
| random_init | max | centered_l2 | 0.853305 | 1.54794 | 0.5912844 | -0.2203885 | 0.0018514 |
| random_init | mean_max | raw | 2.09238 | 3.91071 | 0.9968002 | 0.9893483 | 0.0063857 |
| random_init | mean_max | l2 | 0.0755195 | 0.141149 | 0.9968002 | 0.9893483 | 8.3188e-06 |
| random_init | mean_max | centered_raw | 2.09238 | 3.91071 | 0.5910622 | -0.2202730 | 0.0063857 |
| random_init | mean_max | centered_l2 | 0.853424 | 1.54772 | 0.5910622 | -0.2202730 | 0.00092544 |
| random_init | encoder_tokens_spatial_aligned | raw | 11.9102 | 22.2135 | 0.9967678 | 0.9892732 | 0.0064289 |
| random_init | encoder_tokens_spatial_aligned | l2 | 0.0759745 | 0.141698 | 0.9967678 | 0.9892732 | 2.616e-07 |
| random_init | encoder_tokens_spatial_aligned | centered_raw | 11.9102 | 22.2135 | 0.5899419 | -0.2204306 | 0.0064289 |
| random_init | encoder_tokens_spatial_aligned | centered_l2 | 0.855252 | 1.54794 | 0.5899419 | -0.2204306 | 2.8902e-05 |
| pilot_trained | mean | raw | 0.179392 | 0.360439 | 0.9999578 | 0.9998177 | 0.0001161 |
| pilot_trained | mean | l2 | 0.00913451 | 0.0183536 | 0.9999578 | 0.9998177 | 3.0103e-07 |
| pilot_trained | mean | centered_raw | 0.179392 | 0.360439 | 0.6023675 | -0.2313843 | 0.0001161 |
| pilot_trained | mean | centered_l2 | 0.842222 | 1.54962 | 0.6023675 | -0.2313843 | 0.0018873 |
| pilot_trained | max | raw | 0.494459 | 0.575731 | 0.9996903 | 0.9995767 | 0.00016782 |
| pilot_trained | max | l2 | 0.0248393 | 0.0289221 | 0.9996903 | 0.9995767 | 4.2349e-07 |
| pilot_trained | max | centered_raw | 0.494459 | 0.575731 | 0.1474109 | -0.1419505 | 0.00016782 |
| pilot_trained | max | centered_l2 | 1.30336 | 1.50908 | 0.1474109 | -0.1419505 | 0.0011202 |
| pilot_trained | mean_max | raw | 0.526316 | 0.68177 | 0.9998222 | 0.9996953 | 0.00014196 |
| pilot_trained | mean_max | l2 | 0.0188256 | 0.0243912 | 0.9998222 | 0.9996953 | 1.8176e-07 |
| pilot_trained | mean_max | centered_raw | 0.526316 | 0.68177 | 0.2731365 | -0.1680091 | 0.00014196 |
| pilot_trained | mean_max | centered_l2 | 1.19762 | 1.52353 | 0.2731365 | -0.1680091 | 0.00066742 |
| pilot_trained | encoder_tokens_spatial_aligned | raw | 7.69694 | 8.53179 | 0.9988013 | 0.9985249 | 0.00050989 |
| pilot_trained | encoder_tokens_spatial_aligned | l2 | 0.0489582 | 0.0542685 | 0.9988013 | 0.9985249 | 2.063e-08 |
| pilot_trained | encoder_tokens_spatial_aligned | centered_raw | 7.69694 | 8.53179 | 0.0786904 | -0.1284599 | 0.00050989 |
| pilot_trained | encoder_tokens_spatial_aligned | centered_l2 | 1.35644 | 1.50172 | 0.0786904 | -0.1284599 | 1.5694e-05 |

局部 Token 的主要比较采用每对输入中心之间的几何最小代价一对一匹配（Hungarian），不使用 Token 编号对应。还保留共同参考匹配的协方差结果，以便描述 PCA/秩。跨植物仅有归一化坐标，没有语义配准或姿态配准；空间匹配不是叶片同源对应，匹配距离写入逐对结果，不能把残差全部解释为语义信息。

`tokenizer_tokens` 是通过只读 forward hook 得到的 Transformer 之前的局部 PointNet 输出，用作定位参考，不更改官方 forward。

| 模型 | 阶段 | 单 Token 范数 | 公共方向比 | 同株 Token Cos | 异株 Token Cos | 中心化同株 Cos | 中心化异株 Cos | 株内空间 Token 方差 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| random_init | encoder_tokens | 19.596 | 0.99555598 | 0.99677953 | 0.98928233 | 0.61067 | -0.17306 | 0.00033369 |
| random_init | tokenizer_tokens | 0.019149 | 0.98265296 | 0.97269217 | 0.96644756 | 0.19102 | 0.02214 | 3.134e-08 |
| pilot_trained | encoder_tokens | 19.652 | 0.99925531 | 0.99881774 | 0.99852931 | 0.19453 | 0.01200 | 0.001353 |
| pilot_trained | tokenizer_tokens | 15.356 | 0.97601534 | 0.96085003 | 0.95289057 | 0.16352 | 0.00963 | 0.028418 |

最终 Token 异株 Cos：0.989282328 → 0.998529311。
最终 Token 株内空间方差：0.000333693 → 0.00135295。
Mean+Max L2 后跨植株方差：8.31882e-06 → 1.81762e-07（训练后/训练前 = 0.0218）。
Mean+Max 原始特征公共方向比为 0.999870910。

## 3. 遮挡检索：共享与独立协议

四种协议均从同一份真实 PLY 读取属性，不生成替代数据：

1. `old_shared`：完全复现 Stage 3；1024 Gallery 上裁切，Query 分别为 717/461/307 个原语，与正确 Gallery 的 Query 索引重叠率 100%。
2. `disjoint_same_cap_same_count`：空间裁切阈值与旧 Query 相同，点数相同，从完整 PLY 的对应半空间采样，排除 Gallery 索引。用于检查共享原语/重复采样敏感性。
3. `independent_full_cap_matched_count`：在完整 PLY 上精确裁掉对应比例，排除 Gallery 索引，点数保持 717/461/307。
4. `independent_full_cap_1024`：要求的新主协议；完整 PLY 上精确裁切，排除 Gallery 索引，独立采样 1024 个原语。

所有非旧协议实测索引重叠率为 0%。旧协议和新主协议同时改变裁切总体、采样点数、共享情况，不能只根据二者差异归因；两个中间控制用于拆分这些影响。

下面是原 Stage 3 使用的 Mean+Max，Cosine 在 CPU float64 下从原始 Pooling 向量计算。完整 Mean/Max/Mean+Max 结果见 CSV。`4` 候选是独立测试主分析；`19` 包含训练见过的 15 株，仅是候选数敏感性分析，Query 仍只有未见过的 4 株。

Margin = 正确身份 Cos − 最强错误身份 Cos；Top1−Top2 不区分命中与否，二者不能混用。每行汇总 12 次相关 Query 测量。

| 模型 | 候选数 | 协议 | 缺失 | R@1 | 正确 Cos | 最强错误 Cos | Margin 均值 | Margin 最小 | Top1−Top2 | Query 重叠 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| random_init | 4 | old_shared | 30% | 0.417 | 0.945923319 | 0.946545366 | -0.000622 | -0.02251 | 0.00907 | 100% |
| random_init | 19 | old_shared | 30% | 0.167 | 0.945923319 | 0.959057927 | -0.01313 | -0.02931 | 0.006629 | 100% |
| random_init | 4 | disjoint_same_cap_same_count | 30% | 0.417 | 0.944025711 | 0.946980891 | -0.002955 | -0.03102 | 0.008885 | 0% |
| random_init | 19 | disjoint_same_cap_same_count | 30% | 0.083 | 0.944025711 | 0.958793327 | -0.01477 | -0.03357 | 0.006213 | 0% |
| random_init | 4 | independent_full_cap_matched_count | 30% | 0.417 | 0.944938537 | 0.946877119 | -0.001939 | -0.02651 | 0.008793 | 0% |
| random_init | 19 | independent_full_cap_matched_count | 30% | 0.083 | 0.944938537 | 0.958803271 | -0.01386 | -0.02821 | 0.006055 | 0% |
| random_init | 4 | independent_full_cap_1024 | 30% | 0.417 | 0.944681821 | 0.947345941 | -0.002664 | -0.03188 | 0.009163 | 0% |
| random_init | 19 | independent_full_cap_1024 | 30% | 0.083 | 0.944681821 | 0.959022992 | -0.01434 | -0.03188 | 0.006094 | 0% |
| random_init | 4 | old_shared | 55% | 0.333 | 0.858413739 | 0.866486799 | -0.008073 | -0.05318 | 0.01224 | 100% |
| random_init | 19 | old_shared | 55% | 0.083 | 0.858413739 | 0.882828231 | -0.02441 | -0.05606 | 0.008082 | 100% |
| random_init | 4 | disjoint_same_cap_same_count | 55% | 0.417 | 0.859382787 | 0.868001705 | -0.008619 | -0.05372 | 0.01164 | 0% |
| random_init | 19 | disjoint_same_cap_same_count | 55% | 0.083 | 0.859382787 | 0.884035137 | -0.02465 | -0.05457 | 0.007229 | 0% |
| random_init | 4 | independent_full_cap_matched_count | 55% | 0.417 | 0.857823422 | 0.865983176 | -0.00816 | -0.05306 | 0.01176 | 0% |
| random_init | 19 | independent_full_cap_matched_count | 55% | 0.083 | 0.857823422 | 0.882774018 | -0.02495 | -0.05728 | 0.007847 | 0% |
| random_init | 4 | independent_full_cap_1024 | 55% | 0.333 | 0.862259462 | 0.870945623 | -0.008686 | -0.05269 | 0.0122 | 0% |
| random_init | 19 | independent_full_cap_1024 | 55% | 0.000 | 0.862259462 | 0.888073380 | -0.02581 | -0.05427 | 0.007973 | 0% |
| random_init | 4 | old_shared | 70% | 0.333 | 0.803260543 | 0.813830290 | -0.01057 | -0.06185 | 0.01354 | 100% |
| random_init | 19 | old_shared | 70% | 0.000 | 0.803260543 | 0.833642664 | -0.03038 | -0.07036 | 0.008286 | 100% |
| random_init | 4 | disjoint_same_cap_same_count | 70% | 0.333 | 0.805749985 | 0.816384697 | -0.01063 | -0.06336 | 0.01387 | 0% |
| random_init | 19 | disjoint_same_cap_same_count | 70% | 0.000 | 0.805749985 | 0.836024139 | -0.03027 | -0.06937 | 0.007465 | 0% |
| random_init | 4 | independent_full_cap_matched_count | 70% | 0.333 | 0.805108193 | 0.815269834 | -0.01016 | -0.06184 | 0.0132 | 0% |
| random_init | 19 | independent_full_cap_matched_count | 70% | 0.000 | 0.805108193 | 0.835414343 | -0.03031 | -0.07116 | 0.008244 | 0% |
| random_init | 4 | independent_full_cap_1024 | 70% | 0.333 | 0.812347988 | 0.822714361 | -0.01037 | -0.06319 | 0.0142 | 0% |
| random_init | 19 | independent_full_cap_1024 | 70% | 0.000 | 0.812347988 | 0.843926337 | -0.03158 | -0.07125 | 0.008142 | 0% |
| pilot_trained | 4 | old_shared | 30% | 1.000 | 0.999874747 | 0.999687468 | 0.0001873 | 0.0001044 | 0.0001873 | 100% |
| pilot_trained | 19 | old_shared | 30% | 1.000 | 0.999874747 | 0.999770599 | 0.0001041 | 5.698e-05 | 0.0001041 | 100% |
| pilot_trained | 4 | disjoint_same_cap_same_count | 30% | 0.917 | 0.999785496 | 0.999721045 | 6.445e-05 | -5.509e-05 | 7.212e-05 | 0% |
| pilot_trained | 19 | disjoint_same_cap_same_count | 30% | 0.333 | 0.999785496 | 0.999791376 | -5.88e-06 | -7.396e-05 | 1.715e-05 | 0% |
| pilot_trained | 4 | independent_full_cap_matched_count | 30% | 0.833 | 0.999772537 | 0.999721992 | 5.055e-05 | -1.012e-05 | 5.338e-05 | 0% |
| pilot_trained | 19 | independent_full_cap_matched_count | 30% | 0.333 | 0.999772537 | 0.999787129 | -1.459e-05 | -6.85e-05 | 1.644e-05 | 0% |
| pilot_trained | 4 | independent_full_cap_1024 | 30% | 0.833 | 0.999789307 | 0.999733228 | 5.608e-05 | -2.585e-05 | 6.357e-05 | 0% |
| pilot_trained | 19 | independent_full_cap_1024 | 30% | 0.417 | 0.999789307 | 0.999796448 | -7.141e-06 | -5.655e-05 | 1.45e-05 | 0% |
| pilot_trained | 4 | old_shared | 55% | 1.000 | 0.999790969 | 0.999612633 | 0.0001783 | 0.0001027 | 0.0001783 | 100% |
| pilot_trained | 19 | old_shared | 55% | 0.917 | 0.999790969 | 0.999723249 | 6.772e-05 | -3.801e-06 | 6.835e-05 | 100% |
| pilot_trained | 4 | disjoint_same_cap_same_count | 55% | 0.917 | 0.999730229 | 0.999655893 | 7.434e-05 | -1.237e-05 | 7.64e-05 | 0% |
| pilot_trained | 19 | disjoint_same_cap_same_count | 55% | 0.333 | 0.999730229 | 0.999750560 | -2.033e-05 | -7.461e-05 | 1.733e-05 | 0% |
| pilot_trained | 4 | independent_full_cap_matched_count | 55% | 0.667 | 0.999723839 | 0.999676044 | 4.78e-05 | -5.553e-05 | 6.941e-05 | 0% |
| pilot_trained | 19 | independent_full_cap_matched_count | 55% | 0.333 | 0.999723839 | 0.999750453 | -2.661e-05 | -7.338e-05 | 1.183e-05 | 0% |
| pilot_trained | 4 | independent_full_cap_1024 | 55% | 0.833 | 0.999769013 | 0.999712025 | 5.699e-05 | -4.261e-05 | 6.659e-05 | 0% |
| pilot_trained | 19 | independent_full_cap_1024 | 55% | 0.333 | 0.999769013 | 0.999785603 | -1.659e-05 | -7.121e-05 | 2.465e-05 | 0% |
| pilot_trained | 4 | old_shared | 70% | 1.000 | 0.999712574 | 0.999547856 | 0.0001647 | 2.275e-05 | 0.0001647 | 100% |
| pilot_trained | 19 | old_shared | 70% | 0.667 | 0.999712574 | 0.999670896 | 4.168e-05 | -8.658e-05 | 6.914e-05 | 100% |
| pilot_trained | 4 | disjoint_same_cap_same_count | 70% | 0.750 | 0.999666455 | 0.999577869 | 8.859e-05 | -5.521e-05 | 0.0001001 | 0% |
| pilot_trained | 19 | disjoint_same_cap_same_count | 70% | 0.250 | 0.999666455 | 0.999696786 | -3.033e-05 | -0.0001518 | 3.32e-05 | 0% |
| pilot_trained | 4 | independent_full_cap_matched_count | 70% | 1.000 | 0.999667441 | 0.999561723 | 0.0001057 | 3.831e-06 | 0.0001057 | 0% |
| pilot_trained | 19 | independent_full_cap_matched_count | 70% | 0.250 | 0.999667441 | 0.999681179 | -1.374e-05 | -8.512e-05 | 3.721e-05 | 0% |
| pilot_trained | 4 | independent_full_cap_1024 | 70% | 0.750 | 0.999773186 | 0.999695486 | 7.77e-05 | -4.453e-05 | 9.022e-05 | 0% |
| pilot_trained | 19 | independent_full_cap_1024 | 70% | 0.417 | 0.999773186 | 0.999783200 | -1.001e-05 | -0.0001173 | 1.986e-05 | 0% |

## 4. 证据与问题定位

### A. Pooling 放大共同方向，但不能解释全部集中现象

训练后单个最终 Token 的公共方向比为 0.999255311，
Mean+Max 后为 0.999870910。
几何匹配后的局部 Token 异株 Cos 已达到 0.998529311，
因此共同方向在 Pooling 前已经存在，不能把全部现象归因于 Pooling。
L2 前后同/异株 Cos 数值一致（`feature_statistics.csv` 的 raw/l2 行，以及 `verify_stage35a.py` 的不变性检查），所以 L2 不是高 Cos 的成因。

不同 Pooling 的更具体问题是 Max 分支对独立重采样更不稳定。训练后同株重复采样的原始 L2 距离：Mean 0.179392，Max 0.494459（均为 384 维、原始范数约 19.6–19.9，见 `pooling_comparison.csv`）。测试均值中心化后，同株 Cos 为 Mean 0.602368、Max 0.147411、拼接 0.273137；这只是描述性残差分析，未用于正式检索。
新独立协议下，三个比例合计 36 次测量：4 候选时 Mean 命中 33/36，Max 与拼接均为 29/36；19 候选辅助分析 Mean 24/36、Max 4/36、拼接 14/36。这支持优先检查 Max 读出分支的采样敏感性，但 36 次不是 36 个独立植物，不能据此完成正式方法选择或泛化认证。

### B. Encoder 存在方向集中；没有完全坍缩证据

训练前到训练后，最终 Token 的异株 Cos 从 0.989282328 升至 0.998529311；同株−异株差距从
0.0074972 缩至
0.000288425。
参考中心对齐后的局部描述符跨植物原始维方差从 0.00642891 降到 0.000509887（约 12.6 倍下降）；Mean+Max 原始方差则约下降 45 倍，L2 方差约下降 45.8 倍。
这些是 Encoder 输出的身份相关变化缩小、聚合后进一步集中的证据。

但局部空间波动没有消失：最终 Token 的株内维方差由 0.000333693 增至 0.00135295（约 4.05 倍），不能写成“所有局部 Token 方差都下降”。
按对配对、减去测试 Token 总均值后，训练后局部 Token 同株 Cos 0.194528、异株 0.012004，仍保留有限身份相关残差；原始 Mean+Max 仍有约 1.59% 的相对残差范数。
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

| 植物 | 30% 命中 | 55% 命中 | 70% 命中 | 各比例 Margin 均值 |
| --- | --- | --- | --- | --- |
| DouBanLv2 | 3/3 | 3/3 | 3/3 | 2.55e-05; 3.4e-05; 6.52e-05 |
| XianKeLai1 | 3/3 | 2/3 | 3/3 | 0.000135; 0.000113; 0.000214 |
| XianKeLai2 | 3/3 | 3/3 | 2/3 | 7e-05; 7.61e-05; 1.98e-05 |
| XiangPiShu2 | 1/3 | 2/3 | 1/3 | -6.11e-06; 4.96e-06; 1.18e-05 |

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

- 对 86 个原文件逐一做前后 SHA256/字节数比较，均未变；包括全部 Stage 3 结果、checkpoint、模型源文件和 19 份 PLY。
- random_init：171 组输入全部重复推理；最大绝对差 0。
- pilot_trained：171 组输入全部重复推理；最大绝对差 0。
- Stage 3 回放 random_init：完整 embedding 最大误差 8.33e-09，旧部分输入最大误差 9.36e-09。
- Stage 3 回放 pilot_trained：完整 embedding 最大误差 7.66e-09，旧部分输入最大误差 9.08e-09。
- 新进程完整复现通过：6 份 CSV 字节一致，5 份 NPZ 内 1064 个数组逐元素完全一致（包含全部 Token、Pooling、索引和检索特征）。
- 索引计划重新计算完全相同；每个新 Query 的原始 PLY 索引与 Gallery 交集为空。
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
