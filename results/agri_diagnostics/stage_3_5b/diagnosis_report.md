# Stage 3.5-B：冻结 Encoder 的全局 Pooling 单变量验证

<!-- Historical output: interpreter paths below are the ones actually used at the time. The project now uses uv (env/bootstrap.sh); re-run with $AGRI_PY_GS / $AGRI_PY_JEPA. -->


固定 checkpoint：`/fj/AgriGaussian-JEPA/Gaussian-JEPA/experiments/agri_pilot/pilot/checkpoint_last.pt`；SHA256 `2f0c5b372db37d40308640f0c89a1cc9327ca037a4c94e26a21ff55b6f3c065c`。
协议在推理前固定于 `2026-10-15T08:43:07.767599+00:00`；SHA256 `080136e4fe121617de0b65c03f0dae745d32618fd07c37f7c4820b80dadf2cb0`。

## 数据支持的结论

按预先写入 protocol.json 的保守描述性规则，Mean 没有满足稳定 R@1 改善的全部条件；保留原始 Mean+Max。
该规则不是显著性检验，也不证明对新植株泛化；本轮不改变任何生产读出配置。

| 指标 | Mean+Max | Mean | 差值 (Mean − Mean+Max) |
| --- | --- | --- | --- |
| R@1 30% | 32/36 (88.89%) | 35/36 (97.22%) | +8.33% |
| R@1 55% | 32/36 (88.89%) | 31/36 (86.11%) | -2.78% |
| R@1 70% | 33/36 (91.67%) | 32/36 (88.89%) | -2.78% |
| R@1 overall | 97/108 (89.81%) | 98/108 (90.74%) | +0.93% |
| Mean Margin | 6.075122286e-05 | 7.156670206e-05 | +1.081547920e-05 |
| Median Margin | 5.103851927e-05 | 5.965109689e-05 | +8.612577615e-06 |
| Minimum Margin | -1.260056074e-04 | -1.534951254e-04 | -2.748951797e-05 |
| 正 Margin 比例 | 89.81% | 90.74% | +0.93% |
| Resampling Drift (L2) | 1.941760323e-02 | 1.058377905e-02 | -8.833824182e-03 |
| Resampling Drift (1-cos) | 1.891887265e-04 | 5.687662250e-05 | -1.323121040e-04 |

逐 Query 命中结果：Mean 胜 6，负 5，平 97（均正确 92，均错误 5）。
Margin 逐 Query 胜 58，负 50，平 0，绝对差 ≤ 1e-12 视为 Margin 打平。

R@1、Margin 和漂移分别解释；更小的归一化漂移也可能受到强公共方向压缩的影响，不能单独视作身份信息增加。

## 逐株与遮挡比例证据

| 植株 | Mean+Max | Mean | R@1 差值 | Mean+Max Mean Margin | Mean Mean Margin |
| --- | --- | --- | --- | --- | --- |
| DouBanLv2 | 27/27 (100.00%) | 26/27 (96.30%) | -3.70% | 5.2047428e-05 | 5.1390763e-05 |
| XianKeLai1 | 24/27 (88.89%) | 27/27 (100.00%) | +11.11% | 1.1749881e-04 | 1.7222738e-04 |
| XianKeLai2 | 20/27 (74.07%) | 21/27 (77.78%) | +3.70% | 2.4978997e-05 | 1.7153608e-05 |
| XiangPiShu2 | 26/27 (96.30%) | 24/27 (88.89%) | -7.41% | 4.8479652e-05 | 4.5495060e-05 |

| 植株 / 缺失比例 | Mean+Max | Mean | R@1 差值 | Mean Margin 差值 |
| --- | --- | --- | --- | --- |
| DouBanLv2/0.3 | 9/9 (100.00%) | 9/9 (100.00%) | +0.00% | +2.0715079e-06 |
| DouBanLv2/0.55 | 9/9 (100.00%) | 9/9 (100.00%) | +0.00% | -6.5428523e-07 |
| DouBanLv2/0.7 | 9/9 (100.00%) | 8/9 (88.89%) | -11.11% | -3.3872182e-06 |
| XianKeLai1/0.3 | 8/9 (88.89%) | 9/9 (100.00%) | +11.11% | +6.5940555e-05 |
| XianKeLai1/0.55 | 7/9 (77.78%) | 9/9 (100.00%) | +22.22% | +4.7374366e-05 |
| XianKeLai1/0.7 | 9/9 (100.00%) | 9/9 (100.00%) | +0.00% | +5.0870770e-05 |
| XianKeLai2/0.3 | 6/9 (66.67%) | 8/9 (88.89%) | +22.22% | +7.7285364e-07 |
| XianKeLai2/0.55 | 8/9 (88.89%) | 6/9 (66.67%) | -22.22% | -2.0548713e-05 |
| XianKeLai2/0.7 | 6/9 (66.67%) | 7/9 (77.78%) | +11.11% | -3.7003090e-06 |
| XiangPiShu2/0.3 | 9/9 (100.00%) | 9/9 (100.00%) | +0.00% | -5.8857789e-06 |
| XiangPiShu2/0.55 | 8/9 (88.89%) | 7/9 (77.78%) | -11.11% | -3.1938481e-06 |
| XiangPiShu2/0.7 | 9/9 (100.00%) | 8/9 (88.89%) | -11.11% | +1.2584984e-07 |

## 每组新种子与方向

| 方案 | Gallery seed | 完整重采样 seed | Query seeds (30%,55%,70%; XYZ) |
| --- | --- | --- | --- |
| scheme_1 | 35201001 | 35201002 | [[35201100, 35201101, 35201102], [35201110, 35201111, 35201112], [35201120, 35201121, 35201122]] |
| scheme_2 | 35202001 | 35202002 | [[35202100, 35202101, 35202102], [35202110, 35202111, 35202112], [35202120, 35202121, 35202122]] |
| scheme_3 | 35203001 | 35203002 | [[35203100, 35203101, 35203102], [35203110, 35203111, 35203112], [35203120, 35203121, 35203122]] |

| 方案 / 比例 | Mean+Max | Mean | R@1 差值 | Mean Margin 差值 |
| --- | --- | --- | --- | --- |
| scheme_1 | 31/36 (86.11%) | 35/36 (97.22%) | +11.11% | +2.0181318e-05 |
| scheme_1/0.3 | 10/12 (83.33%) | 12/12 (100.00%) | +16.67% | +1.9701983e-05 |
| scheme_1/0.55 | 10/12 (83.33%) | 11/12 (91.67%) | +8.33% | +1.9966566e-05 |
| scheme_1/0.7 | 11/12 (91.67%) | 12/12 (100.00%) | +8.33% | +2.0875405e-05 |
| scheme_2 | 34/36 (94.44%) | 31/36 (86.11%) | -8.33% | +6.4147397e-06 |
| scheme_2/0.3 | 12/12 (100.00%) | 12/12 (100.00%) | +0.00% | +1.8396509e-05 |
| scheme_2/0.55 | 11/12 (91.67%) | 10/12 (83.33%) | -8.33% | +4.9491233e-07 |
| scheme_2/0.7 | 11/12 (91.67%) | 9/12 (75.00%) | -16.67% | +3.5279754e-07 |
| scheme_3 | 32/36 (88.89%) | 32/36 (88.89%) | +0.00% | +5.8503801e-06 |
| scheme_3/0.3 | 10/12 (83.33%) | 11/12 (91.67%) | +8.33% | +9.0758614e-06 |
| scheme_3/0.55 | 11/12 (91.67%) | 10/12 (83.33%) | -8.33% | -3.2283380e-06 |
| scheme_3/0.7 | 11/12 (91.67%) | 11/12 (91.67%) | +0.00% | +1.1703617e-05 |
| X | 33/36 (91.67%) | 35/36 (97.22%) | +5.56% | +6.6461270e-06 |
| Y | 34/36 (94.44%) | 31/36 (86.11%) | -8.33% | +1.2731678e-05 |
| Z | 30/36 (83.33%) | 32/36 (88.89%) | +5.56% | +1.3068632e-05 |

逐株 × 方案 × 遮挡比例的全部结果见 `per_plant_comparison.csv`；每个 Query 的预测、两个 Cosine、Margin、Top1−Top2 间隔、种子和重叠率见 `retrieval_per_query.csv`。

## 完整植株重采样漂移

每方案每株 1 对完整重采样，共 12 对；Probe 从完整资产采样，排除该 Gallery 的原语 ID，两者均为 1024。原始种子、原始 PLY 路径和索引已保存。

| 植株 / 方案 | Mean+Max L2 drift | Mean L2 drift | 差值 |
| --- | --- | --- | --- |
| DouBanLv2/scheme_1 | 1.83025198e-02 | 9.32432703e-03 | -8.97819278e-03 |
| XianKeLai1/scheme_1 | 2.15474002e-02 | 1.15644405e-02 | -9.98295976e-03 |
| XianKeLai2/scheme_1 | 1.85668741e-02 | 9.37864399e-03 | -9.18823010e-03 |
| XiangPiShu2/scheme_1 | 1.80289349e-02 | 1.09361987e-02 | -7.09273616e-03 |
| DouBanLv2/scheme_2 | 1.89695282e-02 | 1.04703021e-02 | -8.49922616e-03 |
| XianKeLai1/scheme_2 | 2.17316713e-02 | 1.13947242e-02 | -1.03369471e-02 |
| XianKeLai2/scheme_2 | 1.89867829e-02 | 8.81397758e-03 | -1.01728053e-02 |
| XiangPiShu2/scheme_2 | 1.94783543e-02 | 9.50793379e-03 | -9.97042055e-03 |
| DouBanLv2/scheme_3 | 1.96265755e-02 | 1.14134567e-02 | -8.21311883e-03 |
| XianKeLai1/scheme_3 | 1.95075682e-02 | 1.28630158e-02 | -6.64455239e-03 |
| XianKeLai2/scheme_3 | 2.00321743e-02 | 1.23808754e-02 | -7.65129885e-03 |
| XiangPiShu2/scheme_3 | 1.82328551e-02 | 8.95745294e-03 | -9.27540219e-03 |

## 单变量控制与审计

- 固定 4 株未参与训练的植物：DouBanLv2、XianKeLai1、XianKeLai2、XiangPiShu2。
- 每方案 4 个 Gallery，36 个遮挡 Query；3 方案合计 108 个 Query。所有输入均 1024 × 14，所有 Encoder 输出均 64 × 384。
- 全 PLY 先按原农业加载器的 pretrain 模式归一化，之后沿 +X/+Y/+Z 删除全资产最高投影的连续空间区域，按原 Stage 3.5-A 的稳定排序与 round 规则删除原语。缺失比例指 Gaussian 数量比例，不是体积比例。Query 不重新归一化。
- 各 Query 从保留区域排除正确 Gallery 的原始行 ID 后无放回采样 1024 个。所有 Query/Gallery 索引交集严格为 0；索引身份相同才算重复，未声称去除了空间邻近。
- 每份输入只前向一次，共 132 次。两种读出同时来自同一个 Token tensor；float32 Encoder 和原 torch Mean/Max；读出先转换 float64 再进行 L2、Cosine、Margin 与漂移计算。
- 未加载随机模型作为额外变量；仅冻结同一训练后 student Encoder。模型参数、缓冲区及 checkpoint SHA256 均校验，禁止训练、反向传播、参数更新、中心化、白化或投影。
- 旧种子审计包含 Stage 3/3.5-A 保存的所有 JSON/NPZ seed 字段、源码常量和 Dataset seed+epoch 推导的全部训练/加载器种子。新种子集合与历史集合无交集。完整种子审计和对应源文件 SHA256 见 protocol.json。
- 所有采样索引见 sampling_indices.npz，索引对应的 PLY、空间阈值与实际删除比例见 sampling_plan.json。原始 Token、分组坐标、两种原始池化分量及两种归一化读出见 encoder_features.npz。

| 预先固定的稳定性规则 | 满足 |
| --- | --- |
| overall_r1_higher | True |
| no_scheme_lower | False |
| at_least_two_schemes_higher | False |
| at_least_two_plants_higher | True |
| no_plant_lower | False |
| no_ratio_lower | False |

## 尚不确定的判断与实验限制

- 新种子避免复用 Stage 3.5-A 的具体随机采样，却仍使用已经参与假设形成的相同 4 株；本轮只验证这些植株上的采样稳健性，不能当作全新生物学验证。
- 独立实验单位是植株 n=4。108 个 Query、12 个完整重采样对和 3 个方案均为重复测量。不把它们用于查询级显著性检验；平衡设计的总 R@1 与植株宏平均一致，逐株成对差值均已列出。
- 候选集合只有 4 株。R@1 不表明在更大或更相似的候选集合仍能成功；余弦均接近 1 时还需结合绝对 Margin、最小 Margin 和逐株失败查看。
- 本轮不能解释 Encoder 公共方向的成因，也不能区分训练目标、数据规模和重建属性等潜在来源。读出方式的效果不等于解决 Encoder 表征集中；未重新计算或要求公共方向比值下降。
- 花盆、标签、残留背景沿用原 PLY，未分离语义部件。无法把任何检索收益专门归因于植物器官。
- 零原始索引重叠不能使相邻的高斯或相似的重建纹理成为独立观测；连续单侧 XYZ 遮挡也不覆盖所有现实遮挡或坐标旋转。

## 下一阶段只建议处理一个问题

检索结论对候选集合规模的敏感性；固定模型、读出和采样预算，单独改变候选数量。现有 19 株含 15 株训练见过的植物，只能用于标明 seen/unseen 的辅助诊断，不能当独立泛化测试。
预先固定候选身份、采样、种子和评价集合，仅添加干扰候选，报告每个未见过测试植株的 R@1 和 Margin 随候选数量的变化；如果该阶段要声称新植株泛化，还需另行获得未用于选择读出方案的独立测试植株。本轮未执行该实验，也没有扩充数据或改动默认 Pooling。

## 运行和复现命令

工作目录 `/fj/AgriGaussian-JEPA/Gaussian-JEPA`，分支 `agri/plant-gs-experiments`。

```bash
/home/test/biosoft/enter/envs/gaussian_jepa/bin/python -m agri.plant_gs.jepa.pooling_stage35b --prepare
/home/test/biosoft/enter/envs/gaussian_jepa/bin/python -m agri.plant_gs.jepa.pooling_stage35b --run --device cuda:0
/home/test/biosoft/enter/envs/gaussian_jepa/bin/python -m agri.plant_gs.jepa.verify_stage35b --output results/agri_diagnostics/stage_3_5b
/home/test/biosoft/enter/envs/gaussian_jepa/bin/python -m agri.plant_gs.jepa.pooling_stage35b --replay-from results/agri_diagnostics/stage_3_5b --output /tmp/agri-stage35b-reproduction-20261015 --device cuda:0
/home/test/biosoft/enter/envs/gaussian_jepa/bin/python -m agri.plant_gs.jepa.verify_stage35b --output /tmp/agri-stage35b-reproduction-20261015
/home/test/biosoft/enter/envs/gaussian_jepa/bin/python -m agri.plant_gs.jepa.verify_stage35b --output results/agri_diagnostics/stage_3_5b --reference /tmp/agri-stage35b-reproduction-20261015
```

prepare 只用于尚不存在的目录；run 只接受仅含固定协议与保护清单的目录，拒绝覆盖已有结果。复现采用原 protocol.json，复制到新目录后重新加载真实 PLY 和 checkpoint，重新采样、重新推理。verification.json 记录有限性、索引约束、同 Token 读出、独立分数重算、产物保护和逐数组复现检查。只有该文件与 run_completion.json 均通过，才把本轮视为完成。

## 数值校验修订记录

首次验证的 float64 单位范数阈值 1e-15 过严；768 维向量实际最大误差 1.7763568394002505e-15。仅把验证阈值改为 32 × float64 epsilon （约 7.11e-15），并记录源代码哈希修订。原始 protocol.json 字节、全部种子、采样索引、Token、两种读出及分数均未改变；评分/统计模块 pooling_stage35b_report.py 的哈希仍与预注册一致。validation_amendment.json 和 verification.json 保存修订与实际误差。
