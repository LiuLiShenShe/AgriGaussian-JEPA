# Gaussian-JEPA 复现环境审计

审计日期：2026-10-15（Asia/Shanghai）  
工作目录：`/fj/AgriGaussian-JEPA/Gaussian-JEPA`  
分支：`agri/plant-gs-experiments`  
当前提交：`af2f510ee1e7f284ffb3f5a2ec6c4605b55bbcc3`  
官方基线提交：`e73f9b66a2cc50e0a3090f920cab8baef0571d60`

## 结论

代码环境具备进行原版模型小批量验证的条件。CUDA 扩展的现有安装已通过最小算子调用和官方合成数据 smoke test；单卡合成 batch 256 的峰值 PyTorch 显存分配为 19.98 GiB。完整复现目前**不能启动**：ShapeSplatsV1 权限未获批，服务器没有数据，公开 checkpoint 不随仓库发布，完整数据展开后的实际磁盘需求也无法在无访问权限时核实。

## 主机与 GPU

以下为采样时状态，不代表独占资源保证：

| 项目 | 观测 |
|---|---|
| GPU | 2 × NVIDIA RTX A6000，每张 49,140 MiB，Compute Capability 8.6 |
| GPU 0 | 47,982 MiB free，利用率 0% |
| GPU 1 | 46,799 MiB free，利用率 94%；有其他 GPU 作业正在运行 |
| CPU | Intel Xeon Gold 6226R；2 sockets × 16 cores × 2 threads，共 64 logical CPUs |
| RAM | 134.7 GB total，66.9 GB available |
| Swap | 8.59 GB total，8.52 GB 已使用 |
| 文件系统 | `/fj` 所在文件系统约 2.01 TB，501.14 GB 可用 |

完整配置的两卡 DataParallel 候选分配为每卡约 128 样本。单卡 batch 256 的合成模型 smoke test 峰值分配 19.98 GiB，batch 128 为 10.19 GiB；两卡真实训练尚未实测，不能把合成输入峰值当作正式训练峰值保证。显存容量看起来可行，但 GPU 1 当前高利用率，需等待空闲或获得独占窗口。主机 RAM 尚可用，但 swap 接近耗尽，16 个默认 DataLoader worker 的 I/O/RAM 行为需观察。

## 软件与 CUDA

| 项目 | 观测 |
|---|---|
| 环境 | Conda `gaussian_jepa` |
| Python | 3.9.23 |
| PyTorch / torchvision | 2.0.1 / 0.15.2 |
| PyTorch CUDA build | 11.8；CUDA 可用，检测到 2 GPU |
| Conda 环境 nvcc | 11.8.89，位于 `$CONDA_PREFIX/bin/nvcc` |
| PyTorch `CUDA_HOME` | 指向 `gaussian_jepa` 环境 |
| shell `PATH` 默认解析的 nvcc | `/usr/local/cuda/bin/nvcc`，版本 12.6；系统同时存在此版本 |
| CUDA 扩展 | `pointnet2_ops`、`knn_cuda` 可导入 |

对 GPU 0 执行了极小的 FPS、gather、KNN CUDA 算子调用，输出形状正确且距离有限。另执行官方 `tools/smoke_test_release.py` 的合成数据前向/反向：batch 128 和 batch 256 均 PASS，分别报告 10.19 GiB、19.98 GiB 峰值分配。该检查使用随机初始化，明确跳过 checkpoint；它不是数据加载测试，也不是正式训练。

CUDA 运行时兼容性通过。若以后重装或重编译 CUDA 扩展，应在 `gaussian_jepa` 环境中明确使用 `$CONDA_PREFIX/bin/nvcc` / `CUDA_HOME=$CONDA_PREFIX`，不要让系统 CUDA 12.6 与 PyTorch CUDA 11.8 混用。未在本阶段重编译扩展。

## 数据与产物

- ShapeSplatsV1 不在项目 `data/`、仓库 `data/`、预期 `shapesplat_ply` 路径或 Hugging Face dataset cache 中；项目 `data/` 当前只有 `rgb`、`plant_gs`、`sam`。
- 本次运行没有下载数据、没有启动正式训练、没有写入 checkpoint。
- 仓库下既有 `experiments/agri_pilot` 和 Stage 3 / Stage 3.5 结果未修改。审计开始前记录了 `agri/`、`experiments/`、`results/` 下 191 个文件的路径、大小和修改时间，清单保存在 `evidence/preserved_artifacts_before.json`。
- 新增的审计文件位于 `results/official_reproduction/preflight/`；Git 当前只显示该新目录为未跟踪项。

## 来源

- 官方 Gaussian-JEPA：<https://github.com/Amazingren/Gaussian-JEPA>
- 官方安装与训练指南：<https://github.com/Amazingren/Gaussian-JEPA/blob/e73f9b66a2cc50e0a3090f920cab8baef0571d60/docs/getting_started.md>
- Hugging Face Hub 下载文档（Context7 获取）：<https://huggingface.co/docs/huggingface_hub/guides/download>
- 搜索说明：博查 Web Search 返回配额不足；本审计改用官方 Git 仓库、Hugging Face 官方 API/数据卡及已认证账号的访问响应，不使用第三方镜像或绕过 gated 权限。
