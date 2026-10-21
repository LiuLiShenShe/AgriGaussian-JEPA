# 环境迁移：conda → uv（无 conda、无 root）

- 执行日期：2026-10-21
- 分支：`agri-stage4c-multicrop`（在 Stage 4C 提交之后）
- 目标：项目不再依赖 conda 管理依赖；全部环境可由仓库内声明文件重建。

---

## 1. 迁移了什么

原来有两个 conda 解释器，分别承担不同任务：

| 用途 | 迁移前 | 迁移后 |
|---|---|---|
| 3DGS 训练/导出/校验/语义 | `/home/test/biosoft/enter/bin/python`（conda base） | `envs/agri-gs/bin/python` |
| Gaussian-JEPA 前向/反向 | `.../envs/gaussian_jepa/bin/python` | `envs/agri-jepa/bin/python` |
| 依赖声明 | `environment.yml` | `env/requirements-{gs,jepa}.in` + `.lock.txt` |
| 激活 | `conda activate` | `source env/activate.sh` |
| 编译器 | conda `gxx_linux-64=11` | `envs/toolchain/gcc-11`（apt `.deb` 解包） |
| CUDA 11.8 | conda `cuda-nvcc` + cccl | `envs/toolchain/cuda-11.8`（NVIDIA runfile 解包） |

`environment.yml` 已删除。

## 2. 为什么必须是两个环境

不是拆分偏好，是 ABI 约束：

- Gaussian-JEPA 使用 PyTorch 2.0.1 + CUDA 11.8，`pointnet2_ops` / `knn_cuda`
  按该 ABI 编译；
- 3DGS 使用 gsplat，需要当前 PyTorch（2.9.1 + cu128）。

两者无法合并，所以仓库里没有任何"全局激活"的动作：每条命令显式指定
`$AGRI_PY_GS` 或 `$AGRI_PY_JEPA`。这也消除了原来"激活了哪个环境"这类隐式状态。

## 3. 无 conda、无 root 的工具链来源

这是迁移中最不确定的部分，实测可行：

| 组件 | 来源 | 关键点 |
|---|---|---|
| gcc 11.5 | `apt-get download` + `dpkg-deb -x` | 不需要 sudo；Ubuntu 24.04 默认 gcc-13 无法编译 CUDA 11.8 头文件 |
| CUDA 11.8 (V11.8.89) | NVIDIA runfile `--silent --toolkit --toolkitpath=... --override` | `--override` 只绕过安装器的编译器检查，实际编译仍用 gcc-11 |
| `pointnet2_ops` / `knn_cuda` | `codeload.github.com` 源码 tarball | `github.com` 本机不可达 |

**踩到的三个坑（已固化进 `bootstrap.sh`）：**

1. `uv venv` 默认会复用 `PATH` 上的解释器，而本机 `PATH` 第一个就是 conda 的
   python。不用 `--python-preference only-managed` 就会在 conda 解释器上建 venv，
   迁移等于没做。bootstrap 里两处都强制了该参数，并有测试保护。
2. `nvidia-cuda-nvcc-cu11` 这个 pip 包**不含 `nvcc`**，只有 `ptxas` 和头文件。
   所以必须拿完整 toolkit。
3. `uv` 的构建隔离会为构建单独装一个更新的 torch（CUDA 13），导致
   `CUDA version 12.6 mismatches 13.0`。gsplat 用 `BUILD_NO_CUDA=1` 复用其源码树里
   已存在的 `csrc.so`；扩展用 `--no-build-isolation`。
4. `setuptools>=81` 移除了 `pkg_resources`，而 torch 2.0.1 的 `cpp_extension` 仍在
   import 它。锁文件固定 `setuptools<81`。

## 4. 等价性验证（不是"能 import"，而是"产出相同"）

| 检查 | 结果 |
|---|---|
| `tools/smoke_test_release.py` | PASS，losses geo/app/ground = 0.463957 / 0.465716 / 0.194473，与历史记录逐位一致 |
| 真实 PLY 前向（Cotton/1，seed 42） | PASS，losses 与 Stage 4B-Fix 记录完全一致；梯度范数 2.281867 |
| 真实 PLY 前向（Cabbage，seed 42） | losses 0.46019089221954346 / 0.4616609811782837 / 0.20179879665374756，与 Stage 4C 记录**完全相同** |
| **重导出 Cotton/1 30k checkpoint** | `point_cloud.ply` SHA256 = `1e9ff42dadc1f056…`，与 Stage 4B-Fix 记录**字节级一致** |
| **重跑 Cabbage 语义 carve** | `plant_only.ply` SHA256 = `421d0a4796f81ae2…`，与 Stage 4C 记录**字节级一致** |
| 60 步真实训练 | `TRAIN_COMPLETE`，loss/PSNR 正常 |
| 测试套件 | 3DGS 侧 191 passed；JEPA 侧 23 passed；共 **214 passed** |

字节级一致的 PLY 是最强的等价证据：它同时覆盖了 torch/gsplat 的数值、
gsplat 导出器的行为、以及 `keep` 掩码的选择。

## 5. 兼容性处理

- 历史报告（`reports/*.md`）和结果文档（`results/**`）**正文不改**——它们记录的是
  当时真实跑的命令。每份文件顶部加了一行注记，指向 `env/README.md`。
- 代码里 5 处硬编码的 conda 解释器路径改为 `$AGRI_PY_JEPA`（报告生成器的可复制命令）。
- `results/official_reproduction/preflight/reproduction_plan.md` 是**计划**不是记录，
  已直接改为 `source env/activate.sh`。
- 新增 `agri/plant_gs/paa/env_resolve.py`：按 `AGRI_PY_*` → `envs/` 的顺序解析解释器；
  找不到就报错并给出 bootstrap 命令，**不会**回退到 PATH 上的 python。

## 6. 新增测试（11 项）

`agri/plant_gs/paa/tests/test_env_management.py`：

- 仓库内不存在 conda manifest；
- 可执行文件中没有 `conda activate/create/run/install/env` 调用（注释除外）；
- 锁文件固定 `torch==2.9.1+cu128` / `torch==2.0.1+cu118`；
- `setuptools<81` 仍被固定；
- bootstrap 两处都使用 `--python-preference only-managed`；
- 解释器解析优先显式变量、缺失时返回 `None` 而非回退。

## 7. 复现方式

```bash
bash env/bootstrap.sh all      # 首次：建两个 venv、取 gcc-11 与 CUDA 11.8、编译扩展
bash env/bootstrap.sh verify   # 只验证，不重建
source env/activate.sh
```

`envs/` 已被 `.gitignore` 忽略（约 20 GB），不会进入仓库。

## 8. 遗留与限制

1. 旧的 conda 目录 `/home/test/biosoft/enter` **未删除**。该安装还服务着本机其他
   项目（25 个 env），删它超出本任务范围。本项目已不再引用它，可用
   `python -m agri.plant_gs.paa.env_resolve --check` 确认。
2. `envs/toolchain` 约 7 GB，`envs/` 总计约 20 GB，需要磁盘余量。
3. CUDA runfile 约 4.3 GB 只在首次 bootstrap 时下载；`bootstrap.sh` 目前不校验其
   校验和，因为 NVIDIA 未在该 URL 旁提供稳定校验和清单。需要的环境可自行加。
4. `knn_cuda` 在**移动环境目录后**首次 import 会触发重新编译，此时需
   `source env/activate-jepa-build.sh`。
