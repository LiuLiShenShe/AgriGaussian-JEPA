# Stage 4B-Fix — 并发事故复盘、两阶段复现与 checkpoint→PLY 溯源

> **环境说明（迁移后补注，2026-10-21）：** 本文件是历史运行的记录，正文保留当时的真实环境路径。项目已改用 uv 管理环境（`env/bootstrap.sh`），不再使用 conda；复现时请先 `source env/activate.sh`，用 `$AGRI_PY_GS` / `$AGRI_PY_JEPA` 代替文中的旧解释器路径。

**状态：完成。Cotton/1 已干净复现，并发修复与来源校验均通过。**

- 执行日期：2026-10-20
- 分支：`agri-stage4b-fix-repro`，基于 `origin/main`（`062dcbe`，PR #2 合并点）
- 硬件：2 × NVIDIA RTX A6000 48GB
- 数据：PAA-Plant-Dataset（`JackTrum938/PAA-Plant-Dataset`，Apache-2.0，
  Hub revision `19e7f2f2…`），只使用**已下载**的 `Cotton/1` 源数据，未新增下载
- 环境：3DGS = base env（torch 2.9.1+cu128 / gsplat 1.6.0）；
  Gaussian-JEPA = `envs/gaussian_jepa`（torch 2.0.1，`knn_cuda` / `pointnet2_ops` 已编译）

本报告中的所有数值都来自本轮实际运行，可追溯到
`data/plant_gs/PAA/Cotton_1_repro_v1/` 下的 `results/`、`final/` 与 `logs/`。
凡属推断或不可验证者，均显式标注。

---

## 0. 结论速览

| 问题 | 结论 |
|---|---|
| 并发安全是否通过 | **通过**。169 项测试，其中 16 项专门驱动真实多进程竞争 |
| 两阶段 Runner 是否修好 | **是**。`results/train_7k` 与 `results/train_30k` 是两次独立训练 |
| Cotton/1 最新可信指标 | **fg PSNR 24.145 dB / IoU 0.9708 / bg alpha 0.00766**，319,969 高斯，step 29,999 |
| 与旧值 23.882 dB 的差异 | **+0.263 dB**，但**两者不可比**：旧值来自被覆盖的 24,000 步 checkpoint，且当时有两进程竞争 |
| checkpoint→PLY 来源链 | **完整**。checkpoint SHA256 `be0e88cb…` → PLY SHA256 `1e9ff42d…`，逐属性误差全 0 |
| 总测试数与失败数 | **169 passed / 0 failed** |
| 是否可正式关闭 Stage 4B | **可以**，附三项移交条件（§9） |
| 是否建议进入 Stage 4C | **建议进入数据规模化，不建议启动 JEPA 正式预训练**（§10） |

---

## 1. 事故复盘：Cotton/1 是怎么被污染的

### 1.1 事实

`data/plant_gs/PAA/Cotton_1/results/train_30k/` 在 2026-10-20 09:17–09:35 之间
被**两个并发的训练进程**同时写入。四个独立证据：

**(1) `history.jsonl` 含两套交错的记录流。** 642 条记录、305 个唯一步号、
**141 处步号逆序**。正常单次训练是单调递增的（其余所有目录逆序数均为 0）。

**(2) 两套不同的 eval 节奏。** 一套是 `eval_every=1000`（1000, 2000, 4000, 5000…），
另一套是 `eval_every=3000` 的恢复调度（3089, 6089, 9089, 12089…，
`+89` 正是 `len(train_idx)`）。两套 eval 在同一份 `history.jsonl` 里各写各的。

**(3) 两套不同的 checkpoint 保存节奏。** 17 个周期 checkpoint 的步号间隔同时出现
`1000` 与 `2000`，即 `save_every=2000` 与 `save_every=5000` 两条序列的交集：

| 序列 | 步号 |
|---|---|
| A（间隔 2000） | 2000, 4000, 6000, 8000, 12000, 14000, 16000, 18000, 22000, 24000, 26000, 28000 |
| B（间隔 5000） | 5000, 10000, 15000, 20000, 25000 |

按文件 mtime 反推，两条序列是**真正并发**的：A 在 09:18:11–09:27:43 推进 24,000 步
（≈23.8 ms/步），B 在 09:24:37–09:31:49 推进 20,000 步（≈21.6 ms/步）。
同一分钟内两个进程各自前进，符合竞争而非串行重跑。

**(4) 报告与磁盘互相矛盾，且旧报告内部自相矛盾。** `final/report.json` 记录
`source_checkpoint = splats_best.pt`、`step = 24000`、
`clean_validation.foreground_psnr = 23.882314`。但当前的 `splats_best.pt`
内部 `step = 29999`。也就是说：导出发生在 24,000 步的 checkpoint 还存在的时刻，
之后另一个进程把它覆盖成了 29,999 步。

同一目录的 `status.json` 又写 `best_step = 29999`、`gaussians = 332859`。
而 Stage 4B 报告 §2.4 的表格把 Cotton/1 的"采用步"写成 **29,999**，
§2.5 却引用 23.882 dB——**与它自己引用的 `report.json`（step 24000）直接冲突**。
同一份报告里，正文说导出用的是 29,999 步，数字却来自 24,000 步。这不是笔误，
而是两个进程各自更新 `splats_best.pt`／`status.json` 时，报告写作者只能看到
前后不一致的两个版本。

独立复核工具（`audit_historical_export.py`）重算了这条链，结论是 **fail**：

| 检查项 | 结果 |
|---|---|
| checkpoint 内部 step | 29,999 |
| `report.json` 声称的 step | 24,000 |
| PLY 行数 / keep 掩码选中行数 | 320,644 / 320,644（行数一致，容易误判为"没问题"） |
| 同一 keep 掩码下的逐属性最大误差 | x 3.03e-3、y 5.59e-3、z 2.88e-3、**opacity 14.03**、scale_2 6.53、rot 0.53–0.57 |

> 注意最后两行：**行数一致但属性不一致**。这正是"只比长度"或"只看 mtime"
> 这类检查会漏掉的失效模式，也是本阶段坚持按 keep 掩码做逐行映射比对的原因。

### 1.2 为什么原锁没有拦住

Stage 4B 的提交 `e7a686d` 号称"添加训练运行锁防止并发覆写结果目录"，但它是失效的：

```python
def acquire_run_lock(result):
    import fcntl
    ...
    handle.write(str(os.getpid()))   # NameError: name 'os' is not defined
```

`train.py` 顶部没有 `import os`。函数在 `flock` 成功后才调用 `os.getpid()`，
看似加锁成功，实际每次都抛 `NameError`。而且这个函数是在
**检查 `history.jsonl` / `splats.pt` 之后**才调用的——即使它能跑，
"先检查再加锁"本身就存在检查与加锁之间的竞态。

### 1.3 事故时间线

| 时间（UTC+8） | 事件 |
|---|---|
| 09:17:27 | `logs/` 创建，新 30k 训练启动（进程 A） |
| 09:18:11–09:27:43 | 进程 A 每 2000 步落一个 checkpoint |
| 09:24:37–09:31:49 | 进程 B 每 5000 步落一个 checkpoint，与 A 交错 |
| 09:32:04 | `final/` 导出，读的是当时的 `splats_best.pt`（24,000 步） |
| 09:32:10 | `validation.json` 写入 23.882 dB |
| 09:33:29 | **`splats_best.pt` 被写成 29,999 步**（另一个进程的最终结果） |
| 09:33:32 | `status.json` 写 `complete / best_step 29999` |
| —— | 此后 `report.json` 的 `source_checkpoint` 已经指不回原来的文件 |

---

## 2. 并发修复：目录级独占锁

新增 `agri/plant_gs/foreground/runlock.py`。

### 2.1 设计

- **`RunLock`**：对 `<result>/.train.lock` 取 `flock(LOCK_EX | LOCK_NB)`。
  第二个进程立即失败（`LockUnavailable`），不排队、不写入。
- **锁由内核持有**：进程被 `kill -9` 后锁自动释放，不会把目录永久锁死。
- **路径归一**：`resolve_result_path()` 用 `os.path.realpath` + `abspath` 归一，
  因此符号链接、相对路径、`./` 拼写都指向同一 inode，争用同一把锁；
  不同目录互不影响。
- **先加锁，再检查**：`train.py` 在读取 `history.jsonl` / `config.json` /
  checkpoint **之前**就取得锁，把"检查"和"写入"放进同一个原子区间。
- **状态机**：`describe_result()` 区分 `empty` / `complete` / `partial` / 配置不符。
  - `complete` → 拒绝重训；
  - `partial` → 默认拒绝，只有 `--resume` 且存在**真正的**续跑 checkpoint 才继续；
  - 配置不符 → 立即拒绝，不允许两个实验混写。

### 2.2 测试（16 项，全部使用真实子进程）

`flock` 是内核状态，进程内 mock 证明不了任何事，因此全部用 `subprocess` 驱动：

| 测试 | 验证内容 |
|---|---|
| `test_second_process_is_refused_while_first_holds_the_directory` | 持锁进程存活时，第二个进程被拒 |
| `test_two_simultaneous_launches_produce_exactly_one_winner` | 无协调同时启动，恰好一个赢 |
| `test_different_result_directories_train_concurrently` | 不同目录不互相串行（防止把锁做成全局锁） |
| `test_lock_is_released_when_the_holder_exits_abnormally` | `kill -9` 后锁释放 |
| `test_lock_is_released_after_an_exception` / `_on_system_exit` | 异常与 `SystemExit` 路径也释放 |
| `test_symlink_relative_and_absolute_paths_contend_for_one_lock` | 三种写法争同一把锁 |
| `test_relative_path_alias_is_locked_too` | 相对路径 + `./` 别名同样被锁 |
| `test_complete_directory_is_never_silently_retrained` | 已完成目录拒绝重训 |
| `test_directory_with_history_is_not_silently_appended` | 有 history 的残缺目录不静默追加 |
| `test_partial_directory_with_a_different_config_is_refused` | 配置不符拒绝 |
| `test_refusal_does_not_touch_the_directory` | 拒绝时目录字节不变 |
| `test_losing_process_writes_nothing` | 输的进程一行都不写 |
| `test_second_run_on_a_completed_directory_is_refused_by_state` | 状态层拒绝 |

### 2.3 真实续跑的定义

`--resume` 只在 checkpoint 同时携带下列状态时才生效：

- 6 组高斯参数（means / scales / quats / opacities / sh0 / shN）
- 6 个 Adam 优化器的 `state_dict`（一阶矩、二阶矩、步数）
- `DefaultStrategy` 的动态高斯状态（`grad2d`、`count`、`radii`、`scene_scale`）
- NumPy 生成器状态 + Torch CPU/CUDA RNG 状态
- 配置哈希、数据集路径、内部 step、`best_score`/`best_step`、`run_id`

缺任意一项即 `REFUSING TO RESUME`。模型专用 checkpoint（`splats_best.pt`、
周期 `splats_*.pt`）**明确拒绝**，不会冒充续跑。

`--append-history` 已删除；传入会直接报错，说明它不是恢复机制。

**实测**：600 步训练在第 100 步后被 `kill -9`，`--resume` 从 `splats_resume.pt`
恢复到 step 100 并跑完 600 步，`history.jsonl` 写入一条 `{"event":"resume"}` 标记，
最终 `status: complete`、`run_id` 保持不变。

### 2.4 原子写

所有 checkpoint 走 `atomic_torch_save`：写 `<name>.pt.tmp` → `fsync` → `os.replace`。
JSON/PLY 同理。读者永远只会看到旧文件或新文件，不会看到半写文件。

---

## 3. 两阶段 Runner

`agri/plant_gs/paa/run_paa_pilot.sh` 重写，可复用逻辑抽到 `paa/stages.sh`
（便于测试直接 source，不必跑训练）。

### 3.1 两阶段是两次独立训练

```
results/train_7k/     质量门控。独立 config / log / status / checkpoint
results/train_30k/    正式训练。全新训练，不续接 7k 模型
```

30k 阶段**不读取** 7k 的任何 checkpoint；它是从初始化重新开始的独立优化。
这与 Stage 4B 的记录不同——那里的 7k 与 30k 实际上是两次独立训练，
却被当成"先试后正式"的一条链来描述，正是这一点让目录级来源变得模糊。

### 3.2 流程

| 步 | 动作 | 失败条件 |
|---|---|---|
| 1 | COLMAP 审计（只读） | 审计不通过 |
| 2 | 逐样本测量掩膜配准、准备数据集 | 配准门限不通过 |
| 3 | `train_7k`（7000 步，eval 1000，save 2000） | 未达 `complete` |
| 4 | 质量门控，读 7k 自己的 `validation.json` | PSNR < 18 / IoU < 0.85 / bg alpha > 0.05 |
| 5 | `train_30k`（30000 步，eval 3000，save 5000） | 未达 `complete` |
| 6 | 从**显式指定**的 `results/train_30k/splats_best.pt` 导出 | checkpoint 缺失 / 训练未完成 / 持锁 |
| 7 | 独立复核溯源 + PLY 结构校验 | 任一项 fail |
| 8 | JEPA 兼容 + 1024 采样前向/反向 + 优化器/EMA 探针 | fail |
| 9 | 高斯统计 + 注册（upsert） | 记录校验失败 |

### 3.3 覆盖与恢复

- **默认不覆盖任何已有训练目录**：`complete` 跳过，`partial` 拒绝。
- `RESUME=1`：跳过已完成阶段；对残缺阶段调用训练器 `--resume`，
  训练器再按 §2.3 逐项校验。缺 checkpoint、配置不一致都会给出具体诊断。
- 门控"读不到 validation"与"指标不达标"返回**不同退出码**（2 vs 1），
  避免把配置错误误报成质量问题。
- 每个阶段写 `stage_manifest.json`：命令原文、git commit、是否 dirty、完成时间。
- 兼容 `PY_GS` / `PY_JEPA` / `GPU` / `REGISTRY`。

### 3.4 复现命令

```bash
# 从头运行（新 workspace）
PY_GS=/home/test/biosoft/enter/bin/python \
PY_JEPA=/home/test/biosoft/enter/envs/gaussian_jepa/bin/python \
GPU=1 SEED=42 \
bash agri/plant_gs/paa/run_paa_pilot.sh \
  data/external/PAA-Plant-Dataset/Cotton/1 \
  data/plant_gs/PAA/Cotton_1_repro_v1

# 中途崩溃后续跑（跳过已完成阶段；残缺阶段只有在真正可续跑时才继续）
... RESUME=1 bash agri/plant_gs/paa/run_paa_pilot.sh <SAMPLE> <WS>

# 仅重做导出/校验（两个训练阶段均已 complete）
... RESUME=1 bash agri/plant_gs/paa/run_paa_pilot.sh <SAMPLE> <WS>
```

---

## 4. checkpoint → PLY 的可信来源链

### 4.1 记录哪些字段

| 字段 | 训练 | 导出 | 说明 |
|---|---|---|---|
| dataset / acquisition | ✓ | — | 绝对路径 + 样本名 |
| git commit / dirty | ✓ | — | 代码版本 |
| seed | ✓ | — | 随机种子 |
| 训练配置 + 配置哈希 | ✓ | — | `sha256(canonical json)` |
| checkpoint 路径 / SHA256 / 字节数 / mtime | ✓ | ✓ | 文件身份 |
| checkpoint 内部实际 step | ✓ | ✓ | 从文件里读，不信任文件名 |
| export 起止时间 | — | ✓ | UTC |
| 输出 PLY SHA256 | — | ✓ | |
| 过滤前后高斯数 | — | ✓ | 332,883 → 319,969 |
| PLY 属性一致性结果 | — | ✓ | 逐属性最大误差 |
| retained index SHA256 | — | ✓ | 过滤后行到原始行的映射 |

### 4.2 三个关键约束

1. **导出前确认训练已完成且 checkpoint 不再变化。**
   `export.py` 先尝试取该目录的锁；训练未完成（`status.json` 非 `complete`
   或缺 `splats.pt`）直接拒绝。导出开始前算一次 checkpoint SHA256，
   结束前再算一次，不一致即失败。
2. **过滤改变长度，所以按索引映射比对，而不是逐元素对齐。**
   清洗会丢弃高斯，PLY 必然比 checkpoint 短。正确做法是
   `cleaned = checkpoint[keep]`，再与 PLY **逐行、同序**比较；
   并额外记录 `sha256(flatnonzero(keep))`，使"用了哪个映射"本身也可复核。
3. **导出记录与 PLY 属性校验分开存。**
   `format_validation` 回答"文件格式对不对"，
   `attribute_validation` 回答"它是不是来自这个 checkpoint"。两者混为一谈时，
   一个格式合法的 PLY 会被当成来源合法的 PLY。

### 4.3 本轮实测链

```
results/train_30k/splats_best.pt
  step            = 29999
  SHA256          = be0e88cb6eb8b1a5986091a32a775b5484f7cb988f374d9d3d7a03f20f3dece5
  bytes           = 78,563,945
        │  export（14.13 s，导出前后哈希一致）
        ▼
final/point_cloud.ply
  rows            = 319,969
  SHA256          = 1e9ff42dadc1f0565e16853b994648e0ad9b5de6b6fec7769ed15882f45a3495
  keep 索引 SHA256 = 8e94bfa6a96b758a4078025f4cc0373a3609997a1976b5d7c35a9c65e79d2068
  高斯数          = 332,883 → 319,969
  逐属性最大误差   = 全 0（x/y/z/opacity/scale/rot/f_dc 共 14 项）
        │  verify_provenance.py（独立重算）
        ▼
final/provenance_verification.json  →  status = pass
```

该 PLY 的 SHA256（`1e9ff42d…`）也被 JEPA 冒烟测试记录为 `source_sha256`，
因此来源链一直延伸到下游的 JEPA 前向/反向。

### 4.4 Registry 的 accepted 状态

`PlantRecord` 新增 `provenance` 字段。规则：

- `acceptance_status = accepted` **必须** `provenance.state == verified`；
  否则 `validate()` 返回问题、`Registry.upsert` 拒绝写入。
- 区分**当次验证**与**事后追溯验证**：
  `verified_retrospectively: true` 表示该运行早于溯源机制，
  是后来用 `audit_historical_export.py` 从磁盘字节重算得到的结论。

---

## 5. Cotton/1 干净复现

工作目录：`data/plant_gs/PAA/Cotton_1_repro_v1/`（全新，未触碰旧目录）。
污染目录 `data/plant_gs/PAA/Cotton_1/` 已 `chmod -R a-w` 设为只读，
作为事故证据保留，不参与任何指标。

### 5.1 输入与配准复核

| 项 | 本轮实测 |
|---|---|
| 注册图像 / 使用帧 | 102 / 102（0 丢弃） |
| 相机模型 | 2 × SIMPLE_PINHOLE |
| 稀疏点（全部 / 前景） | 23,388 / 22,444 |
| 重投影误差 mean / med / p95 | 0.586 / 0.329 / 1.851 px |
| 掩膜命中率 min / med | 0.8588 / 0.9594 |
| 掩膜 vs 图像 IoU min / med | 0.8267 / 0.8946 |
| 统一画布 | 904 × 1600 |
| 滚动相机数 | 1（36 帧） |
| 准备耗时 | 74.9 s |

配准结果与 Stage 4B 记录逐位一致（同一源数据、同一估计器），
说明复现的输入侧没有漂移。

### 5.2 7k 质量门控

| 指标 | 实测 | 门限 | 结论 |
|---|---|---|---|
| foreground PSNR | **22.168 dB** | ≥ 18 | PASS |
| silhouette IoU | **0.96565** | ≥ 0.85 | PASS |
| background alpha | **0.00922** | ≤ 0.05 | PASS |
| 训练耗时 | 282.2 s | — | |
| 高斯数 | 343,872 | — | |
| 最佳步 | 6,999 | — | |

### 5.3 30k 正式训练（独立新训练）

| 项 | 实测 |
|---|---|
| 状态 | `complete` |
| 步数 | 30,000 |
| 最佳步 / 最佳评分 | **29,999** / 28.8442 |
| 高斯数（训练结束） | 332,883 |
| 训练耗时 | 1,122.95 s |
| run_id | `20261020T064528Z-3273295` |
| 配置哈希 | `982451ea38471b14…` |

### 5.4 导出与渲染质量（13 个单场景留出视角）

| 指标 | 原始 | 清洗后 |
|---|---|---|
| foreground PSNR | 24.1441 dB | **24.1447 dB** |
| silhouette IoU | 0.97079 | **0.97080** |
| background alpha | 0.007691 | **0.007664** |
| 背景不透明比例 | 0.006796 | 0.006794 |
| 高斯数 | 332,883 | **319,969**（移除 12,914） |

> 留出视角是**同一场景内**的视角，只用于重建质量报告，
> **不是**独立植株泛化测试。

### 5.5 PLY 与 Gaussian-JEPA

| 项 | 实测 |
|---|---|
| PLY 属性数 | 59 |
| JEPA 必需 14 属性 | 全部存在 |
| NaN / Inf | 无 |
| `gsjepa_compatible` | **true** |
| opacity ∈ [0,1] / scales > 0 / quat 单位范数 | 全部 true |

**前向/反向**（`gaussian_jepa` env，1024 高斯均匀采样，seed 42）：

| 项 | 实测 |
|---|---|
| losses geo / app / ground | 0.463949 / 0.459002 / 0.201430 |
| 总损失 | 1.124382 |
| 有限梯度张量 | **215 / 215** |
| 梯度全局 L2 | 2.281867 |
| 权重未改变 | true |
| 初始化 | `random`（官方预训练 checkpoint 仍不可用） |
| 语义有效性 | **NOT_EVALUATED** |

**优化器 / EMA 探针**（AdamW lr 1e-4 / wd 0.05，EMA 0.996，5 步）：

| 项 | 实测 |
|---|---|
| 总损失 | finite |
| 参数移动 L2 | 0.204657 |
| EMA gap | 0.0 → **1.2995** |
| EMA teacher 每步更新 | true |
| 状态 | **PASS** |

### 5.6 新旧 Cotton 指标对比

| 指标 | Stage 4B（已撤回） | Stage 4B-Fix（可信） | 差异 |
|---|---|---|---|
| 来源 | `splats_best.pt` @ 24,000 步，**导出后被覆盖** | `splats_best.pt` @ 29,999 步，**哈希可验证** | — |
| fg PSNR | 23.882 dB | **24.145 dB** | +0.263 dB |
| silhouette IoU | 0.97032 | **0.97080** | +0.00048 |
| background alpha | 0.007831 | **0.007664** | −0.000167 |
| 高斯数 | 320,644 | 319,969 | −675 |
| 来源链 | **无效** | verified | — |

三个诚实的解读：

1. **+0.263 dB 不是"提升"。** 旧值是一个来源不可证明的数字，不是有效基线。
   把它当基线算提升，等于给一个错误读数记功。
2. **两个数字本身并不矛盾。** 旧值取自 24,000 步；新值取自 29,999 步。
   旧目录里 29,999 步那一路（`eval_every=3000` 流）自己的记录是 24.121 dB，
   另一路是 24.154 dB，新值 24.145 落在两者之间——与"同配置、同种子、
   仅受并发影响"的预期一致。
3. **真正被修复的是可追溯性，不是精度。** 新值可以一路追溯到
   `splats_best.pt` 的 SHA256；旧值不能。

### 5.7 训练耗时说明

本轮 30k 用时 1,122.95 s，旧记录 630 s。**这不是训练变慢**，而是测量条件不同：

- 旧运行是两个进程**共享**同一张卡，各自 30k 步耗时约 630–650 s，
  折合 21.6–23.8 ms/步；
- 本轮运行时宿主 **load average ≈ 55**，且 GPU 上有其他租户进程（约 25 GB 常驻），
  单进程实测 37.4 ms/步；
- 本轮每个周期还额外写一份完整可续跑 checkpoint（模型 + 优化器 + 策略 + RNG，
  实测 241.5 MB vs 模型专用 78.6 MB，即 **3.07 倍**），增加了 I/O。

因此两个墙钟时间**不构成性能对比**。真实训练步速需要在独占机器上另测。

---

## 6. 另外三组资产的独立审计（只读）

审计工具：`agri/plant_gs/foreground/audit_historical_export.py`。
它不信任任何已记录的数字，而是**从磁盘字节重算**：
用 `cleanup_scores.npz` 里的 keep 掩码筛选 checkpoint 的高斯，
再与 PLY 逐属性比对，并检查 checkpoint 内部 step 是否等于 `report.json` 声称的 step。

结果写入 `results/agri_paa/stage4b_fix/historical_export_audit.json`。

| 资产 | 并发污染特征 | checkpoint step vs report step | 逐属性最大误差 | 结论 |
|---|---|---|---|---|
| `Broccoli/20260327-10` | 无（逆序 0，单一保存节奏） | 27,000 = 27,000 | **全 0** | **pass** |
| `Broccoli/20260402-10` | 无 | 24,000 = 24,000 | **全 0** | **pass** |
| `Broccoli/20260327-5` | 无 | 29,999 = 29,999 | **全 0** | **pass** |
| `Cotton/1` | **有**（逆序 141，两套节奏） | 29,999 ≠ 24,000 | 最大 14.03（opacity） | **fail** |

三组 Broccoli 的 `history.jsonl` 均为 311 条记录、305 个唯一步号、
**0 处逆序**、单一保存间隔（30k 为 5000，7k 为 2000），
且 `config.json` 与实际落盘节奏一致。**没有** Cotton/1 的污染特征。

### 6.1 可追溯程度的如实标注

三组 Broccoli **没有**当次记录的哈希链——它们运行在溯源机制存在之前。
本轮能建立的是**事后一致性**：PLY 与它声称的 checkpoint 至今仍逐属性一致。
这比"只看 mtime"强得多，但仍弱于当次哈希链：

- 它不能证明"导出那一刻"用的就是这个文件，只能证明"现在两者一致"；
- 若中间有人用数值相近的文件替换过 checkpoint，逐属性比对可能漏判
  （Cotton/1 的 opacity 误差达 14.03，属于容易发现的情形；
  若替换的是相邻步的 checkpoint，误差可能很小）。

因此 registry 中这三行记为 `state = verified` +
**`verified_retrospectively = true`**，与 Cotton/1 复现行的当次验证区分开。
报告层面，三组 Broccoli 的数值**继续有效**，但应理解其来源强度为
"事后可验证"，而非"当次哈希锁定"。

---

## 7. 哪些旧数据可信、哪些需要撤回

| 资产 | 训练指标 | 来源链 | 处置 |
|---|---|---|---|
| `Broccoli/20260327-10` | 22.908 dB / IoU 0.9866 | 事后逐属性一致 | **保留**，标注事后验证 |
| `Broccoli/20260402-10` | 24.513 dB / IoU 0.9498 | 事后逐属性一致 | **保留**，标注事后验证 |
| `Broccoli/20260327-5` | 26.135 dB / IoU 0.9813 | 事后逐属性一致 | **保留**，标注事后验证 |
| `Cotton/1`（旧） | 23.882 dB 等 | **断裂**（step 不符、属性不符） | **撤回正式验收**，由本轮复现取代 |
| `Cotton/1`（旧）7k 阶段 | 22.180 dB | 目录本身干净（逆序 0、单一节奏） | **技术有效**，但与 30k 无来源关联，不作为验收依据 |
| 旧报告 §2.4–§2.6 的 Cotton 行 | — | — | 已在原报告中标注 `〔已撤回〕`，数值保留不删 |

**没有为了让旧数字"补上哈希"而重训三株 Broccoli。** 现有证据已足以判断它们的
PLY 与 checkpoint 一致；重训只会引入新的不确定性，不增加信息。

---

## 8. 回归与测试

| 套件 | 数量 | 结果 |
|---|---|---|
| `agri/plant_gs/foreground/tests` | 74 | passed |
| `agri/plant_gs/paa/tests` | 72 | passed |
| `agri/plant_gs/jepa/tests` | 23 | passed |
| **合计** | **169** | **169 passed / 0 failed** |

新增的测试文件：

| 文件 | 覆盖 |
|---|---|
| `foreground/tests/test_runlock.py` | 真实多进程锁、异常退出、符号链接、状态机、拒绝不写入 |
| `foreground/tests/test_trainer_contract.py` | 加锁先于检查、`--append-history` 拒绝、真实续跑门控、原子写 |
| `foreground/tests/test_provenance.py` | 哈希稳定性、篡改检测、keep 掩码映射、缺失字段 |
| `foreground/tests/test_historical_audit.py` | 事后审计的 pass / fail / insufficient 三态 |
| `paa/tests/test_runner_stages.py` | 两阶段独立、完成跳过、残缺拒绝、门控、manifest |
| `paa/tests/test_registry.py`（扩充） | accepted 必须 verified、事后验证标记、溯源随 upsert 保留 |

顺带修复的一个潜在缺陷：Runner 的环境探测原写作
`torch.get_device_name(i)`（不存在），实际是 `torch.cuda.get_device_name(i)`。
Stage 4B 没有暴露它，因为那次是按步骤手工执行的，没有走完整脚本。

---

## 9. 是否可以正式关闭 Stage 4B

**可以关闭。** 四项验收条件均已满足：

1. **并发安全**：169 项测试通过，其中 16 项用真实多进程验证；
   原锁的 `NameError` 失效原因已定位并修复。
2. **Runner 可复用**：两阶段协议在 Cotton/1 上端到端跑通，
   从 COLMAP 审计到 registry 注册全流程无人工干预。
3. **Cotton/1 干净复现**：24.145 dB，来源链完整可验证。
4. **来源校验**：checkpoint→PLY 的 SHA256 与逐属性一致性均由独立工具复算通过。

**移交时需一并说明的三个限制**（不构成阻塞，但必须写进后续记录）：

- 三组 Broccoli 的来源强度是"事后可验证"，不是"当次哈希锁定"；
- Cotton/1 的 24.145 dB 是**单场景留出视角**指标，不是植株泛化指标；
- 本轮所有 JEPA 数字仍来自**随机初始化**模型，`semantic_effectiveness = NOT_EVALUATED`。

---

## 10. 是否建议进入 Stage 4C

**建议进入 Stage 4C，但把 4C 定义为"数据规模化与来源治理"，而不是 JEPA 正式预训练。**

理由：

1. **来源链已就绪，正是规模化的前提。** 41 组样本如果在没有溯源的情况下铺开，
   任何一次并发或覆盖都会重现 Cotton/1。现在可以先规模化，代价最低。
2. **官方预训练 checkpoint 仍不可用**（ShapeSplatsV1 未获批）。
   在此前提下启动"正式预训练"没有意义——没有权重可加载，
   任何损失曲线都不能解释为学到了农业语义。
3. **现有阻塞项在数据侧，不在模型侧。** 语义分离只验证过 1 个样本，
   且 `Cotton/1` 这类白色主体已知会让零样本模型失效（大片棉花被标为"支架"）。
   这在 4C 里应优先解决。

建议的 4C 前置门限：

| 门限 | 依据 |
|---|---|
| 新增样本全部走新 Runner（两阶段 + 溯源） | 本报告 §3–§4 |
| 每个作物至少 1 个样本人工检查语义分离参考面板 | Stage 4B §6.2 风险 1 |
| 覆盖 ≥ 3 种相机模型 | 目前只有 SIMPLE_PINHOLE / FULL_OPENCV |
| Registry 无 `provenance != verified` 的 accepted 行 | 本报告 §4.4 |

---

## 11. 产物位置

| 内容 | 路径 |
|---|---|
| 新 Cotton/1 工作区 | `data/plant_gs/PAA/Cotton_1_repro_v1/` |
| 污染目录（只读证据） | `data/plant_gs/PAA/Cotton_1/` |
| 事故证据哈希 | `results/agri_paa/stage4b_fix/cotton1_incident_evidence.json` |
| 三组 Broccoli 事后审计 | `results/agri_paa/stage4b_fix/historical_export_audit.json` |
| 导出溯源记录 | `.../Cotton_1_repro_v1/final/export.provenance.json` |
| 独立复核结果 | `.../Cotton_1_repro_v1/final/provenance_verification.json` |
| Registry | `data/plant_gs/PAA/registry.jsonl` |
| 旧报告（已标注撤回） | `reports/stage4b_paa_multiplant.md` |