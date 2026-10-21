# Plant 3DGS reconstruction (stage 1)

Reconstruct a single real plant as a standard 3D Gaussian Splatting model and
export a `point_cloud.ply` that Gaussian-JEPA can consume directly.

Input: 250 RGB frames extracted from a handheld video of one potted cyclamen
(`XianKeLai1`), plus SAM plant masks for 208 of them.

Output: a Gaussian splat PLY containing the plant only —
**172,535 Gaussians**, delivered to
`data/plant_gs/plant_001/point_cloud.ply`.

This stage deliberately stops here — it does not touch Gaussian-JEPA's model
code, does not pretrain, and says nothing about downstream tasks.

---

## 0. Result at a glance

| | |
|---|---|
| source frames | 250, of which **247** usable (3 corrupt) |
| frames with SAM mask | 208 (205 usable) |
| SfM | COLMAP 3.9.1, **247/247 registered**, 71,772 points, **0.723 px** mean reprojection error |
| 3DGS | gsplat 1.5.3, 30k steps, full-frame L1+SSIM, degree-3 SH |
| best checkpoint | step 5000 — plant PSNR **16.02**, full-frame PSNR **17.12** |
| scene model | 276,343 Gaussians (step-5000 checkpoint) |
| extracted plant | **172,535 Gaussians**, 59 PLY properties, no NaN/Inf |
| validation | `validate_ply.py` exit 0, `verify_gsjepa_compatible.py` exit 0, feature width **14** |

## 1. What the capture actually is (and why it matters)

The frames are **not** a turntable sequence. The camera orbits the plant by
hand indoors: the plant stays centred while the background sweeps past
(bicycles, chairs, a table, floor tiles). Verified by comparing frame-to-frame
background motion against subject motion.

This distinction drives the whole pipeline:

| | turntable | **handheld orbit (this capture)** |
|---|---|---|
| background motion | static | large |
| features usable for SfM | subject only | subject **and** room |
| typical failure | degenerate/rotating poses | none — this is the easy case |

So COLMAP is run on the **full frames** and happily gets its poses from the
room. The plant masks are *not* used for pose estimation at all; they are only
used later, to decide what belongs to the plant.

Measured orbit geometry (from the registered poses): one camera,
`OPENCV` 1080×1920, `f = 1612 px`; orbit-centre distance median **3.12**
(range 1.46–6.17) in COLMAP units; median consecutive baseline **0.60**, i.e.
ample parallax with no degenerate view.

## 2. Data problems found (reported, not papered over)

* **3 of the 250 frames are unrecoverable.** `0111.jpg`, `0130.jpg`, `0134.jpg`
  do not start with a JPEG SOI marker — their bytes are unrelated to JPEG
  (`1b 1b 00 59 …`). These are transfer/copy errors, not truncated files, so
  there is nothing to salvage. They are excluded (`prepare_frames.py` verifies
  every frame decodes before use). 247 frames remain.
* **Masks exist for only 208 frames** (0000–0207). Since three of those are the
  corrupt frames, 205 frames have both pixels and a mask; those are what the
  3DGS stage trains on.
* **The masks are not a clean plant/pot split.** Inspecting them shows the
  silhouette is inconsistent — the union bounding box spans the whole frame,
  and coverage ranges from 9% to 45% (mean 24%). They are good enough to define
  "plant region" for extraction, but they are not a segmentation-quality
  annotation. Nothing downstream should treat them as ground truth.
* **Motion blur is real and uneven.** Laplacian variance across the 205 frames
  spans 15–217, so the blurriest 20% of frames are excluded from the fit
  (`--min-sharp-pct 20`), leaving 143 train / 21 validation frames.

## 3. Pipeline

```
250 RGB frames ──► prepare_frames.py ──► 247 frames @ 1080x1920 + 205 masks
                                            (3 corrupt frames dropped)
        │
        ├─► run_colmap.sh ──► COLMAP SfM on FULL frames
        │        feature_extractor (OPENCV, single camera)
        │        sequential_matcher (overlap 30, quadratic)
        │        mapper
        │
        └─► build_dataset.py ──► undistort to PINHOLE (images AND masks, same mapping)
                                  filter the sparse cloud to plant points
                    │
                    ├─► train_plant_gsplat.py ──► full-frame 3DGS (gsplat), splats.pt
                    │
                    └─► extract_plant.py ──► depth-aware visibility + SAM masks
                                             ──► point_cloud.ply  (plant only)
                    │
                    ├─► validate_ply.py ──► field/format report
                    └─► verify_gsjepa_compatible.py ──► Gaussian-JEPA loader replay
```

### Step 1 — prepare frames

```bash
python prepare_frames.py
```

Decodes every frame (dropping the 3 corrupt ones), writes working-resolution
images (0.5×: 1080×1920) and resized masks.

### Step 2 — COLMAP

```bash
./run_colmap.sh            # or RESUME_FROM=match|mapper ./run_colmap.sh
```

Two environment quirks are baked in, both of which fail loudly otherwise:

* This COLMAP is built **without CUDA**. Leaving `SiftMatching.use_gpu` at its
  default makes the matcher try to create a GL context and abort inside Qt
  (`could not connect to display`). Both `use_gpu` flags must be `0`.
* In COLMAP 3.9.1 the camera model belongs to `feature_extractor`;
  `database_creator` rejects `--camera_model`, and the mapper option is
  `init_min_num_inliers`, not `init_min_num_registered`.

### Step 3 — dataset

```bash
"$AGRI_PY_GS" build_dataset.py
```

* Undistorts **images and masks** from OPENCV to PINHOLE. COLMAP's
  `image_undistorter` cannot undistort masks, so `build_dataset.py` remaps them
  with `cv2.initUndistortRectifyMap` using the same original→output mapping.
* Filters the 71,772-point sparse cloud down to **30,100 plant points**, in two
  passes:
  1. **Mask voting** — keep points that land inside the mask in ≥50% of the
     views that observe them.
  2. **Geometric bound** — keep only points within the median camera-orbit
     radius.

### Step 4 — train

```bash
"$AGRI_PY_GS" train_plant_gsplat.py --max-steps 30000
```

A standard, *unmasked* 3DGS fit with gsplat: L1 + SSIM over the whole frame,
SH warm-up, `DefaultStrategy` densification. `--min-sharp-pct 20` applies the
blur filter described above.

### Step 5 — extract the plant

```bash
"$AGRI_PY_GS" extract_plant.py --min-score 0.5 --min-visible 2 --min-opacity 0.01 \
                        --prune-floater-mult 6
```

276,343 → **172,535** Gaussians. `sweep_thresholds.py` is the tool used to
choose the thresholds by measurement rather than taste: it scores each setting
on inside-silhouette MAE (does the plant still reproduce?) against outside-ink
(how much non-plant survives?) over sampled views, and the knee is picked.

### Step 6 — validate

```bash
"$AGRI_PY_GS" validate_ply.py point_cloud.ply
"$AGRI_PY_GS" verify_gsjepa_compatible.py point_cloud.ply
```

---

## 4. Two design decisions that are easy to get wrong

### 4.1 Mask-voting alone cannot find the plant

The obvious way to isolate the plant is "keep every Gaussian that projects
inside the plant mask". It does not work, and the failure is silent.

A point on the wall **behind** the plant projects inside the plant silhouette in
every view where the plant occludes it — and because the camera completes a full
orbit around the plant, the plant is in front of it from *every* angle. Such a
point therefore scores a perfect mask-consistency ratio. Measured on the real
sparse cloud, this test passed 31,075 of 71,772 points with a mean on-mask
fraction of 0.73.

Two things are needed to separate them, and both are implemented:

* **Visibility.** A Gaussian only counts as evidence in views where it is the
  *frontmost* surface. A wall point is never frontmost, so it never accumulates
  in-mask evidence. This is what `extract_plant.py` does, by comparing each
  Gaussian's depth against the rendered expected depth with a relative
  tolerance.
* **A geometric bound.** During initialisation, points outside the swept camera
  volume are dropped. Without this, their 3-nearest-neighbour distance is metres,
  so 3DGS seeds them as splats metres across — and the very first render is
  already a wash of background colour. (Observed: outside-mask render luma 0.3
  before the bound, 0.0 after.)

### 4.2 Training on masked pixels alone is ill-posed

Restricting the photometric loss to the plant silhouette looks like the natural
way to build a plant-only model. It converges to a blur, and the reason is
structural: inside a silhouette mask the model is only ever asked to match 2D
appearance, never to place a surface. Splats are free to smear anywhere within
the silhouette, so that is what they do. Measured behaviour: validation masked
PSNR *fell* from 7.7 to 4.0 over 10k steps while the training loss kept looking
healthy.

The pipeline therefore fits the full frame with a well-posed loss (recovering
the room as well) and extracts the plant afterwards. `train_plant_gsplat.py`
keeps `--mask-mode masked` available — with an explicit alpha-to-zero penalty
outside the mask, which is what a masked loss otherwise lacks — purely so the
two regimes can be compared.

## 5. Known limitation, stated honestly

Full-frame PSNR peaks at **step 5000** (17.12) and declines to 15.97 by step
30,000, while the scene over-densifies to 782,130 Gaussians. The plant region
shows the same shape (16.02 → 14.02). The cause is the densification schedule:
opacity resets plus continued splitting keep adding capacity the data cannot
constrain. The step-5000 checkpoint is therefore the one used for extraction.

The fix (lower `grow_grad2d`, longer `reset_every`, or an earlier stop) was
identified but not applied — doing so means retraining, and this stage's
deliverable is already met by the step-5000 model. It is the obvious first
improvement for a second pass.

The extracted model also retains a small number of non-plant objects that are
genuinely present in the room (a small blue box on the floor shows up under the
pot in the render check) plus some peripheral semi-transparent debris. Tighter
floater pruning was measured (`--prune-floater-mult 4` removes only 5,021 of
172,535 Gaussians, 1.7%, while inside-MAE degrades from 12.8 to 28.5), so a
gentle setting was kept deliberately.

## 6. Output format

`point_cloud.ply` is a standard 3DGS file, binary little-endian, with
`x y z`, `f_dc_0..2`, `f_rest_0..44`, `opacity`, `scale_0..2`, `rot_0..3`.

**The activation convention matters.** Following the 3DGS convention, the values
stored are the *un-activated* optimisation parameters: `opacity` is a **logit**
and `scale_*` is a **log-scale**. Consumers apply `sigmoid`/`exp` themselves —
Gaussian-JEPA does exactly this in `datasets/ShapeNet55Gaussian.py`
(`np_sigmoid(vertex["opacity"])`, `np.exp(scales)`). Exporting
`sigmoid(opacity)` or `exp(scale)` would double-apply the activation and
silently produce a wrong model, so `validate_ply.py` checks the convention
explicitly and reports `opacity_convention` / `scale_convention`.

A PLY containing only `x y z` (or `x y z` + RGB) is a plain point cloud and is
**not** accepted by `validate_ply.py` — that is the specific confusion it exists
to prevent.

Measured on the delivered file:

```
num_gaussians         172535        file_format   binary_little_endian
num_properties        59            num_f_rest    45
required_missing      []            problems      []
is_valid_3dgs         True
bbox_min              [-0.4458, -0.4344, -0.3021]     (normalised scene units)
bbox_max              [ 0.5683,  0.5590,  0.5844]
opacity_range         [-4.595, 18.051]                -> logit (correct)
scale_range           [-13.999, -2.479]               -> log (correct)
scale_linear          [8.33e-07, 0.0839]
quat_norm             [1.0000000, 1.0000001], non-unit: 0
degenerate_scale_count 0
```

`verify_gsjepa_compatible.py` replays Gaussian-JEPA's own
`read_gaussian_attribute` verbatim against the file and gets a **14-wide**
feature vector (3 xyz + 1 opacity + 3 scale + 4 rotation + 3 SH), matching the
`[2, 1024, 14]` shape its release smoke test builds. It reports
`gsjepa_compatible: true`.

## 7. Reproducing

```bash
source env/activate.sh           # sets AGRI_PY_GS / AGRI_PY_JEPA (uv, no conda)
cd <this directory>
"$AGRI_PY_GS" prepare_frames.py
./run_colmap.sh
"$AGRI_PY_GS" build_dataset.py
"$AGRI_PY_GS" train_plant_gsplat.py --max-steps 30000
"$AGRI_PY_GS" extract_plant.py --min-score 0.5 --min-visible 2 --min-opacity 0.01 \
                        --prune-floater-mult 6
"$AGRI_PY_GS" validate_ply.py point_cloud.ply
"$AGRI_PY_GS" verify_gsjepa_compatible.py point_cloud.ply
```

`colmap_io.py` is a self-contained reader for COLMAP's binary sparse format,
written because gsplat's own parser needs `pycolmap`, which is not installed
here and would pull in a numpy version incompatible with this environment.

The large binary artefacts (the scene PLY, the extracted PLY, the checkpoints,
the source frames) are **not** committed to this branch; they live in the data
workspace at `/fj/AgriGaussian-JEPA/data/plant_gs/plant_001/` and are
reproducible with the commands above.
