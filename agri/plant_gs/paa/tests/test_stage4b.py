"""Tests for the Stage 4B additions: identity manifest, label merge, sampling.

These modules carry the decisions that are easy to get quietly wrong, so the
properties that matter are tested directly:

* **The manifest groups by plant, not by acquisition.** ``Broccoli/20260327-10``
  and ``Broccoli/20260402-10`` must land in one split group; counting them as
  two plants would let a splitter leak one plant across the split.
* **The label merge is an intersection for the plant class.** A pixel enters
  the plant class only if both backends call it plant, and the merge never
  invents a class that neither backend proposed.
* **The balanced sampler is spatially fairer than the uniform one.** It must
  cover strictly more occupied cells, and it must not be a no-op.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from agri.plant_gs.paa import identity_manifest, merge_labels  # noqa: E402
from agri.plant_gs.paa.sampling_analysis import (  # noqa: E402
    sample_balanced, sample_uniform)


# --------------------------------------------------------------------------- #
# identity manifest
# --------------------------------------------------------------------------- #
def test_manifest_groups_two_dates_of_one_plant():
    crops = {"Broccoli": ["20260327-10", "20260402-10", "20260327-5"]}
    manifest = identity_manifest.build_manifest(crops)
    assert manifest["n_acquisitions"] == 3
    assert manifest["n_distinct_plants"] == 2
    plants = manifest["plants"]
    assert plants["Broccoli/10"]["acquisitions"] == ["20260327-10", "20260402-10"]
    assert plants["Broccoli/10"]["time_span_days"] == 6
    assert plants["Broccoli/5"]["time_span_days"] == 0


def test_manifest_undated_samples_are_one_plant_each():
    manifest = identity_manifest.build_manifest({"Cotton": ["1", "2", "3"]})
    assert manifest["n_distinct_plants"] == 3
    assert manifest["n_acquisitions"] == 3
    assert manifest["time_span"]["days"] == 0


def test_manifest_time_span_covers_the_dated_crops():
    crops = {
        "Broccoli": ["20260327-10", "20260416-10"],
        "Cotton": ["1"],
    }
    manifest = identity_manifest.build_manifest(crops)
    assert manifest["time_span"]["first"] == "2026-03-27"
    assert manifest["time_span"]["last"] == "2026-04-16"
    assert manifest["time_span"]["days"] == 20
    assert manifest["n_crops"] == 2


def test_registration_report_refuses_to_claim_alignment():
    """Two dates of a plant must not be reported as co-registered."""
    import json
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        for acquisition, date in (("20260327-10", "2026-03-27"),
                                   ("20260402-10", "2026-04-02")):
            workspace = root / acquisition
            workspace.mkdir()
            (workspace / "preparation.json").write_text(json.dumps({
                "acquisition_id": acquisition, "acquisition_date": date,
                "registered_images": 102, "frames_used": 102,
                "orbit_radius": 3.9, "reprojection_error_mean_px": 0.77,
            }))
        report = identity_manifest.registration_report(
            {"Broccoli/10": [str(root / "20260327-10"), str(root / "20260402-10")]})
    entry = report["Broccoli/10"]
    assert entry["n_acquisitions"] == 2
    assert entry["cross_date_registration"]["available"] is False
    assert "NOT in a common coordinate system" in \
        entry["cross_date_registration"]["reason"]


# --------------------------------------------------------------------------- #
# label merge
# --------------------------------------------------------------------------- #
def test_merge_takes_plant_only_where_both_agree(tmp_path):
    from PIL import Image
    from agri.plant_gs.paa.semantic import BACKGROUND, PLANT, POT

    plant_dir, other_dir, out = tmp_path / "p", tmp_path / "o", tmp_path / "m"
    plant_dir.mkdir(); other_dir.mkdir(); out.mkdir()

    a = np.zeros((10, 10), np.uint8)
    a[0:4, 0:4] = PLANT          # both call plant
    a[4:8, 0:4] = POT           # only the plant dir calls it plant
    b = np.zeros((10, 10), np.uint8)
    b[0:4, 0:4] = PLANT         # both call plant
    b[8:10, 0:2] = POT         # only the other dir calls it pot
    Image.fromarray(a).save(plant_dir / "1.png")
    Image.fromarray(b).save(other_dir / "1.png")

    merge_labels.merge(plant_dir, other_dir, out)

    merged = np.asarray(Image.open(out / "1.png"))
    assert (merged[0:4, 0:4] == PLANT).all()      # agreement -> plant
    assert (merged[4:8, 0:4] == BACKGROUND).all()  # disagreement -> abstained
    assert (merged[8:10, 0:2] == POT).all()       # non-plant class carried over


def test_merge_is_an_intersection_for_plant(tmp_path):
    from PIL import Image
    from agri.plant_gs.paa.semantic import PLANT

    plant_dir, other_dir, out = tmp_path / "p", tmp_path / "o", tmp_path / "m"
    plant_dir.mkdir(); other_dir.mkdir(); out.mkdir()
    a = np.full((6, 6), PLANT, np.uint8)
    b = np.zeros((6, 6), np.uint8)
    b[0:2, 0:2] = PLANT
    Image.fromarray(a).save(plant_dir / "1.png")
    Image.fromarray(b).save(other_dir / "1.png")

    merge_labels.merge(plant_dir, other_dir, out)
    merged = np.asarray(Image.open(out / "1.png"))
    assert (merged == PLANT).sum() == 4


# --------------------------------------------------------------------------- #
# sampling
# --------------------------------------------------------------------------- #
def test_uniform_sampler_is_uniform_over_rows():
    rng = np.random.default_rng(0)
    counts = np.zeros(50, np.int64)
    for _ in range(200):
        counts[sample_uniform(rng, 50, 10)] += 1
    assert counts.sum() == 2000
    # No row should be wildly over-represented.
    assert counts.max() < 2000 / 50 * 3


def test_balanced_sampler_is_less_clumped_than_uniform():
    """The point of balancing is that the sample is not piled into one cell.

    A uniform draw over rows is uniform over *Gaussians*, not over *space*: a
    dense region contributes many of the sample and a sparse region almost
    none. The balanced sampler takes at most one point per occupied cell, so
    the busiest cell must end up with fewer points than under uniform sampling.
    """
    rng = np.random.default_rng(0)
    xyz = np.concatenate([
        rng.normal(0, 0.02, size=(4000, 3)),   # a dense blob
        rng.uniform(-1, 1, size=(200, 3)),      # a sparse tail
    ])
    lo, hi = xyz.min(axis=0), xyz.max(axis=0)
    extent = np.maximum(hi - lo, 1e-9)

    def per_cell(points, grid=16):
        cell = np.clip(((points - lo) / extent * grid).astype(int), 0, grid - 1)
        keys = cell[:, 0] * 256 + cell[:, 1] * 16 + cell[:, 2]
        # minlength, or bincount truncates trailing empty cells and the two
        # counts stop being comparable.
        return np.bincount(keys, minlength=grid ** 3)

    uniform = sample_uniform(rng, len(xyz), 1024)
    balanced = sample_balanced(rng, xyz, 1024)

    assert len(np.unique(uniform)) == 1024
    assert len(np.unique(balanced)) == 1024
    assert per_cell(xyz[balanced]).max() < per_cell(xyz[uniform]).max()
    # ... and it spreads over strictly more cells.
    assert (per_cell(xyz[balanced]) > 0).sum() > (per_cell(xyz[uniform]) > 0).sum()


# --------------------------------------------------------------------------- #
# PLY -> splats loader
# --------------------------------------------------------------------------- #
def test_ply_loader_restores_the_sh_channel_major_layout(tmp_path):
    """The PLY's f_rest fields are channel-major; getting it wrong is silent.

    A wrong layout still loads and still renders -- the Gaussian is just
    carrying somebody else's spherical harmonics, which shows up as a PSNR
    loss that has nothing to do with the experiment under test. So the layout
    is pinned here against a synthetic PLY whose true layout is known.
    """
    from plyfile import PlyData, PlyElement
    from agri.plant_gs.paa.plant_only import load_splats_from_ply

    n, coefficients = 4, 15
    rng = np.random.default_rng(0)
    sh = rng.normal(size=(n, coefficients, 3)).astype(np.float32)
    fields = [("x", "f4"), ("y", "f4"), ("z", "f4"),
              ("f_dc_0", "f4"), ("f_dc_1", "f4"), ("f_dc_2", "f4"),
              ("opacity", "f4"), ("scale_0", "f4"), ("scale_1", "f4"),
              ("scale_2", "f4"), ("rot_0", "f4"), ("rot_1", "f4"),
              ("rot_2", "f4"), ("rot_3", "f4")]
    fields += [(f"f_rest_{c * coefficients + k}", "f4")
               for c in range(3) for k in range(coefficients)]
    dtype = [(name, kind) for name, kind in fields]
    vertex = np.zeros(n, dtype=dtype)
    for i in range(n):
        vertex[i]["x"] = i
        vertex[i]["f_dc_0"] = 0.1 * i
    # channel-major: f_rest_{c*15+k} is coefficient k of channel c
    for c in range(3):
        for k in range(coefficients):
            vertex[f"f_rest_{c * coefficients + k}"] = sh[:, k, c]

    path = tmp_path / "test.ply"
    PlyData([PlyElement.describe(vertex, "vertex")], text=False).write(str(path))

    splats, degree, count = load_splats_from_ply(path, "cpu")
    assert count == n
    assert degree == 3
    restored = splats["shN"].detach().numpy()
    assert np.abs(restored - sh).max() < 1e-6
    assert np.abs(splats["sh0"].detach().numpy()[:, 0, 0]
                  - 0.1 * np.arange(n)).max() < 1e-6
