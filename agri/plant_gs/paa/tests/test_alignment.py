"""Tests for the PAA alignment module: the crop that registers masks to frames.

These use tiny synthetic fixtures so the geometric claim is checked without
depending on any download:

* ``Camera/images`` smaller than the mask by a different ratio per camera
  -> the relationship is a crop, and the offsets are recovered by NCC, not
  assumed.
* Masks are in the *display* orientation of their source frame while COLMAP's
  intrinsics describe the *stored* frame -> the crop must be composed with the
  EXIF-undoing rotation.
* An EXIF-rotated image cannot be matched by comparing display-orientation
  arrays directly.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from agri.plant_gs.paa.alignment import (  # noqa: E402
    crop_mask_to_camera_frame, mask_vs_whitebg, match_template_offset,
    rotate_to_raw)
from agri.plant_gs.paa.prepare_paa import (  # noqa: E402
    SIM_MIN_INLIERS, SIM_RMS_MAX_PX, resolve_crops)


def make_photo(height, width, seed=0, background=200):
    rng = np.random.default_rng(seed)
    arr = np.full((height, width, 3), background, np.uint8)
    # a distinguishable blob near the centre
    y, x = np.ogrid[:height, :width]
    arr[(y - height // 2) ** 2 + (x - width // 2) ** 2 < (min(height, width) // 3) ** 2] = \
        (30, 160, 60)
    return arr


def save_photo(path, arr, exif_orientation=None):
    image = Image.fromarray(arr)
    if exif_orientation is not None:
        exif = image.getexif()
        exif[274] = exif_orientation
        path = path.with_suffix(".jpg")
        image.save(path, exif=exif, quality=95)
    else:
        image.save(path)
    return path


# --------------------------------------------------------------------------- #
# crop recovery
# --------------------------------------------------------------------------- #
def test_match_template_offset_recovers_crop():
    big = make_photo(300, 400, seed=1)
    crop = big[40:40 + 220, 60:60 + 150].copy()      # offset (dx=60, dy=40)
    dx, dy, ncc = match_template_offset(crop, big)
    assert (dx, dy) == (60, 40)
    assert ncc > 0.95


def test_match_template_offset_refuses_when_smaller_than_source():
    big = make_photo(100, 100)
    small = make_photo(150, 150)
    assert match_template_offset(small, big) is None


def test_match_template_offset_handles_zero_offset():
    big = make_photo(200, 240, seed=2)
    crop = big.copy()
    dx, dy, ncc = match_template_offset(crop, big)
    assert (dx, dy) == (0, 0)
    assert ncc > 0.98


# --------------------------------------------------------------------------- #
# orientation
# --------------------------------------------------------------------------- #
def test_rotate_to_raw_undoes_exif_orientation():
    arr = make_photo(300, 400, seed=3)
    for orientation, expand in ((6, True), (8, True), (3, True), (None, False), (1, False)):
        # PIL's rotate(-angle) is the inverse of the viewer's +angle; the
        # module uses rotate(90) for orientation 6, which is what PIL itself
        # applies when honouring EXIF 6, so the round trip must be exact.
        viewer = Image.fromarray(arr)
        if orientation in (6, 8, 3):
            angle = {6: -90, 8: 90, 3: 180}[orientation]
            viewer = viewer.rotate(angle, expand=expand)
        back = rotate_to_raw(viewer, orientation)
        assert np.asarray(back).shape == arr.shape
        assert np.array_equal(np.asarray(back), arr), orientation


def test_exif_rotated_image_needs_undoing_to_match(tmp_path):
    """The stored frame, not the viewer frame, is what COLMAP solved on."""
    stored = make_photo(200, 120, seed=4)                 # landscape, EXIF-ignored
    viewer_arr = np.asarray(Image.fromarray(stored).rotate(-90, expand=True))
    # Camera frames are the smaller ones in PAA; make a crop of the stored frame
    crop = stored[20:20 + 160, 10:10 + 100].copy()
    # Matching against the stored frame works.
    dx, dy, ncc = match_template_offset(crop, stored)
    assert (dx, dy) == (10, 20)
    # Matching the same crop against the viewer-orientation frame must fail or
    # score badly, which is exactly the failure mode this module guards against.
    bad = match_template_offset(crop, viewer_arr)
    assert bad is None or bad[2] < 0.5


# --------------------------------------------------------------------------- #
# mask alignment
# --------------------------------------------------------------------------- #
def test_mask_vs_whitebg_reports_none_for_shape_mismatch():
    mask = np.zeros((100, 100), np.uint8)
    mask[20:80, 20:80] = 255
    whitebg = np.full((120, 100, 3), 255, np.uint8)
    assert mask_vs_whitebg(mask, whitebg) is None


def test_mask_vs_whitebg_measures_agreement():
    mask = np.zeros((100, 100), np.uint8)
    mask[20:80, 20:80] = 255
    whitebg = np.full((100, 100, 3), 255, np.uint8)
    whitebg[20:80, 20:80] = (30, 160, 60)
    result = mask_vs_whitebg(mask, whitebg)
    assert result["iou"] == pytest.approx(1.0)
    # Shrink the mask so it covers only half the object.
    half = np.zeros((100, 100), np.uint8)
    half[20:50, 20:80] = 255
    partial = mask_vs_whitebg(half, whitebg)
    assert 0.3 < partial["iou"] < 0.7


def test_crop_mask_to_camera_frame_applies_rotation_and_offset():
    """A portrait mask plus an EXIF-6 frame must yield the landscape camera frame."""
    # The mask is authored in the DISPLAY (portrait) orientation.
    display = np.zeros((400, 300), np.uint8)          # H=400, W=300
    display[60:340, 40:260] = 255
    # After the undo rotation the array is 300x400 (H, W), so a camera frame of
    # 300 wide by 200 tall fits inside it at offset (40, 60).
    cropped = crop_mask_to_camera_frame(display, 6, dx=40, dy=60,
                                        width=300, height=200)
    assert cropped.shape == (200, 300)
    assert cropped.dtype == np.uint8
    # The mask must survive the rotation: the bright band is still present.
    assert (cropped > 127).mean() > 0.5


def test_crop_mask_to_camera_frame_rejects_too_small_a_mask():
    tiny = np.zeros((50, 50), np.uint8)
    with pytest.raises(ValueError, match="cannot yield"):
        crop_mask_to_camera_frame(tiny, None, 0, 0, 100, 100)


def test_crop_mask_matches_crop_of_rotated_image():
    """The mask crop must equal the same crop taken after the same rotation."""
    display = make_photo(400, 300, seed=5, background=0)
    display[:] = np.asarray(Image.fromarray(display).convert("L"))[:, :, None]
    # reference: rotate then crop, as a viewer-independent implementation would
    reference = np.asarray(Image.fromarray(display[:, :, 0]).rotate(90, expand=True))[30:30+180, 50:50+120]
    got = crop_mask_to_camera_frame(display[:, :, 0], 6, 50, 30, 120, 180)
    assert np.array_equal(got, reference)


# --------------------------------------------------------------------------- #
# Stage 4B: the mask crop offsets are measured per sample.
#
# The Stage 4A bug this closes: offsets measured on Broccoli/20260327-10 were
# the default for every other sample, so a plant whose camera framing had
# shifted would have been silently registered with the wrong mask.
# --------------------------------------------------------------------------- #
class _FakeAlignment:
    """Just the shape resolve_crops reads out of a SampleAlignment."""

    def __init__(self, crops):
        self.crops = crops


def _strong_crop(tx, ty, support=100, total=100, inliers=2500, rms=1.2):
    """A fit in the shape :func:`resolve_transforms` actually reads.

    The Stage 4B pipeline measures a per-camera *similarity* (ORB + RANSAC +
    constrained similarity fit) between the white-background rendering and the
    camera frame, not an integer crop. The schema below is the one
    ``alignment.probe_sample`` writes: per-parameter medians plus the
    cross-frame spread and the weakest frame's correspondence count.
    """
    return {"sx": 1.0, "sy": 1.0, "tx": tx, "ty": ty, "rms": rms,
            "support_images": support, "total_images": total,
            "inliers_min": inliers, "matches_min": inliers + 100,
            "spread": {"sx": 0.001, "sy": 0.001, "tx": 0.2, "ty": 0.2},
            "A_iou_median": 0.99, "A_iou_min": 0.97}


def test_resolve_crops_uses_the_measurement_of_this_sample():
    alignment = _FakeAlignment({0: _strong_crop(16, 24), 1: _strong_crop(440, 80)})
    assert resolve_crops("20260402-10", alignment, trust_defaults=False) == \
        {0: (16, 24), 1: (440, 80)}


def test_resolve_crops_does_not_inherit_another_samples_offsets():
    """A different measured offset is accepted as-is, not corrected to 16/24."""
    alignment = _FakeAlignment({0: _strong_crop(3, 7)})
    assert resolve_crops("20260402-10", alignment, trust_defaults=False) == {0: (3, 7)}


def test_resolve_crops_refuses_a_weak_correspondence():
    """Too few surviving correspondences means the mask was never matched."""
    alignment = _FakeAlignment({0: _strong_crop(16, 24, inliers=SIM_MIN_INLIERS - 1)})
    with pytest.raises(SystemExit, match="inliers"):
        resolve_crops("20260402-10", alignment, trust_defaults=False)


def test_resolve_crops_refuses_a_loose_fit():
    """A high-residual similarity is not a registration."""
    alignment = _FakeAlignment({0: _strong_crop(16, 24, rms=SIM_RMS_MAX_PX + 0.1)})
    with pytest.raises(SystemExit, match="residual"):
        resolve_crops("20260402-10", alignment, trust_defaults=False)


def test_resolve_crops_refuses_a_split_vote():
    """Half the frames agreeing is not a determined offset."""
    alignment = _FakeAlignment({0: _strong_crop(16, 24, support=40, total=100)})
    with pytest.raises(SystemExit, match="not determined"):
        resolve_crops("20260402-10", alignment, trust_defaults=False)


def test_resolve_crops_trust_defaults_only_for_the_reference_sample():
    alignment = _FakeAlignment({0: _strong_crop(16, 24), 1: _strong_crop(440, 80)})
    # The Stage 4A sample is the only one allowed to lean on the reference.
    assert resolve_crops("20260327-10", alignment, trust_defaults=True) == \
        {0: (16, 24), 1: (440, 80)}
    with pytest.raises(SystemExit, match="only valid for"):
        resolve_crops("20260402-10", alignment, trust_defaults=True)


def test_resolve_crops_rejects_reference_values_the_data_disagrees_with():
    """--trust-default-crops verifies; it does not override the measurement."""
    alignment = _FakeAlignment({0: _strong_crop(3, 7), 1: _strong_crop(440, 80)})
    with pytest.raises(SystemExit, match="disagrees"):
        resolve_crops("20260327-10", alignment, trust_defaults=True)
