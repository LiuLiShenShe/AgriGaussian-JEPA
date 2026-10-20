"""Tests for the PAA Stage 4A ingestion layer.

Two properties here are worth more than they look, so both are tested
directly rather than through the pipeline:

* **The 90-degree camera roll is an exact inverse of the 90-degree image roll.**
  If those two disagree, a rolled frame still loads and still trains -- it just
  trains a plant that is not the one in its photograph. A synthetic pixel that
  is projected through the rolled camera must land on the same rotated pixel
  that the rolled image produced.

* **Split grouping uses plant identity, not acquisition.**
  A dated sample id splits into a plant id and a date, and the split group is
  the plant id. A test that only checked ``parse_sample_id`` would still pass
  if someone later made ``group_key`` use the acquisition id.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from agri.plant_gs.paa import registry  # noqa: E402
from agri.plant_gs.paa.prepare_paa import (  # noqa: E402
    R_ROLL_CCW90, roll_camera_ccw90, roll_image_ccw90)


# --------------------------------------------------------------------------- #
# sample-id parsing
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("sample,plant,date,dated", [
    ("20260327-10", "10", "2026-03-27", True),
    ("20260402-10", "10", "2026-04-02", True),
    ("20260401-23", "23", "2026-04-01", True),
    ("3", "3", None, False),
    ("12", "12", None, False),
])
def test_parse_sample_id(sample, plant, date, dated):
    parsed = registry.parse_sample_id(sample)
    assert parsed["plant_id"] == plant
    assert parsed["acquisition_date"] == date
    assert parsed["dated"] is dated
    assert parsed["acquisition_id"] == sample


def test_split_groups_plant_not_acquisition():
    """Same plant on two dates must share one split group."""
    early = registry.parse_sample_id("20260327-10")
    late = registry.parse_sample_id("20260402-10")
    assert early["acquisition_id"] != late["acquisition_id"]
    assert registry.group_key("Broccoli", early["plant_id"]) == \
        registry.group_key("Broccoli", late["plant_id"])
    # Different plants must NOT share a group.
    assert registry.group_key("Broccoli", "10") != registry.group_key("Broccoli", "5")


def test_check_split_detects_leak():
    reg = registry.Registry(Path("/nonexistent.jsonl"))
    leak = reg.check_split(["Broccoli/10", "Broccoli/5"], ["Broccoli/5", "Rice/3"])
    assert leak == ["Broccoli/5"]
    assert reg.check_split(["Broccoli/10"], ["Broccoli/5"]) == []


def test_undated_samples_do_not_share_groups():
    for sample in ("1", "2", "3"):
        parsed = registry.parse_sample_id(sample)
        assert registry.group_key("Cotton", parsed["plant_id"]) == f"Cotton/{sample}"


# --------------------------------------------------------------------------- #
# camera roll == image roll
# --------------------------------------------------------------------------- #
def test_roll_camera_matches_roll_image():
    """A pixel tracked through the rolled camera lands where the rolled image put it.

    ``roll_image_ccw90`` is the ground truth for where a pixel ends up; the
    rolled camera has to agree, or every rolled frame is mis-registered.
    """
    width, height = 640, 480
    K = np.array([[800.0, 0.0, 300.0],
                  [0.0, 800.0, 200.0],
                  [0.0, 0.0, 1.0]])
    c2w = np.eye(4)
    c2w[:3, :3] = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    c2w[:3, 3] = [0.3, -0.2, 1.5]

    K_new, c2w_new, new_w, new_h = roll_camera_ccw90(K, c2w, width, height)
    assert (new_w, new_h) == (height, width)

    # A synthetic image with one bright pixel, rolled the way images are rolled.
    rng = np.random.default_rng(0)
    pixels = rng.integers(0, 255, size=(height, width, 3), dtype=np.uint8)
    from PIL import Image
    rolled = np.asarray(roll_image_ccw90(Image.fromarray(pixels)))
    # roll_camera_ccw90 returns a PIL-style (width, height); numpy arrays are
    # (rows, cols, ...). Keeping that straight is exactly the trap this test is
    # here to catch, so assert both orderings explicitly.
    assert (new_w, new_h) == (height, width)
    assert rolled.shape == (new_h, new_w, 3) == (width, height, 3)

    # Unproject a pixel to a world ray, reproject with the rolled camera, and
    # check the two views of the same 3D point agree on the same ray.
    def ray_of(K, px, py):
        d = np.array([(px - K[0, 2]) / K[0, 0],
                      (py - K[1, 2]) / K[1, 1], 1.0])
        return d / np.linalg.norm(d)

    def reproject(K, c2w, direction_world, distance=1.7):
        w2c = np.linalg.inv(c2w)
        point = c2w[:3, 3] + distance * direction_world
        cam = point @ w2c[:3, :3].T + w2c[:3, 3]
        return (K[0, 0] * cam[0] / cam[2] + K[0, 2],
                K[1, 1] * cam[1] / cam[2] + K[1, 2], cam[2])

    # Sample interior pixels that survive both frames.
    for px, py in [(320, 240), (100, 60), (500, 400), (640 - 1, 480 - 1), (0, 0)]:
        direction = c2w[:3, :3] @ ray_of(K, px, py)
        # where does that 3D point land in the rolled view?
        u_new, v_new, depth = reproject(K_new, c2w_new, direction)
        if not (0 <= u_new < new_w and 0 <= v_new < new_h and depth > 0):
            continue
        # Where the rolled camera looks, the rolled image must show the same
        # pixel the original camera saw. This is the whole point of the test: it
        # is the only check that a rolled frame is still registered to the right
        # photograph, and it holds because Image.rotate(90) sends (x, y) ->
        # (y, width-1-x) while the rolled camera sends a ray from (px, py) to
        # exactly that same pixel.
        assert rolled[int(round(v_new)), int(round(u_new))].tolist() == \
            pixels[py, px].tolist(), (px, py, u_new, v_new)


def test_roll_camera_is_orthonormal():
    K = np.array([[900.0, 0.0, 400.0], [0.0, 900.0, 250.0], [0.0, 0.0, 1.0]])
    c2w = np.eye(4)
    c2w[:3, :3] = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
    _, c2w_new, _, _ = roll_camera_ccw90(K, c2w, 800, 600)
    R = c2w_new[:3, :3]
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-12)
    assert np.isclose(np.linalg.det(R), 1.0, atol=1e-12)
    assert np.allclose(R_ROLL_CCW90 @ R_ROLL_CCW90.T, np.eye(3))


def test_roll_camera_swaps_axes_consistently():
    # Square focal length: the axis-swap assertion below then holds for both
    # axes at once, so it cannot pass for the wrong reason.
    K = np.array([[900.0, 0.0, 400.0], [0.0, 900.0, 250.0], [0.0, 0.0, 1.0]])
    c2w = np.eye(4)
    K_new, _, new_w, new_h = roll_camera_ccw90(K, c2w, 800, 600)
    # An 800x600 (W x H) frame becomes 600x800.
    assert (new_w, new_h) == (600, 800)
    # Exactly where the principal point lands: rolling about the optical axis
    # does NOT keep it centred unless it already was, so assert the mapping
    # rather than a "still centred" property that happens to be false here
    # (cx=400 is centred for width 800, but cy=250 is not for height 600).
    assert np.isclose(K_new[0, 2], K[1, 2])
    assert np.isclose(K_new[1, 2], 800 - 1 - K[0, 2])
    # Focal lengths: the old vertical focal length now samples horizontally,
    # and the old horizontal one samples vertically.
    assert np.isclose(K_new[0, 0], K[1, 1])
    assert np.isclose(K_new[1, 1], K[0, 0])


# --------------------------------------------------------------------------- #
# record validation
# --------------------------------------------------------------------------- #
def test_accepted_record_needs_evidence():
    record = registry.PlantRecord(crop="Broccoli", plant_id="10",
                                  acquisition_id="20260327-10")
    record.acceptance_status = "accepted"
    problems = record.validate()
    assert any("colmap_status" in p for p in problems)
    assert any("gaussian_ply_path" in p for p in problems)
    assert any("gaussian_count" in p for p in problems)
    assert any("jepa" in p for p in problems)


def test_valid_accepted_record_has_no_problems():
    record = registry.PlantRecord(
        crop="Broccoli", plant_id="10", acquisition_id="20260327-10",
        colmap_status="pass", gaussian_ply_path="/tmp/x.ply", gaussian_count=100,
        jepa_compatibility={"status": "pass"})
    assert record.validate() == []


def test_registry_rejects_invalid_records(tmp_path):
    reg = registry.Registry(tmp_path / "registry.jsonl")
    bad = registry.PlantRecord(crop="", plant_id="", acquisition_id="x")
    with pytest.raises(ValueError):
        reg.append(bad)
    assert not (tmp_path / "registry.jsonl").exists()


def test_registry_roundtrip_and_groups(tmp_path):
    reg = registry.Registry(tmp_path / "registry.jsonl")
    for sample, date in (("20260327-10", "2026-03-27"), ("20260402-10", "2026-04-02")):
        record = registry.PlantRecord(
            crop="Broccoli", plant_id="10", acquisition_date=date,
            acquisition_id=sample, colmap_status="pass",
            gaussian_ply_path="/tmp/x.ply", gaussian_count=10,
            jepa_compatibility={"status": "pass"}, acceptance_status="accepted")
        reg.append(record)
    groups = reg.groups()
    assert groups == {"Broccoli/10": ["20260327-10", "20260402-10"]}
    rows = reg.read()
    assert len(rows) == 2
    assert all(r["split_group"] == "Broccoli/10" for r in rows)
    # A splitter that groups by sample directory would see two groups.
    assert len({r["acquisition_id"] for r in rows}) == 2


def test_record_from_workspace_marks_unknown_not_accepted(tmp_path):
    record = registry.record_from_workspace(tmp_path, "20260327-10", "Broccoli")
    assert record.acceptance_status == "pending"
    assert record.gaussian_ply_path is None
    assert record.notes  # something is missing, and it is recorded
    parsed = registry.parse_sample_id("20260327-10")
    assert record.plant_id == parsed["plant_id"]
    assert record.acquisition_date == parsed["acquisition_date"]


# --------------------------------------------------------------------------- #
# Stage 4B: registration must be idempotent per acquisition, and lossless
# about which plant/date/source each row describes.
# --------------------------------------------------------------------------- #
def _accepted(sample, date, crop="Broccoli", plant="10", count=10, status="accepted",
              revision=None):
    return registry.PlantRecord(
        crop=crop, plant_id=plant, acquisition_date=date, acquisition_id=sample,
        colmap_status="pass", gaussian_ply_path=f"/tmp/{sample}.ply",
        gaussian_count=count, jepa_compatibility={"status": "pass"},
        acceptance_status=status, source_revision=revision)


def test_upsert_does_not_duplicate_an_acquisition(tmp_path):
    """Re-running one sample must leave one row, not two."""
    reg = registry.Registry(tmp_path / "registry.jsonl")
    reg.upsert(_accepted("20260327-10", "2026-03-27", count=292517, revision="aaa"))
    outcome = reg.upsert(_accepted("20260327-10", "2026-03-27", count=331903, revision="bbb"))

    assert outcome["status"] == "updated"
    assert outcome["replaced"] == 1
    rows = reg.read()
    assert len(rows) == 1
    assert rows[0]["gaussian_count"] == 331903
    assert rows[0]["source_revision"] == "bbb"
    # The superseded build is not silently discarded.
    assert rows[0]["history"][0]["gaussian_count"] == 292517
    assert rows[0]["history"][0]["source_revision"] == "aaa"


def test_upsert_keeps_different_acquisitions_of_one_plant(tmp_path):
    """Dedup is per acquisition; the two dates of plant 10 both survive."""
    reg = registry.Registry(tmp_path / "registry.jsonl")
    reg.upsert(_accepted("20260327-10", "2026-03-27"))
    reg.upsert(_accepted("20260402-10", "2026-04-02"))
    assert len(reg.read()) == 2
    assert reg.groups() == {"Broccoli/10": ["20260327-10", "20260402-10"]}

    # Keyed by acquisition, not by plant: re-registering one date must not
    # touch the other.
    reg.upsert(_accepted("20260402-10", "2026-04-02", count=99))
    rows = {r["acquisition_id"]: r for r in reg.read()}
    assert len(rows) == 2
    assert rows["20260327-10"]["gaussian_count"] == 10
    assert rows["20260402-10"]["gaussian_count"] == 99


def test_dedupe_collapses_a_legacy_append_only_registry(tmp_path):
    """Rows written by the old append-only path collapse to the latest."""
    path = tmp_path / "registry.jsonl"
    reg = registry.Registry(path)
    reg.append(_accepted("20260327-10", "2026-03-27", count=1))
    reg.append(_accepted("20260327-10", "2026-03-27", count=2))
    assert len(reg.read()) == 2
    latest = reg.latest()
    assert len(latest) == 1
    assert latest["Broccoli/20260327-10"]["gaussian_count"] == 2


def test_record_carries_source_provenance(tmp_path):
    """The row must name its own source revision, not point at a file."""
    record = registry.record_from_workspace(
        tmp_path, "20260327-10", "Broccoli", revision="19e7f2f2",
        source={"hub_sha": "19e7f2f2", "file_counts": {"camera": 102}})
    payload = record.to_dict()
    assert payload["source_revision"] == "19e7f2f2"
    assert payload["source"]["file_counts"] == {"camera": 102}
    assert payload["plant_id"] == "10"
    assert payload["acquisition_id"] == "20260327-10"
    assert payload["split_group"] == "Broccoli/10"
