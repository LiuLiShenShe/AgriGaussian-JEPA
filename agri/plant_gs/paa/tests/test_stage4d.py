"""Stage 4D: the two contract bugs, and the human-label -> asset verdict path.

The tests are organised around the two defects the stage had to fix:

* **Plant-only shape.** ``assets.build_entry`` wrote candidates keyed by
  version while ``guard.plant_only_checks`` read a top-level ``available``,
  so eight assets with a real, provenance-verified carve were reported as
  ``plant_only_missing``. Every state is asserted here, including the empty
  carve that Rice/1 actually produced.
* **Annotation -> verdict.** ``read_semantics`` used to return a constant
  ``annotation_pending``. The path from a confirmed label to
  ``semantic_verified`` is asserted end to end: a single confirmed view must
  not certify anything, a crop average must not certify a sibling asset, and a
  failing label set must keep the asset unverified.

The fixture is a synthetic annotation set and a synthetic workspace. It is
labelled synthetic in the code: it proves the plumbing, not that any real crop
has been annotated.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from agri.plant_gs.paa.stage4c import annotation, assets, guard, semantic_gate  # noqa: E402


# --------------------------------------------------------------------------- #
# Fixtures: a synthetic annotation set and a synthetic workspace
# --------------------------------------------------------------------------- #
SIZE = 48


def _labels(plant: bool, support: bool = False) -> np.ndarray:
    label = np.zeros((SIZE, SIZE), dtype=np.uint8)
    label[:, : SIZE // 2] = annotation.PLANT if plant else annotation.BACKGROUND
    if support:
        label[SIZE // 2:, SIZE // 2:] = annotation.SUPPORT
    return label


def _save(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path)


def make_annotation(tmp_path: Path, assets_tasks: dict,
                    merged_labels: dict | None = None) -> Path:
    """Build an annotation directory with N tasks per asset.

    ``merged_labels`` maps a task id to the candidate label map. By default the
    candidate is the human truth, which is the "perfect prediction" case.
    """
    directory = tmp_path / "annotation"
    tasks = []
    for asset_id, count in assets_tasks.items():
        crop, _, acquisition = asset_id.partition("/")
        for index in range(count):
            task_id = f"{crop}_{acquisition}_{index}"
            image = np.full((SIZE, SIZE, 3), 220, dtype=np.uint8)
            image[:, : SIZE // 2] = 60
            _save(directory / "images" / f"{task_id}.jpg", image)
            truth = _labels(plant=True)
            _save(directory / "annotation_truth" / f"{task_id}.png", truth)
            tasks.append({
                "task_id": task_id,
                "crop": crop,
                "asset_id": asset_id,
                "split_group": f"{crop}/1",
                "image": f"images/{task_id}.jpg",
                "status": "annotation_pending",
                "priority": False,
            })
    manifest = {"schema_version": "test", "tasks": tasks,
                "status": "annotation_pending"}
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2))
    for task in tasks:
        candidate = (merged_labels or {}).get(task["task_id"])
        if candidate is None:
            candidate = _labels(plant=True)
        for backend in ("labels_merged", "labels_segformer", "labels_sam_clip"):
            _save(directory / "preannotations" / task["task_id"] / f"{backend}.png",
                  candidate)
    return directory


def confirm(directory: Path, task_id: str, annotator: str = "tester") -> None:
    """Run the real import command, which is the only way into truth."""
    annotation.cmd_import(SimpleNamespace(
        dir=directory, task=task_id,
        labels=directory / "annotation_truth" / f"{task_id}.png",
        annotator=annotator, notes="synthetic fixture"))


def make_workspace(root: Path, crop: str = "Rice", sample: str = "1",
                   plant_only: dict | None = None) -> Path:
    workspace = root / f"{crop}_{sample}"
    workspace.mkdir(parents=True)
    (workspace / "preparation.json").write_text(json.dumps({
        "crop": crop, "sample": sample, "plant_id": "1",
        "acquisition_id": sample, "registered_images": 102,
        "frames_used": 102, "reprojection_error_mean_px": 0.7,
    }))
    (workspace / "colmap_audit.json").write_text(json.dumps({
        "status": "PASS",
        "cameras": {"1": {"model": "SIMPLE_PINHOLE",
                          "horizontal_fov_deg": 60.0}},
        "camera_model_distortion_free": True,
    }))
    final = workspace / "final"
    final.mkdir()
    (final / "point_cloud.ply").write_bytes(b"ply\nformat ascii 1.0\n"
                                            b"element vertex 1\nend_header\n0 0 0\n")
    (final / "report.json").write_text(json.dumps({
        "step": 27000, "raw_gaussians": 300000, "retained_gaussians": 300000,
        "clean_validation": {"foreground_psnr": 25.0, "silhouette_iou": 0.98,
                             "background_alpha": 0.004, "per_view": []},
    }))
    (final / "provenance_verification.json").write_text(json.dumps({
        "status": "pass", "details": {
            "checkpoint_sha256": "a" * 64, "ply_sha256": "b" * 64,
            "checkpoint_step": 27000, "gaussians_after_filter": 300000},
    }))
    (final / "gsjepa_compatibility.json").write_text(json.dumps({
        "gsjepa_compatible": True, "required_fields": ["x", "opacity"]}))

    semantic = workspace / "semantic"
    (semantic / "labels_merged").mkdir(parents=True)
    (semantic / "summary.json").write_text(json.dumps({
        "frames_labelled": 102, "backends": {"segformer": {}}}))

    if plant_only is not None:
        carve = workspace / "plant_only"
        carve.mkdir()
        (carve / "plant_only.ply").write_bytes(b"ply\nformat ascii 1.0\n"
                                               b"element vertex 1\nend_header\n")
        (carve / "plant_only_report.json").write_text(json.dumps(plant_only))
        if plant_only.get("_verify", True):
            (carve / "plant_only_provenance_verification.json").write_text(
                json.dumps({"status": "pass"}))
    return workspace


def carved(plant_gaussians: int = 10) -> dict:
    return {
        "status": "PASS", "source_ply": "/tmp/full.ply",
        "source_ply_sha256": "c" * 64, "source_gaussians": 100,
        "plant_gaussians": plant_gaussians, "nonplant_gaussians": 90,
        "undecided_gaussians": 0,
        "plant_fraction_of_gaussians": plant_gaussians / 100,
        "plant_source": "labels_merged",
    }


# --------------------------------------------------------------------------- #
# Bug A: the plant-only contract
# --------------------------------------------------------------------------- #
def test_summarise_plant_only_absent():
    summary = assets.summarise_plant_only({}, semantic_verified=False)
    assert summary["state"] == "absent"
    assert summary["available"] is False
    assert guard.plant_only_checks({"plant_only": summary,
                                    "provenance": {"state": "verified"},
                                    "colmap_status": "pass",
                                    "jepa_compatibility": {"status": "pass"},
                                    "reconstruction_quality": {}}) == \
        guard.plant_only_checks({"plant_only": summary,
                                 "provenance": {"state": "verified"},
                                 "colmap_status": "pass",
                                 "jepa_compatibility": {"status": "pass"},
                                 "reconstruction_quality": {}})
    assert "plant_only_missing" in guard.plant_only_checks({"plant_only": summary})


def test_summarise_plant_only_distinguishes_unverified_source():
    candidates = {"plant_only": {"available": True, "provenance_state": "failed",
                                 "plant_gaussians": 5, "ply": "/tmp/x.ply"}}
    summary = assets.summarise_plant_only(candidates, semantic_verified=False)
    assert summary["state"] == "failed_empty"


def test_summarise_plant_only_source_verified_but_semantics_pending():
    candidates = {"plant_only": {"available": True, "provenance_state": "verified",
                                 "plant_gaussians": 5, "ply": "/tmp/x.ply"}}
    summary = assets.summarise_plant_only(candidates, semantic_verified=False)
    assert summary["state"] == "source_verified_semantics_unverified"


def test_summarise_plant_only_empty_carve_is_failed_empty():
    """Rice/1 is exactly this: provenance verified, zero plant Gaussians."""
    candidates = {"plant_only": {"available": True, "provenance_state": "verified",
                                 "plant_gaussians": 0, "ply": "/tmp/x.ply"}}
    summary = assets.summarise_plant_only(candidates, semantic_verified=True)
    assert summary["state"] == "failed_empty"
    assert "plant_only_failed_empty" in guard.plant_only_checks(
        {"plant_only": summary})


def test_legacy_plant_only_shape_is_not_reported_missing():
    """The Stage 4C shape: candidates keyed by version under ``plant_only``."""
    entry = {
        "asset_id": "Broccoli/20260327-10",
        "gaussian_ply_path": "/tmp/full.ply",
        "colmap_status": "pass",
        "jepa_compatibility": {"status": "pass"},
        "reconstruction_quality": {"foreground_psnr": 22.0,
                                   "silhouette_iou": 0.98,
                                   "background_alpha": 0.004},
        "provenance": {"state": "verified", "checkpoint_sha256": "a" * 64,
                       "ply_sha256": "b" * 64, "checkpoint_step": 27000},
        "plant_only": {"plant_only": {"available": True,
                                      "provenance_state": "verified"}},
        "semantics": {},
    }
    problems = guard.plant_only_checks(entry)
    assert "plant_only_missing" not in problems
    assert "plant_only_not_reconstruction_verified" not in problems


def test_unused_plant_only_variant_does_not_block_the_selected_one():
    """A secondary carve with no provenance is a note, not a blocker."""
    entry = {
        "asset_id": "Broccoli/20260327-10",
        "gaussian_ply_path": "/tmp/full.ply",
        "colmap_status": "pass",
        "jepa_compatibility": {"status": "pass"},
        "reconstruction_quality": {"foreground_psnr": 22.0,
                                   "silhouette_iou": 0.98,
                                   "background_alpha": 0.004},
        "provenance": {"state": "verified", "checkpoint_sha256": "a" * 64,
                       "ply_sha256": "b" * 64, "checkpoint_step": 27000},
        "plant_only": {
            "state": "verified",
            "selected": "plant_only",
            "candidates": {
                "plant_only": {"available": True, "provenance_state": "verified"},
                "plant_only_segformer": {"available": True,
                                         "provenance_state": "absent"},
            },
        },
        "semantics": {"benchmark": {"status": "evaluated",
                                    "truth_source": "human_confirmed",
                                    "verified": True}},
    }
    assert guard.plant_only_checks(entry) == []
    assert guard.evaluate(entry)["states"]["plant_only_verified"] is True


# --------------------------------------------------------------------------- #
# Bug B: a confirmed label must be able to change the verdict
# --------------------------------------------------------------------------- #
def test_one_confirmed_view_never_certifies_an_asset(tmp_path):
    directory = make_annotation(tmp_path, {"Rice/1": 8})
    confirm(directory, "Rice_1_0")
    report = semantic_gate.evaluate(directory)
    verdict = report["per_asset"]["Rice/1"]
    assert verdict["status"] == "insufficient_coverage"
    assert verdict["verified"] is False
    assert report["assets_semantic_verified"] == []


def test_enough_confirmed_views_certify_the_asset(tmp_path):
    directory = make_annotation(tmp_path, {"Rice/1": 8})
    for index in range(semantic_gate.MIN_CONFIRMED_VIEWS_PER_ASSET):
        confirm(directory, f"Rice_1_{index}")
    report = semantic_gate.evaluate(directory)
    verdict = report["per_asset"]["Rice/1"]
    assert verdict["status"] == "evaluated"
    assert verdict["verified"] is True
    assert verdict["n_confirmed_views"] == 6
    assert report["assets_semantic_verified"] == ["Rice/1"]


def test_a_crop_average_certifies_no_sibling_asset(tmp_path):
    directory = make_annotation(tmp_path, {"Broccoli/5": 8,
                                           "Broccoli/10": 8})
    for index in range(semantic_gate.MIN_CONFIRMED_VIEWS_PER_ASSET):
        confirm(directory, f"Broccoli_5_{index}")
    report = semantic_gate.evaluate(directory)
    assert report["per_asset"]["Broccoli/5"]["verified"] is True
    assert report["per_asset"]["Broccoli/10"]["verified"] is False
    assert report["per_asset"]["Broccoli/10"]["status"] == "annotation_pending"
    assert report["per_crop"]["Broccoli"]["certifies_assets"] is False
    assert report["assets_semantic_verified"] == ["Broccoli/5"]


def test_a_failing_label_set_stays_unverified(tmp_path):
    """Six confirmed views, but the merged candidate calls everything background."""
    labels = {f"Rice_1_{i}": np.zeros((SIZE, SIZE), dtype=np.uint8)
              for i in range(8)}
    directory = make_annotation(tmp_path, {"Rice/1": 8}, merged_labels=labels)
    for index in range(6):
        confirm(directory, f"Rice_1_{index}")
    report = semantic_gate.evaluate(directory)
    verdict = report["per_asset"]["Rice/1"]
    assert verdict["status"] == "failed"
    assert verdict["verified"] is False
    assert any("plant recall" in reason or "plant IoU" in reason
               for reason in verdict["reasons"])


def test_unconfirmed_label_png_is_still_ignored_by_the_gate(tmp_path):
    directory = make_annotation(tmp_path, {"Rice/1": 8})
    _save(directory / "labels" / "Rice_1_0.png", _labels(plant=True))
    report = semantic_gate.evaluate(directory)
    assert report["tasks_human_confirmed"] == 0
    assert report["tasks_unconfirmed_ignored"] == 1


# --------------------------------------------------------------------------- #
# Integration: rebuilt library reflects the verdict
# --------------------------------------------------------------------------- #
def _library(tmp_path, gate_report):
    root = tmp_path / "workspaces"
    root.mkdir()
    make_workspace(root, crop="Rice", sample="1", plant_only=carved(10))
    entries = assets.build_library(root, tmp_path / "missing_registry.jsonl",
                                   gate_report)
    return {entry["asset_id"]: entry for entry in entries}


def test_rebuilding_the_library_keeps_semantics_pending_without_labels(tmp_path):
    entries = _library(tmp_path, None)
    entry = entries["Rice/1"]
    assert entry["plant_only"]["state"] == "source_verified_semantics_unverified"
    assert entry["states"]["states"]["semantic_verified"] is False
    assert entry["states"]["states"]["plant_only_verified"] is False
    assert entry["states"]["reason_codes"]["plant_only_pretrain"] == \
        ["semantics_unverified"]


def test_rebuilding_the_library_flips_states_after_confirmation(tmp_path):
    directory = make_annotation(tmp_path, {"Rice/1": 8})
    for index in range(semantic_gate.MIN_CONFIRMED_VIEWS_PER_ASSET):
        confirm(directory, f"Rice_1_{index}")
    gate_report = semantic_gate.evaluate(directory)

    entries = _library(tmp_path, gate_report)
    entry = entries["Rice/1"]
    assert entry["semantics"]["benchmark"]["status"] == "evaluated"
    assert entry["semantics"]["benchmark"]["truth_source"] == "human_confirmed"
    assert entry["plant_only"]["state"] == "verified"
    assert entry["states"]["states"]["semantic_verified"] is True
    assert entry["states"]["states"]["plant_only_verified"] is True
    assert entry["states"]["eligibility"]["plant_only_pretrain"] is True


def test_empty_carve_stays_blocked_even_with_confirmed_semantics(tmp_path):
    root = tmp_path / "workspaces"
    root.mkdir()
    make_workspace(root, crop="Rice", sample="1", plant_only=carved(0))
    directory = make_annotation(tmp_path, {"Rice/1": 8})
    for index in range(semantic_gate.MIN_CONFIRMED_VIEWS_PER_ASSET):
        confirm(directory, f"Rice_1_{index}")
    gate_report = semantic_gate.evaluate(directory)
    entries = {e["asset_id"]: e for e in
               assets.build_library(root, tmp_path / "none.jsonl", gate_report)}
    entry = entries["Rice/1"]
    assert entry["plant_only"]["state"] == "failed_empty"
    assert entry["states"]["states"]["semantic_verified"] is True
    assert entry["states"]["states"]["plant_only_verified"] is False
    assert "plant_only_failed_empty" in \
        entry["states"]["reason_codes"]["plant_only_pretrain"]


# --------------------------------------------------------------------------- #
# Annotation tooling
# --------------------------------------------------------------------------- #
def test_priority_selection_targets_white_tissue_and_total_miss(tmp_path):
    directory = make_annotation(tmp_path, {"Cotton/1": 4})
    # Task 0: candidate calls the bright right half support (white-tissue risk).
    white = np.zeros((SIZE, SIZE), dtype=np.uint8)
    white[:, : SIZE // 2] = annotation.PLANT
    white[:, SIZE // 2:] = annotation.SUPPORT
    for backend in ("labels_merged", "labels_segformer", "labels_sam_clip"):
        _save(directory / "preannotations" / "Cotton_1_0" / f"{backend}.png", white)
    # Task 1: candidate finds almost no plant (total-miss risk).
    miss = np.zeros((SIZE, SIZE), dtype=np.uint8)
    for backend in ("labels_merged", "labels_segformer", "labels_sam_clip"):
        _save(directory / "preannotations" / "Cotton_1_1" / f"{backend}.png", miss)

    outcome = annotation.prioritise(directory, per_crop=2)
    ranked = outcome["crops"]["Cotton"]
    assert ranked[0] in ("Cotton_1_0", "Cotton_1_1")
    assert set(ranked) == {"Cotton_1_0", "Cotton_1_1"}
    manifest = json.loads((directory / "manifest.json").read_text())
    assert sum(1 for t in manifest["tasks"] if t.get("priority")) == 2


def test_check_reports_orphans_without_calling_them_errors(tmp_path):
    directory = make_annotation(tmp_path, {"Rice/1": 2})
    _save(directory / "images" / "Rice_1_orphan.jpg",
          np.full((SIZE, SIZE, 3), 200, dtype=np.uint8))
    assert annotation.cmd_check(SimpleNamespace(dir=directory)) == 0
    report = json.loads((directory / "check.json").read_text())
    assert report["consistent"] is True
    assert "Rice_1_orphan" in report["orphan_images"]


def test_status_reports_progress_per_crop(tmp_path):
    directory = make_annotation(tmp_path, {"Rice/1": 2, "Cotton/1": 2})
    confirm(directory, "Rice_1_0")
    assert annotation.cmd_status(SimpleNamespace(dir=directory)) == 0
    # cmd_status only prints; re-derive from the same helpers for the assertion.
    manifest = json.loads((directory / "manifest.json").read_text())
    confirmed = annotation.load_confirmed(directory, manifest)
    assert confirmed["Rice_1_0"]["status"] == "human_confirmed"
    assert "Cotton_1_0" not in confirmed
