"""Tests for the Stage 4C guard, audit, annotation benchmark and provenance.

Three properties are worth more than the rest, and each is tested directly
rather than only through a helper:

* **Loading is not evidence.** A well-formed PLY with a passing
  ``jepa_compatibility`` but no verified provenance chain must not be
  ``reconstruction_verified``, and nothing about JEPA must be able to flip it.
* **A model label is never a human label.** Annotating from a prediction, or
  importing a PNG without a confirmation record, must leave the task
  ``annotation_pending`` and produce no accuracy numbers.
* **A carve is traceable to its source rows.** The plant-only verifier must
  fail on a PLY whose *row count* matches but whose *rows* differ -- the exact
  blind spot that let the Cotton/1 incident look healthy.
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

from agri.plant_gs.paa.stage4c import annotation, camera_summary, guard  # noqa: E402
from agri.plant_gs.paa.stage4c import input_quality, plant_only_provenance  # noqa: E402


# --------------------------------------------------------------------------- #
# guard: what counts as evidence
# --------------------------------------------------------------------------- #
def base_entry(**overrides):
    entry = {
        "asset_id": "Broccoli/20260327-10",
        "crop": "Broccoli",
        "plant_id": "10",
        "acquisition_id": "20260327-10",
        "split_group": "Broccoli/10",
        "gaussian_ply_path": "/tmp/full.ply",
        "gaussian_count": 292517,
        "colmap_status": "pass",
        "jepa_compatibility": {"status": "pass"},
        "reconstruction_quality": {"foreground_psnr": 22.9, "silhouette_iou": 0.98,
                                   "background_alpha": 0.004},
        "provenance": {"state": "verified", "checkpoint_sha256": "a" * 64,
                       "ply_sha256": "b" * 64, "checkpoint_step": 27000},
        "plant_only": {},
        "semantics": {},
    }
    entry.update(overrides)
    return entry


def test_verified_chain_and_metrics_earn_reconstruction_verified():
    report = guard.evaluate(base_entry())
    assert report["states"]["reconstruction_verified"] is True
    assert report["eligibility"]["full_specimen_pretrain"] is True


def test_jepa_compatibility_alone_is_not_enough():
    """The Stage 4B blind spot: a PLY that loads is not a PLY that is audited."""
    entry = base_entry(provenance={"state": "absent"})
    report = guard.evaluate(entry)
    assert report["states"]["reconstruction_verified"] is False
    reasons = " ".join(report["reasons"]["full_specimen_pretrain"])
    assert "no checkpoint->PLY provenance record" in reasons
    assert "not a PLY that is audited" not in reasons


@pytest.mark.parametrize("quality,code", [
    ({"foreground_psnr": 12.0, "silhouette_iou": 0.9, "background_alpha": 0.01},
     "held-out reconstruction metrics are below the gate"),
    ({"foreground_psnr": None, "silhouette_iou": 0.9, "background_alpha": 0.01},
     "held-out reconstruction metrics are missing"),
])
def test_gate_and_missing_metrics_both_block(quality, code):
    report = guard.evaluate(base_entry(reconstruction_quality=quality))
    assert report["states"]["reconstruction_verified"] is False
    assert code in report["reasons"]["full_specimen_pretrain"]


def test_model_pseudo_labels_never_verify_semantics():
    entry = base_entry(semantics={
        "available": True,
        "benchmark": {"status": "evaluated", "truth_source": "model_candidate"}})
    report = guard.evaluate(entry)
    assert report["states"]["semantic_verified"] is False


def test_human_confirmed_labels_verify_semantics():
    entry = base_entry(semantics={
        "available": True,
        "benchmark": {"status": "evaluated", "truth_source": "human_confirmed"}})
    report = guard.evaluate(entry)
    assert report["states"]["semantic_verified"] is True


def test_plant_only_requires_semantics_not_just_a_carved_file():
    carved = {"available": True, "provenance_state": "verified"}
    without = guard.evaluate(base_entry(plant_only=carved))
    assert without["states"]["plant_only_verified"] is False
    assert ("semantic separation has no human-confirmed evidence"
            in without["reasons"]["plant_only_pretrain"])

    with_labels = guard.evaluate(base_entry(
        plant_only=carved,
        semantics={"available": True, "benchmark": {
            "status": "evaluated", "truth_source": "human_confirmed"}}))
    assert with_labels["states"]["plant_only_verified"] is True
    assert with_labels["eligibility"]["plant_only_pretrain"] is True


def test_plant_only_with_unverified_carve_is_not_verified():
    entry = base_entry(
        plant_only={"available": True, "provenance_state": "failed"},
        semantics={"available": True, "benchmark": {
            "status": "evaluated", "truth_source": "human_confirmed"}})
    report = guard.evaluate(entry)
    assert report["states"]["plant_only_verified"] is False
    assert ("plant-only asset's own reconstruction/provenance did not verify"
            in report["reasons"]["plant_only_pretrain"])


def test_annotation_role_requires_human_labels():
    pending = guard.evaluate(base_entry(semantics={"available": True, "benchmark": {
        "status": "annotation_pending", "truth_source": "model_candidate"}}))
    assert pending["eligibility"]["annotation_benchmark"] is False
    assert "no human annotation has been imported" in \
        pending["reasons"]["annotation_benchmark"]


# --------------------------------------------------------------------------- #
# guard: library audit and split
# --------------------------------------------------------------------------- #
def test_audit_flags_hash_collision_and_missing_revision():
    a = base_entry(asset_id="A/1", crop="A", plant_id="1", split_group="A/1",
                   provenance={"state": "verified", "ply_sha256": "x" * 64,
                               "checkpoint_sha256": "c" * 64, "checkpoint_step": 1000},
                   source={})
    b = base_entry(asset_id="B/1", crop="B", plant_id="1", split_group="B/1",
                   provenance={"state": "verified", "ply_sha256": "x" * 64,
                               "checkpoint_sha256": "d" * 64, "checkpoint_step": 1000},
                   source={})
    audit = guard.audit_library([a, b])
    codes = {p["code"] for p in audit["problems"]}
    assert "ply_hash_collision" in codes
    assert "missing_source_revision" in codes
    assert audit["clean"] is False


def test_audit_detects_a_plant_placed_in_two_splits():
    a = base_entry(asset_id="A/1", crop="A", plant_id="1", split_group="A/1")
    split = {"groups": {"train": ["A/1"], "test": ["A/1"]}}
    audit = guard.audit_library([a], split=split)
    assert audit["split_clean"] is False
    assert audit["split_leakage"]


def test_make_split_refuses_unknown_and_duplicate_groups():
    entries = [base_entry(asset_id="A/1", crop="A", plant_id="1", split_group="A/1")]
    split = guard.make_split(entries, train=["A/1"], val=["A/1"], test=["Z/9"])
    assert split["unknown_groups"] == ["Z/9"]
    assert split["duplicate_placement"] == {"A/1": ["train", "val"]}
    assert split["leakage"] is True


# --------------------------------------------------------------------------- #
# guard: the loader gate
# --------------------------------------------------------------------------- #
def test_assert_train_eligible_raises_with_reasons():
    with pytest.raises(guard.NotTrainEligible) as excinfo:
        guard.assert_train_eligible(base_entry(provenance={"state": "absent"}))
    assert "no checkpoint->PLY provenance record" in str(excinfo.value)


def test_assert_train_eligible_passes_for_a_verified_asset():
    guard.assert_train_eligible(base_entry())


def test_eligible_assets_splits_and_explains():
    good = base_entry(asset_id="A/1", crop="A", plant_id="1", split_group="A/1")
    bad = base_entry(asset_id="B/1", crop="B", plant_id="1", split_group="B/1",
                     provenance={"state": "failed"})
    eligible, refused = guard.eligible_assets([good, bad])
    assert [e["asset_id"] for e in eligible] == ["A/1"]
    assert refused[0]["asset_id"] == "B/1"
    assert "provenance_failed" in refused[0]["blocking"]


def test_sealed_workspace_is_refused_even_with_a_perfect_chain():
    entry = base_entry(sealed=True)
    report = guard.evaluate(entry)
    assert report["states"]["reconstruction_verified"] is False
    assert "asset_sealed" in report["reason_codes"]["full_specimen_pretrain"]


def test_check_ply_hashes_reports_a_changed_file(tmp_path):
    ply = tmp_path / "p.ply"
    ply.write_bytes(b"first")
    import hashlib
    entry = {"asset_id": "A/1", "gaussian_ply_path": str(ply),
             "provenance": {"ply_sha256": hashlib.sha256(b"first").hexdigest()}}
    assert guard.check_ply_hashes([entry]) == []
    ply.write_bytes(b"second")
    mismatches = guard.check_ply_hashes([entry])
    assert mismatches and mismatches[0]["asset_id"] == "A/1"


# --------------------------------------------------------------------------- #
# annotation: scoring
# --------------------------------------------------------------------------- #
def test_merge_labels_writes_a_summary_naming_the_agreement(tmp_path):
    """Regression: merge() wrote every PNG then died on an undefined name.

    The label directory existed and looked complete, so a re-run skipped the
    merge and every downstream step read labels with no merge provenance.
    """
    from agri.plant_gs.paa import merge_labels
    root = tmp_path
    plant_dir = root / "labels_segformer"
    other_dir = root / "labels_sam_clip"
    plant_dir.mkdir()
    other_dir.mkdir()
    from PIL import Image
    a = np.zeros((8, 8), np.uint8)
    a[:4] = 1
    b = np.zeros((8, 8), np.uint8)
    b[2:] = 1
    Image.fromarray(a).save(plant_dir / "1.png")
    Image.fromarray(b).save(other_dir / "1.png")
    merged = merge_labels.merge(plant_dir, other_dir, root / "labels_merged")
    assert merged["totals"]["frames"] == 1
    assert merged["per_frame"]["1"]["plant_both"] == 16


def test_score_perfect_prediction():
    truth = np.array([[1, 1], [0, 2]], dtype=np.uint8)
    score = annotation.score_against(truth, truth.copy())
    assert score["plant_iou"] == 1.0
    assert score["plant_precision"] == 1.0
    assert score["plant_recall"] == 1.0
    assert score["plant_false_rate"] == 0.0


def test_score_false_plant_rate_counts_nonplant_called_plant():
    truth = np.array([[0, 0, 1, 1]], dtype=np.uint8)
    prediction = np.array([[1, 0, 1, 1]], dtype=np.uint8)
    score = annotation.score_against(truth, prediction)
    assert score["plant_false_positive"] == 1
    assert score["plant_true_negative"] == 1
    assert score["plant_false_rate"] == pytest.approx(0.5)
    assert score["plant_recall"] == 1.0


def test_uncertain_pixels_are_excluded_and_reported():
    truth = np.array([[annotation.UNCERTAIN, 1, 0]], dtype=np.uint8)
    prediction = np.array([[1, 1, 1]], dtype=np.uint8)
    score = annotation.score_against(truth, prediction)
    assert score["pixels_scored"] == 2
    assert score["uncertain_fraction"] == pytest.approx(1 / 3)
    # The uncertain pixel's disagreement must not appear as a false positive.
    assert score["plant_false_positive"] == 1


def test_uncertain_can_be_kept_when_asked():
    truth = np.array([[annotation.UNCERTAIN, 1]], dtype=np.uint8)
    prediction = np.array([[1, 1]], dtype=np.uint8)
    score = annotation.score_against(truth, prediction, exclude_uncertain=False)
    assert score["pixels_scored"] == 2
    assert score["plant_true_positive"] == 1


# --------------------------------------------------------------------------- #
# annotation: tasks, import, evaluate
# --------------------------------------------------------------------------- #
def _fake_dataset(root: Path, frames: int = 12):
    dataset = root / "dataset"
    (dataset / "images").mkdir(parents=True)
    (dataset / "masks").mkdir(parents=True)
    rows = []
    for index in range(frames):
        angle = 2 * np.pi * index / frames
        camtoworld = np.eye(4)
        camtoworld[:3, 3] = [3 * np.cos(angle), 3 * np.sin(angle), 1.0]
        image = dataset / "images" / f"{index:03d}.jpg"
        mask = dataset / "masks" / f"{index:03d}.png"
        from PIL import Image
        Image.fromarray(np.full((32, 24, 3), 200, np.uint8)).save(image)
        Image.fromarray(np.full((32, 24), 255, np.uint8)).save(mask)
        rows.append({"name": f"{index:03d}.jpg", "image": str(image),
                     "mask": str(mask), "camtoworld": camtoworld.tolist(),
                     "K": [[10, 0, 12], [0, 10, 16], [0, 0, 1]]})
    (dataset / "frames.json").write_text(json.dumps(rows))
    semantic = root / "semantic"
    for name in ("labels_merged", "labels_segformer", "labels_sam_clip"):
        (semantic / name).mkdir(parents=True, exist_ok=True)
        for index in range(frames):
            label = np.zeros((32, 24), np.uint8)
            label[8:24, 6:18] = 1
            from PIL import Image
            Image.fromarray(label).save(semantic / name / f"{index:03d}.png")
    return root


def test_export_writes_pending_tasks_and_never_touches_labels(tmp_path):
    workspace = _fake_dataset(tmp_path / "Cabbage_20260328-9")
    library = {"entries": [{
        "asset_id": "Cabbage/20260328-9", "crop": "Cabbage",
        "acquisition_id": "20260328-9", "split_group": "Cabbage/9",
        "workspace": str(workspace),
        "states": {"states": {"reconstruction_verified": True}},
    }]}
    out = tmp_path / "annotation"
    annotation.write_task_set(library, out, views_per_crop=4)
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["n_tasks"] == 4
    assert all(t["status"] == "annotation_pending" for t in manifest["tasks"])
    assert list((out / "labels").glob("*.png")) == []
    assert (out / "preannotations").is_dir()
    assert manifest["status"] == "annotation_pending"
    assert "NOT verified" in manifest["success_state"]


def test_import_rejects_ids_outside_the_taxonomy(tmp_path):
    workspace = _fake_dataset(tmp_path / "Cabbage_20260328-9")
    library = {"entries": [{
        "asset_id": "Cabbage/20260328-9", "crop": "Cabbage",
        "acquisition_id": "20260328-9", "split_group": "Cabbage/9",
        "workspace": str(workspace),
        "states": {"states": {"reconstruction_verified": True}}}]}
    out = tmp_path / "annotation"
    annotation.write_task_set(library, out, views_per_crop=2)
    manifest = json.loads((out / "manifest.json").read_text())
    task_id = manifest["tasks"][0]["task_id"]

    from PIL import Image
    bad = tmp_path / "bad.png"
    Image.fromarray(np.full((4, 4), 9, np.uint8)).save(bad)

    class Args:
        dir = out
        task = task_id
        labels = bad
        annotator = "tester"
        notes = ""

    assert annotation.cmd_import(Args()) == 1
    assert not (out / "labels" / f"{task_id}.png").exists()


def test_import_then_evaluate_produces_metrics(tmp_path):
    workspace = _fake_dataset(tmp_path / "Cabbage_20260328-9")
    library = {"entries": [{
        "asset_id": "Cabbage/20260328-9", "crop": "Cabbage",
        "acquisition_id": "20260328-9", "split_group": "Cabbage/9",
        "workspace": str(workspace),
        "states": {"states": {"reconstruction_verified": True}}}]}
    out = tmp_path / "annotation"
    annotation.write_task_set(library, out, views_per_crop=2)
    manifest = json.loads((out / "manifest.json").read_text())
    task_id = manifest["tasks"][0]["task_id"]

    from PIL import Image
    human = np.zeros((32, 24), np.uint8)
    human[8:24, 6:18] = annotation.PLANT
    human[8:24, 6:8] = annotation.UNCERTAIN
    label_path = tmp_path / "human.png"
    Image.fromarray(human).save(label_path)

    class Args:
        dir = out
        task = task_id
        labels = label_path
        annotator = "alice"
        notes = "first pass"

    assert annotation.cmd_import(Args()) == 0
    assert annotation.cmd_evaluate(type("A", (), {"dir": out,
                                                 "predictions": "preannotations"})()) == 0
    report = json.loads((out / "evaluation.json").read_text())
    assert report["status"] == "evaluated"
    assert report["truth_source"] == "human_confirmed"
    assert report["tasks_human_confirmed"] == 1
    assert report["per_backend"]["labels_merged"]["plant_iou"] == 1.0


def test_unconfirmed_label_png_is_ignored_not_scored(tmp_path):
    workspace = _fake_dataset(tmp_path / "Cabbage_20260328-9")
    library = {"entries": [{
        "asset_id": "Cabbage/20260328-9", "crop": "Cabbage",
        "acquisition_id": "20260328-9", "split_group": "Cabbage/9",
        "workspace": str(workspace),
        "states": {"states": {"reconstruction_verified": True}}}]}
    out = tmp_path / "annotation"
    annotation.write_task_set(library, out, views_per_crop=2)
    manifest = json.loads((out / "manifest.json").read_text())
    task_id = manifest["tasks"][0]["task_id"]
    from PIL import Image
    Image.fromarray(np.full((32, 24), 1, np.uint8)).save(out / "labels" / f"{task_id}.png")

    annotation.cmd_evaluate(type("A", (), {"dir": out, "predictions": "preannotations"})())
    report = json.loads((out / "evaluation.json").read_text())
    assert report["status"] == "annotation_pending"
    assert report["tasks_human_confirmed"] == 0
    assert report["tasks_unconfirmed_ignored"] == 1
    assert report["per_backend"] == {}
    assert "NO segmentation accuracy" in report["caveat"]


# --------------------------------------------------------------------------- #
# plant-only provenance
# --------------------------------------------------------------------------- #
def _ply_vertex(rows: int, seed: int, offset: float = 0.0):
    """A structured vertex array with the 3DGS properties the verifier reads."""
    rng = np.random.default_rng(seed)
    dtype = np.dtype([(name, "f4") for name in
                      ("x", "y", "z", "opacity",
                       "scale_0", "scale_1", "scale_2",
                       "rot_0", "rot_1", "rot_2", "rot_3",
                       "f_dc_0", "f_dc_1", "f_dc_2")])
    vertex = np.zeros(rows, dtype=dtype)
    for name in ("x", "y", "z"):
        vertex[name] = rng.normal(size=rows).astype(np.float32) + offset
    vertex["opacity"] = rng.normal(size=rows).astype(np.float32)
    for index in range(3):
        vertex[f"scale_{index}"] = rng.normal(size=rows).astype(np.float32)
    vertex["rot_3"] = np.float32(1.0)
    for index in range(3):
        vertex[f"f_dc_{index}"] = rng.normal(size=rows).astype(np.float32)
    return vertex


def _write_ply(path: Path, rows: int, seed: int, offset: float = 0.0):
    from plyfile import PlyData, PlyElement
    PlyData([PlyElement.describe(_ply_vertex(rows, seed, offset), "vertex")]).write(str(path))


def _carve_fixture(tmp_path: Path, tamper=False):
    directory = tmp_path / "plant_only"
    directory.mkdir()
    source = tmp_path / "full.ply"
    _write_ply(source, 40, seed=1)
    keep = np.zeros(40, dtype=bool)
    keep[::2] = True
    retained = np.flatnonzero(keep)
    carved = directory / "plant_only.ply"
    from plyfile import PlyData, PlyElement
    subset = np.asarray(PlyData.read(str(source))["vertex"].data)[retained].copy()
    if tamper:
        subset["opacity"] = subset["opacity"] + np.float32(5.0)
    PlyData([PlyElement.describe(subset, "vertex")]).write(str(carved))
    np.savez_compressed(directory / "gaussian_classes.npz", keep=keep)
    import hashlib
    report = {
        "source_ply": str(source),
        "source_ply_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "source_gaussians": 40, "plant_gaussians": int(keep.sum()),
    }
    (directory / "plant_only_report.json").write_text(json.dumps(report))
    return directory


def test_plant_only_provenance_passes_on_a_true_subset(tmp_path):
    directory = _carve_fixture(tmp_path)
    result = plant_only_provenance.verify(directory)
    assert result["status"] == "pass", result["problems"]
    assert result["details"]["plant_only_rows"] == 20


def test_plant_only_provenance_fails_when_rows_differ_but_count_matches(tmp_path):
    """The Cotton/1 failure mode: same length, different numbers."""
    directory = _carve_fixture(tmp_path, tamper=True)
    result = plant_only_provenance.verify(directory)
    assert result["status"] == "fail"
    assert any("opacity" in problem for problem in result["problems"])


def test_plant_only_provenance_fails_without_a_source(tmp_path):
    directory = tmp_path / "plant_only"
    directory.mkdir()
    _write_ply(directory / "plant_only.ply", 5, seed=3)
    np.savez_compressed(directory / "gaussian_classes.npz",
                        keep=np.ones(5, dtype=bool))
    (directory / "plant_only_report.json").write_text(json.dumps({}))
    result = plant_only_provenance.verify(directory)
    assert result["status"] == "fail"
    assert "no source_ply" in result["problems"][0]


# --------------------------------------------------------------------------- #
# input_quality: the sampler arms
# --------------------------------------------------------------------------- #
def _synthetic_asset(rows: int = 500, seed: int = 0):
    rng = np.random.default_rng(seed)
    decoded = np.zeros((rows, 14), dtype=np.float32)
    decoded[:, :3] = rng.normal(size=(rows, 3))
    decoded[:, 3] = rng.normal(size=rows)
    decoded[:, 4:7] = np.abs(rng.normal(size=(rows, 3))) + 0.001
    classes = np.full(rows, 2, dtype=np.int64)
    classes[: int(rows * 0.3)] = 1        # 30% plant
    classes[-5:] = -1                     # a few undecided
    return decoded, classes


def test_plant_aware_draw_uses_the_same_budget_and_only_plant_rows():
    decoded, classes = _synthetic_asset()
    _, indices, _ = input_quality.draw_plant_aware(decoded, classes, 100, seed=7)
    assert len(indices) == 100
    assert set(np.unique(indices)).issubset(set(input_quality.plant_pool(classes)))


def test_undecided_gaussians_are_not_treated_as_plant():
    decoded, classes = _synthetic_asset()
    pool = input_quality.plant_pool(classes)
    assert not (classes[pool] == -1).any()


def test_composition_and_scale_profiles_are_well_formed():
    decoded, classes = _synthetic_asset()
    _, indices, _ = input_quality.draw_uniform(decoded, 200, seed=1)
    composition = input_quality.composition(classes, indices)
    assert sum(composition.values()) == pytest.approx(1.0)
    profile = input_quality.scale_opacity(decoded, indices)
    assert profile["scale_median"] > 0
    assert 0.0 <= profile["small_gaussian_share_sample"] <= 1.0


def test_fine_structure_retention_is_plant_relative():
    decoded, classes = _synthetic_asset()
    pool = input_quality.plant_pool(classes)
    indices = pool[:40]
    result = input_quality.fine_structure(decoded, classes, indices)
    assert 0.0 <= result["fine_population_share"] <= 1.0
    assert result["fine_retention_ratio"] >= 0.0


# --------------------------------------------------------------------------- #
# camera summary
# --------------------------------------------------------------------------- #
def test_camera_summary_reads_models_and_roll(tmp_path):
    workspace = tmp_path / "Cotton_1"
    workspace.mkdir()
    (workspace / "colmap_audit.json").write_text(json.dumps({
        "status": "PASS",
        "cameras": {"0": {"model": "SIMPLE_PINHOLE", "images": 66,
                          "width": 2140, "height": 3788,
                          "horizontal_fov_deg": 12.4, "distortion_free": True}},
        "camera_model_distortion_free": True,
        "camera_models_needing_undistortion": [],
        "camera_models_unsupported": [],
        "trainer_ready_without_undistortion": True,
        "reprojection_error_px": {"mean": 0.5, "p95": 1.8},
    }))
    (workspace / "preparation.json").write_text(json.dumps({
        "frames_used": 102, "frames_rolled_ccw90": 36,
        "registered_images": 102, "reprojection_error_mean_px": 0.586,
        "orbit_radius": 5.75,
    }))
    summary = camera_summary.summarise_sample(
        {"asset_id": "Cotton/1", "crop": "Cotton", "workspace": str(workspace)})
    assert summary["camera_models"] == ["SIMPLE_PINHOLE"]
    assert summary["frames_rolled_ccw90"] == 36
    assert summary["n_cameras"] == 1
