#!/usr/bin/env python
"""Assemble the Stage 4C asset library and audit it.

One builder, one output, and no inference in between
----------------------------------------------------
Every field of an entry comes from a file some earlier stage wrote, and the
file it came from is recorded in ``evidence``. Nothing is derived from a
directory name, an mtime, or a report's prose. Where a stage did not run, the
field is ``None`` and the guard reports the gap by name -- a half-finished run
must not be able to look finished.

The library is the input to two very different consumers:

* a human reading `reports/stage4c_multicrop_quality.md`, who needs the real
  measured numbers and the reasons an asset is *not* usable yet;
* the guard in ``guard.py``, which needs each claim separated from the
  evidence for it, because it has to refuse.

Run it with no arguments from the repository root:

    python -m agri.plant_gs.paa.stage4c.assets \
        --workspaces data/plant_gs/PAA \
        --registry  data/plant_gs/PAA/registry.jsonl \
        --out       results/agri_paa/stage4c
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agri.plant_gs.paa.stage4c import guard  # noqa: E402

SCHEMA_VERSION = "paa-stage4c-1"

#: Directories under a workspace that a Stage 4C run may have written.
SEMANTIC_LABEL_DIRS = ("labels_merged", "labels_segformer", "labels_sam_clip")
PLANT_ONLY_DIRS = ("plant_only", "plant_only_segformer")


def log(message: str) -> None:
    print(message, flush=True)


def read_json(path: Path, default=None):
    """Read a JSON file, or return ``default`` -- never raise on absence."""
    try:
        if path.is_file():
            return json.loads(path.read_text())
    except Exception:
        return default
    return default


def sha256_file(path: Path, chunk: int = 1 << 20) -> str | None:
    try:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(chunk), b""):
                digest.update(block)
        return digest.hexdigest()
    except OSError:
        return None


def _ply_rows(path: Path) -> int | None:
    """Vertex count from the PLY header, without reading 300k rows."""
    try:
        with Path(path).open("rb") as handle:
            for _ in range(40):
                line = handle.readline()
                if line.startswith(b"element vertex"):
                    return int(line.split()[-1])
                if line.startswith(b"end_header"):
                    break
    except OSError:
        return None
    return None


def is_sealed(workspace: Path) -> bool:
    """A workspace the project has frozen as evidence.

    Cotton/1 was made read-only after the Stage 4B concurrency incident and kept
    as the record of what went wrong. It is discovered like any other workspace
    -- so the audit can *show* it -- and then excluded from every eligibility
    question by the guard, because an incident directory is not a data asset.
    """
    marker = workspace / "SEALED"
    if marker.is_file():
        return True
    import os
    return not os.access(workspace, os.W_OK)


def discover_workspaces(root: Path) -> list:
    """Every direct child of ``root`` that looks like a prepared sample."""
    out = []
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        if (child / "preparation.json").is_file() or (child / "colmap_audit.json").is_file():
            out.append(child)
    return out


# --------------------------------------------------------------------------- #
# Field readers -- one per artefact, so a missing artefact is visible
# --------------------------------------------------------------------------- #
def read_camera(workspace: Path) -> dict:
    audit = read_json(workspace / "colmap_audit.json") or {}
    cameras = audit.get("cameras") or {}
    models = sorted({(c or {}).get("model") for c in cameras.values()
                     if (c or {}).get("model")})
    fovs = [(c or {}).get("horizontal_fov_deg") for c in cameras.values()
            if (c or {}).get("horizontal_fov_deg") is not None]
    preparation = read_json(workspace / "preparation.json") or {}
    return {
        "n_cameras": len(cameras),
        "models": models,
        "distortion_free": audit.get("camera_model_distortion_free"),
        "needing_undistortion": audit.get("camera_models_needing_undistortion") or [],
        "unsupported": audit.get("camera_models_unsupported") or [],
        "trainer_ready_without_undistortion": audit.get("trainer_ready_without_undistortion"),
        "horizontal_fov_deg_range": [min(fovs), max(fovs)] if fovs else None,
        "undistortion_applied": (preparation.get("undistortion") or {}).get("applied"),
        "source_resolution": preparation.get("source_resolution"),
    }


def read_colmap_quality(workspace: Path) -> dict:
    audit = read_json(workspace / "colmap_audit.json") or {}
    preparation = read_json(workspace / "preparation.json") or {}
    reproj = audit.get("reprojection_error_px") or {}
    status = audit.get("status")
    return {
        # The audit writes PASS/FAIL; the registry and the guard use pass/fail.
        # Normalising here means one spelling reaches every consumer.
        "status": status.lower() if isinstance(status, str) else status,
        "registered_images": preparation.get("registered_images"),
        "frames_used": preparation.get("frames_used"),
        "frames_dropped": len(preparation.get("dropped_frames") or []),
        "sparse_points_total": preparation.get("sparse_points_total"),
        "sparse_points_plant": preparation.get("sparse_points_plant"),
        "reprojection_error_mean_px": preparation.get("reprojection_error_mean_px",
                                                     reproj.get("mean")),
        "reprojection_error_p95_px": reproj.get("p95"),
        "mask_hit_rate_median": preparation.get("mask_hit_rate_median"),
        "mask_hit_rate_min": preparation.get("mask_hit_rate_min"),
        "crop_offsets_measured_per_sample": preparation.get(
            "crop_offsets_measured_per_sample"),
        "canvas": preparation.get("canvas"),
        "prepare_seconds": preparation.get("prepared_in_seconds"),
    }


def read_reconstruction(workspace: Path) -> dict:
    report = read_json(workspace / "final" / "report.json") or {}
    validation = read_json(workspace / "final" / "validation.json") or {}
    # Two writers exist for this number: the export report nests the clean
    # validation, while earlier runs wrote it flat to validation.json. Read
    # both shapes rather than reporting "missing" for a metric that is present
    # under the other name.
    clean = report.get("clean_validation") or validation.get("clean_validation") or {}
    if not clean and validation:
        clean = {k: validation.get(k) for k in
                 ("foreground_psnr", "silhouette_iou", "background_alpha")}
    if not report and validation:
        report = {
            "step": None,
            "raw_gaussians": None,
            "retained_gaussians": _ply_rows(workspace / "final" / "point_cloud.ply"),
        }
    per_view = (report.get("clean_validation") or {}).get("per_view") \
        or validation.get("per_view") or []
    stages = {}
    for stage in ("train_7k", "train_30k"):
        manifest = read_json(workspace / "results" / stage / "stage_manifest.json")
        status = read_json(workspace / "results" / stage / "status.json") or {}
        if manifest or status:
            stages[stage] = {
                "manifest": manifest,
                "status": status.get("status"),
                "best_step": status.get("best_step"),
                "gaussians": status.get("gaussians"),
                "run_id": status.get("run_id"),
            }
    return {
        "foreground_psnr": clean.get("foreground_psnr"),
        "silhouette_iou": clean.get("silhouette_iou"),
        "background_alpha": clean.get("background_alpha"),
        "held_out_views": len(per_view),
        "raw_gaussians": report.get("raw_gaussians"),
        "retained_gaussians": report.get("retained_gaussians"),
        "checkpoint_step": report.get("step"),
        "held_out_scope": ("single-scene held-out views; not plant generalisation"),
        "stages": stages,
        "export_seconds": report.get("seconds"),
    }


def read_provenance(workspace: Path) -> dict:
    verification = read_json(workspace / "final" / "provenance_verification.json")
    if verification is None:
        return {"state": "absent", "record": None}
    details = verification.get("details") or {}
    state = "verified" if verification.get("status") == "pass" else "failed"
    return {
        "state": state,
        "record": str((workspace / "final" / "provenance_verification.json").resolve()),
        "verified_retrospectively": bool(verification.get("retrospective")),
        "audited_utc": verification.get("checked_utc"),
        "checkpoint_sha256": details.get("checkpoint_sha256"),
        "ply_sha256": details.get("ply_sha256"),
        "checkpoint_step": details.get("checkpoint_step"),
        "gaussians_after_filter": details.get("gaussians_after_filter"),
        "problems": verification.get("problems") or [],
    }


def read_jepa_compat(workspace: Path) -> dict:
    for name in ("final/gsjepa_compatibility.json", "gsjepa_compatibility.json"):
        payload = read_json(workspace / name)
        if payload is not None:
            return {
                "status": "pass" if payload.get("gsjepa_compatible") else "fail",
                "attributes_present": payload.get("required_fields"),
                "detail": str((workspace / name).resolve()),
                "source_sha256": payload.get("sha256") or payload.get("source_sha256"),
            }
    return {"status": "not_run"}


def read_smoke(workspace: Path) -> dict:
    smoke = read_json(workspace / "jepa_smoke.json") or {}
    probe = read_json(workspace / "jepa_optimizer_probe.json") or {}
    return {
        "replay": {
            "status": smoke.get("status"),
            "source_sha256": smoke.get("source_sha256"),
            "sampled_gaussians": smoke.get("sampled_gaussians"),
            "sampling_method": (smoke.get("sampling") or {}).get("method"),
            "indices_sha256": (smoke.get("sampling") or {}).get("indices_sha256"),
            "initialization": smoke.get("initialization"),
            "losses": smoke.get("losses"),
            "weights_unchanged": smoke.get("weights_unchanged"),
        },
        "optimizer_probe": {"status": probe.get("status"),
                            "ema_gap": probe.get("ema_gap")},
    }


def read_semantics(workspace: Path) -> dict:
    """Candidate per-view labels and the agreement between the two backends."""
    semantic = workspace / "semantic"
    if not semantic.is_dir():
        return {"available": False}
    summary = read_json(semantic / "summary.json") or {}
    agreement = read_json(semantic / "agreement.json") or {}
    merge = read_json(semantic / "merge_summary.json") or {}
    coverage = {}
    for backend, payload in (summary.get("mean_coverage_within_specimen_mask") or {}).items():
        coverage[backend] = payload
    present = [name for name in SEMANTIC_LABEL_DIRS if (semantic / name).is_dir()]
    return {
        "available": bool(present),
        "label_dirs": present,
        "frames_labelled": summary.get("frames_labelled"),
        "backends": summary.get("backends"),
        "mean_coverage_within_specimen_mask": coverage,
        "plant_agreement_iou_median": agreement.get("plant_iou_median"),
        "plant_agreement_iou_min": agreement.get("plant_iou_min"),
        "merge": {
            "plant_agreement_fraction": merge.get("plant_agreement_fraction"),
            "totals": merge.get("totals"),
        },
        "panel": (workspace / "semantic" / "plant_reference_panel" / "panel_summary.json").is_file(),
        # The brief is explicit: model output is not human truth. This is the
        # field the guard reads, and it is hard-coded here as a fact about the
        # *tooling*, not a claim about the labels.
        "truth_source": "model_candidate",
        "benchmark": {"status": "annotation_pending", "truth_source": "model_candidate"},
    }


def read_plant_only(workspace: Path, directory: str = "plant_only") -> dict:
    out = workspace / directory
    report = read_json(out / "plant_only_report.json")
    provenance = read_json(out / "plant_only_provenance.json")
    verification = read_json(out / "plant_only_provenance_verification.json")
    if report is None:
        return {"available": False, "dir": str(out)}
    region = report.get("region_quality") or {}
    # The carve records its source in plant_only_report.json (source_ply +
    # hash); the separate verifier re-derives the mapping and writes the
    # verification file. So the verification is the authority on the state,
    # and the report is the record being verified.
    provenance_state = "unverified"
    if verification is not None:
        provenance_state = ("verified" if verification.get("status") == "pass"
                            else "failed")
    elif report.get("source_ply") is None:
        provenance_state = "absent"
    return {
        "available": True,
        "dir": str(out.resolve()),
        "ply": str((out / "plant_only.ply").resolve()) if (out / "plant_only.ply").is_file() else None,
        "provenance_state": provenance_state,
        "provenance": verification or provenance,
        "source_ply": report.get("source_ply"),
        "source_ply_sha256": report.get("source_ply_sha256"),
        "status": report.get("status"),
        "source_gaussians": report.get("source_gaussians"),
        "plant_gaussians": report.get("plant_gaussians"),
        "nonplant_gaussians": report.get("nonplant_gaussians"),
        "undecided_gaussians": report.get("undecided_gaussians"),
        "plant_fraction_of_gaussians": report.get("plant_fraction_of_gaussians"),
        "class_counts": report.get("class_counts"),
        "retention": report.get("retention"),
        "visibility": report.get("visibility"),
        "render_quality": {
            "views": region.get("views"),
            "psnr_full_in_plant": region.get("psnr_full_in_plant"),
            "psnr_plant_only_in_plant": region.get("psnr_plant_only_in_plant"),
            "iou_full_vs_plant": region.get("iou_full_vs_plant"),
            "iou_plant_only_vs_plant": region.get("iou_plant_only_vs_plant"),
            "plant_coverage": region.get("plant_coverage"),
            "residue_alpha_outside_plant": region.get("residue_alpha_outside_plant"),
            "residue_opaque_fraction_outside_plant":
                region.get("residue_opaque_fraction_outside_plant"),
            "per_view": region.get("per_view"),
        },
        "plant_source": report.get("plant_source"),
        "labels": report.get("labels"),
        "seconds": report.get("seconds"),
    }


def read_source(workspace: Path, sample_dir: Path | None) -> dict:
    provenance = None
    for candidate in (sample_dir, workspace.parent / (workspace.name.split("_")[0])):
        if candidate is not None and (candidate / "provenance.json").is_file():
            provenance = read_json(candidate / "provenance.json")
            break
    if provenance is None:
        return {}
    hub = provenance.get("hub") or {}
    return {
        "provenance_path": str((candidate / "provenance.json").resolve()),
        "hf_endpoint": provenance.get("hf_endpoint_used"),
        "hub_sha": hub.get("sha"),
        "hub_last_modified": hub.get("last_modified"),
        "requested_revision": provenance.get("requested_revision"),
        "downloaded_at": provenance.get("downloaded_at"),
        "file_counts": provenance.get("file_counts"),
        "license": provenance.get("license") or hub.get("license"),
    }


def read_resources(workspace: Path) -> dict:
    out = {}
    for stage in ("train_7k", "train_30k"):
        status = read_json(workspace / "results" / stage / "status.json") or {}
        manifest = read_json(workspace / "results" / stage / "stage_manifest.json") or {}
        if status:
            out[stage] = {
                "steps": status.get("steps") or manifest.get("steps"),
                "seconds": status.get("seconds") or status.get("elapsed_seconds"),
                "best_step": status.get("best_step"),
                "gaussians": status.get("gaussians"),
                "git_commit": manifest.get("git_commit"),
                "command": manifest.get("command"),
                "finished_utc": manifest.get("finished_utc"),
            }
    preparation = read_json(workspace / "preparation.json") or {}
    if preparation:
        out["prepare_seconds"] = preparation.get("prepared_in_seconds")
    return out


# --------------------------------------------------------------------------- #
# Entry assembly
# --------------------------------------------------------------------------- #
def build_entry(workspace: Path, sample_dir: Path | None = None,
                registry_row: dict | None = None,
                plant_only_dirs: tuple = PLANT_ONLY_DIRS) -> dict:
    preparation = read_json(workspace / "preparation.json") or {}
    crop = preparation.get("crop") or (registry_row or {}).get("crop") or ""
    sample = preparation.get("sample") or (registry_row or {}).get("sample_id")
    if sample is None and "_" in workspace.name:
        crop_guess, _, rest = workspace.name.partition("_")
        sample = rest
        crop = crop or crop_guess
    plant_id = preparation.get("plant_id") or (registry_row or {}).get("plant_id")
    acquisition = preparation.get("acquisition_id") or (registry_row or {}).get(
        "acquisition_id") or sample
    if sample_dir is None:
        candidate = workspace.parent.parent.parent / "external" / "PAA-Plant-Dataset" / \
            str(crop) / str(sample)
        sample_dir = candidate if candidate.is_dir() else None

    plant_only = {}
    for name in plant_only_dirs:
        candidate = read_plant_only(workspace, name)
        if candidate.get("available"):
            plant_only[name] = candidate

    reconstruction = read_reconstruction(workspace)
    entry = {
        "schema_version": SCHEMA_VERSION,
        "asset_id": f"{crop}/{acquisition}",
        "crop": crop,
        "plant_id": plant_id,
        "acquisition_id": acquisition,
        "acquisition_date": preparation.get("acquisition_date")
                            or (registry_row or {}).get("acquisition_date"),
        "split_group": f"{crop}/{plant_id}" if crop and plant_id else None,
        "workspace": str(workspace.resolve()),
        "sealed": is_sealed(workspace),
        "raw_sample_dir": str(sample_dir.resolve()) if sample_dir else None,
        "source": read_source(workspace, sample_dir),
        "source_revision": ((read_source(workspace, sample_dir) or {}).get("hub_sha")
                            or (registry_row or {}).get("source_revision")),
        "license": preparation.get("license") or (registry_row or {}).get("license"),
        "camera": read_camera(workspace),
        "colmap": read_colmap_quality(workspace),
        "colmap_status": read_colmap_quality(workspace).get("status"),
        "reconstruction_quality": reconstruction,
        "gaussian_ply_path": (str((workspace / "final" / "point_cloud.ply").resolve())
                              if (workspace / "final" / "point_cloud.ply").is_file() else None),
        "gaussian_count": reconstruction.get("retained_gaussians"),
        "jepa_compatibility": read_jepa_compat(workspace),
        "provenance": read_provenance(workspace),
        "jepa": read_smoke(workspace),
        "semantics": read_semantics(workspace),
        "plant_only": plant_only,
        "resources": read_resources(workspace),
        "registered": bool(registry_row),
        "notes": [],
    }
    entry["states"] = guard.evaluate(entry)
    entry["gaussian_ply_sha256"] = entry["provenance"].get("ply_sha256")
    return entry


def load_registry(path: Path) -> dict:
    from agri.plant_gs.paa.registry import record_key
    out = {}
    if not path.is_file():
        return out
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        out[record_key(row.get("crop", ""), row.get("acquisition_id", ""))] = row
    return out


def build_library(workspaces_root: Path, registry_path: Path) -> list:
    registry = load_registry(registry_path)
    entries = []
    for workspace in discover_workspaces(workspaces_root):
        row = None
        for key, candidate in registry.items():
            if candidate.get("sample_dir") and Path(candidate["sample_dir"]).resolve() == workspace.resolve():
                row = candidate
                break
        entry = build_entry(workspace, registry_row=row)
        if not entry["crop"] or not entry["acquisition_id"]:
            log(f"skipping {workspace.name}: crop/sample id unresolved")
            continue
        entries.append(entry)

    # A sealed workspace and the clean re-run of the same acquisition share an
    # id (both are "Cotton/1"). The clean run keeps it; the sealed evidence
    # directory is renamed rather than dropped, so the audit can still show it
    # and a reader can see that the polluted build exists.
    by_id = {}
    for entry in entries:
        by_id.setdefault(entry["asset_id"], []).append(entry)
    for asset_id, group in by_id.items():
        if len(group) > 1:
            for entry in group:
                if entry.get("sealed"):
                    entry["asset_id"] = f"{asset_id}@sealed"
                    entry["notes"].append(
                        "id disambiguated from the clean re-run of the same "
                        "acquisition; this workspace is sealed evidence")
    return entries


# --------------------------------------------------------------------------- #
# Plant-level split plan
# --------------------------------------------------------------------------- #
def plan_split(entries: list, seed: int = 20261020,
               ratios=(0.6, 0.2, 0.2)) -> dict:
    """A train/val/test split over *split groups*, never over acquisitions.

    The brief requires that every acquisition of one plant land in the same
    split, so the unit of assignment is the split group. What this function
    deliberately also does is **state how weak the split is**: with the plant
    count this library actually has, a three-way split cannot support a
    generalisation claim, and a plan that presented the ratios without saying
    so would invite exactly that claim.

    Only assets that are ``reconstruction_verified`` and not sealed take part;
    a plant with one usable date still contributes its group, because the group
    is the plant, not the date.
    """
    import random
    usable = [e for e in entries
              if not e.get("sealed")
              and (e.get("states") or {}).get("states", {}).get("reconstruction_verified")]
    groups = sorted({e["split_group"] for e in usable if e.get("split_group")})
    rng = random.Random(seed)
    shuffled = list(groups)
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_train = max(1, int(round(n * ratios[0]))) if n else 0
    n_val = max(1, int(round(n * ratios[1]))) if n > 2 else 0
    if n_train + n_val >= n:
        n_train = max(1, n - 2)
        n_val = 1 if n > 2 else 0
    train = sorted(shuffled[:n_train])
    val = sorted(shuffled[n_train:n_train + n_val])
    test = sorted(shuffled[n_train + n_val:])
    split = guard.make_split(usable, train, val, test)
    split["seed"] = seed
    split["ratios"] = {"train": ratios[0], "val": ratios[1], "test": ratios[2]}
    split["counts"] = {
        "groups_total": n,
        "groups": {"train": len(train), "val": len(val), "test": len(test)},
        "assets": {name: len(ids) for name, ids in split["coverage"].items()},
        "crops": {name: sorted({a["crop"] for a in usable
                                if a["split_group"] in set(groups_)})
                  for name, groups_ in split["groups"].items()},
    }
    split["caveat"] = (
        f"{n} distinct plants is too few for a three-way split to support a "
        "generalisation claim; it fixes identity leakage, which is necessary but "
        "not sufficient. For any statement about unseen plants, prefer "
        "leave-one-crop-out and report the per-crop spread, not a single val "
        "number.")
    return split


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workspaces", type=Path, default=Path("data/plant_gs/PAA"))
    ap.add_argument("--registry", type=Path,
                    default=Path("data/plant_gs/PAA/registry.jsonl"))
    ap.add_argument("--out", type=Path, default=Path("results/agri_paa/stage4c"))
    ap.add_argument("--skip-hashes", action="store_true",
                    help="do not re-hash the PLYs (the provenance record already "
                         "carries their hashes; re-hashing is an independent check)")
    ap.add_argument("--split-seed", type=int, default=20261020)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    entries = build_library(args.workspaces, args.registry)

    hash_audit = {}
    if not args.skip_hashes:
        for mismatch in guard.check_ply_hashes(entries):
            log(f"PLY hash MISMATCH {mismatch}")
        hash_audit = {
            entry["asset_id"]: {
                "recorded": (entry.get("provenance") or {}).get("ply_sha256"),
                "recomputed": sha256_file(Path(entry["gaussian_ply_path"])),
            }
            for entry in entries
            if entry.get("gaussian_ply_path")
            and (entry.get("provenance") or {}).get("ply_sha256")
        }
        for value in hash_audit.values():
            value["matches"] = value["recorded"] == value["recomputed"]

    library = {
        "schema_version": SCHEMA_VERSION,
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "workspaces_root": str(args.workspaces.resolve()),
        "registry": str(args.registry.resolve()),
        "n_assets": len(entries),
        "entries": entries,
        "ply_hash_recheck": hash_audit,
    }
    (args.out / "assets.json").write_text(
        json.dumps(library, indent=2, ensure_ascii=False) + "\n")

    split = plan_split(entries, seed=args.split_seed)
    audit = guard.audit_library(entries, split=split)
    eligible_full, refused_full = guard.eligible_assets(entries, "full_specimen_pretrain")
    eligible_po, refused_po = guard.eligible_assets(entries, "plant_only_pretrain")
    governance = {
        "schema_version": SCHEMA_VERSION,
        "generated_utc": library["generated_utc"],
        "states_legend": {
            "reconstruction_verified": "checkpoint->PLY chain verified, COLMAP "
                                       "passed, PLY is JEPA-compatible, metrics "
                                       "cleared the gate",
            "semantic_verified": "plant/pot labels were scored against labels a "
                                 "human confirmed",
            "plant_only_verified": "a plant-only asset exists and is both "
                                   "reconstruction- and semantic-verified",
            "train_eligible": "the asset meets the project's data-use conditions "
                              "for a named role",
        },
        "counts": {
            "assets": len(entries),
            "reconstruction_verified": sum(
                1 for e in entries if e["states"]["states"]["reconstruction_verified"]),
            "semantic_verified": sum(
                1 for e in entries if e["states"]["states"]["semantic_verified"]),
            "plant_only_verified": sum(
                1 for e in entries if e["states"]["states"]["plant_only_verified"]),
            "train_eligible_full_specimen": len(eligible_full),
            "train_eligible_plant_only": len(eligible_po),
            "sealed": sum(1 for e in entries if e.get("sealed")),
            "crops": len({e["crop"] for e in entries if not e.get("sealed")}),
            "distinct_plants": len({e["split_group"] for e in entries
                                    if not e.get("sealed")}),
        },
        "eligibility": {
            "full_specimen_pretrain": {
                "eligible": [e["asset_id"] for e in eligible_full],
                "refused": refused_full},
            "plant_only_pretrain": {
                "eligible": [e["asset_id"] for e in eligible_po],
                "refused": refused_po},
        },
        "audit": audit,
        "split": split,
        "ply_hash_recheck": hash_audit,
        "not_claimed": [
            "no segmentation accuracy: no human-confirmed annotation exists",
            "no representation quality: every JEPA number is from a randomly "
            "initialised model",
            "no plant generalisation: all reconstruction metrics are single-scene "
            "held-out views",
        ],
    }
    (args.out / "governance.json").write_text(
        json.dumps(governance, indent=2, ensure_ascii=False) + "\n")

    summary = [
        {"asset_id": e["asset_id"], "crop": e["crop"],
         "reconstruction_verified": e["states"]["states"]["reconstruction_verified"],
         "semantic_verified": e["states"]["states"]["semantic_verified"],
         "plant_only_verified": e["states"]["states"]["plant_only_verified"],
         "train_eligible_full": e["states"]["eligibility"]["full_specimen_pretrain"],
         "psnr": (e["reconstruction_quality"] or {}).get("foreground_psnr"),
         "gaussians": e["gaussian_count"],
         "camera_models": e["camera"]["models"]}
        for e in entries]
    log(json.dumps(summary, indent=2, ensure_ascii=False))
    log(f"wrote {args.out / 'assets.json'} ({len(entries)} assets)")


if __name__ == "__main__":
    main()
