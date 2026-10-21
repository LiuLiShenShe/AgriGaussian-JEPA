#!/usr/bin/env python
"""Eligibility decisions for the Stage 4C asset library.

The one rule this module exists to enforce
------------------------------------------
**Loading successfully is not evidence.** Gaussian-JEPA will happily read any
well-formed 3DGS PLY. A file that loads, renders, and produces finite losses
has demonstrated that the *reader* works -- nothing about whether the asset
should be in a pretraining set. Stage 4B closed with exactly that ambiguity:
`jepa_compatibility: pass` was recorded next to assets whose semantic
separation had never been checked.

So eligibility is computed here from named evidence, and every evidence item
must be *present and passing on its own terms*. Missing evidence is never
treated as neutral; it fails the requirement that asked for it.

The four states
---------------
``reconstruction_verified``
    The checkpoint -> PLY chain verified, COLMAP passed, the PLY is
    JEPA-compatible, and the held-out metrics cleared the project gate.
``semantic_verified``
    Plant/pot labels were scored against labels a human confirmed. Candidate
    labels produced by a model never set this state.
``plant_only_verified``
    A plant-only asset exists *and* it is both reconstruction-verified and
    semantic-verified. A carved asset whose labels were never checked is
    ``plant_only_candidate``, not verified -- the carve depends on those labels.
``train_eligible``
    The asset meets the project's data-use conditions for the *role* it is
    being considered for. The role matters: a full-specimen asset needs
    reconstruction evidence, a plant-only asset additionally needs verified
    semantics, and `annotation_benchmark` images need a human label.

Two assets can therefore both be ``reconstruction_verified`` and only one be
``train_eligible``; that difference is the whole point of the distinction.
"""

from __future__ import annotations

from dataclasses import dataclass, field

#: Project thresholds, the same ones the 7k quality gate applies. Duplicated as
#: data rather than imported from the shell helper because this module is used
#: by tests and by the audit, neither of which can source a bash file.
DEFAULT_GATE = {
    "min_foreground_psnr": 18.0,
    "min_silhouette_iou": 0.85,
    "max_background_alpha": 0.05,
}

#: The roles an asset can be used for in a later experiment.
ROLES = ("full_specimen_pretrain", "plant_only_pretrain", "annotation_benchmark",
         "semantic_review_only")

#: Why a requirement failed, in words a reader can act on.
REASONS = {
    "no_ply": "no Gaussian PLY recorded",
    "no_provenance": "no checkpoint->PLY provenance record",
    "provenance_failed": "checkpoint->PLY provenance did not verify",
    "provenance_absent": "no checkpoint->PLY provenance record (missing or unreadable)",
    "colmap_not_pass": "COLMAP audit did not pass",
    "jepa_not_pass": "PLY is not Gaussian-JEPA compatible",
    "quality_unknown": "held-out reconstruction metrics are missing",
    "quality_below_gate": "held-out reconstruction metrics are below the gate",
    "semantics_unverified": "semantic separation has no human-confirmed evidence",
    "plant_only_missing": "no plant-only asset",
    "plant_only_failed_empty":
        "the plant-only carve failed or retained no Gaussians",
    "plant_only_source_unverified":
        "a plant-only asset exists but its carve provenance did not verify",
    "plant_only_not_reconstruction_verified":
        "plant-only asset's own reconstruction/provenance did not verify",
    "annotation_not_human": "labels are not human-confirmed",
    "annotation_pending": "no human annotation has been imported",
    "asset_sealed": "workspace is sealed as incident evidence and is excluded by policy",
}


def reconstruction_checks(entry: dict, gate: dict | None = None) -> list:
    """Every reason the entry has not earned ``reconstruction_verified``."""
    gate = dict(DEFAULT_GATE, **(gate or {}))
    problems = []
    if entry.get("sealed"):
        # A sealed workspace is the evidence for a past incident. Whatever it
        # contains, it is not an asset, and no amount of passing checks changes
        # that -- so the refusal comes first and is not overridable by evidence.
        problems.append("asset_sealed")
    if not entry.get("gaussian_ply_path"):
        problems.append("no_ply")
    provenance = entry.get("provenance") or {}
    state = provenance.get("state")
    if state == "verified":
        pass
    elif state == "failed":
        problems.append("provenance_failed")
    elif state is None:
        problems.append("no_provenance")
    else:
        problems.append("provenance_absent")
    if entry.get("colmap_status") != "pass":
        problems.append("colmap_not_pass")
    if (entry.get("jepa_compatibility") or {}).get("status") != "pass":
        problems.append("jepa_not_pass")
    quality = entry.get("reconstruction_quality") or {}
    values = [quality.get("foreground_psnr"), quality.get("silhouette_iou"),
              quality.get("background_alpha")]
    if any(v is None for v in values):
        problems.append("quality_unknown")
    else:
        psnr, iou, bg = (float(v) for v in values)
        if (psnr < gate["min_foreground_psnr"]
                or iou < gate["min_silhouette_iou"]
                or bg > gate["max_background_alpha"]):
            problems.append("quality_below_gate")
    return problems


def semantic_checks(entry: dict) -> list:
    """Every reason the entry has not earned ``semantic_verified``.

    The only accepted source of truth is a benchmark evaluation whose labels
    were imported from a human confirmation. ``source="model_pseudo_label"``
    or ``"candidate"`` counts as *evidence of work done*, never as verified.

    A benchmark that was evaluated but whose per-asset gate failed is also not
    verified: ``status`` is ``"failed"`` there, and ``verified`` is ``False``.
    Reading both fields means a report written by an older tool (which had no
    ``verified`` key) still decides on ``status`` alone.
    """
    semantics = entry.get("semantics") or {}
    benchmark = semantics.get("benchmark") or {}
    if benchmark.get("truth_source") != "human_confirmed":
        return ["semantics_unverified"]
    if benchmark.get("status") != "evaluated":
        return ["semantics_unverified"]
    if benchmark.get("verified") is False:
        return ["semantics_unverified"]
    return []


def plant_only_checks(entry: dict, gate: dict | None = None) -> list:
    """Every reason the entry has not earned ``plant_only_verified``.

    ``assets.py`` collapses the per-version plant-only candidates into a single
    ``state`` (see ``assets.summarise_plant_only``). The five states are
    distinct on purpose, because "there is no carve", "the carve exists but its
    source is unverified", "the carve is sound but its labels never met a
    human" and "the carve is empty" call for different next actions. Collapsing
    them into one ``plant_only_missing`` -- the Stage 4C bug -- told a reader
    that eight existing assets had no carve at all.
    """
    plant = entry.get("plant_only") or {}
    state = plant.get("state")

    # Resolve the candidate the verdict is about. Three shapes exist:
    #   stage4d: {"state": ..., "selected": "plant_only", "candidates": {...}}
    #   stage4c: {"plant_only": {...}, "plant_only_segformer": {...}}
    #   older:   {"available": True, "provenance_state": "verified"}
    nested = plant.get("candidates") if isinstance(plant.get("candidates"), dict) else {}
    if nested:
        candidates = {k: v for k, v in nested.items()
                      if isinstance(v, dict) and v.get("available")}
        selected_name = plant.get("selected")
    elif plant.get("available") is True:
        candidates = {"_flat": plant}
        selected_name = "_flat"
    else:
        candidates = {k: v for k, v in plant.items() if isinstance(v, dict)
                      and v.get("available")}
        selected_name = sorted(candidates)[0] if candidates else None
    if state is None:
        state = "absent" if not candidates else "_legacy"

    if state == "absent":
        return ["plant_only_missing"]

    problems = []
    if state == "failed_empty":
        return ["plant_only_failed_empty"]
    if state == "candidate_source_unverified":
        return ["plant_only_source_unverified"]

    # The carved asset's own provenance: it was rendered from the same
    # checkpoint, but it is a *different* PLY and its rows must be traceable.
    # Only the *selected* candidate decides the verdict; an unused variant with
    # an unverified carve is a note for a reader, not a blocker.
    selected = candidates.get(selected_name) if selected_name else None
    if selected is None and len(candidates) == 1:
        selected = next(iter(candidates.values()))
    if selected is not None and selected.get("provenance_state") not in (
            None, "verified"):
        problems.append("plant_only_not_reconstruction_verified")

    problems.extend(reconstruction_checks(entry, gate))
    problems.extend(semantic_checks(entry))
    return problems




def evaluate(entry: dict, gate: dict | None = None) -> dict:
    """Compute the four states plus, for each failing requirement, the reason."""
    recon = reconstruction_checks(entry, gate)
    semantic = semantic_checks(entry)
    plant = plant_only_checks(entry, gate)

    states = {
        "reconstruction_verified": not recon,
        "semantic_verified": not semantic,
        "plant_only_verified": not plant,
    }

    eligibility = {}
    reasons = {}
    for role in ROLES:
        if role == "full_specimen_pretrain":
            failure = recon
        elif role == "plant_only_pretrain":
            failure = plant
        elif role == "semantic_review_only":
            # A human may look at an asset for review even if nothing is
            # verified; what it must NOT do is count as evidence.
            failure = recon
        else:  # annotation_benchmark
            failure = recon
            benchmark = (entry.get("semantics") or {}).get("benchmark") or {}
            if benchmark.get("truth_source") == "human_confirmed":
                pass
            elif benchmark.get("status") == "annotation_pending":
                failure = failure + ["annotation_pending"]
            else:
                failure = failure + ["annotation_not_human"]
        eligibility[role] = not failure
        reasons[role] = sorted(set(failure))

    return {
        "asset_id": entry.get("asset_id"),
        "states": states,
        "eligibility": eligibility,
        "reasons": {role: [REASONS.get(code, code) for code in codes]
                    for role, codes in reasons.items()},
        "reason_codes": reasons,
    }


# --------------------------------------------------------------------------- #
# The gate a loader must go through
# --------------------------------------------------------------------------- #
class NotTrainEligible(RuntimeError):
    """Raised when an asset is asked to be used for a role it has not earned."""


def eligible_assets(entries: list, role: str = "full_specimen_pretrain") -> tuple:
    """Split the library into (eligible, refused-with-reasons) for one role."""
    if role not in ROLES:
        raise ValueError(f"unknown role {role!r}; expected one of {ROLES}")
    eligible, refused = [], []
    for entry in entries:
        report = evaluate(entry)
        if report["eligibility"][role]:
            eligible.append(entry)
        else:
            refused.append({"asset_id": entry.get("asset_id"),
                            "blocking": report["reason_codes"][role],
                            "reasons": report["reasons"][role]})
    return eligible, refused


def check_ply_hashes(entries: list) -> list:
    """Re-hash each asset's PLY and compare with the recorded provenance hash.

    The provenance record already carries the hash. Recomputing it here answers
    a different question: "is the file that exists *now* the file whose hash is
    on the record?" -- which is the check that catches a workspace edited after
    its audit. Returns the mismatches, so an empty list means every file still
    matches.
    """
    import hashlib
    mismatches = []
    for entry in entries:
        path = entry.get("gaussian_ply_path")
        recorded = (entry.get("provenance") or {}).get("ply_sha256")
        if not path or not recorded:
            continue
        try:
            digest = hashlib.sha256()
            with open(path, "rb") as handle:
                for block in iter(lambda: handle.read(1 << 20), b""):
                    digest.update(block)
            actual = digest.hexdigest()
        except OSError as exc:
            mismatches.append({"asset_id": entry.get("asset_id"), "error": str(exc)})
            continue
        if actual != recorded:
            mismatches.append({"asset_id": entry.get("asset_id"),
                               "recorded": recorded, "actual": actual})
    return mismatches


def assert_train_eligible(entry: dict, role: str = "full_specimen_pretrain") -> None:
    """Raise unless the asset has earned ``role``. The only way in."""
    report = evaluate(entry)
    if not report["eligibility"][role]:
        raise NotTrainEligible(
            f"{entry.get('asset_id')} is not eligible for {role}: "
            + "; ".join(report["reasons"][role]))


# --------------------------------------------------------------------------- #
# Identity audit
# --------------------------------------------------------------------------- #
@dataclass
class AuditRow:
    """One problem found in the asset library."""

    severity: str          # "error" | "warning" | "info"
    code: str
    asset_id: str
    detail: str = ""


def audit_library(entries: list, split: dict | None = None) -> dict:
    """Cross-check plant identity, hashes, provenance and split leakage.

    The checks are deliberately about *internal consistency* of the library.
    A library can be internally consistent and still be wrong about a plant,
    so what this proves is stated narrowly: these identities do not collide,
    these hashes do not repeat, and no plant crosses a split boundary.
    """
    rows: list = []
    by_id = {}
    by_ply_hash = {}
    by_ckpt_hash = {}
    for entry in entries:
        asset_id = entry.get("asset_id")
        if not asset_id:
            rows.append(AuditRow("error", "missing_asset_id", str(entry)[:60]))
            continue
        if asset_id in by_id:
            rows.append(AuditRow("error", "duplicate_asset_id", asset_id))
        by_id[asset_id] = entry

        provenance = entry.get("provenance") or {}
        ply_sha = provenance.get("ply_sha256")
        ckpt_sha = provenance.get("checkpoint_sha256")
        if ply_sha:
            by_ply_hash.setdefault(ply_sha, []).append(asset_id)
        if entry.get("gaussian_ply_path") and not provenance.get("state"):
            rows.append(AuditRow("error", "missing_provenance", asset_id,
                                 "a PLY exists with no provenance block"))
        if provenance.get("state") == "verified":
            if not ply_sha or not ckpt_sha:
                rows.append(AuditRow("error", "provenance_without_hashes", asset_id))
            if provenance.get("checkpoint_step") is None:
                rows.append(AuditRow("error", "checkpoint_step_unrecorded", asset_id))
        elif provenance.get("state") == "failed":
            rows.append(AuditRow("error", "provenance_failed", asset_id,
                                 "; ".join(provenance.get("problems") or [])[:200]))

        source = entry.get("source") or {}
        if not (source.get("hub_sha") or entry.get("source_revision")):
            rows.append(AuditRow("warning", "missing_source_revision", asset_id,
                                 "Hub revision was not recorded for this build"))
        if entry.get("plant_id") in (None, ""):
            rows.append(AuditRow("error", "missing_plant_id", asset_id))
        if entry.get("split_group") != f"{entry.get('crop')}/{entry.get('plant_id')}":
            rows.append(AuditRow("error", "split_group_mismatch", asset_id,
                                 f"{entry.get('split_group')!r}"))

    for sha, owners in by_ply_hash.items():
        if len(owners) > 1:
            rows.append(AuditRow("error", "ply_hash_collision", ",".join(owners),
                                 f"identical PLY bytes under {len(owners)} assets"))
    for sha, owners in _checkpoint_collisions(entries):
        rows.append(AuditRow("warning", "checkpoint_hash_reused", ",".join(owners),
                             f"same checkpoint bytes exported to {len(owners)} assets"))

    leakage = []
    if split:
        groups = split.get("groups") or {}
        placed_in = {}
        for name, members in groups.items():
            for member in members:
                placed_in.setdefault(member, []).append(name)

        # (a) one group in two splits is direct leakage.
        for group, where in placed_in.items():
            if len(set(where)) > 1:
                leakage.append({"split_group": group, "placements": sorted(set(where))})

        # (b) every asset's own split_group must match the split its group was
        # put in. Keying by ``split_group`` makes (a) imply (b), but this runs
        # anyway because it is the check a reader can act on: it names the
        # asset, not the group.
        for entry in entries:
            asset_id = entry.get("asset_id")
            group_of_asset = entry.get("split_group")
            if not group_of_asset:
                leakage.append({"asset_id": asset_id,
                                "problem": "asset has no split_group"})
                continue
            where = placed_in.get(group_of_asset)
            if not where:
                continue
            if len(set(where)) > 1:
                leakage.append({"asset_id": asset_id, "split_group": group_of_asset,
                                "placements": sorted(set(where))})
        # (c) every acquisition of one plant must be on one side, which
        # split_group membership already guarantees; what can still go wrong is
        # one group naming two plants, so that is the check.
        plant_to_groups = {}
        for entry in entries:
            if entry.get("split_group"):
                plant_to_groups.setdefault(entry["split_group"], set()).add(
                    entry.get("plant_id"))
        for group, plants in plant_to_groups.items():
            if len(plants) > 1:
                leakage.append({"split_group": group, "plants": sorted(map(str, plants))})

    errors = [r for r in rows if r.severity == "error"]
    warnings = [r for r in rows if r.severity == "warning"]
    return {
        "assets_checked": len(entries),
        "problems": [r.__dict__ for r in rows],
        "error_count": len(errors),
        "warning_count": len(warnings),
        "split_leakage": leakage,
        "split_clean": not leakage,
        "clean": not errors and not leakage,
        "note": ("this audit proves internal consistency of the library "
                 "(identities, hashes, provenance, split grouping); it does not "
                 "prove that a plant id names the physical plant the data came from"),
    }


def _checkpoint_collisions(entries: list):
    """Group assets that share one checkpoint SHA256."""
    buckets = {}
    for entry in entries:
        sha = (entry.get("provenance") or {}).get("checkpoint_sha256")
        if sha:
            buckets.setdefault(sha, []).append(entry.get("asset_id"))
    return [(sha, owners) for sha, owners in buckets.items() if len(owners) > 1]


# --------------------------------------------------------------------------- #
# Split construction
# --------------------------------------------------------------------------- #
def make_split(entries: list, train: list, val: list, test: list) -> dict:
    """Build a split from *split groups* and verify nothing crossed over.

    Callers name groups, never acquisitions: grouping by acquisition is the
    mistake this whole identity layer exists to prevent, so the API does not
    offer it.
    """
    groups = {"train": list(train), "val": list(val), "test": list(test)}
    placed = {}
    for name, members in groups.items():
        for member in members:
            placed.setdefault(member, []).append(name)
    duplicate_placement = {k: v for k, v in placed.items() if len(v) > 1}
    known = {entry["split_group"] for entry in entries}
    unknown = sorted(set(placed) - known)
    coverage = {}
    for name, members in groups.items():
        coverage[name] = [e["asset_id"] for e in entries
                          if e.get("split_group") in set(members)]
    covered = {asset for ids in coverage.values() for asset in ids}
    return {
        "groups": groups,
        "coverage": coverage,
        "unassigned_assets": sorted({e["asset_id"] for e in entries} - covered),
        "unknown_groups": unknown,
        "duplicate_placement": duplicate_placement,
        "leakage": bool(duplicate_placement) or bool(unknown),
        "rule": "a split assigns whole split groups (<crop>/<plant_id>); "
                "an acquisition is never assigned on its own",
    }