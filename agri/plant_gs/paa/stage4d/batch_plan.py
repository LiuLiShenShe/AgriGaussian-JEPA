#!/usr/bin/env python
"""Stage 4D: what a PAA batch expansion would actually cost, and whether to run it.

The decision this module supports is *not* "can the pipeline run again" -- it
has run nine times. It is "has anything changed since Stage 4C said stop", and
the answer is recorded as data: the Rice failure is not explained, the human
semantic check has not happened, and one crop's white tissue is still labelled
support by the candidate models.

So the module emits a **conditional** plan: an exact list of the remaining
acquisitions, ordered so that independent plants are added before second dates,
with measured storage and time per unit, and with the preconditions written
next to the batch they gate. A reader can execute it the day the conditions are
met, and cannot mistake it for authorisation today.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

SCHEMA_VERSION = "paa-stage4d-batch-plan-1"


def log(message: str) -> None:
    print(message, flush=True)


def _utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def directory_bytes(path: Path) -> int:
    total = 0
    for item in Path(path).rglob("*"):
        try:
            if item.is_file():
                total += item.stat().st_size
        except OSError:
            continue
    return total


def measured_unit_cost(workspaces: list) -> dict:
    """Storage and wall-clock of the units that already exist, measured.

    Training time is summed **per workspace** (7k + 30k), then averaged across
    workspaces. Averaging the stage values instead would halve the number by
    mixing a 7k run with a 30k run.
    """
    sizes, per_unit_seconds, prepare = [], [], []
    for workspace in workspaces:
        if workspace.is_dir():
            sizes.append(directory_bytes(workspace))
        total = 0.0
        for stage in ("train_7k", "train_30k"):
            status = workspace / "results" / stage / "status.json"
            if status.is_file():
                payload = json.loads(status.read_text())
                value = payload.get("seconds") or payload.get("elapsed_seconds")
                if value:
                    total += float(value)
        if total:
            per_unit_seconds.append(total)
        prep = workspace / "preparation.json"
        if prep.is_file():
            value = json.loads(prep.read_text()).get("prepared_in_seconds")
            if value:
                prepare.append(float(value))
    return {
        "workspaces_measured": sum(1 for w in workspaces if w.is_dir()),
        "disk_bytes_per_unit_mean": int(sum(sizes) / len(sizes)) if sizes else None,
        "disk_bytes_per_unit_max": max(sizes) if sizes else None,
        "train_7k_plus_30k_seconds_mean": (sum(per_unit_seconds)
                                           / len(per_unit_seconds))
        if per_unit_seconds else None,
        "prepare_seconds_mean": (sum(prepare) / len(prepare)) if prepare else None,
        "not_measured_here": {
            "download_seconds": 180,
            "semantic_label_seconds": 1110,
            "export_verify_seconds": 180,
            "carve_eval_seconds": 120,
            "source": ("Stage 4C report section 2.5 measured on the same host; "
                       "kept separate from the per-stage numbers above because "
                       "they were timed as a block rather than by stage"),
        },
        "note": ("wall-clock is from a shared host (load 30-140); it is a plan "
                 "input, not a benchmark"),
    }


def build_plan(identity: dict, entries: list, workspaces_root: Path,
               gate_report: dict | None) -> dict:
    acquisitions = identity.get("acquisitions") or []
    trusted = [e for e in entries
               if not e.get("sealed")
               and (e.get("states") or {}).get("states", {}).get(
                   "reconstruction_verified")]
    trusted_keys = {(e["crop"], e["acquisition_id"]) for e in trusted}
    trusted_plants = {(e["crop"], e.get("plant_id")) for e in trusted}

    remaining = [a for a in acquisitions
                 if (a["crop"], a["acquisition_id"]) not in trusted_keys]
    new_plants = [a for a in remaining
                  if (a["crop"], a.get("plant_id")) not in trusted_plants]
    extra_dates = [a for a in remaining
                   if (a["crop"], a.get("plant_id")) in trusted_plants]

    # New plants first: they buy a new split group. A second date of an
    # existing plant buys a within-plant comparison but no new group, and it
    # must stay in that plant's split.
    ordered = ([{**a, "batch_reason": "new independent plant"}
                for a in sorted(new_plants, key=lambda a: (a["crop"], a["acquisition_id"]))]
               + [{**a, "batch_reason": "additional date of a known plant"}
                  for a in sorted(extra_dates, key=lambda a: (a["crop"], a["acquisition_id"]))])

    workspaces = [workspaces_root / f"{a['crop']}_{a['acquisition_id']}"
                  for a in acquisitions]
    cost = measured_unit_cost(workspaces)
    n = len(ordered)
    storage = (cost["disk_bytes_per_unit_mean"] or 0) * n
    other = sum((cost.get("not_measured_here") or {}).get(k, 0) for k in
                ("download_seconds", "semantic_label_seconds",
                 "export_verify_seconds", "carve_eval_seconds"))
    seconds = (((cost["train_7k_plus_30k_seconds_mean"] or 0)
                + (cost["prepare_seconds_mean"] or 0) + other) * n) if n else 0
    single_gpu_hours = seconds / 3600 if seconds else None

    conditions = {
        "rice_root_cause": {
            "met": False,
            "why": ("Rice/1 carved to 0 Gaussians because both backends miss "
                    "the thin stalks; with 4 more Rice acquisitions in the "
                    "remaining set, running them first repeats the failure four "
                    "times"),
            "how_to_satisfy": ("a candidate 2D labeller that keeps rice stalks "
                               "stable across the orbit, checked on the Rice "
                               "priority views"),
        },
        "human_semantic_check": {
            "met": bool((gate_report or {}).get("assets_semantic_verified")),
            "why": ("no human-confirmed label exists, so plant-only assets "
                    "remain unverified and the white-tissue failure is not "
                    "independently confirmed"),
            "how_to_satisfy": ("the 12 priority tasks confirmed, then the "
                               "semantic gate re-run"),
        },
        "cabbage_consistency": {
            "met": False,
            "why": ("Cabbage's A/B plant IoU is 0.38 and the diagnostic shows "
                    "B labels 65% of white specimen pixels support; expanding "
                    "Cabbage first would multiply a known failure"),
            "how_to_satisfy": "confirm the Cabbage priority views, then re-score",
        },
    }
    blocked = [name for name, value in conditions.items() if not value["met"]]

    return {
        "schema_version": SCHEMA_VERSION,
        "generated_utc": _utc(),
        "library": {
            "acquisitions_total": len(acquisitions),
            "plants_total": identity.get("n_distinct_plants"),
            "crops_total": identity.get("n_crops"),
            "trusted_assets": len(trusted),
            "trusted_plants": len(trusted_plants),
            "remaining_acquisitions": n,
            "remaining_new_plants": len(new_plants),
            "remaining_extra_dates": len(extra_dates),
        },
        "measured_unit_cost": cost,
        "estimate": {
            "remaining_units": n,
            "storage_bytes": int(storage),
            "storage_gib": round(storage / (1024 ** 3), 1),
            "single_gpu_hours": round(seconds / 3600, 1) if seconds else None,
            "two_gpu_hours": round(seconds / 3600 / 2, 1) if seconds else None,
            "basis": ("per-unit training (7k+30k) and preparation measured from "
                      "the eight existing workspaces, plus the download/"
                      "semantic/export timings measured in Stage 4C section 2.5"),
            "assumption": ("two independent workers; wall-clock halves only if "
                           "both GPUs are actually free"),
        },
        "preconditions": conditions,
        "blocked_by": blocked,
        "decision": ("conditional: do not start a bulk expansion until every "
                     "precondition is met"
                     if blocked else
                     "go: preconditions met; start with the new-plant batches only"),
        "recommended_first_batch": [a for a in ordered
                                    if a["batch_reason"] == "new independent plant"][:6]
        if blocked else ordered,
        "batch_order": ordered,
        "rules": [
            "only reconstruction_verified, source-complete full-specimen assets "
            "enter the candidate pool",
            "plant-only export must pass its own semantic gate; a loadable PLY "
            "is not evidence",
            "new independent plants before additional dates",
            "every asset keeps a checkpoint->PLY provenance record and a "
            "re-hashed PLY",
        ],
        "not_authorised": ("this file is a plan, not a permission. The decision "
                           "field is conditional and names what is missing."),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--identity", type=Path,
                    default=Path("results/agri_paa/stage4c/identity_manifest.json"))
    ap.add_argument("--library", type=Path,
                    default=Path("results/agri_paa/stage4d/assets.json"))
    ap.add_argument("--workspaces", type=Path, default=Path("data/plant_gs/PAA"))
    ap.add_argument("--gate", type=Path,
                    default=Path("results/agri_paa/stage4d/semantic_gate.json"))
    ap.add_argument("--out", type=Path,
                    default=Path("results/agri_paa/stage4d/paa_batch_plan.json"))
    args = ap.parse_args()

    identity = json.loads(args.identity.read_text())
    library = json.loads(args.library.read_text()) if args.library.is_file() else {"entries": []}
    gate = json.loads(args.gate.read_text()) if args.gate.is_file() else None
    plan = build_plan(identity, library.get("entries", []), args.workspaces, gate)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(plan, indent=2, ensure_ascii=False) + "\n")
    log(json.dumps({"library": plan["library"], "estimate": plan["estimate"],
                    "decision": plan["decision"],
                    "blocked_by": plan["blocked_by"]}, indent=2))
    log(f"wrote {args.out}")


if __name__ == "__main__":
    main()
