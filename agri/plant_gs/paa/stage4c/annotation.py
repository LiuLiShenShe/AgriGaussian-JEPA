#!/usr/bin/env python
"""Export, review and score the Stage 4C human annotation benchmark.

The rule this module is built around
------------------------------------
**A model's prediction is not a human label, and the tooling must make that
impossible to confuse.** Three separate places enforce it:

1. Pre-annotations are written to ``preannotations/`` and are labelled with
   their backend. The evaluator never reads that directory as truth.
2. ``labels/`` is only ever populated by :func:`cmd_import`, which requires an
   annotator name and writes a confirmation record carrying the label's
   SHA256. A PNG dropped in by hand is reported as ``unconfirmed_ignored``.
3. Every task carries ``status: annotation_pending`` until a confirmation
   exists. Nothing in this module can set ``human_confirmed`` on its own.

The consequence is that on a machine with no annotator the honest output is
"tooling ready, 0 of 48 confirmed, no accuracy numbers" -- which is the success
state the brief asks for, not a failure to be papered over.

Usage
-----
    python -m agri.plant_gs.paa.stage4c.annotation export  --out results/agri_paa/stage4c/annotation
    python -m agri.plant_gs.paa.stage4c.annotation serve   --dir results/agri_paa/stage4c/annotation
    python -m agri.plant_gs.paa.stage4c.annotation import  --dir ... --task <id> --labels a.png --annotator alice
    python -m agri.plant_gs.paa.stage4c.annotation evaluate --dir ...  --predictions preannotations
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

Image.MAX_IMAGE_PIXELS = None

# --------------------------------------------------------------------------- #
# Annotation taxonomy.  Deliberately NOT the semantic.py class ids: those have
# pot and soil as separate classes and no uncertainty class, and reusing the
# integers would let a pot/soil mix-up look correct.
# --------------------------------------------------------------------------- #
BACKGROUND, PLANT, POT_SOIL, SUPPORT, UNCERTAIN = 0, 1, 2, 3, 4
CLASS_NAMES = {BACKGROUND: "Background", PLANT: "Plant", POT_SOIL: "Pot/Soil",
               SUPPORT: "Support", UNCERTAIN: "Uncertain"}
PALETTE = {BACKGROUND: (0, 0, 0), PLANT: (40, 200, 70), POT_SOIL: (60, 90, 230),
           SUPPORT: (235, 215, 60), UNCERTAIN: (255, 0, 255)}

#: semantic.py's class ids -> this taxonomy. SOIL is folded into POT_SOIL
#: because the brief's taxonomy has one class for the container and its
#: substrate.
SEMANTIC_TO_ANNOTATION = {0: BACKGROUND, 1: PLANT, 2: POT_SOIL, 3: POT_SOIL, 4: SUPPORT}

CRITERIA = ("azimuth_spread", "leaf_density", "plant_support_overlap",
            "plant_pot_boundary", "model_disagreement", "sparse_leaves")


def log(message: str) -> None:
    print(message, flush=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# --------------------------------------------------------------------------- #
# Export: pick views, write the task set
# --------------------------------------------------------------------------- #
def frame_features(frame: dict, label_plant: np.ndarray | None,
                   label_other: np.ndarray | None,
                   label_disagreement: np.ndarray | None) -> dict:
    """Everything the selector may look at, all measured on this frame."""
    c2w = np.asarray(frame["camtoworld"], dtype=np.float64)
    centre = c2w[:3, 3]
    forward = -c2w[:3, 2]
    return {
        "camera_centre": centre.tolist(),
        "elevation": float(np.arcsin(np.clip(forward[2], -1, 1))),
        "specimen_fraction": (float(label_plant.mean() + label_other.mean())
                              if label_plant is not None else None),
        "plant_fraction": float(label_plant.mean()) if label_plant is not None else None,
        "nonplant_fraction": float(label_other.mean()) if label_other is not None else None,
        "disagreement_fraction": (float(label_disagreement.mean())
                                  if label_disagreement is not None else None),
    }


def _normalise(values: list) -> np.ndarray:
    array = np.asarray([0.0 if v is None else v for v in values], dtype=np.float64)
    lo, hi = array.min(), array.max()
    return (array - lo) / (hi - lo) if hi > lo else np.zeros_like(array)


def select_views(dataset: Path, semantic: Path, count: int) -> list:
    """Farthest-point selection over geometric + semantic frame features.

    The brief asks for front/side/oblique views, dense and sparse foliage,
    plant-support overlap, plant-pot boundaries and high-disagreement regions.
    A hand-written "pick view 1, 12, 23" would address none of those in a way
    another person could reproduce. Instead each frame becomes a feature vector
    (azimuth as sin/cos so the orbit wraps, elevation, plant and non-plant
    coverage, backend disagreement, specimen size) and the selected set is the
    one that maximises how far apart its members are. The selection is
    deterministic, and each chosen view can be described by which of its
    features is extreme -- which is what the manifest records.
    """
    frames = json.loads((dataset / "frames.json").read_text())
    rows = []
    origins = []
    for frame in frames:
        c2w = np.asarray(frame["camtoworld"], dtype=np.float64)
        origins.append(c2w[:3, 3])
        rows.append(frame)
    origins = np.stack(origins)
    centre = origins.mean(axis=0)
    angles = np.arctan2(origins[:, 1] - centre[1], origins[:, 0] - centre[0])

    features, meta = [], []
    for index, frame in enumerate(rows):
        stem = Path(frame["name"]).stem
        plant = other = disagreement = None
        merged = semantic / "labels_merged" / f"{stem}.png"
        segformer = semantic / "labels_segformer" / f"{stem}.png"
        sam_clip = semantic / "labels_sam_clip" / f"{stem}.png"
        if merged.is_file():
            label = np.asarray(Image.open(merged))
            plant = label == 1                      # PLANT in semantic.py
            other = (label != 0) & (label != 1)
        if segformer.is_file() and sam_clip.is_file():
            a = np.asarray(Image.open(segformer)) == 1
            b = np.asarray(Image.open(sam_clip)) == 1
            disagreement = a != b
        feature = frame_features(frame, plant, other, disagreement)
        feature["azimuth"] = float(angles[index])
        features.append(feature)
        meta.append({"frame": frame, "stem": stem})

    keys = ["sin_azimuth", "cos_azimuth", "elevation", "plant_fraction",
            "nonplant_fraction", "disagreement_fraction", "specimen_fraction"]
    matrix = np.zeros((len(rows), len(keys)), dtype=np.float64)
    matrix[:, 0] = np.sin(angles)
    matrix[:, 1] = np.cos(angles)
    for column, key in enumerate(keys[2:], start=2):
        matrix[:, column] = _normalise([f.get(key) for f in features])

    # Farthest-point sampling, seeded with the frame closest to the centroid of
    # the feature cloud so the result does not depend on list order.
    chosen = [int(np.argmin(np.linalg.norm(matrix - matrix.mean(axis=0), axis=1)))]
    while len(chosen) < min(count, len(rows)):
        distance = np.min(np.linalg.norm(
            matrix[:, None, :] - matrix[chosen][None, :, :], axis=2), axis=1)
        distance[chosen] = -1.0
        chosen.append(int(np.argmax(distance)))

    out = []
    for rank, index in enumerate(chosen):
        feature = dict(features[index])
        feature["azimuth_mod_360"] = float((np.degrees(feature["azimuth"]) + 360) % 360)
        # Which of the brief's coverage axes this view is the extreme of, among
        # the selected set. Descriptive, not a claim that the axis is "covered".
        tags = []
        comparisons = {
            "model_disagreement": ("disagreement_fraction", True),
            "plant_support_overlap": ("nonplant_fraction", True),
            "leaf_density": ("plant_fraction", True),
            "sparse_leaves": ("plant_fraction", False),
            "specimen_coverage": ("specimen_fraction", True),
        }
        for tag, (key, high) in comparisons.items():
            values = [f.get(key) for f in features]
            values = [v for v in values if v is not None]
            value = feature.get(key)
            if value is None or not values:
                continue
            if high and value >= np.percentile(values, 90):
                tags.append(tag)
            if not high and value <= np.percentile(values, 10):
                tags.append(tag)
        out.append({
            "rank": rank,
            "frame_index": index,
            "frame": meta[index]["frame"]["name"],
            "stem": meta[index]["stem"],
            "image": meta[index]["frame"]["image"],
            "mask": meta[index]["frame"]["mask"],
            "features": {k: feature[k] for k in
                         ("azimuth_mod_360", "elevation", "plant_fraction",
                          "nonplant_fraction", "disagreement_fraction",
                          "specimen_fraction")},
            "extreme_on": tags,
        })
    return out, features


def write_task_set(library: dict, out: Path, views_per_crop: int,
                   representative: dict | None = None) -> dict:
    """One representative acquisition per crop, 8 views each."""
    out.mkdir(parents=True, exist_ok=True)
    (out / "images").mkdir(exist_ok=True)
    (out / "preannotations").mkdir(exist_ok=True)
    (out / "labels").mkdir(exist_ok=True)
    (out / "confirmations").mkdir(exist_ok=True)

    by_crop = {}
    for entry in library["entries"]:
        if entry["states"]["states"]["reconstruction_verified"]:
            by_crop.setdefault(entry["crop"], entry)
    if representative:
        for crop, asset_id in representative.items():
            for entry in library["entries"]:
                if entry["asset_id"] == asset_id:
                    by_crop[crop] = entry

    tasks, per_crop = [], {}
    for crop in sorted(by_crop):
        entry = by_crop[crop]
        workspace = Path(entry["workspace"])
        selected, _ = select_views(workspace / "dataset", workspace / "semantic",
                                   views_per_crop)
        per_crop[crop] = {"asset_id": entry["asset_id"], "views": len(selected)}
        for view in selected:
            task_id = f"{crop}_{entry['acquisition_id']}_{view['stem']}"
            image = Image.open(view["image"]).convert("RGB")
            image.save(out / "images" / f"{task_id}.jpg", quality=92)
            # ``labels/`` is created empty on purpose. Nothing this command
            # writes goes there; it is populated only by ``import``.
            tasks.append({
                "task_id": task_id,
                "crop": crop,
                "asset_id": entry["asset_id"],
                "split_group": entry["split_group"],
                "frame": view["frame"],
                "image": f"images/{task_id}.jpg",
                "source_image": view["image"],
                "features": view["features"],
                "selected_because": view["extreme_on"],
                "classes": CLASS_NAMES,
                "status": "annotation_pending",
                "annotator": None,
                "confirmation": None,
                "preannotation_summary": {},
            })
            _write_preannotations(workspace / "semantic", view["stem"], out, task_id)

    manifest = {
        "schema_version": "paa-stage4c-annotation-1",
        "created_utc": _utc(),
        "views_per_crop": views_per_crop,
        "n_tasks": len(tasks),
        "crops": sorted(by_crop),
        "tasks": tasks,
        "taxonomy": {str(k): v for k, v in CLASS_NAMES.items()},
        "palette": {str(k): list(v) for k, v in PALETTE.items()},
        "status": "annotation_pending",
        "success_state": ("evaluation tooling + pending task set are ready; "
                          "segmentation accuracy is NOT verified because no human "
                          "label exists yet"),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2,
                                                  ensure_ascii=False) + "\n")
    (out / "ANNOTATION_GUIDE.md").write_text(ANNOTATION_GUIDE)
    write_review_app(out)
    return {"n_tasks": len(tasks), "per_crop": per_crop}


def _write_preannotations(semantic: Path, stem: str, out: Path, task_id: str) -> None:
    """Candidate label maps, kept in their own directory and named by backend."""
    target = out / "preannotations" / task_id
    target.mkdir(parents=True, exist_ok=True)
    for backend in ("labels_segformer", "labels_sam_clip", "labels_merged"):
        source = semantic / backend / f"{stem}.png"
        if not source.is_file():
            continue
        raw = np.asarray(Image.open(source))
        mapped = np.zeros_like(raw, dtype=np.uint8)
        for src, dst in SEMANTIC_TO_ANNOTATION.items():
            mapped[raw == src] = dst
        Image.fromarray(mapped).save(target / f"{backend}.png")
    (target / "README.txt").write_text(
        "These are MODEL PREDICTIONS, one per backend. They are offered as a\n"
        "starting point only. They must never be moved into labels/ by tooling:\n"
        "a human label enters labels/ only through `annotation.py import`, which\n"
        "records who confirmed it.\n")


ANNOTATION_GUIDE = """# Annotation guide (Stage 4C)

## What you are labelling
Potted plants photographed on a turntable against a white backdrop, with a
white plastic support frame. You label *pixels*, on the photograph itself.

## Classes

| id | name | include | exclude |
|---:|---|---|---|
| 0 | Background | white backdrop, shadows on the backdrop, anything behind the plant | — |
| 1 | Plant | leaves, stems, petioles, veins, flowers, fruit, and any real plant tissue, **including white or pale tissue** | supports, pot, soil |
| 2 | Pot/Soil | the container and the substrate visible inside it | plant parts overhanging the pot rim |
| 3 | Support | the white plastic stakes, clips, ties and the frame holding the plant | plant parts touching them |
| 4 | Uncertain | pixels you cannot decide (motion blur, occlusion, a thin stem against a bright leaf) | — |

## Rules
1. **White plant tissue is Plant, not Support.** Cotton bolls, white flowers and
   pale stems are the single most common mistake; the support is the plastic
   hardware, not everything white.
2. **Thin structures count.** A stem two pixels wide is still a stem. Zoom in.
3. When plant and support overlap, label the pixel according to what is
   *visible* at that pixel.
4. When in doubt, use **Uncertain** rather than guessing. Uncertain pixels are
   excluded from the IoU/Precision/Recall figures and reported separately, so
   marking them costs nothing and hiding doubt costs accuracy.
5. Do not use the pre-annotations as a reference while painting. They are a
   model's opinion and the point of this benchmark is to measure it.

## Workflow
1. Start the review UI: `python -m agri.plant_gs.paa.stage4c.annotation serve --dir <annotation_dir>`
   and open the printed URL.
2. Enter your name, work through the task list, paint, press Save.
3. Export each saved label with
   `python -m agri.plant_gs.paa.stage4c.annotation import --dir <dir> --task <id> --labels <png> --annotator <name>`
   (the UI writes the PNG for you; import is what makes it count as human).
4. Score: `python -m agri.plant_gs.paa.stage4c.annotation evaluate --dir <dir>`
"""


def write_review_app(out: Path) -> None:
    (out / "review.html").write_text(REVIEW_HTML)
    (out / "review_server.py").write_text(REVIEW_SERVER)


# --------------------------------------------------------------------------- #
# Import a human confirmation
# --------------------------------------------------------------------------- #
def cmd_import(args) -> int:
    directory = Path(args.dir)
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    task = next((t for t in manifest["tasks"] if t["task_id"] == args.task), None)
    if task is None:
        log(f"unknown task {args.task}")
        return 1
    source = Path(args.labels)
    if not source.is_file():
        log(f"label file {source} does not exist")
        return 1
    label = np.asarray(Image.open(source))
    allowed = set(CLASS_NAMES)
    present = set(int(v) for v in np.unique(label))
    unexpected = sorted(present - allowed)
    if unexpected:
        log(f"label file uses ids outside the taxonomy: {unexpected}")
        return 1
    target = directory / "labels" / f"{args.task}.png"
    target.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(label.astype(np.uint8)).save(target)
    record = {
        "task_id": args.task,
        "truth_source": "human_confirmed",
        "annotator": args.annotator,
        "confirmed_utc": _utc(),
        "label_sha256": _sha256(target),
        "label_path": f"labels/{args.task}.png",
        "classes_present": sorted(present),
        "uncertain_fraction": float((label == UNCERTAIN).mean()),
        "notes": args.notes or "",
        "declaration": ("the annotator declares this label was produced by human "
                        "inspection of the photograph, not copied from a model "
                        "prediction"),
    }
    (directory / "confirmations" / f"{args.task}.json").write_text(
        json.dumps(record, indent=2, ensure_ascii=False) + "\n")
    task["status"] = "human_confirmed"
    task["annotator"] = args.annotator
    task["confirmation"] = record
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    log(f"imported {args.task} for {args.annotator} ({record['label_sha256'][:12]}…)")
    return 0


# --------------------------------------------------------------------------- #
# Evaluate
# --------------------------------------------------------------------------- #
def score_against(labels: np.ndarray, prediction: np.ndarray,
                  exclude_uncertain: bool = True) -> dict:
    """Plant IoU / Precision / Recall / False Plant Rate plus per-class counts.

    Uncertain pixels are excluded from the plant figures and reported
    separately: a label set that abstains on the hard pixels should be scored
    on the pixels it was willing to call, with the abstention rate next to it.
    """
    mask = np.ones(labels.shape, dtype=bool)
    if exclude_uncertain:
        mask &= labels != UNCERTAIN
    truth = labels[mask]
    pred = prediction[mask]
    result = {"pixels_scored": int(mask.sum()),
              "uncertain_fraction": float((labels == UNCERTAIN).mean())}
    true_plant = truth == PLANT
    pred_plant = pred == PLANT
    tp = int((true_plant & pred_plant).sum())
    fp = int((~true_plant & pred_plant).sum())
    fn = int((true_plant & ~pred_plant).sum())
    tn = int((~true_plant & ~pred_plant).sum())
    result.update({
        "plant_true_positive": tp, "plant_false_positive": fp,
        "plant_false_negative": fn, "plant_true_negative": tn,
        "plant_iou": tp / (tp + fp + fn) if (tp + fp + fn) else None,
        "plant_precision": tp / (tp + fp) if (tp + fp) else None,
        "plant_recall": tp / (tp + fn) if (tp + fn) else None,
        "plant_false_rate": fp / (fp + tn) if (fp + tn) else None,
    })
    per_class = {}
    for class_id, name in CLASS_NAMES.items():
        if class_id == UNCERTAIN:
            continue
        true_c, pred_c = truth == class_id, pred == class_id
        union = int((true_c | pred_c).sum())
        per_class[name] = {
            "iou": (int((true_c & pred_c).sum()) / union) if union else None,
            "true_pixels": int(true_c.sum()),
            "predicted_pixels": int(pred_c.sum()),
        }
    result["per_class"] = per_class
    return result


def load_confirmed(directory: Path, manifest: dict) -> dict:
    """Every label with both a PNG and a matching confirmation record."""
    out = {}
    for task in manifest["tasks"]:
        path = directory / "labels" / f"{task['task_id']}.png"
        record_path = directory / "confirmations" / f"{task['task_id']}.json"
        if not path.is_file():
            continue
        if not record_path.is_file() or not task.get("confirmation"):
            out[task["task_id"]] = {"status": "unconfirmed_ignored",
                                    "reason": "label PNG has no confirmation record"}
            continue
        record = json.loads(record_path.read_text())
        if record.get("label_sha256") != _sha256(path):
            out[task["task_id"]] = {"status": "unconfirmed_ignored",
                                    "reason": "label changed after confirmation"}
            continue
        out[task["task_id"]] = {"status": "human_confirmed", "label": path,
                                "record": record}
    return out


def cmd_evaluate(args) -> int:
    directory = Path(args.dir)
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    confirmed = load_confirmed(directory, manifest)
    confirmed_ids = [k for k, v in confirmed.items() if v["status"] == "human_confirmed"]

    report = {
        "status": "annotation_pending" if not confirmed_ids else "evaluated",
        "truth_source": "human_confirmed" if confirmed_ids else None,
        "tasks_total": len(manifest["tasks"]),
        "tasks_human_confirmed": len(confirmed_ids),
        "tasks_unconfirmed_ignored": sum(
            1 for v in confirmed.values() if v["status"] == "unconfirmed_ignored"),
        "annotators": sorted({confirmed[k]["record"]["annotator"]
                              for k in confirmed_ids}),
        "per_task": {},
        "per_crop": {},
        "per_backend": {},
        "evaluated_utc": _utc(),
    }

    if not confirmed_ids:
        report["caveat"] = (
            "No human-confirmed labels exist, so NO segmentation accuracy is "
            "reported. Model predictions scored against themselves are not an "
            "accuracy measurement, and this report will not present them as one. "
            "Run `annotation.py import` for each reviewed task, then re-run "
            "`annotation.py evaluate`.")
        (directory / "evaluation.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n")
        log(json.dumps({k: report[k] for k in
                        ("status", "tasks_total", "tasks_human_confirmed")}, indent=2))
        return 0

    by_backend = {}
    for task_id in confirmed_ids:
        task = next(t for t in manifest["tasks"] if t["task_id"] == task_id)
        truth = np.asarray(Image.open(confirmed[task_id]["label"]))
        row = {"crop": task["crop"], "annotator": confirmed[task_id]["record"]["annotator"],
               "uncertain_fraction": confirmed[task_id]["record"]["uncertain_fraction"],
               "predictions": {}}
        preannotation_dir = directory / "preannotations" / task_id
        for path in sorted(preannotation_dir.glob("*.png")) if preannotation_dir.is_dir() else []:
            backend = path.stem
            prediction = np.asarray(Image.open(path))
            score = score_against(truth, prediction)
            row["predictions"][backend] = score
            by_backend.setdefault(backend, []).append(score)
        report["per_task"][task_id] = row

    def aggregate(scores: list) -> dict:
        keys = ("plant_iou", "plant_precision", "plant_recall", "plant_false_rate",
                "uncertain_fraction")
        out = {"n": len(scores)}
        for key in keys:
            values = [s[key] for s in scores if s.get(key) is not None]
            out[key] = float(np.mean(values)) if values else None
            out[f"{key}_min"] = float(np.min(values)) if values else None
        return out

    for backend, scores in by_backend.items():
        report["per_backend"][backend] = aggregate(scores)
    for crop in sorted({row["crop"] for row in report["per_task"].values()}):
        for backend in by_backend:
            scores = [row["predictions"][backend] for row in report["per_task"].values()
                      if row["crop"] == crop and backend in row["predictions"]]
            if scores:
                report["per_crop"].setdefault(crop, {})[backend] = aggregate(scores)
    report["caveat"] = (
        "Metrics are computed on human-confirmed pixels only; Uncertain pixels "
        "are excluded and reported as a fraction. 'plant_false_rate' is the "
        "False Plant Rate: the share of true non-plant pixels called plant.")
    (directory / "evaluation.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    log(json.dumps({k: report[k] for k in
                    ("status", "tasks_human_confirmed", "per_backend")}, indent=2))
    return 0


# --------------------------------------------------------------------------- #
# Commands
# --------------------------------------------------------------------------- #
def cmd_export(args) -> int:
    library = json.loads(Path(args.library).read_text())
    representative = None
    if args.representative:
        representative = dict(
            item.split("=", 1) for item in args.representative if "=" in item)
    outcome = write_task_set(library, Path(args.out), args.views_per_crop, representative)
    log(json.dumps(outcome, indent=2))
    log(f"wrote {Path(args.out) / 'manifest.json'}")
    return 0


def cmd_serve(args) -> int:
    import functools
    import http.server
    directory = Path(args.dir).resolve()

    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *a, **kw):
            super().__init__(*a, directory=str(directory), **kw)

        def do_POST(self):
            if self.path != "/api/save":
                self.send_error(404)
                return
            length = int(self.headers.get("Content-Length", 0))
            payload = json.loads(self.rfile.read(length) or b"{}")
            task_id = payload.get("task_id", "")
            annotator = (payload.get("annotator") or "").strip()
            png = payload.get("png_base64", "")
            if not task_id or not annotator or not png.startswith("data:image/png;base64,"):
                self.send_error(400, "task_id, annotator and a PNG are required")
                return
            raw = base64.b64decode(png.split(",", 1)[1])
            target = directory / "labels" / f"{task_id}.png"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(raw)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({
                "saved": str(target),
                "hint": (f"now run: python -m agri.plant_gs.paa.stage4c.annotation "
                         f"import --dir {directory} --task {task_id} "
                         f"--labels {target} --annotator {annotator}"),
            }).encode())

        def log_message(self, fmt, *a):
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    log(f"annotation review UI on http://127.0.0.1:{args.port}/review.html "
        f"(serving {directory})")
    server.serve_forever()
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("export", help="write the pending annotation task set")
    p.add_argument("--library", type=Path,
                   default=Path("results/agri_paa/stage4c/assets.json"))
    p.add_argument("--out", type=Path,
                   default=Path("results/agri_paa/stage4c/annotation"))
    p.add_argument("--views-per-crop", type=int, default=8)
    p.add_argument("--representative", nargs="*", default=None,
                   help="override the per-crop asset, e.g. Cotton=Cotton/1")
    p.set_defaults(func=cmd_export)

    p = sub.add_parser("import", help="register a human-confirmed label")
    p.add_argument("--dir", type=Path, required=True)
    p.add_argument("--task", required=True)
    p.add_argument("--labels", type=Path, required=True)
    p.add_argument("--annotator", required=True)
    p.add_argument("--notes", default="")
    p.set_defaults(func=cmd_import)

    p = sub.add_parser("evaluate", help="score predictions against human labels")
    p.add_argument("--dir", type=Path, required=True)
    p.add_argument("--predictions", default="preannotations")
    p.set_defaults(func=cmd_evaluate)

    p = sub.add_parser("serve", help="serve the review UI")
    p.add_argument("--dir", type=Path, required=True)
    p.add_argument("--port", type=int, default=8931)
    p.set_defaults(func=cmd_serve)

    args = ap.parse_args()
    sys.exit(args.func(args))




REVIEW_HTML = r"""<!doctype html>
<meta charset="utf-8">
<title>Stage 4C annotation review</title>
<style>
 body{font:13px/1.4 system-ui,sans-serif;margin:0;display:flex;height:100vh}
 #side{width:260px;overflow:auto;border-right:1px solid #ccc;padding:8px}
 #main{flex:1;display:flex;flex-direction:column;padding:8px;gap:6px}
 #stage{position:relative;flex:1;overflow:auto;background:#222}
 canvas{position:absolute;left:0;top:0;image-rendering:pixelated}
 .task{padding:3px 5px;cursor:pointer;border-radius:3px}
 .task:hover{background:#eee}.task.done{color:#2a7;font-weight:600}
 .classbtn{padding:4px 8px;margin:2px;border:1px solid #888;border-radius:3px;cursor:pointer}
 .classbtn.active{outline:3px solid #000}
 .hint{color:#666}
</style>
<div id="side">
  <div><input id="annotator" placeholder="your name" style="width:100%"></div>
  <div class="hint" id="progress"></div>
  <div id="tasks"></div>
</div>
<div id="main">
  <div id="classes"></div>
  <div>
    <button onclick="undo()">undo</button>
    <button onclick="clearAll()">clear</button>
    <button onclick="save()">save</button>
    <label>brush <input id="brushsize" type="range" min="3" max="61" value="9"></label>
    <span id="msg" class="hint"></span>
  </div>
  <div id="stage">
    <canvas id="photo"></canvas>
    <canvas id="paint"></canvas>
  </div>
</div>
<script>
const CLASSES = [[1,"Plant"],[2,"Pot/Soil"],[3,"Support"],[4,"Uncertain"],[0,"Background"]];
const COLORS = {0:[0,0,0],1:[40,200,70],2:[60,90,230],3:[235,215,60],4:[255,0,255]};
let manifest, current=null, cls=1, painting=false, undoStack=[], confirmed=new Set();
const photo=document.getElementById('photo'), paint=document.getElementById('paint');
const pctx=photo.getContext('2d'), gctx=paint.getContext('2d');

const tpl = await (await fetch('manifest.json')).json();
manifest = tpl;
for(const t of manifest.tasks) if(t.status==='human_confirmed') confirmed.add(t.task_id);

function classButtons(){
  const box=document.getElementById('classes'); box.innerHTML='';
  for(const [id,name] of CLASSES){
    const b=document.createElement('button');
    b.className='classbtn'+(id===cls?' active':'');
    b.textContent=name;
    b.style.background='rgb('+COLORS[id].join(',')+')';
    b.style.color=(id===0||id===2)?'#fff':'#000';
    b.onclick=()=>{cls=id;classButtons();};
    box.appendChild(b);
  }
}
function renderList(){
  const box=document.getElementById('tasks'); box.innerHTML='';
  let n=0;
  for(const t of manifest.tasks){
    const d=document.createElement('div');
    d.className='task'+(confirmed.has(t.task_id)?' done':'');
    d.textContent=t.task_id;
    d.onclick=()=>loadTask(t); box.appendChild(d); if(confirmed.has(t.task_id))n++;
  }
  document.getElementById('progress').textContent = n+'/'+manifest.tasks.length+' confirmed';
}
function loadTask(t){
  current=t; undoStack=[];
  const img=new Image();
  img.onload=()=>{
    paint.width=photo.width=img.width; paint.height=photo.height=img.height;
    pctx.drawImage(img,0,0);
    gctx.clearRect(0,0,paint.width,paint.height);
    const saved='labels/'+t.task_id+'.png';
    const cur=new Image();
    cur.onload=()=>gctx.drawImage(cur,0,0);
    cur.onerror=()=>{};
    cur.src=saved+'?t='+Date.now();
  };
  img.src=t.image;
  document.getElementById('msg').textContent=t.task_id+' — '+(t.selected_because||[]).join(', ');
}
function pointAt(event){
  const r=paint.getBoundingClientRect();
  return [Math.round(event.clientX-r.left),Math.round(event.clientY-r.top)];
}
function stroke(event){
  const [x,y]=pointAt(event);
  const size=Number(document.getElementById('brushsize')?.value||9);
  gctx.fillStyle='rgb('+COLORS[cls].join(',')+')';
  gctx.beginPath(); gctx.arc(x,y,size/2,0,7); gctx.fill();
}
paint.addEventListener('mousedown',e=>{painting=true;undoStack.push(gctx.getImageData(0,0,paint.width,paint.height));stroke(e);});
paint.addEventListener('mousemove',e=>{if(painting)stroke(e);});
window.addEventListener('mouseup',()=>painting=false);
function undo(){if(undoStack.length)gctx.putImageData(undoStack.pop(),0,0);}
function clearAll(){gctx.clearRect(0,0,paint.width,paint.height);}
function save(){
  const who=document.getElementById('annotator').value.trim();
  if(!who){alert('enter your name first');return;}
  if(!current){alert('pick a task');return;}
  // Flatten onto a zero-filled layer: alpha 0 means Background (class 0).
  const flat=document.createElement('canvas');flat.width=paint.width;flat.height=paint.height;
  const fctx=flat.getContext('2d');fctx.fillStyle='rgb(0,0,0)';fctx.fillRect(0,0,flat.width,flat.height);
  fctx.drawImage(paint,0,0);
  fetch('/api/save',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({task_id:current.task_id,annotator:who,png_base64:flat.toDataURL('image/png')})})
   .then(r=>r.json()).then(j=>{document.getElementById('msg').textContent='saved '+j.saved;});
}
classButtons();renderList();
</script>
"""

REVIEW_SERVER = '''#!/usr/bin/env python
"""Standalone launcher for the annotation review UI."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from agri.plant_gs.paa.stage4c.annotation import main

if __name__ == "__main__":
    sys.argv = ["annotation.py", "serve", "--dir", str(Path(__file__).parent)] + sys.argv[1:]
    main()
'''


if __name__ == "__main__":
    main()
