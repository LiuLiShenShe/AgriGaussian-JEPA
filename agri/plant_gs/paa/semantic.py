#!/usr/bin/env python
"""Per-view semantic labels (plant / pot / soil / support) for a PAA dataset.

Why this exists
---------------
PAA's ``RGB_seg/masks`` cover the whole specimen: plant *and* pot *and* support
frame. Stage 4A confirmed this by pixel statistics. A plant-only Gaussian asset
therefore cannot come out of that mask; it needs an extra labelling step, and
the brief is explicit that colour, a fixed height or an XYZ box are not
acceptable on their own.

Two independent backends, deliberately different in kind
--------------------------------------------------------
``segformer``  ``nvidia/segformer-b5-finetuned-ade-640-640`` -- a *trained*
               semantic segmenter (ADE20K, 150 classes). It has real classes for
               ``plant``, ``flower``, ``pot``, ``vase`` and ``earth``. It knows
               these objects from supervised training, not from a prompt.

``sam_clip``   SAM (region proposals) + CLIP (zero-shot text scoring). Class
               boundaries come from SAM, class identity from CLIP.

Neither is ground truth. They disagree, and this module's job is to record that
disagreement rather than average it away: ``--backend both`` writes each
backend's label map plus an ``agreement`` map (plant only where *both* agree).

Honesty about accuracy
----------------------
PAA has no plant/pot annotation, so nothing here can be scored as "accuracy".
The ADE20K model is out of domain for white-backdrop potted-plant photography
and mislabels the blue pot as ``chair`` in some views; the CLIP backend has high
plant recall and low plant precision. What is reported is **agreement between
methods**, and the downstream plant-only asset is evaluated on retention and
residue against those candidate masks -- never claimed as a true segmentation.

Output layout
-------------
``<out>/labels_<backend>/<stem>.png``  uint8 class id per pixel
``<out>/agreement/<stem>.png``         classes on which both backends agree
``<out>/summary_<backend>.json``       per-frame class coverage + provenance
``<out>/agreement.json``               agreement statistics between backends
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
Image.MAX_IMAGE_PIXELS = None

BACKGROUND, PLANT, POT, SOIL, SUPPORT = 0, 1, 2, 3, 4

CLASS_NAMES = {
    BACKGROUND: "background", PLANT: "plant", POT: "pot", SOIL: "soil",
    SUPPORT: "support",
}

CLASS_COLORS = {
    BACKGROUND: (0, 0, 0), PLANT: (40, 200, 70), POT: (60, 90, 230),
    SOIL: (140, 90, 40), SUPPORT: (235, 215, 60),
}

# ADE20K class ids that map onto the taxonomy. Everything else in the specimen
# mask is left as "other" (class 0) rather than forced into a bucket -- a
# wrong label is worse here than an abstention, because a wrong *plant* label
# puts pot or soil into the plant-only asset.
ADE_PLANT = (17, 66)          # plant, flower
ADE_POT = (125, 135)          # pot, vase
ADE_SOIL = (13,)              # earth

PROMPTS = {
    PLANT: [
        "a green leafy plant with stems and leaves",
        "a plant with green leaves",
        "a potted plant growing",
    ],
    POT: ["a blue plastic flower pot", "a plant container pot",
          "a plastic bucket holding a plant"],
    SOIL: ["dark potting soil", "brown soil", "earth and substrate"],
    SUPPORT: ["a white plastic support stake", "a plant support frame",
              "white plastic clips and poles"],
    BACKGROUND: ["a plain white studio background", "empty white backdrop"],
}

DEFAULT_SEGFORMER = (
    "/home/test/.cache/huggingface/hub/models--nvidia--segformer-b5-finetuned-ade-640-640"
    "/snapshots/739f5d4692954e4a185eac280dec1ba5a7d52f1d")
DEFAULT_CLIP = (
    "/home/test/.cache/huggingface/hub/models--openai--clip-vit-base-patch32"
    "/snapshots/3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268")
DEFAULT_SAM = "/data/models/image/sam-vit/1.0.0/sam_vit_h_4b8939.pth"


def log(message: str) -> None:
    print(message, flush=True)


# --------------------------------------------------------------------------- #
# backend: ADE20K SegFormer
# --------------------------------------------------------------------------- #
class SegformerBackend:
    """Trained semantic segmentation; direct class ids, no prompts involved."""

    name = "segformer"

    def __init__(self, model_dir: str, device: str = "cuda"):
        import torch
        from transformers import (SegformerForSemanticSegmentation,
                                  SegformerImageProcessor)
        self.torch = torch
        self.device = device
        self.processor = SegformerImageProcessor.from_pretrained(model_dir)
        self.model = SegformerForSemanticSegmentation.from_pretrained(
            model_dir).to(device).eval()
        self.id2label = dict(self.model.config.id2label)
        self.model_dir = model_dir

    def provenance(self) -> dict:
        return {"backend": self.name, "model": "nvidia/segformer-b5-finetuned-ade-640-640",
                "model_dir": self.model_dir, "training": "supervised, ADE20K 150 classes",
                "class_map": {"plant": list(ADE_PLANT), "pot": list(ADE_POT),
                              "soil": list(ADE_SOIL),
                              "labels": {str(i): self.id2label.get(i, str(i))
                                         for i in list(ADE_PLANT) + list(ADE_POT)
                                         + list(ADE_SOIL)}}}

    def label_frame(self, image: np.ndarray, specimen_mask) -> tuple:
        torch = self.torch
        inputs = self.processor(images=Image.fromarray(image),
                                return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        logits = torch.nn.functional.interpolate(
            outputs.logits, size=image.shape[:2], mode="bilinear",
            align_corners=False)
        pred = logits.argmax(1)[0].cpu().numpy()

        label_map = np.zeros(pred.shape, dtype=np.uint8)
        label_map[np.isin(pred, list(ADE_PLANT))] = PLANT
        label_map[np.isin(pred, list(ADE_POT))] = POT
        label_map[np.isin(pred, list(ADE_SOIL))] = SOIL
        if specimen_mask is not None:
            label_map[~specimen_mask] = BACKGROUND

        counts = {int(c): int((pred == c).sum()) for c in
                  np.unique(pred[specimen_mask])} if specimen_mask is not None else {}
        detail = {
            "top_classes_in_specimen": sorted(
                ((self.id2label.get(c, str(c)), n) for c, n in counts.items()),
                key=lambda kv: -kv[1])[:8],
            "label_counts": {CLASS_NAMES[c]: int((label_map == c).sum())
                             for c in CLASS_NAMES},
        }
        return label_map, detail


# --------------------------------------------------------------------------- #
# backend: SAM proposals + CLIP zero-shot scoring
# --------------------------------------------------------------------------- #
class SamClipBackend:
    """Class-agnostic regions from SAM, class identity from CLIP text prompts."""

    name = "sam_clip"

    def __init__(self, sam_checkpoint: str, clip_dir: str, device: str = "cuda",
                 points_per_side: int = 32):
        import torch
        from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
        from transformers import CLIPModel, CLIPProcessor
        self.torch = torch
        self.device = device
        sam = sam_model_registry["vit_h"](checkpoint=sam_checkpoint).to(device)
        self.generator = SamAutomaticMaskGenerator(
            sam, points_per_side=points_per_side, pred_iou_thresh=0.86,
            stability_score_thresh=0.90, min_mask_region_area=500)
        self.clip = CLIPModel.from_pretrained(clip_dir).to(device).eval()
        self.processor = CLIPProcessor.from_pretrained(clip_dir)
        self.sam_checkpoint = sam_checkpoint
        self.clip_dir = clip_dir
        flat, owner = [], []
        for class_id, prompts in PROMPTS.items():
            for prompt in prompts:
                flat.append(prompt)
                owner.append(class_id)
        self.flat_prompts = flat
        self.owner = np.asarray(owner)

    def provenance(self) -> dict:
        return {"backend": self.name, "sam": "vit_h", "sam_checkpoint": self.sam_checkpoint,
                "clip": "openai/clip-vit-base-patch32", "clip_dir": self.clip_dir,
                "training": "SAM supervised (SA-1B) + CLIP zero-shot text prompts",
                "prompts": {CLASS_NAMES[k]: v for k, v in PROMPTS.items()}}

    def label_frame(self, image: np.ndarray, specimen_mask, downscale: int = 2) -> tuple:
        torch = self.torch
        height, width = image.shape[:2]
        small = image
        if downscale > 1:
            small = np.asarray(Image.fromarray(image).resize(
                (max(1, width // downscale), max(1, height // downscale)),
                Image.Resampling.LANCZOS))
        masks = self.generator.generate(small)
        if not masks:
            return np.zeros((height, width), np.uint8), {"regions": 0}

        crops = []
        for mask in masks:
            x, y, w, h = mask["bbox"]
            crop = np.asarray(small[y:y + h, x:x + w]).copy()
            if crop.size == 0:
                crops.append(Image.fromarray(np.full((8, 8, 3), 128, np.uint8)))
                continue
            crop[~mask["segmentation"][y:y + h, x:x + w]] = 255
            crops.append(Image.fromarray(crop))
        inputs = self.processor(text=self.flat_prompts, images=crops,
                                return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            logits = self.clip(**inputs).logits_per_image
        # Per-class score is the MAX over that class's prompts, on the raw
        # similarity scale. Averaging over prompts of unequal count biases the
        # comparison, and a softmax over all prompts mixes classes together.
        scores = {c: logits[:, self.owner == c].max(dim=1).values.cpu().numpy()
                  for c in PROMPTS}
        keys = list(PROMPTS)
        stacked = np.stack([scores[c] for c in keys], axis=1)
        best = stacked.argmax(axis=1)

        label_small = np.zeros(small.shape[:2], np.uint8)
        order = np.argsort([-m["area"] for m in masks])
        for i in order:
            label_small[masks[i]["segmentation"]] = keys[best[i]]
        label_map = label_small if downscale == 1 else np.asarray(
            Image.fromarray(label_small).resize((width, height),
                                                Image.Resampling.NEAREST))
        label_map = np.array(label_map)
        if specimen_mask is not None:
            label_map[~specimen_mask] = BACKGROUND

        regions = [{
            "area": int(m["area"]), "bbox": [int(v) for v in m["bbox"]],
            "class": CLASS_NAMES[keys[best[i]]],
            "scores": {CLASS_NAMES[c]: round(float(stacked[i, j]), 4)
                       for j, c in enumerate(keys)},
        } for i, m in enumerate(masks)]
        return label_map, {"regions": len(masks), "region_detail": regions}


BACKENDS = {"segformer": SegformerBackend, "sam_clip": SamClipBackend}


def build_backend(name: str, args):
    if name == "segformer":
        return SegformerBackend(args.segformer_dir, args.device)
    return SamClipBackend(args.sam_checkpoint, args.clip_dir, args.device,
                          args.points_per_side)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--backend", choices=["segformer", "sam_clip", "both"],
                    default="both")
    ap.add_argument("--segformer-dir", default=DEFAULT_SEGFORMER)
    ap.add_argument("--clip-dir", default=DEFAULT_CLIP)
    ap.add_argument("--sam-checkpoint", default=DEFAULT_SAM)
    ap.add_argument("--points-per-side", type=int, default=32)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--downscale", type=int, default=2,
                    help="SAM/CLIP working scale (sam_clip backend only)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--every", type=int, default=1)
    ap.add_argument("--overlay-limit", type=int, default=12)
    args = ap.parse_args()

    started = time.time()
    frames = json.loads((args.dataset / "frames.json").read_text())
    if args.every > 1:
        frames = frames[::args.every]
    if args.limit:
        frames = frames[:args.limit]

    names = ["segformer", "sam_clip"] if args.backend == "both" else [args.backend]
    models = {n: build_backend(n, args) for n in names}
    for n in names:
        (args.out / f"labels_{n}").mkdir(parents=True, exist_ok=True)
    (args.out / "agreement").mkdir(parents=True, exist_ok=True)
    (args.out / "overlays").mkdir(parents=True, exist_ok=True)

    coverage = {n: {} for n in names}
    details = {n: {} for n in names}
    agreement_stats = {}
    for index, frame in enumerate(frames):
        stem = Path(frame["name"]).stem
        image = np.asarray(Image.open(frame["image"]).convert("RGB"))
        specimen = np.asarray(Image.open(frame["mask"]).convert("L")) > 127
        label_maps = {}
        for n in names:
            if n == "sam_clip":
                label_map, detail = models[n].label_frame(image, specimen,
                                                          args.downscale)
            else:
                label_map, detail = models[n].label_frame(image, specimen)
            label_maps[n] = label_map
            details[n][frame["name"]] = detail
            Image.fromarray(label_map).save(args.out / f"labels_{n}" / f"{stem}.png")
            spec = label_map[specimen]
            coverage[n][frame["name"]] = {
                cls: float((spec == cid).mean()) for cid, cls in CLASS_NAMES.items()}

        if len(names) == 2:
            a, b = label_maps["segformer"], label_maps["sam_clip"]
            agree = np.where(a == b, a, BACKGROUND).astype(np.uint8)
            agree[~specimen] = BACKGROUND
            Image.fromarray(agree).save(args.out / "agreement" / f"{stem}.png")
            pa, pb = (a == PLANT), (b == PLANT)
            union = int((pa | pb).sum())
            agreement_stats[frame["name"]] = {
                "plant_iou": float((pa & pb).sum() / union) if union else None,
                "plant_in_both": float((pa & pb).mean()),
                "plant_only_segformer": float((pa & ~pb).mean()),
                "plant_only_sam_clip": float((pb & ~pa).mean()),
                "both_agree_plant_fraction_of_specimen":
                    float(agree[specimen].astype(int).__eq__(PLANT).mean()),
            }

        if index < args.overlay_limit:
            for n in names:
                overlay = image.copy()
                for class_id, colour in CLASS_COLORS.items():
                    if class_id == BACKGROUND:
                        continue
                    overlay[label_maps[n] == class_id] = colour
                blended = (0.55 * image.astype(np.float32)
                           + 0.45 * overlay.astype(np.float32)).astype(np.uint8)
                Image.fromarray(blended).save(
                    args.out / "overlays" / f"{stem}_{n}.jpg", quality=90)
        summary = " ".join(
            f"{n}[{CLASS_NAMES[c]}={coverage[n][frame['name']][CLASS_NAMES[c]]:.2f}]"
            for n in names for c in (PLANT, POT))
        log(f"[{index + 1}/{len(frames)}] {frame['name']} {summary}")

    payload = {
        "status": "PASS",
        "dataset": str(args.dataset.resolve()),
        "frames_labelled": len(frames),
        "backends": {n: models[n].provenance() for n in names},
        "mean_coverage_within_specimen_mask": {
            n: {cls: float(np.mean([v[cls] for v in coverage[n].values()]))
                for cls in CLASS_NAMES.values()} for n in names},
        "per_frame_coverage": coverage,
        "seconds": round(time.time() - started, 1),
        "scope": ("candidate semantic labelling; PAA has no plant/pot annotation, so "
                  "this is agreement between methods, not segmentation accuracy"),
    }
    (args.out / "summary.json").write_text(json.dumps(payload, indent=2) + "\n")
    for n in names:
        (args.out / f"details_{n}.json").write_text(json.dumps(details[n], indent=2) + "\n")
    if agreement_stats:
        ious = [v["plant_iou"] for v in agreement_stats.values()
                if v["plant_iou"] is not None]
        payload_agreement = {
            "per_frame": agreement_stats,
            "plant_iou_median": float(np.median(ious)) if ious else None,
            "plant_iou_min": float(np.min(ious)) if ious else None,
            "note": ("IoU here measures agreement between two independent candidate "
                     "labellings, not accuracy against ground truth"),
        }
        (args.out / "agreement.json").write_text(
            json.dumps(payload_agreement, indent=2) + "\n")
        log(f"plant agreement IoU median {payload_agreement['plant_iou_median']}")
    log(json.dumps(payload["mean_coverage_within_specimen_mask"], indent=2))


if __name__ == "__main__":
    main()
