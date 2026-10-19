"""Establish, and independently check, how PAA's RGB_seg masks map onto Camera/images.

Three facts have to be reconciled before any masked 3DGS run is trustworthy:

  * Camera/images are SMALLER than RGB/whitebg/RGB_seg/masks, by a different
    ratio per camera -> the relationship is a crop, not a resize.
  * Images 100-102 carry EXIF orientation 6; COLMAP's intrinsics for camera 1
    (3070x2020, principal point at 1535/1010) match the RAW, EXIF-ignored frame
    (verified separately by keypoint-gradient contrast: 3.8 RAW vs 1.8 EXIF).
  * The masks carry no EXIF at all and are 2160x3840 portrait for every image,
    i.e. they are in the DISPLAY orientation of the source frames.

Two independent checks, deliberately using different signal:

  A. mask(display) vs whitebg(display): the mask should coincide with the
     non-white pixels of the white-background render. Reported as IoU.
  B. whitebg(display) vs Camera/images(raw): photo-to-photo NCC gives the
     rotation needed for the EXIF'd frames and the crop offset.

Composing A and B yields mask -> Camera/images(raw), and both are reported so
the composition is auditable rather than assumed.
"""
import os, json, glob, argparse
import numpy as np
from PIL import Image, ImageOps

Image.MAX_IMAGE_PIXELS = None
EXIF_ORIENT = 274


def exif_orient(path):
    try:
        return Image.open(path).getexif().get(EXIF_ORIENT)
    except Exception:
        return None


def load_display(path):
    """Load as a viewer would render it (EXIF applied)."""
    return ImageOps.exif_transpose(Image.open(path))


def align_slide(small_img, big_img, scales=(8, 4, 2, 1)):
    """Find where `small_img` sits inside `big_img` (both PIL, same orientation).

    Returns (dx, dy, ncc) with small_img == big_img[dy:dy+h, dx:dx+w].
    Implemented on numpy so it does not depend on OpenCV's Sobel/resize quirks.
    """
    import cv2
    # ascontiguousarray: arrays produced by np.asarray(PIL) are read-only views
    # that OpenCV 5 rejects ("src is not a numpy array").
    big0 = np.ascontiguousarray(np.asarray(big_img.convert("L"), dtype=np.uint8))
    small0 = np.ascontiguousarray(np.asarray(small_img.convert("L"), dtype=np.uint8))
    bh, bw = big0.shape
    sh, sw = small0.shape
    assert bh >= sh and bw >= sw
    best = None
    for f in scales:
        b = cv2.resize(big0, (max(1, bw // f), max(1, bh // f)), interpolation=cv2.INTER_AREA)
        s = cv2.resize(small0, (max(1, sw // f), max(1, sh // f)), interpolation=cv2.INTER_AREA)
        if s.shape[0] > b.shape[0] or s.shape[1] > b.shape[1]:
            continue
        r = cv2.matchTemplate(b, s, cv2.TM_CCOEFF_NORMED)
        _, mx, _, loc = cv2.minMaxLoc(r)
        dx = int(round(loc[0] * f)); dy = int(round(loc[1] * f))
        dx = max(0, min(dx, bw - sw)); dy = max(0, min(dy, bh - sh))
        if best is None or mx > best[2]:
            best = (dx, dy, float(mx))
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample-dir", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    S = a.sample_dir
    cdir = os.path.join(S, "Camera/images")
    mdir = os.path.join(S, "RGB_seg/masks")
    wdir = os.path.join(S, "RGB_seg/whitebg")

    stems = sorted((os.path.splitext(os.path.basename(p))[0]
                    for p in glob.glob(cdir + "/*")), key=int)
    report = {"n_images": len(stems), "per_image": {}}
    for s in stems:
        cam_p = os.path.join(cdir, s + ".jpg")
        mask_p = os.path.join(mdir, s + ".png")
        wb_p = os.path.join(wdir, s + ".jpg")
        orient = exif_orient(cam_p)
        cam_raw = Image.open(cam_p)                       # raw, EXIF ignored
        cam_disp = load_display(cam_p)
        mask = Image.open(mask_p)                          # no EXIF -> display frame
        wb_disp = load_display(wb_p)

        rec = {"exif_orient": orient,
               "cam_raw_size": list(cam_raw.size), "cam_disp_size": list(cam_disp.size),
               "mask_size": list(mask.size), "whitebg_disp_size": list(wb_disp.size)}

        # ---- check A: mask vs non-white pixels of whitebg (display frame) ----
        if mask.size == wb_disp.size:
            wb = np.asarray(wb_disp.convert("RGB"), dtype=np.uint8)
            nonwhite = (wb.min(axis=2) < 235)
            m = np.ascontiguousarray(np.asarray(mask.convert("L"))) > 127
            inter = np.logical_and(m, nonwhite).sum()
            union = np.logical_or(m, nonwhite).sum()
            rec["A_mask_vs_nonwhite_iou"] = float(inter / max(union, 1))
            rec["A_mask_cov"] = float(m.mean()); rec["A_nonwhite_cov"] = float(nonwhite.mean())
        else:
            rec["A_mask_vs_nonwhite_iou"] = None

        # ---- check B: whitebg(display) inside Camera/images(raw) ----
        # Bring whitebg into the raw frame first (undo the EXIF rotation).
        if orient == 6:
            wb_for_cam = wb_disp.rotate(90, expand=True)   # display -> raw
        elif orient == 8:
            wb_for_cam = wb_disp.rotate(-90, expand=True)
        elif orient == 3:
            wb_for_cam = wb_disp.rotate(180, expand=True)
        else:
            wb_for_cam = wb_disp
        rec["wb_for_cam_size"] = list(wb_for_cam.size)
        dx, dy, ncc = align_slide(cam_raw, wb_for_cam)
        rec.update(dx_raw=dx, dy_raw=dy, B_ncc=ncc,
                   residual=[wb_for_cam.size[1]-cam_raw.size[1]-dy,
                             wb_for_cam.size[0]-cam_raw.size[0]-dx])
        report["per_image"][s] = rec

    # aggregate
    cams = {}
    for s, r in report["per_image"].items():
        cams.setdefault((r["cam_raw_size"][0], r["cam_raw_size"][1], r["exif_orient"]), []).append(s)
    report["groups"] = {f"{k[0]}x{k[1]}_exif{k[2]}": {"n": len(v), "stems": sorted(v, key=int)[:6]} for k, v in cams.items()}
    json.dump(report, open(a.out, "w"), indent=1)

    print(f"images: {len(stems)}")
    print(f"{'stem':>5} {'cam_raw':>12} {'exif':>4} {'mask':>10} {'A_iou':>7} {'dx':>5} {'dy':>5} {'B_ncc':>7} {'resid':>12}")
    for s in stems:
        r = report["per_image"][s]
        if int(s) not in (1, 20, 40, 60, 80, 99, 100, 101, 102):
            continue
        print(f"{s:>5} {str(r['cam_raw_size']):>12} {str(r['exif_orient']):>4} {str(r['mask_size']):>10} "
              f"{('%.4f'%r['A_mask_vs_nonwhite_iou']) if r['A_mask_vs_nonwhite_iou'] is not None else '  n/a ':>7} "
              f"{r['dx_raw']:>5} {r['dy_raw']:>5} {r['B_ncc']:>7.4f} {str(r['residual']):>12}")
    # consistency within group
    import collections
    for k, v in sorted(cams.items()):
        offs = collections.Counter((report["per_image"][s]["dx_raw"], report["per_image"][s]["dy_raw"]) for s in v)
        ious = [report["per_image"][s]["A_mask_vs_nonwhite_iou"] for s in v if report["per_image"][s]["A_mask_vs_nonwhite_iou"]]
        print(f"group {k[0]}x{k[1]} exif={k[2]}: n={len(v)} offsets={offs.most_common(4)} "
              f"iou mean={np.mean(ious):.4f} min={np.min(ious):.4f}")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
