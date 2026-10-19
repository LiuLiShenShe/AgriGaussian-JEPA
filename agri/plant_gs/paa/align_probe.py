"""Probe the geometric relationship between PAA's Camera/images and RGB/whitebg/masks.

Camera/images are smaller than RGB/whitebg/masks and by a *different* ratio per
camera, so the relationship is a crop, not a resize. Recover the crop offset by
multi-scale normalised cross-correlation (cv2.matchTemplate, TM_CCOEFF_NORMED)
over the full valid offset range at each scale (the range is tiny, so no window
search is needed), and report the NCC score so the mapping is verified, not assumed.
"""
import os, json, glob, argparse
import numpy as np, cv2


def load_gray(p, scale=1.0):
    im = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
    if im is None:
        raise IOError(p)
    if scale != 1.0:
        im = cv2.resize(im, None, fx=1.0/scale, fy=1.0/scale, interpolation=cv2.INTER_AREA)
    return im


def match(cam_path, src_path, scales=(8, 4, 2, 1), verbose=False):
    full_cam = load_gray(cam_path, 1.0)
    ch, cw = full_cam.shape
    sh, sw = load_gray(src_path, 1.0).shape
    if sh < ch or sw < cw:
        raise ValueError(f"source {sh}x{sw} smaller than camera {ch}x{cw}")
    best = None
    for f in scales:
        cam = load_gray(cam_path, f)
        src = load_gray(src_path, f)
        r = cv2.matchTemplate(src, cam, cv2.TM_CCOEFF_NORMED)
        _, maxv, _, maxloc = cv2.minMaxLoc(r)
        # maxloc is at scale f; convert to full-res and clamp to the valid range
        dx = int(round(maxloc[0] * f)); dy = int(round(maxloc[1] * f))
        dx = max(0, min(dx, sw - cw)); dy = max(0, min(dy, sh - ch))
        if verbose:
            print(f"    f={f} range=({sw-cw+1}x{sh-ch+1}) -> dx={dx} dy={dy} ncc={maxv:.5f}")
        if best is None or maxv > best[2]:
            best = (dx, dy, float(maxv))
    return dict(dx=best[0], dy=best[1], ncc=round(best[2], 6),
                crop_hw=[ch, cw], source_hw=[sh, sw],
                residual_hw=[sh-ch-best[1], sw-cw-best[0]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera-images", required=True)
    ap.add_argument("--source", required=True)
    ap.add_argument("--ext", default=".jpg")
    ap.add_argument("--limit", type=int, default=6)
    ap.add_argument("--sample", default="spread", choices=["spread", "first"])
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    stems = sorted((os.path.splitext(os.path.basename(p))[0]
                    for p in glob.glob(a.camera_images + "/*")), key=lambda s: int(s))
    n = min(a.limit, len(stems))
    idx = list(range(n)) if a.sample == "first" else list(np.linspace(0, len(stems)-1, n).astype(int))
    res = {}
    for i in idx:
        s = stems[i]
        cam = os.path.join(a.camera_images, s + ".jpg")
        src = os.path.join(a.source, s + a.ext)
        if not os.path.exists(src):
            print("skip (missing source)", s); continue
        r = match(cam, src)
        res[s] = r
        print(f"{s}: dx={r['dx']:5d} dy={r['dy']:5d} ncc={r['ncc']:.5f} "
              f"cam={r['crop_hw']} src={r['source_hw']} residual={r['residual_hw']}")
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)
        print("wrote", a.out)


if __name__ == "__main__":
    main()
