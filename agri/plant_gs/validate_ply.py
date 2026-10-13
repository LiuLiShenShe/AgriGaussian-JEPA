#!/usr/bin/env python
"""Validate that a PLY really is a 3DGS Gaussian splat file.

The point of this script is to refuse to be fooled. A plain xyz (or xyz+rgb)
point cloud is NOT a Gaussian splat cloud and cannot be fed to Gaussian-JEPA
even though both are ".ply". So the check is explicit and strict:

  required-any: x y z
  required:     opacity
  required:     scale_0 scale_1 scale_2
  required:     rot_0 rot_1 rot_2 rot_3         (normalised quaternion)
  required:     f_dc_0 f_dc_1 f_dc_2            (SH band 0 / base colour)

and it reports, per field, whether the values are in a physically plausible
range, plus NaN/Inf counts and the bounding box.

Usage: validate_ply.py <file.ply> [--json out.json]
"""
import argparse
import json
import struct
import sys

import numpy as np

PLY_TYPES = {
    "char": "b", "uchar": "B", "short": "h", "ushort": "H",
    "int": "i", "uint": "I", "float": "f", "double": "d",
}
NP_TYPES = {
    "char": "i1", "uchar": "u1", "short": "i2", "ushort": "u2",
    "int": "i4", "uint": "u4", "float": "f4", "double": "f8",
}


def read_ply(path):
    """Read an ascii or binary_little_endian PLY into {name: ndarray}."""
    with open(path, "rb") as f:
        raw = f.read()
    if not raw.startswith(b"ply"):
        raise ValueError("not a PLY file (bad magic)")

    end = raw.find(b"end_header\n")
    if end < 0:
        raise ValueError("PLY header has no end_header")
    header = raw[:end].decode("ascii", errors="replace").splitlines()
    body_start = end + len(b"end_header\n")

    fmt = None
    elements = []          # (name, count, [(prop, type)])
    cur = None
    for line in header[1:]:
        t = line.split()
        if not t:
            continue
        if t[0] == "format":
            fmt = t[1]
        elif t[0] == "element":
            cur = (t[1], int(t[2]), [])
            elements.append(cur)
        elif t[0] == "property" and cur is not None:
            if t[1] == "list":
                raise ValueError("list properties are not supported here")
            cur[2].append((t[2], t[1]))

    counts = {name: cnt for name, cnt, _ in elements}
    info = {"format": fmt, "elements": {n: c for n, c in counts.items()}}

    if len(elements) != 1:
        raise ValueError("expected exactly one element, got %s" % [e[0] for e in elements])
    ename, n, props = elements[0]
    names = [p[0] for p in props]

    if fmt == "ascii":
        vals = np.loadtxt(path, skiprows=header.index("end_header") + 1)
        data = {nm: vals[:, i] for i, nm in enumerate(names)}
    elif fmt == "binary_little_endian":
        dt = np.dtype([(nm, "<" + NP_TYPES[ty]) for nm, ty in props])
        arr = np.frombuffer(raw, dtype=dt, count=n, offset=body_start)
        data = {nm: np.asarray(arr[nm]) for nm in names}
    else:
        raise ValueError("unsupported PLY format: %s" % fmt)

    return data, info


def check(path):
    data, info = read_ply(path)
    names = list(data.keys())
    n = len(next(iter(data.values())))

    required = ["x", "y", "z", "opacity",
                "scale_0", "scale_1", "scale_2",
                "rot_0", "rot_1", "rot_2", "rot_3",
                "f_dc_0", "f_dc_1", "f_dc_2"]
    missing = [k for k in required if k not in data]
    present = [k for k in required if k in data]
    f_rest = sorted([k for k in names if k.startswith("f_rest_")],
                    key=lambda s: int(s.split("_")[-1]))

    report = {
        "path": path,
        "file_format": info["format"],
        "num_gaussians": n,
        "num_properties": len(names),
        "required_present": present,
        "required_missing": missing,
        "is_valid_3dgs": len(missing) == 0,
        "has_higher_order_sh": len(f_rest) > 0,
        "num_f_rest": len(f_rest),
        "fields": {},
        "problems": [],
    }

    for k in names:
        v = data[k].astype(np.float64)
        nan = int(np.isnan(v).sum())
        inf = int(np.isinf(v).sum())
        finite = v[np.isfinite(v)]
        rec = {
            "dtype": str(data[k].dtype),
            "nan": nan, "inf": inf,
            "min": float(finite.min()) if finite.size else None,
            "max": float(finite.max()) if finite.size else None,
            "mean": float(finite.mean()) if finite.size else None,
        }
        report["fields"][k] = rec
        if nan or inf:
            report["problems"].append(f"{k}: {nan} NaN, {inf} Inf")

    if missing:
        report["problems"].append(
            "MISSING 3DGS attributes: %s -- this is a plain point cloud, NOT a "
            "Gaussian splat file" % ", ".join(missing))
        return report

    xyz = np.stack([data["x"], data["y"], data["z"]], axis=1).astype(np.float64)
    report["bbox_min"] = xyz.min(axis=0).tolist()
    report["bbox_max"] = xyz.max(axis=0).tolist()
    report["bbox_extent"] = (xyz.max(axis=0) - xyz.min(axis=0)).tolist()

    op = data["opacity"].astype(np.float64)
    report["opacity_range"] = [float(op.min()), float(op.max())]
    # 3DGS stores opacity as a RAW LOGIT; consumers apply sigmoid themselves
    # (Gaussian-JEPA's loader does `np_sigmoid(vertex["opacity"])`). A file whose
    # opacity already lies inside [0,1] has been written pre-activated, which
    # would get sigmoid() applied a second time downstream.
    n_out01 = int(((op >= 0) & (op <= 1)).sum())
    report["opacity_fraction_in_unit_interval"] = n_out01 / max(n, 1)
    report["opacity_convention"] = ("logit (correct)" if op.min() < -0.05 or op.max() > 1.05
                                    else "already sigmoid-activated (SUSPECT)")
    if report["opacity_convention"].startswith("already"):
        report["problems"].append(
            "opacity looks already sigmoid-activated; a 3DGS loader will apply "
            "sigmoid() again. Expected raw logits.")

    scales = np.stack([data["scale_%d" % i] for i in range(3)], axis=1).astype(np.float64)
    linear = np.exp(scales)
    report["scale_range"] = [float(scales.min()), float(scales.max())]
    report["scale_linear_min"] = float(linear.min())
    report["scale_linear_max"] = float(linear.max())
    # Log-scales are overwhelmingly negative; linear scales are all positive and
    # typically small. Use the sign of the minimum to tell them apart.
    report["scale_convention"] = ("log (correct)" if scales.min() < -0.5
                                  else "already exp'd linear scale (SUSPECT)")
    if scales.min() >= 0:
        report["problems"].append(
            "all scale_* values are non-negative, i.e. exp() was already applied; "
            "a 3DGS loader will apply exp() again. Expected raw log-scales.")
    if scales.shape[1] == 3 and np.allclose(scales.min(axis=1), scales.max(axis=1), atol=1e-9):
        report["scale_is_isotropic"] = True

    quats = np.stack([data["rot_%d" % i] for i in range(4)], axis=1).astype(np.float64)
    norms = np.linalg.norm(quats, axis=1)
    report["quat_norm_min"] = float(norms.min())
    report["quat_norm_max"] = float(norms.max())
    n_bad = int((np.abs(norms - 1.0) > 0.01).sum())
    report["quat_non_unit_count"] = n_bad
    if n_bad > 0.01 * n:
        report["problems"].append(
            "%d/%d quaternions are not unit-norm (>1%% deviation)" % (n_bad, n))

    zero_scale = int((linear < 1e-8).all(axis=1).sum())
    report["degenerate_scale_count"] = zero_scale
    if zero_scale > 0.05 * n:
        report["problems"].append("%d Gaussians have a degenerate scale" % zero_scale)

    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ply")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    rep = check(a.ply)
    print(json.dumps(rep, indent=2))
    if a.json:
        with open(a.json, "w") as f:
            json.dump(rep, f, indent=2)
    sys.exit(0 if rep["is_valid_3dgs"] and not rep["problems"] else 1)


if __name__ == "__main__":
    main()