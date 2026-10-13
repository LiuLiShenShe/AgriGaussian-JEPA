#!/usr/bin/env python
"""End-to-end check: can Gaussian-JEPA's own reader consume this PLY?

`validate_ply.py` checks the file's structure. This goes further and replays
Gaussian-JEPA's exact loading path -- the body of
`datasets/ShapeNet55Gaussian.py::read_gaussian_attribute` -- against the PLY,
then reports the resulting per-Gaussian feature vector.

That vector is the thing the model actually consumes, and its width is the
`C` in Gaussian-JEPA's `[B, N, C]` input. With the standard attribute set
(xyz, opacity, scale, rotation, sh) it must be 3+1+3+4+3 = 14, matching the
shape the release smoke test builds (`[2, 1024, 14]`).

Usage: verify_gsjepa_compatible.py <file.ply>
"""
import argparse
import json
import sys

import numpy as np
from plyfile import PlyData


def np_sigmoid(x):
    return 1 / (1 + np.exp(-x))


def read_gaussian_attribute(vertex, attribute=("xyz", "opacity", "scale",
                                               "rotation", "sh")):
    """Verbatim re-implementation of Gaussian-JEPA's loader body."""
    attribute_index = {}
    data = None
    if "xyz" in attribute:
        x = vertex["x"].astype(np.float32)
        y = vertex["y"].astype(np.float32)
        z = vertex["z"].astype(np.float32)
        data = np.stack((x, y, z), axis=-1)

    if "opacity" in attribute:
        opacity = vertex["opacity"].astype(np.float32).reshape(-1, 1)
        opacity = np_sigmoid(opacity)
        data = np.concatenate((data, opacity), axis=-1)

    if "scale" in attribute and "rotation" in attribute:
        scale_names = [p.name for p in vertex.properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key=lambda s: int(s.split("_")[-1]))
        scales = np.zeros((data.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = vertex[attr_name].astype(np.float32)
        scales = np.exp(scales)  # scale normalization

        rot_names = [p.name for p in vertex.properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key=lambda s: int(s.split("_")[-1]))
        rots = np.zeros((data.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = vertex[attr_name].astype(np.float32)
        rots = rots / (np.linalg.norm(rots, axis=1, keepdims=True) + 1e-9)
        signs_vector = np.sign(rots[:, 0])
        rots = rots * signs_vector[:, None]

        data = np.concatenate((data, scales, rots), axis=-1)

    if "sh" in attribute:
        features_dc = np.zeros((data.shape[0], 3, 1))
        features_dc[:, 0, 0] = vertex["f_dc_0"].astype(np.float32)
        features_dc[:, 1, 0] = vertex["f_dc_1"].astype(np.float32)
        features_dc[:, 2, 0] = vertex["f_dc_2"].astype(np.float32)
        feature_pc = features_dc.reshape(-1, 3)
        data = np.concatenate((data, feature_pc), axis=1)

    return data, attribute_index


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ply")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    ply = PlyData.read(a.ply)
    # Pass the PlyElement, not .data: the loader uses `vertex.properties` as
    # well as `vertex[name]`, and only the element supports both.
    vertex = ply["vertex"]
    n = len(vertex.data)
    data, _ = read_gaussian_attribute(vertex)

    rep = {
        "ply": a.ply,
        "n_gaussians": int(n),
        "ply_element": ply.elements[0].name,
        "feature_vector_width": int(data.shape[1]),
        "expected_width_for_default_attributes": 14,
        "matches_smoke_test_input_width": bool(data.shape[1] == 14),
        "has_nan": bool(np.isnan(data).any()),
        "has_inf": bool(np.isinf(data).any()),
    }
    names = ["x", "y", "z", "opacity", "scale_0", "scale_1", "scale_2",
             "rot_0", "rot_1", "rot_2", "rot_3", "sh_0", "sh_1", "sh_2"]
    rep["columns"] = {}
    for i, nm in enumerate(names[:data.shape[1]]):
        c = data[:, i]
        rep["columns"][nm] = {"min": float(c.min()), "max": float(c.max()),
                              "mean": float(c.mean())}
    # The loaders' own post-conditions.
    op = data[:, 3]
    rep["opacity_in_unit_interval"] = bool(op.min() >= 0.0 and op.max() <= 1.0)
    sc = data[:, 4:7]
    rep["all_scales_positive"] = bool((sc > 0).all())
    rq = data[:, 7:11]
    rep["quat_norms_unit"] = bool(np.allclose(np.linalg.norm(rq, axis=1), 1.0, atol=1e-3))
    rep["quat_first_component_non_negative"] = bool((rq[:, 0] >= 0).all())

    ok = (rep["matches_smoke_test_input_width"] and not rep["has_nan"]
          and not rep["has_inf"] and rep["opacity_in_unit_interval"]
          and rep["all_scales_positive"] and rep["quat_norms_unit"])
    rep["gsjepa_compatible"] = bool(ok)
    print(json.dumps(rep, indent=2))
    if a.json:
        with open(a.json, "w") as f:
            json.dump(rep, f, indent=2)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()