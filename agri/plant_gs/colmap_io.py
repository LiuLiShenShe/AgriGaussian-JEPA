"""Minimal reader/writer for COLMAP binary sparse models.

Self-contained on purpose: gsplat's own `datasets.colmap.Parser` requires
`pycolmap`, which is not installed here and which would drag in a numpy version
incompatible with this env. Only the subset of the format we need is
implemented.

Reference: https://colmap.github.io/format.html
"""
import collections
import struct

import numpy as np

CameraModel = collections.namedtuple("CameraModel", ["model_id", "model_name", "num_params"])
Camera = collections.namedtuple("Camera", ["id", "model", "width", "height", "params"])
Image = collections.namedtuple("Image", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"])
Point3D = collections.namedtuple("Point3D", ["id", "xyz", "rgb", "error", "image_ids", "point2D_idxs"])

CAMERA_MODELS = {
    0: CameraModel(0, "SIMPLE_PINHOLE", 3),
    1: CameraModel(1, "PINHOLE", 4),
    2: CameraModel(2, "SIMPLE_RADIAL", 4),
    3: CameraModel(3, "RADIAL", 5),
    4: CameraModel(4, "OPENCV", 8),
    5: CameraModel(5, "OPENCV_FISHEYE", 8),
    6: CameraModel(6, "FULL_OPENCV", 12),
    7: CameraModel(7, "FOV", 5),
    8: CameraModel(8, "SIMPLE_RADIAL_FISHEYE", 4),
    9: CameraModel(9, "RADIAL_FISHEYE", 5),
    10: CameraModel(10, "THIN_PRISM_FISHEYE", 12),
}
CAMERA_MODEL_IDS = {m.model_name: m.model_id for m in CAMERA_MODELS.values()}


def _read(f, fmt):
    size = struct.calcsize(fmt)
    data = f.read(size)
    if len(data) != size:
        raise EOFError("unexpected end of COLMAP binary file")
    return struct.unpack(fmt, data)


def read_cameras_binary(path):
    cameras = {}
    with open(path, "rb") as f:
        (num_cameras,) = _read(f, "<Q")
        for _ in range(num_cameras):
            camera_id, model_id, width, height = _read(f, "<iiQQ")
            model = CAMERA_MODELS[model_id]
            params = _read(f, "<" + "d" * model.num_params)
            cameras[camera_id] = Camera(camera_id, model.model_name, width, height,
                                        np.array(params, dtype=np.float64))
    return cameras


def read_images_binary(path):
    images = {}
    with open(path, "rb") as f:
        (num_reg_images,) = _read(f, "<Q")
        for _ in range(num_reg_images):
            props = _read(f, "<i" + "d" * 7 + "i")  # id, qvec(4), tvec(3), camera_id
            image_id, qvec, tvec, camera_id = props[0], np.array(props[1:5]), np.array(props[5:8]), props[8]
            name_chars = []
            while True:
                ch = f.read(1)
                if ch == b"\x00":
                    break
                name_chars.append(ch)
            name = b"".join(name_chars).decode("utf-8")
            (num_points2D,) = _read(f, "<Q")
            data = _read(f, "<" + "ddq" * num_points2D)
            xys = np.column_stack([tuple(map(float, data[0::3])), tuple(map(float, data[1::3]))])
            point3D_ids = np.array(tuple(map(int, data[2::3])))
            images[image_id] = Image(image_id, qvec, tvec, camera_id, name, xys, point3D_ids)
    return images


def read_points3D_binary(path):
    points = {}
    with open(path, "rb") as f:
        (num_points,) = _read(f, "<Q")
        for _ in range(num_points):
            props = _read(f, "<QdddBBBd")
            point_id = props[0]
            xyz = np.array(props[1:4])
            rgb = np.array(props[4:7])
            error = np.array(props[7])
            track_length, = _read(f, "<Q")
            track = _read(f, "<" + "ii" * track_length) if track_length else ()
            image_ids = np.array(tuple(map(int, track[0::2])))
            point2D_idxs = np.array(tuple(map(int, track[1::2])))
            points[point_id] = Point3D(point_id, xyz, rgb, error, image_ids, point2D_idxs)
    return points


def read_model(path):
    return (read_cameras_binary(path + "/cameras.bin"),
            read_images_binary(path + "/images.bin"),
            read_points3D_binary(path + "/points3D.bin"))


def qvec2rotmat(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
        [2 * x * y + 2 * z * w, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * x * w],
        [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x * x - 2 * y * y],
    ])


def image_camtoworld(img):
    """COLMAP stores world-to-camera; return the cam-to-world 4x4."""
    R = qvec2rotmat(img.qvec)
    t = img.tvec
    c2w = np.eye(4)
    c2w[:3, :3] = R.T
    c2w[:3, 3] = -R.T @ t
    return c2w