"""Standard 3DGS PLY decoding for the isolated real-plant JEPA smoke test."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from plyfile import PlyData
from torch.utils.data import Dataset

ATTRIBUTES = ['xyz', 'opacity', 'scale', 'rotation', 'sh']
REQUIRED_FIELDS = (
    ['x', 'y', 'z', 'opacity']
    + [f'scale_{i}' for i in range(3)]
    + [f'rot_{i}' for i in range(4)]
    + [f'f_dc_{i}' for i in range(3)]
)


def load_gaussian_ply(path: Path) -> np.ndarray:
    """Decode logit opacity/log scales into the official 14-D attribute order."""
    from datasets.ShapeNet55Gaussian import read_gaussian_attribute

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    ply = PlyData.read(path)
    if 'vertex' not in ply:
        raise ValueError(f'PLY has no vertex element: {path}')
    vertex = ply['vertex']
    missing = sorted(set(REQUIRED_FIELDS) - set(vertex.data.dtype.names))
    if missing:
        raise ValueError(f'missing Gaussian attributes in {path}: {missing}')
    if not len(vertex):
        raise ValueError(f'empty Gaussian PLY: {path}')
    raw = np.column_stack([vertex[name] for name in REQUIRED_FIELDS])
    if not np.isfinite(raw).all():
        raise ValueError(f'non-finite Gaussian attributes in {path}')
    # Reuse the official decoder; it sorts field suffixes and ignores higher SH.
    with np.errstate(over='ignore', under='ignore'):
        decoded = read_gaussian_attribute(vertex, ATTRIBUTES)
    # The upstream sign canonicalization uses sign(qw), which maps an exact
    # qw == 0 quaternion to all zeros. Preserve the valid half-turn boundary.
    rotation_names = [f'rot_{i}' for i in range(4)]
    rotation = np.column_stack([vertex[name] for name in rotation_names]).astype(np.float32)
    rotation_norm = np.linalg.norm(rotation, axis=1, keepdims=True)
    if not np.isfinite(rotation_norm).all() or (rotation_norm <= 1e-8).any():
        raise ValueError(f'invalid zero/non-finite quaternion in {path}')
    rotation /= rotation_norm
    rotation *= np.where(rotation[:, :1] < 0, -1.0, 1.0)
    decoded[:, 7:11] = rotation
    if decoded.shape != (len(vertex), 14):
        raise ValueError(f'expected N x 14 decoded attributes, got {decoded.shape}')
    result = np.ascontiguousarray(decoded, dtype=np.float32)
    if not np.isfinite(result).all() or not (result[:, 4:7] > 0).all():
        raise ValueError(f'invalid decoded Gaussian values/scales in {path}')
    return result


def normalize_gaussians(data: np.ndarray, mode: str):
    """Normalize the full asset BEFORE subsampling, matching an official profile.

    ``pretrain`` matches ShapeNetGaussian.pc_norm_gs with all norm attributes.
    ``downstream`` matches tools/extract_features.py (xyz + physical scales).
    Returned metadata can invert the geometry/scale normalization. This tensor
    is a model representation, not a renderable 3DGS PLY (scales can be negative
    in the pretraining representation).
    """
    if mode not in ('pretrain', 'downstream'):
        raise ValueError(f'unknown normalization mode: {mode}')
    result = np.array(data, dtype=np.float32, order='C', copy=True)
    if result.ndim != 2 or result.shape[1] != 14 or not len(result):
        raise ValueError(f'expected non-empty N x 14 data, got {result.shape}')
    if not np.isfinite(result).all():
        raise ValueError('non-finite input Gaussian attributes')
    center = result[:, :3].mean(axis=0)
    result[:, :3] -= center
    radius = float(np.linalg.norm(result[:, :3], axis=1).max())
    if not np.isfinite(radius) or radius <= 1e-8:
        raise ValueError(f'degenerate object radius: {radius}')
    result[:, :3] /= radius
    result[:, 4:7] /= radius
    scale_center = np.zeros(3, dtype=np.float32)
    scale_factor = 1.0
    if mode == 'pretrain':
        result[:, 3] = result[:, 3] * 2 - 1
        scale_center = result[:, 4:7].mean(axis=0)
        result[:, 4:7] -= scale_center
        scale_factor = float(np.linalg.norm(result[:, 4:7], axis=1).max())
        if not np.isfinite(scale_factor) or scale_factor <= 1e-12:
            raise ValueError(f'degenerate scale normalization: {scale_factor}')
        result[:, 4:7] /= scale_factor
        sh = np.clip(result[:, 11:14] * 0.28209479177387814, -0.5, 0.5)
        result[:, 11:14] = 2 * sh / np.sqrt(3)
    if not np.isfinite(result).all():
        raise ValueError('non-finite normalized Gaussian attributes')
    metadata = {
        'mode': mode, 'xyz_center': center.tolist(), 'xyz_radius': radius,
        'scale_center': scale_center.tolist(), 'scale_factor': scale_factor,
        'computed_from_full_asset': True,
    }
    return result, metadata


def sample_gaussians(data: np.ndarray, count: int, seed: int):
    """Uniform whole-row sampling; replacement only when the asset is too small."""
    if count < 1:
        raise ValueError('sample count must be positive')
    if data.ndim != 2 or data.shape[1] != 14 or not len(data):
        raise ValueError(f'expected non-empty N x 14 data, got {data.shape}')
    indices = np.random.RandomState(seed).choice(
        len(data), count, replace=len(data) < count
    ).astype(np.int64)
    return np.ascontiguousarray(data[indices], dtype=np.float32), indices


class PlantGaussianDataset(Dataset):
    """One PLY per item, with deterministic epoch-aware sampling metadata."""

    def __init__(self, ply_paths, num_gaussians: int = 1024, seed: int = 0,
                 normalization: str = 'pretrain'):
        self.ply_paths = [Path(p).resolve() for p in ply_paths]
        self.num_gaussians = int(num_gaussians)
        self.seed = int(seed)
        self.normalization = normalization
        self.epoch = 0
        if self.num_gaussians < 1:
            raise ValueError('num_gaussians must be positive')
        if not self.ply_paths:
            raise ValueError('at least one PLY path is required')
        self._cache = {}

    def __len__(self):
        return len(self.ply_paths)

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def __getitem__(self, index: int):
        path = self.ply_paths[index]
        if index not in self._cache:
            decoded = load_gaussian_ply(path)
            self._cache[index] = normalize_gaussians(decoded, self.normalization)
        normalized, metadata = self._cache[index]
        sampled, indices = sample_gaussians(
            normalized, self.num_gaussians, self.seed + self.epoch
        )
        import torch
        return {
            'points': torch.from_numpy(sampled),
            'indices': torch.from_numpy(indices.copy()),
            'source_ply': str(path),
            'num_source_gaussians': int(len(normalized)),
            'normalization': metadata,
        }
