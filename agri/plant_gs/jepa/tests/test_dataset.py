from pathlib import Path

import numpy as np
import pytest
from plyfile import PlyData, PlyElement

from agri.plant_gs.jepa.dataset import load_gaussian_ply


def write_ply(path: Path, n: int = 128, quaternion=(1.0, 0.0, 0.0, 0.0)):
    fields = ['x', 'y', 'z', 'opacity']
    fields += [f'scale_{i}' for i in range(3)]
    fields += [f'rot_{i}' for i in range(4)]
    fields += [f'f_dc_{i}' for i in range(3)]
    vertex = np.zeros(n, dtype=[(name, 'f4') for name in fields])
    rng = np.random.RandomState(3)
    xyz = rng.normal(size=(n, 3)).astype(np.float32) + [3, -2, 5]
    for i, name in enumerate(['x', 'y', 'z']):
        vertex[name] = xyz[:, i]
    vertex['opacity'] = np.linspace(-2, 2, n)
    for i in range(3):
        vertex[f'scale_{i}'] = np.log(np.linspace(0.01, 0.08, n) * (i + 1))
        vertex[f'f_dc_{i}'] = np.linspace(-2, 2, n) * (i + 1)
    for i in range(4):
        vertex[f'rot_{i}'] = quaternion[i]
    PlyData([PlyElement.describe(vertex, 'vertex')], text=False).write(path)
    return vertex


def test_standard_ply_decodes_real_attributes_in_model_order(tmp_path):
    path = tmp_path / 'plant.ply'
    vertex = write_ply(path, quaternion=(-2.0, 0.0, 0.0, 0.0))
    decoded = load_gaussian_ply(path)
    assert decoded.shape == (128, 14)
    assert decoded.dtype == np.float32 and decoded.flags.c_contiguous
    np.testing.assert_allclose(decoded[:, 0], vertex['x'])
    np.testing.assert_allclose(decoded[:, 3], 1 / (1 + np.exp(-vertex['opacity'])))
    np.testing.assert_allclose(decoded[:, 4], np.exp(vertex['scale_0']), rtol=1e-6)
    np.testing.assert_allclose(decoded[:, 7:11], np.tile([1, 0, 0, 0], (128, 1)))
    np.testing.assert_allclose(decoded[:, 11], vertex['f_dc_0'])


def test_pretrain_normalization_matches_official_and_preserves_input(tmp_path):
    from agri.plant_gs.jepa.dataset import normalize_gaussians
    from datasets.ShapeNet55Gaussian import ShapeNetGaussian

    path = tmp_path / 'plant.ply'
    write_ply(path)
    decoded = load_gaussian_ply(path)
    before = decoded.copy()
    expected, scale_center, scale_factor = ShapeNetGaussian.pc_norm_gs(
        None, decoded.copy(), ['xyz', 'opacity', 'scale', 'rotation', 'sh']
    )
    result, metadata = normalize_gaussians(decoded, 'pretrain')
    np.testing.assert_allclose(result, expected, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(decoded, before)
    np.testing.assert_allclose(metadata['scale_center'], scale_center)
    np.testing.assert_allclose(metadata['scale_factor'], scale_factor)
    assert np.linalg.norm(result[:, :3], axis=1).max() == pytest.approx(1)
    assert np.abs(result[:, 11:14]).max() <= 1 / np.sqrt(3) + 1e-6
    assert metadata['mode'] == 'pretrain'


def test_sampling_keeps_rows_aligned_and_is_reproducible(tmp_path):
    from agri.plant_gs.jepa.dataset import sample_gaussians

    path = tmp_path / 'plant.ply'
    write_ply(path)
    data = load_gaussian_ply(path)
    sampled, indices = sample_gaussians(data, 32, seed=7)
    again, again_indices = sample_gaussians(data, 32, seed=7)
    _, different_indices = sample_gaussians(data, 32, seed=8)
    np.testing.assert_array_equal(sampled, data[indices])
    np.testing.assert_array_equal(again, sampled)
    np.testing.assert_array_equal(again_indices, indices)
    assert len(np.unique(indices)) == 32
    assert not np.array_equal(indices, different_indices)
    assert sampled.flags.c_contiguous and indices.dtype == np.int64


def test_dataset_uses_full_asset_norm_and_dataloader_returns_real_sample(tmp_path):
    import torch
    from torch.utils.data import DataLoader
    from agri.plant_gs.jepa.dataset import PlantGaussianDataset, normalize_gaussians

    path = tmp_path / 'plant.ply'
    write_ply(path)
    dataset = PlantGaussianDataset([path], num_gaussians=64, seed=12)
    assert len(dataset) == 1
    sample = dataset[0]
    normalized, metadata = normalize_gaussians(load_gaussian_ply(path), 'pretrain')
    np.testing.assert_array_equal(sample['points'].numpy(), normalized[sample['indices']])
    assert sample['num_source_gaussians'] == 128
    assert sample['source_ply'] == str(path.resolve())
    assert sample['normalization'] == metadata
    assert sample['points'].dtype == torch.float32 and sample['points'].is_contiguous()
    batch = next(iter(DataLoader(dataset, batch_size=1, num_workers=0, drop_last=False)))
    assert tuple(batch['points'].shape) == (1, 64, 14)
    dataset.set_epoch(1)
    assert not torch.equal(dataset[0]['indices'], sample['indices'])
    dataset.set_epoch(0)
    assert torch.equal(dataset[0]['indices'], sample['indices'])


def test_half_turn_quaternion_is_not_destroyed_by_sign_zero(tmp_path):
    path = tmp_path / 'half_turn.ply'
    write_ply(path, quaternion=(0.0, -2.0, 0.0, 0.0))
    decoded = load_gaussian_ply(path)
    np.testing.assert_allclose(decoded[:, 7:11], np.tile([0, -1, 0, 0], (128, 1)))
    np.testing.assert_allclose(np.linalg.norm(decoded[:, 7:11], axis=1), 1)


def test_zero_quaternion_is_rejected_instead_of_silently_fed_to_model(tmp_path):
    path = tmp_path / 'zero_rotation.ply'
    write_ply(path, quaternion=(0.0, 0.0, 0.0, 0.0))
    with pytest.raises(ValueError, match='quaternion'):
        load_gaussian_ply(path)


def test_downstream_profile_matches_official_feature_extractor(tmp_path):
    from agri.plant_gs.jepa.dataset import normalize_gaussians
    # Avoid tools/__init__.py eagerly importing the CUDA training runners.
    import importlib.util
    script = Path(__file__).resolve().parents[4] / 'tools/extract_features.py'
    spec = importlib.util.spec_from_file_location('official_feature_script', script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    official_normalize = module.normalize_gaussians

    path = tmp_path / 'plant.ply'
    write_ply(path)
    decoded = load_gaussian_ply(path)
    actual, meta = normalize_gaussians(decoded, 'downstream')
    expected = official_normalize(decoded)
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)
    np.testing.assert_array_equal(actual[:, 3], decoded[:, 3])
    np.testing.assert_array_equal(actual[:, 7:], decoded[:, 7:])
    assert meta['scale_center'] == [0, 0, 0] and meta['scale_factor'] == 1


def test_sampling_small_asset_repeats_real_rows_never_pads_fake_gaussians():
    from agri.plant_gs.jepa.dataset import sample_gaussians
    data = np.arange(8 * 14, dtype=np.float32).reshape(8, 14)
    sampled, indices = sample_gaussians(data, 32, seed=1)
    assert len(indices) == 32 and len(np.unique(indices)) < 32
    np.testing.assert_array_equal(sampled, data[indices])


@pytest.mark.parametrize('bad_field', ['x', 'opacity', 'scale_0', 'rot_1', 'f_dc_2'])
def test_nonfinite_attributes_fail_instead_of_falling_back_to_random(tmp_path, bad_field):
    path = tmp_path / 'bad.ply'
    vertex = write_ply(path)
    vertex[bad_field][0] = np.nan
    PlyData([PlyElement.describe(vertex, 'vertex')], text=False).write(path)
    with pytest.raises(ValueError, match='non-finite'):
        load_gaussian_ply(path)


def test_xyzrgb_ply_is_rejected_with_missing_attributes(tmp_path):
    path = tmp_path / 'centers.ply'
    vertex = np.zeros(100, dtype=[(n, 'f4') for n in ['x', 'y', 'z']])
    PlyData([PlyElement.describe(vertex, 'vertex')], text=False).write(path)
    with pytest.raises(ValueError, match='missing Gaussian attributes'):
        load_gaussian_ply(path)


def test_degenerate_normalization_is_rejected(tmp_path):
    from agri.plant_gs.jepa.dataset import normalize_gaussians
    path = tmp_path / 'plant.ply'
    write_ply(path)
    data = load_gaussian_ply(path)
    data[:, :3] = 0
    with pytest.raises(ValueError, match='radius'):
        normalize_gaussians(data, 'pretrain')
    data = load_gaussian_ply(path)
    data[:, 4:7] = 0.02
    with pytest.raises(ValueError, match='scale normalization'):
        normalize_gaussians(data, 'pretrain')
