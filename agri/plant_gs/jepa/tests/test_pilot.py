import numpy as np
import pytest

from agri.plant_gs.jepa.pilot_data import fixed_split
from agri.plant_gs.jepa.pilot_eval import spatial_cap_keep_mask


PLANT_IDS = [f'plant_{index:02d}' for index in range(19)]


def test_fixed_split_is_deterministic_disjoint_and_15_4():
    first = fixed_split(PLANT_IDS, seed=123)
    second = fixed_split(list(reversed(PLANT_IDS)), seed=123)
    assert first == second
    assert first['train_count'] == 15
    assert first['test_count'] == 4
    assert not set(first['train_plant_ids']) & set(first['test_plant_ids'])
    assert set(first['train_plant_ids']) | set(first['test_plant_ids']) == set(PLANT_IDS)


def test_fixed_split_rejects_duplicates_or_wrong_plant_count():
    with pytest.raises(ValueError):
        fixed_split(PLANT_IDS[:-1])
    with pytest.raises(ValueError):
        fixed_split(PLANT_IDS[:-1] + [PLANT_IDS[0]])


@pytest.mark.parametrize('ratio', [0.30, 0.55, 0.70])
def test_spatial_cap_removes_exact_fraction_from_one_contiguous_side(ratio):
    xyz = np.column_stack([
        np.linspace(-1, 1, 1000),
        np.zeros(1000),
        np.zeros(1000),
    ])
    keep = spatial_cap_keep_mask(xyz, ratio, np.asarray([1, 0, 0]))
    removed = ~keep
    assert removed.sum() == round(1000 * ratio)
    assert np.all(xyz[removed, 0] >= xyz[keep, 0].max())


def test_spatial_cap_rejects_invalid_input():
    with pytest.raises(ValueError):
        spatial_cap_keep_mask(np.zeros((10, 2)), 0.3, np.ones(3))
    with pytest.raises(ValueError):
        spatial_cap_keep_mask(np.zeros((10, 3)), 1.0, np.ones(3))
