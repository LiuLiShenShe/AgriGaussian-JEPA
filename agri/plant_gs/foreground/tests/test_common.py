import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import find_rgb_match


def test_matches_foreground_pixels_even_when_mask_filename_is_renumbered():
    rng = np.random.default_rng(5)
    rgb = rng.integers(0, 256, size=(3, 5, 6, 3), dtype=np.uint8)
    mask = np.zeros((5, 6), dtype=bool)
    mask[1:4, 1:5] = True
    crop = np.zeros((5, 6, 3), dtype=np.uint8)
    crop[mask] = rgb[2][mask]
    match = find_rgb_match(crop, mask, rgb, ['0000.jpg', '0165.jpg', '0199.jpg'])
    assert match['rgb'] == '0199.jpg'
    assert match['mae'] == 0
    assert match['exact_fraction'] == 1


def test_rejects_ambiguous_or_inexact_pairings_and_empty_masks():
    crop = np.ones((4, 4, 3), dtype=np.uint8) * 125
    mask = np.ones((4, 4), dtype=bool)
    with pytest.raises(ValueError, match='got 2'):
        find_rgb_match(crop, mask, np.stack([crop, crop]), ['a.jpg', 'b.jpg'])
    with pytest.raises(ValueError, match='got 0'):
        find_rgb_match(crop, mask, np.stack([crop + 1]), ['a.jpg'])
    with pytest.raises(ValueError, match='empty foreground'):
        find_rgb_match(crop, np.zeros((4, 4)), np.stack([crop]), ['a.jpg'])


def test_silhouette_support_ignores_out_of_frame_and_behind_camera():
    from common import silhouette_support
    K = np.array([[1., 0., 2.], [0., 1., 2.], [0., 0., 1.]])
    frames = [{'K': K, 'camtoworld': np.eye(4), 'width': 5, 'height': 5}]
    mask = np.zeros((5, 5), dtype=bool)
    mask[2, 2] = True
    points = np.array([[0, 0, 1], [1, 0, 1], [10, 0, 1], [0, 0, -1.]])
    inside, observed = silhouette_support(points, frames, [mask])
    np.testing.assert_array_equal(inside, [1, 0, 0, 0])
    np.testing.assert_array_equal(observed, [1, 1, 0, 0])
