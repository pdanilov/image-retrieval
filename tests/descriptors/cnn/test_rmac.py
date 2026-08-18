import numpy as np
import pytest
import torch
from PIL import Image

from cbir.descriptors.cnn.pooling import CHANNELS
from cbir.descriptors.cnn.rmac import RMAC, regions


def _pooler(levels: int):
    """`RMAC.pool` bound to an object carrying only the attribute it reads."""
    return RMAC.pool.__get__(type("_", (), {"levels": levels})(), RMAC)


def test_level_one_squares_span_the_shorter_side():
    # The paper's level 1: squares of side min(h, w), stepped along the longer side --
    # three of them on a 2:1 map, which is what lands nearest the 40% target overlap.
    boxes = regions(10, 20, levels=1)
    assert all(size == 10 for _, _, size, _ in boxes)
    assert len(boxes) == 3


def test_a_square_map_gets_one_region_at_level_one():
    # Nothing to spread along, so level 1 is the whole map -- which is exactly MAC.
    assert regions(16, 16, levels=1) == [(0, 0, 16, 16)]


def test_regions_get_smaller_and_more_numerous_with_level():
    coarse = regions(20, 30, levels=1)
    fine = regions(20, 30, levels=3)
    assert len(fine) > len(coarse)
    assert min(size for _, _, size, _ in fine) < min(size for _, _, size, _ in coarse)


def test_every_region_fits_inside_the_map():
    # An out-of-bounds slice does not raise in torch, it silently returns a smaller
    # region -- so the grid would quietly stop being the grid.
    for top, left, box_h, box_w in regions(13, 29, levels=3):
        assert top >= 0 and top + box_h <= 13
        assert left >= 0 and left + box_w <= 29


def test_regions_cover_both_ends_of_the_map():
    # Regions must reach the far edge, or the descriptor ignores part of the frame.
    boxes = regions(20, 40, levels=2)
    assert min(left for _, left, _, _ in boxes) == 0
    assert max(left + w for _, left, _, w in boxes) == 40


def test_a_square_map_is_symmetric_in_its_two_axes():
    boxes = regions(16, 16, levels=2)
    assert sorted((top, left) for top, left, _, _ in boxes) == sorted((left, top) for top, left, _, _ in boxes)


def test_pool_returns_one_vector_per_channel():
    pooled = _pooler(3)(torch.rand(2, 7, 12, 16))
    assert pooled.shape == (2, 7)


def test_pool_sums_l2_normalized_regions_not_raw_maxima():
    # Without the per-region L2 the largest region dominates and R-MAC degenerates
    # towards MAC. Checked by scaling one corner far up: the raw sum would follow it.
    features = torch.rand(1, 4, 12, 12) + 0.1
    boosted = features.clone()
    boosted[:, :, :6, :6] *= 100

    plain = _pooler(2)(features)
    loud = _pooler(2)(boosted)
    # Normalization bounds each region's contribution, so the sum cannot grow 100x.
    assert (loud.norm() / plain.norm()).item() < 2.0


def test_a_uniform_map_pools_to_a_uniform_vector():
    features = torch.full((1, 5, 9, 9), 0.3)
    pooled = _pooler(2)(features)
    assert pooled.std().item() == pytest.approx(0.0, abs=1e-6)


def test_pool_is_finite_on_an_all_zero_map():
    # Conv maps are post-ReLU; an empty region's norm is zero and dividing by it would
    # produce NaN rather than raising.
    assert torch.isfinite(_pooler(3)(torch.zeros(1, 3, 8, 8))).all()


def test_rmac_differs_from_plain_max_pooling():
    # If the region grid were bypassed, R-MAC would silently be MAC under another name.
    # Needs *localized* activations to show it: on uniform noise every region has
    # roughly the same per-channel max, so the two agree by construction.
    features = torch.full((1, 6, 14, 18), 0.05)
    for channel in range(6):
        features[0, channel, channel, channel * 3] = 1.0

    regional = _pooler(3)(features)
    regional = regional / regional.norm()
    globally = features.amax(dim=(2, 3))
    globally = globally / globally.norm()
    assert torch.dot(regional[0], globally[0]).item() < 0.999


def test_levels_must_be_at_least_one():
    with pytest.raises(ValueError, match="levels must be at least 1"):
        RMAC("alexnet", levels=0, device="cpu")


def test_extract_produces_native_width_descriptors():
    model = RMAC("alexnet", levels=3, device="cpu")
    out = model.extract([Image.new("RGB", (300, 200), "gray"), Image.new("RGB", (131, 90), "gray")])

    assert out.shape == (2, CHANNELS["alexnet"])
    assert np.linalg.norm(out, axis=1) == pytest.approx(1.0, abs=1e-5)


def test_p_is_pinned_to_max():
    # R-MAC has no exponent; carrying one would let a record claim a knob that did
    # nothing to the numbers.
    assert RMAC("alexnet", device="cpu").p is None
