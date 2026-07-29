import numpy as np
import pytest
from PIL import Image

from cbir.descriptors.classic.rootsift import extract, extract_many, pool_descriptors, root_sift, sift_descriptors


def _checkerboard(size: int = 64, square: int = 8) -> Image.Image:
    xs, ys = np.meshgrid(np.arange(size), np.arange(size))
    pattern = (((xs // square) + (ys // square)) % 2) * 255
    return Image.fromarray(pattern.astype(np.uint8), mode="L")


def _flat_image(size: int = 64) -> Image.Image:
    return Image.fromarray(np.full((size, size), 128, dtype=np.uint8), mode="L")


def test_root_sift_hand_computed():
    # L1 norm of [3,4] is 7 -> [3/7, 4/7] -> sqrt.
    row = np.array([[3.0, 4.0]], dtype=np.float32)
    result = root_sift(row)
    expected = np.sqrt(np.array([[3.0 / 7.0, 4.0 / 7.0]]))
    assert result == pytest.approx(expected, abs=1e-6)


def test_root_sift_zero_row_stays_zero():
    row = np.zeros((1, 128), dtype=np.float32)
    result = root_sift(row)
    assert np.all(result == 0.0)


def test_root_sift_output_is_l2_unit_norm_for_nonzero_rows():
    rng = np.random.default_rng(0)
    rows = rng.uniform(0, 10, size=(5, 128)).astype(np.float32)
    result = root_sift(rows)
    norms = np.linalg.norm(result, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-5)


def test_sift_descriptors_finds_keypoints_on_structured_image():
    descriptors = sift_descriptors(_checkerboard())
    assert descriptors.ndim == 2
    assert descriptors.shape[1] == 128
    assert descriptors.shape[0] > 0
    assert descriptors.dtype == np.float32


def test_sift_descriptors_empty_on_flat_image():
    descriptors = sift_descriptors(_flat_image())
    assert descriptors.shape == (0, 128)


def test_extract_composes_sift_and_root_sift():
    descriptors = extract(_checkerboard())
    assert descriptors.shape[0] > 0
    assert descriptors.shape[1] == 128
    norms = np.linalg.norm(descriptors, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-5)


def test_extract_empty_on_flat_image():
    descriptors = extract(_flat_image())
    assert descriptors.shape == (0, 128)


def test_extract_many_preserves_order_and_length():
    images = [_checkerboard(), _flat_image(), _checkerboard()]
    results = extract_many(images)
    assert len(results) == 3
    assert results[1].shape == (0, 128)
    for image, result in zip([images[0], images[2]], [results[0], results[2]], strict=True):
        np.testing.assert_array_equal(result, extract(image))


def test_pool_descriptors_concatenates_all_images():
    images = [_checkerboard(), _flat_image(), _checkerboard()]
    pooled = pool_descriptors(images)
    per_image = extract_many(images)
    assert pooled.shape[0] == sum(len(d) for d in per_image)
    assert pooled.shape[1] == 128
    np.testing.assert_array_equal(pooled, np.concatenate(per_image, axis=0))
