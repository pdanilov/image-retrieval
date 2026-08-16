import numpy as np
import pytest
from PIL import Image

from cbir.descriptors.classic.local.rootsift import RootSIFT

# One extractor shared across tests, which is also how it's meant to be used: the
# cv2.SIFT detector is built once and reused, not rebuilt per image.
ROOTSIFT = RootSIFT()


def _checkerboard(size: int = 64, square: int = 8) -> Image.Image:
    xs, ys = np.meshgrid(np.arange(size), np.arange(size))
    pattern = (((xs // square) + (ys // square)) % 2) * 255
    return Image.fromarray(pattern.astype(np.uint8), mode="L")


def _flat_image(size: int = 64) -> Image.Image:
    return Image.fromarray(np.full((size, size), 128, dtype=np.uint8), mode="L")


def test_hellinger_hand_computed():
    # L1 norm of [3,4] is 7 -> [3/7, 4/7] -> sqrt.
    row = np.array([[3.0, 4.0]], dtype=np.float32)
    result = RootSIFT.hellinger(row)
    expected = np.sqrt(np.array([[3.0 / 7.0, 4.0 / 7.0]]))
    assert result == pytest.approx(expected, abs=1e-6)


def test_hellinger_zero_row_stays_zero():
    row = np.zeros((1, 128), dtype=np.float32)
    assert np.all(RootSIFT.hellinger(row) == 0.0)


def test_hellinger_output_is_l2_unit_norm_for_nonzero_rows():
    rng = np.random.default_rng(0)
    rows = rng.uniform(0, 10, size=(5, 128)).astype(np.float32)
    norms = np.linalg.norm(RootSIFT.hellinger(rows), axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-5)


def test_hellinger_is_callable_without_an_instance():
    # It's a staticmethod: pure math on a descriptor array, needing no detector.
    assert RootSIFT.hellinger(np.ones((1, 4), dtype=np.float32)).shape == (1, 4)


def test_sift_descriptors_finds_keypoints_on_structured_image():
    descriptors = ROOTSIFT.sift_descriptors(_checkerboard())
    assert descriptors.ndim == 2
    assert descriptors.shape[1] == 128
    assert descriptors.shape[0] > 0
    assert descriptors.dtype == np.float32


def test_sift_descriptors_empty_on_flat_image():
    assert ROOTSIFT.sift_descriptors(_flat_image()).shape == (0, 128)


def test_extract_composes_sift_and_hellinger():
    descriptors = ROOTSIFT.extract(_checkerboard())
    assert descriptors.shape[0] > 0
    assert descriptors.shape[1] == 128
    norms = np.linalg.norm(descriptors, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-5)


def test_extract_empty_on_flat_image():
    assert ROOTSIFT.extract(_flat_image()).shape == (0, 128)


def test_extract_many_preserves_order_and_length():
    images = [_checkerboard(), _flat_image(), _checkerboard()]
    results = ROOTSIFT.extract_many(images)
    assert len(results) == 3
    assert results[1].shape == (0, 128)
    for image, result in zip([images[0], images[2]], [results[0], results[2]], strict=True):
        np.testing.assert_array_equal(result, ROOTSIFT.extract(image))


def test_reusing_one_extractor_gives_identical_results():
    # The detector is instance state now; extracting twice through the same instance,
    # or through two instances, must not drift.
    image = _checkerboard()
    np.testing.assert_array_equal(ROOTSIFT.extract(image), ROOTSIFT.extract(image))
    np.testing.assert_array_equal(ROOTSIFT.extract(image), RootSIFT().extract(image))


def test_pool_concatenates_all_images():
    images = [_checkerboard(), _flat_image(), _checkerboard()]
    pooled = ROOTSIFT.pool(images)
    per_image = ROOTSIFT.extract_many(images)
    assert pooled.shape[0] == sum(len(d) for d in per_image)
    assert pooled.shape[1] == 128
    np.testing.assert_array_equal(pooled, np.concatenate(per_image, axis=0))
