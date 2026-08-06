import numpy as np
from PIL import Image

from cbir.data.crop import crop_query


def _gradient_image(width: int = 20, height: int = 10) -> Image.Image:
    # Each pixel's value is its x-coordinate, so cropped content is hand-verifiable.
    row = np.arange(width, dtype=np.uint8)
    return Image.fromarray(np.tile(row, (height, 1)), mode="L")


def test_crop_query_matches_hand_computed_box():
    image = _gradient_image()
    cropped = crop_query(image, (5, 2, 15, 8))
    assert cropped.size == (10, 6)
    arr = np.asarray(cropped)
    assert arr[0].tolist() == list(range(5, 15))


def test_crop_query_rounds_fractional_bbx_to_nearest_pixel():
    image = _gradient_image()
    cropped = crop_query(image, (5.4, 2.0, 14.6, 8.0))
    # round(5.4)=5, round(14.6)=15 -> same box as the hand-computed test above.
    assert cropped.size == (10, 6)


def test_crop_query_full_image_bbx_is_a_no_op():
    image = _gradient_image()
    cropped = crop_query(image, (0, 0, image.width, image.height))
    assert cropped.size == image.size
    np.testing.assert_array_equal(np.asarray(cropped), np.asarray(image))
