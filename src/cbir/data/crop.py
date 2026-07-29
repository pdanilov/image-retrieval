"""Crop query images to their ground-truth region.

Every method — classic and neural alike — is evaluated on queries cropped to `bbx`,
never the full image: the ground-truth `easy`/`hard`/`junk` labels were made against
the cropped region, so scoring an uncropped query against them is scoring it against
labels that don't describe it.
"""

from PIL import Image


def crop_query(
    image: Image.Image,
    bbox: tuple[int, int, int, int] | tuple[float, float, float, float],
) -> Image.Image:
    """Crop `image` to its query region.

    `bbox` is `(x1, y1, x2, y2)`, zero-based — exactly PIL's `Image.crop` box, no `+1`
    (the reference MATLAB code's `+1` is an artifact of 1-based indexing and must not
    be ported here). Rounds to the nearest pixel.
    """
    x1, y1, x2, y2 = map(round, bbox)
    return image.crop((x1, y1, x2, y2))
