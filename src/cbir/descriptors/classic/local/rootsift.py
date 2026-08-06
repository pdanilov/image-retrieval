"""RootSIFT local descriptors: SIFT keypoints, then a Hellinger-kernel transform.

SIFT (Lowe, "Distinctive Image Features from Scale-Invariant Keypoints", IJCV 2004)
produces a 128-D gradient-orientation histogram per keypoint. RootSIFT (Arandjelović
& Zisserman, "Three things everyone should know to improve object retrieval", CVPR
2012) L1-normalizes each descriptor and then takes its element-wise square root. That
one change makes L2 distance between the transformed vectors equal the Hellinger
distance between the original histograms — a much better match for histogram-like
data than raw SIFT's L2 — for a fixed cost, which is why RootSIFT rather than raw
SIFT is the actual local descriptor used downstream (`codebook/`, `aggregate/` all
consume its output).
"""

from collections.abc import Iterable

import cv2
import numpy as np
from PIL.Image import Image as PILImage

DIM = 128


class RootSIFT:
    """Extracts RootSIFT descriptors, reusing one OpenCV SIFT detector.

    The detector is built once per instance rather than once per image. It carries no
    per-image state — `detectAndCompute` takes the image as an argument — so reusing it
    is safe, and it avoids re-running `cv2.SIFT_create()` for every one of the ~11k
    images in a dataset.

    There are no configurable parameters yet (fixed `cv2.SIFT_create()` defaults). If
    any are added, `cache/rootsift.py`'s cache key must grow to include them, or a
    config change would silently reuse descriptors extracted under the old settings.
    """

    def __init__(self) -> None:
        self._sift = cv2.SIFT_create()  # type: ignore[attr-defined]  # cv2's bundled stubs lag the runtime API

    def sift_descriptors(self, image: PILImage) -> np.ndarray:
        """Raw SIFT descriptors for one image.

        Converts to grayscale internally. Returns `(n, 128)` float32; `(0, 128)` if no
        keypoints are found (`cv2` returns `None` in that case, not an empty array).
        """
        gray = np.asarray(image.convert("L"))
        _, descriptors = self._sift.detectAndCompute(gray, None)
        if descriptors is None:
            return np.zeros((0, DIM), dtype=np.float32)
        return descriptors.astype(np.float32)

    @staticmethod
    def hellinger(descriptors: np.ndarray) -> np.ndarray:
        """Hellinger-kernel transform: L1-normalize each row, then element-wise sqrt.

        SIFT descriptor entries are non-negative gradient-magnitude histogram counts, so
        the L1 norm is a plain sum (no `abs` needed). Rows with L1 norm 0 stay all-zero
        rather than becoming NaN.
        """
        l1_norms = descriptors.sum(axis=1, keepdims=True)
        l1_norms[l1_norms == 0] = 1.0
        return np.sqrt(descriptors / l1_norms).astype(np.float32)

    def extract(self, image: PILImage) -> np.ndarray:
        """RootSIFT local descriptors for one image: `sift_descriptors` then `hellinger`."""
        return self.hellinger(self.sift_descriptors(image))

    def extract_many(self, images: Iterable[PILImage]) -> list[np.ndarray]:
        """RootSIFT descriptors per image, order preserved (one `(n_i, 128)` array each).

        `images` only needs to be iterable, not a materialized sequence — pass a lazy
        generator (`data.images.iter_images`) rather than a list of decoded images, so
        only one image is decoded at a time instead of the whole dataset at once.
        """
        return [self.extract(image) for image in images]

    def pool(self, images: Iterable[PILImage]) -> np.ndarray:
        """All images' descriptors concatenated into one `(M, 128)` array.

        For fitting a codebook, which clusters over descriptors regardless of which
        image they came from — unlike `encode()`, image boundaries don't matter here.
        """
        return np.concatenate(self.extract_many(images), axis=0)
