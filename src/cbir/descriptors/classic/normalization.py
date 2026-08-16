"""Shared L2-normalization helper for the classic aggregators.

BoW's tf-idf weighting, VLAD's intra/global normalization, and Fisher's post-processing
each L2-normalize rows (or, for VLAD's intra-normalization, per-word blocks) and must
treat an all-zero input as a legitimate case — an image with no local descriptors, or a
word/component nothing was assigned to — rather than let it divide into NaN.
"""

import numpy as np


def safe_l2_normalize(x: np.ndarray, axis: int) -> np.ndarray:
    """L2-normalize `x` along `axis`; entries with zero norm stay zero, not NaN."""
    norms = np.linalg.norm(x, axis=axis, keepdims=True)
    norms[norms == 0] = 1.0
    return (x / norms).astype(np.float32)
