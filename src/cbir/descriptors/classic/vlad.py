"""VLAD aggregation: first-order residuals against the visual vocabulary.

VLAD (Jégou et al., "Aggregating local descriptors into a compact image
representation", CVPR 2010) records, per visual word, the *sum of residuals* of the
descriptors assigned to it — where they landed relative to the center, not merely how
many landed there (BoW). For `k` words of dimension `d` the image becomes a dense
`k * d` vector.

Normalization is not cosmetic here; it is most of why VLAD works, and the choices are
from Arandjelović & Zisserman, "All about VLAD" (CVPR 2013) and Jégou & Chum,
"Negative evidences and co-occurences" (ECCV 2012):

- **intra-normalization** — L2-normalize each per-word `d`-dim block independently,
  before concatenation. Stops a single bursty word (repeated texture: windows, foliage)
  from dominating the descriptor. This is the recommended default.
- **power-law / signed square root** — `f(x) = sign(x) |x|^alpha` element-wise, another
  burstiness suppressor, applied before the global normalization.
- **global L2** — final L2-normalization so similarity is a dot product.

VLAD has no corpus-level fit (unlike BoW's idf), so queries are encoded exactly like the
database — `encode` is called the same way on both.
"""

from __future__ import annotations

import numpy as np

from cbir.descriptors.classic.normalization import safe_l2_normalize
from cbir.descriptors.classic.prepare import ClassicDescriptorInputs
from cbir.descriptors.classic.vocabulary import Vocabulary
from cbir.descriptors.classic.vocabulary_cache import cached_train


def raw_vlad(vocabulary: Vocabulary, descriptors: np.ndarray) -> np.ndarray:
    """Un-normalized `(k * d,)` VLAD vector for one image's `(n, d)` descriptors.

    Block `j` (rows `j*d : (j+1)*d`) is the sum of `x - center_j` over descriptors
    hard-assigned to word `j`. An image with no descriptors yields an all-zero vector.
    """
    k, d = vocabulary.k, vocabulary.d
    vlad = np.zeros((k, d), dtype=np.float32)
    if len(descriptors) == 0:
        return vlad.reshape(-1)
    assignments = vocabulary.assign(descriptors)
    residuals = np.ascontiguousarray(descriptors, dtype=np.float32) - vocabulary.centers[assignments]
    np.add.at(vlad, assignments, residuals)
    return vlad.reshape(-1)


def normalize(
    vectors: np.ndarray,
    k: int,
    d: int,
    *,
    intra_norm: bool = True,
    power: float | None = None,
) -> np.ndarray:
    """Normalize `(N, k*d)` raw VLAD rows.

    Order: power-law (if given) -> intra-normalization (if enabled) -> global L2. Each
    step is skipped cleanly for all-zero rows, which stay all-zero rather than NaN.
    """
    out = np.array(vectors, dtype=np.float32, copy=True)

    if power is not None:
        out = np.sign(out) * np.abs(out) ** power

    if intra_norm:
        blocks = out.reshape(len(out), k, d)
        out = safe_l2_normalize(blocks, axis=2).reshape(len(out), k * d)

    return safe_l2_normalize(out, axis=1)


def encode(
    vocabulary: Vocabulary,
    images: list[np.ndarray],
    *,
    intra_norm: bool = True,
    power: float | None = None,
) -> np.ndarray:
    """Encode per-image descriptor arrays into `(N, k*d)` normalized VLAD vectors."""
    raw = np.stack([raw_vlad(vocabulary, descriptors) for descriptors in images])
    return normalize(raw, vocabulary.k, vocabulary.d, intra_norm=intra_norm, power=power)


def fit_and_encode(
    inputs: ClassicDescriptorInputs,
    k: int,
    seed: int,
    *,
    intra_norm: bool = True,
    power: float | None = None,
) -> tuple[Vocabulary, np.ndarray, np.ndarray]:
    """Train a vocabulary on `inputs.held_out_descriptors`, then encode its database/queries.

    VLAD has no corpus-level fit, so database and queries are two separate `encode()`
    calls rather than BoW's single dual-return call. Vocabulary training is cached by
    `(inputs.held_out_dataset, k, seed)` and shared with BoW (see `vocabulary_cache.py`).
    Returns `(vocabulary, database_vectors, query_vectors)`.
    """
    vocabulary = cached_train(inputs.held_out_dataset, inputs.held_out_descriptors, k=k, seed=seed)
    database_vectors = encode(vocabulary, inputs.database_descriptors, intra_norm=intra_norm, power=power)
    query_vectors = encode(vocabulary, inputs.query_descriptors, intra_norm=intra_norm, power=power)
    return vocabulary, database_vectors, query_vectors
