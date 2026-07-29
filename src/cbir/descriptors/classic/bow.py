"""Bag-of-Words aggregation: hard-assignment histograms with tf-idf weighting.

Each image's local descriptors are hard-assigned to visual words (nearest center in the
`Vocabulary`), counted into a `k`-length histogram, weighted by tf-idf, and
L2-normalized. Under cosine similarity these dense vectors reproduce exactly what an
inverted index would rank — the inverted index is a search-time optimization that only
pays off at corpus sizes this project treats as out of scope (revisitop1m). At
roxford5k/rparis6k scale the dense `(N, k)` matrix drops straight into `exact_search`.

Weighting follows standard BoW retrieval:
    tf   = raw word counts in the image
    idf  = log(N / df),  df = number of database images containing the word
    vec  = L2_normalize(tf * idf)
so the L2 normalization absorbs image length and similarity is a plain dot product.

The idf is a property of the database corpus and must be estimated from it once, then
applied to both database and query encodings — a query is weighted by the database's
idf, not its own.
"""

from __future__ import annotations

import numpy as np

from cbir.descriptors.classic.normalization import safe_l2_normalize
from cbir.descriptors.classic.prepare import ClassicDescriptorInputs
from cbir.descriptors.classic.vocabulary import Vocabulary
from cbir.descriptors.classic.vocabulary_cache import cached_train


def term_frequencies(assignments: np.ndarray, k: int) -> np.ndarray:
    """`(k,)` float32 raw word counts from `(n,)` hard word assignments."""
    return np.bincount(assignments, minlength=k).astype(np.float32)


def histograms(vocabulary: Vocabulary, images: list[np.ndarray]) -> np.ndarray:
    """`(N, k)` raw count histograms for a list of per-image descriptor arrays.

    `images[i]` is an `(n_i, d)` array of that image's local descriptors; `n_i` varies.
    An image with no descriptors yields an all-zero row.
    """
    out = np.zeros((len(images), vocabulary.k), dtype=np.float32)
    for i, descriptors in enumerate(images):
        if len(descriptors) == 0:
            continue
        out[i] = term_frequencies(vocabulary.assign(descriptors), vocabulary.k)
    return out


def inverse_document_frequency(hist: np.ndarray) -> np.ndarray:
    """`(k,)` idf weights `log(N / df)` from an `(N, k)` matrix of raw count histograms.

    `df` is the number of images (rows) in which a word occurs at least once. Words that
    never occur (df=0) get idf 0, contributing nothing rather than dividing by zero.
    """
    num_images = hist.shape[0]
    document_frequency = np.count_nonzero(hist > 0, axis=0)
    idf = np.zeros(hist.shape[1], dtype=np.float32)
    seen = document_frequency > 0
    idf[seen] = np.log(num_images / document_frequency[seen]).astype(np.float32)
    return idf


def apply_tfidf(hist: np.ndarray, idf: np.ndarray) -> np.ndarray:
    """Weight `(N, k)` raw histograms by `idf` and L2-normalize each row.

    Rows that are all zero (image with no descriptors, or no surviving words) stay all
    zero rather than becoming NaN.
    """
    weighted = hist * idf[None, :]
    return safe_l2_normalize(weighted, axis=1)


def encode(
    vocabulary: Vocabulary,
    database: list[np.ndarray],
    queries: list[np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Encode a database (and optional queries) into L2-normalized tf-idf vectors.

    idf is estimated from the database only and applied to both, so query and database
    vectors live in the same weighted space. Returns `(database_vectors, query_vectors)`;
    `query_vectors` is None when `queries` is None. Both are `(N, k)` float32.
    """
    database_hist = histograms(vocabulary, database)
    idf = inverse_document_frequency(database_hist)
    database_vectors = apply_tfidf(database_hist, idf)
    if queries is None:
        return database_vectors, None
    query_vectors = apply_tfidf(histograms(vocabulary, queries), idf)
    return database_vectors, query_vectors


def fit_and_encode(inputs: ClassicDescriptorInputs, k: int, seed: int) -> tuple[Vocabulary, np.ndarray, np.ndarray]:
    """Train a vocabulary on `inputs.held_out_descriptors`, then encode its database/queries.

    Vocabulary training is cached by `(inputs.held_out_dataset, k, seed)` and shared
    with VLAD (see `vocabulary_cache.py`) -- a rerun at a `k` already trained, by BoW or
    VLAD, is loaded from disk instead of re-clustering. Returns
    `(vocabulary, database_vectors, query_vectors)`.
    """
    vocabulary = cached_train(inputs.held_out_dataset, inputs.held_out_descriptors, k=k, seed=seed)
    database_vectors, query_vectors = encode(vocabulary, inputs.database_descriptors, inputs.query_descriptors)
    assert query_vectors is not None  # queries is not None above, so encode() always returns an array here
    return vocabulary, database_vectors, query_vectors
