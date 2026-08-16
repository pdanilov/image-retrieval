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

The idf is a property of the database corpus, which is why it is instance state fitted
once by `train` and then applied by every `encode` call: a query must be weighted by the
*database's* idf, not by one derived from the queries themselves.
"""

from __future__ import annotations

import numpy as np

from cbir.descriptors.classic.cache.vocabulary import VocabularyCache
from cbir.descriptors.classic.codebook.vocabulary import Vocabulary
from cbir.descriptors.classic.normalization import safe_l2_normalize
from cbir.descriptors.classic.prepare import ClassicDescriptorInputs


class BagOfWords:
    """A vocabulary plus the database idf weights fitted against it."""

    def __init__(self, vocabulary: Vocabulary, idf: np.ndarray) -> None:
        self.vocabulary = vocabulary
        self.idf = np.asarray(idf, dtype=np.float32)

    @classmethod
    def train(cls, inputs: ClassicDescriptorInputs, k: int, seed: int) -> BagOfWords:
        """Train a vocabulary on the held-out pool, then fit idf on the eval database.

        Vocabulary training is cached by `(inputs.held_out_dataset, k, seed)` and shared
        with VLAD (see `cache/vocabulary.py`) -- a rerun at a `k` already trained, by BoW
        or VLAD, is loaded from disk instead of re-clustering.

        Fitting idf walks the whole database. If the database vectors are wanted too --
        the usual case -- use `train_and_encode_database`, which gets them out of that
        same walk instead of paying for a second one.
        """
        vocabulary = VocabularyCache.train(inputs.held_out_dataset, inputs.held_out_descriptors, k=k, seed=seed)
        idf = cls.inverse_document_frequency(cls._histograms(vocabulary, inputs.database_descriptors))
        return cls(vocabulary, idf)

    @classmethod
    def train_and_encode_database(
        cls, inputs: ClassicDescriptorInputs, k: int, seed: int
    ) -> tuple[BagOfWords, np.ndarray]:
        """Train, and encode the database from the very histograms idf was fitted on.

        `train(...)` followed by `encode(inputs.database_descriptors)` walks the database
        twice -- once to fit idf, once to encode -- and that walk is the dominant cost of
        BoW: one `(n_i, 128) @ (128, k)` distance matrix per image, ~2e13 FLOPs across
        roxford5k at k=5000. Both walks produce the same histograms, so this does the one
        pass and uses it for both.

        Returns `(model, database_vectors)`. Queries still go through `encode`, which is
        negligible by comparison (70 images against ~5000). The histograms stay local to
        this call rather than being kept on the model: they are `(N, k)` float32 -- ~100
        MB at k=5000, ~1 GB at k=50000 -- which is not worth holding for the lifetime of
        a model that may never be asked for them again.
        """
        vocabulary = VocabularyCache.train(inputs.held_out_dataset, inputs.held_out_descriptors, k=k, seed=seed)
        histograms = cls._histograms(vocabulary, inputs.database_descriptors)
        model = cls(vocabulary, cls.inverse_document_frequency(histograms))
        return model, model.apply_tfidf(histograms)

    @staticmethod
    def term_frequencies(assignments: np.ndarray, k: int) -> np.ndarray:
        """`(k,)` float32 raw word counts from `(n,)` hard word assignments."""
        return np.bincount(assignments, minlength=k).astype(np.float32)

    @staticmethod
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

    @staticmethod
    def _histograms(vocabulary: Vocabulary, images: list[np.ndarray]) -> np.ndarray:
        """`(N, k)` raw count histograms, against an explicit vocabulary.

        Static because `train` needs it *before* an instance exists (idf must be fitted
        to build one); `histograms` below is the instance-facing entry point.
        """
        out = np.zeros((len(images), vocabulary.k), dtype=np.float32)
        for i, descriptors in enumerate(images):
            if len(descriptors) == 0:
                continue
            out[i] = BagOfWords.term_frequencies(vocabulary.assign(descriptors), vocabulary.k)
        return out

    def histograms(self, images: list[np.ndarray]) -> np.ndarray:
        """`(N, k)` raw count histograms for a list of per-image descriptor arrays.

        `images[i]` is an `(n_i, d)` array of that image's local descriptors; `n_i` varies.
        An image with no descriptors yields an all-zero row.
        """
        return self._histograms(self.vocabulary, images)

    def apply_tfidf(self, hist: np.ndarray) -> np.ndarray:
        """Weight `(N, k)` raw histograms by this model's idf and L2-normalize each row.

        Rows that are all zero (image with no descriptors, or no surviving words) stay all
        zero rather than becoming NaN.
        """
        return safe_l2_normalize(hist * self.idf[None, :], axis=1)

    def encode(self, images: list[np.ndarray]) -> np.ndarray:
        """Encode per-image descriptor arrays into `(N, k)` L2-normalized tf-idf vectors.

        Database and queries go through this same call — both are weighted by the
        database idf fitted in `train`, so they land in the same weighted space.
        """
        return self.apply_tfidf(self.histograms(images))
