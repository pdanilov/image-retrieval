"""Visual vocabulary shared by the classic aggregators (BoW, VLAD).

A vocabulary is `k` cluster centers in local-descriptor space, obtained by k-means on
a set of local descriptors (RootSIFT in this project). BoW and VLAD both encode an
image by comparing its local descriptors against these centers; they differ only in
what they record per center (a count vs. a residual).

The vocabulary MUST be trained on a held-out set, never on the evaluation database —
training the vocabulary on the images it will later be used to search is a form of
fitting on the test set. This module does not enforce that (it only sees descriptor
arrays); the caller is responsible for the split. See AGENTS.md.

Fisher vectors use a Gaussian mixture instead of hard k-means centers and so have their
own model (`fisher.py`), not this class.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Vocabulary:
    """`k` cluster centers of dimension `d`, row order defining word indices.

    Constructed either by :meth:`train` (k-means on held-out descriptors) or directly
    from known centers — the latter is what unit tests use, so the encoding math can be
    checked against hand-computed values without depending on a clustering result.
    """

    centers: np.ndarray  # (k, d) float32

    def __post_init__(self) -> None:
        if self.centers.ndim != 2:
            raise ValueError(f"centers must be 2-D (k, d), got shape {self.centers.shape}")
        if self.centers.dtype != np.float32:
            # Normalize dtype rather than reject: descriptors are float32 everywhere in
            # this codebase, and a float64 center set would silently up-cast encodings.
            object.__setattr__(self, "centers", self.centers.astype(np.float32))

    @property
    def k(self) -> int:
        return self.centers.shape[0]

    @property
    def d(self) -> int:
        return self.centers.shape[1]

    @classmethod
    def train(
        cls,
        descriptors: np.ndarray,
        k: int,
        *,
        seed: int,
        minibatch: bool = True,
        batch_size: int = 10_000,
    ) -> Vocabulary:
        """Cluster `(m, d)` held-out descriptors into `k` centers.

        `minibatch` (default) uses MiniBatchKMeans, which is the standard choice at
        SIFT scale (millions of descriptors) — full Lloyd k-means is impractical there.
        `seed` is recorded by the caller for reproducibility; identical inputs and seed
        give an identical vocabulary.
        """
        if descriptors.ndim != 2:
            raise ValueError(f"descriptors must be 2-D (m, d), got shape {descriptors.shape}")
        if k > len(descriptors):
            raise ValueError(f"cannot train {k} clusters from {len(descriptors)} descriptors")

        descriptors = np.ascontiguousarray(descriptors, dtype=np.float32)

        # Imported lazily so importing the package (e.g. for eval) doesn't pull sklearn.
        if minibatch:
            from sklearn.cluster import MiniBatchKMeans

            model = MiniBatchKMeans(n_clusters=k, random_state=seed, batch_size=batch_size, n_init="auto")
        else:
            from sklearn.cluster import KMeans

            model = KMeans(n_clusters=k, random_state=seed, n_init="auto")

        model.fit(descriptors)
        return cls(centers=np.asarray(model.cluster_centers_, dtype=np.float32))

    def assign(self, descriptors: np.ndarray) -> np.ndarray:
        """Hard-assign each of `(n, d)` descriptors to its nearest center (L2).

        Returns `(n,)` int64 word indices. Ties break to the lowest index (argmin).
        """
        return self._squared_distances(descriptors).argmin(axis=1)

    def _squared_distances(self, descriptors: np.ndarray) -> np.ndarray:
        """`(n, k)` squared Euclidean distances from each descriptor to each center.

        Uses ||x - c||^2 = ||x||^2 - 2 x·c + ||c||^2, which avoids materializing the
        `(n, k, d)` difference tensor.
        """
        if descriptors.ndim != 2:
            raise ValueError(f"descriptors must be 2-D (n, d), got shape {descriptors.shape}")
        if descriptors.shape[1] != self.d:
            raise ValueError(f"descriptor dim {descriptors.shape[1]} != vocabulary dim {self.d}")
        x = np.ascontiguousarray(descriptors, dtype=np.float32)
        x_sq = np.einsum("nd,nd->n", x, x)[:, None]
        c_sq = np.einsum("kd,kd->k", self.centers, self.centers)[None, :]
        cross = x @ self.centers.T
        # Clamp tiny negatives from float round-off; distances are only used for argmin.
        return np.maximum(x_sq - 2.0 * cross + c_sq, 0.0)
