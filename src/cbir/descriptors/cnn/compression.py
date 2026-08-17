"""PCA compression of global descriptors, fitted on held-out images.

Part of Neural Codes as published, not an enhancement bolted on: the paper's claim is
that a 4096-D code compresses to a few hundred dimensions with little loss, and that
compact form is what makes deep features competitive with the classic tier's much
longer vectors. Reporting only the uncompressed 4096-D code would leave the actual
result unmeasured.

Fitted on the **held-out** dataset, never on the images it will be used to search —
the same rule the classic tier's vocabularies follow, and for the same reason: a
projection fitted on the evaluation database is fitted on the test set.

No whitening. Babenko's method is plain PCA compression; whitening is a later,
separately-evaluated idea (Jégou & Chum, ECCV 2012) and folding it in silently would
mean reporting one method's number under another's name.
"""

from __future__ import annotations

import numpy as np

from cbir.descriptors.classic.normalization import safe_l2_normalize


class PCACompression:
    """A fitted PCA projection, applied as `L2 -> project -> L2`."""

    def __init__(self, mean: np.ndarray, components: np.ndarray) -> None:
        self.mean = np.ascontiguousarray(mean, dtype=np.float32)
        self.components = np.ascontiguousarray(components, dtype=np.float32)  # (dim, D)

    @property
    def dim(self) -> int:
        return self.components.shape[0]

    @classmethod
    def fit(cls, descriptors: np.ndarray, dim: int, *, seed: int = 0) -> PCACompression:
        """Fit on `(n, D)` held-out descriptors, keeping `dim` components.

        `seed` is passed through to the randomized solver so a refit is reproducible;
        at 4096-D sklearn may choose it over the exact one.
        """
        if dim <= 0 or dim > descriptors.shape[1]:
            raise ValueError(f"dim must be in 1..{descriptors.shape[1]}, got {dim}")
        if dim > len(descriptors):
            raise ValueError(f"cannot fit {dim} components from {len(descriptors)} descriptors")

        from sklearn.decomposition import PCA

        model = PCA(n_components=dim, whiten=False, random_state=seed)
        model.fit(np.ascontiguousarray(descriptors, dtype=np.float32))
        return cls(mean=model.mean_, components=model.components_)

    def transform(self, descriptors: np.ndarray) -> np.ndarray:
        """Project `(n, D)` to `(n, dim)` and re-normalize.

        The trailing L2 matters: projection changes vector lengths, and `exact_search`
        reads a dot product as cosine similarity, so skipping it would score the
        ranking partly on magnitude.
        """
        x = np.ascontiguousarray(descriptors, dtype=np.float32) - self.mean
        return safe_l2_normalize(x @ self.components.T, axis=1)
