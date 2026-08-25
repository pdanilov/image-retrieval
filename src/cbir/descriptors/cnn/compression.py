"""PCA compression of global descriptors, fitted on held-out images.

Part of Neural Codes as published, not an enhancement bolted on: the paper's claim is
that a 4096-D code compresses to a few hundred dimensions with little loss, and that
compact form is what makes deep features competitive with the classic tier's much
longer vectors. Reporting only the uncompressed 4096-D code would leave the actual
result unmeasured.

Fitted on the **held-out** dataset, never on the images it will be used to search —
the same rule the classic tier's vocabularies follow, and for the same reason: a
projection fitted on the evaluation database is fitted on the test set.

Whitening is opt-in and recorded, never implied. Babenko's Neural Codes is plain PCA
compression, so folding whitening in there would report one method's number under
another's name. The pooled conv descriptors are the opposite case: Radenović et al.
call whitening "essential" and apply it to every off-the-shelf CNN baseline they
publish, so a pooled run without it is not comparable to their table. Making it a
parameter rather than a default keeps both honest, and puts the choice in `params`.
"""

from __future__ import annotations

import numpy as np

from cbir.descriptors.classic.normalization import safe_l2_normalize


class PCACompression:
    """A fitted PCA projection, applied as `L2 -> project -> L2`."""

    def __init__(self, mean: np.ndarray, components: np.ndarray) -> None:
        self.mean = np.ascontiguousarray(mean, dtype=np.float32)
        # (dim, D). When whitening, the per-component variance scaling is folded in
        # here at fit time, so `transform` stays one matmul either way.
        self.components = np.ascontiguousarray(components, dtype=np.float32)

    @property
    def dim(self) -> int:
        return self.components.shape[0]

    @classmethod
    def fit(
        cls,
        descriptors: np.ndarray,
        dim: int,
        *,
        whiten: bool = False,
        shrinkage: float = 0.0,
        seed: int = 0,
    ) -> PCACompression:
        """Fit on `(n, D)` held-out descriptors, keeping `dim` components.

        `whiten` divides each component by its standard deviation, so every retained
        direction contributes equally to the similarity instead of the first few
        dominating.

        `shrinkage` floors that divisor at `sqrt(lambda + eps)`, with `eps` a fraction of
        the mean eigenvalue — scale-free, so one value transfers across backbones. It
        exists because whitening is unstable when the fitting set is not much larger than
        the descriptor: at 2048 components fitted on 2100 images this pipeline scores 6.7
        mAP unshrunk and 39.3 at `shrinkage=0.1`, because the smallest eigenvalues are
        estimated from almost nothing and dividing by them amplifies pure noise. Large
        directions are barely touched, so it costs little where whitening already works.

        `seed` is passed through to the randomized solver so a refit is reproducible; at
        4096-D sklearn may choose it over the exact one.
        """
        if shrinkage < 0:
            raise ValueError(f"shrinkage must be non-negative, got {shrinkage}")
        if dim <= 0 or dim > descriptors.shape[1]:
            raise ValueError(f"dim must be in 1..{descriptors.shape[1]}, got {dim}")
        if dim > len(descriptors):
            raise ValueError(f"cannot fit {dim} components from {len(descriptors)} descriptors")

        from sklearn.decomposition import PCA

        model = PCA(n_components=dim, random_state=seed)
        model.fit(np.ascontiguousarray(descriptors, dtype=np.float32))

        components = model.components_
        if whiten:
            # Guarded against a zero-variance direction, which a rank-deficient
            # held-out set can produce and which would otherwise divide by zero.
            variance = model.explained_variance_
            floor = shrinkage * variance.mean()
            deviation = np.sqrt(np.maximum(variance + floor, 1e-12))
            components = components / deviation[:, None]
        return cls(mean=model.mean_, components=components)

    def transform(self, descriptors: np.ndarray) -> np.ndarray:
        """Project `(n, D)` to `(n, dim)` and re-normalize.

        The trailing L2 matters: projection changes vector lengths, and `exact_search`
        reads a dot product as cosine similarity, so skipping it would score the
        ranking partly on magnitude.
        """
        x = np.ascontiguousarray(descriptors, dtype=np.float32) - self.mean
        return safe_l2_normalize(x @ self.components.T, axis=1)
