"""Diagonal-covariance Gaussian Mixture Model, fit via EM on held-out descriptors.

This is Fisher vector aggregation's (`fisher.py`) "vocabulary" — a soft, probabilistic
alternative to the hard k-means clusters `Vocabulary` (`vocabulary.py`) provides to
BoW and VLAD.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np


@dataclass(frozen=True)
class GaussianMixture:
    """A diagonal-covariance GMM: `k` components of dimension `d`.

    Constructed by :meth:`train` (EM on held-out descriptors) or directly from arrays,
    the latter for unit tests. `variances` are the diagonals of the covariances,
    floored at `VAR_FLOOR` — never exactly 0, safe to divide by unconditionally.
    """

    weights: np.ndarray  # (k,)      mixture weights, sum to 1
    means: np.ndarray  # (k, d)
    variances: np.ndarray  # (k, d)  diagonal covariances (> 0)

    # GMM components can collapse to near-zero variance on a tight held-out cluster,
    # which would blow up the whitened residual. sklearn applies its own reg_covar
    # during fitting; this guards hand-constructed models too, and stays well below
    # SIFT variance scales. A ClassVar, not a dataclass field: it's a numerical safety
    # guard, not a per-instance modeling choice, so it isn't part of __init__/equality.
    VAR_FLOOR: ClassVar[float] = 1e-12

    def __post_init__(self) -> None:
        w, m, v = self.weights, self.means, self.variances
        if m.ndim != 2 or v.shape != m.shape or w.shape != (m.shape[0],):
            raise ValueError(f"shape mismatch: weights {w.shape}, means {m.shape}, variances {v.shape}")
        object.__setattr__(self, "weights", np.ascontiguousarray(w, dtype=np.float32))
        object.__setattr__(self, "means", np.ascontiguousarray(m, dtype=np.float32))
        object.__setattr__(self, "variances", np.maximum(np.ascontiguousarray(v, dtype=np.float32), self.VAR_FLOOR))

    @property
    def k(self) -> int:
        return self.means.shape[0]

    @property
    def d(self) -> int:
        return self.means.shape[1]

    @staticmethod
    def subsample(descriptors: np.ndarray, sample: int, seed: int) -> np.ndarray:
        """`sample` rows drawn without replacement, or all of them if there are fewer.

        Separate from `train` so the draw can be tested on its own, and so the caller
        can see that `seed` controls both the draw and EM — one seed reproduces the
        whole fit.
        """
        if sample <= 0 or sample >= len(descriptors):
            return descriptors
        rows = np.random.default_rng(seed).choice(len(descriptors), size=sample, replace=False)
        rows.sort()  # keep the memory access sequential; the row order carries no meaning
        return descriptors[rows]

    @classmethod
    def train(cls, descriptors: np.ndarray, k: int, *, seed: int, sample: int = 0) -> GaussianMixture:
        """Fit a `k`-component diagonal GMM to `(m, d)` held-out descriptors via EM.

        `sample` caps how many descriptors EM sees (0 = all of them). This is not an
        optimization detail to leave at a default and forget: sklearn's GaussianMixture
        is full-batch, and the held-out pools here are ~21M x 128, so each E-step
        allocates an (m, k) responsibility matrix of several GB and `init_params`
        defaults to a *full* Lloyd k-means over the same array. Fitting on a seeded
        subsample of ~1M is both tractable and what the Fisher-vector literature does.
        Whatever value is used belongs in the cache key and in the recorded run params —
        a GMM fitted on 1M descriptors is a different model from one fitted on 21M.
        """
        if descriptors.ndim != 2:
            raise ValueError(f"descriptors must be 2-D (m, d), got shape {descriptors.shape}")
        if k > len(descriptors):
            raise ValueError(f"cannot fit {k} components from {len(descriptors)} descriptors")

        from sklearn.mixture import GaussianMixture as SkGaussianMixture

        fitting = cls.subsample(descriptors, sample, seed)
        if k > len(fitting):
            raise ValueError(f"cannot fit {k} components from a {len(fitting)}-descriptor sample")

        model = SkGaussianMixture(n_components=k, covariance_type="diag", random_state=seed)
        model.fit(np.ascontiguousarray(fitting, dtype=np.float32))
        return cls(
            weights=np.asarray(model.weights_, dtype=np.float32),
            means=np.asarray(model.means_, dtype=np.float32),
            variances=np.asarray(model.covariances_, dtype=np.float32),
        )

    def log_responsibilities(self, descriptors: np.ndarray) -> np.ndarray:
        """`(n, k)` log posteriors `log p(component=k | x_i)` for `(n, d)` descriptors."""
        from scipy.special import logsumexp

        x = np.ascontiguousarray(descriptors, dtype=np.float32)
        if x.shape[1] != self.d:
            raise ValueError(f"descriptor dim {x.shape[1]} != model dim {self.d}")
        diff = x[:, None, :] - self.means[None, :, :]  # (n, k, d)
        # log N(x; mu_k, diag(var_k)) for each component. self.variances is already
        # floored (see __post_init__), so no re-flooring needed here.
        log_gauss = -0.5 * (
            self.d * np.log(2.0 * np.pi)
            + np.log(self.variances).sum(axis=1)[None, :]
            + np.einsum("nkd,kd->nk", diff**2, 1.0 / self.variances)
        )
        log_weighted = np.log(self.weights)[None, :] + log_gauss
        return log_weighted - logsumexp(log_weighted, axis=1, keepdims=True)

    def responsibilities(self, descriptors: np.ndarray) -> np.ndarray:
        """`(n, k)` posterior probabilities; rows sum to 1."""
        return np.exp(self.log_responsibilities(descriptors))
