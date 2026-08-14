"""Fisher vector aggregation: first- and second-order statistics under a GMM.

The Fisher vector (Perronnin & Dance, "Fisher kernels on visual vocabularies", CVPR
2007; Perronnin et al., "Improving the Fisher kernel for large-scale image
classification", ECCV 2010) generalizes VLAD. It replaces hard k-means assignment with
a Gaussian mixture and its soft posteriors, and records two statistics per component:
the gradient of the data log-likelihood with respect to the component means (first
order, VLAD-like) and with respect to the component variances (second order — whether
descriptors near a component are more or less spread out than the model expects).

With `k` diagonal-covariance components of dimension `d`, define per descriptor and
component the whitened residual `u = (x - mu_k) / sigma_k`. Then

    G_mu_k    = 1 / (N * sqrt(w_k))     * sum_i  gamma_ik * u_ik
    G_sigma_k = 1 / (N * sqrt(2 * w_k)) * sum_i  gamma_ik * (u_ik^2 - 1)

where `gamma_ik` is the posterior of component `k` for descriptor `i` and `w_k` its
mixture weight. The Fisher vector is `[G_mu (k*d) , G_sigma (k*d)]`, dimension `2*k*d`
— twice VLAD's, half of it the second-order information VLAD discards. VLAD is
essentially the first-order block with hard assignment and unit variances.

Post-processing follows the "improved Fisher vector": element-wise signed square root
(power-law, `alpha=0.5`) then global L2, both burstiness suppressors. Like VLAD, Fisher
has no corpus-level fit, so queries are encoded identically to the database.

The GMM itself (`GaussianMixture`, fit via EM on held-out descriptors) lives in
`codebook/gaussian_mixture.py` — this module only aggregates against one.
"""

from __future__ import annotations

import numpy as np

from cbir.descriptors.classic.cache.gaussian_mixture import GaussianMixtureCache
from cbir.descriptors.classic.codebook.gaussian_mixture import GaussianMixture
from cbir.descriptors.classic.normalization import safe_l2_normalize
from cbir.descriptors.classic.prepare import ClassicDescriptorInputs


class FisherVector:
    """A GMM plus the power-law exponent applied to every encoding."""

    def __init__(self, model: GaussianMixture, *, power: float | None = 0.5) -> None:
        self.model = model
        self.power = power

    @classmethod
    def train(
        cls,
        inputs: ClassicDescriptorInputs,
        k: int,
        seed: int,
        *,
        power: float | None = 0.5,
        sample: int = 0,
    ) -> FisherVector:
        """Fit a GMM on the held-out pool. Fisher has no corpus-level fit beyond it.

        GMM fitting is cached by `(inputs.held_out_dataset, k, seed, sample)` — see
        `cache/gaussian_mixture.py`. `sample` caps how much of the held-out pool EM
        sees; at these pool sizes leaving it at 0 means hours of full-batch EM.
        """
        model = GaussianMixtureCache.train(
            inputs.held_out_dataset, inputs.held_out_descriptors, k=k, seed=seed, sample=sample
        )
        return cls(model, power=power)

    @staticmethod
    def raw(model: GaussianMixture, descriptors: np.ndarray) -> np.ndarray:
        """Un-normalized `(2*k*d,)` Fisher vector for one image's `(n, d)` descriptors.

        Layout is `[G_mu (k*d) , G_sigma (k*d)]`, each block row-major over components. An
        image with no descriptors yields an all-zero vector.
        """
        k, d = model.k, model.d
        if len(descriptors) == 0:
            return np.zeros(2 * k * d, dtype=np.float32)

        x = np.ascontiguousarray(descriptors, dtype=np.float32)
        n = len(x)
        sigma = np.sqrt(model.variances)  # (k, d), already floored (see GaussianMixture.__post_init__)
        resp = model.responsibilities(x)  # (n, k)
        u = (x[:, None, :] - model.means[None, :, :]) / sigma[None, :, :]  # (n, k, d) whitened

        # Weighted sums over descriptors, then per-component Fisher scaling.
        g_mu = np.einsum("nk,nkd->kd", resp, u) / (n * np.sqrt(model.weights)[:, None])
        g_sigma = np.einsum("nk,nkd->kd", resp, u**2 - 1.0) / (n * np.sqrt(2.0 * model.weights)[:, None])
        return np.concatenate([g_mu.reshape(-1), g_sigma.reshape(-1)]).astype(np.float32)

    def normalize(self, vectors: np.ndarray) -> np.ndarray:
        """Improved-Fisher post-processing on `(N, 2*k*d)` rows: power-law then global L2.

        `power=0.5` is the signed square root; `power=None` skips it. All-zero rows stay
        all-zero rather than becoming NaN.
        """
        out = np.array(vectors, dtype=np.float32, copy=True)
        if self.power is not None:
            out = np.sign(out) * np.abs(out) ** self.power
        return safe_l2_normalize(out, axis=1)

    def encode(self, images: list[np.ndarray]) -> np.ndarray:
        """Encode per-image descriptor arrays into `(N, 2*k*d)` normalized Fisher vectors."""
        raw = np.stack([self.raw(self.model, descriptors) for descriptors in images])
        return self.normalize(raw)
