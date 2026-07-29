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
`gaussian_mixture.py` — this module only aggregates against one.
"""

from __future__ import annotations

import numpy as np

from cbir.descriptors.classic.gaussian_mixture import GaussianMixture
from cbir.descriptors.classic.gaussian_mixture_cache import cached_train
from cbir.descriptors.classic.normalization import safe_l2_normalize
from cbir.descriptors.classic.prepare import ClassicDescriptorInputs


def fisher_vector(model: GaussianMixture, descriptors: np.ndarray) -> np.ndarray:
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


def normalize(vectors: np.ndarray, *, power: float | None = 0.5) -> np.ndarray:
    """Improved-Fisher post-processing on `(N, 2*k*d)` rows: power-law then global L2.

    `power=0.5` is the signed square root; `power=None` skips it. All-zero rows stay
    all-zero rather than becoming NaN.
    """
    out = np.array(vectors, dtype=np.float32, copy=True)
    if power is not None:
        out = np.sign(out) * np.abs(out) ** power
    return safe_l2_normalize(out, axis=1)


def encode(
    model: GaussianMixture,
    images: list[np.ndarray],
    *,
    power: float | None = 0.5,
) -> np.ndarray:
    """Encode per-image descriptor arrays into `(N, 2*k*d)` normalized Fisher vectors."""
    raw = np.stack([fisher_vector(model, descriptors) for descriptors in images])
    return normalize(raw, power=power)


def fit_and_encode(
    inputs: ClassicDescriptorInputs,
    k: int,
    seed: int,
    *,
    power: float | None = 0.5,
) -> tuple[GaussianMixture, np.ndarray, np.ndarray]:
    """Fit a GMM on `inputs.held_out_descriptors`, then encode its database/queries.

    Like VLAD, Fisher has no corpus-level fit, so database and queries are two
    separate `encode()` calls. GMM fitting is cached by `(inputs.held_out_dataset, k,
    seed)` (see `gaussian_mixture_cache.py`). Returns
    `(model, database_vectors, query_vectors)`.
    """
    model = cached_train(inputs.held_out_dataset, inputs.held_out_descriptors, k=k, seed=seed)
    database_vectors = encode(model, inputs.database_descriptors, power=power)
    query_vectors = encode(model, inputs.query_descriptors, power=power)
    return model, database_vectors, query_vectors
