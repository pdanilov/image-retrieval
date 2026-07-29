import numpy as np
import pytest

from cbir.descriptors.classic.gaussian_mixture import GaussianMixture


def test_single_component_responsibility_is_one():
    # One component -> every descriptor belongs to it with probability 1.
    gmm = GaussianMixture(
        weights=np.array([1.0]),
        means=np.array([[0.0]]),
        variances=np.array([[1.0]]),
    )
    resp = gmm.responsibilities(np.array([[2.0], [-5.0]], dtype=np.float32))
    assert resp.ravel() == pytest.approx([1.0, 1.0])


def test_train_recovers_two_gaussians():
    rng = np.random.default_rng(1)
    x = np.vstack([rng.normal(0.0, 0.3, size=(200, 2)), rng.normal(8.0, 0.3, size=(200, 2))]).astype(np.float32)
    gmm = GaussianMixture.train(x, k=2, seed=0)
    # Means should land near 0 and 8 (in some order).
    recovered = np.sort(gmm.means.mean(axis=1))
    assert recovered == pytest.approx([0.0, 8.0], abs=0.5)
