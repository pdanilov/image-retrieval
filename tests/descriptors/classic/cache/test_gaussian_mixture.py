import numpy as np

import cbir.descriptors.classic.cache.gaussian_mixture as cache_module
from cbir.descriptors.classic.cache.gaussian_mixture import GaussianMixtureCache


def test_cached_train_writes_and_reuses_cache(tmp_cache, monkeypatch):

    train_calls = []
    real_train = cache_module.GaussianMixture.train.__func__

    def counting_train(cls, descriptors, k, *, seed, **kwargs):
        train_calls.append((k, seed))
        return real_train(cls, descriptors, k, seed=seed, **kwargs)

    monkeypatch.setattr(cache_module.GaussianMixture, "train", classmethod(counting_train))

    rng = np.random.default_rng(0)
    descriptors = np.vstack([rng.normal(0.0, 0.3, size=(30, 2)), rng.normal(8.0, 0.3, size=(30, 2))]).astype(np.float32)
    first = GaussianMixtureCache.train("dsA", descriptors, k=2, seed=0)
    assert len(train_calls) == 1
    assert (tmp_cache / "gmm" / "dsA_k2_seed0.npz").exists()

    second = GaussianMixtureCache.train("dsA", descriptors, k=2, seed=0)
    assert len(train_calls) == 1  # served from disk, not refit
    np.testing.assert_array_equal(first.weights, second.weights)
    np.testing.assert_array_equal(first.means, second.means)
    np.testing.assert_array_equal(first.variances, second.variances)


def test_cached_train_keys_are_independent_per_dataset_k_seed(tmp_cache, monkeypatch):
    rng = np.random.default_rng(0)
    descriptors = np.vstack([rng.normal(0.0, 0.3, size=(30, 2)), rng.normal(8.0, 0.3, size=(30, 2))]).astype(np.float32)

    GaussianMixtureCache.train("dsA", descriptors, k=2, seed=0)
    GaussianMixtureCache.train("dsA", descriptors, k=3, seed=0)
    GaussianMixtureCache.train("dsB", descriptors, k=2, seed=0)
    GaussianMixtureCache.train("dsA", descriptors, k=2, seed=1)

    assert (tmp_cache / "gmm" / "dsA_k2_seed0.npz").exists()
    assert (tmp_cache / "gmm" / "dsA_k3_seed0.npz").exists()
    assert (tmp_cache / "gmm" / "dsB_k2_seed0.npz").exists()
    assert (tmp_cache / "gmm" / "dsA_k2_seed1.npz").exists()
