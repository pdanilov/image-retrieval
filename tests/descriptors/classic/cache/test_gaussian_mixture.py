import numpy as np

import cbir.descriptors.classic.cache.gaussian_mixture as cache_module
from cbir.descriptors.classic.cache.gaussian_mixture import GaussianMixtureCache
from cbir.descriptors.classic.codebook.gaussian_mixture import GaussianMixture


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
    assert (tmp_cache / "gmm" / "dsA_k2_seed0_n0.npz").exists()

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

    assert (tmp_cache / "gmm" / "dsA_k2_seed0_n0.npz").exists()
    assert (tmp_cache / "gmm" / "dsA_k3_seed0_n0.npz").exists()
    assert (tmp_cache / "gmm" / "dsB_k2_seed0_n0.npz").exists()
    assert (tmp_cache / "gmm" / "dsA_k2_seed1_n0.npz").exists()


def test_sample_is_part_of_the_cache_key(tmp_cache):
    # A GMM fitted on 1000 descriptors is a different model from one fitted on all of
    # them. If `sample` were not in the key, the first fit would be served for the
    # second and a sweep over it would silently plot one model under several labels.
    rng = np.random.default_rng(0)
    descriptors = np.vstack([rng.normal(0.0, 0.3, size=(60, 2)), rng.normal(8.0, 0.3, size=(60, 2))]).astype(np.float32)

    full = GaussianMixtureCache.train("dsA", descriptors, k=2, seed=0, sample=0)
    partial = GaussianMixtureCache.train("dsA", descriptors, k=2, seed=0, sample=40)

    assert (tmp_cache / "gmm" / "dsA_k2_seed0_n0.npz").exists()
    assert (tmp_cache / "gmm" / "dsA_k2_seed0_n40.npz").exists()
    # Different blobs, and genuinely different fits -- not the same model written twice.
    assert not np.array_equal(full.means, partial.means)


def test_subsample_is_seeded_and_capped():
    descriptors = np.arange(200, dtype=np.float32).reshape(100, 2)

    a = GaussianMixture.subsample(descriptors, sample=10, seed=0)
    b = GaussianMixture.subsample(descriptors, sample=10, seed=0)
    c = GaussianMixture.subsample(descriptors, sample=10, seed=1)

    assert a.shape == (10, 2)
    np.testing.assert_array_equal(a, b)  # same seed -> same draw, so a fit is reproducible
    assert not np.array_equal(a, c)
    # Rows must be distinct: sampling with replacement would quietly reweight the fit.
    assert len(np.unique(a, axis=0)) == 10


def test_subsample_passes_everything_through_when_not_capping():
    descriptors = np.arange(20, dtype=np.float32).reshape(10, 2)

    # 0 means "no cap", and a sample larger than the pool is not an error -- both must
    # return the pool itself rather than an error or a resampled copy.
    np.testing.assert_array_equal(GaussianMixture.subsample(descriptors, sample=0, seed=0), descriptors)
    np.testing.assert_array_equal(GaussianMixture.subsample(descriptors, sample=999, seed=0), descriptors)
