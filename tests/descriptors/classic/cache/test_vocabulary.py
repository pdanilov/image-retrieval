import numpy as np

import cbir.descriptors.classic.cache.vocabulary as cache_module
from cbir.descriptors.classic.cache.vocabulary import VocabularyCache
from cbir.descriptors.classic.prepare import ClassicDescriptorInputs


def test_cached_train_writes_and_reuses_cache(tmp_cache, monkeypatch):

    train_calls = []
    real_train = cache_module.Vocabulary.train.__func__

    def counting_train(cls, descriptors, k, *, seed, **kwargs):
        train_calls.append((k, seed))
        return real_train(cls, descriptors, k, seed=seed, **kwargs)

    monkeypatch.setattr(cache_module.Vocabulary, "train", classmethod(counting_train))

    descriptors = np.random.default_rng(0).normal(size=(20, 2)).astype(np.float32)
    first = VocabularyCache.train("dsA", descriptors, k=2, seed=0)
    assert len(train_calls) == 1
    assert (tmp_cache / "kmeans" / "dsA_k2_seed0.npz").exists()

    second = VocabularyCache.train("dsA", descriptors, k=2, seed=0)
    assert len(train_calls) == 1  # served from disk, not retrained
    np.testing.assert_array_equal(first.centers, second.centers)


def test_cached_train_keys_are_independent_per_dataset_k_seed(tmp_cache, monkeypatch):
    descriptors = np.random.default_rng(0).normal(size=(20, 2)).astype(np.float32)

    VocabularyCache.train("dsA", descriptors, k=2, seed=0)
    VocabularyCache.train("dsA", descriptors, k=3, seed=0)
    VocabularyCache.train("dsB", descriptors, k=2, seed=0)
    VocabularyCache.train("dsA", descriptors, k=2, seed=1)

    assert (tmp_cache / "kmeans" / "dsA_k2_seed0.npz").exists()
    assert (tmp_cache / "kmeans" / "dsA_k3_seed0.npz").exists()
    assert (tmp_cache / "kmeans" / "dsB_k2_seed0.npz").exists()
    assert (tmp_cache / "kmeans" / "dsA_k2_seed1.npz").exists()


def test_bow_and_vlad_share_one_vocabulary_cache_entry(tmp_cache, monkeypatch):
    # The whole point of extracting a shared cache module: a vocabulary trained once
    # by BoW at (dataset, k, seed) must be reused by VLAD at the same (dataset, k,
    # seed) rather than re-clustering the same held-out descriptors a second time.
    import cbir.descriptors.classic.aggregate.bow as bow_module
    import cbir.descriptors.classic.aggregate.vlad as vlad_module

    rng = np.random.default_rng(0)
    blobs = [rng.normal(mu, 0.1, size=(30, 2)) for mu in ([0, 0], [10, 0])]
    held_out = np.vstack(blobs).astype(np.float32)
    inputs = ClassicDescriptorInputs(
        held_out_dataset="shared",
        held_out_descriptors=held_out,
        database_descriptors=[np.array([[0.0, 0.0]], dtype=np.float32)],
        query_descriptors=[np.array([[10.0, 0.0]], dtype=np.float32)],
    )

    bow_model = bow_module.BagOfWords.train(inputs, k=2, seed=0)

    def exploding_train(cls, *args, **kwargs):
        raise AssertionError("Vocabulary.train must not run again -- VLAD should reuse BoW's cached vocabulary")

    monkeypatch.setattr(cache_module.Vocabulary, "train", classmethod(exploding_train))

    vlad_model = vlad_module.VLAD.train(inputs, k=2, seed=0)
    np.testing.assert_array_equal(bow_model.vocabulary.centers, vlad_model.vocabulary.centers)
