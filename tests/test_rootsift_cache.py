from collections.abc import Iterator

import numpy as np
from PIL import Image

import cbir.descriptors.classic.rootsift_cache as cache_module
from cbir.descriptors.classic.rootsift_cache import (
    _pack,
    _unpack,
    cached_extract_many,
    cached_pooled_descriptors,
)


def _images(count: int) -> list[Image.Image]:
    """`count` tiny images.

    The pixels are irrelevant — `extract_many` is faked in every test below, so
    nothing ever reads them. The *type* is not irrelevant: these are passed to the
    real `cached_extract_many`/`cached_pooled_descriptors`, whose contract is
    `Iterable[Image.Image]`, so placeholder strings would be a lie about what the
    cache is being handed.
    """
    return [Image.new("RGB", (1, 1)) for _ in range(count)]


class _ExplodingImages:
    """An `images` iterable that fails the test if anything consumes it.

    A cache hit must not touch `images` at all: consuming it means decoding every
    image only to throw the result away, which is the OOM this cache exists to avoid.
    """

    def __iter__(self) -> Iterator[Image.Image]:
        raise AssertionError("images must not be consumed on a cache hit")


def test_pack_unpack_round_trip():
    descriptors = [
        np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        np.zeros((0, 2), dtype=np.float32),
        np.array([[5.0, 6.0]], dtype=np.float32),
    ]
    flat, offsets = _pack(descriptors)
    assert flat.shape == (3, 2)
    assert offsets.tolist() == [0, 2, 2, 3]

    restored = _unpack(flat, offsets)
    assert len(restored) == 3
    for original, result in zip(descriptors, restored, strict=True):
        np.testing.assert_array_equal(original, result)


def test_cached_extract_many_writes_and_reuses_cache(tmp_cache, monkeypatch):

    calls = []

    def fake_extract_many(images):
        images = list(images)
        calls.append(images)
        return [np.full((1, 128), float(i), dtype=np.float32) for i, _ in enumerate(images)]

    monkeypatch.setattr(cache_module, "extract_many", fake_extract_many)

    images = _images(3)
    first = cached_extract_many("roxford5k", "database", images)
    assert len(calls) == 1
    assert (tmp_cache / "rootsift" / "roxford5k_database.npz").exists()

    second = cached_extract_many("roxford5k", "database", images)
    # Second call must not touch extract_many again -- served from disk.
    assert len(calls) == 1

    for a, b in zip(first, second, strict=True):
        np.testing.assert_array_equal(a, b)


def test_cached_extract_many_never_consumes_images_on_cache_hit(tmp_cache, monkeypatch):
    # Regression test: an earlier version had the caller materialize every decoded
    # image *before* the cache-hit check could happen, so a hit still paid to decode
    # thousands of images just to discard them -- a real OOM. `images` must stay
    # untouched on a hit.
    monkeypatch.setattr(cache_module, "extract_many", lambda images: [np.zeros((1, 128), dtype=np.float32)])

    cached_extract_many("roxford5k", "database", _images(1))  # populates the cache

    result = cached_extract_many("roxford5k", "database", _ExplodingImages())
    assert len(result) == 1


def test_cached_extract_many_keys_are_independent_per_dataset_and_role(tmp_cache, monkeypatch):

    def fake_extract_many(images):
        return [np.zeros((1, 128), dtype=np.float32) for _ in images]

    monkeypatch.setattr(cache_module, "extract_many", fake_extract_many)

    cached_extract_many("roxford5k", "database", _images(1))
    cached_extract_many("roxford5k", "query", _images(1))
    cached_extract_many("rparis6k", "database", _images(1))

    assert (tmp_cache / "rootsift" / "roxford5k_database.npz").exists()
    assert (tmp_cache / "rootsift" / "roxford5k_query.npz").exists()
    assert (tmp_cache / "rootsift" / "rparis6k_database.npz").exists()


def test_cached_pooled_descriptors_returns_flat_array_directly(tmp_cache, monkeypatch):
    monkeypatch.setattr(
        cache_module,
        "extract_many",
        lambda images: [np.full((2, 128), float(i), dtype=np.float32) for i, _ in enumerate(images)],
    )

    pooled = cached_pooled_descriptors("roxford5k", "database", _images(2))
    assert pooled.shape == (4, 128)

    # Second call must be served from disk, not by recomputing via extract_many.
    pooled_again = cached_pooled_descriptors("roxford5k", "database", _ExplodingImages())
    np.testing.assert_array_equal(pooled, pooled_again)


def test_cached_extract_many_and_cached_pooled_descriptors_share_one_cache_entry(tmp_cache, monkeypatch):
    # Both are views over the same (dataset, role) cache -- populating via one must
    # be readable via the other, since prepare.py relies on exactly this (a dataset's
    # "database" role is both its own database cache and another eval direction's
    # held-out pool).
    monkeypatch.setattr(
        cache_module,
        "extract_many",
        lambda images: [np.full((2, 128), float(i), dtype=np.float32) for i, _ in enumerate(images)],
    )

    per_image = cached_extract_many("roxford5k", "database", _images(2))
    pooled = cached_pooled_descriptors("roxford5k", "database", _ExplodingImages())

    np.testing.assert_array_equal(pooled, np.concatenate(per_image, axis=0))
