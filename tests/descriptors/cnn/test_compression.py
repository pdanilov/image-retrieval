import numpy as np
import pytest

from cbir.descriptors.cnn.compression import PCACompression


def _descriptors(n: int = 200, d: int = 32, rank: int = 5, seed: int = 0) -> np.ndarray:
    """`n` vectors that genuinely live on a `rank`-dimensional subspace."""
    rng = np.random.default_rng(seed)
    basis = rng.normal(size=(rank, d)).astype(np.float32)
    return (rng.normal(size=(n, rank)).astype(np.float32) @ basis).astype(np.float32)


def test_transform_compresses_to_the_requested_width():
    pca = PCACompression.fit(_descriptors(), dim=8)
    out = pca.transform(_descriptors(n=10))

    assert pca.dim == 8
    assert out.shape == (10, 8)
    assert out.dtype == np.float32


def test_output_is_l2_normalized():
    # Projection changes vector lengths, and exact_search reads a dot product as cosine
    # similarity -- without the trailing L2 the ranking is scored partly on magnitude.
    out = PCACompression.fit(_descriptors(), dim=8).transform(_descriptors(n=10))
    assert np.linalg.norm(out, axis=1) == pytest.approx(1.0, abs=1e-5)


def test_compression_preserves_relative_similarity():
    # The point of the method: a 4096-D code survives compression. If neighbours were
    # scrambled, mAP would drop while every shape check above still passed.
    data = _descriptors(n=60)
    pca = PCACompression.fit(data, dim=5)  # data is genuinely rank-5

    def normalize(x):
        return x / np.linalg.norm(x, axis=1, keepdims=True)

    before = normalize(data) @ normalize(data).T
    after = pca.transform(data) @ pca.transform(data).T
    # Rank ordering of each row's neighbours, not the exact values.
    assert np.corrcoef(before.ravel(), after.ravel())[0, 1] > 0.95


def test_fit_is_reproducible_for_a_seed():
    data = _descriptors()
    a = PCACompression.fit(data, dim=8, seed=0)
    b = PCACompression.fit(data, dim=8, seed=0)
    np.testing.assert_allclose(a.transform(data), b.transform(data), atol=1e-5)


def test_transform_centres_by_the_fitted_mean_not_the_input():
    # Centring on the input would make the projection depend on which images happen to
    # be transformed together -- queries and database would land in different spaces.
    data = _descriptors()
    pca = PCACompression.fit(data, dim=4)

    one = pca.transform(data[:1])
    within_batch = pca.transform(data)[:1]
    np.testing.assert_allclose(one, within_batch, atol=1e-5)


@pytest.mark.parametrize("dim", [0, -1, 33])
def test_impossible_widths_are_rejected(dim):
    with pytest.raises(ValueError, match="dim must be in"):
        PCACompression.fit(_descriptors(d=32), dim=dim)


def test_more_components_than_samples_is_rejected():
    # sklearn would raise something less specific; this says which of the two numbers
    # is the problem.
    with pytest.raises(ValueError, match="cannot fit"):
        PCACompression.fit(_descriptors(n=4, d=32), dim=10)


def test_whitening_equalizes_component_variance():
    # The point of whitening: without it the first few directions dominate every
    # similarity. With it, each retained direction contributes equally.
    rng = np.random.default_rng(0)
    scales = np.array([50.0, 5.0, 0.5, 0.05], dtype=np.float32)
    data = rng.normal(size=(500, 4)).astype(np.float32) * scales

    plain = PCACompression.fit(data, dim=4, whiten=False)
    white = PCACompression.fit(data, dim=4, whiten=True)

    # Compare before the trailing L2, which would hide the effect.
    projected_plain = (data - plain.mean) @ plain.components.T
    projected_white = (data - white.mean) @ white.components.T

    assert projected_plain.std(axis=0).max() / projected_plain.std(axis=0).min() > 100
    assert projected_white.std(axis=0).max() / projected_white.std(axis=0).min() == pytest.approx(1.0, abs=0.1)


def test_whitening_is_off_by_default():
    # Never implied: Neural Codes is plain compression, and a silently whitened run
    # would report one method's number under another's name.
    data = _descriptors()
    default = PCACompression.fit(data, dim=8)
    explicit = PCACompression.fit(data, dim=8, whiten=False)
    np.testing.assert_allclose(default.transform(data), explicit.transform(data))


def test_whitening_survives_a_zero_variance_direction():
    # A rank-deficient held-out set yields components with no variance; dividing by
    # their standard deviation would produce NaN rather than raising.
    data = _descriptors(n=100, d=16, rank=3)
    out = PCACompression.fit(data, dim=8, whiten=True).transform(data)
    assert np.isfinite(out).all()


def _near_singular(n: int = 40, d: int = 32, seed: int = 0) -> np.ndarray:
    """`n` vectors barely outnumbering `d` — the regime whitening breaks in."""
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, d)).astype(np.float32)


def test_shrinkage_tames_the_smallest_directions():
    # The failure it exists for: with n close to d the trailing eigenvalues are noise,
    # and dividing by their square roots amplifies that noise enormously.
    data = _near_singular()
    plain = PCACompression.fit(data, dim=32, whiten=True)
    shrunk = PCACompression.fit(data, dim=32, whiten=True, shrinkage=0.1)

    assert np.abs(shrunk.components).max() < np.abs(plain.components).max()


def test_shrinkage_barely_touches_the_leading_directions():
    # It must not be a blunt instrument: the large eigenvalues dwarf eps, so the
    # directions that carry the signal are left almost exactly as they were.
    data = _descriptors(n=500, d=32, rank=32)
    plain = PCACompression.fit(data, dim=4, whiten=True)
    shrunk = PCACompression.fit(data, dim=4, whiten=True, shrinkage=1e-4)

    ratio = np.linalg.norm(shrunk.components, axis=1) / np.linalg.norm(plain.components, axis=1)
    assert ratio.min() > 0.99


def test_shrinkage_defaults_to_off():
    data = _descriptors()
    default = PCACompression.fit(data, dim=8, whiten=True)
    explicit = PCACompression.fit(data, dim=8, whiten=True, shrinkage=0.0)
    np.testing.assert_allclose(default.transform(data), explicit.transform(data))


def test_shrinkage_does_nothing_without_whitening():
    # It only ever enters the whitening divisor, so a plain PCA must be unaffected --
    # otherwise the parameter would silently change runs that never asked for it.
    data = _descriptors()
    plain = PCACompression.fit(data, dim=8, shrinkage=0.5)
    none = PCACompression.fit(data, dim=8)
    np.testing.assert_allclose(plain.transform(data), none.transform(data))


def test_negative_shrinkage_is_rejected():
    # A negative eps would subtract from the eigenvalues and can flip one negative,
    # which yields NaN through the square root rather than an error.
    with pytest.raises(ValueError, match="shrinkage must be non-negative"):
        PCACompression.fit(_descriptors(), dim=8, whiten=True, shrinkage=-0.1)
