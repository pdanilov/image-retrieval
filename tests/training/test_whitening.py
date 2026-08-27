"""Supervised whitening: the algebra it must satisfy, and the reference it must match."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from scipy.linalg import cholesky

from cbir.descriptors.cnn.compression import PCACompression
from cbir.descriptors.cnn.finetuned import load
from cbir.training.tuples import Corpus
from cbir.training.whitening import attach, learn, sample_pairs


def _reference(x, qidxs, pidxs):
    """cirtorch's `whitenlearn`, transcribed column-major and otherwise untouched.

    Kept verbatim — `np.linalg.eig` and all — so that our row-major port is checked
    against the code that produced the published `Lw`, not against a tidied-up idea of
    it. `x` is `(D, N)` here, the opposite of ours.
    """
    m = x[:, qidxs].mean(axis=1, keepdims=True)
    df = x[:, qidxs] - x[:, pidxs]
    S = np.dot(df, df.T) / df.shape[1]  # noqa: N806 - verbatim from the reference
    P = np.linalg.inv(cholesky(S, lower=True))  # noqa: N806 - verbatim from the reference
    df = np.dot(P, x - m)
    D = np.dot(df, df.T)  # noqa: N806 - verbatim from the reference
    eigval, eigvec = np.linalg.eig(D)
    order = eigval.argsort()[::-1]
    eigvec = eigvec[:, order]
    return {"m": m, "P": np.dot(eigvec.T, P)}


@pytest.fixture
def pairs():
    """Descriptors with a real nuisance direction: dimension 0 varies within a landmark.

    Built so the answer is known in advance — a whitening that works must suppress
    dimension 0 relative to the others.
    """
    rng = np.random.default_rng(0)
    dim, clusters, per = 6, 400, 2
    centres = rng.normal(size=(clusters, dim))
    x, qidxs, pidxs = [], [], []
    for c in range(clusters):
        base = len(x)
        for _ in range(per):
            noise = rng.normal(scale=0.05, size=dim)
            noise[0] = rng.normal(scale=1.5)  # the nuisance dimension
            x.append(centres[c] + noise)
        qidxs.append(base)
        pidxs.append(base + 1)
    return np.array(x), qidxs, pidxs


def test_it_matches_the_reference_implementation(pairs):
    x, qidxs, pidxs = pairs
    projection, mean = learn(x, qidxs, pidxs)
    expected = _reference(x.T, qidxs, pidxs)

    assert np.allclose(mean.ravel(), expected["m"].ravel(), atol=1e-5)
    # An eigenvector's sign is arbitrary and the two solvers do not agree on it, so rows
    # are compared up to sign. `P.T @ P` below pins down everything sign cannot affect.
    for ours, theirs in zip(projection, expected["P"], strict=True):
        assert np.allclose(ours, theirs, atol=1e-4) or np.allclose(ours, -theirs, atol=1e-4)
    assert np.allclose(projection.T @ projection, expected["P"].T @ expected["P"], atol=1e-4)


def test_it_whitens_the_intra_class_covariance_to_the_identity(pairs):
    """The defining property: after projection, pair differences have unit covariance."""
    x, qidxs, pidxs = pairs
    projection, _ = learn(x, qidxs, pidxs)

    difference = x[qidxs] - x[pidxs]
    intra = difference.T @ difference / len(difference)

    assert np.allclose(projection @ intra @ projection.T, np.eye(x.shape[1]), atol=1e-4)


def test_it_suppresses_the_direction_that_varies_within_a_landmark(pairs):
    """Dimension 0 is nuisance by construction, so it must lose weight."""
    x, qidxs, pidxs = pairs
    projection, mean = learn(x, qidxs, pidxs)
    whitened = PCACompression(mean=mean.ravel(), components=projection)

    def separation(vectors):
        matched = np.linalg.norm(vectors[qidxs] - vectors[pidxs], axis=1).mean()
        shuffled = np.roll(pidxs, 1)  # pair each query with someone else's positive
        other = np.linalg.norm(vectors[qidxs] - vectors[shuffled], axis=1).mean()
        return other / matched

    raw = x / np.linalg.norm(x, axis=1, keepdims=True)
    assert separation(whitened.transform(x)) > separation(raw)


def test_the_output_loads_straight_into_the_eval_path(pairs):
    """`(D, D)` rows and a `(D, 1)` mean are what `finetuned.whitening` unpacks."""
    x, qidxs, pidxs = pairs
    projection, mean = learn(x, qidxs, pidxs)

    assert projection.shape == (x.shape[1], x.shape[1])
    assert mean.shape == (x.shape[1], 1)
    # Truncation is dropping trailing rows, which is only valid because they are ordered.
    compression = PCACompression(mean=mean.reshape(-1), components=projection[:3])
    assert compression.transform(x).shape == (len(x), 3)


def test_too_few_pairs_is_refused_rather_than_silently_singular(pairs):
    x, qidxs, pidxs = pairs
    with pytest.raises(ValueError, match="cannot determine"):
        learn(x, qidxs[:4], pidxs[:4])


def test_pairs_must_pair_up(pairs):
    x, qidxs, pidxs = pairs
    with pytest.raises(ValueError, match="must pair up"):
        learn(x, qidxs, pidxs[:-1])


def test_no_pairs_at_all_is_refused(pairs):
    x, _, _ = pairs
    with pytest.raises(ValueError, match="no matching pairs"):
        learn(x, [], [])


def test_sampling_pairs_remaps_indices_onto_the_extracted_images():
    """The returned indices address the extracted subset, not the original corpus."""
    corpus = Corpus(cids=[f"{i}" for i in range(100)], cluster=[0] * 100, qidxs=[10, 20, 30], pidxs=[11, 21, 31])

    images, qidxs, pidxs = sample_pairs(corpus, count=3, seed=0)

    assert images == [10, 11, 20, 21, 30, 31]
    # Every remapped index points back at the corpus index it came from.
    for q, p in zip(qidxs, pidxs, strict=True):
        assert images[p] == images[q] + 1


def test_sampling_more_pairs_than_exist_takes_them_all():
    corpus = Corpus(cids=[f"{i}" for i in range(10)], cluster=[0] * 10, qidxs=[0, 2], pidxs=[1, 3])

    _, qidxs, _ = sample_pairs(corpus, count=999, seed=0)

    assert len(qidxs) == 2


def test_attaching_makes_the_projection_readable_by_the_eval_path(tmp_path, pairs):
    """The whole point: afterwards `--whiten-source learned` finds something."""
    x, qidxs, pidxs = pairs
    projection, mean = learn(x, qidxs, pidxs)
    path = tmp_path / "best.pth"
    torch.save(
        {
            "state_dict": {"features.0.weight": torch.zeros(1), "pool.p": torch.tensor(2.9)},
            "meta": {"architecture": "vgg16", "pooling": "gem", "whitening": False, "epoch": 30},
        },
        path,
    )

    attach(path, {variant: {"P": projection, "m": mean} for variant in ("ss", "ms")})
    loaded = load("vgg16", str(path))

    assert loaded.p == pytest.approx(2.9)
    compression = loaded.whitening(scales=(1.0,))
    assert np.allclose(compression.components, projection)
    assert np.allclose(compression.mean, mean.ravel())


def test_attaching_keeps_everything_else_in_the_checkpoint(tmp_path, pairs):
    """It edits one key of a training artifact in place; nothing else may move."""
    x, qidxs, pidxs = pairs
    projection, mean = learn(x, qidxs, pidxs)
    path = tmp_path / "best.pth"
    original = {
        "state_dict": {"features.0.weight": torch.arange(4.0), "pool.p": torch.tensor(2.9)},
        "meta": {"architecture": "vgg16", "pooling": "gem", "whitening": False, "epoch": 30, "outputdim": 512},
    }
    torch.save(original, path)

    attach(path, {"ss": {"P": projection, "m": mean}})
    reloaded = torch.load(path, map_location="cpu", weights_only=False)

    assert torch.equal(reloaded["state_dict"]["features.0.weight"], original["state_dict"]["features.0.weight"])
    assert reloaded["meta"]["epoch"] == 30
    assert reloaded["meta"]["outputdim"] == 512
    assert not (path.parent / "best.tmp").exists()  # the atomic write cleaned up
