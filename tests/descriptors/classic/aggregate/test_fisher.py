import numpy as np
import pytest

from cbir.descriptors.classic.aggregate.fisher import FisherVector
from cbir.descriptors.classic.codebook.gaussian_mixture import GaussianMixture
from cbir.descriptors.classic.prepare import ClassicDescriptorInputs

# normalize() only reads `self.power`, never the model, but a FisherVector needs one.
_UNIT_GMM = GaussianMixture(
    weights=np.array([1.0]),
    means=np.array([[0.0]]),
    variances=np.array([[1.0]]),
)


def test_fisher_vector_single_component_hand_computed():
    # k=1, d=1, unit variance, x=2, N=1, gamma=1, u=(2-0)/1=2.
    #   G_mu    = 1/sqrt(1)      * 1 * u          = 2
    #   G_sigma = 1/sqrt(2*1)    * 1 * (u^2 - 1)  = 3/sqrt(2)
    gmm = GaussianMixture(
        weights=np.array([1.0]),
        means=np.array([[0.0]]),
        variances=np.array([[1.0]]),
    )
    fv = FisherVector.raw(gmm, np.array([[2.0]], dtype=np.float32))
    assert fv == pytest.approx([2.0, 3.0 / np.sqrt(2.0)], abs=1e-6)


def test_fisher_vector_layout_is_mu_then_sigma():
    # Two far-apart components, one descriptor sitting on component 0. It should drive
    # component 0's blocks and leave component 1's ~0. Layout: [mu0, mu1, sig0, sig1].
    gmm = GaussianMixture(
        weights=np.array([0.5, 0.5]),
        means=np.array([[0.0], [10.0]]),
        variances=np.array([[1.0], [1.0]]),
    )
    fv = FisherVector.raw(gmm, np.array([[0.0]], dtype=np.float32))
    mu0, mu1, sig0, sig1 = fv
    # x sits exactly on mean0 -> u0 = 0 -> G_mu0 = 0, and G_sigma0 = (0 - 1)/sqrt(2*0.5) = -1.
    assert mu0 == pytest.approx(0.0, abs=1e-6)
    assert sig0 == pytest.approx(-1.0, abs=1e-5)
    # Component 1 is ~50 log-units away -> negligible responsibility.
    assert mu1 == pytest.approx(0.0, abs=1e-6)
    assert sig1 == pytest.approx(0.0, abs=1e-6)


def test_fisher_vector_dimensionality_is_two_k_d():
    gmm = GaussianMixture(
        weights=np.array([0.5, 0.5]),
        means=np.zeros((2, 4)),
        variances=np.ones((2, 4)),
    )
    fv = FisherVector.raw(gmm, np.random.default_rng(0).normal(size=(7, 4)))
    assert fv.shape == (2 * 2 * 4,)


def test_fisher_vector_empty_image_is_zero():
    gmm = GaussianMixture(
        weights=np.array([1.0]),
        means=np.array([[0.0]]),
        variances=np.array([[1.0]]),
    )
    assert list(FisherVector.raw(gmm, np.zeros((0, 1), dtype=np.float32))) == [0.0, 0.0]


def test_normalize_power_law_then_global_l2():
    raw = np.array([[2.0, 3.0 / np.sqrt(2.0)]], dtype=np.float32)
    out = FisherVector(_UNIT_GMM, power=0.5).normalize(raw)
    ssr = np.sign(raw[0]) * np.abs(raw[0]) ** 0.5
    assert out[0] == pytest.approx(ssr / np.linalg.norm(ssr), abs=1e-6)
    assert np.linalg.norm(out[0]) == pytest.approx(1.0)


def test_normalize_zero_row_stays_zero():
    out = FisherVector(_UNIT_GMM, power=0.5).normalize(np.zeros((1, 4), dtype=np.float32))
    assert not np.isnan(out).any()
    assert np.linalg.norm(out[0]) == 0.0


def test_encode_produces_unit_rows_of_expected_width():
    gmm = GaussianMixture(
        weights=np.array([0.5, 0.5]),
        means=np.array([[0.0], [10.0]]),
        variances=np.array([[1.0], [1.0]]),
    )
    images = [np.array([[0.0], [1.0]], dtype=np.float32), np.array([[9.0]], dtype=np.float32)]
    vectors = FisherVector(gmm).encode(images)
    assert vectors.shape == (2, 2 * 2 * 1)
    assert np.linalg.norm(vectors[0]) == pytest.approx(1.0)


def test_train_then_encode_database_and_queries(tmp_cache):
    rng = np.random.default_rng(0)
    held_out = np.vstack([rng.normal(0.0, 0.3, size=(50, 2)), rng.normal(8.0, 0.3, size=(50, 2))]).astype(np.float32)
    inputs = ClassicDescriptorInputs(
        held_out_dataset="test",
        held_out_descriptors=held_out,
        database_descriptors=[np.array([[0.0, 0.0]], dtype=np.float32)],
        query_descriptors=[np.array([[8.0, 8.0]], dtype=np.float32)],
    )

    model = FisherVector.train(inputs, k=2, seed=0)
    db_vectors = model.encode(inputs.database_descriptors)
    q_vectors = model.encode(inputs.query_descriptors)

    assert model.model.k == 2
    assert db_vectors.shape == (1, 2 * 2 * 2)
    assert q_vectors.shape == (1, 2 * 2 * 2)
    assert np.linalg.norm(db_vectors[0]) == pytest.approx(1.0)
