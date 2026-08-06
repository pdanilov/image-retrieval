import numpy as np
import pytest

from cbir.descriptors.classic.aggregate.vlad import VLAD
from cbir.descriptors.classic.codebook.vocabulary import Vocabulary
from cbir.descriptors.classic.prepare import ClassicDescriptorInputs

# Two centers, so a VLAD vector is 2*2 = 4-dimensional and hand-checkable.
CENTERS = np.array([[0.0, 0.0], [10.0, 0.0]], dtype=np.float32)

# Descriptors: [1,1] and [2,0] assign to word0, [9,0] assigns to word1.
#   block0 = (1-0,1-0) + (2-0,0-0) = (3, 1)
#   block1 = (9-10, 0-0)          = (-1, 0)
IMAGE = np.array([[1.0, 1.0], [2.0, 0.0], [9.0, 0.0]], dtype=np.float32)
RAW = np.array([3.0, 1.0, -1.0, 0.0], dtype=np.float32)


def test_raw_vlad_sums_residuals_per_word():
    vocab = Vocabulary(CENTERS)
    assert list(VLAD.raw(vocab, IMAGE)) == list(RAW)


def test_raw_vlad_empty_image_is_zero():
    vocab = Vocabulary(CENTERS)
    assert list(VLAD.raw(vocab, np.zeros((0, 2), dtype=np.float32))) == [0.0, 0.0, 0.0, 0.0]


def test_normalize_global_l2_only():
    out = VLAD(Vocabulary(CENTERS), intra_norm=False).normalize(RAW[None, :])
    expected = RAW / np.sqrt(11.0)  # ||[3,1,-1,0]|| = sqrt(11)
    assert out[0] == pytest.approx(expected, abs=1e-6)
    assert np.linalg.norm(out[0]) == pytest.approx(1.0)


def test_normalize_intra_then_global():
    out = VLAD(Vocabulary(CENTERS), intra_norm=True).normalize(RAW[None, :])
    # block0 (3,1)/sqrt(10) = (0.948683, 0.316228); block1 (-1,0)/1 = (-1, 0).
    # Then global L2 over [0.948683, 0.316228, -1, 0] (norm = sqrt(2)).
    expected = np.array([0.948683, 0.316228, -1.0, 0.0]) / np.sqrt(2.0)
    assert out[0] == pytest.approx(expected, abs=1e-5)
    assert np.linalg.norm(out[0]) == pytest.approx(1.0)


def test_normalize_power_law_then_global():
    out = VLAD(Vocabulary(CENTERS), intra_norm=False, power=0.5).normalize(RAW[None, :])
    # signed sqrt of [3,1,-1,0] = [1.732051, 1, -1, 0], global norm = sqrt(5).
    expected = np.array([np.sqrt(3.0), 1.0, -1.0, 0.0]) / np.sqrt(5.0)
    assert out[0] == pytest.approx(expected, abs=1e-6)


def test_normalize_zero_row_stays_zero():
    out = VLAD(Vocabulary(CENTERS), intra_norm=True, power=0.5).normalize(np.zeros((1, 4), dtype=np.float32))
    assert not np.isnan(out).any()
    assert np.linalg.norm(out[0]) == 0.0


def test_encode_produces_unit_rows():
    vocab = Vocabulary(CENTERS)
    vectors = VLAD(vocab).encode([IMAGE, IMAGE])
    assert vectors.shape == (2, 4)
    assert np.linalg.norm(vectors[0]) == pytest.approx(1.0)


def test_train_then_encode_database_and_queries(tmp_cache):
    rng = np.random.default_rng(0)
    blobs = [rng.normal(mu, 0.1, size=(30, 2)) for mu in ([0, 0], [10, 0])]
    held_out = np.vstack(blobs).astype(np.float32)
    inputs = ClassicDescriptorInputs(
        held_out_dataset="test",
        held_out_descriptors=held_out,
        database_descriptors=[IMAGE],
        query_descriptors=[IMAGE],
    )

    model = VLAD.train(inputs, k=2, seed=0)
    db_vectors = model.encode(inputs.database_descriptors)
    q_vectors = model.encode(inputs.query_descriptors)

    assert model.vocabulary.k == 2
    assert db_vectors.shape == (1, 4)
    assert q_vectors.shape == (1, 4)
    assert np.linalg.norm(db_vectors[0]) == pytest.approx(1.0)
