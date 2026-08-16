import numpy as np
import pytest

from cbir.descriptors.classic.codebook.vocabulary import Vocabulary

# Three well-separated centers used across the assignment tests.
CENTERS = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0]], dtype=np.float32)


def test_k_and_d_report_center_shape():
    vocab = Vocabulary(CENTERS)
    assert vocab.k == 3
    assert vocab.d == 2


def test_float64_centers_are_downcast():
    vocab = Vocabulary(np.zeros((2, 4), dtype=np.float64))
    assert vocab.centers.dtype == np.float32


def test_assign_picks_nearest_center():
    vocab = Vocabulary(CENTERS)
    points = np.array([[1.0, 1.0], [9.0, 1.0], [1.0, 9.0]], dtype=np.float32)
    assert list(vocab.assign(points)) == [0, 1, 2]


def test_assign_breaks_ties_to_lowest_index():
    # [5, 0] is equidistant (d^2 = 25) from centers 0 and 1; argmin takes the lower.
    vocab = Vocabulary(CENTERS)
    assert vocab.assign(np.array([[5.0, 0.0]], dtype=np.float32))[0] == 0


def test_assign_rejects_dimension_mismatch():
    vocab = Vocabulary(CENTERS)
    with pytest.raises(ValueError, match="!= vocabulary dim"):
        vocab.assign(np.zeros((3, 5), dtype=np.float32))


def test_squared_distances_hand_computed():
    vocab = Vocabulary(CENTERS)
    d2 = vocab._squared_distances(np.array([[5.0, 0.0]], dtype=np.float32))
    # [5,0]: to (0,0)=25, to (10,0)=25, to (0,10)=25+100=125.
    assert d2[0] == pytest.approx([25.0, 25.0, 125.0])


def test_train_recovers_separated_blobs():
    rng = np.random.default_rng(0)
    blobs = [rng.normal(mu, 0.1, size=(50, 2)) for mu in ([0, 0], [10, 0], [0, 10])]
    descriptors = np.vstack(blobs).astype(np.float32)

    vocab = Vocabulary.train(descriptors, k=3, seed=0, minibatch=False)

    # The three true centers must map to three distinct words, and every point within a
    # blob must share one word — i.e. k-means recovered the blobs (up to label order).
    assert len(set(vocab.assign(CENTERS).tolist())) == 3
    for blob in blobs:
        assert len(set(vocab.assign(blob.astype(np.float32)).tolist())) == 1


def test_train_rejects_more_clusters_than_points():
    with pytest.raises(ValueError, match="cannot train"):
        Vocabulary.train(np.zeros((3, 2), dtype=np.float32), k=5, seed=0)
