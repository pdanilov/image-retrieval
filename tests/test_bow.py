import numpy as np
import pytest

from cbir.descriptors.classic.bow import (
    apply_tfidf,
    encode,
    fit_and_encode,
    histograms,
    inverse_document_frequency,
    term_frequencies,
)
from cbir.descriptors.classic.prepare import ClassicDescriptorInputs
from cbir.descriptors.classic.vocabulary import Vocabulary

CENTERS = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0]], dtype=np.float32)

# A 4-image corpus of raw count histograms (k=3) used for the idf/tf-idf tests.
#   word0 occurs in images 0,1,2 -> df=3
#   word1 occurs in image  1     -> df=1
#   word2 occurs in images 2,3   -> df=2
CORPUS = np.array(
    [
        [1, 0, 0],
        [1, 1, 0],
        [1, 0, 1],
        [0, 0, 1],
    ],
    dtype=np.float32,
)


def test_term_frequencies_counts_words():
    assignments = np.array([0, 0, 2, 1, 1, 1])
    assert list(term_frequencies(assignments, k=3)) == [2.0, 3.0, 1.0]


def test_inverse_document_frequency_hand_computed():
    idf = inverse_document_frequency(CORPUS)
    expected = [np.log(4 / 3), np.log(4 / 1), np.log(4 / 2)]
    assert idf == pytest.approx(expected)


def test_idf_zero_for_never_seen_word():
    hist = np.array([[1, 0], [1, 0]], dtype=np.float32)  # word1 never occurs
    assert inverse_document_frequency(hist)[1] == 0.0


def test_apply_tfidf_row_is_l2_normalized_and_directional():
    idf = inverse_document_frequency(CORPUS)
    vectors = apply_tfidf(CORPUS, idf)
    # img1 = [1,1,0]: weighted = [log(4/3), log(4), 0], then L2-normalized.
    w = np.array([np.log(4 / 3), np.log(4), 0.0])
    expected = w / np.linalg.norm(w)
    assert vectors[1] == pytest.approx(expected, abs=1e-6)
    assert np.linalg.norm(vectors[1]) == pytest.approx(1.0)


def test_apply_tfidf_zero_row_stays_zero():
    idf = inverse_document_frequency(CORPUS)
    hist = np.zeros((1, 3), dtype=np.float32)
    assert not np.isnan(apply_tfidf(hist, idf)).any()
    assert np.linalg.norm(apply_tfidf(hist, idf)[0]) == 0.0


def test_histograms_assign_and_count():
    vocab = Vocabulary(CENTERS)
    # Image with 2 descriptors near word0 and 1 near word2.
    image = np.array([[0.5, 0.5], [1.0, 0.0], [0.5, 9.0]], dtype=np.float32)
    hist = histograms(vocab, [image])
    assert list(hist[0]) == [2.0, 0.0, 1.0]


def test_histograms_empty_image_is_zero_row():
    vocab = Vocabulary(CENTERS)
    hist = histograms(vocab, [np.zeros((0, 2), dtype=np.float32)])
    assert list(hist[0]) == [0.0, 0.0, 0.0]


def test_encode_applies_database_idf_to_queries():
    vocab = Vocabulary(CENTERS)
    near0 = np.array([[0.1, 0.0]], dtype=np.float32)
    near1 = np.array([[10.0, 0.1]], dtype=np.float32)
    near2 = np.array([[0.0, 10.0]], dtype=np.float32)
    database = [near0, near1, near2, near0]  # word0 common (df high -> low idf)
    queries = [near0]

    db_vectors, q_vectors = encode(vocab, database, queries)

    assert db_vectors.shape == (4, 3)
    assert q_vectors is not None and q_vectors.shape == (1, 3)
    # The query has only word0, which is the most common word in the database, so after
    # idf weighting its single nonzero component still normalizes to a unit vector.
    assert np.linalg.norm(q_vectors[0]) == pytest.approx(1.0)
    # Query vector direction is word0 only.
    assert q_vectors[0][0] == pytest.approx(1.0)


def test_encode_without_queries_returns_none():
    vocab = Vocabulary(CENTERS)
    db_vectors, q_vectors = encode(vocab, [np.array([[0.0, 0.0]], dtype=np.float32)])
    assert q_vectors is None
    assert db_vectors.shape == (1, 3)


def test_fit_and_encode_trains_vocabulary_and_encodes_database_and_queries(tmp_cache):

    rng = np.random.default_rng(0)
    blobs = [rng.normal(mu, 0.1, size=(30, 2)) for mu in ([0, 0], [10, 0], [0, 10])]
    held_out = np.vstack(blobs).astype(np.float32)
    inputs = ClassicDescriptorInputs(
        held_out_dataset="test",
        held_out_descriptors=held_out,
        # All three words must appear in the database, or idf zeroes out any word the
        # database never saw (idf is a database-corpus property) -- including one
        # the query then relies on.
        database_descriptors=[
            np.array([[0.1, 0.1]], dtype=np.float32),
            np.array([[10.0, 0.1]], dtype=np.float32),
            np.array([[0.0, 10.0]], dtype=np.float32),
        ],
        query_descriptors=[np.array([[0.0, 10.0]], dtype=np.float32)],
    )

    vocabulary, db_vectors, q_vectors = fit_and_encode(inputs, k=3, seed=0)

    assert vocabulary.k == 3
    assert db_vectors.shape == (3, 3)
    assert q_vectors.shape == (1, 3)
    assert np.linalg.norm(q_vectors[0]) == pytest.approx(1.0)
