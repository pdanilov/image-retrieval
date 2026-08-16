import numpy as np
import pytest

from cbir.descriptors.classic.aggregate.bow import BagOfWords
from cbir.descriptors.classic.codebook.vocabulary import Vocabulary
from cbir.descriptors.classic.prepare import ClassicDescriptorInputs

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
    assert list(BagOfWords.term_frequencies(assignments, k=3)) == [2.0, 3.0, 1.0]


def test_inverse_document_frequency_hand_computed():
    idf = BagOfWords.inverse_document_frequency(CORPUS)
    expected = [np.log(4 / 3), np.log(4 / 1), np.log(4 / 2)]
    assert idf == pytest.approx(expected)


def test_idf_zero_for_never_seen_word():
    hist = np.array([[1, 0], [1, 0]], dtype=np.float32)  # word1 never occurs
    assert BagOfWords.inverse_document_frequency(hist)[1] == 0.0


def test_pure_helpers_need_no_instance():
    # Both are staticmethods -- callable straight off the class, no vocabulary or idf
    # required, which is what keeps this hand-computed math testable in isolation.
    assert BagOfWords.term_frequencies(np.array([0, 0]), k=2)[0] == 2.0
    assert BagOfWords.inverse_document_frequency(CORPUS).shape == (3,)


def test_apply_tfidf_row_is_l2_normalized_and_directional():
    model = BagOfWords(Vocabulary(CENTERS), BagOfWords.inverse_document_frequency(CORPUS))
    vectors = model.apply_tfidf(CORPUS)
    # img1 = [1,1,0]: weighted = [log(4/3), log(4), 0], then L2-normalized.
    w = np.array([np.log(4 / 3), np.log(4), 0.0])
    expected = w / np.linalg.norm(w)
    assert vectors[1] == pytest.approx(expected, abs=1e-6)
    assert np.linalg.norm(vectors[1]) == pytest.approx(1.0)


def test_apply_tfidf_zero_row_stays_zero():
    model = BagOfWords(Vocabulary(CENTERS), BagOfWords.inverse_document_frequency(CORPUS))
    out = model.apply_tfidf(np.zeros((1, 3), dtype=np.float32))
    assert not np.isnan(out).any()
    assert np.linalg.norm(out[0]) == 0.0


def test_histograms_assign_and_count():
    model = BagOfWords(Vocabulary(CENTERS), np.ones(3, dtype=np.float32))
    # Image with 2 descriptors near word0 and 1 near word2.
    image = np.array([[0.5, 0.5], [1.0, 0.0], [0.5, 9.0]], dtype=np.float32)
    assert list(model.histograms([image])[0]) == [2.0, 0.0, 1.0]


def test_histograms_empty_image_is_zero_row():
    model = BagOfWords(Vocabulary(CENTERS), np.ones(3, dtype=np.float32))
    assert list(model.histograms([np.zeros((0, 2), dtype=np.float32)])[0]) == [0.0, 0.0, 0.0]


def test_encode_applies_the_same_database_idf_to_queries():
    # idf is fitted state, so a query encoded later is weighted by the *database's* idf
    # rather than one derived from the query itself.
    vocab = Vocabulary(CENTERS)
    near0 = np.array([[0.1, 0.0]], dtype=np.float32)
    near1 = np.array([[10.0, 0.1]], dtype=np.float32)
    near2 = np.array([[0.0, 10.0]], dtype=np.float32)
    database = [near0, near1, near2, near0]  # word0 common (df high -> low idf)

    idf = BagOfWords.inverse_document_frequency(BagOfWords._histograms(vocab, database))
    model = BagOfWords(vocab, idf)

    db_vectors = model.encode(database)
    q_vectors = model.encode([near0])

    assert db_vectors.shape == (4, 3)
    assert q_vectors.shape == (1, 3)
    # The query has only word0, so after idf weighting its single nonzero component
    # still normalizes to a unit vector pointing along word0.
    assert np.linalg.norm(q_vectors[0]) == pytest.approx(1.0)
    assert q_vectors[0][0] == pytest.approx(1.0)


def test_train_fits_vocabulary_and_idf_then_encodes_both_corpora(tmp_cache):
    rng = np.random.default_rng(0)
    blobs = [rng.normal(mu, 0.1, size=(30, 2)) for mu in ([0, 0], [10, 0], [0, 10])]
    inputs = ClassicDescriptorInputs(
        held_out_dataset="test",
        held_out_descriptors=np.vstack(blobs).astype(np.float32),
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

    model = BagOfWords.train(inputs, k=3, seed=0)
    db_vectors = model.encode(inputs.database_descriptors)
    q_vectors = model.encode(inputs.query_descriptors)

    assert model.vocabulary.k == 3
    assert model.idf.shape == (3,)
    assert db_vectors.shape == (3, 3)
    assert q_vectors.shape == (1, 3)
    assert np.linalg.norm(q_vectors[0]) == pytest.approx(1.0)


def _counting_assign(monkeypatch):
    """Record how many images get hard-assigned, the dominant cost of BoW encoding."""
    assigned = []
    real_assign = Vocabulary.assign

    def counting(self, descriptors):
        assigned.append(len(descriptors))
        return real_assign(self, descriptors)

    monkeypatch.setattr(Vocabulary, "assign", counting)
    return assigned


def _blob_inputs():
    rng = np.random.default_rng(0)
    blobs = [rng.normal(mu, 0.1, size=(30, 2)) for mu in ([0, 0], [10, 0], [0, 10])]
    return ClassicDescriptorInputs(
        held_out_dataset="test",
        held_out_descriptors=np.vstack(blobs).astype(np.float32),
        database_descriptors=[
            np.array([[0.1, 0.1]], dtype=np.float32),
            np.array([[10.0, 0.1]], dtype=np.float32),
            np.array([[0.0, 10.0]], dtype=np.float32),
        ],
        query_descriptors=[np.array([[0.0, 10.0]], dtype=np.float32)],
    )


def test_train_and_encode_database_walks_the_database_once(tmp_cache, monkeypatch):
    # train() + encode(database) fits idf on one pass and then recomputes the identical
    # histograms to encode -- 2x the dominant cost. train_and_encode_database does one.
    inputs = _blob_inputs()
    n_database = len(inputs.database_descriptors)

    assigned = _counting_assign(monkeypatch)
    BagOfWords.train_and_encode_database(inputs, k=3, seed=0)
    assert len(assigned) == n_database

    assigned = _counting_assign(monkeypatch)
    two_pass = BagOfWords.train(inputs, k=3, seed=0)
    two_pass.encode(inputs.database_descriptors)
    assert len(assigned) == 2 * n_database


def test_train_and_encode_database_matches_the_two_pass_result(tmp_cache):
    # The saving must be pure bookkeeping: identical vocabulary, idf, and vectors.
    inputs = _blob_inputs()

    model, db_vectors = BagOfWords.train_and_encode_database(inputs, k=3, seed=0)
    reference = BagOfWords.train(inputs, k=3, seed=0)

    np.testing.assert_array_equal(model.vocabulary.centers, reference.vocabulary.centers)
    np.testing.assert_array_equal(model.idf, reference.idf)
    np.testing.assert_array_equal(db_vectors, reference.encode(inputs.database_descriptors))
    # The returned model is a normal one -- queries still go through encode().
    np.testing.assert_array_equal(model.encode(inputs.query_descriptors), reference.encode(inputs.query_descriptors))
