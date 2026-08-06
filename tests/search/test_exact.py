import numpy as np

from cbir.search.exact import exact_search

# Unit vectors, forced onto CPU so this runs without a GPU (matches CI).
QUERIES = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
DATABASE = np.array([[1.0, 0.0], [0.0, 1.0], [0.7071, 0.7071]], dtype=np.float32)


def test_exact_search_orders_by_descending_similarity():
    # query 0 = [1,0]: similarities to db = [1, 0, 0.7071] -> ranking [0, 2, 1]
    # query 1 = [0,1]: similarities to db = [0, 1, 0.7071] -> ranking [1, 2, 0]
    rankings = exact_search(QUERIES, DATABASE, device="cpu")
    np.testing.assert_array_equal(rankings, [[0, 2, 1], [1, 2, 0]])


def test_exact_search_output_shape_and_is_a_permutation_per_row():
    rankings = exact_search(QUERIES, DATABASE, device="cpu")
    assert rankings.shape == (QUERIES.shape[0], DATABASE.shape[0])
    for row in rankings:
        assert sorted(row.tolist()) == list(range(DATABASE.shape[0]))


def test_exact_search_defaults_to_available_device():
    # No device override -> must still run and produce a valid result.
    rankings = exact_search(QUERIES, DATABASE)
    np.testing.assert_array_equal(rankings, [[0, 2, 1], [1, 2, 0]])
