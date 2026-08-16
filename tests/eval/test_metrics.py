import math

import pytest

from cbir.eval.metrics import (
    QueryGroundTruth,
    compute_ap,
    evaluate,
    precision_at_k,
    protocol_sets,
    query_ap,
)


def test_protocol_sets_easy_medium_hard():
    easy, hard, junk = [1, 2], [3], [4]
    assert protocol_sets(easy, hard, junk, "easy") == ({1, 2}, {3, 4})
    assert protocol_sets(easy, hard, junk, "medium") == ({1, 2, 3}, {4})
    assert protocol_sets(easy, hard, junk, "hard") == ({3}, {1, 2, 4})


def test_protocol_sets_rejects_unknown_protocol():
    with pytest.raises(ValueError, match="unknown protocol"):
        # Deliberately not a valid Protocol literal, to exercise the runtime guard
        # (e.g. a bad value from a CLI arg or config file wouldn't be caught by typing alone).
        protocol_sets([], [], [], "bogus")  # type: ignore[arg-type]


def test_compute_ap_perfect_ranking_is_one():
    # Both relevant items ranked first -> perfect precision/recall curve.
    assert compute_ap([0, 1], num_relevant=2) == pytest.approx(1.0)


def test_compute_ap_hand_computed_imperfect_ranking():
    # Relevant items at 0-based ranks 3 and 4 out of 5 -> AP = 0.225 (hand-derived
    # via the trapezoidal formula: term0=(0/3+1/4)/2*0.5=0.0625, term1=(1/4+2/5)/2*0.5=0.1625).
    assert compute_ap([3, 4], num_relevant=2) == pytest.approx(0.225)


def test_compute_ap_rejects_zero_relevant():
    with pytest.raises(ValueError, match="num_relevant"):
        compute_ap([0], num_relevant=0)


def test_query_ap_removes_ignored_before_scoring():
    # ranking: [0(junk), 1(pos), 2, 3(pos), 4]. Dropping junk index 0 shifts positives
    # to 0-based ranks [0, 2] among the remaining 4 items -> AP = 0.7917 (hand-derived:
    # term0=(1+1)/2*0.5=0.5, term1=(0.5+2/3)/2*0.5=0.291667, sum=0.791667).
    ranking = [0, 1, 2, 3, 4]
    ap = query_ap(ranking, positive={1, 3}, ignored={0})
    assert ap == pytest.approx(0.791667, abs=1e-5)


def test_query_ap_no_positives_is_nan():
    assert math.isnan(query_ap([0, 1, 2], positive=set(), ignored=set()))


def test_precision_at_k_caps_denominator_for_small_ground_truth():
    # Only 2 positives total; both retrieved within the first 2 ranks.
    ranking = [1, 3, 0, 2, 4]
    assert precision_at_k(ranking, positive={1, 3}, ignored=set(), k=1) == pytest.approx(1.0)
    assert precision_at_k(ranking, positive={1, 3}, ignored=set(), k=10) == pytest.approx(1.0)


def test_precision_at_k_hand_computed():
    # Positives at 1-based ranks 4 and 5 -> P@10 capped at kq=min(5,10)=5 -> 2/5.
    ranking = [0, 2, 4, 1, 3]
    assert precision_at_k(ranking, positive={1, 3}, ignored=set(), k=10) == pytest.approx(0.4)


def test_evaluate_excludes_queries_with_no_positives():
    ground_truth = [
        QueryGroundTruth(easy=[0], hard=[], junk=[]),
        QueryGroundTruth(easy=[], hard=[], junk=[]),  # no easy positives -> excluded from "easy"
    ]
    rankings = [[0, 1, 2], [0, 1, 2]]
    result = evaluate(rankings, ground_truth, protocol="easy", k=10)
    assert result.num_queries == 1
    assert result.num_excluded == 1
    assert result.map == pytest.approx(1.0)


def test_evaluate_averages_ap_across_queries():
    ground_truth = [
        QueryGroundTruth(easy=[0, 1], hard=[], junk=[]),
        QueryGroundTruth(easy=[1], hard=[], junk=[]),
    ]
    rankings = [
        [0, 1, 2],  # both positives first -> AP = 1.0
        [2, 0, 1],  # positive at 0-based rank 2 -> AP = compute_ap([2], 1)
    ]
    result = evaluate(rankings, ground_truth, protocol="easy", k=10)
    expected_ap_1 = 1.0
    expected_ap_2 = compute_ap([2], num_relevant=1)
    assert result.map == pytest.approx((expected_ap_1 + expected_ap_2) / 2)
    assert result.num_queries == 2
    assert result.num_excluded == 0
