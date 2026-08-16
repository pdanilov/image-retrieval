import pytest

from cbir.eval.frame import LOCAL_DIM, descriptor_dim, tidy_rows
from cbir.eval.results import RunRecord


def _record(technique="bow", k=5000, **params) -> RunRecord:
    return RunRecord(
        dataset="roxford5k",
        held_out_dataset="rparis6k",
        technique=technique,
        params={"k": k, "seed": 0, **params},
        metrics={
            "easy": {"map": 0.1775, "mp_at_k": 0.3211, "k": 10, "num_queries": 68, "num_excluded": 2},
            "medium": {"map": 0.1484, "mp_at_k": 0.3050, "k": 10, "num_queries": 70, "num_excluded": 0},
            "hard": {"map": 0.0570, "mp_at_k": 0.0855, "k": 10, "num_queries": 70, "num_excluded": 0},
        },
        seconds=1135.3,
        commit="abc1234",
    )


def test_one_row_per_run_and_protocol():
    rows = tidy_rows(records=[_record(), _record(technique="vlad", k=64)])

    assert len(rows) == 6
    assert {row["protocol"] for row in rows} == {"easy", "medium", "hard"}
    assert rows[0]["map"] == pytest.approx(0.1775)


def test_dim_is_the_axis_the_techniques_are_comparable_on():
    # At equal k these three are wildly unequal descriptors, which is the whole reason
    # this column exists: plotting mAP against k alone flatters BoW by ~128x.
    assert descriptor_dim("bow", {"k": 64}) == 64
    assert descriptor_dim("vlad", {"k": 64}) == 64 * LOCAL_DIM
    assert descriptor_dim("fisher", {"k": 64}) == 2 * 64 * LOCAL_DIM


def test_local_dim_matches_the_extractor_it_describes():
    # frame.LOCAL_DIM is hard-coded so that plotting doesn't import cv2. That
    # duplication is only safe if it cannot drift: if RootSIFT's dimensionality ever
    # changes, every VLAD and Fisher point would land at the wrong x with nothing to
    # signal it. Imported here rather than in frame.py -- a test can afford OpenCV.
    from cbir.descriptors.classic.local import rootsift

    assert LOCAL_DIM == rootsift.DIM


def test_dim_is_none_rather_than_wrong_for_an_unknown_technique():
    # A guessed dimensionality would put a point on the comparison axis in the wrong
    # place; None leaves it off the chart, which is the honest failure.
    assert descriptor_dim("gem", {"k": 64}) is None
    assert descriptor_dim("bow", {}) is None


def test_rows_carry_the_full_params_so_runs_stay_distinguishable():
    # Two Fisher runs differing only by `sample` share (technique, k). Without the
    # params dict they would be indistinguishable rows and would plot as one line.
    rows = tidy_rows(records=[_record(technique="fisher", k=64, sample=1_000_000)])
    assert rows[0]["params"]["sample"] == 1_000_000


def test_excluded_counts_survive_into_the_frame():
    # A protocol scored over fewer queries is not comparable to one scored over all of
    # them; the chart must be able to say so rather than silently plotting both.
    rows = {row["protocol"]: row for row in tidy_rows(records=[_record()])}
    assert rows["easy"]["num_excluded"] == 2
    assert rows["medium"]["num_excluded"] == 0


def test_empty_store_gives_no_rows(tmp_path):
    assert tidy_rows(tmp_path / "missing.jsonl") == []
