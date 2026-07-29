import pytest

import cbir.data.holdout as holdout_module
from cbir.data.holdout import HeldOutSplit, held_out_database


def test_for_eval_roxford5k_holds_out_rparis6k():
    split = HeldOutSplit.for_eval("roxford5k")
    assert split.eval_dataset == "roxford5k"
    assert split.held_out_dataset == "rparis6k"


def test_for_eval_rparis6k_holds_out_roxford5k():
    split = HeldOutSplit.for_eval("rparis6k")
    assert split.eval_dataset == "rparis6k"
    assert split.held_out_dataset == "roxford5k"


def test_for_eval_rejects_unknown_dataset():
    with pytest.raises(ValueError, match="unknown dataset"):
        HeldOutSplit.for_eval("not-a-real-dataset")  # type: ignore[arg-type]


def test_direct_construction_with_correct_pair_succeeds():
    split = HeldOutSplit(eval_dataset="roxford5k", held_out_dataset="rparis6k")
    assert split.held_out_dataset == "rparis6k"


def test_direct_construction_cannot_hold_out_the_eval_dataset_itself():
    with pytest.raises(ValueError, match="invalidates the resulting mAP"):
        HeldOutSplit(eval_dataset="roxford5k", held_out_dataset="roxford5k")


def test_direct_construction_rejects_mismatched_pair():
    # roxford5k's held-out counterpart is rparis6k, not some other value.
    with pytest.raises(ValueError, match="held_out_dataset must be"):
        HeldOutSplit(eval_dataset="roxford5k", held_out_dataset="not-a-real-dataset")  # type: ignore[arg-type]


def test_direct_construction_rejects_unknown_eval_dataset():
    with pytest.raises(ValueError, match="unknown eval_dataset"):
        HeldOutSplit(eval_dataset="not-a-real-dataset", held_out_dataset="roxford5k")  # type: ignore[arg-type]


def test_held_out_database_downloads_the_held_out_dataset_and_returns_the_db_split(monkeypatch):
    calls = []

    def fake_download(name):
        calls.append(name)
        return ("query-sentinel", "db-sentinel")

    monkeypatch.setattr(holdout_module, "download", fake_download)

    split = HeldOutSplit.for_eval("roxford5k")
    result = held_out_database(split)

    assert calls == ["rparis6k"]
    assert result == "db-sentinel"
