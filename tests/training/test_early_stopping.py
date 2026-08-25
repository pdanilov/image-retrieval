"""Early stopping: when the loop gives up, and what it keeps when it does.

`train()` is driven here with every expensive part stubbed — no corpus, no mining, no
real forward pass — because the thing under test is purely the bookkeeping around the
validation score, and that bookkeeping is what decides whether hours of GPU get spent.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from cbir.training import loop
from cbir.training.loop import TrainConfig, train


class _Stub(torch.nn.Module):
    """Enough of `RetrievalNet` for the loop: a parameter to optimize and a `pool.p`."""

    dim = 8

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.pool = torch.nn.Module()
        self.pool.p = torch.nn.Parameter(torch.tensor(3.0))


@pytest.fixture
def scores(monkeypatch):
    """Drive the loop off a fixed list of validation mAPs, one per epoch."""

    def install(values: list[float]):
        seen = iter(values)
        monkeypatch.setattr(loop, "RetrievalNet", _Stub)
        monkeypatch.setattr(loop.Corpus, "load", classmethod(lambda cls, split: SimpleNamespace(cids=[])))
        monkeypatch.setattr(loop, "sample_epoch", lambda *a, **k: [])
        monkeypatch.setattr(loop, "train_epoch", lambda *a, **k: 0.1)
        monkeypatch.setattr(loop, "_track", lambda *a, **k: None)
        monkeypatch.setattr(loop, "validate", lambda *a, **k: {"mean_average_precision": next(seen)})

    return install


def _epochs_run(directory) -> int:
    return int(torch.load(directory / "last.pth", weights_only=False)["meta"]["epoch"])


def test_it_stops_once_the_gains_stay_under_min_delta(scores, tmp_path):
    # baseline 0.50, then a real gain, then four epochs of crumbs.
    scores([0.50, 0.60, 0.6002, 0.6004, 0.6006, 0.6008, 0.6010, 0.6012])
    config = TrainConfig(epochs=20, patience=3, min_delta=0.001, out=tmp_path, resume=False)

    best_path = train(config, device="cpu", log=lambda *_: None)

    # Epoch 1 was the last real gain; 2, 3, 4 were crumbs, so it stops after epoch 4.
    assert _epochs_run(best_path.parent) == 4


def test_stopping_early_still_keeps_the_genuinely_best_checkpoint(scores, tmp_path):
    """The crumbs are too small to reset patience but still improve the model."""
    scores([0.50, 0.60, 0.6002, 0.6004, 0.6006, 0.6008])
    config = TrainConfig(epochs=20, patience=3, min_delta=0.001, out=tmp_path, resume=False)

    best_path = train(config, device="cpu", log=lambda *_: None)
    payload = torch.load(best_path, weights_only=False)

    # Not epoch 1's 0.60: every crumb was still an improvement worth checkpointing.
    assert payload["meta"]["metrics"]["mean_average_precision"] == pytest.approx(0.6006)
    assert payload["meta"]["epoch"] == 4


def test_steady_gains_over_min_delta_run_the_full_schedule(scores, tmp_path):
    scores([0.50, 0.52, 0.54, 0.56, 0.58])
    config = TrainConfig(epochs=4, patience=3, min_delta=0.001, out=tmp_path, resume=False)

    best_path = train(config, device="cpu", log=lambda *_: None)

    assert _epochs_run(best_path.parent) == 4


def test_a_single_bad_epoch_does_not_end_the_run(scores, tmp_path):
    """The observed curve dips by ~0.0001 now and then and recovers next epoch."""
    scores([0.50, 0.60, 0.5998, 0.62, 0.6199, 0.64])
    config = TrainConfig(epochs=5, patience=2, min_delta=0.001, out=tmp_path, resume=False)

    best_path = train(config, device="cpu", log=lambda *_: None)

    assert _epochs_run(best_path.parent) == 5


def test_patience_none_disables_the_rule(scores, tmp_path):
    scores([0.50, 0.50, 0.50, 0.50, 0.50])
    config = TrainConfig(epochs=4, patience=None, min_delta=0.001, out=tmp_path, resume=False)

    best_path = train(config, device="cpu", log=lambda *_: None)

    assert _epochs_run(best_path.parent) == 4
