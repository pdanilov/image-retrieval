"""Resume, which is what makes a restart policy worth having.

Without it a supervisor that restarts a crashed run is worse than no supervisor: it
throws away every completed epoch and starts from zero. These tests use a fake model and
corpus — the point is the bookkeeping (which epoch, which optimizer state, which best),
not the training, and a real epoch is fifteen minutes.
"""

from __future__ import annotations

import pytest
import torch

from cbir.training.loop import TrainConfig, _resume, _save
from cbir.training.net import RetrievalNet


@pytest.fixture
def run(tmp_path):
    """A model, its optimizer and scheduler, mid-training and saved to `last.pth`."""
    model = RetrievalNet("vgg16", weights="torchvision")
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-7)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99005)

    # Two real steps, so optimizer and scheduler both carry state worth restoring.
    for _ in range(2):
        model(torch.randn(1, 3, 64, 64)).sum().backward()
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()
    with torch.no_grad():
        model.pool.p.fill_(2.77)

    path = tmp_path / "last.pth"
    _save(path, model, TrainConfig(backbone="vgg16"), 7, {"mean_average_precision": 0.5}, optimizer, scheduler, 0.61)
    return path, model, optimizer, scheduler


def test_nothing_to_resume_is_not_an_error(tmp_path):
    # A first run and a resumed run must be the same code path, or the common case is
    # the one that goes untested.
    model = RetrievalNet("vgg16", weights="torchvision")
    optimizer = torch.optim.Adam(model.parameters())
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

    assert _resume(tmp_path / "absent.pth", model, optimizer, scheduler, TrainConfig(), "cpu", print) is None


def test_it_restarts_at_the_epoch_after_the_one_that_finished(run):
    path, _, _, _ = run
    fresh, optimizer, scheduler = _fresh()

    start, best, best_epoch = _resume(
        path, fresh, optimizer, scheduler, TrainConfig(backbone="vgg16"), "cpu", lambda *_: None
    )

    assert start == 8  # epoch 7 completed, so 8 is next
    assert best == pytest.approx(0.61)
    # `_save` was called without `best_epoch`, as pre-early-stopping checkpoints were:
    # the patience clock restarts at the resumed epoch rather than firing immediately.
    assert best_epoch == 7


def test_the_patience_clock_survives_a_resume(tmp_path):
    """Otherwise a supervisor that restarts the job hands it unlimited patience."""
    model, optimizer, scheduler = _fresh()
    path = tmp_path / "last.pth"
    _save(path, model, TrainConfig(), 9, {}, optimizer, scheduler, 0.61, best_epoch=4)

    fresh, fresh_optimizer, fresh_scheduler = _fresh()
    start, _, best_epoch = _resume(path, fresh, fresh_optimizer, fresh_scheduler, TrainConfig(), "cpu", lambda *_: None)

    assert (start, best_epoch) == (10, 4)  # five epochs stale, not zero


def test_the_trained_weights_come_back(run):
    path, trained, _, _ = run
    fresh, optimizer, scheduler = _fresh()

    _resume(path, fresh, optimizer, scheduler, TrainConfig(backbone="vgg16"), "cpu", lambda *_: None)

    assert fresh.pool.p.item() == pytest.approx(2.77)
    for restored, original in zip(fresh.parameters(), trained.parameters(), strict=True):
        assert torch.equal(restored, original)


def test_the_optimizer_moments_come_back(run):
    # Adam without its moments is not Adam. Dropping them silently would give the resumed
    # run a different trajectory from the one it claims to continue.
    path, _, trained_optimizer, _ = run
    fresh, optimizer, scheduler = _fresh()

    _resume(path, fresh, optimizer, scheduler, TrainConfig(backbone="vgg16"), "cpu", lambda *_: None)

    restored = optimizer.state_dict()["state"]
    original = trained_optimizer.state_dict()["state"]
    assert set(restored) == set(original)
    assert restored[0]["step"] == original[0]["step"]
    assert torch.equal(restored[0]["exp_avg"], original[0]["exp_avg"])


def test_the_decayed_learning_rate_comes_back(run):
    # The schedule decays per epoch. Restarting it would hand epoch 8 the learning rate
    # of epoch 1 -- a slow, silent divergence rather than a crash.
    path, _, _, trained_scheduler = run
    fresh, optimizer, scheduler = _fresh()

    _resume(path, fresh, optimizer, scheduler, TrainConfig(backbone="vgg16"), "cpu", lambda *_: None)

    assert scheduler.get_last_lr() == pytest.approx(trained_scheduler.get_last_lr())
    assert scheduler.last_epoch == trained_scheduler.last_epoch


def test_resuming_another_architecture_is_refused(run):
    path, _, _, _ = run
    fresh = RetrievalNet("resnet101", weights="torchvision")
    optimizer = torch.optim.Adam(fresh.parameters())
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

    with pytest.raises(ValueError, match="refusing to resume"):
        _resume(path, fresh, optimizer, scheduler, TrainConfig(backbone="resnet101"), "cpu", lambda *_: None)


def test_the_evaluated_checkpoint_stays_free_of_training_state(tmp_path):
    # best.pth is the artifact; carrying Adam's moments would triple its size for
    # something the eval path never reads.
    model = RetrievalNet("vgg16", weights="torchvision")

    _save(tmp_path / "best.pth", model, TrainConfig(), 3, {})
    payload = torch.load(tmp_path / "best.pth", map_location="cpu", weights_only=False)

    assert set(payload) == {"state_dict", "meta"}


def _fresh():
    model = RetrievalNet("vgg16", weights="torchvision")
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-7)
    return model, optimizer, torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99005)
