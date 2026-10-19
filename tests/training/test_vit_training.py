"""IRT's network in the training loop: the knobs that resolve per family, the KoLeo term,
and the one property the whole comparison rests on — that a ViT trained here is scored
by the evaluation path as exactly the network that was trained."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from PIL import Image

from cbir.descriptors.transformer.vit import FrozenViT
from cbir.training.loop import TrainConfig, _save
from cbir.training.loss import KOLEO_EPS, koleo
from cbir.training.net import RetrievalViT

SMALL = "dinov2-vits14"


# --- configuration -------------------------------------------------------------------


def test_every_transformer_gets_irts_margin():
    assert TrainConfig(backbone="deit-vitb16-dist-384").resolved_margin() == 0.85
    assert TrainConfig(backbone=SMALL).resolved_margin() == 0.85


def test_whitening_defaults_on_for_a_cnn_and_off_for_a_vit():
    assert TrainConfig(backbone="resnet101").resolved_whiten() is True
    assert TrainConfig(backbone="deit-vits16-dist-224").resolved_whiten() is False


def test_supervised_whitening_is_refused_for_a_vit():
    # The projection is fitted with PooledCNN; IRT reduces with PCA at evaluation instead.
    with pytest.raises(ValueError, match="CNN checkpoints only"):
        TrainConfig(backbone="deit-vits16-dist-224", whiten=True).resolved_whiten()


def test_siglip_is_refused_before_anything_downloads():
    with pytest.raises(ValueError, match="no CLS token"):
        RetrievalViT("siglip2-vitl16-256")


def test_a_fixed_resolution_deit_refuses_another_size_before_anything_downloads():
    with pytest.raises(ValueError, match="384 x 384 only"):
        RetrievalViT("deit-vitb16-dist-384", image_size=224)


# --- KoLeo -----------------------------------------------------------------------------


def test_koleo_is_minus_the_mean_log_nearest_neighbour_distance():
    # Three points at 90 degree steps on the unit circle: every nearest neighbour is
    # sqrt(2) away, so the loss is -log sqrt(2).
    z = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    assert float(koleo(z)) == pytest.approx(-math.log(math.sqrt(2) + KOLEO_EPS), abs=1e-6)


def test_koleo_uses_each_points_own_nearest_neighbour():
    # Distances 1 (a-b), 1 (b-a), 2 (c to b): mean of -log(1), -log(1), -log(2).
    z = torch.tensor([[0.0], [1.0], [3.0]])
    assert float(koleo(z)) == pytest.approx(-math.log(2) / 3, abs=1e-6)


def test_koleo_stays_finite_on_duplicates():
    # A near-duplicate positive is common in a landmark corpus; -log 0 would be inf.
    z = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]], requires_grad=True)
    loss = koleo(z)
    loss.backward()
    assert torch.isfinite(loss)


def test_koleo_falls_as_the_points_spread():
    tight = torch.nn.functional.normalize(torch.tensor([[1.0, 0.1], [1.0, -0.1], [0.0, 1.0]]), dim=1)
    spread = torch.nn.functional.normalize(torch.tensor([[1.0, 1.0], [1.0, -1.0], [-1.0, 0.0]]), dim=1)
    assert float(koleo(spread)) < float(koleo(tight))


# --- the trained network is the evaluated one --------------------------------------------


@pytest.fixture(scope="module")
def image_path(tmp_path_factory) -> str:
    path = tmp_path_factory.mktemp("img") / "a.png"
    rng = np.random.default_rng(0)
    Image.fromarray(rng.integers(0, 255, (60, 90, 3), dtype=np.uint8)).save(path)
    return str(path)


@pytest.fixture(scope="module")
def net() -> RetrievalViT:
    return RetrievalViT(SMALL).eval()


def test_the_training_network_matches_the_frozen_readout(net, image_path):
    # Same weights, same squash, same CLS: the "before" row and the network that starts
    # training must be one function, or training's gain is measured against a stranger.
    with torch.inference_mode():
        trained = net(net.load(image_path).unsqueeze(0)).numpy()
    frozen = FrozenViT(SMALL, resize="stretch", device="cpu").extract([Image.open(image_path)])
    np.testing.assert_allclose(trained, frozen, atol=1e-5)


def test_a_checkpoint_we_write_is_what_evaluate_scores(net, image_path, tmp_path):
    # Perturb a weight so the fine-tuned network differs from the pretrained one, write
    # it as the loop does, and read it back the way `cbir evaluate vit --checkpoint` does.
    # Random, not a constant: a constant added to a token is exactly what the next
    # LayerNorm subtracts, which leaves the output unchanged and the test vacuous.
    with torch.no_grad():
        weight = net.network.encoder.layer[0].mlp.fc1.weight
        weight.add_(torch.randn(weight.shape, generator=torch.Generator().manual_seed(0)) * 0.1)
    path = tmp_path / "best.pth"
    _save(path, net, TrainConfig(backbone=SMALL), epoch=1, metrics={})

    with torch.inference_mode():
        trained = net(net.load(image_path).unsqueeze(0)).numpy()
    evaluated = FrozenViT(SMALL, resize="stretch", checkpoint=str(path), device="cpu").extract([Image.open(image_path)])
    pretrained = FrozenViT(SMALL, resize="stretch", device="cpu").extract([Image.open(image_path)])

    np.testing.assert_allclose(evaluated, trained, atol=1e-5)
    assert not np.allclose(evaluated, pretrained, atol=1e-3)
    assert torch.load(path, weights_only=False)["meta"]["pooling"] == "cls"


def test_a_checkpoint_for_another_model_is_refused(net, tmp_path):
    path = tmp_path / "best.pth"
    _save(path, net, TrainConfig(backbone=SMALL), epoch=1, metrics={})
    with pytest.raises(ValueError, match="not dinov2-vitb14"):
        FrozenViT("dinov2-vitb14", checkpoint=str(path), device="cpu")


# --- batching changes speed, not descriptors ---------------------------------------------


@pytest.fixture(scope="module")
def paths(tmp_path_factory) -> list[str]:
    directory = tmp_path_factory.mktemp("many")
    rng = np.random.default_rng(1)
    out = []
    for i, size in enumerate([(60, 90), (80, 40), (50, 50), (90, 30), (40, 70)]):
        path = directory / f"{i}.png"
        Image.fromarray(rng.integers(0, 255, (*size, 3), dtype=np.uint8)).save(path)
        out.append(str(path))
    return out


@pytest.mark.parametrize("workers", [0, 2])
def test_batched_describe_matches_one_forward_per_image_in_order(net, paths, workers, monkeypatch):
    from cbir.training import tuples

    monkeypatch.setattr(tuples, "BATCH", 2)  # several batches, one of them short
    with torch.inference_mode():
        batched = tuples.describe(net, paths, "cpu", None, "", workers=workers)
        single = torch.cat([net(net.load(path).unsqueeze(0)) for path in paths])
    torch.testing.assert_close(batched, single, atol=1e-5, rtol=0)


def test_a_tuple_through_the_loop_is_one_batch_with_the_same_loss(net, paths):
    # train_epoch stacks a square tuple into one forward; the loss must be the one seven
    # separate forwards give, or batching changed the objective.
    from cbir.training.loop import train_epoch
    from cbir.training.loss import contrastive_loss

    images = [net.load(path) for path in paths]
    calls = []
    forward = net.forward
    net.forward = lambda x: calls.append(len(x)) or forward(x)
    try:
        optimizer = torch.optim.SGD(net.parameters(), lr=0.0)
        loss = train_epoch(net, [images], optimizer, 0.85, batch_size=1, device="cpu", workers=0)
    finally:
        del net.forward
    with torch.no_grad():
        d = torch.cat([forward(image.unsqueeze(0)) for image in images])
        expected = contrastive_loss(d[:1], d[1:2], d[2:].unsqueeze(0), 0.85)

    assert calls == [len(paths)]
    assert loss == pytest.approx(float(expected), abs=1e-5)
