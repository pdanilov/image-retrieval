"""LoRA on IRT's network: it starts as the frozen model, trains only its adapters, and
leaves a checkpoint the evaluation path reads as a plain network."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image

from cbir.descriptors.transformer.vit import FrozenViT
from cbir.training.loop import TrainConfig, _resume, _save, train_epoch
from cbir.training.net import RetrievalViT

SMALL = "dinov2-vits14"


@pytest.fixture(scope="module")
def paths(tmp_path_factory) -> list[str]:
    directory = tmp_path_factory.mktemp("lora")
    rng = np.random.default_rng(2)
    out = []
    for i in range(7):
        path = directory / f"{i}.png"
        Image.fromarray(rng.integers(0, 255, (50 + 7 * i, 70, 3), dtype=np.uint8)).save(path)
        out.append(str(path))
    return out


def _describe(net, paths) -> torch.Tensor:
    with torch.inference_mode():
        return net(torch.stack([net.load(path) for path in paths]))


def _perturb_adapters(net) -> None:
    # lora_B starts at zero, which makes the update vanish; give it something to merge.
    generator = torch.Generator().manual_seed(0)
    with torch.no_grad():
        for name, parameter in net.named_parameters():
            if "lora_B" in name:
                parameter.add_(torch.randn(parameter.shape, generator=generator) * 0.05)


def test_lora_starts_as_exactly_the_frozen_network(paths):
    # Zero-initialized B: before the first step the adapters add nothing, so the "before"
    # row and LoRA's starting point are one function.
    adapted = _describe(RetrievalViT(SMALL, lora_rank=4).eval(), paths)
    np.testing.assert_allclose(adapted, _describe(RetrievalViT(SMALL).eval(), paths), atol=1e-6)


def test_only_the_adapters_train():
    net = RetrievalViT(SMALL, lora_rank=4)
    trainable = {name for name, parameter in net.named_parameters() if parameter.requires_grad}
    assert trainable and all("lora_" in name for name in trainable)
    # 12 blocks x 4 projections x (A: 4x384 + B: 384x4) for ViT-S at rank 4.
    assert sum(p.numel() for p in net.parameters() if p.requires_grad) == 12 * 4 * 2 * 4 * 384


def test_a_step_moves_adapters_and_leaves_pretrained_weights_alone(paths):
    net = RetrievalViT(SMALL, lora_rank=4)
    base = {name: p.detach().clone() for name, p in net.named_parameters() if not p.requires_grad}
    adapters = {name: p.detach().clone() for name, p in net.named_parameters() if p.requires_grad}
    optimizer = torch.optim.Adam([p for p in net.parameters() if p.requires_grad], lr=1e-3)

    train_epoch(net, [[net.load(path) for path in paths]], optimizer, 0.85, batch_size=1, device="cpu", workers=0)

    after = dict(net.named_parameters())
    assert all(torch.equal(after[name], value) for name, value in base.items())
    assert any(not torch.equal(after[name], value) for name, value in adapters.items())


def test_best_checkpoint_is_merged_and_scored_by_evaluate(paths, tmp_path):
    net = RetrievalViT(SMALL, lora_rank=4).eval()
    _perturb_adapters(net)
    path = tmp_path / "best.pth"
    _save(path, net, TrainConfig(backbone=SMALL, lora_rank=4, lr=1e-4), epoch=1, metrics={})

    state = torch.load(path, weights_only=False)["state_dict"]
    assert not any("lora_" in key or "base_model" in key for key in state)

    images = [Image.open(p) for p in paths]
    evaluated = FrozenViT(SMALL, resize="stretch", checkpoint=str(path), device="cpu").extract(images)
    pretrained = FrozenViT(SMALL, resize="stretch", device="cpu").extract(images)
    np.testing.assert_allclose(evaluated, _describe(net, paths).numpy(), atol=1e-5)
    assert not np.allclose(evaluated, pretrained, atol=1e-3)


def test_last_checkpoint_keeps_adapters_and_resumes_them(paths, tmp_path):
    net = RetrievalViT(SMALL, lora_rank=4).eval()
    _perturb_adapters(net)
    config = TrainConfig(backbone=SMALL, lora_rank=4, lr=1e-4)
    optimizer = torch.optim.Adam([p for p in net.parameters() if p.requires_grad], lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
    path = tmp_path / "last.pth"
    _save(path, net, config, 3, {}, optimizer, scheduler, 0.5, best_epoch=3)

    fresh = RetrievalViT(SMALL, lora_rank=4).eval()
    fresh_optimizer = torch.optim.Adam([p for p in fresh.parameters() if p.requires_grad], lr=1e-4)
    fresh_scheduler = torch.optim.lr_scheduler.ExponentialLR(fresh_optimizer, gamma=0.99)
    _resume(path, fresh, fresh_optimizer, fresh_scheduler, config, "cpu", lambda *_: None)
    torch.testing.assert_close(_describe(fresh, paths), _describe(net, paths))


def test_lora_has_no_default_learning_rate():
    # The recipe's 5e-7 is a full fine-tuning value; zero-initialized adapters would barely
    # move under it, and the run would look like "LoRA adds nothing".
    with pytest.raises(ValueError, match="pass --lr"):
        TrainConfig(backbone=SMALL, lora_rank=8).resolved_lr()
    assert TrainConfig(backbone=SMALL, lora_rank=8, lr=1e-4).resolved_lr() == 1e-4


def test_lora_knobs_are_refused_where_they_mean_nothing():
    from cbir.training.loop import _build

    with pytest.raises(ValueError, match="transformer backbones only"):
        _build(TrainConfig(backbone="vgg16", lora_rank=8))
    with pytest.raises(ValueError, match="needs lora_rank"):
        RetrievalViT(SMALL, lora_alpha=16.0)
