"""BatchNorm during training, which is the difference between resnet101 and rubbish.

A tuple is seven images, but each gets its own forward pass -- they keep their aspect
ratios, so they cannot be stacked -- and a BatchNorm left in training mode therefore sees
N=1 and becomes instance normalization. The fatal consequence is not that its statistics
are noisy (these layers see a median 391 spatial positions per channel) but that training
and evaluation stop computing the same function, so the loss optimizes a network that
validation never scores.

VGG16 has no BatchNorm and so cannot catch this; resnet101 has 104 and collapsed from
0.6465 mAP to 0.1545 when it was missed.
"""

from __future__ import annotations

import pytest
import torch

from cbir.training.loop import _freeze_batchnorm, train_epoch
from cbir.training.net import RetrievalNet


class _Tiny(torch.nn.Module):
    """A conv-BN-pool stack shaped like the real thing but small enough to test."""

    def __init__(self):
        super().__init__()
        self.features = torch.nn.Sequential(torch.nn.Conv2d(3, 4, 3, padding=1), torch.nn.BatchNorm2d(4))
        self.pool = torch.nn.Module()
        self.pool.p = torch.nn.Parameter(torch.tensor(3.0))

    @property
    def bn(self):
        return self.features[1]

    def forward(self, x):
        features = self.features(x)
        pooled = features.clamp(min=1e-6).pow(self.pool.p).mean(dim=(2, 3)).pow(1.0 / self.pool.p)
        return torch.nn.functional.normalize(pooled, dim=1)


@pytest.fixture
def tuples():
    """Two tuples of (query, positive, two negatives), each image a different size.

    Two negatives, not one: a single tuple with a single negative is the degenerate case
    the loss refuses, since PML silently returns zero for it.
    """
    torch.manual_seed(0)
    return [
        [torch.rand(3, 24, 32), torch.rand(3, 32, 24), torch.rand(3, 28, 28), torch.rand(3, 20, 30)] for _ in range(2)
    ]


def test_running_statistics_are_not_touched_by_training(tuples):
    """The batch is one image, so what it estimates is biased, not merely imprecise.

    Population variance is within-image plus between-image variance; N=1 can only see
    the first, so the running estimate is dragged consistently downward.
    """
    model = _Tiny()
    before_mean = model.bn.running_mean.clone()
    before_var = model.bn.running_var.clone()

    train_epoch(model, tuples, torch.optim.Adam(model.parameters(), lr=1e-3), 0.85, 2, "cpu")

    assert torch.equal(model.bn.running_mean, before_mean)
    assert torch.equal(model.bn.running_var, before_var)
    assert model.bn.num_batches_tracked.item() == 0


def test_the_rest_of_the_model_still_trains(tuples):
    """Freezing BatchNorm must not freeze the network around it."""
    model = _Tiny()
    before = model.features[0].weight.clone()

    train_epoch(model, tuples, torch.optim.Adam(model.parameters(), lr=1e-2), 0.85, 2, "cpu")

    assert not torch.equal(model.features[0].weight, before)


def test_the_batchnorm_affine_parameters_still_train(tuples):
    """The reference freezes the statistics, not the learnable scale and shift."""
    model = _Tiny()
    before = model.bn.weight.clone()

    train_epoch(model, tuples, torch.optim.Adam(model.parameters(), lr=1e-2), 0.85, 2, "cpu")

    assert not torch.equal(model.bn.weight, before)


def test_batchnorm_is_in_eval_mode_while_the_model_trains(tuples):
    model = _Tiny()
    train_epoch(model, tuples, torch.optim.Adam(model.parameters(), lr=1e-3), 0.85, 2, "cpu")

    assert model.bn.training is False
    assert model.features[0].training is True


def test_resnet_is_the_architecture_this_protects_and_vgg_cannot_catch_it():
    """Pins the asymmetry that let this ship: vgg16 has nothing to freeze."""
    counts = {}
    for backbone in ("vgg16", "resnet101"):
        model = RetrievalNet(backbone, weights="torchvision")
        counts[backbone] = sum(isinstance(m, torch.nn.modules.batchnorm._BatchNorm) for m in model.modules())
    assert counts["vgg16"] == 0
    assert counts["resnet101"] == 104


def test_training_and_evaluation_see_the_same_network(tuples):
    """The property that actually matters, and the one that failed.

    With the statistics frozen and no dropout anywhere, a forward pass in training mode
    must equal one in eval mode. Without the freeze they diverge -- measured on the real
    resnet101, the two descriptors for one image had cosine similarity 0.58. The loss was
    therefore optimizing a different network from the one validation scored, which is how
    a falling loss and a collapsing mAP coexisted.
    """
    torch.manual_seed(0)
    model = _Tiny()
    image = tuples[0][0].unsqueeze(0)

    model.eval()
    with torch.no_grad():
        reference = model(image)

    model.train()
    model.apply(_freeze_batchnorm)
    with torch.no_grad():
        training_mode = model(image)

    assert torch.allclose(reference, training_mode, atol=1e-6)
