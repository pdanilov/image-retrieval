"""The trainable network: backbone, generalized-mean pooling, L2.

`descriptors/cnn/pooling.py` has the same three stages but cannot be trained through:
it returns numpy, describes one image per call, and holds `p` as a float. This is the
same computation as an `nn.Module` with `p` as a parameter.

The backbone is built by `PooledCNN._build` rather than re-derived, so truncation and
weight provenance stay defined in exactly one place. What is *not* shared is the
pooling: a `float` and an `nn.Parameter` are different objects, and threading a
trainable scalar through the inference path would complicate the code that produces
every recorded row in order to serve the one that produces none.
"""

from __future__ import annotations

import torch

from cbir.descriptors.cnn.pooling import CHANNELS, Backbone, PooledCNN
from cbir.descriptors.cnn.weights import WeightSource

EPS = 1e-6
"""Floor before the power. Conv maps are post-ReLU so never negative, but an exact zero
raised to a fractional power has an infinite gradient — which in training is not a
cosmetic concern the way it is at inference."""


class GeM(torch.nn.Module):
    """Generalized mean over spatial positions, with a learnable exponent.

    `p` is a single scalar shared across channels, which is what makes it trainable at
    all: per-channel exponents would be 2048 parameters fitted on one scalar of signal
    each. It is stored as a raw parameter rather than through a softplus — the reference
    does the same, and clamping the input at `EPS` already keeps the power well-defined
    for any positive `p`.
    """

    def __init__(self, p: float = 3.0) -> None:
        super().__init__()
        self.p = torch.nn.Parameter(torch.tensor(float(p)))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """`(B, C, h, w)` conv map to `(B, C)`."""
        return features.clamp(min=EPS).pow(self.p).mean(dim=(2, 3)).pow(1.0 / self.p)

    def extra_repr(self) -> str:
        return f"p={self.p.item():.4f}"


class RetrievalNet(torch.nn.Module):
    """Backbone to L2-normalized global descriptor, trainable end to end."""

    def __init__(
        self,
        backbone: Backbone = "vgg16",
        *,
        p: float = 3.0,
        weights: WeightSource = "torchvision",
        last_pool: bool = False,
    ) -> None:
        super().__init__()
        # `last_pool=False` is the default here and `True` in PooledCNN: the reference
        # trains on `features[:-1]`, and a network fine-tuned with the trailing pool
        # would not be comparable to the checkpoints this package exists to reproduce.
        # Checked before `_build`, not after it: building would download and read the
        # checkpoint first, so an invalid configuration answered with "go fetch 500 MB"
        # before answering "this configuration describes nothing". It also let the guard
        # depend on a manual download, which is not a property the guard has.
        if weights == "sfm120k":
            raise ValueError("weights='sfm120k' is already fine-tuned; training starts from ImageNet")
        features, _ = PooledCNN._build(backbone, weights, last_pool)
        self.backbone = backbone
        self.features = features
        self.pool = GeM(p)
        self.dim = CHANNELS[backbone]

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """`(B, 3, h, w)` batch to `(B, C)` unit vectors."""
        pooled = self.pool(self.features(images))
        return torch.nn.functional.normalize(pooled, p=2.0, dim=1, eps=EPS)
