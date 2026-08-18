"""Global descriptors by pooling a frozen CNN's last convolutional feature map.

The whole CNN era after Neural Codes is this one idea: drop the classifier, read the
last conv map `(C, h, w)` as `h*w` local descriptors of width `C`, and pool them into a
single `C`-vector. Dropping the fully-connected layer removes the fixed input size with
it, so images keep their aspect ratio and their detail, and pooling over positions makes
the descriptor translation-invariant — the two things `neural_codes.py` cannot do.

SPoC, MAC and GeM differ *only* in the pooling, and generalized mean subsumes all three:

    f_k = (mean_x x_k^p)^(1/p)      p = 1 is SPoC (average);  p -> inf is MAC (max)

So there is one module with `p` as a knob rather than three classes (AGENTS.md). `p=None`
selects true max rather than a large finite `p`, which would overflow float32 long before
it converged on the maximum. Only R-MAC needs separate code, for its region grid.

Multi-scale is the second knob. The published off-the-shelf numbers are measured over
several input resolutions per image, the per-scale descriptors L2-normalized and averaged
into one vector of the same width. A conv stack has a fixed receptive field, so the only
way it sees an object at another size is to be shown the image at another size; averaging
across scales is what makes the descriptor tolerate the scale change between a query crop
and the same landmark in a database photo.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Literal

import numpy as np
import torch
from PIL.Image import Image as PILImage

from cbir.descriptors.classic.normalization import safe_l2_normalize
from cbir.descriptors.cnn.neural_codes import IMAGENET_MEAN, IMAGENET_STD

Backbone = Literal["alexnet", "vgg16", "vgg19", "resnet18", "resnet34", "resnet50", "resnet101"]

CHANNELS: dict[str, int] = {
    "alexnet": 256,
    "vgg16": 512,
    "vgg19": 512,
    "resnet18": 512,
    "resnet34": 512,
    "resnet50": 2048,
    "resnet101": 2048,
}
"""Width of each backbone's last conv map — and therefore of the descriptor.

Note that depth and width are separate axes: resnet18 is deeper than vgg16 but produces
the same 512-D descriptor, and resnet50 matches resnet101's 2048. Comparing across
backbones therefore compares two things at once unless the width is held fixed."""

EPS = 1e-6
"""Floor before the power in GeM. Conv maps are post-ReLU so never negative, but an
exact zero raised to a fractional power has an infinite gradient and yields NaN under
`1/p` on some backends."""

MULTI_SCALE: tuple[float, ...] = (1.0, 2**-0.5, 0.5)
"""The scale set the off-the-shelf CNN baselines are published at: full, 1/sqrt(2), 1/2."""

MIN_SIDE: dict[str, int] = {
    "alexnet": 63,
    "vgg16": 32,
    "vgg19": 32,
    "resnet18": 32,
    "resnet34": 32,
    "resnet50": 32,
    "resnet101": 32,
}
"""Smallest input each conv stack accepts, below which its own pooling layers raise.

Measured, not assumed: AlexNet's stride-4 first conv plus three max-pools exhaust a
62-px side. Query crops here go down to 131 px on the long side and are far narrower on
the short one, so a 1/2-scale pass lands underneath this without a floor.

The ResNets technically survive down to 8 px — adaptive pooling does not fail the way a
fixed kernel does — but 32 is kept as the floor rather than their true minimum: below it
the last conv map is a single cell, which makes pooling and the R-MAC region grid
degenerate without raising."""


class PooledCNN:
    """A frozen backbone truncated to its conv stack, plus a pooling exponent."""

    def __init__(
        self,
        backbone: Backbone = "alexnet",
        *,
        p: float | None = 3.0,
        max_side: int = 1024,
        scales: tuple[float, ...] = (1.0,),
        device: str | None = None,
    ) -> None:
        if not scales:
            raise ValueError("scales must not be empty")
        if any(s <= 0 for s in scales):
            raise ValueError(f"scales must all be positive, got {scales}")
        self.backbone = backbone
        self.p = p
        self.max_side = max_side
        self.scales = scales
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model = self._build(backbone).to(self.device).eval()

    @staticmethod
    def _build(backbone: Backbone) -> torch.nn.Module:
        """Everything up to and including the last conv block, classifier discarded."""
        import torchvision.models as tv

        match backbone:
            case "alexnet":
                return tv.alexnet(weights=tv.AlexNet_Weights.IMAGENET1K_V1).features
            case "vgg16" | "vgg19":
                builder = getattr(tv, backbone)
                return builder(weights=getattr(tv, f"{backbone.upper()}_Weights").IMAGENET1K_V1).features
            case "resnet18" | "resnet34" | "resnet50" | "resnet101":
                # Same surgery for every depth; only the constructor and weights differ.
                builder = getattr(tv, backbone)
                weights = getattr(tv, f"ResNet{backbone.removeprefix('resnet')}_Weights").IMAGENET1K_V1
                model = builder(weights=weights)
                # Drop avgpool and fc; children()[:-2] ends at layer4's output.
                return torch.nn.Sequential(*list(model.children())[:-2])
            case _:
                raise ValueError(f"unknown backbone {backbone!r}")

    def preprocess(self, image: PILImage, scale: float = 1.0) -> torch.Tensor:
        """Normalized `(1, 3, h, w)` tensor, aspect ratio preserved.

        Caps the longest side at `max_side` and does not enlarge to reach it. Upscaling
        adds no information, and query crops here go down to 131 px — interpolating one
        8x would only cost compute.

        The one exception is `MIN_SIDE`: a conv stack accepts almost any size, but not
        any size. A narrow crop at scale 1/2 lands under what the pooling layers can
        consume and the forward pass raises, so the shorter side is floored there. That
        only ever triggers below full scale — an image that failed the floor at scale 1
        would have raised in the single-scale runs already recorded, so those still
        reproduce exactly.

        `scale` multiplies the capped size, which is how multi-scale extraction asks
        for the same image at a lower resolution.
        """
        from torchvision.transforms import functional as tv

        image = image.convert("RGB")
        factor = min(1.0, self.max_side / max(image.size)) * scale
        floor = MIN_SIDE[self.backbone] / min(image.size)
        factor = max(factor, floor)
        if factor != 1.0:
            image = image.resize((max(1, round(image.width * factor)), max(1, round(image.height * factor))))
        tensor = tv.normalize(tv.to_tensor(image), mean=IMAGENET_MEAN, std=IMAGENET_STD)
        return tensor.unsqueeze(0)

    def pool(self, features: torch.Tensor) -> torch.Tensor:
        """`(B, C, h, w)` conv map to a `(B, C)` vector."""
        if self.p is None:
            return features.amax(dim=(2, 3))
        return features.clamp(min=EPS).pow(self.p).mean(dim=(2, 3)).pow(1.0 / self.p)

    def describe(self, image: PILImage) -> np.ndarray:
        """One `(1, C)` descriptor combining every scale in `self.scales`.

        Each scale's vector is L2-normalized *before* combining. Without that the
        largest scale dominates: it pools over more positions, so its raw magnitude is
        the biggest, and the combination would be it plus a nudge.

        The combination follows the same rule as the reference table (Radenović et al.,
        §"Multi-scale"): plain averaging for every method **except** GeM, which reuses
        its own generalized mean across scales as well as across positions. Getting this
        wrong is invisible for MAC and costs several mAP for GeM, which is precisely the
        discrepancy it was found by.
        """
        pooled = []
        for scale in self.scales:
            features = self._model(self.preprocess(image, scale).to(self.device))
            vector = self.pool(features).float().cpu().numpy()
            pooled.append(safe_l2_normalize(vector, axis=1))

        # Short-circuited rather than left to the general path: for one scale the
        # generalized mean is an identity only up to float error, and single-scale rows
        # already recorded must reproduce exactly.
        if len(pooled) == 1:
            return pooled[0]
        stacked = np.stack(pooled)
        if self.p is None:
            return stacked.mean(axis=0)
        return (np.maximum(stacked, EPS) ** self.p).mean(axis=0) ** (1.0 / self.p)

    def extract(self, images: Iterable[PILImage]) -> np.ndarray:
        """`(N, C)` L2-normalized descriptors, in input order.

        One image per forward pass: sizes differ after the cap, so they cannot be
        stacked into a batch without padding, and padding would feed zeros into the
        pooling. At these dataset sizes the GPU is not the bottleneck anyway — JPEG
        decoding is.
        """
        with torch.inference_mode():
            vectors = [self.describe(image) for image in images]
        if not vectors:
            return np.zeros((0, CHANNELS[self.backbone]), dtype=np.float32)
        return safe_l2_normalize(np.concatenate(vectors).astype(np.float32), axis=1)
