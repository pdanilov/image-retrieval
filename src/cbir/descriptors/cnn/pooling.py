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
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Literal

import numpy as np
import torch
from PIL.Image import Image as PILImage

from cbir.descriptors.classic.normalization import safe_l2_normalize
from cbir.descriptors.cnn.neural_codes import IMAGENET_MEAN, IMAGENET_STD

Backbone = Literal["alexnet", "vgg16", "resnet101"]

CHANNELS: dict[str, int] = {"alexnet": 256, "vgg16": 512, "resnet101": 2048}
"""Width of each backbone's last conv map — and therefore of the descriptor."""

EPS = 1e-6
"""Floor before the power in GeM. Conv maps are post-ReLU so never negative, but an
exact zero raised to a fractional power has an infinite gradient and yields NaN under
`1/p` on some backends."""


class PooledCNN:
    """A frozen backbone truncated to its conv stack, plus a pooling exponent."""

    def __init__(
        self,
        backbone: Backbone = "alexnet",
        *,
        p: float | None = 3.0,
        max_side: int = 1024,
        device: str | None = None,
    ) -> None:
        self.backbone = backbone
        self.p = p
        self.max_side = max_side
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model = self._build(backbone).to(self.device).eval()

    @staticmethod
    def _build(backbone: Backbone) -> torch.nn.Module:
        """Everything up to and including the last conv block, classifier discarded."""
        import torchvision.models as tv

        match backbone:
            case "alexnet":
                return tv.alexnet(weights=tv.AlexNet_Weights.IMAGENET1K_V1).features
            case "vgg16":
                return tv.vgg16(weights=tv.VGG16_Weights.IMAGENET1K_V1).features
            case "resnet101":
                model = tv.resnet101(weights=tv.ResNet101_Weights.IMAGENET1K_V1)
                # Drop avgpool and fc; children()[:-2] ends at layer4's output.
                return torch.nn.Sequential(*list(model.children())[:-2])
            case _:
                raise ValueError(f"unknown backbone {backbone!r}")

    def preprocess(self, image: PILImage) -> torch.Tensor:
        """Normalized `(1, 3, h, w)` tensor, aspect ratio preserved.

        Caps the longest side at `max_side` and never enlarges. Upscaling adds no
        information, and query crops here go down to 131 px — interpolating one 8x
        would only cost compute. A conv stack accepts whatever size it is given, so
        unlike `neural_codes.py` there is nothing forcing a fixed resolution.
        """
        from torchvision.transforms import functional as tv

        image = image.convert("RGB")
        scale = min(1.0, self.max_side / max(image.size))
        if scale < 1.0:
            image = image.resize((max(1, round(image.width * scale)), max(1, round(image.height * scale))))
        tensor = tv.normalize(tv.to_tensor(image), mean=IMAGENET_MEAN, std=IMAGENET_STD)
        return tensor.unsqueeze(0)

    def pool(self, features: torch.Tensor) -> torch.Tensor:
        """`(B, C, h, w)` conv map to a `(B, C)` vector."""
        if self.p is None:
            return features.amax(dim=(2, 3))
        return features.clamp(min=EPS).pow(self.p).mean(dim=(2, 3)).pow(1.0 / self.p)

    def extract(self, images: Iterable[PILImage]) -> np.ndarray:
        """`(N, C)` L2-normalized descriptors, in input order.

        One image per forward pass: sizes differ after the cap, so they cannot be
        stacked into a batch without padding, and padding would feed zeros into the
        pooling. At these dataset sizes the GPU is not the bottleneck anyway — JPEG
        decoding is.
        """
        vectors = []
        with torch.inference_mode():
            for image in images:
                features = self._model(self.preprocess(image).to(self.device))
                vectors.append(self.pool(features).float().cpu().numpy())
        if not vectors:
            return np.zeros((0, CHANNELS[self.backbone]), dtype=np.float32)
        return safe_l2_normalize(np.concatenate(vectors).astype(np.float32), axis=1)
