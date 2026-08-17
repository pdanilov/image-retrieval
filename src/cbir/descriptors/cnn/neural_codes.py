"""Neural Codes: activations of a frozen ImageNet CNN's first fully-connected layer.

Babenko et al., ECCV 2014 (`1404.1777`). The 2014 starting point of the CNN era and the
method AGENTS.md names as the diploma-era one: take the 4096-D output of `fc6`,
L2-normalize, PCA-compress, L2-normalize again, compare by cosine.

It has no pooling stage at all. The fully-connected layer has a fixed input size, which
is what forces every image to one resolution — and removing that constraint is exactly
what SPoC, MAC, R-MAC and GeM are for. So this class deliberately does not generalize:
the next tier replaces the FC layer with a pooled conv map rather than extending it.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import Literal

import numpy as np
import torch
from PIL.Image import Image as PILImage

from cbir.descriptors.classic.normalization import safe_l2_normalize

Backbone = Literal["alexnet"]

DIM = 4096
"""Width of `fc6` in both AlexNet and VGG — the canonical Neural Code size."""

INPUT_SIZE = 224
"""What the fully-connected layer's fixed input size forces every image down to."""

# ImageNet channel statistics the frozen weights were trained under. Skipping these
# does not fail loudly -- it just shifts every activation and quietly costs mAP.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class NeuralCodes:
    """A frozen backbone truncated at `fc6`, plus the preprocessing it expects."""

    def __init__(self, backbone: Backbone = "alexnet", *, device: str | None = None, batch_size: int = 32) -> None:
        self.backbone = backbone
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self._model = self._build(backbone).to(self.device).eval()

    @staticmethod
    def _build(backbone: Backbone) -> torch.nn.Module:
        """The classifier truncated after `fc6`'s ReLU.

        AlexNet's `classifier` is
        `[Dropout, Linear(9216, 4096), ReLU, Dropout, Linear(4096, 4096), ReLU, Linear(4096, 1000)]`,
        so `fc6` is index 1 and `[:3]` keeps it *with* its ReLU. Post-activation is what
        "neural codes" means in the paper and in every reimplementation — the raw
        pre-ReLU projection is a different, signed feature.
        """
        from torchvision.models import AlexNet_Weights, alexnet

        if backbone != "alexnet":
            raise ValueError(f"unknown backbone {backbone!r}")
        model = alexnet(weights=AlexNet_Weights.IMAGENET1K_V1)
        model.classifier = torch.nn.Sequential(*list(model.classifier.children())[:3])
        return model

    @staticmethod
    def preprocess(image: PILImage) -> torch.Tensor:
        """One image to a normalized `(3, 224, 224)` tensor.

        Resized outright rather than resize-then-center-crop, which is what the
        ImageNet classification transform does. A center crop discards the edges of
        the frame — on a retrieval benchmark that can remove the landmark entirely, and
        it would silently undo the bbx crop the query was given for exactly that reason.
        Aspect ratio is not preserved; the fixed FC input leaves no alternative.
        """
        from torchvision.transforms import functional as tv

        tensor = tv.to_tensor(image.convert("RGB").resize((INPUT_SIZE, INPUT_SIZE)))
        return tv.normalize(tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD)

    def _batches(self, images: Iterable[PILImage]) -> Iterator[torch.Tensor]:
        batch: list[torch.Tensor] = []
        for image in images:
            batch.append(self.preprocess(image))
            if len(batch) == self.batch_size:
                yield torch.stack(batch)
                batch = []
        if batch:
            yield torch.stack(batch)

    def extract(self, images: Iterable[PILImage]) -> np.ndarray:
        """`(N, 4096)` L2-normalized codes, in input order.

        `torch.inference_mode` rather than plain `no_grad`: nothing here is ever
        backpropagated, and the model is in `eval()` so AlexNet's dropout layers are
        identities — without that, every call would return different vectors.
        """
        codes = []
        with torch.inference_mode():
            for batch in self._batches(images):
                codes.append(self._model(batch.to(self.device)).float().cpu().numpy())
        if not codes:
            return np.zeros((0, DIM), dtype=np.float32)
        # L2 first, before any PCA: it is what makes the codes comparable across images
        # of different activation magnitude, and the paper's order is L2 -> PCA -> L2.
        return safe_l2_normalize(np.concatenate(codes).astype(np.float32), axis=1)
