"""Frozen vision transformers, described by their CLS token.

The transformer era's off-the-shelf baseline, and the counterpart of `cnn/pooling.py`:
a network trained for something other than retrieval, read out as a global descriptor
with no training here. What changed is the pretraining. DINOv2 (Oquab et al., TMLR
2024, `2304.07193`) is self-distilled on 142M curated images, an objective that already
pulls different views of the same content together — which is close to what the CNN
tier had to *fine-tune* its ImageNet classifiers to do. Frozen DINOv2 ViT-L/14 scores
0.729 / 0.528 on roxford5k Medium / Hard here, against 0.654 / 0.404 for the GeM
ResNet101 this project fine-tuned for the task.

The descriptor is the CLS token after the final LayerNorm — DINOv2's `x_norm_clstoken`,
which is what Hugging Face exposes as `pooler_output`. Not pooled patch tokens: the
published instance-retrieval numbers are measured on CLS, so CLS is what has to be
reproduced before any other readout is worth comparing to it.

Preprocessing follows the one protocol either paper documents (DINOv3, Siméoni et al.
2025, `2508.10104`, App. D.8): resize so the *longer* side is `image_size`, keeping the
aspect ratio, then "take a full center crop" to `image_size` x `image_size`. A center
crop larger than the image pads it, so the shorter side ends up framed by black bars
rather than cut — the reading taken here, since cropping a 224-px-long image to 224
cannot remove anything else.

The DINOv2 paper's own table is not reproducible under this benchmark's rules, and the
reason was found rather than guessed: it matches, within 0.1-1.6 mAP, what the DINO
group's public evaluation script measures — and that script never crops queries to their
`bbx`. Cropped, as the benchmark requires, small models lose up to 11 points on roxford
and large ones about one. `Resize` lists the preprocessing alternatives that were ruled
out on the way; none moved ViT-S toward the table.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import Literal

import numpy as np
import torch
from PIL import Image
from PIL.Image import Image as PILImage

from cbir.descriptors.classic.normalization import safe_l2_normalize
from cbir.descriptors.cnn.neural_codes import IMAGENET_MEAN, IMAGENET_STD

Model = Literal["dinov2-vits14", "dinov2-vitb14", "dinov2-vitl14", "dinov2-vitg14"]

REPOS: dict[str, str] = {
    "dinov2-vits14": "facebook/dinov2-small",
    "dinov2-vitb14": "facebook/dinov2-base",
    "dinov2-vitl14": "facebook/dinov2-large",
    "dinov2-vitg14": "facebook/dinov2-giant",
}
"""Hugging Face checkpoints, all Apache-2.0 and ungated. The original DINOv2 releases
without registers: those are the networks the published rows were measured on, so a
registers variant is an ablation to add after these reproduce, not a substitute."""

WIDTH: dict[str, int] = {
    "dinov2-vits14": 384,
    "dinov2-vitb14": 768,
    "dinov2-vitl14": 1024,
    "dinov2-vitg14": 1536,
}
"""Hidden size of each model, and therefore the CLS descriptor's width."""

PATCH = 14
"""DINOv2's patch size. An input side that is not a multiple of it is not an error to
the patch embedding — a stride-14 convolution just drops the remainder — so the last
few rows and columns of every image would be silently ignored. Rejected instead."""

IMAGE_SIZE = 224
"""The published protocol's input size: 16 x 16 patches."""

Resize = Literal["pad", "aspect", "crop"]
"""How an image becomes the network's input — a part of the protocol the DINOv2 paper
leaves unstated, and the first suspect when its rows did not reproduce. None of the three
explained that gap (the query crop did; see the module docstring):

* `pad` — DINOv3 App. D.8: longer side to `image_size`, centred on a black square.
* `aspect` — longer side to `image_size`, aspect ratio kept and no padding; each side
  rounded to a multiple of the patch. DINOv2 interpolates its position embeddings, so a
  non-square input is legitimate, and no patch is spent on black bars.
* `crop` — DINOv2's own classification eval transform: shorter side to
  `image_size * 256 / 224`, then a centre crop to `image_size`. It cuts the frame's
  edges, which on a retrieval benchmark can cut the landmark; recorded as a candidate
  for what the paper did, not as a recommendation."""

Interpolation = Literal["bicubic", "bilinear"]

RESAMPLING = {"bicubic": Image.Resampling.BICUBIC, "bilinear": Image.Resampling.BILINEAR}


class FrozenViT:
    """A frozen ViT, the preprocessing its retrieval rows assume, and its CLS readout."""

    def __init__(
        self,
        model: Model = "dinov2-vitl14",
        *,
        image_size: int = IMAGE_SIZE,
        resize: Resize = "pad",
        interpolation: Interpolation = "bicubic",
        device: str | None = None,
        batch_size: int = 64,
    ) -> None:
        if model not in REPOS:
            raise ValueError(f"unknown model {model!r}, expected one of {sorted(REPOS)}")
        if resize not in ("pad", "aspect", "crop"):
            raise ValueError(f"unknown resize {resize!r}, expected pad, aspect or crop")
        if interpolation not in RESAMPLING:
            raise ValueError(f"unknown interpolation {interpolation!r}, expected one of {sorted(RESAMPLING)}")
        if image_size % PATCH:
            raise ValueError(f"image_size must be a multiple of the patch size {PATCH}, got {image_size}")
        self.model = model
        self.image_size = image_size
        self.resize = resize
        self.interpolation = interpolation
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size

        # Imported here, not at module scope: transformers is a multi-second import that
        # nothing but this class needs, and `cbir results` should not pay for it.
        from transformers import AutoModel

        self._model = AutoModel.from_pretrained(REPOS[model]).to(self.device).eval()

    def preprocess(self, image: PILImage) -> torch.Tensor:
        """One image to a normalized `(3, h, w)` tensor, shaped as `self.resize` says.

        `pad` and `crop` give `image_size` x `image_size`; `aspect` gives a rectangle
        whose longer side is `image_size`. Every mode resizes up as well as down — the
        protocol fixes the side rather than capping it, so a small query crop is
        enlarged.
        """
        from torchvision.transforms import functional as tv

        image = image.convert("RGB")
        resampling = RESAMPLING[self.interpolation]
        side = self.image_size
        match self.resize:
            case "pad":
                image = _scale(image, side / max(image.size), resampling)
                # torchvision pads a crop larger than the image with zeros and centres
                # it, which is what "take a full center crop" of a 224-px-long image can
                # only mean.
                image = tv.center_crop(image, [side, side])
            case "aspect":
                factor = side / max(image.size)
                size = (_patches(image.width * factor), _patches(image.height * factor))
                image = image.resize(size, resampling)
            case "crop":
                image = _scale(image, round(side * 256 / 224) / min(image.size), resampling)
                image = tv.center_crop(image, [side, side])
        return tv.normalize(tv.to_tensor(image), mean=IMAGENET_MEAN, std=IMAGENET_STD)

    def _batches(self, images: Iterable[PILImage]) -> Iterator[torch.Tensor]:
        batch: list[torch.Tensor] = []
        for image in images:
            tensor = self.preprocess(image)
            # `aspect` inputs differ in shape and cannot be stacked: a shape change ends
            # the batch. The square modes never trigger it.
            if batch and tensor.shape != batch[0].shape:
                yield torch.stack(batch)
                batch = []
            batch.append(tensor)
            if len(batch) == self.batch_size:
                yield torch.stack(batch)
                batch = []
        if batch:
            yield torch.stack(batch)

    def extract(self, images: Iterable[PILImage]) -> np.ndarray:
        """`(N, width)` L2-normalized CLS descriptors, in input order.

        Batched, unlike the CNN tier: the square modes give every input the same size,
        so stacking needs no padding beyond what the protocol already applies. `aspect`
        inputs batch only while consecutive shapes agree. Float32
        throughout — these rows are compared to published numbers, and half precision
        is a second variable a mismatch would then have to rule out.
        """
        vectors = []
        with torch.inference_mode():
            for batch in self._batches(images):
                output = self._model(pixel_values=batch.to(self.device))
                vectors.append(output.pooler_output.float().cpu().numpy())
        if not vectors:
            return np.zeros((0, WIDTH[self.model]), dtype=np.float32)
        return safe_l2_normalize(np.concatenate(vectors).astype(np.float32), axis=1)


def _scale(image: PILImage, factor: float, resampling: Image.Resampling) -> PILImage:
    size = (max(1, round(image.width * factor)), max(1, round(image.height * factor)))
    return image.resize(size, resampling)


def _patches(length: float) -> int:
    """`length` rounded to a whole number of patches, at least one."""
    return max(PATCH, round(length / PATCH) * PATCH)
