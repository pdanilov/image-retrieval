"""Frozen vision transformers, read out as one global vector each.

The transformer era's off-the-shelf baseline, and the counterpart of `cnn/pooling.py`:
a network trained for something other than retrieval, read out as a global descriptor
with no training here. What changed is the pretraining. DINOv2 (Oquab et al., TMLR
2024, `2304.07193`) is self-distilled on 142M curated images, an objective that already
pulls different views of the same content together — which is close to what the CNN
tier had to *fine-tune* its ImageNet classifiers to do. Frozen DINOv2 ViT-L/14 scores
0.729 / 0.528 on roxford5k Medium / Hard here, against 0.654 / 0.404 for Radenović et
al.'s ResNet101 GeM checkpoint fine-tuned for the task.

Four families, one readout each:

* **DINOv2** and **DINOv3** (Siméoni et al. 2025, `2508.10104`): the CLS token after
  the final LayerNorm — DINOv2's `x_norm_clstoken`. DINOv3 also carries register
  tokens; they sit after CLS and are not part of the descriptor. Not pooled patch
  tokens: the published instance-retrieval numbers are measured on CLS.
* **DeiT** (Touvron et al. 2021, `2012.12877`): the same ViT architecture trained as a
  supervised ImageNet classifier, distilled from a CNN teacher through a second token.
  The descriptor is CLS, as in IRT, which fine-tunes exactly these checkpoints; the
  distillation token beside it is not used. DeiT is the transformer counterpart of the
  CNN tier's ImageNet ResNets: the "before" row for retrieval fine-tuning.
* **SigLIP 2** (Tschannen et al. 2025, `2502.14786`): there is no CLS token. The vision
  tower ends in an attention-pooling head, and its output is the image embedding the
  model matches against text. It is here as the image-text contrast — trained to say
  *what* a picture shows, not *which* building it is.

Preprocessing follows the one protocol either DINO paper documents (DINOv3 App. D.8):
resize so the *longer* side is `image_size`, keeping the aspect ratio, then "take a full
center crop" to `image_size` x `image_size`. A center crop larger than the image pads it,
so the shorter side ends up framed by black bars rather than cut — the reading taken
here, since cropping a 224-px-long image to 224 cannot remove anything else. Every family
gets the same geometry and its own normalization, so a gap between families is the
network's, not the preprocessing's.

The DINOv2 paper's own table is not reproducible under this benchmark's rules, and the
reason was found rather than guessed: it matches, within 0.1-1.6 mAP, what the DINO
group's public evaluation script measures — and that script never crops queries to their
`bbx`. Cropped, as the benchmark requires, small models lose up to 11 points on roxford
and large ones about one. `Resize` lists the preprocessing alternatives that were ruled
out on the way; none moved ViT-S toward the table.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np
import torch
from PIL import Image
from PIL.Image import Image as PILImage

from cbir.descriptors.classic.normalization import safe_l2_normalize
from cbir.descriptors.cnn.neural_codes import IMAGENET_MEAN, IMAGENET_STD

if TYPE_CHECKING:
    from transformers.modeling_outputs import BaseModelOutputWithPooling

Model = Literal[
    "dinov2-vits14",
    "dinov2-vitb14",
    "dinov2-vitl14",
    "dinov2-vitg14",
    "dinov3-vits16",
    "dinov3-vitb16",
    "dinov3-vitl16",
    "siglip2-vitl16-256",
    "siglip2-vitl16-384",
    "siglip2-vitl16-512",
    "deit-vits16-dist-224",
    "deit-vitb16-dist-224",
    "deit-vitb16-dist-384",
]

SIGLIP_MEAN = (0.5, 0.5, 0.5)
SIGLIP_STD = (0.5, 0.5, 0.5)


@dataclass(frozen=True)
class Spec:
    """What one checkpoint needs to be loaded, fed and read out.

    Args:
        repo: Hugging Face checkpoint.
        width: Hidden size, and therefore the descriptor's width.
        patch: Patch size. An input side that is not a multiple of it is not an error to
            the patch embedding — a strided convolution just drops the remainder — so the
            last rows and columns of every image would be silently ignored. Rejected.
        mean: Per-channel normalization mean the network was trained under.
        std: Per-channel normalization std.
        native: The only input size the checkpoint accepts, for fixed-resolution
            checkpoints. SigLIP 2 ships one checkpoint per resolution, with position
            embeddings learned for that grid; running it at another size means
            interpolating them, which is a different network from the one released.
        siglip: Load only the vision tower of an image-text checkpoint, and read out its
            attention-pooling head rather than a CLS token.
        prefix: Tokens ahead of the patches in the output sequence, which a pooled
            readout must skip: CLS (DINOv2), CLS + 4 registers (DINOv3), CLS +
            distillation (DeiT†), none (SigLIP). Checked against the patch count on every
            batch, since a wrong value would fold CLS or a register into the mean without
            changing its shape.
    """

    repo: str
    width: int
    patch: int
    prefix: int
    mean: tuple[float, float, float] = IMAGENET_MEAN
    std: tuple[float, float, float] = IMAGENET_STD
    native: int | None = None
    siglip: bool = False


SPECS: dict[str, Spec] = {
    # The original DINOv2 releases, without registers: those are the networks the
    # published rows were measured on. All Apache-2.0 and ungated.
    "dinov2-vits14": Spec("facebook/dinov2-small", 384, 14, prefix=1),
    "dinov2-vitb14": Spec("facebook/dinov2-base", 768, 14, prefix=1),
    "dinov2-vitl14": Spec("facebook/dinov2-large", 1024, 14, prefix=1),
    "dinov2-vitg14": Spec("facebook/dinov2-giant", 1536, 14, prefix=1),
    # Distilled from the 7B teacher on LVD-1689M. **Gated**: accept the license on the
    # model page before the first download, or loading fails with a 403. RoPE position
    # encoding, so any multiple of 16 is a native input size.
    "dinov3-vits16": Spec("facebook/dinov3-vits16-pretrain-lvd1689m", 384, 16, prefix=5),
    "dinov3-vitb16": Spec("facebook/dinov3-vitb16-pretrain-lvd1689m", 768, 16, prefix=5),
    "dinov3-vitl16": Spec("facebook/dinov3-vitl16-pretrain-lvd1689m", 1024, 16, prefix=5),
    "siglip2-vitl16-256": Spec(
        "google/siglip2-large-patch16-256",
        1024,
        16,
        prefix=0,
        mean=SIGLIP_MEAN,
        std=SIGLIP_STD,
        native=256,
        siglip=True,
    ),
    "siglip2-vitl16-384": Spec(
        "google/siglip2-large-patch16-384",
        1024,
        16,
        prefix=0,
        mean=SIGLIP_MEAN,
        std=SIGLIP_STD,
        native=384,
        siglip=True,
    ),
    "siglip2-vitl16-512": Spec(
        "google/siglip2-large-patch16-512",
        1024,
        16,
        prefix=0,
        mean=SIGLIP_MEAN,
        std=SIGLIP_STD,
        native=512,
        siglip=True,
    ),
    # The distilled (†) DeiT checkpoints IRT starts from. Learned absolute position
    # embeddings for one grid each, so like SigLIP 2 they take only their native size.
    "deit-vits16-dist-224": Spec("facebook/deit-small-distilled-patch16-224", 384, 16, prefix=2, native=224),
    "deit-vitb16-dist-224": Spec("facebook/deit-base-distilled-patch16-224", 768, 16, prefix=2, native=224),
    "deit-vitb16-dist-384": Spec("facebook/deit-base-distilled-patch16-384", 768, 16, prefix=2, native=384),
}

UNUSED_HEADS = ("pooler.",)
"""Weights a checkpoint may legitimately lack. Hugging Face's ViT-style models build a
`pooler` (a dense layer and tanh over CLS) that classification checkpoints such as DeiT's
never trained; it is created randomly at load, and nothing here reads it. Any *other*
missing weight means part of the network is random, so loading fails."""

IMAGE_SIZE = 224
"""The published protocol's input size."""

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

Readout = Literal["native", "mean", "gem"]
"""Which vector of the output sequence becomes the descriptor:

* `native` — each family's own image embedding: CLS after the final LayerNorm
  (DINOv2/DINOv3/DeiT), the attention-pooling head (SigLIP 2). Every published row.
* `mean` — the average of the patch tokens, prefix tokens excluded. The patch grid is
  the transformer's counterpart of a conv map, so this is SPoC over tokens.
* `gem` — generalized mean over the patch tokens, GeM over tokens. The tokens are
  LayerNorm outputs and *signed*, unlike a post-ReLU conv map, and GeM's clamp to a
  small positive floor discards every negative component before the power — so `gem`
  is not `mean` at `p = 1`, and `mean` is the honest first comparison."""

EPS = 1e-6
"""GeM's floor before the power, as in the CNN tier."""

RESAMPLING = {"bicubic": Image.Resampling.BICUBIC, "bilinear": Image.Resampling.BILINEAR}


class FrozenViT:
    """A frozen ViT, the preprocessing its retrieval rows assume, and its global readout."""

    def __init__(
        self,
        model: Model = "dinov2-vitl14",
        *,
        image_size: int = IMAGE_SIZE,
        resize: Resize = "pad",
        interpolation: Interpolation = "bicubic",
        readout: Readout = "native",
        p: float | None = None,
        device: str | None = None,
        batch_size: int = 64,
    ) -> None:
        if model not in SPECS:
            raise ValueError(f"unknown model {model!r}, expected one of {sorted(SPECS)}")
        spec = SPECS[model]
        if resize not in ("pad", "aspect", "crop"):
            raise ValueError(f"unknown resize {resize!r}, expected pad, aspect or crop")
        if interpolation not in RESAMPLING:
            raise ValueError(f"unknown interpolation {interpolation!r}, expected one of {sorted(RESAMPLING)}")
        if readout not in ("native", "mean", "gem"):
            raise ValueError(f"unknown readout {readout!r}, expected native, mean or gem")
        if (readout == "gem") != (p is not None):
            raise ValueError(f"p is GeM's exponent: required for readout='gem' and only for it, got p={p!r}")
        if p is not None and p <= 0:
            raise ValueError(f"p must be positive, got {p}")
        if image_size % spec.patch:
            raise ValueError(f"image_size must be a multiple of the patch size {spec.patch}, got {image_size}")
        if spec.native is not None and (image_size != spec.native or resize == "aspect"):
            raise ValueError(
                f"{model} is a fixed-resolution checkpoint: it takes {spec.native} x {spec.native} only, "
                f"so image_size must be {spec.native} and resize cannot be 'aspect'. For another "
                "resolution, pick the checkpoint released at that size."
            )
        self.model = model
        self.spec = spec
        self.image_size = image_size
        self.resize = resize
        self.interpolation = interpolation
        self.readout = readout
        self.p = p
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size

        # Imported here, not at module scope: transformers is a multi-second import that
        # nothing but this class needs, and `cbir results` should not pay for it.
        if spec.siglip:
            # The vision tower alone. `AutoModel` would load the text tower too, which
            # nothing here calls, and its forward requires text inputs.
            from transformers import SiglipVisionModel as Loader
        else:
            from transformers import AutoModel as Loader

        network, info = Loader.from_pretrained(spec.repo, output_loading_info=True)
        missing = _unexpectedly_missing(info["missing_keys"])
        if missing:
            raise RuntimeError(f"{spec.repo} left {len(missing)} weights uninitialized, e.g. {sorted(missing)[:3]}")
        self._model = network.to(self.device).eval()

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
                patch = self.spec.patch
                size = (_patches(image.width * factor, patch), _patches(image.height * factor, patch))
                image = image.resize(size, resampling)
            case "crop":
                image = _scale(image, round(side * 256 / 224) / min(image.size), resampling)
                image = tv.center_crop(image, [side, side])
        return tv.normalize(tv.to_tensor(image), mean=self.spec.mean, std=self.spec.std)

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

    def _read_out(self, output: BaseModelOutputWithPooling, batch: torch.Tensor) -> torch.Tensor:
        """`(B, width)` descriptors from one batch's model output, per `self.readout`."""
        if self.readout == "native":
            # Not `pooler_output` across the board: for DeiT that is the untrained tanh
            # head (see `UNUSED_HEADS`), and for DINO it equals CLS anyway.
            return output.pooler_output if self.spec.siglip else output.last_hidden_state[:, 0]
        height, width = batch.shape[-2:]
        patches = (height // self.spec.patch) * (width // self.spec.patch)
        tokens = _patch_tokens(output.last_hidden_state, self.spec.prefix, patches)
        if self.readout == "mean":
            return tokens.mean(dim=1)
        assert self.p is not None  # guaranteed by __init__
        return tokens.clamp(min=EPS).pow(self.p).mean(dim=1).pow(1.0 / self.p)

    def extract(self, images: Iterable[PILImage]) -> np.ndarray:
        """`(N, width)` L2-normalized descriptors, in input order.

        Batched, unlike the CNN tier: the square modes give every input the same size,
        so stacking needs no padding beyond what the protocol already applies. `aspect`
        inputs batch only while consecutive shapes agree. Float32 throughout — these
        rows are compared to published numbers, and half precision is a second variable
        a mismatch would then have to rule out.
        """
        vectors = []
        with torch.inference_mode():
            for batch in self._batches(images):
                output = self._model(pixel_values=batch.to(self.device))
                vectors.append(self._read_out(output, batch).float().cpu().numpy())
        if not vectors:
            return np.zeros((0, self.spec.width), dtype=np.float32)
        return safe_l2_normalize(np.concatenate(vectors).astype(np.float32), axis=1)


def _patch_tokens(hidden: torch.Tensor, prefix: int, patches: int) -> torch.Tensor:
    """The patch tokens of a `(B, tokens, D)` sequence, after `prefix` leading ones.

    Raises if what is left is not exactly `patches` tokens: a wrong prefix would fold CLS
    or a register into a pooled descriptor while leaving its shape unchanged.
    """
    tokens = hidden[:, prefix:]
    if tokens.shape[1] != patches:
        raise RuntimeError(f"expected {patches} patch tokens after {prefix} prefix tokens, got {tokens.shape[1]}")
    return tokens


def _unexpectedly_missing(missing_keys: Iterable[str]) -> set[str]:
    """Missing checkpoint weights other than the heads nothing reads."""
    return {key for key in missing_keys if not key.startswith(UNUSED_HEADS)}


def _scale(image: PILImage, factor: float, resampling: Image.Resampling) -> PILImage:
    size = (max(1, round(image.width * factor)), max(1, round(image.height * factor)))
    return image.resize(size, resampling)


def _patches(length: float, patch: int) -> int:
    """`length` rounded to a whole number of patches, at least one."""
    return max(patch, round(length / patch) * patch)
