"""The trainable networks: a CNN with generalized-mean pooling, or a ViT read out by CLS.

`descriptors/cnn/pooling.py` has the same three stages but cannot be trained through:
it returns numpy, describes one image per call, and holds `p` as a float. This is the
same computation as an `nn.Module` with `p` as a parameter.

The backbone is built by `PooledCNN._build` rather than re-derived, so truncation and
weight provenance stay defined in exactly one place. What is *not* shared is the
pooling: a `float` and an `nn.Parameter` are different objects, and threading a
trainable scalar through the inference path would complicate the code that produces
every recorded row in order to serve the one that produces none.

The ViT is IRT's network (El-Nouby et al. 2021): the frozen tier's checkpoint, read out
by its CLS token, trained end to end. Its network and preprocessing come from
`descriptors/transformer/vit.py` rather than being re-derived, for the same reason.

Each network also owns `load`, the path-to-tensor step for its own family, so mining,
training and validation need not know which one they are driving: the CNN keeps the
aspect ratio and caps the longer side at 362, the ViT squashes into IRT's square. `load`
is a plain function rather than a method so DataLoader workers can decode with it without
holding the network, and `square` says whether its outputs can be stacked into a batch.
"""

from __future__ import annotations

import functools

import torch
from PIL import Image

from cbir.descriptors.cnn.pooling import CHANNELS, Backbone, PooledCNN
from cbir.descriptors.cnn.weights import WeightSource
from cbir.descriptors.transformer.vit import CHECKPOINT_PREFIX, SPECS, Model, Spec, load_network, preprocess
from cbir.training.tuples import IMAGE_SIZE, load_tensor

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
        image_size: int = IMAGE_SIZE,
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
        self.image_size = image_size
        # Aspect kept, longer side capped: shapes differ, so nothing stacks.
        self.load = functools.partial(load_tensor, image_size=image_size)
        self.square = False

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """`(B, 3, h, w)` batch to `(B, C)` unit vectors."""
        pooled = self.pool(self.features(images))
        return torch.nn.functional.normalize(pooled, p=2.0, dim=1, eps=EPS)


def load_square(path: str, *, spec: Spec, side: int) -> torch.Tensor:
    """One image squashed into the square, as IRT trains and evaluates."""
    with Image.open(path) as image:
        return preprocess(image, spec, side, "stretch", "bicubic")


class RetrievalViT(torch.nn.Module):
    """A pretrained ViT to L2-normalized CLS token, trainable end to end — IRT's network.

    CLS is read from `last_hidden_state`, after the final LayerNorm, exactly as the
    frozen tier reads it: DeiT's `pooler_output` is an untrained head. SigLIP 2 has no
    CLS token and is refused.

    The network is held as `network`, which is the prefix `vit.load_checkpoint` strips —
    so a checkpoint written by the loop is read back by `cbir evaluate vit --checkpoint`.
    """

    def __init__(self, model: Model, *, image_size: int | None = None) -> None:
        super().__init__()
        if model not in SPECS:
            raise ValueError(f"unknown model {model!r}, expected one of {sorted(SPECS)}")
        spec = SPECS[model]
        if spec.siglip:
            raise ValueError(f"{model} has no CLS token to train; IRT's recipe reads CLS")
        image_size = image_size if image_size is not None else (spec.native or 224)
        if spec.native is not None and image_size != spec.native:
            raise ValueError(f"{model} takes {spec.native} x {spec.native} only, got image_size={image_size}")
        if image_size % spec.patch:
            raise ValueError(f"image_size must be a multiple of the patch size {spec.patch}, got {image_size}")
        assert CHECKPOINT_PREFIX == "network."  # the attribute name below must match it
        self.network = load_network(spec)
        self.backbone = model
        self.spec = spec
        self.dim = spec.width
        self.image_size = image_size
        self.load = functools.partial(load_square, spec=spec, side=image_size)
        self.square = True

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """`(B, 3, s, s)` batch to `(B, width)` unit vectors."""
        cls = self.network(pixel_values=images).last_hidden_state[:, 0]
        return torch.nn.functional.normalize(cls, p=2.0, dim=1, eps=EPS)
