"""Alternative ImageNet weights: the Caffe-converted ones the reference actually uses.

`cnnimageretrieval-pytorch` builds a torchvision architecture but loads its own
Caffe-converted ImageNet weights into it, not torchvision's. They are different
networks numerically — conv1 alone has cosine 0.008 against torchvision's — so an
off-the-shelf row measured on one is not the same experiment as on the other.

That matters here because our ResNet101 rows miss the published ones by 7-15 mAP while
VGG reproduces, and weight provenance is the difference we had not tested.

Manual download, like the SfM corpus. The files omit `num_batches_tracked`, which the
architecture only uses to update running statistics during training; loading is
therefore non-strict, but *only* those buffers may be absent.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import torch

WeightSource = Literal["torchvision", "caffe", "sfm120k"]
"""Where the architecture's weights come from.

`torchvision` and `caffe` are both frozen ImageNet classifiers, differing only in whose
training run produced them. `sfm120k` is a different kind of thing: fine-tuned for
retrieval itself, and carrying its own pooling exponent and whitening — see
`finetuned.py`."""

CAFFE_URL = "https://cmp.felk.cvut.cz/cnnimageretrieval/data/networks/imagenet"
CAFFE_FILES = {
    "vgg16": "imagenet-caffe-vgg16-features-d369c8e.pth",
    "resnet50": "imagenet-caffe-resnet50-features-ac468af.pth",
    "resnet101": "imagenet-caffe-resnet101-features-10a101d.pth",
    "resnet152": "imagenet-caffe-resnet152-features-1011020.pth",
}


def root() -> Path:
    """Where the weight files live; `CBIR_WEIGHTS_ROOT` overrides."""
    return Path(os.environ.get("CBIR_WEIGHTS_ROOT", Path.home() / ".cache" / "cbir" / "weights"))


def load_caffe(model: torch.nn.Module, backbone: str) -> torch.nn.Module:
    """Replace `model`'s weights in place with the Caffe-converted ImageNet ones."""
    if backbone not in CAFFE_FILES:
        raise ValueError(f"no caffe weights published for {backbone!r}; have {sorted(CAFFE_FILES)}")

    path = root() / CAFFE_FILES[backbone]
    if not path.exists():
        raise FileNotFoundError(f"{path} not found — download it from {CAFFE_URL}/{CAFFE_FILES[backbone]}")

    state = torch.load(path, map_location="cpu", weights_only=True)
    return apply_state(model, state, source=path.name, backbone=backbone)


def apply_state(
    model: torch.nn.Module, state: dict[str, torch.Tensor], *, source: str, backbone: str
) -> torch.nn.Module:
    """Load `state` into `model`, tolerating only the buffers these files never carry.

    Both the Caffe and the fine-tuned checkpoints omit `num_batches_tracked`, which only
    ever updates running statistics during training. Anything *else* absent would mean
    the file does not describe this architecture, and the run would otherwise proceed on
    a half-initialized network without failing — which is the one failure mode that
    produces a plausible-looking wrong number.
    """
    missing, unexpected = model.load_state_dict(state, strict=False)
    unknown = [k for k in missing if not k.endswith("num_batches_tracked")]
    if unknown or unexpected:
        raise ValueError(f"{source} does not match {backbone}: missing {unknown[:3]}, unexpected {unexpected[:3]}")
    return model
