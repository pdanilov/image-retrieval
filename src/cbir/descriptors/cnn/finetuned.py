"""Checkpoints fine-tuned for retrieval, rather than borrowed from classification.

Every other backbone in this tier is a frozen ImageNet classifier: it was optimized to
tell a golden retriever from a poodle, and retrieval works only because those features
happen to transfer. These checkpoints were optimized for the task itself — the same
architectures, trained on retrieval-SfM-120k with a contrastive loss over matching and
non-matching landmark pairs mined by structure-from-motion (Radenović et al., TPAMI
2018). That is the difference between 40-46 mAP and the published 61.9 / 64.7.

Fine-tuning happened on a third landmark corpus with Oxford/Paris overlaps removed, so
nothing here was fitted on what it searches — the same rule the vocabularies and the
whitening follow.

A checkpoint carries three things, not one:

  * `features.*` — the fine-tuned conv stack. For VGG the names are already torchvision's
    once the prefix is stripped; ResNet is stored flat and needs the mapping below.
  * `pool.p` — the generalized-mean exponent, which is differentiable and so was trained
    rather than chosen. It is *not* 3.0, and running the trained weights at a different
    exponent evaluates a network at a pooling it was never trained for.
  * `meta['Lw']` — supervised whitening, fitted on SfM pairs after training rather than
    learned end-to-end (`meta['whitening']` is False for exactly that reason). Plain
    `P`/`m` arrays applied as `P @ (x - m)` then L2, which is what `PCACompression`
    already does — so it is loaded into one rather than given its own class.

`Lw` comes in `ss` and `ms` variants, fitted for single- and multi-scale extraction. They
are not interchangeable; the variant is chosen from the run's `scales` rather than
exposed as a knob, so it cannot contradict them.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from cbir.descriptors.cnn.compression import PCACompression
from cbir.descriptors.cnn.weights import apply_state, root

URL = "https://cmp.felk.cvut.cz/cnnimageretrieval/data/networks/retrieval-SfM-120k"
FILES = {
    "vgg16": "retrievalSfM120k-vgg16-gem-b4dcdc6.pth",
    "resnet101": "retrievalSfM120k-resnet101-gem-b80fb85.pth",
}
"""The published GeM checkpoints. Only these two architectures were released."""

CORPUS = "retrieval-SfM-120k"
"""Key `meta['Lw']` files its whitening under — the corpus it was fitted on."""


@dataclass(frozen=True)
class FineTuned:
    """One checkpoint's three parts, already split apart."""

    state: dict[str, torch.Tensor]
    """Conv-stack weights under the architecture's own parameter names."""

    p: float
    """The trained generalized-mean exponent."""

    _whitening: dict[str, tuple[np.ndarray, np.ndarray]]
    """Per-variant `(P, m)`, keyed `ss` / `ms`."""

    def whitening(self, scales: tuple[float, ...], dim: int | None = None) -> PCACompression:
        """The supervised whitening fitted for this run's number of scales.

        `P @ (x - m)` on column vectors is `(x - m) @ P.T` on ours, which is precisely
        `PCACompression.transform` with `P` as the components and `m` as the mean.

        `dim` keeps only the leading rows of `P`, which is how the reference shortens
        these descriptors — the projection is already ordered, so there is nothing to
        refit.
        """
        variant = "ss" if len(scales) == 1 else "ms"
        projection, mean = self._whitening[variant]
        if dim is not None:
            if dim <= 0 or dim > len(projection):
                raise ValueError(f"dim must be in 1..{len(projection)}, got {dim}")
            projection = projection[:dim]
        return PCACompression(mean=mean.reshape(-1), components=projection)


def load(backbone: str) -> FineTuned:
    """Read the fine-tuned checkpoint for `backbone`, or say what is missing."""
    if backbone not in FILES:
        raise ValueError(f"no fine-tuned checkpoint published for {backbone!r}; have {sorted(FILES)}")

    path = root() / FILES[backbone]
    if not path.exists():
        raise FileNotFoundError(f"{path} not found — download it from {URL}/{FILES[backbone]}")

    # `weights_only=False` because `meta` holds numpy arrays, not just tensors. The file
    # is one we downloaded from a known URL, not user input.
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state, meta = checkpoint["state_dict"], checkpoint["meta"]

    if meta["architecture"] != backbone:
        raise ValueError(f"{path.name} is a {meta['architecture']} checkpoint, not {backbone}")
    if meta["pooling"] != "gem":
        raise ValueError(f"{path.name} pools by {meta['pooling']!r}; only gem is wired up")

    learned = meta["Lw"][CORPUS]
    return FineTuned(
        state={_rename(key, backbone): value for key, value in state.items() if key.startswith("features.")},
        p=float(state["pool.p"].item()),
        _whitening={variant: (learned[variant]["P"], learned[variant]["m"]) for variant in ("ss", "ms")},
    )


def load_into(model: torch.nn.Module, backbone: str) -> tuple[torch.nn.Module, FineTuned]:
    """Fill `model` with the fine-tuned conv weights, returning it and the rest."""
    checkpoint = load(backbone)
    return apply_state(model, checkpoint.state, source=FILES[backbone], backbone=backbone), checkpoint


def _rename(key: str, backbone: str) -> str:
    """Checkpoint parameter name to the one the torchvision module uses.

    VGG needs only the prefix gone: the checkpoint was saved from a `Sequential` built
    out of torchvision's own `features`, so the indices already line up. ResNet was saved
    the same way but from `Sequential(*resnet.children()[:-2])`, whose children are
    unnamed — so the flat indices are what we rebuild it as too, and the same strip
    works.
    """
    del backbone  # kept in the signature: a checkpoint needing real remapping goes here
    return key.removeprefix("features.")
