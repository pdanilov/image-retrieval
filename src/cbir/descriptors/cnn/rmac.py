"""R-MAC: max-pooling over a grid of regions rather than the whole feature map.

Tolias, Sicre and Jégou, ICLR 2016 (`1511.05879`). MAC throws away where an activation
fired: one max per channel over the entire map. R-MAC keeps a coarse sense of layout by
max-pooling over overlapping square regions at several sizes, L2-normalizing each region
vector, and summing them. A landmark occupying a quarter of the frame then dominates the
regions it falls in, instead of competing with the whole scene for one max.

This is the one member of the family that `pooling.py` cannot express as an exponent:
the generalized mean parameterizes *how* activations are combined, while R-MAC changes
*what is combined over*. So it subclasses `PooledCNN` and overrides `pool` alone —
backbone, preprocessing, multi-scale and extraction are all inherited unchanged.

Deviation from the paper, recorded here rather than buried: the published method
PCA-whitens each region vector before summing, with the projection learned on a separate
landmark set. Whitening here stays where the rest of the tier keeps it — applied to the
finished descriptor via `PooledConfig`/`RMACConfig`'s `dim`/`whiten`, fitted on the
held-out dataset. Per-region whitening would need the projection fitted on region
vectors, which is a different fit, and AGENTS.md rules out training on a third dataset.
"""

from __future__ import annotations

import numpy as np
import torch

from cbir.descriptors.cnn.pooling import Backbone, PooledCNN

OVERLAP = 0.4
"""Target overlap between neighbouring regions, from the paper."""

CANDIDATE_STEPS = np.arange(2, 8, dtype=np.float64)
"""Candidate region counts along the longer side; the one closest to `OVERLAP` wins."""


def regions(height: int, width: int, levels: int) -> list[tuple[int, int, int, int]]:
    """Square region boxes `(top, left, size, size)` over a `height x width` map.

    Level `l` uses squares of side `2*min(h, w)/(l+1)`, so level 1 is two large regions
    and each level after is finer. Counts along the longer side are chosen to land as
    close to `OVERLAP` as the geometry allows, which is why the number of regions is not
    a round formula.
    """
    short = min(height, width)
    long_side = max(height, width)

    # How many regions fit along the longer side at the target overlap.
    stride = (long_side - short) / np.maximum(CANDIDATE_STEPS - 1, 1)
    steps = int(CANDIDATE_STEPS[np.argmin(np.abs((short - stride) / short - OVERLAP))])
    # Strictly greater, not `>=`: on a square map neither axis is the longer one, and
    # granting the extra regions to width would make the grid asymmetric in an image
    # that has no long side to spread them along.
    extra_w = steps - 1 if width > height else 0
    extra_h = steps - 1 if height > width else 0

    boxes: list[tuple[int, int, int, int]] = []
    for level in range(1, levels + 1):
        size = int(2 * short / (level + 1))
        if size < 1:
            break
        across, down = level + extra_w, level + extra_h
        lefts = _starts(width, size, across)
        tops = _starts(height, size, down)
        boxes.extend((top, left, size, size) for top in tops for left in lefts)
    return boxes


def _starts(extent: int, size: int, count: int) -> list[int]:
    """`count` evenly spaced start offsets for a window of `size` within `extent`."""
    if count <= 1 or extent <= size:
        return [0]
    step = (extent - size) / (count - 1)
    return [round(i * step) for i in range(count)]


class RMAC(PooledCNN):
    """`PooledCNN` with regional max-pooling in place of a single global pool.

    `p` is inherited but unused — R-MAC is max-pooling by definition, and the paper has
    no exponent. It is fixed to `None` so a record cannot claim an exponent that had no
    effect.
    """

    def __init__(self, backbone: Backbone = "alexnet", *, levels: int = 3, **kwargs) -> None:
        if levels < 1:
            raise ValueError(f"levels must be at least 1, got {levels}")
        super().__init__(backbone, p=None, **kwargs)
        self.levels = levels

    def pool(self, features: torch.Tensor) -> torch.Tensor:
        """`(B, C, h, w)` conv map to `(B, C)`, summing L2-normalized region maxima."""
        _, _, height, width = features.shape
        total = torch.zeros(features.shape[:2], dtype=features.dtype, device=features.device)
        for top, left, box_h, box_w in regions(height, width, self.levels):
            region = features[:, :, top : top + box_h, left : left + box_w]
            vector = region.amax(dim=(2, 3))
            # Per-region L2 before summing: without it a region with strong activations
            # (usually the largest, which covers most of the frame) dominates the sum
            # and R-MAC degenerates towards MAC.
            total = total + vector / vector.norm(dim=1, keepdim=True).clamp(min=1e-6)
        return total
