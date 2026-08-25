"""Contrastive loss over (query, positive, negatives) tuples.

The supervision retrieval actually wants: not "which class is this" but "are these two
photographs of the same thing". Matching pairs are pulled together without limit;
non-matching pairs are pushed apart only until they are `margin` apart, after which they
contribute nothing. That hinge is what stops the easy negatives — which are almost all
of them — from dominating the gradient.

Both terms are distances between L2-normalized vectors, so the whole loss lives on the
same scale the evaluation's cosine similarity does.
"""

from __future__ import annotations

import torch

EPS = 1e-6
"""Added inside the square before the root. At `D = 0` — a query against an identical
positive — the derivative of `sqrt` is infinite, and that case is common enough in a
landmark corpus with near-duplicate photographs to matter."""

MARGIN = {"vgg16": 0.7, "resnet101": 0.85, "resnet50": 0.85}
"""Per-backbone hinge width.

0.85 is the reference's own published command for resnet101; 0.7 is its argparse default
and what vgg16 falls back to, since no vgg16 command was ever published. Recorded here
rather than passed in so a run cannot silently use the wrong one for its backbone."""


def contrastive_loss(
    query: torch.Tensor,
    positive: torch.Tensor,
    negatives: torch.Tensor,
    margin: float,
) -> torch.Tensor:
    """Loss for a batch of tuples.

    Args:
        query: `(B, D)` L2-normalized query descriptors.
        positive: `(B, D)` the matching image for each query.
        negatives: `(B, K, D)` non-matching images, usually mined to be hard.
        margin: Distance beyond which a negative pair is already far enough.

    Returns:
        Scalar, summed over the batch rather than averaged — the reference sums, and the
        learning rates that go with this recipe are calibrated to that scale.
    """
    if query.shape != positive.shape:
        raise ValueError(f"query and positive must match: {tuple(query.shape)} vs {tuple(positive.shape)}")
    if negatives.ndim != 3 or negatives.shape[0] != query.shape[0]:
        raise ValueError(f"negatives must be (B, K, D) matching the batch, got {tuple(negatives.shape)}")

    # `+ EPS` inside the square, not outside the root: it keeps the gradient finite at
    # coincident points without shifting the distance itself by a constant.
    positive_distance = ((query - positive) + EPS).pow(2).sum(dim=1).sqrt()
    negative_distance = ((query.unsqueeze(1) - negatives) + EPS).pow(2).sum(dim=2).sqrt()

    attract = positive_distance.pow(2)
    repel = (margin - negative_distance).clamp(min=0).pow(2).sum(dim=1)
    return 0.5 * (attract + repel).sum()
