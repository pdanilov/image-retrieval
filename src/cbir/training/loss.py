"""Contrastive loss over (query, positive, negatives) tuples.

The supervision retrieval actually wants: not "which class is this" but "are these two
photographs of the same thing". Matching pairs are pulled together without limit;
non-matching pairs are pushed apart only until they are `margin` apart, after which they
contribute nothing. That hinge is what stops the easy negatives — which are almost all
of them — from dominating the gradient.

Built on `pytorch_metric_learning` for the pair machinery (distance, pair enumeration,
reduction), which makes the other objectives in that library — triplet, multi-similarity,
ArcFace — drop-in alternatives to try later. But it is **not** PML's `ContrastiveLoss`
unmodified, and the difference is not cosmetic:

* PML's hinge is **linear**, `relu(d - margin)`; the recipe being reproduced **squares**
  both terms. `SquaredContrastiveLoss` overrides the two per-pair calculations to
  restore that.
* PML defaults to `AvgNonZeroReducer`; the reference **sums**, and the learning rate that
  goes with this recipe is calibrated to that scale. Hence `SumReducer`.

One thing the squared hinge buys for free: at `D = 0` — a query against a near-duplicate
photograph, common in a landmark corpus — the derivative of `sqrt` is infinite, but the
`2D` factor from the square cancels it. The reference carries an epsilon inside the
square for this; here it is unnecessary, and a test covers the coincident-point case.

Substituting the stock loss would quietly optimize a different objective, and this branch
exists to reproduce a published number — AGENTS.md is explicit that a gap to one is a bug
until proven otherwise, never something to tune. `tests/training/test_loss.py` therefore
pins this implementation against the reference formula numerically; if a PML upgrade
changes the semantics, that test fails rather than the mAP quietly dropping.
"""

from __future__ import annotations

from functools import lru_cache

import torch
from pytorch_metric_learning import losses, reducers
from pytorch_metric_learning.distances import LpDistance

MARGIN = {"vgg16": 0.7, "resnet101": 0.85, "resnet50": 0.85}
"""Per-backbone hinge width.

0.85 is the reference's own published command for resnet101; 0.7 is its argparse default
and what vgg16 falls back to, since no vgg16 command was ever published. Recorded here
rather than passed in so a run cannot silently use the wrong one for its backbone."""


class SquaredContrastiveLoss(losses.ContrastiveLoss):
    """PML's contrastive loss with the reference's squared hinge and summed reduction.

    Only the two per-pair calculations change. Pair enumeration, the distance, and
    reduction all stay PML's, so swapping in another distance or reducer works as it
    does for any of its losses.
    """

    def __init__(self, margin: float) -> None:
        super().__init__(
            pos_margin=0.0,
            neg_margin=margin,
            # Embeddings arrive already L2-normalized from the network; normalizing again
            # is a no-op that costs a kernel and obscures where normalization happens.
            distance=LpDistance(p=2, normalize_embeddings=False),
            reducer=reducers.SumReducer(),
        )

    def pos_calc(self, pos_pair_dist, margin):
        return torch.nn.functional.relu(self.distance.margin(pos_pair_dist, margin)).pow(2) * 0.5

    def neg_calc(self, neg_pair_dist, margin):
        return torch.nn.functional.relu(self.distance.margin(margin, neg_pair_dist)).pow(2) * 0.5


def tuple_indices(batch: int, negatives: int, device: torch.device | str = "cpu"):
    """PML `indices_tuple` for a batch of tuples laid out as `[q, p, n_1 … n_K]`.

    Passed explicitly rather than letting PML derive pairs from labels, because deriving
    them would also pair each *positive* against every negative. The reference forms only
    query-anchored pairs, and the extra ones would be a different objective.

    Returns `(anchor_pos, pos, anchor_neg, neg)` as flat index tensors into a batch
    flattened to `(B * (2 + K), D)`.
    """
    width = 2 + negatives
    starts = torch.arange(batch, device=device) * width
    anchors = starts
    positives = starts + 1
    negative_anchors = starts.repeat_interleave(negatives)
    negative_indices = (starts.unsqueeze(1) + torch.arange(2, width, device=device)).reshape(-1)
    return anchors, positives, negative_anchors, negative_indices


@lru_cache(maxsize=8)
def _criterion(margin: float) -> SquaredContrastiveLoss:
    """One loss module per margin. It holds no parameters, but the training loop calls
    this thousands of times per epoch and rebuilding PML's submodules each time is pure
    overhead."""
    return SquaredContrastiveLoss(margin)


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
        Scalar, summed over the batch rather than averaged.
    """
    if query.shape != positive.shape:
        raise ValueError(f"query and positive must match: {tuple(query.shape)} vs {tuple(positive.shape)}")
    if negatives.ndim != 3 or negatives.shape[0] != query.shape[0]:
        raise ValueError(f"negatives must be (B, K, D) matching the batch, got {tuple(negatives.shape)}")

    batch, count, dim = negatives.shape
    # PML returns zero when every index list holds at most one entry, which for this
    # layout means exactly one tuple carrying exactly one negative. Silently training on
    # a zero loss is worse than refusing, and real tuples carry five negatives.
    if batch * count == 1:
        raise ValueError("a single tuple with a single negative is degenerate for PML; use more of either")

    flat = torch.cat([query.unsqueeze(1), positive.unsqueeze(1), negatives], dim=1).reshape(batch * (2 + count), dim)
    indices = tuple_indices(batch, count, device=flat.device)
    return _criterion(margin)(flat, indices_tuple=indices)
