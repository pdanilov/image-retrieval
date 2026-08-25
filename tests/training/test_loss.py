"""The loss's shape, not its value: what must hold is the *sign* of each term."""

from __future__ import annotations

import pytest
import torch

from cbir.training.loss import MARGIN, contrastive_loss


def _unit(*shape):
    x = torch.randn(*shape)
    return torch.nn.functional.normalize(x, dim=-1)


def test_a_perfect_tuple_costs_nothing():
    # Query identical to its positive, negatives already beyond the margin: there is
    # nothing left to learn and the loss must say so.
    query = _unit(2, 8)
    negatives = -query.unsqueeze(1).expand(2, 3, 8)  # antipodal, distance 2 > margin

    loss = contrastive_loss(query, query.clone(), negatives, margin=0.7)

    assert loss.item() == pytest.approx(0.0, abs=1e-6)


def test_a_distant_positive_is_penalized_without_limit():
    # The attractive term has no hinge, so pushing the positive further always costs
    # more -- this is what stops the network collapsing everything to one point.
    query = _unit(1, 8)
    near = contrastive_loss(query, query * 0.9 + _unit(1, 8) * 0.1, torch.zeros(1, 1, 8) + 9, margin=0.7)
    far = contrastive_loss(query, -query, torch.zeros(1, 1, 8) + 9, margin=0.7)

    assert far > near


def test_a_negative_beyond_the_margin_contributes_nothing():
    # The whole point of the hinge: most negatives are easy, and without it they would
    # dominate the gradient with a signal that says nothing.
    query, positive = _unit(1, 8), _unit(1, 8)
    far = query.new_zeros(1, 1, 8)
    far[0, 0, 0] = 9.0  # distance far beyond any margin

    with_far = contrastive_loss(query, positive, far, margin=0.7)
    without = contrastive_loss(query, positive, far.clone(), margin=0.0)

    assert with_far.item() == pytest.approx(without.item(), abs=1e-6)


def test_a_hard_negative_costs_more_than_an_easy_one():
    query, positive = _unit(1, 8), _unit(1, 8)
    hard = (query * 0.99).unsqueeze(1)
    easy = (-query).unsqueeze(1)

    assert contrastive_loss(query, positive, hard, 0.7) > contrastive_loss(query, positive, easy, 0.7)


def test_the_loss_is_summed_not_averaged():
    # The reference sums, and its learning rate is calibrated to that scale; averaging
    # would quietly divide every gradient by the batch size.
    query, positive = _unit(1, 8), _unit(1, 8)
    negatives = _unit(1, 3, 8)

    one = contrastive_loss(query, positive, negatives, 0.7)
    four = contrastive_loss(query.repeat(4, 1), positive.repeat(4, 1), negatives.repeat(4, 1, 1), 0.7)

    assert four.item() == pytest.approx(4 * one.item(), rel=1e-5)


def test_gradients_reach_every_input():
    query = _unit(2, 8).requires_grad_()
    positive = _unit(2, 8).requires_grad_()
    negatives = _unit(2, 3, 8).requires_grad_()

    contrastive_loss(query, positive, negatives, 0.7).backward()

    assert query.grad is not None and query.grad.abs().sum() > 0
    assert positive.grad is not None and positive.grad.abs().sum() > 0


def test_mismatched_shapes_are_refused():
    with pytest.raises(ValueError, match="query and positive"):
        contrastive_loss(_unit(2, 8), _unit(3, 8), _unit(2, 1, 8), 0.7)
    with pytest.raises(ValueError, match=r"\(B, K, D\)"):
        contrastive_loss(_unit(2, 8), _unit(2, 8), _unit(2, 8), 0.7)


def test_each_backbone_carries_its_published_margin():
    # resnet101's 0.85 is from the reference's own published command; vgg16 has no
    # published command, so it takes the argparse default.
    assert MARGIN["resnet101"] == 0.85
    assert MARGIN["vgg16"] == 0.7
