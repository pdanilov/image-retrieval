"""Fine-tuning a backbone for retrieval, rather than borrowing one from classification.

Everything in `descriptors/` is frozen: the network was trained for ImageNet, and
retrieval works because the features transfer. This package trains one for the task
itself, on retrieval-SfM-120k with a contrastive loss over SfM-mined landmark pairs.

AGENTS.md scoped this out until it was justified separately, and the justification is
that the frozen tier is finished and validated: `descriptors/cnn/finetuned.py` already
loads Radenovic et al.'s own checkpoints and reproduces their published rows. That gives
this package something the rest of the project never had -- a target it can be *wrong*
against, in three independent ways:

  * final mAP, against 0.619 (vgg16) and 0.647 (resnet101);
  * the learned exponent, against their 2.9208 / 2.9033;
  * the descriptors themselves, which can be compared to their checkpoint's directly.

The recipe is theirs, deliberately: the point is reproduction, and AGENTS.md is explicit
that a gap to a published number is a bug until proven otherwise, never something to
tune away.
"""
