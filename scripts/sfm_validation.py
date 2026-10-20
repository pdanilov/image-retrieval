"""Score a frozen ViT on the retrieval-SfM-120k validation split, as training validates.

A diagnostic outside the harness: it writes nothing to `results/`. It answers how much
of the task the fine-tuning loop optimizes a network already solves before any training
— the number every `cbir train` run logs as "epoch 0 (untrained)", for a model one has
not trained. DeiT-B† 224 starts at 0.702 and ends IRT's recipe at 0.824; a model that
already starts near that has little left for the same recipe to add.

Same network, preprocessing and metric as the loop: `RetrievalViT` (CLS, inputs squashed
into the square) and `validate` (mAP by SfM cluster over the 6403 validation images).

    uv run python scripts/sfm_validation.py dinov2-vits14 dinov3-vitl16
"""

from __future__ import annotations

import argparse

import torch

from cbir.descriptors.transformer.vit import SPECS
from cbir.training.loop import validate
from cbir.training.net import RetrievalViT
from cbir.training.tuples import Corpus


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("models", nargs="+", choices=sorted(name for name, spec in SPECS.items() if not spec.siglip))
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    corpus = Corpus.load("val")
    for model in args.models:
        net = RetrievalViT(model).to(device)
        scores = validate(net, corpus, device)
        print(f"{model}: SfM val mAP {scores['mean_average_precision']:.4f}  P@1 {scores['precision_at_1']:.4f}")
        del net
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
