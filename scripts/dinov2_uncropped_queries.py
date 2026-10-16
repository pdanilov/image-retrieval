"""Replicate the DINO group's retrieval script, to explain DINOv2's Table 9.

A diagnostic, deliberately outside the harness: it writes nothing to `results/` and is
not reachable from the `cbir` CLI. It exists because the frozen-DINOv2 rows the harness
records sit below the DINOv2 paper's Table 9 (by up to 11 mAP for ViT-S on roxford5k),
and the paper does not state how it evaluated.

What it replicates is facebookresearch/dino's `eval_image_retrieval.py`, written by the
same group:

  * queries are loaded **whole** — the script never applies their `bbx`;
  * every image is `thumbnail`ed to 224: longer side down to 224, aspect kept, never
    enlarged, never padded;
  * ImageNet normalization, one image per forward pass, CLS token, cosine.

Run this way, Table 9 reproduces within 0.1-1.6 mAP for ViT-S, B and L on both datasets.
That is what pins the gap on the protocol rather than on this implementation. The
uncropped numbers are not comparable to the rest of the project: the benchmark's labels
describe the crop, which is why the harness has no option to skip it.

`--queries crop` keeps everything else identical but crops the queries, isolating what
the crop alone costs.

    uv run python scripts/dinov2_uncropped_queries.py dinov2-vits14 roxford5k --queries full
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable

import numpy as np
import torch
from PIL import Image
from PIL.Image import Image as PILImage
from torchvision.transforms import functional as tv
from transformers import AutoModel

from cbir.data.crop import crop_query
from cbir.data.images import image_paths, iter_images
from cbir.data.revisitop import download
from cbir.descriptors.cnn.neural_codes import IMAGENET_MEAN, IMAGENET_STD
from cbir.descriptors.transformer.vit import REPOS
from cbir.eval.metrics import evaluate
from cbir.eval.runner import ground_truth
from cbir.search.exact import exact_search

SIDE = 224
"""The DINO script's default `--imsize`, and what Table 9 matches."""


def describe(net: torch.nn.Module, images: Iterable[PILImage], device: str) -> np.ndarray:
    """L2-normalized CLS tokens, one image per pass, preprocessed as the DINO script does."""
    vectors = []
    with torch.inference_mode():
        for image in images:
            image = image.convert("RGB")
            # In place, as in the DINO script; LANCZOS is what its `Image.ANTIALIAS` was.
            image.thumbnail((SIDE, SIDE), Image.Resampling.LANCZOS)
            tensor = tv.normalize(tv.to_tensor(image), mean=IMAGENET_MEAN, std=IMAGENET_STD)
            vectors.append(net(pixel_values=tensor.unsqueeze(0).to(device)).pooler_output.float().cpu().numpy())
    stacked = np.concatenate(vectors)
    return stacked / np.linalg.norm(stacked, axis=1, keepdims=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("model", choices=sorted(REPOS))
    parser.add_argument("dataset", choices=["roxford5k", "rparis6k"])
    parser.add_argument("--queries", choices=["full", "crop"], default="full")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = AutoModel.from_pretrained(REPOS[args.model]).to(device).eval()

    query_dataset, database_dataset = download(args.dataset)
    database = describe(net, iter_images(image_paths(database_dataset)), device)
    queries = iter_images(image_paths(query_dataset))
    if args.queries == "crop":
        queries = (crop_query(image, box) for image, box in zip(queries, query_dataset["bbx"], strict=True))
    query_vectors = describe(net, queries, device)

    rankings = exact_search(query_vectors, database)
    truth = ground_truth(query_dataset)
    medium, hard = (evaluate(rankings, truth, protocol).map for protocol in ("medium", "hard"))
    print(f"{args.model} {args.dataset} queries={args.queries}: Medium {medium:.4f}  Hard {hard:.4f}")


if __name__ == "__main__":
    main()
