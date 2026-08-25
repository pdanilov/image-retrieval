"""Training tuples from retrieval-SfM-120k, with hard negatives re-mined each epoch.

The corpus supplies matching pairs, mined by structure-from-motion: two photographs are
positives when they were reconstructed into the same 3D model. Negatives are not
supplied, because almost every other image is one — and that is the problem. A random
negative is beyond the margin from the start and contributes nothing, so a model trained
on random negatives sees a gradient of almost pure zeros.

So negatives are **mined against the current model**: each epoch, descriptors are
extracted for a large random pool and the highest-scoring non-matching images become
that query's negatives. They are hard by construction, and they get harder as the model
improves — the difficulty of the task tracks the network rather than being fixed up
front.

Two details in the mining are not obvious and both matter:

* **One negative per cluster.** Walking the ranking and taking the top five would often
  take five photographs of the same wrong building, which is one mistake counted five
  times. The reference takes at most one image from any cluster, so five negatives are
  five distinct ways to be wrong.
* **The query's own cluster is excluded**, since those are positives that happen not to
  be listed as pairs — training against them teaches the opposite of the intended thing.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
from PIL import Image

from cbir.descriptors.cnn.neural_codes import IMAGENET_MEAN, IMAGENET_STD
from cbir.descriptors.cnn.sfm import image_path, root

Split = Literal["train", "val"]

IMAGE_SIZE = 362
"""Longest side during training, from the reference's published command.

Far below the 1024 used at evaluation, and deliberately: training cost scales with area,
and the network is fully convolutional, so it transfers to the larger evaluation size."""

QUERY_SIZE = 2000
"""Queries sampled per epoch, out of ~181k available pairs."""

POOL_SIZE = 22000
"""Images descriptors are extracted for, to mine negatives from."""

NEG_NUM = 5
"""Negatives per tuple, each from a different cluster."""


def load_tensor(path: str, image_size: int = IMAGE_SIZE) -> torch.Tensor:
    """One normalized `(3, h, w)` image, longest side capped, aspect preserved.

    Shrinks only. Enlarging a small training image would spend compute inventing detail
    the network then learns to rely on.
    """
    from torchvision.transforms import functional as tv

    image = Image.open(path).convert("RGB")
    image.thumbnail((image_size, image_size), Image.Resampling.LANCZOS)
    return tv.normalize(tv.to_tensor(image), mean=IMAGENET_MEAN, std=IMAGENET_STD)


@dataclass(frozen=True)
class Corpus:
    """One split of retrieval-SfM-120k: images, their clusters, and the matching pairs."""

    cids: list[str]
    cluster: list[int]
    qidxs: list[int]
    pidxs: list[int]

    @classmethod
    def load(cls, split: Split) -> Corpus:
        with (root() / "retrieval-SfM-120k.pkl").open("rb") as handle:
            data = pickle.load(handle)[split]
        return cls(
            cids=list(data["cids"]),
            cluster=[int(c) for c in data["cluster"]],
            qidxs=[int(i) for i in data["qidxs"]],
            pidxs=[int(i) for i in data["pidxs"]],
        )

    def path(self, index: int) -> str:
        return image_path(self.cids[index])


def mine_negatives(
    scores: torch.Tensor,
    pool: list[int],
    query_clusters: list[int],
    cluster: list[int],
    count: int = NEG_NUM,
) -> list[list[int]]:
    """Per query, the hardest `count` pool images from distinct non-matching clusters.

    Args:
        scores: `(pool, queries)` similarities, pool images down each column.
        pool: Corpus indices of the pool images, in the order `scores` rows use.
        query_clusters: Cluster of each query, in `scores` column order.
        cluster: Cluster of every corpus image.
        count: Negatives to collect per query.

    Returns:
        One list of corpus indices per query. Shorter than `count` only if the pool ran
        out of eligible clusters, which a pool of 22000 against ~550 clusters will not do.
    """
    ranked = scores.argsort(dim=0, descending=True)
    mined = []
    for column, query_cluster in enumerate(query_clusters):
        seen = {query_cluster}
        negatives: list[int] = []
        for row in ranked[:, column].tolist():
            candidate = pool[row]
            if cluster[candidate] in seen:
                continue
            seen.add(cluster[candidate])
            negatives.append(candidate)
            if len(negatives) == count:
                break
        mined.append(negatives)
    return mined


class EpochTuples:
    """The tuples for one epoch: sampled queries, their positives, and mined negatives."""

    def __init__(self, corpus: Corpus, queries: list[int], positives: list[int], negatives: list[list[int]]) -> None:
        self.corpus = corpus
        self.queries = queries
        self.positives = positives
        self.negatives = negatives

    def __len__(self) -> int:
        return len(self.queries)

    def __getitem__(self, index: int) -> list[torch.Tensor]:
        """Query, positive, then the negatives — as separate tensors.

        Not stacked: aspect ratios are preserved, so a tuple's images have different
        shapes and the network sees them one at a time. That is what the reference does,
        and batching would require either padding (which feeds zeros into the pooling)
        or distorting the aspect ratio.
        """
        members = [self.queries[index], self.positives[index], *self.negatives[index]]
        return [load_tensor(self.corpus.path(i)) for i in members]


def sample_epoch(
    corpus: Corpus,
    model: torch.nn.Module,
    *,
    query_size: int = QUERY_SIZE,
    pool_size: int = POOL_SIZE,
    neg_num: int = NEG_NUM,
    device: str = "cuda",
    seed: int = 0,
    log=None,
) -> EpochTuples:
    """Sample queries and a negative pool, then mine negatives with the current model."""
    rng = np.random.default_rng(seed)
    pair_choice = rng.choice(len(corpus.qidxs), size=min(query_size, len(corpus.qidxs)), replace=False)
    queries = [corpus.qidxs[i] for i in pair_choice]
    positives = [corpus.pidxs[i] for i in pair_choice]
    pool = rng.choice(len(corpus.cids), size=min(pool_size, len(corpus.cids)), replace=False).tolist()

    model.eval()
    with torch.inference_mode():
        query_vectors = _describe(model, [corpus.path(i) for i in queries], device, log, "queries")
        pool_vectors = _describe(model, [corpus.path(i) for i in pool], device, log, "pool")
        scores = pool_vectors @ query_vectors.T

    negatives = mine_negatives(scores.cpu(), pool, [corpus.cluster[i] for i in queries], corpus.cluster, neg_num)
    return EpochTuples(corpus, queries, positives, negatives)


def _describe(model: torch.nn.Module, paths: list[str], device: str, log, what: str) -> torch.Tensor:
    """`(n, D)` descriptors, one forward per image — sizes differ, so they cannot batch."""
    vectors = []
    for i, path in enumerate(paths):
        vectors.append(model(load_tensor(path).unsqueeze(0).to(device)).squeeze(0))
        if log is not None and (i + 1) % 2000 == 0:
            log(f"  mining: {what} {i + 1}/{len(paths)}")
    return torch.stack(vectors)
