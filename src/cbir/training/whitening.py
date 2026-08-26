"""Supervised whitening, fitted on SfM matching pairs.

The unsupervised whitening this project has used until now (`PCACompression.fit`) asks
one question of an unlabelled corpus: which directions carry the most variance? It then
rescales them all to equal footing. That is better than nothing — Radenović et al. call
it "essential" for off-the-shelf descriptors — but it cannot know which of that variance
is *nuisance*. Two photographs of the same building differ by viewpoint, time of day and
occlusion, and those differences are exactly what a retrieval descriptor should ignore.

With matching pairs the nuisance directions become measurable: they are the directions
in which descriptors of the *same* landmark disagree. This fits the intra-class
covariance of pair differences, whitens it to the identity — flattening precisely the
directions that vary within a landmark — and then rotates by the scatter of the whitened
data so the surviving directions come out ordered, which is what makes truncating to
`dim` a matter of dropping trailing rows.

It is fitted *after* training rather than learned end-to-end, which is why a published
checkpoint has `meta['whitening'] = False` and yet ships a `meta['Lw']`. The measured
cost of doing without it is large: the published vgg16 checkpoint scores 0.5747 Medium
on roxford5k whitened by PCA on the held-out set, against 0.6073 with the projection
fitted here — roughly four times what the whole learning-rate change was worth.

Follows Radenović et al. (TPAMI 2018) §4.3, itself after Mikulik et al. (2013).
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch

from cbir.data.images import iter_images
from cbir.descriptors.cnn.finetuned import CORPUS
from cbir.descriptors.cnn.pooling import MULTI_SCALE, PooledCNN
from cbir.training.tuples import Corpus, Split


def learn(descriptors: np.ndarray, qidxs: Sequence[int], pidxs: Sequence[int]) -> tuple[np.ndarray, np.ndarray]:
    """Fit `(P, m)` from descriptors and the pairs that match within them.

    `descriptors` is `(N, D)`, one L2-normalized row per image; `qidxs`/`pidxs` index
    into it, one entry each per matching pair. Returns `P` as `(D, D)` — rows ordered
    most- to least-discriminative — and `m` as `(D, 1)`, the layout the published
    checkpoints use, so the result can be written straight into `meta['Lw']`.

    Applied as `P @ (x - m)` then L2, which is `PCACompression(mean=m, components=P)`.
    """
    # float64 throughout: this inverts a Cholesky factor and eigendecomposes a scatter
    # matrix, and float32 descriptors give a covariance whose small eigenvalues are noise
    # at single precision. Cast back on the way out.
    x = np.asarray(descriptors, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"descriptors must be (N, D), got shape {x.shape}")
    if len(qidxs) != len(pidxs):
        raise ValueError(f"qidxs and pidxs must pair up, got {len(qidxs)} and {len(pidxs)}")
    if not len(qidxs):
        raise ValueError("no matching pairs: supervised whitening is exactly what they provide")

    dim = x.shape[1]
    if len(qidxs) < dim:
        raise ValueError(
            f"{len(qidxs)} pairs cannot determine a {dim}x{dim} intra-class covariance — "
            f"it is singular below {dim} pairs and unstable near it. Extract more pairs."
        )

    mean = x[list(qidxs)].mean(axis=0)
    difference = x[list(qidxs)] - x[list(pidxs)]
    # The nuisance covariance: how descriptors of the *same* landmark disagree. Not
    # mean-centred, deliberately — a matching pair's expected difference is zero, and
    # centring would subtract the very bias that a systematic nuisance direction has.
    intra = difference.T @ difference / len(difference)

    try:
        factor = np.linalg.cholesky(intra)
    except np.linalg.LinAlgError as error:  # pragma: no cover - needs a degenerate fit
        raise ValueError(
            "the intra-class covariance is not positive definite, so it cannot be "
            "whitened. Usually too few pairs, or duplicate images making some pair "
            "differences exactly zero."
        ) from error
    whitener = np.linalg.inv(factor)

    projected = (x - mean) @ whitener.T
    # `eigh`, not `eig`: this scatter is symmetric positive semi-definite by
    # construction, and the general solver would do more work to return the same
    # eigenvectors with a spurious imaginary part attached.
    eigenvalues, eigenvectors = np.linalg.eigh(projected.T @ projected)
    rotation = eigenvectors[:, eigenvalues.argsort()[::-1]].T

    return (rotation @ whitener).astype(np.float32), mean.astype(np.float32).reshape(-1, 1)


PAIRS = 20000
"""Matching pairs sampled to fit on, and the knob that decides what this costs.

The statistically binding quantity is pairs against descriptor width: the intra-class
covariance is `D x D` and singular below `D` pairs. 20000 is 39x vgg16's 512 and 10x
resnet101's 2048, comfortably clear of both, while touching roughly 35000 images rather
than the corpus's 91642.
"""

VARIANTS: dict[str, tuple[float, ...]] = {"ss": (1.0,), "ms": MULTI_SCALE}
"""The two extraction settings a projection can be fitted for, keyed as `meta['Lw']` is.

Fitted separately and not interchangeable: a multi-scale descriptor is a generalized mean
over three resolutions, so its nuisance directions are not the single-scale ones. The
reference ships both for the same reason, and `finetuned.whitening` picks between them by
the run's `scales` rather than exposing the choice.
"""


def sample_pairs(corpus: Corpus, count: int, seed: int) -> tuple[list[int], list[int], list[int]]:
    """Pick `count` matching pairs, returning the images they touch and remapped indices.

    Sampling pairs rather than images is deliberate. Sampling images and keeping the
    pairs internal to them retains only the *square* of the sampling fraction — a fifth
    of the corpus keeps a twenty-fifth of the pairs — which puts the count that actually
    has to clear `D` at the mercy of the one that does not.
    """
    total = len(corpus.qidxs)
    rng = np.random.default_rng(seed)
    chosen = rng.permutation(total)[: min(count, total)]

    images = sorted({corpus.qidxs[i] for i in chosen} | {corpus.pidxs[i] for i in chosen})
    position = {image: index for index, image in enumerate(images)}
    return (
        images,
        [position[corpus.qidxs[i]] for i in chosen],
        [position[corpus.pidxs[i]] for i in chosen],
    )


def fit(
    checkpoint: str,
    backbone: str,
    *,
    pairs: int = PAIRS,
    max_side: int = 1024,
    seed: int = 0,
    split: Split = "train",
    log=print,
) -> dict[str, dict[str, np.ndarray]]:
    """Fit both variants for a checkpoint, ready to write into `meta['Lw']`.

    Extraction runs at the *evaluation* resolution, not the training one: the projection
    has to match the descriptors it will be applied to, and a network trained at 362 is
    still evaluated at 1024.
    """
    corpus = Corpus.load(split)
    images, qidxs, pidxs = sample_pairs(corpus, pairs, seed)
    paths = [corpus.path(index) for index in images]
    log(f"fitting on {len(qidxs)} pairs over {len(paths)} images from the {split} split")

    fitted: dict[str, dict[str, np.ndarray]] = {}
    for variant, scales in VARIANTS.items():
        model = PooledCNN(
            backbone,
            p="learned",
            weights="sfm120k",
            # The fine-tuned checkpoints were trained without VGG's trailing pool, and
            # ResNet has none to drop, so this is right for both.
            last_pool=False,
            checkpoint=checkpoint,
            max_side=max_side,
            scales=scales,
        )
        started = time.time()
        descriptors = model.extract(iter_images(paths))
        projection, mean = learn(descriptors, qidxs, pidxs)
        fitted[variant] = {"P": projection, "m": mean}
        log(f"  {variant}: {descriptors.shape[1]}-D, fitted in {time.time() - started:.0f}s")
    return fitted


def attach(path: Path, fitted: dict[str, dict[str, np.ndarray]], corpus: str = CORPUS) -> None:
    """Write a fitted projection into a checkpoint's `meta['Lw']`, in place.

    In place, and under the key the loader already reads, because that is what makes the
    result usable without touching the evaluation path: `--whiten-source learned` then
    finds a projection where before it found nothing. Everything else in the file is
    carried across untouched, and the write goes through a temporary file so an
    interrupted run cannot leave a truncated checkpoint where a good one was.
    """
    payload = torch.load(path, map_location="cpu", weights_only=False)
    payload.setdefault("meta", {}).setdefault("Lw", {})[corpus] = fitted
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)
