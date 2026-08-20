"""Typed configs for the CNN tier.

Same structure as the classic tier: one frozen dataclass per technique, carrying only
the knobs that technique honours, plus `prepare` and `train_and_encode`. Nothing in
`eval/runner.py` knows which tier it is running.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from cbir.configs.types import Encoded
from cbir.data.crop import crop_query
from cbir.data.holdout import EvalDataset
from cbir.data.images import iter_images
from cbir.descriptors.cnn.compression import PCACompression
from cbir.descriptors.cnn.neural_codes import DIM as NEURAL_CODES_DIM
from cbir.descriptors.cnn.neural_codes import Backbone, NeuralCodes
from cbir.descriptors.cnn.pooling import CHANNELS, PooledCNN
from cbir.descriptors.cnn.pooling import Backbone as PoolBackbone
from cbir.descriptors.cnn.prepare import EvalImages, prepare_image_inputs
from cbir.descriptors.cnn.rmac import RMAC
from cbir.descriptors.cnn.sfm import WhitenSource, whitening_paths


@dataclass(frozen=True)
class NeuralCodesConfig:
    """Frozen-CNN fully-connected activations (Babenko et al., ECCV 2014).

    Args:
        backbone: Which frozen ImageNet network supplies `fc6`.
        dim: PCA output dimensionality. The compression is part of the method, not an
            option on top of it — the paper's claim is that a 4096-D code survives it —
            so the default is a compressed width and `None` is the *ablation*, giving
            the raw 4096-D code. Fitted on the held-out dataset, never the eval database.
        seed: Seeds PCA's randomized solver, so a refit reproduces.
    """

    backbone: Backbone = "alexnet"
    dim: int | None = 256
    seed: int = 0

    technique: ClassVar[str] = "neural_codes"
    inputs_kind: ClassVar[str] = "images"

    def prepare(self, dataset: EvalDataset) -> EvalImages:
        return prepare_image_inputs(dataset)

    def train_and_encode(self, inputs: EvalImages) -> Encoded:
        model = NeuralCodes(self.backbone)

        database_vectors = model.extract(iter_images(inputs.database_paths))
        # Queries are cropped to their ground-truth region before the network sees
        # them -- AGENTS.md requires it of every method, classic and neural alike.
        cropped = (
            crop_query(image, box)
            for image, box in zip(iter_images(inputs.query_paths), inputs.query_boxes, strict=True)
        )
        query_vectors = model.extract(cropped)

        if self.dim is None:
            return database_vectors, query_vectors

        # Fitted on the held-out dataset's images, which is a third pass through the
        # network. Worth it rather than reusing the database codes: fitting the
        # projection on the images it will search is fitting on the test set.
        held_out = model.extract(iter_images(inputs.held_out_paths))
        pca = PCACompression.fit(held_out, dim=self.dim, seed=self.seed)
        return pca.transform(database_vectors), pca.transform(query_vectors)


@dataclass(frozen=True)
class PooledConfig:
    """Pooled conv-map descriptors: SPoC, MAC and GeM as one parameterized method.

    `technique` is `"gem"` for all of them because they *are* one method — the
    generalized mean — and `p` is what names the variant in the literature. A row with
    `p=1.0` is SPoC and `p=None` is MAC; both are recorded as `gem` plus their `p`,
    which is what keeps them comparable in one table instead of split across three.

    Args:
        backbone: Frozen ImageNet network whose last conv map is pooled.
        p: Generalized-mean exponent. `1.0` = SPoC (average), `None` = MAC (max),
            `3.0` = the GeM default from Radenović et al.
        max_side: Longest image side fed to the network. Caps only, never enlarges.
        scales: Input resolutions each image is described at, as factors of the capped
            size; the per-scale descriptors are L2-normalized and averaged. The default
            `(1.0,)` is single-scale — the published off-the-shelf rows use
            `MULTI_SCALE`, `(1, 1/sqrt(2), 1/2)`, and cost one forward pass per scale.
        dim: PCA width, fitted on the held-out set. `None` keeps the backbone's
            native width, which is already compact — with `whiten` on that still means
            a PCA is fitted, at full width, since whitening is a rotation and rescale
            rather than a compression.
        whiten: Divide each PCA direction by its standard deviation. Radenović et al.
            call this "essential" for off-the-shelf CNN descriptors and apply it to
            every such row they publish, so their numbers are not comparable without
            it. Off by default so that a run says which it was.
        whiten_source: Which corpus the whitening is fitted on. `held_out` is the
            sibling benchmark (6322 images); `sfm30k`/`sfm120k` are the retrieval-SfM
            landmark sets the reference uses, which need a manual download. Recorded so
            a row always says where its projection came from.
        shrinkage: Floors whitening's divisor at `sqrt(lambda + eps)`, `eps` being this
            fraction of the mean eigenvalue. Only whitening reads it. Whitening is
            unstable when the held-out set is not much larger than the descriptor width,
            which is the regime every 2048-D backbone is in here.
        seed: Seeds PCA's randomized solver.
    """

    backbone: PoolBackbone = "alexnet"
    p: float | None = 3.0
    max_side: int = 1024
    scales: tuple[float, ...] = (1.0,)
    dim: int | None = None
    whiten: bool = False
    whiten_source: WhitenSource = "held_out"
    shrinkage: float = 0.0
    seed: int = 0

    technique: ClassVar[str] = "gem"
    inputs_kind: ClassVar[str] = "images"

    def prepare(self, dataset: EvalDataset) -> EvalImages:
        return prepare_image_inputs(dataset)

    def _fitting_paths(self, inputs: EvalImages) -> list[str]:
        """Images the PCA/whitening is fitted on — never the ones being searched."""
        if self.whiten_source == "held_out":
            return inputs.held_out_paths
        return whitening_paths(self.whiten_source)

    def train_and_encode(self, inputs: EvalImages) -> Encoded:
        model = PooledCNN(self.backbone, p=self.p, max_side=self.max_side, scales=self.scales)

        database_vectors = model.extract(iter_images(inputs.database_paths))
        cropped = (
            crop_query(image, box)
            for image, box in zip(iter_images(inputs.query_paths), inputs.query_boxes, strict=True)
        )
        query_vectors = model.extract(cropped)

        if self.dim is None and not self.whiten:
            return database_vectors, query_vectors

        held_out = model.extract(iter_images(self._fitting_paths(inputs)))
        # Whitening with no explicit width still needs a PCA, fitted at the backbone's
        # full descriptor size: it rescales the axes without discarding any.
        width = self.dim if self.dim is not None else CHANNELS[self.backbone]
        pca = PCACompression.fit(held_out, dim=width, whiten=self.whiten, shrinkage=self.shrinkage, seed=self.seed)
        return pca.transform(database_vectors), pca.transform(query_vectors)


@dataclass(frozen=True)
class RMACConfig:
    """Regional max-pooling (Tolias et al., ICLR 2016).

    Separate from `PooledConfig` because R-MAC is not a point on the generalized-mean
    axis: it changes what the max is taken over, not how activations are combined. It
    therefore has no `p`, and a shared config would carry one that does nothing.

    Args:
        backbone: Frozen ImageNet network whose last conv map is pooled.
        levels: Region grid depth. Level 1 is two large squares, each level after is
            finer; 3 is the paper's setting.
        max_side: Longest image side fed to the network. Caps only, never enlarges.
        scales: Input resolutions, combined by averaging as the reference table
            specifies for every method except GeM.
        dim: PCA width, fitted on the held-out set. `None` keeps the native width.
        whiten: PCA-whiten the finished descriptor. The paper whitens each region
            vector instead, with a projection learned on a separate landmark set —
            see `descriptors/cnn/rmac.py` for why that is not what happens here.
        whiten_source: Which corpus the whitening is fitted on. `held_out` is the
            sibling benchmark (6322 images); `sfm30k`/`sfm120k` are the retrieval-SfM
            landmark sets the reference uses, which need a manual download. Recorded so
            a row always says where its projection came from.
        shrinkage: Floors whitening's divisor at `sqrt(lambda + eps)`, `eps` being this
            fraction of the mean eigenvalue. Only whitening reads it. Whitening is
            unstable when the held-out set is not much larger than the descriptor width,
            which is the regime every 2048-D backbone is in here.
        seed: Seeds PCA's randomized solver.
    """

    backbone: PoolBackbone = "alexnet"
    levels: int = 3
    max_side: int = 1024
    scales: tuple[float, ...] = (1.0,)
    dim: int | None = None
    whiten: bool = False
    whiten_source: WhitenSource = "held_out"
    shrinkage: float = 0.0
    seed: int = 0

    technique: ClassVar[str] = "rmac"
    inputs_kind: ClassVar[str] = "images"

    def prepare(self, dataset: EvalDataset) -> EvalImages:
        return prepare_image_inputs(dataset)

    def _fitting_paths(self, inputs: EvalImages) -> list[str]:
        """Images the PCA/whitening is fitted on — never the ones being searched."""
        if self.whiten_source == "held_out":
            return inputs.held_out_paths
        return whitening_paths(self.whiten_source)

    def train_and_encode(self, inputs: EvalImages) -> Encoded:
        model = RMAC(self.backbone, levels=self.levels, max_side=self.max_side, scales=self.scales)

        database_vectors = model.extract(iter_images(inputs.database_paths))
        cropped = (
            crop_query(image, box)
            for image, box in zip(iter_images(inputs.query_paths), inputs.query_boxes, strict=True)
        )
        query_vectors = model.extract(cropped)

        if self.dim is None and not self.whiten:
            return database_vectors, query_vectors

        held_out = model.extract(iter_images(self._fitting_paths(inputs)))
        width = self.dim if self.dim is not None else CHANNELS[self.backbone]
        pca = PCACompression.fit(held_out, dim=width, whiten=self.whiten, shrinkage=self.shrinkage, seed=self.seed)
        return pca.transform(database_vectors), pca.transform(query_vectors)


CNNConfig = NeuralCodesConfig | PooledConfig | RMACConfig
"""Union of this tier's configs, composed by `configs/run.py`."""


def descriptor_dim(config: CNNConfig) -> int:
    """Encoded vector length: the PCA width if set, else the descriptor's native size."""
    if config.dim is not None:
        return config.dim
    if isinstance(config, NeuralCodesConfig):
        return NEURAL_CODES_DIM
    return CHANNELS[config.backbone]
