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
from cbir.descriptors.cnn.neural_codes import Backbone, NeuralCodes
from cbir.descriptors.cnn.prepare import EvalImages, prepare_image_inputs


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


CNNConfig = NeuralCodesConfig
"""Union of this tier's configs. A single member today; the pooling family joins it.

Kept as a named alias so `configs/run.py` composes tiers rather than techniques, and
adding GeM means changing this line rather than every place a union is spelled out.
"""


def descriptor_dim(config: NeuralCodesConfig) -> int:
    """Encoded vector length — 4096 uncompressed, otherwise the PCA output size."""
    from cbir.descriptors.cnn.neural_codes import DIM

    return DIM if config.dim is None else config.dim
