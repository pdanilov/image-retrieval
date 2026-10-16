"""Typed configs for the transformer tier.

Same shape as the other tiers: a frozen dataclass carrying only the knobs its technique
honours, plus `prepare` and `train_and_encode`, so `eval/runner.py` runs it without
knowing which era it belongs to.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from cbir.configs.types import Encoded
from cbir.data.crop import crop_query
from cbir.data.holdout import EvalDataset
from cbir.data.images import iter_images
from cbir.descriptors.cnn.prepare import EvalImages, prepare_image_inputs
from cbir.descriptors.transformer.vit import IMAGE_SIZE, FrozenViT, Interpolation, Model, Resize


@dataclass(frozen=True)
class ViTConfig:
    """A frozen vision transformer's global readout, compared by cosine.

    The readout is the CLS token for DINOv2/DINOv3 and the attention-pooling head for
    SigLIP 2 — each family's own image embedding. Nothing is fitted: no PCA, no
    whitening, so the held-out images are never extracted.

    Args:
        model: Which frozen checkpoint to read out. SigLIP 2 names carry their
            resolution because each one is a separate fixed-size checkpoint.
        image_size: Input side: the square's for `pad` and `crop`, the longer side's for
            `aspect`. 224 is the published protocol; anything else is an ablation. Must
            be a multiple of the patch size, and equal to the native size for SigLIP 2.
        resize: How an image is fitted to `image_size`. `pad` is the documented
            protocol; `aspect` and `crop` were ruled out as the reason the DINOv2
            paper's table does not reproduce — see `descriptors/transformer/vit.py`.
        interpolation: Resampling filter. The papers name none; bicubic is what
            DINOv2's own evaluation transforms use.
    """

    model: Model = "dinov2-vitl14"
    image_size: int = IMAGE_SIZE
    resize: Resize = "pad"
    interpolation: Interpolation = "bicubic"

    technique: ClassVar[str] = "vit"
    inputs_kind: ClassVar[str] = "images"

    def prepare(self, dataset: EvalDataset) -> EvalImages:
        return prepare_image_inputs(dataset)

    def train_and_encode(self, inputs: EvalImages) -> Encoded:
        model = FrozenViT(self.model, image_size=self.image_size, resize=self.resize, interpolation=self.interpolation)

        database_vectors = model.extract(iter_images(inputs.database_paths))
        # Cropped to the ground-truth region before resizing: the labels describe the
        # crop, and the protocol's resize then applies to the crop as to any image.
        cropped = (
            crop_query(image, box)
            for image, box in zip(iter_images(inputs.query_paths), inputs.query_boxes, strict=True)
        )
        return database_vectors, model.extract(cropped)
