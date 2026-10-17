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
from cbir.descriptors.cnn.compression import PCACompression
from cbir.descriptors.cnn.prepare import EvalImages, prepare_image_inputs
from cbir.descriptors.cnn.sfm import WhitenSource, whitening_paths
from cbir.descriptors.transformer.vit import IMAGE_SIZE, SPECS, FrozenViT, Interpolation, Model, Readout, Resize


@dataclass(frozen=True)
class ViTConfig:
    """A frozen vision transformer's global readout, compared by cosine.

    The readout is the CLS token for DINOv2/DINOv3/DeiT and the attention-pooling head
    for SigLIP 2 — each family's own image embedding. By default nothing is fitted, so
    the held-out images are never extracted; `dim`/`whiten` add the same post-processing
    the CNN tier's off-the-shelf rows use, so the two tiers can be compared like for like.

    Args:
        model: Which frozen checkpoint to read out. SigLIP 2 and DeiT names carry their
            resolution because each one is a separate fixed-size checkpoint.
        image_size: Input side: the square's for `pad` and `crop`, the longer side's for
            `aspect`. 224 is the published protocol; anything else is an ablation. Must
            be a multiple of the patch size, and equal to the native size for SigLIP 2
            and DeiT.
        resize: How an image is fitted to `image_size`. `pad` is the documented
            protocol; `aspect` and `crop` were ruled out as the reason the DINOv2
            paper's table does not reproduce — see `descriptors/transformer/vit.py`.
        interpolation: Resampling filter. The papers name none; bicubic is what
            DINOv2's own evaluation transforms use.
        readout: Which vector becomes the descriptor. `native` is the family's own
            image embedding and what every published row uses; `mean` and `gem` pool
            the patch tokens instead, as the CNN tier pools a conv map.
        p: GeM's exponent, for `readout="gem"` only. The tokens are signed, so GeM's
            clamp discards their negative half — see `Readout`.
        dim: PCA width, fitted on `whiten_source`. `None` keeps the model's native width
            — with `whiten` on that still fits a full-width PCA, since whitening is a
            rotation and rescale rather than a compression.
        whiten: Divide each PCA direction by its standard deviation. The published
            transformer rows use none, so it is off by default; the CNN tier's
            off-the-shelf rows all use it, which is why it exists here.
        whiten_source: Corpus the projection is fitted on: `held_out` (the sibling
            benchmark, as the CNN rows use) or the retrieval-SfM landmark sets
            (`sfm30k`/`sfm120k`, a manual download). Never the images being searched.
        shrinkage: Floors whitening's divisor at `sqrt(lambda + eps)`, `eps` being this
            fraction of the mean eigenvalue; see `PCACompression.fit`.
        seed: Seeds PCA's randomized solver, so a refit reproduces.
    """

    model: Model = "dinov2-vitl14"
    image_size: int = IMAGE_SIZE
    resize: Resize = "pad"
    interpolation: Interpolation = "bicubic"
    readout: Readout = "native"
    p: float | None = None
    dim: int | None = None
    whiten: bool = False
    whiten_source: WhitenSource = "held_out"
    shrinkage: float = 0.0
    seed: int = 0

    technique: ClassVar[str] = "vit"
    inputs_kind: ClassVar[str] = "images"

    def __post_init__(self) -> None:
        # A frozen ViT ships no projection of its own; `learned` names the one inside a
        # retrieval-fine-tuned CNN checkpoint and has nothing to read here.
        if self.whiten_source == "learned":
            raise ValueError(
                "whiten_source='learned' needs a checkpoint that ships a projection; a frozen ViT has none"
            )

    def prepare(self, dataset: EvalDataset) -> EvalImages:
        return prepare_image_inputs(dataset)

    def _fitting_paths(self, inputs: EvalImages) -> list[str]:
        """Images the PCA/whitening is fitted on — never the ones being searched."""
        if self.whiten_source == "held_out":
            return inputs.held_out_paths
        return whitening_paths(self.whiten_source)

    def train_and_encode(self, inputs: EvalImages) -> Encoded:
        model = FrozenViT(
            self.model,
            image_size=self.image_size,
            resize=self.resize,
            interpolation=self.interpolation,
            readout=self.readout,
            p=self.p,
        )

        database_vectors = model.extract(iter_images(inputs.database_paths))
        # Cropped to the ground-truth region before resizing: the labels describe the
        # crop, and the protocol's resize then applies to the crop as to any image.
        cropped = (
            crop_query(image, box)
            for image, box in zip(iter_images(inputs.query_paths), inputs.query_boxes, strict=True)
        )
        query_vectors = model.extract(cropped)

        if self.dim is None and not self.whiten:
            return database_vectors, query_vectors

        fitting = model.extract(iter_images(self._fitting_paths(inputs)))
        width = self.dim if self.dim is not None else SPECS[self.model].width
        pca = PCACompression.fit(fitting, dim=width, whiten=self.whiten, shrinkage=self.shrinkage, seed=self.seed)
        return pca.transform(database_vectors), pca.transform(query_vectors)
