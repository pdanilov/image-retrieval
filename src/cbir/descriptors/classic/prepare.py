"""Turn an eval dataset name into RootSIFT descriptors ready for any classic aggregator.

RootSIFT extraction doesn't depend on which aggregator (BoW/VLAD/Fisher) will later
consume it, so it happens once here rather than being repeated per technique — each
technique's `fit_and_encode` (in `bow.py`/`vlad.py`/`fisher.py`) takes the result of
`prepare_classic_inputs` and only does the part that's actually technique-specific:
training its own vocabulary/GMM and aggregating.
"""

from dataclasses import dataclass

import numpy as np

from cbir.data.crop import crop_query
from cbir.data.holdout import EvalDataset, HeldOutSplit, held_out_database
from cbir.data.images import image_paths, iter_images
from cbir.data.revisitop import download
from cbir.descriptors.classic.cache.rootsift import RootSIFTCache


@dataclass(frozen=True)
class ClassicDescriptorInputs:
    """RootSIFT descriptors ready to train/encode any classic aggregator."""

    held_out_dataset: str  # which dataset trained held_out_descriptors -- cache key for vocabulary/GMM caches
    held_out_descriptors: np.ndarray  # (M, 128) pooled, for Vocabulary.train / GaussianMixture.train
    database_descriptors: list[np.ndarray]  # per-image, order = database list order
    query_descriptors: list[np.ndarray]  # per-image, bbx-cropped, order = query list order


def prepare_classic_inputs(eval_dataset: EvalDataset) -> ClassicDescriptorInputs:
    """Extract everything BoW/VLAD/Fisher need for one eval direction.

    Runs RootSIFT once each over the held-out set (for training), the eval database,
    and the bbx-cropped eval queries (both for encoding) — computed once regardless
    of how many techniques get trained from the result.
    """
    split = HeldOutSplit.for_eval(eval_dataset)
    # `image_paths` reads the whole image column without decoding any of it (a few
    # thousand strings), and `iter_images` decodes one at a time as extraction walks
    # it -- so a cache hit decodes nothing and a cache miss holds one image at a
    # time. The held-out pool is just the *other* dataset's own database descriptors,
    # so this reuses (or populates) that dataset's "database" cache entry rather than
    # a separate one -- and it's read as one pooled array directly
    # (`RootSIFTCache.pooled`), not unpacked-then-reconcatenated, since at
    # held-out scale (tens of millions of rows) that round trip would transiently
    # hold two full copies in memory.
    held_out_ds = held_out_database(split)
    held_out_descriptors = RootSIFTCache.pooled(
        split.held_out_dataset, "database", iter_images(image_paths(held_out_ds))
    )

    query_ds, database_ds = download(eval_dataset)
    database_descriptors = RootSIFTCache.extract_many(eval_dataset, "database", iter_images(image_paths(database_ds)))

    # Queries are cropped to their ground-truth region before extraction. `bbx` is a
    # plain float column, so reading it whole costs nothing worth deferring.
    cropped_queries = (
        crop_query(image, bbx) for image, bbx in zip(iter_images(image_paths(query_ds)), query_ds["bbx"], strict=True)
    )
    query_descriptors = RootSIFTCache.extract_many(eval_dataset, "query", cropped_queries)

    return ClassicDescriptorInputs(
        held_out_dataset=split.held_out_dataset,
        held_out_descriptors=held_out_descriptors,
        database_descriptors=database_descriptors,
        query_descriptors=query_descriptors,
    )
