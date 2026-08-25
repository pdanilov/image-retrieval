"""Turn an eval dataset name into the images the CNN tier consumes.

The classic tier's `prepare` runs RootSIFT over every image and hands back descriptor
arrays, because that extraction costs hours and must be cached and shared. A CNN needs
no such stage: the network *is* the extractor, and a few thousand images through a
frozen backbone on a GPU is a couple of minutes. So this prepares paths, not pixels —
decoding happens inside the descriptor, one image at a time.

Queries carry their bbx here rather than being cropped up front, for the same reason:
cropping is cheap and belongs next to the decode that feeds the network.
"""

from dataclasses import dataclass

from cbir.data.holdout import EvalDataset, HeldOutSplit, held_out_database
from cbir.data.images import image_paths
from cbir.data.revisitop import download


@dataclass(frozen=True)
class EvalImages:
    """Every image one eval direction needs, as on-disk paths.

    `held_out_paths` are the *other* dataset's database images — the only images a CNN
    run may fit anything on (here, the PCA). `query_boxes[i]` is the ground-truth
    region of `query_paths[i]`, in the same order.
    """

    held_out_paths: list[str]
    database_paths: list[str]
    query_paths: list[str]
    query_boxes: list[list[float]]


def prepare_image_inputs(eval_dataset: EvalDataset) -> EvalImages:
    """Paths for the held-out pool, the eval database, and the eval queries.

    Reads the image columns without decoding any of them (`image_paths` casts the
    column to an undecoded feature first), so this is a few thousand strings and costs
    nothing worth caching.
    """
    split = HeldOutSplit.for_eval(eval_dataset)
    held_out_paths = image_paths(held_out_database(split))

    query_dataset, database_dataset = download(eval_dataset)
    return EvalImages(
        held_out_paths=held_out_paths,
        database_paths=image_paths(database_dataset),
        query_paths=image_paths(query_dataset),
        # `bbx` is a plain float column; reading it whole costs nothing worth deferring.
        query_boxes=[list(box) for box in query_dataset["bbx"]],
    )
