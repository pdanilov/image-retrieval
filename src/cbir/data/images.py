"""Reading images out of an HF dataset without decoding more than one at a time.

The `image` column of the revisitop datasets is an HF `Image` feature with
`decode=True`, so *touching* it hands back a decoded `PIL.Image` — and touching the
whole column (`dataset["image"]`) decodes every image in the dataset into memory
before the caller sees any of them. At roxford5k/rparis6k scale that is several tens
of GB of pixel data, and it caused a real OOM here once.

Re-casting the column to `decode=False` makes it hand back the underlying
`{"path": ..., "bytes": ...}` record instead. For these datasets the loader extracts
archives to disk and stores *paths* (`bytes` is None), so the whole column is a few
thousand strings — cheap enough to read eagerly, with decoding deferred to
`iter_images`, which opens exactly one file at a time.
"""

from collections.abc import Generator, Iterable

from PIL import Image
from PIL.Image import Image as PILImage


def image_paths(dataset) -> list[str]:
    """Every image's on-disk path, decoding none of them."""
    # Imported lazily, matching revisitop.download -- the rest of the CLI shouldn't
    # pay for importing `datasets` on every invocation.
    import datasets

    undecoded = dataset.cast_column("image", datasets.Image(decode=False))
    return [record["path"] for record in undecoded["image"]]


def iter_images(paths: Iterable[str]) -> Generator[PILImage]:
    """Open each path in turn, closing it before moving on to the next.

    Exactly one image is open at a time, so peak memory is one decoded image no
    matter how many paths are passed. Each image stays open for as long as the
    consumer is working on it (the `with` only exits when the next image is
    requested), so consumers can read pixel data from the yielded image directly.
    """
    for path in paths:
        with Image.open(path) as image:
            yield image
