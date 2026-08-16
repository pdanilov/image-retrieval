"""Disk cache for RootSIFT extraction, the expensive step in the classic pipeline
(~18 minutes for a single eval direction's held-out + database + queries, most of it
real per-image SIFT compute rather than anything this project controls the cost of).

Cached under this repo's `data/rootsift/` (gitignored — see `.gitignore`'s `/data/`
rule), keyed by `(dataset, role)`, `role` being `"database"` or `"query"` images of
`roxford5k`/`rparis6k`. There is deliberately no separate held-out cache entry: the
held-out pool for evaluating one dataset is always the *other* dataset's own database
descriptors (see `holdout.py`), so pooling the other dataset's cached `"database"`
entry serves both purposes for free — evaluating the second direction later reuses
both datasets' database caches, only the query crops differ.

A cache hit/miss is decided by a row lookup in the `rootsift` table of the shared
SQLite index (`db.py`, column `split` = `role`), not by a file existing at a
conventionally-named path -- see that module's docstring for why. It also owns where
blobs get written, so this module holds no paths of its own, only its table name; a
fresh result's actual path is recorded in the database and read back on future hits.

The `(dataset, role)` key lives here rather than on `RootSIFT` itself: which eval
dataset a batch of images belongs to, and whether they are database or query images,
are facts about the retrieval pipeline, not about a SIFT extractor. Keeping them out of
`local/rootsift.py` is what lets that module stay pure — no filesystem, no SQLite, and
tests that need no fixtures.

`RootSIFT` currently has no configurable parameters (fixed `cv2.SIFT_create()`
defaults), so `(dataset, role)` is the whole cache key. If that ever changes, the key
must grow to include it — a config change silently reusing a stale cache is exactly
the failure mode this cache exists to avoid, not introduce.
"""

from collections.abc import Iterable
from typing import Literal

import numpy as np
from PIL.Image import Image as PILImage

from cbir.descriptors.classic.cache import db
from cbir.descriptors.classic.local.rootsift import DIM, RootSIFT

TABLE = "rootsift"

Role = Literal["database", "query"]


class RootSIFTCache:
    """Cached RootSIFT extraction, keyed by `(dataset, role)`."""

    @staticmethod
    def pack(descriptors: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        """Ragged list of `(n_i, 128)` arrays -> one flat `(M, 128)` array + `(N+1,)` offsets.

        Image `i`'s descriptors are `flat[offsets[i]:offsets[i+1]]`. Avoids storing
        thousands of separately-named arrays in one npz for what's really one ragged array.
        """
        offsets = np.zeros(len(descriptors) + 1, dtype=np.int64)
        offsets[1:] = np.cumsum([len(d) for d in descriptors])
        flat = np.concatenate(descriptors, axis=0) if descriptors else np.zeros((0, DIM), dtype=np.float32)
        return flat, offsets

    @staticmethod
    def unpack(flat: np.ndarray, offsets: np.ndarray) -> list[np.ndarray]:
        """Inverse of `pack`. The returned arrays are views into `flat`, not copies."""
        return [flat[offsets[i] : offsets[i + 1]] for i in range(len(offsets) - 1)]

    @staticmethod
    def _load_or_compute(dataset: str, role: Role, images: Iterable[PILImage]) -> tuple[np.ndarray, np.ndarray]:
        """`(flat, offsets)` for `(dataset, role)`, from disk on a hit, else computed and cached.

        `images` is only required to be `Iterable`, never a materialized `Sequence`, and
        is consumed exactly once — on a cache hit it is not consumed at all. Pass a lazy
        generator (`data.images.iter_images`) rather than a list of decoded images, so a
        hit decodes nothing and a miss decodes one image at a time. Decoding every image
        up front is what caused a real OOM here once, at tens of GB of pixel data.

        `images` must yield the same images in the same order for a given `(dataset,
        role)` — the cache trusts the caller rather than re-verifying image content, the
        same way `revisitop.download`'s expected-count assertions trust the dataset loader
        rather than re-hashing every image.
        """
        with db.connect() as conn:
            row = conn.execute(
                f"SELECT filepath FROM {TABLE} WHERE dataset = ? AND split = ?", (dataset, role)
            ).fetchone()

        if row is not None:
            with np.load(row[0]) as data:
                return data["flat"], data["offsets"]

        flat, offsets = RootSIFTCache.pack(RootSIFT().extract_many(images))
        path = db.reserve_blob_path(TABLE, f"{dataset}_{role}")
        np.savez(path, flat=flat, offsets=offsets)

        with db.connect() as conn:
            conn.execute(
                f"INSERT OR REPLACE INTO {TABLE} (dataset, split, filepath) VALUES (?, ?, ?)",
                (dataset, role, str(path)),
            )

        return flat, offsets

    @staticmethod
    def extract_many(dataset: str, role: Role, images: Iterable[PILImage]) -> list[np.ndarray]:
        """Per-image descriptor list for `(dataset, role)` — for an aggregator's `encode()`."""
        flat, offsets = RootSIFTCache._load_or_compute(dataset, role, images)
        return RootSIFTCache.unpack(flat, offsets)

    @staticmethod
    def pooled(dataset: str, role: Role, images: Iterable[PILImage]) -> np.ndarray:
        """One pooled `(M, 128)` array for `(dataset, role)` — for fitting a codebook.

        Returns the cache's `flat` array directly rather than unpacking per-image and
        re-concatenating: at held-out scale (tens of millions of rows, multi-GB) that
        round trip would transiently hold two full copies in memory for no reason, since
        the cached `flat` array already *is* the pooled result.
        """
        flat, _offsets = RootSIFTCache._load_or_compute(dataset, role, images)
        return flat
