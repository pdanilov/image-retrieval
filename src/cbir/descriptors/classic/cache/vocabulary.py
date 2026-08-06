"""Disk cache for trained `Vocabulary` (k-means) models, shared by BoW and VLAD.

Training a k-means vocabulary on tens of millions of held-out RootSIFT descriptors is
the other expensive step in the classic pipeline, alongside RootSIFT extraction itself
(cached by `cache/rootsift.py`). BoW and VLAD both train the exact same kind of model
(`Vocabulary`, k-means centers) from the exact same held-out descriptors for a given
`(held-out dataset, k, seed)` -- a vocabulary trained once for BoW at k=64 is directly
reusable by VLAD at k=64 (or a later BoW/VLAD rerun at that same k), with no retraining.

Cached under this repo's `data/kmeans/` (gitignored — see `.gitignore`'s `/data/`
rule), keyed by `(dataset, k, seed)`. `dataset` is the *held-out* dataset name (the
training data's identity, `ClassicDescriptorInputs.held_out_dataset`) -- not the eval
dataset, since two different eval directions can share one held-out dataset (e.g. a
`roxford5k` eval trains on rparis6k, same as it would if evaluating some third dataset
whose held-out set was also rparis6k).

A cache hit/miss is decided by a row lookup in the `kmeans` table of the shared SQLite
index (`db.py`), not by a file existing at a conventionally-named path -- see that
module's docstring for why — it also owns where blobs get written, so this module holds
no paths of its own, only its table name. Keeping all of that here rather than on
`Vocabulary` itself is what lets `codebook/vocabulary.py` stay pure k-means: no
filesystem, no SQLite, no notion of which dataset trained it.
"""

import numpy as np

from cbir.descriptors.classic.cache import db
from cbir.descriptors.classic.codebook.vocabulary import Vocabulary

TABLE = "kmeans"


class VocabularyCache:
    """Cached k-means vocabulary training, keyed by `(dataset, k, seed)`."""

    @staticmethod
    def train(dataset: str, descriptors: np.ndarray, k: int, seed: int) -> Vocabulary:
        """`Vocabulary` for `(dataset, k, seed)`: trained fresh, or loaded from disk on a hit."""
        with db.connect() as conn:
            row = conn.execute(
                f"SELECT filepath FROM {TABLE} WHERE dataset = ? AND k = ? AND seed = ?", (dataset, k, seed)
            ).fetchone()

        if row is not None:
            with np.load(row[0]) as data:
                return Vocabulary(centers=data["centers"])

        vocabulary = Vocabulary.train(descriptors, k=k, seed=seed)
        path = db.reserve_blob_path(TABLE, f"{dataset}_k{k}_seed{seed}")
        np.savez(path, centers=vocabulary.centers)

        with db.connect() as conn:
            conn.execute(
                f"INSERT OR REPLACE INTO {TABLE} (dataset, k, seed, filepath) VALUES (?, ?, ?, ?)",
                (dataset, k, seed, str(path)),
            )

        return vocabulary
